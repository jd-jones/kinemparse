import argparse
import os
import collections
import itertools

import yaml
import torch
import joblib

from mathtools import utils, torchutils, metrics
from kinemparse import imu

import pdb


class ConvClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_set_size, kernel_size, binary_labels=False):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.binary_labels = binary_labels

        self.conv1d = torch.nn.Conv1d(self.input_dim, self.out_set_size, kernel_size, padding=(kernel_size//2))

        logger.info(
            f'Initialized 1D convolutional classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        # want (batch_size, num_channels, seq_length)
        input_seq = input_seq.transpose(1, 2)
        # want (batch_size, num_channels, seq_length)
        output_seq = self.conv1d(input_seq)
        return output_seq.transpose(1, 2)

    def predict(self, outputs):
        if self.binary_labels:
            return (outputs > 0.5).float()
        __, preds = torch.max(outputs, -1)
        return preds


def split(imu_feature_seqs, imu_label_seqs, trial_ids):
    num_signals = imu_label_seqs[0].shape[1]

    def validate(seqs):
        return all(seq.shape[1] == num_signals for seq in seqs)
    all_valid = all(validate(x) for x in (imu_feature_seqs, imu_label_seqs))
    if not all_valid:
        raise AssertionError("IMU and labels don't all have the same number of sequences")

    trial_ids = tuple(
        itertools.chain(
            *(
                tuple(t_id + 0.01 * (i + 1) for i in range(num_signals))
                for t_id in trial_ids
            )
        )
    )

    def splitSeq(arrays):
        return tuple(row for array in arrays for row in array)
    imu_feature_seqs = splitSeq(map(lambda x: x.swapaxes(0, 1), imu_feature_seqs))
    imu_label_seqs = splitSeq(map(lambda x: x.T, imu_label_seqs))

    return imu_feature_seqs, imu_label_seqs, trial_ids


def main(
        out_dir=None, data_dir=None, model_name=None, plot_predictions=None, 
        gpu_dev_id=None, batch_size=None, learning_rate=None, independent_signals=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = loadVariable('trial_ids')
    imu_sample_seqs = loadVariable('imu_sample_seqs')
    imu_label_seqs = loadVariable('imu_label_seqs')

    device = torchutils.selectDevice(gpu_dev_id)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(
        imu_sample_seqs, imu_label_seqs, trial_ids,
        **cv_params
    )

    for cv_index, (train_data, val_data, test_data) in enumerate(cv_folds):
        if independent_signals:
            criterion = torch.nn.CrossEntropyLoss()
            labels_dtype = torch.long
            fig_type = None
            train_data = split(*train_data)
            val_data = split(*val_data)
            test_data = split(*test_data)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
            labels_dtype = torch.float
            fig_type = 'multi'

        if train_data == ((), (), (),):
            train_set = None
            train_loader = None
            train_ids = tuple()
        else:
            train_obsv, train_labels, train_ids = train_data
            train_set = torchutils.SequenceDataset(
                train_obsv, train_labels,
                device=device, labels_dtype=labels_dtype, seq_ids=train_ids
            )
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True
            )

        if test_data == ((), (), (),):
            test_set = None
            test_loader = None
            test_ids = tuple()
        else:
            test_obsv, test_labels, test_ids = test_data
            test_set = torchutils.SequenceDataset(
                test_obsv, test_labels,
                device=device, labels_dtype=labels_dtype, seq_ids=test_ids
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False
            )

        if val_data == ((), (), (),):
            val_set = None
            val_loader = None
            val_ids = tuple()
        else:
            val_obsv, val_labels, val_ids = val_data
            val_set = torchutils.SequenceDataset(
                val_obsv, val_labels,
                device=device, labels_dtype=labels_dtype, seq_ids=val_ids
            )
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        input_dim = train_set.num_obsv_dims
        output_dim = train_set.num_label_types
        if model_name == 'linear':
            model = torchutils.LinearClassifier(
                input_dim, output_dim, **model_params
            ).to(device=device)
        elif model_name == 'conv':
            model = ConvClassifier(input_dim, output_dim, **model_params).to(device=device)
        else:
            raise AssertionError()

        train_epoch_log = collections.defaultdict(list)
        val_epoch_log = collections.defaultdict(list)
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy(),
            'Precision': metrics.Precision(),
            'Recall': metrics.Recall(),
            'F1': metrics.Fmeasure()
        }

        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

        model, last_model_wts = torchutils.trainModel(
            model, criterion, optimizer_ft, exp_lr_scheduler,
            train_loader, val_loader,
            device=device,
            metrics=metric_dict,
            train_epoch_log=train_epoch_log,
            val_epoch_log=val_epoch_log,
            **train_params
        )

        # Test model
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy(),
            'Precision': metrics.Precision(),
            'Recall': metrics.Recall(),
            'F1': metrics.Fmeasure()
        }
        test_io_history = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )
        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        if plot_predictions:
            imu.plot_prediction_eg(test_io_history, fig_dir, fig_type=fig_type, **viz_params)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        model.load_state_dict(last_model_wts)
        saveVariable(model, f'cvfold={cv_index}_{model_name}-last')

        torchutils.plotEpochLog(
            train_epoch_log,
            subfig_size=(10, 2.5),
            title='Training performance',
            fn=os.path.join(fig_dir, 'train-plot.png')
        )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(fig_dir, 'val-plot.png')
            )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--model_params')
    parser.add_argument('--data_dir')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    config.update(args)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
