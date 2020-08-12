import os
import collections
import logging

import yaml
import torch
import joblib

from mathtools import utils, torchutils, metrics


logger = logging.getLogger(__name__)


class DummyClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_set_size, binary_multiclass=False):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size

        logger.info(
            f'Initialized dummy classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        return input_seq.transpose(1, 2)

    def predict(self, outputs):
        __, preds = torch.max(outputs, -1)
        return preds


class ConvClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_set_size, kernel_size=3, binary_multiclass=False):
        super().__init__()

        if not (kernel_size % 2):
            raise NotImplementedError("Kernel size must be an odd number!")

        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.binary_labels = binary_multiclass

        self.conv1d = torch.nn.Conv1d(
            self.input_dim, self.out_set_size, kernel_size,
            padding=(kernel_size // 2)
        )

        logger.info(
            f'Initialized 1D convolutional classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        output_seq = self.conv1d(input_seq).transpose(1, 2)
        return output_seq

    def predict(self, outputs):
        if self.binary_labels:
            return (outputs > 0.5).float()
        __, preds = torch.max(outputs, -1)
        return preds


class TcnClassifier(torch.nn.Module):
    def __init__(
            self, input_dim, out_set_size,
            binary_multiclass=False, tcn_channels=None, **tcn_kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.binary_multiclass = binary_multiclass

        self.TCN = torchutils.TemporalConvNet(input_dim, tcn_channels, **tcn_kwargs)
        self.linear = torch.nn.Linear(tcn_channels[-1], self.out_set_size)

        logger.info(
            f'Initialized TCN classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq, return_feats=False):
        tcn_out = self.TCN(input_seq).transpose(1, 2)
        linear_out = self.linear(tcn_out)

        if return_feats:
            return linear_out, tcn_out
        return linear_out

    def predict(self, outputs, return_scores=False):
        if self.binary_multiclass:
            return (outputs > 0.5).float()

        __, preds = torch.max(outputs, -1)

        if return_scores:
            scores = torch.nn.softmax(outputs, dim=-1)
            return preds, scores
        return preds


def main(
        out_dir=None, data_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        plot_predictions=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, f'results.csv')
        write_mode = 'w'
    else:
        results_file = os.path.expanduser(results_file)
        write_mode = 'a'

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=')
    feature_seqs = loadAll(trial_ids, 'feature-seq.pkl', data_dir)
    label_seqs = loadAll(trial_ids, 'label-seq.pkl', data_dir)

    device = torchutils.selectDevice(gpu_dev_id)

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, label_seqs, trial_ids)
        )
        return split_data

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        train_feats, train_labels, train_ids = train_data
        train_set = torchutils.SequenceDataset(
            train_feats, train_labels,
            device=device, labels_dtype=torch.long, seq_ids=train_ids,
            transpose_data=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        test_feats, test_labels, test_ids = test_data
        test_set = torchutils.SequenceDataset(
            test_feats, test_labels,
            device=device, labels_dtype=torch.long, seq_ids=test_ids,
            transpose_data=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        val_feats, val_labels, val_ids = val_data
        val_set = torchutils.SequenceDataset(
            val_feats, val_labels,
            device=device, labels_dtype=torch.long, seq_ids=val_ids,
            transpose_data=True
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(trial_ids)} total '
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
        elif model_name == 'TCN':
            model = TcnClassifier(input_dim, output_dim, **model_params).to(device=device)
        elif model_name == 'dummy':
            model = DummyClassifier(input_dim, output_dim, **model_params)
        else:
            raise AssertionError()

        criterion = torch.nn.CrossEntropyLoss()
        if model_name != 'dummy':
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
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

            model, last_model_wts = torchutils.trainModel(
                model, criterion, optimizer_ft, lr_scheduler,
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

        d = {k: v.value for k, v in metric_dict.items()}
        utils.writeResults(
            results_file, d, sweep_param_name, model_params,
            write_mode=write_mode
        )

        if plot_predictions:
            io_fig_dir = os.path.join(fig_dir, 'model-io')
            if not os.path.exists(io_fig_dir):
                os.makedirs(io_fig_dir)

            label_names = ('gt', 'pred')
            preds, scores, inputs, gt_labels, ids = zip(*test_io_history)
            for batch in test_io_history:
                batch = map(lambda x: x.cpu().numpy(), batch)
                for preds, _, inputs, gt_labels, seq_id in zip(*batch):
                    fn = os.path.join(io_fig_dir, f"trial={seq_id}_model-io.png")
                    utils.plot_array(inputs, (gt_labels, preds), label_names, fn=fn)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq, f'trial={trial_id}_score-seq')
            saveVariable(label_seq, f'trial={trial_id}_true-label-seq')
        for batch in test_io_history:
            batch = map(lambda x: x.cpu().numpy(), batch)
            for io in zip(*batch):
                saveTrialData(*io)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        train_fig_dir = os.path.join(fig_dir, 'train-plots')
        if not os.path.exists(train_fig_dir):
            os.makedirs(train_fig_dir)

        torchutils.plotEpochLog(
            train_epoch_log,
            subfig_size=(10, 2.5),
            title='Training performance',
            fn=os.path.join(train_fig_dir, f'cvfold={cv_index}_train-plot.png')
        )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(train_fig_dir, f'cvfold={cv_index}_val-plot.png')
            )


if __name__ == "__main__":
    # Parse command-line args and config file
    cl_args = utils.parse_args(main)
    config, config_fn = utils.parse_config(cl_args, script_name=__file__)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    main(**config)
