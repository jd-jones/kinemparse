import argparse
import os
import collections

import yaml
import torch
import pyplot as plt
import joblib

from mathtools import utils, torchutils, metrics
from blocks import notebookutils
import seqtools.torchutils


def main(
        out_dir=None, data_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        cv_params={}, train_params={}):

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
        joblib.dump(os.path.join(out_data_dir, f'{var_name}.pkl'))

    device = torchutils.selectDevice(gpu_dev_id)

    # Load data
    trial_ids = utils.loadVariable('trial_ids', data_dir)
    accel_seqs = utils.loadVariable('accel_samples', data_dir)
    gyro_seqs = utils.loadVariable('gyro_samples', data_dir)
    action_seqs = utils.loadVariable('action_seqs', data_dir)

    # Reformat action_seqs
    action_seqs = tuple(
        tuple(event for action in action_seq for event in action)
        for action_seq in action_seqs
    )

    # Compute signal magnitude
    imu_sample_seqs, imu_timestamp_seqs = utils.batchProcess(
        notebookutils.makeImuSeq,
        accel_seqs, gyro_seqs,
        static_kwargs={'mag_only': True},
        unzip=True
    )

    label_seqs = None

    # Define cross-validation folds
    cv_folds = notebookutils.makeDataSplits(imu_sample_seqs, label_seqs, **cv_params)

    for cv_index, (train_data, val_data, test_data) in enumerate(cv_folds):
        if train_data == ((), (), (),):
            train_set = None
            train_loader = None
            train_ids = tuple()
        else:
            train_obsv, train_labels, train_ids = train_data
            train_set = seqtools.torchutils.SequenceDataset(train_obsv, train_labels, device=device)
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=2
            )

        if test_data == ((), (), (),):
            test_set = None
            test_loader = None
            test_ids = tuple()
        else:
            test_obsv, test_labels, test_ids = test_data
            test_set = seqtools.torchutils.SequenceDataset(test_obsv, test_labels, device=device)
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=True, num_workers=2
            )

        if val_data == ((), (), (),):
            val_set = None
            val_loader = None
            val_ids = tuple()
        else:
            val_obsv, val_labels, val_ids = val_data
            val_set = seqtools.torchutils.SequenceDataset(val_obsv, val_labels, device=device)
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, shuffle=True, num_workers=2
            )

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )
        logger.info(f'  test set: {test_ids}')

        input_dim = train_set.num_obsv_dims
        output_dim = train_set.num_label_types
        model = torchutils.LinearClassifier(input_dim, output_dim)

        train_epoch_log = collections.defaultdict(list)
        val_epoch_log = collections.defaultdict(list)
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy(),
            'Precision': metrics.Precision(),
            'Recall': metrics.Recall(),
            'F1': metrics.Fmeasure()
        }

        criterion = torch.nn.CrossEntropyLoss()
        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

        model, last_model_wts = torchutils.trainModel(
            model, criterion, optimizer_ft, exp_lr_scheduler,
            train_loader, val_loader,
            metrics=metric_dict,
            train_epoch_log=train_epoch_log,
            val_epoch_log=val_epoch_log,
            **train_params
        )

        utils.saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        utils.saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        utils.saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        utils.saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        utils.saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        utils.saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        utils.saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        model.load_state_dict(last_model_wts)
        utils.saveVariable(model, f'cvfold={cv_index}_{model_name}-last')

        # Plot training performance
        subfig_size = (10, 2.5)
        fig_title = 'Training performance'
        torchutils.plotEpochLog(train_epoch_log, subfig_size=subfig_size, title=fig_title)
        notebookutils.saveExptFig(fig_dir, fig_title)
        plt.close()

        if val_epoch_log:
            # Plot heldout performance
            subfig_size = (10, 2.5)
            fig_title = 'Heldout performance'
            torchutils.plotEpochLog(val_epoch_log, subfig_size=subfig_size, title=fig_title)
            notebookutils.saveExptFig(fig_dir, fig_title)
            plt.close()

        # Test model
        model, avg_loss, avg_acc, avg_prc, avg_rec = torchutils.epochValPhase(
            model, criterion, optimizer_ft, exp_lr_scheduler, test_loader,
            device=device, update_interval=None
        )

        fmt_str = '[{}]  Loss: {:.4f}   Acc: {:5.2f}%   Prc: {:5.2f}%   Rec: {:5.2f}%'
        logger.info(
            fmt_str.format(
                'TEST', avg_loss,
                avg_acc * 100, avg_prc * 100, avg_rec * 100
            )
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

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
