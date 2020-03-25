import argparse
import os
import collections
import itertools
import copy

import yaml
import torch
import numpy as np
from matplotlib import pyplot as plt
import joblib

from mathtools import utils, torchutils, metrics
from blocks.estimation import notebookutils
from blocks.core import labels
import seqtools.torchutils


def main(
        out_dir=None, data_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None, as_array=None,
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
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    device = torchutils.selectDevice(gpu_dev_id)

    # Load data
    trial_ids = loadVariable('trial_ids')
    accel_seqs = loadVariable('accel_samples')
    gyro_seqs = loadVariable('gyro_samples')
    action_seqs = loadVariable('action_seqs')
    orig_rgb_frame_timestamp_seqs = loadVariable('orig_rgb_timestamps')

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

    # Compute block activity labels from action annotations
    obj_label_seqs = utils.batchProcess(
        labels.extractBlockActionSeq,
        action_seqs, orig_rgb_frame_timestamp_seqs,
        static_kwargs={
            'split_samples': False,
            'include_adj_target': False,
            'action_type': 'object'
        }
    )

    tgt_label_seqs = utils.batchProcess(
        labels.extractBlockActionSeq,
        action_seqs, orig_rgb_frame_timestamp_seqs,
        static_kwargs={
            'split_samples': False,
            'include_adj_target': False,
            'action_type': 'target'
        }
    )

    imu_obj_label_seqs = utils.batchProcess(
        notebookutils.makeImuLabelSeq,
        obj_label_seqs, orig_rgb_frame_timestamp_seqs, imu_timestamp_seqs
    )

    imu_tgt_label_seqs = utils.batchProcess(
        notebookutils.makeImuLabelSeq,
        tgt_label_seqs, orig_rgb_frame_timestamp_seqs, imu_timestamp_seqs
    )

    # Post-process data
    imu_is_resting = utils.batchProcess(
        notebookutils.imuResting, imu_obj_label_seqs, imu_tgt_label_seqs
    )

    centered_imu_sample_seqs = utils.batchProcess(
        notebookutils.centerSignals, imu_sample_seqs, imu_is_resting
    )

    label_seqs = utils.batchProcess(
        notebookutils.makeImuActivityLabels,
        centered_imu_sample_seqs, imu_obj_label_seqs, imu_tgt_label_seqs
    )

    if as_array:
        def stackSeqs(seq_dict):
            num_seqs = len(seq_dict)
            return np.hstack(tuple(seq_dict[i] for i in range(num_seqs)))
        centered_imu_sample_seqs = tuple(map(stackSeqs, centered_imu_sample_seqs))
        # imu_timestamp_seqs = tuple(map(stackSeqs, imu_timestamp_seqs))
        label_seqs = tuple(map(stackSeqs, label_seqs))
    else:
        def splitSeqs(seq_dict):
            num_seqs = len(seq_dict)
            return tuple(seq_dict[i] for i in range(num_seqs))
        trial_ids = tuple(itertools.chain(
            *((t_id,) * len(label_dict) for t_id, label_dict in zip(trial_ids, label_seqs))
        ))
        imu_sample_seqs = tuple(itertools.chain(*map(splitSeqs, centered_imu_sample_seqs)))
        # imu_timestamp_seqs = itertools.chain(*map(splitSeqs, imu_timestamp_seqs))
        label_seqs = tuple(itertools.chain(*map(splitSeqs, label_seqs)))

    # Define cross-validation folds
    cv_folds = notebookutils.makeDataSplits(imu_sample_seqs, label_seqs, trial_ids, **cv_params)

    for cv_index, (train_data, val_data, test_data) in enumerate(cv_folds):
        if train_data == ((), (), (),):
            train_set = None
            train_loader = None
            train_ids = tuple()
        else:
            train_obsv, train_labels, train_ids = train_data
            train_set = seqtools.torchutils.SequenceDataset(
                train_obsv, train_labels, device=device, labels_dtype=torch.long
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
            test_set = seqtools.torchutils.SequenceDataset(
                test_obsv, test_labels,
                device=device, labels_dtype=torch.long
            )
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

        if val_data == ((), (), (),):
            val_set = None
            val_loader = None
            val_ids = tuple()
        else:
            val_obsv, val_labels, val_ids = val_data
            val_set = seqtools.torchutils.SequenceDataset(
                val_obsv, val_labels,
                device=device, labels_dtype=torch.long
            )
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )
        # logger.info(f'  test set: {test_ids}')

        input_dim = train_set.num_obsv_dims
        output_dim = train_set.num_label_types
        model = torchutils.LinearClassifier(input_dim, output_dim).to(device=device)

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

        # Test model
        metric_dict = copy.deepcopy(metric_dict)
        _ = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch']
        )
        for metric_name, metric in metric_dict.items():
            val_epoch_log[metric_name].append(metric.evaluate())
        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        model.load_state_dict(last_model_wts)
        saveVariable(model, f'cvfold={cv_index}_{model_name}-last')

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
