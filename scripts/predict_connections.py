import argparse
import os
import collections

import yaml
import torch
from matplotlib import pyplot as plt
import joblib

from mathtools import utils, torchutils, metrics
from blocks.estimation import notebookutils
import seqtools.torchutils


def plot_prediction_eg(*args, fig_type=None, **kwargs):
    if fig_type is None:
        return plot_prediction_eg_standard(*args, **kwargs)
    elif fig_type == 'array':
        return plot_prediction_eg_array(*args, **kwargs)
    elif fig_type == 'multi':
        return plot_prediction_eg_multi(*args, **kwargs)


def plot_prediction_eg_array(io_history, expt_out_path):
    subplot_width = 12
    subplot_height = 3

    for fig_idx, io_sample in enumerate(io_history):
        preds, inputs, true_labels = map(lambda x: x.squeeze().cpu().numpy().T, io_sample)
        figsize = (subplot_width, 3 * subplot_height)
        fig, axes = plt.subplots(3, figsize=figsize)
        axes[0].imshow(preds, interpolation='none', aspect='auto')
        axes[0].set_ylabel('Predicted')
        axes[1].imshow(true_labels, interpolation='none', aspect='auto')
        axes[1].set_ylabel('Ground truth')
        axes[2].imshow(inputs, interpolation='none', aspect='auto')
        axes[2].set_ylabel('Input')
        plt.tight_layout()
        fig_title = f'model-predictions-{fig_idx}.png'
        notebookutils.saveExptFig(expt_out_path, fig_title=fig_title)
        plt.close()


def plot_prediction_eg_multi(io_history, expt_out_path):
    subplot_width = 12
    subplot_height = 2

    for fig_idx, io_sample in enumerate(io_history):
        preds, inputs, true_labels = map(lambda x: x.squeeze().cpu().numpy(), io_sample)
        num_seqs = true_labels.shape[1]
        figsize = (subplot_width, num_seqs * subplot_height)
        fig, axes = plt.subplots(num_seqs, figsize=figsize)
        if num_seqs == 1:
            axes = (axes,)
        for i in range(num_seqs):
            axis = axes[i]
            input_seq = inputs[:, [i, i + num_seqs]]
            pred_seq = preds[:, i]
            gt_seq = true_labels[:, i]
            _ = notebookutils.plotImu(
                (input_seq,), (pred_seq, gt_seq),
                label_names=('preds', 'labels'), axis=axis
            )
        plt.tight_layout()
        fig_title = f'model-predictions-{fig_idx}.png'
        notebookutils.saveExptFig(expt_out_path, fig_title=fig_title)
        plt.close()


def plot_prediction_eg_standard(io_history, expt_out_path, num_samples_per_fig=8, fig_type=None):
    subplot_width = 12
    subplot_height = 2

    s_idxs = tuple(range(0, len(io_history), num_samples_per_fig))
    e_idxs = s_idxs[1:] + (len(io_history),)
    io_histories = tuple(io_history[s_idx:e_idx] for s_idx, e_idx in zip(s_idxs, e_idxs))

    for fig_idx, io_samples in enumerate(io_histories):
        num_seqs = len(io_samples)
        figsize = (subplot_width, num_seqs * subplot_height)
        fig, axes = plt.subplots(num_seqs, figsize=figsize)
        if num_seqs == 1:
            axes = (axes,)
        for axis, io_sample in zip(axes, io_samples):
            preds, inputs, true_labels = map(lambda x: x.squeeze().cpu().numpy(), io_sample)
            _ = notebookutils.plotImu(
                (inputs,), (preds, true_labels),
                label_names=('preds', 'labels'), axis=axis
            )
        plt.tight_layout()

        fig_title = f'model-predictions-{fig_idx}.png'
        notebookutils.saveExptFig(expt_out_path, fig_title=fig_title)
        plt.close()


def main(
        out_dir=None, data_dir=None, model_name=None,
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
    cv_folds = notebookutils.makeDataSplits(
        imu_sample_seqs, imu_label_seqs, trial_ids,
        **cv_params
    )

    for cv_index, (train_data, val_data, test_data) in enumerate(cv_folds):
        if independent_signals:
            criterion = torch.nn.CrossEntropyLoss()
            labels_dtype = torch.long
            fig_type = None
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
            train_set = seqtools.torchutils.SequenceDataset(
                train_obsv, train_labels,
                device=device,
                labels_dtype=labels_dtype
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
                device=device,
                labels_dtype=labels_dtype
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
                device=device,
                labels_dtype=labels_dtype
            )
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        input_dim = train_set.num_obsv_dims
        output_dim = train_set.num_label_types
        model = torchutils.LinearClassifier(
            input_dim, output_dim, **model_params
        ).to(device=device)

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
        # plot_prediction_eg(test_io_history, fig_dir)
        plot_prediction_eg(test_io_history, fig_dir, fig_type=fig_type, **viz_params)

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
