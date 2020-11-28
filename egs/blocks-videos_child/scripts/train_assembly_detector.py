import os
import collections
import logging

import yaml
import torch
import joblib
import numpy as np

from mathtools import utils, metrics, torchutils
from kinemparse import sim2real


logger = logging.getLogger(__name__)


def main(
        out_dir=None, data_dir=None, pretrain_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        num_disp_imgs=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    if pretrain_dir is not None:
        pretrain_dir = os.path.expanduser(pretrain_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    io_dir = os.path.join(fig_dir, 'model-io')
    if not os.path.exists(io_dir):
        os.makedirs(io_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadAssemblies(seq_id, vocab):
        assembly_seq = joblib.load(os.path.join(data_dir, f"trial={seq_id}_assembly-seq.pkl"))
        labels = np.array([utils.getIndex(assembly, vocab) for assembly in assembly_seq])
        return labels

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadVariable(var_name):
        return joblib.load(os.path.join(pretrain_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    vocab = []
    label_seqs = tuple(loadAssemblies(t_id, vocab) for t_id in trial_ids)

    logger.info(f"Loaded {len(trial_ids)} sequences; {len(vocab)} unique assemblies")

    trial_ids = np.array([
        trial_id for trial_id, l_seq in zip(trial_ids, label_seqs)
        if l_seq is not None
    ])
    label_seqs = tuple(
        l_seq for l_seq in label_seqs
        if l_seq is not None
    )

    device = torchutils.selectDevice(gpu_dev_id)

    if model_name == 'AAE':
        dataset = sim2real.DenoisingDataset
    elif model_name == 'Resnet':
        dataset = sim2real.RenderDataset
    elif model_name == 'Connections':
        dataset = sim2real.ConnectionDataset

    for cv_index, cv_splits in enumerate(range(1)):
        train_set = dataset(vocab, device=device)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = dataset(vocab, device=device)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

        val_set = dataset(vocab, device=device)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        if model_name == 'AAE':
            model = sim2real.AugmentedAutoEncoder(
                train_set.data_shape, train_set.num_classes,
                # debug_fig_dir=io_dir
            )
            criterion = torchutils.BootstrappedCriterion(
                0.25, base_criterion=torch.nn.functional.mse_loss,
            )
            metric_names = ('Reciprocal Loss',)
        elif model_name == 'Resnet':
            model = sim2real.ImageClassifier(train_set.num_classes)
            criterion = torch.nn.CrossEntropy()
            metric_names = ('Loss', 'Accuracy')
        elif model_name == 'Connections':
            model = sim2real.ConnectionClassifier(train_set.label_shape[0])
            criterion = torch.nn.BCEWithLogitsLoss()
            metric_names = ('Loss', 'Accuracy', 'Precision', 'Recall', 'F1')

        model = model.to(device=device)

        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

        train_epoch_log = collections.defaultdict(list)
        val_epoch_log = collections.defaultdict(list)
        metric_dict = {name: metrics.makeMetric(name) for name in metric_names}
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
        metric_dict = {name: metrics.makeMetric(name) for name in metric_names}
        test_io_batches = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )
        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq.cpu().numpy(), f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq.cpu().numpy(), f'trial={trial_id}_score-seq')
            saveVariable(label_seq.cpu().numpy(), f'trial={trial_id}_true-label-seq')
        for batch in test_io_batches:
            saveTrialData(*batch)

        cv_str = f"cvfold={cv_index}"

        saveVariable(train_epoch_log, f'{cv_str}_model-train-epoch-log')
        saveVariable(val_epoch_log, f'{cv_str}_model-val-epoch-log')
        saveVariable(metric_dict, f'{cv_str}_model-metric-dict')
        saveVariable(model, f'{cv_str}_model-best')

        model.load_state_dict(last_model_wts)
        saveVariable(model, f'{cv_str}_model-last')

        if train_epoch_log:
            torchutils.plotEpochLog(
                train_epoch_log,
                subfig_size=(10, 2.5),
                title='Training performance',
                fn=os.path.join(fig_dir, f'{cv_str}_train-plot.png')
            )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(fig_dir, f'{cv_str}_val-plot.png')
            )

        if num_disp_imgs is not None:
            model.plotBatches(test_io_batches, io_dir, dataset=test_set)


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
