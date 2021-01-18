import os
import collections
import functools
import logging

import yaml
import torch
import joblib

from mathtools import utils, torchutils, metrics
from kinemparse import imu

from seqtools import torch_lctm


logger = logging.getLogger(__name__)


class ConvClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_set_size, kernel_size, binary_labels=False):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.binary_labels = binary_labels

        self.conv1d = torch.nn.Conv1d(
            self.input_dim, self.out_set_size, kernel_size,
            padding=(kernel_size // 2)
        )

        logger.info(
            f'Initialized 1D convolutional classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        return self.conv1d(input_seq).transpose(1, 2)

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


class SegmentalTcnClassifier(TcnClassifier):
    def __init__(self, max_segs, pw, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)

        self.max_segs = max_segs
        self.pw = pw

    def predict(self, outputs, return_scores=False):
        if self.binary_multiclass:
            raise NotImplementedError()

        predict_segmental = functools.partial(
            torch_lctm.segmental_inference,
            max_segs=self.max_segs, pw=self.pw, return_scores=False
        )

        preds = torch.stack([predict_segmental(x) for x in outputs])

        if return_scores:
            scores = torch.nn.softmax(outputs, dim=-1)
            return preds, scores

        return preds


def main(
        out_dir=None, data_dir=None, model_name=None, pretrained_model_dir=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        independent_signals=None, active_only=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        plot_predictions=None, results_file=None, sweep_param_name=None,
        label_mapping=None, eval_label_mapping=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
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
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)
    feature_seqs = loadAll(trial_ids, 'feature-seq.pkl', data_dir)
    label_seqs = loadAll(trial_ids, 'label-seq.pkl', data_dir)

    device = torchutils.selectDevice(gpu_dev_id)

    if label_mapping is not None:
        def map_labels(labels):
            for i, j in label_mapping.items():
                labels[labels == i] = j
            return labels
        label_seqs = tuple(map(map_labels, label_seqs))

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
        if pretrained_model_dir is not None:
            def loadFromPretrain(fn):
                return joblib.load(os.path.join(pretrained_model_dir, f"{fn}.pkl"))
            model = loadFromPretrain(f'cvfold={cv_index}_{model_name}-best')
            train_ids = loadFromPretrain(f'cvfold={cv_index}_train-ids')
            val_ids = loadFromPretrain(f'cvfold={cv_index}_val-ids')
            test_ids = tuple(i for i in trial_ids if i not in (train_ids + val_ids))
            test_idxs = tuple(trial_ids.tolist().index(i) for i in test_ids)
            test_data = getSplit(test_idxs)

            if independent_signals:
                criterion = torch.nn.CrossEntropyLoss()
                labels_dtype = torch.long
                test_data = splitSeqs(*test_data, active_only=False)
            else:
                # FIXME
                # criterion = torch.nn.BCEWithLogitsLoss()
                # labels_dtype = torch.float
                criterion = torch.nn.CrossEntropyLoss()
                labels_dtype = torch.long

            test_feats, test_labels, test_ids = test_data
            test_set = torchutils.SequenceDataset(
                test_feats, test_labels,
                device=device, labels_dtype=labels_dtype, seq_ids=test_ids,
                transpose_data=True
            )
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=batch_size, shuffle=False
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
            if independent_signals:
                test_io_history = tuple(joinSeqs(test_io_history))

            metric_str = '  '.join(str(m) for m in metric_dict.values())
            logger.info('[TST]  ' + metric_str)

            d = {k: v.value for k, v in metric_dict.items()}
            utils.writeResults(results_file, d, sweep_param_name, model_params)

            if plot_predictions:
                imu.plot_prediction_eg(test_io_history, fig_dir, **viz_params)

            def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
                if label_mapping is not None:
                    def dup_score_cols(scores):
                        num_cols = scores.shape[-1] + len(label_mapping)
                        col_idxs = torch.arange(num_cols)
                        for i, j in label_mapping.items():
                            col_idxs[i] = j
                        return scores[..., col_idxs]
                    score_seq = dup_score_cols(score_seq)
                saveVariable(pred_seq.cpu().numpy(), f'trial={trial_id}_pred-label-seq')
                saveVariable(score_seq.cpu().numpy(), f'trial={trial_id}_score-seq')
                saveVariable(label_seq.cpu().numpy(), f'trial={trial_id}_true-label-seq')
            for io in test_io_history:
                saveTrialData(*io)
            continue

        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        if independent_signals:
            criterion = torch.nn.CrossEntropyLoss()
            labels_dtype = torch.long
            split_ = functools.partial(splitSeqs, active_only=active_only)
            train_data = split_(*train_data)
            val_data = split_(*val_data)
            test_data = splitSeqs(*test_data, active_only=False)
        else:
            # FIXME
            # criterion = torch.nn.BCEWithLogitsLoss()
            # labels_dtype = torch.float
            criterion = torch.nn.CrossEntropyLoss()
            labels_dtype = torch.long

        train_feats, train_labels, train_ids = train_data
        train_set = torchutils.SequenceDataset(
            train_feats, train_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=train_ids,
            transpose_data=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        test_feats, test_labels, test_ids = test_data
        test_set = torchutils.SequenceDataset(
            test_feats, test_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=test_ids,
            transpose_data=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        val_feats, val_labels, val_ids = val_data
        val_set = torchutils.SequenceDataset(
            val_feats, val_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=val_ids,
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
            model = TcnClassifier(input_dim, output_dim, **model_params)
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
        if independent_signals:
            test_io_history = tuple(joinSeqs(test_io_history))

        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        d = {k: v.value for k, v in metric_dict.items()}
        utils.writeResults(results_file, d, sweep_param_name, model_params)

        if plot_predictions:
            # imu.plot_prediction_eg(test_io_history, fig_dir, fig_type=fig_type, **viz_params)
            imu.plot_prediction_eg(test_io_history, fig_dir, **viz_params)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            if label_mapping is not None:
                def dup_score_cols(scores):
                    num_cols = scores.shape[-1] + len(label_mapping)
                    col_idxs = torch.arange(num_cols)
                    for i, j in label_mapping.items():
                        col_idxs[i] = j
                    return scores[..., col_idxs]
                score_seq = dup_score_cols(score_seq)
            saveVariable(pred_seq.cpu().numpy(), f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq.cpu().numpy(), f'trial={trial_id}_score-seq')
            saveVariable(label_seq.cpu().numpy(), f'trial={trial_id}_true-label-seq')
        for io in test_io_history:
            saveTrialData(*io)

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
            fn=os.path.join(fig_dir, f'cvfold={cv_index}_train-plot.png')
        )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(fig_dir, f'cvfold={cv_index}_val-plot.png')
            )

        if eval_label_mapping is not None:
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
                return_io_history=True,
                label_mapping=eval_label_mapping
            )
            if independent_signals:
                test_io_history = joinSeqs(test_io_history)
            metric_str = '  '.join(str(m) for m in metric_dict.values())
            logger.info('[TST]  ' + metric_str)


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
