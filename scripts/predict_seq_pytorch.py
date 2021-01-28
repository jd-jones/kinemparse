import os
import collections
import logging
import itertools
import math

import yaml
import torch

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
            self, input_dim, out_set_size, num_multiclass=None,
            binary_multiclass=False, tcn_channels=None, **tcn_kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.binary_multiclass = binary_multiclass
        self.num_multiclass = num_multiclass

        self.TCN = torchutils.TemporalConvNet(input_dim, tcn_channels, **tcn_kwargs)
        if self.num_multiclass is None:
            self.linear = torch.nn.Linear(tcn_channels[-1], self.out_set_size)
        else:
            self.linear = torch.nn.Linear(
                tcn_channels[-1],
                self.out_set_size * self.num_multiclass
            )

        logger.info(
            f'Initialized TCN classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq, return_feats=False):
        tcn_out = self.TCN(input_seq).transpose(1, 2)
        linear_out = self.linear(tcn_out)

        if self.num_multiclass is not None:
            linear_out = linear_out.view(
                *linear_out.shape[0:2], self.out_set_size, self.num_multiclass
            )

        if return_feats:
            return linear_out, tcn_out
        return linear_out

    def predict(self, outputs, return_scores=False):
        """ outputs has shape (num_batch, num_samples, num_classes, ...) """

        if self.binary_multiclass:
            return (outputs > 0.5).float()

        __, preds = torch.max(outputs, dim=2)

        if return_scores:
            scores = torch.nn.softmax(outputs, dim=1)
            return preds, scores

        return preds


class LstmClassifier(torch.nn.Module):
    def __init__(
            self, input_dim, out_set_size, num_multiclass=None,
            binary_multiclass=False, hidden_dim=512, **lstm_kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.binary_multiclass = binary_multiclass
        self.num_multiclass = num_multiclass

        num_directions = 2
        self.LSTM = torch.nn.LSTM(input_dim, hidden_dim, **lstm_kwargs)

        if self.num_multiclass is None:
            self.linear = torch.nn.Linear(num_directions * hidden_dim, self.out_set_size)
        else:
            self.linear = torch.nn.Linear(
                num_directions * hidden_dim,
                self.out_set_size * self.num_multiclass
            )

        logger.info(
            f'Initialized LSTM classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq, return_feats=False):
        batch_size = input_seq.shape[0]
        hidden_size = 512
        num_layers = 1
        num_directions = 2
        h0 = torch.randn(
            num_layers * num_directions, batch_size, hidden_size,
            device=input_seq.device
        )
        c0 = torch.randn(
            num_layers * num_directions, batch_size, hidden_size,
            device=input_seq.device
        )
        lstm_out, (hn, cn) = self.LSTM(input_seq, (h0, c0))
        # import pdb; pdb.set_trace()
        linear_out = self.linear(lstm_out)

        if self.num_multiclass is not None:
            linear_out = linear_out.view(
                *linear_out.shape[0:2], self.out_set_size, self.num_multiclass
            )

        if return_feats:
            return linear_out, lstm_out
        return linear_out

    def predict(self, outputs, return_scores=False):
        """ outputs has shape (num_batch, num_samples, num_classes, ...) """

        if self.binary_multiclass:
            return (outputs > 0.5).float()

        __, preds = torch.max(outputs, dim=2)

        if return_scores:
            scores = torch.nn.softmax(outputs, dim=1)
            return preds, scores

        return preds


def splitSeqs(feature_seqs, label_seqs, trial_ids, active_only=False):
    num_signals = label_seqs[0].shape[1]
    if num_signals >= 100:
        raise ValueError("{num_signals} signals will cause overflow in sequence ID (max is 99)")

    def validate(seqs):
        return all(seq.shape[1] == num_signals for seq in seqs)
    all_valid = all(validate(x) for x in (feature_seqs, label_seqs))
    if not all_valid:
        raise AssertionError("Features and labels don't all have the same number of sequences")

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

    feature_seqs = splitSeq(map(lambda x: x.swapaxes(0, 1), feature_seqs))
    label_seqs = splitSeq(map(lambda x: x.T, label_seqs))

    if active_only:
        is_active = tuple(map(lambda x: x.any(), label_seqs))

        def filterInactive(arrays):
            return tuple(arr for arr, act in zip(arrays, is_active) if act)
        return tuple(map(filterInactive, (feature_seqs, label_seqs, trial_ids)))

    return feature_seqs, label_seqs, trial_ids


def joinSeqs(batches):
    def stack(seq):
        stacked = torch.stack(seq, dim=-1)
        return stacked[None, ...]

    all_seqs = collections.defaultdict(dict)
    for batch in batches:
        for b in zip(*batch):
            i = b[-1]
            seqs = b[:-1]

            seq_id, trial_id = math.modf(i)
            seq_id = int(round(seq_id * 100))
            trial_id = int(round(trial_id))

            all_seqs[trial_id][seq_id] = seqs

    for trial_id, seq_dict in all_seqs.items():
        seqs = (seq_dict[k] for k in sorted(seq_dict.keys()))
        seqs = map(stack, zip(*seqs))
        yield tuple(seqs) + ((trial_id,),)


def main(
        out_dir=None, data_dir=None, model_name=None, predict_mode='classify',
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        independent_signals=None, active_only=None,
        feature_fn_format='feature-seq.pkl', label_fn_format='label_seq.pkl',
        dataset_params={}, model_params={}, cv_params={}, train_params={}, viz_params={},
        metric_names=['Loss', 'Accuracy', 'Precision', 'Recall', 'F1'],
        plot_predictions=None, results_file=None, sweep_param_name=None):

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

    def saveVariable(var, var_name, to_dir=out_data_dir):
        return utils.saveVariable(var, var_name, to_dir)

    # Load data
    device = torchutils.selectDevice(gpu_dev_id)
    trial_ids = utils.getUniqueIds(
        data_dir, prefix='trial=', suffix=feature_fn_format,
        to_array=True
    )
    dataset = utils.CvDataset(
        trial_ids, data_dir,
        feature_fn_format=feature_fn_format, label_fn_format=label_fn_format
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)
    utils.saveVariable(dataset.vocab, 'vocab', out_data_dir)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(trial_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    if predict_mode == 'binary multiclass':
        criterion = torchutils.BootstrappedCriterion(
            0.25, base_criterion=torch.nn.functional.binary_cross_entropy_with_logits,
        )
        labels_dtype = torch.float
    elif predict_mode == 'multiclass':
        criterion = torch.nn.CrossEntropyLoss()
        labels_dtype = torch.long
    elif predict_mode == 'classify':
        criterion = torch.nn.CrossEntropyLoss()
        labels_dtype = torch.long
    else:
        raise AssertionError()

    def make_dataset(feats, labels, ids, shuffle=True):
        dataset = torchutils.SequenceDataset(
            feats, labels, device=device, labels_dtype=labels_dtype, seq_ids=ids,
            **dataset_params
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset, loader

    for cv_index, cv_fold in enumerate(cv_folds):
        train_data, val_data, test_data = dataset.getFold(cv_fold)
        if independent_signals:
            train_data = splitSeqs(*train_data, active_only=active_only)
            val_data = splitSeqs(*val_data, active_only=active_only)
            test_data = splitSeqs(*test_data, active_only=False)
        train_set, train_loader = make_dataset(*train_data, shuffle=True)
        test_set, test_loader = make_dataset(*test_data, shuffle=False)
        val_set, val_loader = make_dataset(*val_data, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(dataset.trial_ids)} total '
            f'({len(train_set)} train, {len(val_set)} val, {len(test_set)} test)'
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
            if predict_mode == 'multiclass':
                num_multiclass = train_set[0][1].shape[-1]
                output_dim = max([
                    train_set.num_label_types,
                    test_set.num_label_types,
                    val_set.num_label_types
                ])
            else:
                num_multiclass = None
            model = TcnClassifier(
                input_dim, output_dim, num_multiclass=num_multiclass,
                **model_params
            ).to(device=device)
        elif model_name == 'LSTM':
            if predict_mode == 'multiclass':
                num_multiclass = train_set[0][1].shape[-1]
                output_dim = max([
                    train_set.num_label_types,
                    test_set.num_label_types,
                    val_set.num_label_types
                ])
            else:
                num_multiclass = None
            model = LstmClassifier(
                input_dim, output_dim, num_multiclass=num_multiclass,
                **model_params
            ).to(device=device)
        else:
            raise AssertionError()

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
        test_io_history = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )
        if independent_signals:
            test_io_history = tuple(joinSeqs(test_io_history))

        logger.info('[TST]  ' + '  '.join(str(m) for m in metric_dict.values()))
        utils.writeResults(
            results_file, {k: v.value for k, v in metric_dict.items()},
            sweep_param_name, model_params
        )

        if plot_predictions:
            io_fig_dir = os.path.join(fig_dir, 'model-io')
            if not os.path.exists(io_fig_dir):
                os.makedirs(io_fig_dir)

            label_names = ('gt', 'pred')
            preds, scores, inputs, gt_labels, ids = zip(*test_io_history)
            for batch in test_io_history:
                batch = tuple(
                    x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    for x in batch
                )
                for preds, _, inputs, gt_labels, seq_id in zip(*batch):
                    fn = os.path.join(io_fig_dir, f"trial={seq_id}_model-io.png")
                    utils.plot_array(inputs, (gt_labels.T, preds.T), label_names, fn=fn)

        for batch in test_io_history:
            batch = tuple(
                x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                for x in batch
            )
            for pred_seq, score_seq, feat_seq, label_seq, trial_id in zip(*batch):
                saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
                saveVariable(score_seq, f'trial={trial_id}_score-seq')
                saveVariable(label_seq, f'trial={trial_id}_true-label-seq')

        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        train_fig_dir = os.path.join(fig_dir, 'train-plots')
        if not os.path.exists(train_fig_dir):
            os.makedirs(train_fig_dir)

        if train_epoch_log:
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
