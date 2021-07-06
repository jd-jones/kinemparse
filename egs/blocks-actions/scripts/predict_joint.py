import os
import logging
import collections

import yaml
import numpy as np
import scipy
import torch

from mathtools import utils, torchutils, metrics


logger = logging.getLogger(__name__)


def make_event_assembly_transition_priors(event_vocab, assembly_vocab):
    def isValid(event, cur_assembly, next_assembly):
        is_valid = diff == event
        return is_valid

    num_events = len(event_vocab)
    num_assemblies = len(assembly_vocab)
    priors = np.zeros((num_events, num_assemblies, num_assemblies), dtype=bool)

    for j, cur_assembly in enumerate(assembly_vocab):
        for k, next_assembly in enumerate(assembly_vocab):
            try:
                diff = next_assembly - cur_assembly
            except ValueError:
                continue

            for i, event in enumerate(event_vocab):
                priors[i, j, k] = diff == event

    return priors


class FusionDataset(object):
    def __init__(
            self, seq_ids, event_scores_dir, assembly_scores_dir, ea_mapper,
            prefix='seq=', label_fn_format=None, feature_fn_format=None,
            device=None, modalities=None, out_type=None):

        # self.metadata = utils.loadMetadata(rgb_data_dir, rows=trial_ids)
        # self.vocab = utils.loadVariable('vocab', rgb_attributes_dir)

        self.out_type = out_type

        self.prefix = prefix
        self.label_fn_format = label_fn_format
        self.feature_fn_format = feature_fn_format

        self.event_scores_dir = event_scores_dir
        self.assembly_scores_dir = assembly_scores_dir
        self.device = device
        self.ea_mapper = ea_mapper

        self.seq_ids = seq_ids
        self.label_seqs = tuple(self.loadTargets(s_id) for s_id in seq_ids)

        self.pair_vocab = np.zeros((len(self.ea_mapper), 2), dtype=int)
        for k, v in self.ea_mapper.items():
            self.pair_vocab[v] = k
        self.event_indices, self.assembly_indices = self.pair_vocab.T

        self.feature_seqs = tuple(self.loadInputs(s_id) for s_id in seq_ids)

    def loadInputs(self, seq_id):
        trial_prefix = f"{self.prefix}{seq_id}"

        event_scores = utils.loadVariable(
            f"{trial_prefix}_{self.feature_fn_format}",
            self.event_scores_dir
        )

        assembly_scores = utils.loadVariable(
            f"{trial_prefix}_{self.feature_fn_format}",
            self.assembly_scores_dir
        )

        feats = np.stack(
            (event_scores[:, self.event_indices], assembly_scores[:, self.assembly_indices]),
            axis=-1
        )

        return feats

    def loadTargets(self, seq_id):
        trial_prefix = f"{self.prefix}{seq_id}"

        e = utils.loadVariable(
            f"{trial_prefix}_{self.label_fn_format}",
            self.event_scores_dir
        )

        a = utils.loadVariable(
            f"{trial_prefix}_{self.label_fn_format}",
            self.assembly_scores_dir
        )

        if self.out_type == 'parts':
            return e  # np.column_stack((e, a))

        labels = np.array(
            [utils.getIndex((i, j), self.ea_mapper) for i, j in zip(e, a)],
            dtype=int
        )

        return labels

    def getFold(self, cv_fold):
        return tuple(self.getSplit(split) for split in cv_fold)

    def getSplit(self, split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (self.feature_seqs, self.label_seqs, self.seq_ids)
        )
        return split_data


class Model(torch.nn.Module):
    def __init__(self, c, label_parts, mean=None, std=None, device=None, out_type=None):
        super().__init__()

        self.device = device
        self.out_type = out_type
        self.label_parts = label_parts
        self.label_to_event, self.label_to_assembly = self.invert_parts(label_parts)

        self.mean = torch.tensor(mean, dtype=torch.float, device=device)
        self.std = torch.tensor(std, dtype=torch.float, device=device)

        scale = torch.randn((c, 2), dtype=torch.float, device=device)
        shift = torch.randn((c,), dtype=torch.float, device=device)

        self.scale = torch.nn.Parameter(scale)
        self.shift = torch.nn.Parameter(shift)

    def invert_parts(self, label_parts):
        num_labels = label_parts.shape[0]
        num_events = label_parts[:, 0].max() + 1
        num_assemblies = label_parts[:, 1].max() + 1

        label_to_event = torch.full(
            (num_labels, num_events), -np.inf,
            dtype=torch.float, device=self.device
        )

        label_to_assembly = torch.full(
            (num_labels, num_assemblies), -np.inf,
            dtype=torch.float, device=self.device
        )

        for i, (e, a) in enumerate(label_parts):
            label_to_event[i, e] = 0
            label_to_assembly[i, a] = 0

        return label_to_event, label_to_assembly

    def forward(self, score_seq):
        if self.mean is not None:
            score_seq = score_seq - self.mean[None, None, :, :]

        if self.std is not None:
            score_seq = score_seq / self.std[None, None, :, :]

        score_seq = torch.einsum('btcm,cm->btc', score_seq, self.scale)
        # score_seq = torch.einsum('btcm,mm->btcm', score_seq, self.scale)
        score_seq = score_seq + self.shift[None, None, :]

        if self.out_type == 'parts':
            event_scores = self.to_parts(score_seq)
            return event_scores

        return score_seq

    def predict(self, score_seq):
        pred_seq = score_seq.argmax(axis=-1)
        # pred_seq = score_seq.argmax(axis=-2)
        return pred_seq

    def to_parts(self, score_seq):

        def makeOne(e):
            event_scores = torch.logsumexp(
                score_seq[:, :, :, None] + self.label_to_event[None, None, :, e],
                dim=2
            )
            return event_scores

        num_events = self.label_to_event.shape[-1]
        event_scores = torch.stack(tuple(makeOne(i) for i in range(num_events)), dim=-1)

        # assembly_scores = torch.logsumexp(
        #     score_seq[:, :, :, None] + self.label_to_assembly[None, None, :, :],
        #     dim=2
        # )

        return event_scores


def main(
        out_dir=None, assembly_scores_dir=None, event_scores_dir=None,
        labels_from='assemblies', plot_predictions=False,
        feature_fn_format='score-seq', label_fn_format='true-label-seq',
        only_fold=None, plot_io=None, prefix='seq=', stop_after=None,
        background_action='', stride=None, standardize_inputs=False,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        dataset_params={}, model_params={}, cv_params={}, train_params={}, viz_params={},
        metric_names=['Loss', 'Accuracy', 'Precision', 'Recall', 'F1'],
        results_file=None, sweep_param_name=None, out_type=None):

    event_scores_dir = os.path.expanduser(event_scores_dir)
    assembly_scores_dir = os.path.expanduser(assembly_scores_dir)
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

    misc_dir = os.path.join(out_dir, 'misc')
    if not os.path.exists(misc_dir):
        os.makedirs(misc_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name, to_dir=out_data_dir):
        return utils.saveVariable(var, var_name, to_dir)

    scores_dirs = {
        'events': event_scores_dir,
        'assemblies': assembly_scores_dir
    }
    data_dir = scores_dirs[labels_from]
    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix=f'{label_fn_format}.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {data_dir}")

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    # Load vocabs; create priors
    event_vocab = utils.loadVariable('vocab', event_scores_dir)
    assembly_vocab = utils.loadVariable('vocab', assembly_scores_dir)

    # vocabs = {
    #     'event_vocab': tuple(range(len(event_vocab))),
    #     'assembly_vocab': tuple(range(len(assembly_vocab)))
    # }

    try:
        event_priors = utils.loadVariable('event-priors', out_data_dir)
    except AssertionError:
        event_priors = make_event_assembly_transition_priors(event_vocab, assembly_vocab)
        utils.saveVariable(event_priors, 'event-priors', out_data_dir)
        np.savetxt(
            os.path.join(misc_dir, "event-transitions.csv"),
            np.column_stack(event_priors.nonzero()),
            delimiter=",", fmt='%d'
        )

    # _vocabs = (event_vocab, assembly_vocab)
    event_assembly_scores = np.log(event_priors)
    ea_scores = scipy.special.logsumexp(event_assembly_scores, axis=-1)
    nonzero_indices = np.column_stack(np.nonzero(~np.isinf(ea_scores)))
    # ea_vocab = tuple(
    #     tuple(_vocabs[i][j] for i, j in enumerate(indices))
    #     for indices in nonzero_indices_ea
    # )
    ea_mapper = {
        tuple(indices.tolist()): i
        for i, indices in enumerate(nonzero_indices)
    }
    # pair_vocab_size = len(ea_mapper)

    device = torchutils.selectDevice(gpu_dev_id)
    criterion = torch.nn.CrossEntropyLoss()

    dataset = FusionDataset(
        seq_ids, event_scores_dir, assembly_scores_dir, ea_mapper,
        prefix=prefix, feature_fn_format=feature_fn_format, label_fn_format=label_fn_format,
        out_type=out_type
    )

    def make_dataset(feats, labels, ids, shuffle=True):
        dataset = torchutils.SequenceDataset(
            feats, labels, device=device, labels_dtype=torch.long, seq_ids=ids,
            **dataset_params
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataset, loader

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        train_data, val_data, test_data = dataset.getFold(cv_fold)
        train_set, train_loader = make_dataset(*train_data, shuffle=True)
        test_set, test_loader = make_dataset(*test_data, shuffle=False)
        val_set, val_loader = make_dataset(*val_data, shuffle=True)

        all_feats = np.concatenate(train_data[0], axis=0)
        mean = all_feats.mean(axis=0)
        std = all_feats.std(axis=0)
        # mean = np.array([all_feats[..., i].mean(axis=0) for i in range(all_feats.shape[-1])])
        # std = np.array([all_feats[..., i].std(axis=0) for i in range(all_feats.shape[-1])])

        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        # cv_str = f'cvfold={cv_index}'
        model = Model(
            len(dataset.ea_mapper), dataset.pair_vocab,
            mean=mean, std=std, device=device, out_type=out_type
        )

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
            train_loader, val_loader=val_loader,
            device=device,
            metrics=metric_dict,
            train_epoch_log=train_epoch_log,
            val_epoch_log=val_epoch_log,
            **train_params
        )

        metric_dict = {name: metrics.makeMetric(name) for name in metric_names}
        test_io_history = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )
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
                    fn = os.path.join(io_fig_dir, f"{prefix}{seq_id}_model-io.png")
                    utils.plot_array(inputs.T, (gt_labels.T, preds.T), label_names, fn=fn)

        for batch in test_io_history:
            batch = tuple(
                x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                for x in batch
            )
            for pred_seq, score_seq, feat_seq, label_seq, trial_id in zip(*batch):
                saveVariable(pred_seq, f'{prefix}{trial_id}_pred-label-seq')
                saveVariable(score_seq, f'{prefix}{trial_id}_score-seq')
                saveVariable(label_seq, f'{prefix}{trial_id}_true-label-seq')

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
