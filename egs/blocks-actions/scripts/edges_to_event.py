import os
import logging
import collections

import yaml
import numpy as np
import scipy
# from matplotlib import pyplot as plt

import LCTM.metrics

from mathtools import utils

# from blocks.core import blockassembly
from blocks.core import labels as labels_lib


logger = logging.getLogger(__name__)


class AttributeClassifier(object):
    def __init__(self, edge_attrs):
        """
        Parameters
        ----------
        action_part_counts : np.ndarray of int with shape (NUM_ACTIONS, NUM_PARTS)
        """

        self.edge_attrs = np.reshape(edge_attrs, (edge_attrs.shape[0], -1))

    def forward(self, edge_scores):
        """ Combine action and part scores into event scores.

        Parameters
        ----------
        action_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS)
        part_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_PARTS)

        Returns
        -------
        event_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS, NUM_PARTS)
        """

        edge_scores = np.reshape(edge_scores, (edge_scores.shape[0], -1))

        event_scores = edge_scores @ self.edge_attrs.T

        # event_scores = scipy.special.log_softmax(event_scores, axis=1)

        return event_scores

    def predict(self, outputs):
        """ Choose the best labels from an array of output activations.

        Parameters
        ----------
        outputs : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS, NUM_PARTS)

        Returns
        -------
        preds : np.ndarray of int with shape (NUM_SAMPLES, 2)
            preds[:, 0] contains the index of the action predicted for each sample.
            preds[:, 1] contains the index of the part predicted for each sample.
        """

        preds = outputs.argmax(axis=1)
        return preds


class DataLoader(object):
    def __init__(self, events_dir, edges_dir, prefix='seq='):
        self.prefix = prefix

        self.events_dir = events_dir
        self.edges_dir = edges_dir

        self.seq_ids = utils.getUniqueIds(
            self.events_dir, prefix=prefix, suffix='label-seq.*',
            to_array=True
        )

        self.vocab = utils.loadVariable('vocab', self.events_dir)
        for i in range(len(self.vocab)):
            sign = self.vocab[i].sign
            if isinstance(sign, np.ndarray):
                self.vocab[i].sign = np.sign(sign.sum())

    def __getitem__(self, index):
        if isinstance(index, collections.abc.Iterable):
            return tuple(self.__getitem__(i) for i in index)

        score_seq = self._getScores(index)
        label_seq = self._getLabels(index)

        score_seq = scipy.special.softmax(score_seq, axis=1)
        score_seq = 2 * score_seq - 1

        return score_seq, label_seq

    def _getScores(self, index):
        if isinstance(index, collections.abc.Iterable):
            return tuple(self._getScores(i) for i in index)

        trial_prefix = f"{self.prefix}{index}"
        score_seq = utils.loadVariable(
            f"{trial_prefix}_score-seq",
            self.edges_dir
        )

        return score_seq

    def _getLabels(self, index):
        if isinstance(index, collections.abc.Iterable):
            return tuple(self._getLabels(i) for i in index)

        trial_prefix = f"{self.prefix}{index}"

        dir_ = self.events_dir
        fn = f"{trial_prefix}_label-seq"

        return utils.loadVariable(fn, dir_)


def eval_metrics(pred_seq, true_seq, name_suffix='', name_prefix='', append_to={}):
    acc = (pred_seq == true_seq).astype(float).mean()
    edit = LCTM.metrics.edit_score(pred_seq, true_seq) / 100
    overlap = LCTM.metrics.overlap_score(pred_seq, true_seq) / 100

    metric_dict = {
        name_prefix + 'Accuracy' + name_suffix: acc,
        name_prefix + 'Edit Score' + name_suffix: edit,
        name_prefix + 'Overlap Score' + name_suffix: overlap
    }

    append_to.update(metric_dict)
    return append_to


def makeEdges(vocab):
    def to_onehot(attrs):
        dim = attrs.max() + 1
        num_events, num_edges = attrs.shape
        onehot_attrs = np.zeros((num_events, dim, num_edges), dtype=bool)
        rows = np.arange(num_events)
        for i_edge in range(num_edges):
            onehot_attrs[rows, attrs[:, i_edge], i_edge] = True
        return onehot_attrs

    parts_vocab, edge_diffs = labels_lib.make_parts_vocab(
        vocab, lower_tri_only=True, append_to_vocab=False
    )
    signs = np.array([a.sign for a in vocab], dtype=int)
    signs[signs == -1] = 2
    edge_diffs = np.concatenate((edge_diffs, signs[:, None]), axis=1)

    # Convert to one-hot and scale
    edge_diffs = to_onehot(edge_diffs).astype(float)
    edge_diffs = 2 * edge_diffs - 1

    return edge_diffs


def make_attrs(event_vocab):
    def sign_to_onehot(sign):
        onehot = np.zeros(4, dtype=bool)
        index = 2 if sign == -1 else sign
        onehot[index] = True
        return onehot

    for i in range(len(event_vocab)):
        e = event_vocab[i]
        if isinstance(e.sign, np.ndarray):
            event_vocab[i].sign = np.sign(e.sign.sum())

    action_attrs = np.row_stack(
        tuple(sign_to_onehot(a.sign) for a in event_vocab)
    )
    part_attrs = np.row_stack(
        tuple(a.symmetrized_connections.any(axis=0) for a in event_vocab)
    )

    # Convert to floating point in [-1, 1]
    part_attrs = 2 * part_attrs.astype(float) - 1
    action_attrs = 2 * action_attrs.astype(float) - 1

    return action_attrs, part_attrs


def main(
        out_dir=None, event_scores_dir=None, edge_scores_dir=None,
        only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    event_scores_dir = os.path.expanduser(event_scores_dir)
    edge_scores_dir = os.path.expanduser(edge_scores_dir)
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

    loader = DataLoader(
        event_scores_dir, edge_scores_dir,
        prefix=prefix
    )
    logger.info(f"Loaded ids for {len(loader.seq_ids)} sequences")

    metadata = utils.loadMetadata(loader.events_dir)

    # loader.vocabs['event'] = [blockassembly.AssemblyAction()] + loader.vocabs['event']

    utils.saveMetadata(metadata, out_data_dir)
    utils.saveVariable(loader.vocab, 'vocab', out_data_dir)

    edge_attrs = makeEdges(loader.vocab)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(loader.seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        model = AttributeClassifier(edge_attrs)

        for i in test_indices:
            seq_id = loader.seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            edge_score_seq, true_label_seq = loader[seq_id]
            score_seq = model.forward(edge_score_seq)
            pred_label_seq = model.predict(score_seq)

            metric_dict = eval_metrics(pred_label_seq, true_label_seq)
            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")

            seq_id_str = f"seq={seq_id}"
            utils.saveVariable(score_seq, f'{seq_id_str}_score-seq', out_data_dir)
            utils.saveVariable(true_label_seq, f'{seq_id_str}_true-label-seq', out_data_dir)
            utils.saveVariable(pred_label_seq, f'{seq_id_str}_pred-label-seq', out_data_dir)
            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if plot_io:
                utils.plot_array(
                    score_seq.T,
                    (true_label_seq.T, pred_label_seq.T), ('true', 'pred'),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}.png")
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
