import os
import logging
import collections

import yaml
import numpy as np
import scipy
# from matplotlib import pyplot as plt

import LCTM.metrics

from mathtools import utils

from blocks.core import blockassembly


logger = logging.getLogger(__name__)


class AttributeClassifier(object):
    def __init__(self, action_attrs, part_attrs, action_weight=1, part_weight=1):
        """
        Parameters
        ----------
        action_part_counts : np.ndarray of int with shape (NUM_ACTIONS, NUM_PARTS)
        """

        self.action_attrs = action_attrs
        self.part_attrs = part_attrs

        self.action_weight = action_weight
        self.part_weight = part_weight

    def forward(self, action_scores, part_scores):
        """ Combine action and part scores into event scores.

        Parameters
        ----------
        action_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS)
        part_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_PARTS)

        Returns
        -------
        event_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS, NUM_PARTS)
        """

        event_scores = (
            self.action_weight * (2 * action_scores - 1) @ self.action_attrs.T
            + self.part_weight * (2 * part_scores - 1) @ self.part_attrs.T
        )

        event_scores = scipy.special.softmax(event_scores, axis=1)

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
    def __init__(self, data_dirs, scores_dirs, prefix='seq='):
        self.prefix = prefix

        self.data_dirs = data_dirs
        self.scores_dirs = scores_dirs

        self.seq_ids = utils.getUniqueIds(
            self.scores_dirs['action'], prefix=prefix, suffix='true-label-seq.*',
            to_array=True
        )

        self.vocabs = {
            name: utils.loadVariable('vocab', dir_)
            for name, dir_ in self.data_dirs.items()
        }

    def __getitem__(self, index):
        if isinstance(index, collections.abc.Iterable):
            return tuple(self.__getitem__(i) for i in index)

        action_score_seq, part_score_seq = self._getScores(index, label_name=['action', 'part'])
        event_labels = self._getLabels(index, label_name='event')

        action_score_seq = scipy.special.softmax(action_score_seq, axis=1)
        part_score_seq = scipy.special.expit(part_score_seq)

        return action_score_seq, part_score_seq, event_labels

    def _getScores(self, index, label_name='event'):
        if isinstance(index, collections.abc.Iterable):
            return tuple(self._getScores(i, label_name=label_name) for i in index)

        if isinstance(label_name, collections.abc.Iterable) and not isinstance(label_name, str):
            return tuple(self._getScores(index, label_name=ln) for ln in label_name)

        trial_prefix = f"{self.prefix}{index}"
        score_seq = utils.loadVariable(
            f"{trial_prefix}_score-seq",
            self.scores_dirs[label_name]
        )

        return score_seq

    def _getLabels(self, index, label_name='event', load_from_data=False):
        if isinstance(index, collections.abc.Iterable):
            return tuple(
                self._getLabels(i, label_name=label_name, load_from_data=load_from_data)
                for i in index
            )

        if isinstance(label_name, collections.abc.Iterable) and not isinstance(label_name, str):
            return tuple(
                self._getLabels(index, label_name=ln, load_from_data=load_from_data)
                for ln in label_name
            )

        trial_prefix = f"{self.prefix}{index}"

        if load_from_data:
            dir_ = self.data_dirs[label_name]
            fn = f"{trial_prefix}_labels"
        else:
            dir_ = self.scores_dirs[label_name]
            fn = f"{trial_prefix}_true-label-seq"

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
        out_dir=None, data_dir=None,
        event_scores_dir=None, action_scores_dir=None, part_scores_dir=None,
        as_atomic_events=False, only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    event_scores_dir = os.path.expanduser(event_scores_dir)
    action_scores_dir = os.path.expanduser(action_scores_dir)
    part_scores_dir = os.path.expanduser(part_scores_dir)
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
        {
            'event': os.path.join(data_dir, 'event-dataset'),
            'action': os.path.join(data_dir, 'action-dataset'),
            'part': os.path.join(data_dir, 'part-dataset')
        },
        {
            'event': event_scores_dir,
            'action': action_scores_dir,
            'part': part_scores_dir
        },
        prefix=prefix
    )
    logger.info(f"Loaded ids for {len(loader.seq_ids)} sequences from {data_dir}")

    metadata = utils.loadMetadata(loader.data_dirs['event'])

    loader.vocabs['event'] = [blockassembly.AssemblyAction()] + loader.vocabs['event']

    utils.saveMetadata(metadata, out_data_dir)
    utils.saveVariable(loader.vocabs['event'], 'vocab', out_data_dir)

    action_attrs, part_attrs = make_attrs(loader.vocabs['event'])

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

        model = AttributeClassifier(action_attrs, part_attrs)

        for i in test_indices:
            seq_id = loader.seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            action_score_seq, part_score_seq, true_label_seq = loader[seq_id]
            score_seq = model.forward(action_score_seq, part_score_seq)
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
