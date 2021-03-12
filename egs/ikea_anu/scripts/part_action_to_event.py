import os
import logging
import collections

import yaml
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import LCTM.metrics

from mathtools import utils


logger = logging.getLogger(__name__)


class AttributeClassifier(object):
    def __init__(self, action_part_counts):
        """
        Parameters
        ----------
        action_part_counts : np.ndarray of int with shape (NUM_ACTIONS, NUM_PARTS)
        """
        self.action_part_counts = action_part_counts
        self.feat_ap = np.log((self.action_part_counts > 0).astype(float))

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
            action_scores[:, :, None]
            + part_scores[:, None, :]
            + self.feat_ap[None, :, :]
        )
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
        pair_scores = outputs.reshape(outputs.shape[0], -1)
        pair_preds = pair_scores.argmax(axis=-1)
        # preds = np.column_stack(np.unravel_index(pair_preds, outputs.shape[1:]))
        preds = np.unravel_index(pair_preds, outputs.shape[1:])
        return preds


class DataLoader(object):
    def __init__(self, data_dirs, scores_dirs, prefix='seq='):
        self.prefix = prefix

        self.data_dirs = data_dirs
        self.scores_dirs = scores_dirs

        self.seq_ids = utils.getUniqueIds(
            self.data_dirs['event'], prefix=prefix, suffix='labels.*',
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
        # event_labels = self._getLabels(index, label_name='event')
        action_labels = self._getLabels(index, label_name='action')
        part_labels = self._getLabels(index, label_name='part')
        return action_score_seq, part_score_seq, action_labels, part_labels

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


def count_action_part_bigrams(action_vocab, part_vocab, action_labels, part_labels):
    """ Count the number of times each (action, part) pair co-occurs.

    Parameters
    ----------
    action_vocab : iterable(string) with len NUM_ACTIONS
    part_vocab : iterable(string) with len NUM_PARTS
    action_labels : np.ndarray of int with shape (NUM_SAMPLES,)
    part_labels : np.ndarray of int with shape (NUM_SAMPLES,)

    Returns
    -------
    counts : np.ndarray of int with shape (NUM_ACTIONS, NUM_PARTS)
    """
    counts = np.zeros((len(action_vocab), len(part_vocab)), dtype=int)

    for part_activity_row, action_index in zip(part_labels, action_labels):
        # FIXME: Convert hardcoded col_index -> part_index mapping to something flexible.
        #    The way I do it below will break if '' is not the first element in vocab.
        if not part_activity_row.any():
            part_index = 0
            counts[action_index, part_index] += 1
            continue

        for i, is_active in enumerate(part_activity_row):
            part_index = i + 1
            counts[action_index, part_index] += int(is_active)

    return counts


def plot_action_part_bigrams(action_vocab, part_vocab, counts, fn):
    """
    Parameters
    ----------
    action_vocab : iterable(string) with len NUM_ACTIONS
    part_vocab : iterable(string) with len NUM_PARTS
    counts : np.ndarray of int with shape (NUM_ACTIONS, NUM_PARTS)
    fn : string
    """
    plt.matshow(counts)
    plt.xticks(ticks=range(len(part_vocab)), labels=part_vocab, rotation='vertical')
    plt.yticks(ticks=range(len(action_vocab)), labels=action_vocab)
    plt.savefig(fn, bbox_inches='tight')


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


def event_to_ap(event_labels):
    raise NotImplementedError()


def ap_to_event(action_labels, part_labels, map_df, vocabs):
    def to_event(i_action, i_part):
        action_name = vocabs['action'][i_action]
        part_name = vocabs['part'][i_part]
        matches_action = map_df['action'] == action_name
        if part_name:
            matches_part = map_df[f'{part_name}_active']
            matches_event = matches_action & matches_part
        else:
            matches_event = matches_action
        event_name = map_df['event'].loc[matches_event].iloc[0]
        i_event = vocabs['event'].index(event_name)
        return i_event

    i_event = np.array(
        [
            to_event(i_action, i_part)
            for i_action, i_part in zip(action_labels, part_labels)
        ], dtype=int
    )

    return i_event


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

    all_metrics = collections.defaultdict(list)

    if as_atomic_events:
        map_df = pd.read_csv(
            os.path.join(data_dir, 'labels', 'event-vocab.csv'),
            index_col=False, keep_default_na=False
        )
    else:
        map_df = pd.DataFrame(
            [
                [f"{action}({part})", action] + [part == n for n in loader.vocabs['part'] if n]
                for action in loader.vocabs['action']
                for part in loader.vocabs['part']
            ],
            columns=['event', 'action'] + [f"{n}_active" for n in loader.vocabs['part'] if n]
        )
    event_vocab = map_df['event'].to_list()
    utils.saveVariable(event_vocab, 'vocab', out_data_dir)

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

        action_part_counts = sum(
            count_action_part_bigrams(loader.vocabs['action'], loader.vocabs['part'], la, lp)
            for la, lp in loader._getLabels(
                train_indices, label_name=['action', 'part'],
                load_from_data=True
            )
        )
        model = AttributeClassifier(action_part_counts)

        plot_action_part_bigrams(
            loader.vocabs['action'], loader.vocabs['part'],
            (model.action_part_counts > 0).astype(int),
            os.path.join(fig_dir, f'cvfold={cv_index}_action-part-coocurrence.png')
        )

        for i in test_indices:
            seq_id = loader.seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            action_score_seq, part_score_seq, true_action_seq, true_part_seq = loader[seq_id]
            event_score_seq = model.forward(action_score_seq, part_score_seq)
            pred_action_seq, pred_part_seq = model.predict(event_score_seq)

            true_event_seq = ap_to_event(true_action_seq, true_part_seq, map_df, loader.vocabs)
            pred_event_seq = ap_to_event(pred_action_seq, pred_part_seq, map_df, loader.vocabs)

            metric_dict = {}
            metric_dict = eval_metrics(
                pred_event_seq, true_event_seq,
                name_prefix='Event ', append_to=metric_dict
            )
            metric_dict = eval_metrics(
                pred_action_seq, true_action_seq,
                name_prefix='Action ', append_to=metric_dict
            )
            metric_dict = eval_metrics(
                pred_part_seq, true_part_seq,
                name_prefix='Part ', append_to=metric_dict
            )
            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")
                all_metrics[name].append(value)

            seq_id_str = f"seq={seq_id}"
            utils.saveVariable(event_score_seq, f'{seq_id_str}_score-seq', out_data_dir)
            utils.saveVariable(true_event_seq, f'{seq_id_str}_true-label-seq', out_data_dir)
            utils.saveVariable(pred_event_seq, f'{seq_id_str}_pred-label-seq', out_data_dir)
            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if plot_io:
                utils.plot_array(
                    event_score_seq.reshape(event_score_seq.shape[0], -1).T,
                    (true_event_seq, pred_event_seq), ('true', 'pred'),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}_event.png")
                )
                utils.plot_array(
                    action_score_seq.T, (true_action_seq, pred_action_seq), ('true', 'pred'),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}_action.png")
                )
                utils.plot_array(
                    part_score_seq.T, (true_part_seq, pred_part_seq), ('true', 'pred'),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}_part.png")
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
