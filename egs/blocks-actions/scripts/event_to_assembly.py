import os
import logging
import json
import itertools
import collections

import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy

import LCTM.metrics

from kinemparse import decode
from mathtools import utils  # , metrics


logger = logging.getLogger(__name__)


def loadPartInfo(assembly_data_dir, event_attr_fn, connection_attr_fn, background_action=''):
    assembly_vocab = utils.loadVariable('vocab', assembly_data_dir)
    assembly_attrs = utils.loadVariable('assembly-attrs', assembly_data_dir)
    # joint_vocab = tuple(tuple(sorted(joint)) for joint in data['joint_vocab'])
    # part_vocab = tuple(data['part_vocab'])
    joint_vocab = None  # FIXME
    part_vocab = None  # FIXME
    part_categories = {}

    final_assembly_attrs = assembly_attrs  # FIXME

    connection_probs, action_vocab, connection_vocab = connection_attrs_to_probs(
        pd.read_csv(connection_attr_fn, index_col=False, keep_default_na=False),
        # normalize=True
    )

    with open(event_attr_fn, 'rt') as file_:
        event_probs, event_vocab = event_attrs_to_probs(
            json.load(file_),
            part_categories, action_vocab, joint_vocab,
            background_action=background_action,
            # normalize=True
        )

    assembly_probs, assembly_vocab = assembly_attrs_to_probs(
        assembly_attrs, joint_vocab, connection_vocab,
        # normalize=True
    )

    num_assemblies = len(assembly_attrs)
    transition_probs = np.zeros((num_assemblies + 1, num_assemblies + 1), dtype=float)
    transition_probs[0, :-1] = prior_probs(assembly_attrs)
    transition_probs[1:, -1] = final_probs(assembly_attrs, final_assembly_attrs)
    transition_probs[1:, :-1] = assembly_transition_probs(
        assembly_attrs,
        allow_self_transitions=True
    )

    probs = (event_probs, connection_probs, assembly_probs, transition_probs)
    vocabs = {
        'event_vocab': event_vocab,
        'part_vocab': part_vocab,
        'action_vocab': action_vocab,
        'joint_vocab': joint_vocab,
        'connection_vocab': connection_vocab,
        'assembly_vocab': assembly_vocab
    }
    return probs, vocabs


def connection_attrs_to_probs(action_attrs, normalize=False):
    """
    Parameters
    ----------
    action_attrs :

    Returns
    -------
    scores :
    action_vocab :
    connecion_vocab :
    """

    # Log-domain values for zero and one
    # zero = -np.inf
    # one = 0
    zero = 0

    tx_sep = '->'
    tx_cols = [name for name in action_attrs.columns if tx_sep in name]
    tx_vocab = tuple(
        tuple(int(x) for x in col_name.split(tx_sep))
        for col_name in tx_cols
    )
    connection_vocab = tuple(
        sorted(frozenset().union(*[frozenset(t) for t in tx_vocab]))
    )
    action_vocab = tuple(action_attrs['action'].to_list())

    num_actions = len(action_vocab)
    num_connections = len(connection_vocab)
    scores = np.full((num_actions, num_connections, num_connections), zero, dtype=float)

    action_attrs = action_attrs.set_index('action')
    for i_action, action_name in enumerate(action_vocab):
        tx_weights = tuple(action_attrs.loc[action_name][c] for c in tx_cols)
        for i_edge, tx_weight in enumerate(tx_weights):
            i_conn_cur, i_conn_next = tx_vocab[i_edge]
            scores[i_action, i_conn_cur, i_conn_next] = tx_weight

    if normalize:
        # Compute P(c_cur, c_next | action)
        action_counts = scores.reshape((scores.shape[0], -1)).sum(axis=1)
        scores /= action_counts[:, None, None]

    return scores, action_vocab, connection_vocab


def event_attrs_to_probs(
        event_attrs, part_categories, action_vocab, joint_vocab,
        background_action='', normalize=False):
    def expand_part_categories(and_rules):
        def expand(parts):
            if isinstance(parts, str):
                parts = [parts]
            parts = set(x for name in parts for x in part_categories.get(name, [name]))
            return sorted(parts)

        def to_or_rules(joint):
            new_joint = [expand(parts) for parts in joint]
            joints_set = frozenset([frozenset(prod) for prod in itertools.product(*new_joint)])
            joints_tup = tuple(tuple(sorted(x)) for x in joints_set)
            return joints_tup

        and_or_rules = tuple(to_or_rules(joint) for joint in and_rules if len(joint) > 1)
        or_and_rules = tuple(itertools.product(*and_or_rules))
        return or_and_rules

    # Log-domain values for zero and one
    # zero = -np.inf
    # one = 0
    zero = 0
    one = 1

    # event index --> all data
    event_vocab = tuple(event_attrs.keys())

    action_integerizer = {x: i for i, x in enumerate(action_vocab)}

    num_actions = len(action_vocab)
    num_joints = len(joint_vocab)
    scores = []
    for i_event, event_name in enumerate(event_vocab):
        event_data = event_attrs[event_name]
        action_name = event_data['action']
        and_rules = event_data['parts']
        or_and_rules = expand_part_categories(and_rules)

        num_modes = len(or_and_rules)
        event_scores = np.full((num_modes, num_actions, num_joints), zero, dtype=float)
        for i_mode, active_joints in enumerate(or_and_rules):
            for i_joint, joint in enumerate(joint_vocab):
                action = action_name if joint in active_joints else background_action
                i_action = action_integerizer[action]
                event_scores[i_mode, i_action, i_joint] = one
        scores.append(event_scores)

    if normalize:
        # Compute P(action, joint | event)
        # event_counts = scores.reshape((scores.shape[0], -1)).sum(axis=1)
        # scores /= event_counts[:, None, None]
        raise NotImplementedError()

    return scores, event_vocab


def assembly_attrs_to_probs(
        assembly_attrs, joint_vocab, connection_vocab,
        disconnected_val=0, connected_val=1, normalize=False):
    """
    Parameters
    ----------
    assembly_attrs :
    joint_vocab :
    connection_vocab :

    Returns
    -------
    scores :
    assembly_vocab :
    """

    # Log-domain values for zero and one
    # zero = -np.inf
    # one = 0
    zero = 0
    one = 1

    assembly_vocab = tuple(
        tuple(sorted(set(part for joint in a for part in joint)))
        # i
        for i, a in enumerate(assembly_attrs)
    )

    num_assemblies = len(assembly_vocab)
    num_joints = len(joint_vocab)
    num_connections = len(connection_vocab)

    joint_integerizer = {x: i for i, x in enumerate(joint_vocab)}
    connection_integerizer = {x: i for i, x in enumerate(connection_vocab)}

    disconnected_index = connection_integerizer[disconnected_val]
    connected_index = connection_integerizer[connected_val]

    scores = np.full((num_assemblies, num_joints, num_connections), zero, dtype=float)
    for i_assembly, _ in enumerate(assembly_vocab):
        joints = assembly_attrs[i_assembly]
        connection_indices = np.full((num_joints,), disconnected_index, dtype=int)
        for joint in joints:
            i_joint = joint_integerizer[joint]
            connection_indices[i_joint] = connected_index
        for i_joint, i_connection in enumerate(connection_indices):
            scores[i_assembly, i_joint, i_connection] = one

    if normalize:
        # Compute P(assembly | joint, connection)
        event_counts = scores.sum(axis=0)
        scores /= event_counts[None, :, :]

    return scores, assembly_vocab


def prior_probs(assembly_attrs):
    zero = 0
    one = 1

    def score(joints):
        if joints:
            return zero
        return one

    num_assemblies = len(assembly_attrs)
    scores = np.full((num_assemblies,), zero, dtype=float)
    for i, joints in enumerate(assembly_attrs):
        scores[i] = score(joints)
    return scores


def final_probs(assembly_attrs, final_assembly_attrs):
    zero = 0
    one = 1

    def score(joints):
        if joints in final_assembly_attrs:
            return one
        return zero

    num_assemblies = len(assembly_attrs)
    scores = np.full((num_assemblies,), zero, dtype=float)
    for i, joints in enumerate(assembly_attrs):
        scores[i] = score(joints)
    return scores


def assembly_transition_probs(assembly_attrs, allow_self_transitions=False):
    """
    """

    zero = 0
    one = 1

    def score(cur_joints, next_joints):
        def num_diff(lhs, rhs):
            return sum(x not in rhs for x in lhs)

        def get_parts(joints):
            return frozenset(p for joint in joints for p in joint)

        if allow_self_transitions and cur_joints == next_joints:
            return one

        if num_diff(cur_joints, next_joints) != 0:
            return zero

        if not cur_joints and len(next_joints) == 1:
            return one

        cur_parts = get_parts(cur_joints)
        next_parts = get_parts(next_joints)
        if num_diff(next_parts, cur_parts) != 1:
            return zero
        return one

    num_assemblies = len(assembly_attrs)
    scores = np.full((num_assemblies, num_assemblies), zero, dtype=float)
    for i_cur, cur_joints in enumerate(assembly_attrs):
        for i_next, next_joints in enumerate(assembly_attrs):
            scores[i_cur, i_next] = score(cur_joints, next_joints)

    return scores


def event_to_assembly_scores(event_probs, connection_probs, assembly_probs, support_only=True):
    def make_score(p_connection, p_cur_assembly, p_next_assembly, p_action):
        joint_probs = np.stack(
            tuple(
                np.einsum(
                    'aij,ik,jk,ak->k',
                    p_connection, p_cur_assembly, p_next_assembly, p
                )
                for p in p_action
            ), axis=0
        )
        score = scipy.special.logsumexp(np.log(joint_probs).sum(axis=1))
        return score

    num_assemblies = assembly_probs.shape[0]
    num_events = len(event_probs)

    # Log-domain values for zero and one
    zero = -np.inf
    # one = 0

    scores = np.full((num_events, num_assemblies, num_assemblies), zero, dtype=float)
    for i_event in range(num_events):
        for i_cur in range(num_assemblies):
            for i_next in range(num_assemblies):
                score = make_score(
                    connection_probs,
                    assembly_probs[i_cur].T, assembly_probs[i_next].T,
                    event_probs[i_event]
                )
                scores[i_event, i_cur, i_next] = score

    if support_only:
        scores[~np.isinf(scores)] = 0

    return scores


def count_priors(label_seqs, num_classes, stride=None, approx_upto=None, support_only=False):
    dur_counts = {}
    class_counts = {}
    for label_seq in label_seqs:
        for label, dur in zip(*utils.computeSegments(label_seq[::stride])):
            class_counts[label] = class_counts.get(label, 0) + 1
            dur_counts[label, dur] = dur_counts.get((label, dur), 0) + 1

    class_priors = np.zeros((num_classes))
    for label, count in class_counts.items():
        class_priors[label] = count
    class_priors /= class_priors.sum()

    max_dur = max(dur for label, dur in dur_counts.keys())
    dur_priors = np.zeros((num_classes, max_dur))
    for (label, dur), count in dur_counts.items():
        assert dur
        dur_priors[label, dur - 1] = count
    dur_priors /= dur_priors.sum(axis=1, keepdims=True)

    if approx_upto is not None:
        cdf = dur_priors.cumsum(axis=1)
        approx_bounds = (cdf >= approx_upto).argmax(axis=1)
        dur_priors = dur_priors[:, :approx_bounds.max()]

    if support_only:
        dur_priors = (dur_priors > 0).astype(float)

    return class_priors, dur_priors


def viz_priors(fn, class_priors, dur_priors):
    fig, axes = plt.subplots(3)
    axes[0].matshow(dur_priors)
    axes[1].stem(class_priors)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def viz_transition_probs(fig_dir, transitions, vocab):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for i, transition_arr in enumerate(transitions):
        plt.matshow(transition_arr)
        plt.title(vocab[i])
        plt.savefig(os.path.join(fig_dir, f"action={i:02d}_{vocab[i].replace(' ', '-')}"))
        plt.close()


def write_transition_probs(fig_dir, transitions, event_vocab, assembly_vocab):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for i, transition_arr in enumerate(transitions):
        fn = os.path.join(fig_dir, f"action={i:02d}_{event_vocab[i].replace(' ', '-')}.txt")
        rows, cols = transition_arr.nonzero()
        strs = tuple(
            f'{assembly_vocab[r]} -> {assembly_vocab[c]}: {transition_arr[r, c]:.2f}'
            for r, c in zip(rows, cols)
        )
        with open(fn, 'wt') as file_:
            file_.write('\n'.join(strs))


def write_labels(fn, label_seq, vocab):
    seg_label_idxs, seg_durs = utils.computeSegments(label_seq)

    seg_durs = np.array(seg_durs)
    seg_ends = np.cumsum(seg_durs) - 1
    seg_starts = np.array([0] + (seg_ends + 1)[:-1].tolist())
    seg_labels = tuple(vocab[i] for i in seg_label_idxs)
    d = {
        'start': seg_starts,
        'end': seg_ends,
        'label': seg_labels
    }
    pd.DataFrame(d).to_csv(fn, index=False)


def makeTimeString(time_elapsed):
    mins_elapsed = time_elapsed // 60
    secs_elapsed = time_elapsed % 60
    time_str = f'{mins_elapsed:.0f}m {secs_elapsed:.0f}s'
    return time_str


def eval_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
    state_acc = (pred_seq == true_seq).astype(float).mean()

    metric_dict = {
        'State Accuracy' + name_suffix: state_acc,
        'State Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'State Overlap Score' + name_suffix: LCTM.metrics.overlap_score(pred_seq, true_seq) / 100
    }

    append_to.update(metric_dict)
    return append_to


def suppress_nonmax(scores):
    col_idxs = scores.argmax(axis=1)
    new_scores = np.zeros_like(scores)
    row_idxs = np.arange(scores.shape[0])
    new_scores[row_idxs, col_idxs] = scores[row_idxs, col_idxs]
    return new_scores


def main(
        out_dir=None, data_dir=None, scores_dir=None, assembly_data_dir=None,
        event_attr_fn=None, connection_attr_fn=None,
        only_fold=None, plot_io=None, prefix='seq=', stop_after=None,
        background_action='', model_params={}, cv_params={},
        results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
    assembly_data_dir = os.path.expanduser(assembly_data_dir)
    event_attr_fn = os.path.expanduser(event_attr_fn)
    connection_attr_fn = os.path.expanduser(connection_attr_fn)
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

    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix='labels.*',
        to_array=True
    )

    dataset = utils.FeaturelessCvDataset(
        seq_ids, data_dir,
        prefix=prefix,
        label_fn_format='labels'
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {scores_dir}")

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    # Load event, connection attributes
    (*probs, assembly_transition_probs), vocabs = loadPartInfo(
        assembly_data_dir, event_attr_fn, connection_attr_fn,
        background_action=background_action
    )
    event_assembly_scores = event_to_assembly_scores(*probs, vocabs)
    assembly_transition_scores = np.log(assembly_transition_probs)
    viz_transition_probs(
        os.path.join(fig_dir, 'action-transitions'),
        np.exp(event_assembly_scores),
        vocabs['event_vocab']
    )
    write_transition_probs(
        os.path.join(misc_dir, 'action-transitions'),
        np.exp(event_assembly_scores),
        vocabs['event_vocab'], vocabs['assembly_vocab']
    )

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        train_data, val_data, test_data = dataset.getFold(cv_fold)

        cv_str = f'cvfold={cv_index}'

        class_priors, event_dur_probs = count_priors(
            train_data[0], len(dataset.vocab),
            stride=10, approx_upto=0.95, support_only=True
        )
        event_dur_scores = np.log(event_dur_probs)
        event_dur_scores = np.zeros_like(event_dur_scores)
        scores = (event_dur_scores, event_assembly_scores, assembly_transition_scores)

        model = decode.AssemblyActionRecognizer(scores, vocabs, model_params)

        viz_priors(
            os.path.join(fig_dir, f'{cv_str}_priors'),
            class_priors, event_dur_probs
        )
        model.write_fsts(os.path.join(misc_dir, f'{cv_str}_fsts'))
        model.save_vocabs(os.path.join(out_data_dir, f'{cv_str}_model-vocabs'))

        for i, (_, seq_id) in enumerate(zip(*test_data)):
            if stop_after is not None and i >= stop_after:
                break

            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"{prefix}{seq_id}"
            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            true_event_seq = utils.loadVariable(f"{trial_prefix}_true-label-seq", scores_dir)

            # FIXME: the serialized variables are probs, not log-probs
            # event_score_seq = suppress_nonmax(event_score_seq)
            event_score_seq = np.log(event_score_seq)

            decode_score_seq = model.forward(event_score_seq)
            pred_seq = model.predict(decode_score_seq)

            if model_params['return_label'] == 'input':
                metric_dict = eval_metrics(pred_seq, true_event_seq)
                for name, value in metric_dict.items():
                    logger.info(f"    {name}: {value * 100:.2f}%")
                utils.saveVariable(
                    true_event_seq, f'{trial_prefix}_true-label-seq',
                    out_data_dir
                )
                utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            utils.saveVariable(decode_score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)

            if plot_io:
                utils.plot_array(
                    decode_score_seq.T, (pred_seq.T,), ('pred',),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}.png")
                )
                write_labels(
                    os.path.join(misc_dir, f"seq={seq_id:03d}_pred-seq.txt"),
                    pred_seq, model.output_vocab.as_raw()
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
