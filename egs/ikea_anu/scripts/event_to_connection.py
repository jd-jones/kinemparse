import os
import logging
import json
import itertools

import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import LCTM.metrics

from kinemparse import decode
from mathtools import utils  # , metrics


logger = logging.getLogger(__name__)


def loadPartInfo(fn):
    with open(fn, 'rt') as file_:
        data = json.load(file_)

    part_vocab = tuple(data['part_vocab'])
    part_categories = data['part_categories']
    part_connections = data['part_connections']

    joint_vocab = tuple(tuple(sorted(joint)) for joint in data['joint_vocab'])
    assembly_vocab = tuple(
        tuple(tuple(sorted(joint)) for joint in joints)
        for joints in data['assembly_vocab']
    )

    return part_vocab, part_categories, part_connections, joint_vocab, assembly_vocab


def event_attrs_to_scores(
        event_attrs, part_categories, event_vocab, action_vocab, joint_vocab,
        background_action=''):
    """

    Parameters
    ----------
    event_attrs : pd.Dataframe
        Columns: (event [str], action [str], <part> active [bool])

    Returns
    -------
    scores : np.array of float, shape (num_events, num_actions, num_joints)
        score(event, action, joint) represented in the log domain
    """

    def makeActiveParts(active_part_categories):
        parts = tuple(
            part_categories.get(part_category, [part_category])
            for part_category in active_part_categories
        )
        parts_set = frozenset([frozenset(prod) for prod in itertools.product(*parts)])
        parts_tup = tuple(tuple(sorted(x)) for x in parts_set)
        return parts_tup

    # Log-domain values for zero and one
    zero = -np.inf
    one = 0

    num_events = len(event_vocab.as_raw())
    num_actions = len(action_vocab.as_raw())
    num_joints = len(joint_vocab.as_raw())

    # event index --> all data
    event_attrs = event_attrs.set_index('event')
    part_suffix = '_active'
    part_cols = [name for name in event_attrs.columns if name.endswith(part_suffix)]

    scores = np.full((num_events, num_actions, num_joints), zero, dtype=float)
    for i_event, event_name in enumerate(event_vocab):
        row = event_attrs.loc[event_name]
        action_name = row['action']
        active_part_categories = tuple(
            name.split(part_suffix)[0]
            for name in part_cols
            if row[name]
        )
        all_active_parts = makeActiveParts(active_part_categories)
        for i_joint, _ in enumerate(joint_vocab):
            for active_parts in all_active_parts:
                active_joint_index = joint_vocab.integerizer.get(active_parts, None)
                if active_joint_index == i_joint:
                    i_action = action_vocab.integerizer[action_name]
                    break
            else:
                i_action = action_vocab.integerizer[background_action]
            scores[i_event, i_action, i_joint] = one

    return scores


def action_attrs_to_scores(action_attrs):
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
    zero = -np.inf
    one = 0

    action_attrs = action_attrs.set_index('action')

    tx_sep = '->'
    tx_cols = [name for name in action_attrs.columns if name.contains(tx_sep)]
    tx_vocab = tuple(
        tuple(int(x) for x in col_name.split(tx_sep))
        for col_name in tx_cols
    )
    connection_vocab = tuple(
        sorted(frozenset().union(*[frozenset(t) for t in tx_vocab]))
    )
    action_vocab = tuple(action_attrs['action'].to_list())

    num_actions = len(action_vocab)
    num_connections = len(action_vocab)
    scores = np.full((num_actions, num_connections, num_connections), zero, dtype=float)
    for i_action, action_name in enumerate(action_vocab):
        row = action_attrs.loc[action_name]
        action_name = row['action']
        for i_edge, tx_weight in enumerate([row[c] for c in tx_cols]):
            i_conn_cur, i_conn_next = tx_vocab[i_edge]
            scores[i_action, i_conn_cur, i_conn_next] = one

    return scores, action_vocab, connection_vocab


def assembly_attrs_to_scores(assembly_attrs, joint_vocab, connection_vocab):
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
    zero = -np.inf
    one = 0

    assembly_vocab = tuple(i for i, _ in enumerate(assembly_attrs))

    num_assemblies = len(assembly_vocab)
    num_joints = len(joint_vocab.as_raw())
    num_connections = len(connection_vocab.as_raw())

    scores = np.full((num_assemblies, num_joints, num_connections), zero, dtype=float)
    for i_assembly, _ in enumerate(assembly_vocab):
        joints = assembly_attrs[i_assembly]
        for joint in joints:
            i_joint = joint_vocab.integerizer[joint]
            i_connection = connection_vocab.integerizer[1]
            scores[i_assembly, i_joint, i_connection] = one

    return scores, assembly_vocab


def count_priors(label_seqs, num_classes, stride=None, approx_upto=None):
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

    return class_priors, dur_priors


def viz_priors(fn, class_priors, dur_priors):
    fig, axes = plt.subplots(2)
    axes[0].matshow(dur_priors)
    axes[1].stem(class_priors)
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


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


def main(
        out_dir=None, data_dir=None, scores_dir=None,
        event_attr_fn=None, connection_attr_fn=None, part_info_fn=None,
        only_fold=None, plot_io=None, prefix='seq=', stop_after=None,
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
    event_attr_fn = os.path.expanduser(event_attr_fn)
    connection_attr_fn = os.path.expanduser(connection_attr_fn)
    part_info_fn = os.path.expanduser(part_info_fn)
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
    event_attrs = pd.read_csv(event_attr_fn, index_col=False, keep_default_na=False)
    connection_attrs = pd.read_csv(connection_attr_fn, index_col=False, keep_default_na=False)
    part_vocab, part_categories, part_connections, joint_vocab, assembly_attrs = loadPartInfo(
        part_info_fn
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
        class_priors, dur_priors = count_priors(
            train_data[0], len(dataset.vocab),
            stride=10, approx_upto=0.95
        )
        model = decode.AttributeClassifier(
            event_attrs, connection_attrs, assembly_attrs,
            dataset.vocab, joint_vocab,
            part_vocab, part_categories, part_connections,
            event_duration_scores=np.log(dur_priors),
            **model_params
        )

        viz_priors(os.path.join(fig_dir, f'{cv_str}_priors'), class_priors, dur_priors)
        model.write_fsts(os.path.join(misc_dir, f'{cv_str}_fsts'))
        model.save_vocabs(os.path.join(out_data_dir, f'{cv_str}_model-vocabs'))

        for i, (_, seq_id) in enumerate(zip(*test_data)):
            if stop_after is not None and i >= stop_after:
                break

            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"{prefix}{seq_id}"
            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            # true_event_seq = utils.loadVariable(f"{trial_prefix}_true-label-seq", scores_dir)

            # FIXME: the serialized variables are probs, not log-probs
            event_score_seq = np.log(event_score_seq)

            decode_score_seq = model.forward(event_score_seq)
            pred_seq = model.predict(decode_score_seq)

            # metric_dict = eval_metrics(pred_event_seq, true_event_seq)
            # for name, value in metric_dict.items():
            #     logger.info(f"    {name}: {value * 100:.2f}%")

            utils.saveVariable(decode_score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            # utils.saveVariable(true_event_seq, f'{seq_id_str}_true-label-seq', out_data_dir)
            # utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if plot_io:
                utils.plot_array(
                    event_score_seq.T, (pred_seq.T,), ('pred',),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}.png")
                )

                joint_fig_dir = os.path.join(fig_dir, f"seq={seq_id:03d}_joint-scores")
                if not os.path.exists(joint_fig_dir):
                    os.makedirs(joint_fig_dir)
                joint_misc_dir = os.path.join(misc_dir, f"seq={seq_id:03d}_joint-preds")
                if not os.path.exists(joint_misc_dir):
                    os.makedirs(joint_misc_dir)
                for i in range(decode_score_seq.shape[-1]):
                    scores = decode_score_seq[..., i]
                    preds = pred_seq[..., i]
                    joint = tuple(model.joint_vocab.as_raw()[i])
                    utils.plot_array(
                        scores.T, (preds,), ('pred',),
                        fn=os.path.join(joint_fig_dir, f"joint={i:02d}_{joint}.png"),
                        title=f"joint {i}: {joint}"
                    )
                    write_labels(
                        os.path.join(joint_misc_dir, f"joint={i:02d}_{joint}.txt"),
                        preds, model.output_vocab.as_raw()
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
