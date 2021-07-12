import os
import logging

import yaml
import numpy as np
from matplotlib import pyplot as plt
# import pandas as pd
# import scipy

import LCTM.metrics

from kinemparse import decode
from mathtools import utils  # , metrics

# from blocks.core import blockassembly


logger = logging.getLogger(__name__)


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


def make_assembly_transition_priors(assembly_vocab):
    def isValid(diff):
        for i in range(diff.connections.shape[0]):
            c = diff.connections.copy()
            c[i, :] = 0
            c[:, i] = 0
            if not c.any():
                return True
        return False

    num_assemblies = len(assembly_vocab)
    priors = np.zeros((num_assemblies, num_assemblies), dtype=bool)

    for j, cur_assembly in enumerate(assembly_vocab):
        for k, next_assembly in enumerate(assembly_vocab):
            if cur_assembly == next_assembly:
                continue

            try:
                diff = next_assembly - cur_assembly
            except ValueError:
                continue

            priors[j, k] = isValid(diff)

    return priors


def count_transitions(label_seqs, num_classes, support_only=False):
    start_counts = np.zeros(num_classes, dtype=float)
    end_counts = np.zeros(num_classes, dtype=float)
    for label_seq in label_seqs:
        start_counts[label_seq[0]] += 1
        end_counts[label_seq[-1]] += 1

    start_probs = start_counts / start_counts.sum()
    end_probs = end_counts / end_counts.sum()

    if support_only:
        start_probs = (start_probs > 0).astype(float)
        end_probs = (end_probs > 0).astype(float)

    return start_probs, end_probs


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


def viz_transition_probs(fig_dir, transitions):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for i, transition_arr in enumerate(transitions):
        plt.matshow(transition_arr)
        plt.savefig(os.path.join(fig_dir, f"action={i:03d}"))
        plt.close()


def pack_scores(transitions, start, end):
    num_assemblies = transitions.shape[0]
    packed = np.zeros((num_assemblies + 1, num_assemblies + 1), dtype=float)
    packed[0, :-1] = start
    packed[1:, -1] = end
    packed[1:, :-1] = transitions
    return packed


def computeMoments(feature_seqs):
    features = np.concatenate(feature_seqs, axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return mean, std


def main(
        out_dir=None, assembly_scores_dir=None, event_scores_dir=None,
        labels_from='assemblies',
        feature_fn_format='score-seq', label_fn_format='true-label-seq',
        only_fold=None, plot_io=None, prefix='seq=', stop_after=None,
        background_action='', stride=None, standardize_inputs=False,
        model_params={}, cv_params={},
        results_file=None, sweep_param_name=None):

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

    scores_dirs = {
        'events': event_scores_dir,
        'assemblies': assembly_scores_dir
    }
    data_dir = scores_dirs[labels_from]
    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix=f'{label_fn_format}.*',
        to_array=True
    )

    event_dataset = utils.FeaturelessCvDataset(
        seq_ids, event_scores_dir,
        prefix=prefix,
        label_fn_format=label_fn_format
    )

    assembly_dataset = utils.FeaturelessCvDataset(
        seq_ids, assembly_scores_dir,
        prefix=prefix,
        label_fn_format=label_fn_format
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {data_dir}")

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    # Load vocabs; create priors
    event_vocab = utils.loadVariable('vocab', event_scores_dir)
    assembly_vocab = utils.loadVariable('vocab', assembly_scores_dir)

    vocabs = {
        'event_vocab': tuple(range(len(event_vocab))),
        'assembly_vocab': tuple(range(len(assembly_vocab)))
    }

    try:
        event_priors = utils.loadVariable('event-priors', out_data_dir)
    except AssertionError:
        event_priors = make_event_assembly_transition_priors(event_vocab, assembly_vocab)
        utils.saveVariable(event_priors, 'event-priors', out_data_dir)
        viz_transition_probs(os.path.join(fig_dir, 'event-priors'), event_priors)
        np.savetxt(
            os.path.join(misc_dir, "event-transitions.csv"),
            np.column_stack(event_priors.nonzero()),
            delimiter=",", fmt='%d'
        )

    try:
        assembly_priors = utils.loadVariable('assembly-priors', out_data_dir)
    except AssertionError:
        assembly_priors = make_assembly_transition_priors(assembly_vocab)
        utils.saveVariable(assembly_priors, 'assembly-priors', out_data_dir)
        viz_transition_probs(os.path.join(fig_dir, 'assembly-priors'), assembly_priors[None, ...])
        np.savetxt(
            os.path.join(misc_dir, "assembly-transitions.csv"),
            np.column_stack(assembly_priors.nonzero()),
            delimiter=",", fmt='%d'
        )

    event_assembly_scores = np.log(event_priors)
    assembly_scores = np.log(assembly_priors)
    assembly_scores = np.zeros_like(assembly_scores)

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        cv_str = f'cvfold={cv_index}'

        (train_event_labels, _), _, (_, test_seq_ids) = event_dataset.getFold(cv_fold)
        (train_assembly_labels, _), _, _ = assembly_dataset.getFold(cv_fold)

        assembly_start_probs, assembly_end_probs = count_transitions(
            train_assembly_labels, len(assembly_vocab),
            support_only=True
        )
        assembly_start_scores = np.log(assembly_start_probs)
        assembly_end_scores = np.log(assembly_end_probs)
        assembly_transition_scores = pack_scores(
            assembly_scores, assembly_start_scores, assembly_end_scores
        )

        class_priors, event_dur_probs = count_priors(
            train_event_labels, len(event_vocab),
            approx_upto=0.95, support_only=True
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

        for i, seq_id in enumerate(test_seq_ids):
            if stop_after is not None and i >= stop_after:
                break

            trial_prefix = f"{prefix}{seq_id}"
            logger.info(f"  Processing sequence {seq_id}...")

            true_label_seq = utils.loadVariable(
                f"{trial_prefix}_true-label-seq",
                data_dir
            )

            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", event_scores_dir)

            score_seq = model.forward(event_score_seq)
            pred_label_seq = model.predict(score_seq)

            metric_dict = eval_metrics(pred_label_seq, true_label_seq)
            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")
            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            utils.saveVariable(score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_label_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            utils.saveVariable(true_label_seq, f'{trial_prefix}_true-label-seq', out_data_dir)

            if plot_io:
                utils.plot_array(
                    event_score_seq.T, (pred_label_seq.T, true_label_seq.T), ('pred', 'true'),
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
