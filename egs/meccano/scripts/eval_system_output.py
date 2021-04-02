import os
import logging
import collections

import yaml
import numpy as np
# from matplotlib import pyplot as plt

import LCTM.metrics

from mathtools import utils, metrics


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


def main(
        out_dir=None, data_dir=None, scores_dir=None, frames_dir=None,
        vocab_from_scores_dir=None, only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
    frames_dir = os.path.expanduser(frames_dir)
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

    io_dir_images = os.path.join(fig_dir, 'model-io_images')
    if not os.path.exists(io_dir_images):
        os.makedirs(io_dir_images)

    io_dir_plots = os.path.join(fig_dir, 'model-io_plots')
    if not os.path.exists(io_dir_plots):
        os.makedirs(io_dir_plots)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix='labels.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {scores_dir}")

    if vocab_from_scores_dir:
        vocab = utils.loadVariable('vocab', scores_dir)
    else:
        vocab = utils.loadVariable('vocab', data_dir)

    all_metrics = collections.defaultdict(list)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    all_pred_seqs = []
    all_true_seqs = []

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        for i in test_indices:
            seq_id = seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"{prefix}{seq_id}"
            score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            pred_seq = utils.loadVariable(f"{trial_prefix}_pred-label-seq", scores_dir)
            true_seq = utils.loadVariable(f"{trial_prefix}_true-label-seq", scores_dir)

            metric_dict = eval_metrics(pred_seq, true_seq)
            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")
                all_metrics[name].append(value)

            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            all_pred_seqs.append(pred_seq)
            all_true_seqs.append(true_seq)

            if plot_io:
                utils.plot_array(
                    score_seq.T, (true_seq, pred_seq), ('true', 'pred'),
                    fn=os.path.join(io_dir_plots, f"seq={seq_id:03d}.png")
                )

    confusions = metrics.confusionMatrix(all_pred_seqs, all_true_seqs, len(vocab))
    utils.saveVariable(confusions, "confusions", out_data_dir)

    per_class_acc, class_counts = metrics.perClassAcc(confusions, return_counts=True)
    class_preds = confusions.sum(axis=1)
    logger.info(f"MACRO ACC: {np.nanmean(per_class_acc) * 100:.2f}%")

    metrics.plotConfusions(os.path.join(fig_dir, 'confusions.png'), confusions, vocab)
    metrics.plotPerClassAcc(
        os.path.join(fig_dir, 'per-class-results.png'),
        vocab, per_class_acc, class_preds, class_counts
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
