import os
import logging
import collections

import yaml
import numpy as np
# from matplotlib import pyplot as plt

import LCTM.metrics

from mathtools import utils, metrics


logger = logging.getLogger(__name__)


def retrievalMetrics(pred_seq, true_seq, background_index=0):
    tp = metrics.truePositives(pred_seq, true_seq, background_index=background_index)
    # tn = metrics.trueNegatives(pred_seq, true_seq, background_index=background_index)
    fp = metrics.falsePositives(pred_seq, true_seq, background_index=background_index)
    fn = metrics.falseNegatives(pred_seq, true_seq, background_index=background_index)

    prc = utils.safeDivide(tp, tp + fp)
    rec = utils.safeDivide(tp, tp + fn)
    f1 = utils.safeDivide(2 * prc * rec, prc + rec)
    acc = (pred_seq == true_seq).astype(float).mean()

    return acc, prc, rec, f1


def eval_metrics_part(pred_seq, true_seq, name_suffix='', append_to={}):
    tp = metrics.truePositives(pred_seq, true_seq)
    # tn = metrics.trueNegatives(pred_seq, true_seq)
    fp = metrics.falsePositives(pred_seq, true_seq)
    fn = metrics.falseNegatives(pred_seq, true_seq)

    prc = utils.safeDivide(tp, tp + fp)
    rec = utils.safeDivide(tp, tp + fn)
    f1 = utils.safeDivide(2 * prc * rec, prc + rec)
    acc = (pred_seq == true_seq).astype(float).mean()

    metric_dict = {
        'Accuracy' + name_suffix: acc,
        'Precision' + name_suffix: prc,
        'Recall' + name_suffix: rec,
        'F1' + name_suffix: f1,
    }

    append_to.update(metric_dict)
    return append_to


def eval_metrics(pred_seq, true_seq, name_suffix='', append_to={}, background_index=0):
    # state_acc = (pred_seq == true_seq).astype(float).mean()
    acc, prc, rec, f1 = retrievalMetrics(pred_seq, true_seq, background_index=0)

    metric_dict = {
        'Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'Accuracy' + name_suffix: acc,
        'Precision' + name_suffix: prc,
        'Recall' + name_suffix: rec,
        'F1' + name_suffix: f1,
    }

    append_to.update(metric_dict)
    return append_to


def make_attrs(s, edge_vocab):
    attrs = np.zeros(len(edge_vocab), dtype=int)
    for e in s:
        i = edge_vocab.index(e)
        attrs[i] = 1
    return attrs


def main(
        out_dir=None, data_dir=None, scores_dir=None, frames_dir=None,
        vocab_from_scores_dir=None, only_fold=None, plot_io=None, prefix='seq=',
        no_cv=False, background_class='NA',
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
        scores_dir, prefix=prefix, suffix='true-label-seq.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {scores_dir}")

    vocab = utils.loadVariable('vocab', data_dir)
    background_index = vocab.index(background_class)

    if False:
        vocab = tuple(
            tuple(sorted(tuple(sorted(joint)) for joint in a))
            for a in utils.loadVariable('vocab', data_dir)
        )
        edge_vocab = tuple(set(e for s in vocab for e in s))
        attr_vocab = np.row_stack(tuple(make_attrs(s, edge_vocab) for s in vocab))

    all_metrics = collections.defaultdict(list)

    # Define cross-validation folds
    if no_cv:
        cv_folds = ((tuple(), tuple(), tuple(range(len(seq_ids)))),)
    else:
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

            metric_dict = eval_metrics(pred_seq, true_seq, background_index=background_index)

            if False:
                pred_edges = attr_vocab[pred_seq]
                true_edges = attr_vocab[true_seq]
                part_metric_dict = eval_metrics_part(pred_edges, true_edges)
                for key, value in part_metric_dict.items():
                    metric_dict[f'Part {key}'] = value

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

    if False:
        confusions = metrics.confusionMatrix(all_pred_seqs, all_true_seqs, len(vocab))
        utils.saveVariable(confusions, "confusions", out_data_dir)

        per_class_acc, class_counts = metrics.perClassAcc(confusions, return_counts=True)
        logger.info(f"MACRO ACC: {per_class_acc.mean() * 100:.2f}%")

        metrics.plotConfusions(os.path.join(fig_dir, 'confusions.png'), confusions, vocab)
        metrics.plotPerClassAcc(
            os.path.join(fig_dir, 'per-class-results.png'),
            vocab, per_class_acc, class_counts
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
