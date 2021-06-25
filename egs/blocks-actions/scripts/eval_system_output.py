import os
import logging
import collections

import yaml
import numpy as np
# from matplotlib import pyplot as plt
import graphviz as gv

import LCTM.metrics

from mathtools import utils, metrics
from blocks.core import labels as labels_lib


logger = logging.getLogger(__name__)


def eval_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
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
        'Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'Overlap Score' + name_suffix: LCTM.metrics.overlap_score(pred_seq, true_seq) / 100
    }

    append_to.update(metric_dict)
    return append_to


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


def drawLabels(paths, fig_fn, base_path, state_img_dir, path_labels=None, img_ext='png'):
    """ Draw a sequence of `BlockAssembly` states using graphviz.

    Parameters
    ----------
    path : iterable( int )
        A path is a list of state indices.
    fig_fn : str
        Filename of the figure
    base_path : str
        Path to the directory where figure will be saved
    state_img_dir : str
        Path to the directory containing source images of the states that make
        up this path. State filenames are assumed to have the format
        `state<state_index>.<img_ext>`
    img_ext : str, optional
        Extension specifying the image file type. Can be 'svg', 'png', etc.
    """

    path_graph = gv.Digraph(
        name=fig_fn, format=img_ext, directory=base_path,
        graph_attr={'rankdir': 'LR'},
        node_attr={'shape': 'plaintext'}
    )

    for j, path in enumerate(paths):
        for i, state_index in enumerate(path):
            image_fn = 'state{}.{}'.format(state_index, img_ext)
            image_path = os.path.join(state_img_dir, image_fn)

            if path_labels is not None:
                label = f"{path_labels[j, i]}"
            else:
                label = ''

            path_graph.node(
                f"{j}, {i}", image=image_path,
                fixedsize='true', width='1', height='0.5', imagescale='true',
                pad='1', fontsize='12', label=label
            )
            if i > 0:
                path_graph.edge(f"{j}, {i - 1}", f"{j}, {i}", fontsize='12')

    path_graph.render(filename=fig_fn, directory=base_path, cleanup=True)


def makeEdges(vocab):
    parts_vocab, edge_diffs = labels_lib.make_parts_vocab(
        vocab, lower_tri_only=True, append_to_vocab=False
    )
    signs = np.array([a.sign for a in vocab], dtype=int)
    signs[signs == -1] = 2
    edge_diffs = np.concatenate((edge_diffs, signs[:, None]), axis=1)

    return edge_diffs


def main(
        out_dir=None, data_dir=None, scores_dir=None,
        vocab_from_scores_dir=None, only_fold=None,
        plot_io=None, draw_labels=None, vocab_fig_dir=None,
        prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
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
        scores_dir, prefix=prefix, suffix='pred-label-seq.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {scores_dir}")

    if vocab_from_scores_dir:
        vocab = utils.loadVariable('vocab', scores_dir)
    else:
        vocab = utils.loadVariable('vocab', data_dir)

    for i in range(len(vocab)):
        sign = vocab[i].sign
        if isinstance(sign, np.ndarray):
            vocab[i].sign = np.sign(sign.sum())

    edge_attrs = makeEdges(vocab)

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

            pred_parts_seq = edge_attrs[pred_seq]
            true_parts_seq = edge_attrs[true_seq]
            # pred_parts_seq = pred_seq
            # true_parts_seq = true_seq

            # metric_dict = {}
            metric_dict = eval_metrics(pred_seq, true_seq)
            part_metric_dict = eval_metrics_part(pred_parts_seq, true_parts_seq)
            for key, value in part_metric_dict.items():
                metric_dict[f'Part {key}'] = value

            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")
                all_metrics[name].append(value)

            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            all_pred_seqs.append(pred_seq)
            all_true_seqs.append(true_seq)

            if plot_io:
                perf_str = '  '.join(
                    f'{name}: {metric_dict[name] * 100:.2f}%'
                    for name in ('Accuracy', 'F1', 'Edit Score')
                )
                title = f'{trial_prefix}  {perf_str}'
                utils.plot_array(
                    score_seq.T, (true_seq.T, pred_seq.T), ('true', 'pred'),
                    fn=os.path.join(io_dir_plots, f"seq={seq_id:03d}.png"),
                    title=title
                )
                utils.plot_array(
                    score_seq.T, (true_parts_seq.T, pred_parts_seq.T), ('true', 'pred'),
                    fn=os.path.join(io_dir_plots, f"seq={seq_id:03d}_parts.png"),
                    title=title
                )

            if draw_labels:
                drawLabels(
                    [
                        utils.computeSegments(pred_seq)[0],
                        utils.computeSegments(true_seq)[0]
                    ],
                    f"seq={seq_id:03d}", io_dir_images,
                    vocab_fig_dir
                )

    if False:
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
