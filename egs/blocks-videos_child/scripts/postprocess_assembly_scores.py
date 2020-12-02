import os
import logging
import collections

import yaml
import joblib
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

from mathtools import utils
from blocks.core import definitions as defn


logger = logging.getLogger(__name__)


def plot_io(rgb_seq, seg_seq, pred_seq, true_seq, file_path=None):
    def plot_graph(ax, array):
        g = nx.from_numpy_array(array, create_using=nx.DiGraph)
        pos = nx.circular_layout(g)

        for color in ("red", "blue", "green", "yellow"):
            color_nodes = [i for i, name in enumerate(defn.blocks) if name.startswith(color)]
            nx.draw_networkx_nodes(
                g, pos, ax=ax, nodelist=color_nodes,
                node_shape='s', node_color=color
            )

        labels = {i: n.split()[1][0].upper() for i, n in enumerate(defn.blocks)}
        nx.draw_networkx_labels(g, pos, labels, ax=ax)

        nx.draw_networkx_edges(g, pos, ax=ax)

    def plot_image(axis, array):
        axis.imshow(array)
        axis.axis('off')

    pred_seq = np.reshape(pred_seq, (pred_seq.shape[0], 8, 8))
    true_seq = np.reshape(true_seq, (true_seq.shape[0], 8, 8))

    num_rows = 4
    num_cols = rgb_seq.shape[0]

    if not all(x.shape[0] == num_cols for x in (rgb_seq, seg_seq, pred_seq, true_seq)):
        raise AssertionError()

    row_size = 3
    col_size = 3
    figsize = (num_cols * col_size, num_rows * row_size)
    f, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i_col in range(num_cols):
        plot_image(axes[0, i_col], rgb_seq[i_col])
        plot_image(axes[1, i_col], seg_seq[i_col])
        plot_graph(axes[2, i_col], pred_seq[i_col])
        plot_graph(axes[3, i_col], true_seq[i_col])
    axes[0, 0].set_ylabel('RGB FRAMES')
    axes[1, 0].set_ylabel('FOREGROUND SEGMENTS')
    axes[2, 0].set_ylabel('PREDICTED CONNECTIONS')
    axes[3, 0].set_ylabel('TRUE CONNECTIONS')
    plt.tight_layout()

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        plt.close()


def main(
        out_dir=None, data_dir=None, segs_dir=None, scores_dir=None,
        start_from=None, stop_at=None, num_disp_imgs=None,
        results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    segs_dir = os.path.expanduser(segs_dir)
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

    io_dir = os.path.join(fig_dir, 'model-io')
    if not os.path.exists(io_dir):
        os.makedirs(io_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadVariable(var_name, from_dir=scores_dir):
        var = joblib.load(os.path.join(from_dir, f"{var_name}.pkl"))
        return var

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def makeSeqBatches(unflatten, seq_ids):
        d = collections.defaultdict(list)
        for batch_index, (seq_index, win_index) in enumerate(unflatten):
            seq_id = seq_ids[seq_index]
            d[seq_id].append(batch_index)
        return d

    def loadBatchData(cv_index, batch_index):
        prefix = f"cvfold={cv_index}_batch={batch_index}"
        batch_score = loadVariable(f"{prefix}_score-seq")
        batch_pred = loadVariable(f"{prefix}_pred-label-seq").astype(int)
        batch_true = loadVariable(f"{prefix}_true-label-seq").astype(int)
        return batch_score, batch_pred, batch_true

    cv_fold_indices = utils.getUniqueIds(scores_dir, prefix='cvfold=', to_array=True)
    num_cv_folds = len(cv_fold_indices)

    for cv_index in cv_fold_indices:
        logger.info(f"CV FOLD {cv_index + 1} / {num_cv_folds}")
        seq_ids = loadVariable(f"cvfold={cv_index}_test-ids")
        unflatten = loadVariable(f"cvfold={cv_index}_test-set-unflatten")
        flatten = makeSeqBatches(unflatten, seq_ids)
        for seq_id in seq_ids:
            logger.info(f"  Processing sequence {seq_id}...")
            batch_idxs = flatten[seq_id]
            score_seq, pred_seq, true_seq = map(
                np.vstack,
                zip(*tuple(loadBatchData(cv_index, i) for i in batch_idxs))
            )

            trial_prefix = f"trial={seq_id}"
            rgb_seq = loadVariable(f"{trial_prefix}_rgb-frame-seq", from_dir=data_dir)
            seg_seq = loadVariable(f"{trial_prefix}_seg-labels-seq", from_dir=segs_dir)
            if score_seq.shape[0] != rgb_seq.shape[0]:
                err_str = f"scores shape {score_seq.shape} != data shape {rgb_seq.shape}"
                raise AssertionError(err_str)

            saveVariable(score_seq, f"{trial_prefix}_score-seq")
            saveVariable(pred_seq, f"{trial_prefix}_pred-label-seq")
            saveVariable(true_seq, f"{trial_prefix}_true-label-seq")

            if num_disp_imgs is not None:
                if rgb_seq.shape[0] > num_disp_imgs:
                    idxs = np.arange(rgb_seq.shape[0])
                    np.random.shuffle(idxs)
                    idxs = idxs[:num_disp_imgs]
                    idxs = np.sort(idxs)
                else:
                    idxs = slice(None, None, None)
                plot_io(
                    rgb_seq[idxs], seg_seq[idxs], pred_seq[idxs], true_seq[idxs],
                    file_path=os.path.join(io_dir, f"{trial_prefix}.png")
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
