import os
import logging

import yaml
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import networkx as nx
import torch

from mathtools import utils, torchutils, metrics
from kinemparse import sim2real

from blocks.core import definitions as defn


logger = logging.getLogger(__name__)


def plot_io_DEPRECATED(rgb_seq, seg_seq, pred_seq, true_seq, file_path=None):
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

    def edgeLabelsAsArray(batch):
        num_vertices = len(defn.blocks)
        edges = np.column_stack(np.tril_indices(num_vertices, k=-1))
        arr = np.zeros(
            (batch.shape[0], num_vertices, num_vertices),
            dtype=batch.dtype,
        )
        arr[:, edges[:, 0], edges[:, 1]] = batch
        arr[:, edges[:, 1], edges[:, 0]] = batch
        return arr

    # pred_seq = np.reshape(pred_seq, (pred_seq.shape[0], 8, 8))
    # true_seq = np.reshape(true_seq, (true_seq.shape[0], 8, 8))
    pred_seq = edgeLabelsAsArray(pred_seq)
    true_seq = edgeLabelsAsArray(true_seq)

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


def plot_io(rgb_seq, seg_seq, pred_seq, true_seq, dataset, file_path=None):
    def canonicalPose(num_samples=1):
        angles = torch.zeros(num_samples, dtype=torch.float)
        rotations = Rotation.from_euler('Z', angles)
        R = torch.tensor(rotations.as_matrix()).float().cuda()
        t = torch.stack((torch.zeros_like(angles),) * 3, dim=1).float().cuda()
        return R, t

    def render(assembly):
        R, t = canonicalPose()
        rgb_batch, depth_batch, label_batch = sim2real.renderTemplates(
            dataset.renderer, assembly, t, R
        )
        rgb_crop = sim2real._crop(rgb_batch[0], label_batch[0], dataset.crop_size // 2)
        rgb_crop /= rgb_crop.max()
        return rgb_crop.cpu().numpy()

    num_vertices = len(defn.blocks)
    edges = np.column_stack(np.tril_indices(num_vertices, k=-1))

    any_nonzero_edges = (pred_seq != 0).any(axis=0)
    pred_seq = pred_seq[:, any_nonzero_edges]
    edges = edges[any_nonzero_edges, :]

    num_samples, num_edges = pred_seq.shape

    num_cols = num_samples
    num_rows = num_edges + 2
    row_size = 1.5
    col_size = 1.5
    figsize = (num_cols * col_size, num_rows * row_size)
    f, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    for t in range(num_samples):
        axes[0, t].imshow(rgb_seq[t])
        axes[0, t].axis('off')
        axes[1, t].imshow(seg_seq[t])
        axes[1, t].axis('off')
        for e in range(num_edges):
            edge_label = pred_seq[t, e]
            if edge_label:
                edge_key = frozenset(edges[e, :].tolist())
                assembly = dataset.parts_vocab[edge_key][edge_label - 1]
                image = render(assembly)
            else:
                image = np.zeros((240, 240, 3))
            axes[e + 2, t].imshow(image)
            axes[e + 2, t].axis('off')

    plt.tight_layout()

    if file_path is None:
        plt.show()
    else:
        plt.savefig(file_path)
        plt.close()


def eval_metrics(pred_seq, true_seq, name_suffix=''):
    tp = metrics.truePositives(pred_seq, true_seq)
    tn = metrics.trueNegatives(pred_seq, true_seq)
    fp = metrics.falsePositives(pred_seq, true_seq)
    fn = metrics.falseNegatives(pred_seq, true_seq)

    state_acc = (pred_seq == true_seq).all(axis=1).astype(float).mean()
    edge_acc = (tp + tn) / (tp + tn + fp + fn)
    edge_prc = tp / (tp + fp)
    edge_rec = tp / (tp + fn)
    edge_f1 = 2 * (edge_prc * edge_rec) / (edge_prc + edge_rec)

    metric_dict = {
        'State Accuracy' + name_suffix: state_acc,
        'Edge Accuracy' + name_suffix: edge_acc,
        'Edge Precision' + name_suffix: edge_prc,
        'Edge Recall' + name_suffix: edge_rec,
        'Edge F1' + name_suffix: edge_f1
    }

    return metric_dict


def main(
        out_dir=None, data_dir=None, segs_dir=None, scores_dir=None,
        gpu_dev_id=None, start_from=None, stop_at=None, num_disp_imgs=None,
        results_file=None, sweep_param_name=None, model_params={}):

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

    io_dir_images = os.path.join(fig_dir, 'model-io_images')
    if not os.path.exists(io_dir_images):
        os.makedirs(io_dir_images)

    io_dir_plots = os.path.join(fig_dir, 'model-io_plots')
    if not os.path.exists(io_dir_plots):
        os.makedirs(io_dir_plots)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadAssemblies(seq_id, vocab):
        assembly_seq = joblib.load(os.path.join(data_dir, f"trial={seq_id}_assembly-seq.pkl"))
        labels = np.array([utils.getIndex(assembly, vocab) for assembly in assembly_seq])
        return labels

    def loadVariable(var_name, from_dir=scores_dir):
        var = joblib.load(os.path.join(from_dir, f"{var_name}.pkl"))
        return var

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    seq_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    vocab = loadVariable('vocab')
    parts_vocab = loadVariable('parts-vocab')
    part_labels = loadVariable('part-labels')

    device = torchutils.selectDevice(gpu_dev_id)
    dataset = sim2real.LabeledConnectionDataset(parts_vocab, part_labels, vocab, device=device)

    for seq_id in seq_ids:
        logger.info(f"  Processing sequence {seq_id}...")

        trial_prefix = f"trial={seq_id}"
        rgb_seq = loadVariable(f"{trial_prefix}_rgb-frame-seq", from_dir=data_dir)
        seg_seq = loadVariable(f"{trial_prefix}_seg-labels-seq", from_dir=segs_dir)
        score_seq = loadVariable(f"{trial_prefix}_score-seq", from_dir=scores_dir)
        pred_seq = loadVariable(f"{trial_prefix}_pred-label-seq", from_dir=scores_dir)
        true_seq = loadVariable(f"{trial_prefix}_true-label-seq", from_dir=scores_dir)
        if score_seq.shape[0] != rgb_seq.shape[0]:
            err_str = f"scores shape {score_seq.shape} != data shape {rgb_seq.shape}"
            raise AssertionError(err_str)

        metric_dict = eval_metrics(pred_seq, true_seq)
        metric_dict_no_labels = eval_metrics(
            (pred_seq != 0).astype(int),
            (true_seq != 0).astype(int),
            name_suffix=' (no labels)'
        )
        metric_dict.update(metric_dict_no_labels)

        for name, value in metric_dict.items():
            logger.info(f"    {name}: {value * 100:.2f}%")

        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

        if num_disp_imgs is not None:
            utils.plot_array(
                score_seq.T, (true_seq.T, pred_seq.T), ('true', 'pred'),
                fn=os.path.join(io_dir_plots, f"seq={seq_id:03d}.png")
            )

            if rgb_seq.shape[0] > num_disp_imgs:
                idxs = np.arange(rgb_seq.shape[0])
                np.random.shuffle(idxs)
                idxs = idxs[:num_disp_imgs]
                idxs = np.sort(idxs)
            else:
                idxs = slice(None, None, None)
            plot_io(
                rgb_seq[idxs], seg_seq[idxs], pred_seq[idxs], true_seq[idxs], dataset,
                file_path=os.path.join(io_dir_images, f"seq={seq_id:03d}.png")
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
