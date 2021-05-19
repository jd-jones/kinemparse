import os
import logging
import collections

import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import networkx as nx
import torch

import LCTM.metrics

from mathtools import utils, torchutils, metrics
from kinemparse import sim2real
from kinemparse import assembly as lib_assembly
from visiontools import imageprocessing

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


def render(dataset, assembly):
    def canonicalPose(num_samples=1):
        angles = torch.zeros(num_samples, dtype=torch.float)
        rotations = Rotation.from_euler('Z', angles)
        R = torch.tensor(rotations.as_matrix()).float().cuda()
        t = torch.stack((torch.zeros_like(angles),) * 3, dim=1).float().cuda()
        return R, t

    if not assembly.links:
        return np.zeros((dataset.crop_size // 2, dataset.crop_size // 2, 3))

    R, t = canonicalPose()
    rgb_batch, depth_batch, label_batch = lib_assembly.render(dataset.renderer, assembly, t, R)
    rgb_crop = sim2real._crop(rgb_batch[0], label_batch[0], dataset.crop_size // 2)
    rgb_crop /= rgb_crop.max()
    return rgb_crop.cpu().numpy()


def plot_io(rgb_seq, seg_seq, pred_seq, true_seq, dataset, file_path=None):
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
                image = render(dataset, assembly)
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


def eval_edge_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
    tp = metrics.truePositives(pred_seq, true_seq)
    tn = metrics.trueNegatives(pred_seq, true_seq)
    fp = metrics.falsePositives(pred_seq, true_seq)
    fn = metrics.falseNegatives(pred_seq, true_seq)

    edge_acc = utils.safeDivide(tp + tn, tp + tn + fp + fn)
    edge_prc = utils.safeDivide(tp, tp + fp)
    edge_rec = utils.safeDivide(tp, tp + fn)
    edge_f1 = utils.safeDivide(2 * edge_prc * edge_rec, edge_prc + edge_rec)

    metric_dict = {
        'Edge Accuracy' + name_suffix: edge_acc,
        'Edge Precision' + name_suffix: edge_prc,
        'Edge Recall' + name_suffix: edge_rec,
        'Edge F1' + name_suffix: edge_f1
    }

    append_to.update(metric_dict)
    return append_to


def eval_state_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
    state_acc = (pred_seq == true_seq).astype(float).mean()

    metric_dict = {
        'State Accuracy' + name_suffix: state_acc,
        'State Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'State Overlap Score' + name_suffix: LCTM.metrics.overlap_score(pred_seq, true_seq) / 100
    }

    append_to.update(metric_dict)
    return append_to


def oov_rate_state(state_seq, state_vocab):
    state_is_oov = ~np.array([s in state_vocab for s in state_seq], dtype=bool)
    prop_state_oov = state_is_oov.sum() / state_is_oov.size
    return prop_state_oov


def oov_rate_edges(edge_seq, edge_vocab):
    matches_edges = edge_seq[None, :, :] == edge_vocab[:, None, :]
    edge_is_oov = ~(matches_edges.any(axis=0))
    prop_edge_oov = edge_is_oov.sum() / edge_is_oov.size
    return prop_edge_oov


def edge_joint_freqs(edge_seq):
    num_samples, num_edges = edge_seq.shape
    bigram_counts = np.zeros((num_edges, num_edges), dtype=int)
    unigram_counts = np.zeros(num_edges, dtype=int)
    for edges in edge_seq:
        nonzero_edges = np.nonzero(edges)[0]
        for i in range(len(nonzero_edges)):
            e_i = nonzero_edges[i]
            unigram_counts[e_i] += 1
            for j in range(i):
                e_j = nonzero_edges[j]
                bigram_counts[e_i, e_j] += 1
                bigram_counts[e_j, e_i] += 1

    bigram_freqs = bigram_counts / num_samples
    unigram_freqs = unigram_counts / num_samples
    return bigram_freqs, unigram_freqs


def edges_to_assemblies(edge_label_seq, assembly_vocab, edge_vocab, assembly_vocab_edges):
    rows, cols = np.tril_indices(8, k=-1)
    keys = {
        i: frozenset((int(r), int(c)))
        for i, (r, c) in enumerate(zip(rows, cols))
    }

    link_vocab = edge_vocab[keys[0]][0].link_vocab
    joint_vocab = edge_vocab[keys[0]][0].joint_vocab
    joint_type_vocab = edge_vocab[keys[0]][0].joint_type_vocab

    def get_assembly_label(edge_labels):
        all_edges_match = (edge_labels == assembly_vocab_edges).all(axis=1)
        if all_edges_match.any():
            assembly_labels = all_edges_match.nonzero()[0]
            if edge_labels.any() and assembly_labels.size != 1:
                AssertionError(f"{assembly_labels.size} assemblies match these edges!")
            return assembly_labels[0]

        edges = tuple(
            edge_vocab[keys[edge_index]][edge_label - 1]
            for edge_index, edge_label in enumerate(edge_labels)
            if edge_label
        )
        assembly = lib_assembly.union(
            *edges,
            link_vocab=link_vocab,
            joint_vocab=joint_vocab,
            joint_type_vocab=joint_type_vocab
        )
        assembly_label = utils.getIndex(assembly, assembly_vocab)
        return assembly_label

    edge_segs, seg_lens = utils.computeSegments(edge_label_seq)
    assembly_segs = tuple(get_assembly_label(edge_labels) for edge_labels in edge_segs)
    assembly_label_seq = np.array(utils.fromSegments(assembly_segs, seg_lens))

    return assembly_label_seq


def loadAssemblies(seq_id, vocab, data_dir):
    assembly_seq = utils.load(f"trial={seq_id}_assembly-seq", data_dir)
    # assembly_seq = joblib.load(os.path.join(data_dir, f"trial={seq_id}_assembly-seq.pkl"))
    labels = np.array([utils.getIndex(assembly, vocab) for assembly in assembly_seq])
    return labels


def load_vocab(link_vocab, joint_vocab, joint_type_vocab, vocab_dir):
    assembly_vocab = utils.loadVariable('vocab', vocab_dir)
    # FIXME: Convert keys from vertex pairs to edge indices
    edge_vocab = utils.loadVariable('parts-vocab', vocab_dir)
    edge_labels = utils.loadVariable('part-labels', vocab_dir)

    assembly_vocab = tuple(
        lib_assembly.Assembly.from_blockassembly(
            a,
            link_vocab=link_vocab,
            joint_vocab=joint_vocab,
            joint_type_vocab=joint_type_vocab
        )
        for a in assembly_vocab
    )

    edge_vocab = {
        edge_key: tuple(
            lib_assembly.Assembly.from_blockassembly(
                a, link_vocab=link_vocab, joint_vocab=joint_vocab,
                joint_type_vocab=joint_type_vocab
            )
            for a in assemblies
        )
        for edge_key, assemblies in edge_vocab.items()
    }

    return assembly_vocab, edge_vocab, edge_labels


def make_scatterplot(fn, x, y, x_label, y_label, classes=None):
    x = np.array(x)
    y = np.array(y)

    if classes is not None:
        data = tuple((x[classes == i], y[classes == i]) for i in np.unique(classes))
    else:
        data = ((x, y),)

    plt.figure()

    for i, (x, y) in enumerate(data):
        coef = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coef)
        x_range = np.linspace(min(x), max(x))
        y_range = poly1d_fn(x_range)
        plt.scatter(x, y)
        plt.plot(x_range, y_range, label=f"{i}: m={coef[0]:.2f}, b={coef[1]:.2f}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(fn)


def main(
        out_dir=None, data_dir=None, segs_dir=None, scores_dir=None, vocab_dir=None,
        label_type='edges', gpu_dev_id=None, start_from=None, stop_at=None, num_disp_imgs=None,
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    segs_dir = os.path.expanduser(segs_dir)
    scores_dir = os.path.expanduser(scores_dir)
    vocab_dir = os.path.expanduser(vocab_dir)
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
        scores_dir, prefix='trial=', suffix='score-seq.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {scores_dir}")

    link_vocab = {}
    joint_vocab = {}
    joint_type_vocab = {}
    vocab, parts_vocab, part_labels = load_vocab(
        link_vocab, joint_vocab, joint_type_vocab, vocab_dir
    )
    pred_vocab = []  # FIXME

    if label_type == 'assembly':
        logger.info("Converting assemblies -> edges")
        state_pred_seqs = tuple(
            utils.loadVariable(f"trial={seq_id}_pred-label-seq", scores_dir)
            for seq_id in seq_ids
        )
        state_true_seqs = tuple(
            utils.loadVariable(f"trial={seq_id}_true-label-seq", scores_dir)
            for seq_id in seq_ids
        )
        edge_pred_seqs = tuple(part_labels[seq] for seq in state_pred_seqs)
        edge_true_seqs = tuple(part_labels[seq] for seq in state_true_seqs)
    elif label_type == 'edge':
        logger.info("Converting edges -> assemblies (will take a few minutes)")
        edge_pred_seqs = tuple(
            utils.loadVariable(f"trial={seq_id}_pred-label-seq", scores_dir)
            for seq_id in seq_ids
        )
        edge_true_seqs = tuple(
            utils.loadVariable(f"trial={seq_id}_true-label-seq", scores_dir)
            for seq_id in seq_ids
        )
        state_pred_seqs = tuple(
            edges_to_assemblies(seq, pred_vocab, parts_vocab, part_labels)
            for seq in edge_pred_seqs
        )
        state_true_seqs = tuple(
            edges_to_assemblies(seq, vocab, parts_vocab, part_labels)
            for seq in edge_true_seqs
        )

    device = torchutils.selectDevice(gpu_dev_id)
    dataset = sim2real.LabeledConnectionDataset(
        utils.loadVariable('parts-vocab', vocab_dir),
        utils.loadVariable('part-labels', vocab_dir),
        utils.loadVariable('vocab', vocab_dir),
        device=device
    )

    all_metrics = collections.defaultdict(list)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    for cv_index, cv_fold in enumerate(cv_folds):
        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        train_states = np.hstack(tuple(state_true_seqs[i] for i in (train_indices)))
        train_edges = part_labels[train_states]
        # state_train_vocab = np.unique(train_states)
        # edge_train_vocab = part_labels[state_train_vocab]
        train_freq_bigram, train_freq_unigram = edge_joint_freqs(train_edges)
        # state_probs = utils.makeHistogram(len(vocab), train_states, normalize=True)

        test_states = np.hstack(tuple(state_true_seqs[i] for i in (test_indices)))
        test_edges = part_labels[test_states]
        # state_test_vocab = np.unique(test_states)
        # edge_test_vocab = part_labels[state_test_vocab]
        test_freq_bigram, test_freq_unigram = edge_joint_freqs(test_edges)

        f, axes = plt.subplots(1, 2)
        axes[0].matshow(train_freq_bigram)
        axes[0].set_title('Train')
        axes[1].matshow(test_freq_bigram)
        axes[1].set_title('Test')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"edge-freqs-bigram_cvfold={cv_index}.png"))

        f, axis = plt.subplots(1)
        axis.stem(train_freq_unigram, label='Train', linefmt='C0-', markerfmt='C0o')
        axis.stem(test_freq_unigram, label='Test', linefmt='C1--', markerfmt='C1o')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"edge-freqs-unigram_cvfold={cv_index}.png"))

        for i in test_indices:
            seq_id = seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"trial={seq_id}"
            # I include the '.' to differentiate between 'rgb-frame-seq' and
            # 'rgb-frame-seq-before-first-touch'
            # rgb_seq = utils.loadVariable(f"{trial_prefix}_rgb-frame-seq.", data_dir)
            # seg_seq = utils.loadVariable(f"{trial_prefix}_seg-labels-seq", segs_dir)
            score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            # if score_seq.shape[0] != rgb_seq.shape[0]:
            #     err_str = f"scores shape {score_seq.shape} != data shape {rgb_seq.shape}"
            #     raise AssertionError(err_str)

            edge_pred_seq = edge_pred_seqs[i]
            edge_true_seq = edge_true_seqs[i]
            state_pred_seq = state_pred_seqs[i]
            state_true_seq = state_true_seqs[i]

            num_types = np.unique(state_pred_seq).shape[0]
            num_samples = state_pred_seq.shape[0]
            num_total = len(pred_vocab)
            logger.info(
                f"    {num_types} assemblies predicted ({num_total} total); "
                f"{num_samples} samples"
            )

            # edge_freq_bigram, edge_freq_unigram = edge_joint_freqs(edge_true_seq)
            # dist_shift = np.linalg.norm(train_freq_unigram - edge_freq_unigram)
            metric_dict = {
                # 'State OOV rate': oov_rate_state(state_true_seq, state_train_vocab),
                # 'Edge OOV rate': oov_rate_edges(edge_true_seq, edge_train_vocab),
                # 'State avg prob, true': state_probs[state_true_seq].mean(),
                # 'State avg prob, pred': state_probs[state_pred_seq].mean(),
                # 'Edge distribution shift': dist_shift
            }
            metric_dict = eval_edge_metrics(edge_pred_seq, edge_true_seq, append_to=metric_dict)
            metric_dict = eval_state_metrics(state_pred_seq, state_true_seq, append_to=metric_dict)
            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")
                all_metrics[name].append(value)

            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if num_disp_imgs is not None:
                pred_images = tuple(
                    render(dataset, vocab[seg_label])
                    for seg_label in utils.computeSegments(state_pred_seq)[0]
                )
                imageprocessing.displayImages(
                    *pred_images,
                    file_path=os.path.join(io_dir_images, f"seq={seq_id:03d}_pred-assemblies.png"),
                    num_rows=None, num_cols=5
                )
                true_images = tuple(
                    render(dataset, vocab[seg_label])
                    for seg_label in utils.computeSegments(state_true_seq)[0]
                )
                imageprocessing.displayImages(
                    *true_images,
                    file_path=os.path.join(io_dir_images, f"seq={seq_id:03d}_true-assemblies.png"),
                    num_rows=None, num_cols=5
                )

                utils.plot_array(
                    score_seq.T, (edge_true_seq.T, edge_pred_seq.T), ('true', 'pred'),
                    fn=os.path.join(io_dir_plots, f"seq={seq_id:03d}.png")
                )

                # if False:  # FIXME
                #     if rgb_seq.shape[0] > num_disp_imgs:
                #         idxs = np.arange(rgb_seq.shape[0])
                #         np.random.shuffle(idxs)
                #         idxs = idxs[:num_disp_imgs]
                #         idxs = np.sort(idxs)
                #     else:
                #         idxs = slice(None, None, None)
                #     plot_io(
                #         rgb_seq[idxs], seg_seq[idxs], edge_pred_seq[idxs], edge_true_seq[idxs],
                #         dataset,
                #         file_path=os.path.join(io_dir_images, f"seq={seq_id:03d}.png")
                #     )

    # make_scatterplot(
    #     os.path.join(fig_dir, "state-oov_vs_state-accuracy.png"),
    #     all_metrics['State OOV rate'],
    #     all_metrics['State Accuracy'],
    #     'State OOV rate', 'State Accuracy',
    #     classes=(np.array(all_metrics['Edge OOV rate']) < 0.05).astype(int)
    # )
    # make_scatterplot(
    #     os.path.join(fig_dir, "state-prob_vs_state-accuracy.png"),
    #     all_metrics['State avg prob, true'],
    #     all_metrics['State Accuracy'],
    #     'State avg prob, true', 'State Accuracy',
    # )
    # make_scatterplot(
    #     os.path.join(fig_dir, "edge-oov_vs_state-accuracy.png"),
    #     all_metrics['Edge OOV rate'],
    #     all_metrics['State Accuracy'],
    #     'Edge OOV rate', 'State Accuracy'
    # )
    # make_scatterplot(
    #     os.path.join(fig_dir, "edge-oov_vs_edge-F1.png"),
    #     all_metrics['Edge OOV rate'],
    #     all_metrics['Edge F1'],
    #     'Edge OOV rate', 'Edge F1'
    # )
    # make_scatterplot(
    #     os.path.join(fig_dir, "edge-oov_vs_state-oov.png"),
    #     all_metrics['Edge OOV rate'],
    #     all_metrics['State OOV rate'],
    #     'Edge OOV rate', 'State OOV rate'
    # )
    # make_scatterplot(
    #     os.path.join(fig_dir, "edge-dist-shift_vs_state-accuracy.png"),
    #     all_metrics['Edge distribution shift'],
    #     all_metrics['State Accuracy'],
    #     'Edge distribution shift', 'State Accuracy'
    # )
    # make_scatterplot(
    #     os.path.join(fig_dir, "edge-dist-shift_vs_edge-F1.png"),
    #     all_metrics['Edge distribution shift'],
    #     all_metrics['Edge F1'],
    #     'Edge distribution shift', 'Edge F1'
    # )

    # for i, a in enumerate(pred_vocab):
    #     pred_image = render(dataset, a)
    #     imageprocessing.displayImage(
    #         pred_image,
    #         file_path=os.path.join(io_dir_images, f"pred_class={i:04d}.png")
    #     )


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
