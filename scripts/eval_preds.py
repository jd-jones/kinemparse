import argparse
import os
import inspect
import functools

import yaml
import numpy as np
import joblib
from matplotlib import pyplot as plt
import graphviz as gv
import pandas as pd

from mathtools import utils, metrics
from blocks.core import labels


def drawPaths(paths, fig_fn, base_path, state_img_dir, path_labels=None, img_ext='png'):
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

    path_graph = gv.Digraph(name=fig_fn, format=img_ext, directory=base_path)

    for j, path in enumerate(paths):
        for i, state_index in enumerate(path):
            image_fn = 'state{}.{}'.format(state_index, img_ext)
            image_path = os.path.join(state_img_dir, image_fn)

            if path_labels is not None:
                label = f"{path_labels[j, i]}"
            else:
                label = None

            path_graph.node(
                f"{j}, {i}", image=image_path,
                fixedsize='true', width='1', height='0.5', imagescale='true',
                pad='1', fontsize='12', label=label
            )
            if i > 0:
                path_graph.edge(f"{j}, {i - 1}", f"{j}, {i}", fontsize='12')

    path_graph.render(filename=fig_fn, directory=base_path, cleanup=True)


def plot_scores(score_seq, k=None, fn=None):
    subplot_width = 12
    subplot_height = 3
    num_axes = score_seq.shape[0] + 1

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize)
    if num_axes == 1:
        axes = [axes]

    score_seq = score_seq.copy()

    for i, scores in enumerate(score_seq):
        if k is not None:
            bottom_k = (-scores).argsort(axis=0)[k:, :]
            for j in range(scores.shape[1]):
                scores[bottom_k[:, j], j] = -np.inf
        axes[i].imshow(scores, interpolation='none', aspect='auto')
    axes[-1].imshow(score_seq.sum(axis=0), interpolation='none', aspect='auto')

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def plot_hists(scores, axis_labels=None, fn=None):
    subplot_width = 12
    subplot_height = 3
    num_axes = scores.shape[0]

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize)
    if num_axes == 1:
        axes = [axes]

    for i, s in enumerate(scores):
        s[np.isinf(s)] = s[~np.isinf(s)].min() - 1
        axes[i].hist(s, bins=50, density=True)
        if axis_labels is not None:
            axes[i].set_ylabel(axis_labels[i])

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def is_goal_error(goal, pred, true):
    return (pred <= goal) == (true <= goal)


def is_goal_layer(goal, assembly):
    def makeLayers(assembly):
        if not assembly.blocks:
            return {}

        def makeLayer(z, assembly):
            layer = assembly.copy()
            for b_id in tuple(layer.blocks.keys()):
                if b_id not in layer.blocks:
                    continue

                block = layer.getBlock(b_id)
                block_height = block.metric_vertices[:, -1].max()
                if block_height > z:
                    layer.connections[b_id, :] = False
                    layer.connections[:, b_id] = False
                    component = layer.connected_components[block.component_index]
                    component.remove(b_id)
                    if not component:
                        del layer.connected_componets[block.component_index]
                    del layer.blocks[b_id]
            return layer

        def max_z(block_vertices):
            return block_vertices[:, -1].max()
        block_heights = np.array(list(map(max_z, assembly.vertices)))
        layer_heights = np.unique(block_heights)
        layers = {z: makeLayer(z, assembly) for z in layer_heights}

        return layers

    for comp_idx in assembly.connected_components.keys():
        assembly.centerComponent(comp_idx, zero_at='smallest_z')

    for comp_idx in goal.connected_components.keys():
        goal.centerComponent(comp_idx, zero_at='smallest_z')

    assembly_layers = makeLayers(assembly)
    goal_layers = makeLayers(goal)

    prev_layer_heights = sorted(assembly_layers.keys())[:-1]
    prev_layers_complete = tuple(
        assembly_layers[z] == goal_layers[z]
        for z in prev_layer_heights
    )

    is_goal_layer = all(prev_layers_complete)

    if len(goal.blocks) > 4:
        # import pdb; pdb.set_trace()
        pass

    # logger.info(f"      z_max prev: {zmax_prev:.1f}")
    # logger.info(f"       z_max cur: {zmax_cur:.1f}")
    # logger.info(f"cur is new layer: {cur_is_new_layer}")
    # logger.info(f" prev is correct: {prev_is_correct}")
    # logger.info(f"   is goal layer: {is_goal_layer}")

    return is_goal_layer


def main(
        out_dir=None, data_dir=None, metadata_file=None,
        plot_predictions=None, results_file=None, sweep_param_name=None):

    logger.info(f"Reading from: {data_dir}")
    logger.info(f"Writing to: {out_dir}")

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    if metadata_file is not None:
        metadata_file = os.path.expanduser(metadata_file)
        metadata = pd.read_csv(metadata_file, index_col=0)

    if results_file is None:
        results_file = os.path.join(out_dir, f'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=')
    pred_assembly_seqs = loadAll(trial_ids, "pred-assembly-seq.pkl")
    gt_assembly_seqs = loadAll(trial_ids, "gt-assembly-seq.pkl")

    all_assemblies = []
    gt_label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(gt_assembly_seq, all_assemblies)))
        for gt_assembly_seq in gt_assembly_seqs
    )
    pred_label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(pred_assembly_seq, all_assemblies)))
        for pred_assembly_seq in pred_assembly_seqs
    )

    if plot_predictions:
        assembly_fig_dir = os.path.join(fig_dir, 'assembly-imgs')
        if not os.path.exists(assembly_fig_dir):
            os.makedirs(assembly_fig_dir)
        for i, assembly in enumerate(all_assemblies):
            assembly.draw(assembly_fig_dir, i)

    logger.info(f"Evaluating {len(trial_ids)} sequence predictions")

    accuracies = {
        'state': [],
        'is_error': [],
        'is_layer': []
    }
    data = zip(trial_ids, pred_assembly_seqs, gt_assembly_seqs)
    for i, trial_id in enumerate(trial_ids):
        pred_assembly_seq = pred_assembly_seqs[i]
        gt_assembly_seq = gt_assembly_seqs[i]

        task = int(metadata.iloc[trial_id]['task id'])
        goal = labels.constructGoalState(task)
        is_error = functools.partial(is_goal_error, goal)
        is_layer = functools.partial(is_goal_layer, goal)
        state = None

        logger.info(f"SEQUENCE {trial_id}: {len(gt_assembly_seq)} items")

        for name in accuracies.keys():
            if name == 'is_layer':
                pred_is_layer = np.array(list(map(is_layer, pred_assembly_seq)))
                gt_is_layer = np.array(list(map(is_layer, gt_assembly_seq)))
                matches = pred_is_layer == gt_is_layer
                acc = matches.sum() / len(matches)
                logger.info(f"  {name}: {acc:.2}")
                logger.info(f"    {gt_is_layer.sum():2}   gt layers")
                logger.info(f"    {pred_is_layer.sum():2} pred layers")
            else:
                acc = metrics.accuracy_upto(
                    # pred_assembly_seq[1:], gt_assembly_seq[1:],
                    pred_assembly_seq, gt_assembly_seq,
                    equivalence=locals()[name]
                )
                logger.info(f"  {name}: {acc:.2}")
            accuracies[name].append(acc)

        if plot_predictions:
            paths_dir = os.path.join(fig_dir, 'path-imgs')
            if not os.path.exists(paths_dir):
                os.makedirs(paths_dir)
            fn = f"trial={trial_id}_paths"
            paths = np.row_stack((gt_label_seqs[i], pred_label_seqs[i]))
            path_labels = np.row_stack((gt_is_layer, pred_is_layer))
            drawPaths(paths, fn, paths_dir, assembly_fig_dir, path_labels=path_labels)

    logger.info("EVALUATION RESULTS:")
    max_width = max(map(len, accuracies.keys()))
    for name, vals in accuracies.items():
        vals = np.array(vals) * 100
        mean = vals.mean()
        std = vals.std()
        logger.info(f"  {name:{max_width}}: {mean:4.1f} +/- {std:4.1f}%")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    for arg_name in inspect.getfullargspec(main).args:
        parser.add_argument(f'--{arg_name}')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    else:
        config_fn = os.path.basename(config_file_path)
    if os.path.exists(config_file_path):
        with open(config_file_path, 'rt') as config_file:
            config = yaml.safe_load(config_file)
    else:
        config = {}

    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
