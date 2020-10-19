import argparse
import os
import inspect
import warnings

import yaml
import numpy as np
import joblib
import graphviz as gv

# Stop numba from throwing a bunch of warnings when it compiles LCTM
from numba import NumbaWarning; warnings.filterwarnings('ignore', category=NumbaWarning)
import LCTM.metrics

from mathtools import utils
from blocks.core import labels, blockassembly


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


def actionsFromAssemblies(assembly_seq):
    def subtract(lhs, rhs):
        try:
            difference = lhs - rhs
        except ValueError as e:
            difference = lhs - blockassembly.BlockAssembly()
            logger.warning(e)
        return difference

    action_seq = (blockassembly.AssemblyAction(),)
    action_seq += tuple(subtract(n, c) for c, n in zip(assembly_seq[:-1], assembly_seq[1:]))
    return action_seq


def main(
        out_dir=None, data_dir=None, metadata_file=None, metric_names=None,
        ignore_initial_state=None,
        draw_paths=None, plot_predictions=None, results_file=None, sweep_param_name=None):

    if metric_names is None:
        metric_names = ('accuracy', 'edit_score', 'overlap_score')

    logger.info(f"Reading from: {data_dir}")
    logger.info(f"Writing to: {out_dir}")

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
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

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)
    pred_assembly_seqs = loadAll(trial_ids, "pred-assembly-seq.pkl")
    gt_assembly_seqs = loadAll(trial_ids, "gt-assembly-seq.pkl")

    all_assemblies = []
    gt_assembly_label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(gt_assembly_seq, all_assemblies)))
        for gt_assembly_seq in gt_assembly_seqs
    )
    pred_assembly_label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(pred_assembly_seq, all_assemblies)))
        for pred_assembly_seq in pred_assembly_seqs
    )

    def checkPredictions(pred_assembly_label_seqs, gt_assembly_label_seqs, trial_ids):
        for i, trial_id in enumerate(trial_ids):
            pred_seq = pred_assembly_label_seqs[i]
            true_seq = gt_assembly_label_seqs[i]

            train_seqs = gt_assembly_label_seqs[:i] + gt_assembly_label_seqs[i + 1:]
            train_vocab = np.unique(np.hstack(train_seqs))
            pred_is_correct = pred_seq == true_seq
            pred_in_test_vocab = np.array([np.any(train_vocab == i) for i in pred_seq])
            pred_is_oov_and_correct = ~pred_in_test_vocab * pred_is_correct
            if not pred_in_test_vocab.all():
                warn_str = (
                    f"trial {trial_id}: {np.sum(~pred_in_test_vocab)} / "
                    f"{len(pred_in_test_vocab)} preds not in test; "
                    f"{np.sum(pred_is_oov_and_correct)} oov preds are correct"
                )
                logger.warning(warn_str)

    checkPredictions(pred_assembly_label_seqs, gt_assembly_label_seqs, trial_ids)

    def estimate_oov_rate(label_seqs):
        num_oov = 0
        num_items = 0
        for i in range(len(label_seqs)):
            test_seq = label_seqs[i]
            train_seqs = label_seqs[:i] + label_seqs[i + 1:]
            train_vocab = np.unique(np.hstack(train_seqs))

            num_oov += sum(float(not np.any(train_vocab == i)) for i in test_seq)
            num_items += len(test_seq)

        return num_oov / num_items

    oov_rate = estimate_oov_rate(gt_assembly_label_seqs)
    logger.info(f"OOV rate: {oov_rate * 100:.2f}%")

    pred_action_seqs = tuple(map(actionsFromAssemblies, pred_assembly_seqs))
    gt_action_seqs = tuple(map(actionsFromAssemblies, gt_assembly_seqs))

    all_actions = []
    gt_action_label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(gt_action_seq, all_actions)))
        for gt_action_seq in gt_action_seqs
    )
    pred_action_label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(pred_action_seq, all_actions)))
        for pred_action_seq in pred_action_seqs
    )

    if draw_paths:
        assembly_fig_dir = os.path.join(fig_dir, 'assembly-imgs')
        if not os.path.exists(assembly_fig_dir):
            os.makedirs(assembly_fig_dir)
        for i, assembly in enumerate(all_assemblies):
            assembly.draw(assembly_fig_dir, i)

        action_fig_dir = os.path.join(fig_dir, 'action-imgs')
        if not os.path.exists(action_fig_dir):
            os.makedirs(action_fig_dir)
        for i, action in enumerate(all_actions):
            action.draw(action_fig_dir, i)

    logger.info(f"Evaluating {len(trial_ids)} sequence predictions")

    batch = []
    for i, trial_id in enumerate(trial_ids):
        logger.info(f"VIDEO {trial_id}:")

        pred_assembly_index_seq = pred_assembly_label_seqs[i]
        true_assembly_index_seq = gt_assembly_label_seqs[i]

        if ignore_initial_state:
            pred_assembly_index_seq = pred_assembly_index_seq[1:]
            true_assembly_index_seq = true_assembly_index_seq[1:]

        pred_action_index_seq = pred_action_label_seqs[i]
        true_action_index_seq = gt_action_label_seqs[i]

        if draw_paths:
            drawPaths(
                [pred_assembly_index_seq, true_assembly_index_seq],
                f"trial={trial_id}_assembly-paths", fig_dir, assembly_fig_dir,
                path_labels=None, img_ext='png'
            )

            drawPaths(
                [pred_action_index_seq, true_action_index_seq],
                f"trial={trial_id}_action-paths", fig_dir, action_fig_dir,
                path_labels=None, img_ext='png'
            )

        metric_dict = {}
        for name in metric_names:
            key = f"{name}_action"
            value = getattr(LCTM.metrics, name)(pred_action_index_seq, true_action_index_seq) / 100
            metric_dict[key] = value
            logger.info(f"  {key}: {value * 100:.1f}%")

            key = f"{name}_assembly"
            value = getattr(LCTM.metrics, name)(
                pred_assembly_index_seq, true_assembly_index_seq
            ) / 100
            metric_dict[key] = value
            logger.info(f"  {key}: {value * 100:.1f}%")

        utils.writeResults(results_file, metric_dict, sweep_param_name, {})

        batch.append((pred_action_index_seq, None, true_action_index_seq, trial_id))

    if plot_predictions:
        label_names = ('gt', 'pred')
        for preds, inputs, gt_labels, seq_id in batch:
            fn = os.path.join(fig_dir, f"trial={seq_id}_model-io.png")
            utils.plot_array(inputs, (gt_labels, preds), label_names, fn=fn, labels_together=True)


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
