import os
import logging
import glob

import yaml
import joblib
import numpy as np

from mathtools import utils
from kinemparse import airplanecorpus


logger = logging.getLogger(__name__)


def makeAssemblyLabels(action_labels, part_names, num_samples, vocabulary={}):
    assembly_labels = np.full(num_samples, -1, dtype=int)

    def makeLabel(cur_assembly, next_assembly, start_index, end_index):
        transition = (frozenset(cur_assembly), frozenset(next_assembly))
        transition_index = vocabulary.get(transition, None)
        if transition_index is None:
            vocabulary[transition] = len(vocabulary)

        assembly_labels[start_index:end_index] = transition_index

    prev_assembly = set()
    prev_end_index = 0
    for part_index, start_index, end_index in action_labels:
        part_name = part_names[part_index]

        cur_assembly = prev_assembly.copy()
        cur_assembly.add(part_name)

        makeLabel(prev_assembly, prev_assembly, prev_end_index, start_index)
        makeLabel(prev_assembly, cur_assembly, start_index, end_index + 1)

        prev_assembly = cur_assembly
        prev_end_index = end_index
    makeLabel(prev_assembly, prev_assembly, prev_end_index, -1)

    if (assembly_labels == -1).any():
        raise AssertionError()

    return assembly_labels


def makeAssemblyScores(bin_scores, part_idxs_to_bins, transition_vocabulary):
    assembly_scores = bin_scores
    return assembly_scores


def main(
        out_dir=None, bin_scores_dir=None, action_labels_dir=None,
        plot_output=None, results_file=None, sweep_param_name=None):

    bin_scores_dir = os.path.expanduser(bin_scores_dir)
    action_labels_dir = os.path.expanduser(action_labels_dir)

    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    logger.info(f"Writing to: {out_dir}")

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
        # write_mode = 'w'
    else:
        results_file = os.path.expanduser(results_file)
        # write_mode = 'a'

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    part_names, part_names_to_idxs, part_idxs_to_bins = airplanecorpus.loadParts()

    transition_vocabulary = {}
    all_assembly_labels = []
    for i, fn in enumerate(glob.glob(os.path.join(bin_scores_dir, '*.pkl'))):
        video_id = utils.stripExtension(fn).replace('-', '_')

        bin_scores = joblib.load(fn)
        action_labels = airplanecorpus.loadLabels(
            video_id, dir_name=action_labels_dir,
            part_idxs_to_bins=part_idxs_to_bins
        )

        # Make assembly labels from action labels
        assembly_labels = makeAssemblyLabels(
            action_labels, part_names, bin_scores.shape[0],
            vocabulary=transition_vocabulary
        )
        all_assembly_labels.append(assembly_labels)

    for i, fn in enumerate(glob.glob(os.path.join(bin_scores_dir, '*.pkl'))):
        video_id = utils.stripExtension(fn).replace('-', '_')

        bin_scores = joblib.load(fn)
        assembly_labels = all_assembly_labels[i]

        assembly_scores = makeAssemblyScores(bin_scores, part_idxs_to_bins, transition_vocabulary)

        fig_fn = os.path.join(fig_dir, f"{video_id}.png")
        utils.plot_array(assembly_scores.T, (assembly_labels,), ('assembly',), fn=fig_fn)

        video_id = video_id.replace('_', '-')
        saveVariable(assembly_scores, f'trial={video_id}_feature-seq')
        saveVariable(assembly_labels, f'trial={video_id}_label-seq')


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
