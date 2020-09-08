import os
import logging

import yaml
import joblib
import numpy as np

from mathtools import utils
from kinemparse import airplanecorpus
from seqtools import utils as su


logger = logging.getLogger(__name__)


def makeAssemblyLabels(action_labels, part_names, num_samples, vocabulary={}):
    assembly_labels = np.full(num_samples, -1, dtype=int)

    def makeLabel(cur_assembly, next_assembly, start_index, end_index):
        transition = (frozenset(cur_assembly), frozenset(next_assembly))
        transition_index = vocabulary.get(transition, None)
        if transition_index is None:
            transition_index = len(vocabulary)
            vocabulary[transition] = transition_index

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
    makeLabel(prev_assembly, prev_assembly, prev_end_index, None)

    if (assembly_labels == -1).any():
        raise AssertionError()

    return assembly_labels


def makeAssemblyScores(
        bin_scores, part_names_to_idxs, part_idxs_to_bins, transition_vocabulary,
        initial_scores=None, final_scores=None):
    def transitionToPart(cur_assembly, next_assembly):
        if next_assembly == cur_assembly:
            return 'null'

        part_difference = tuple(next_assembly ^ cur_assembly)
        if len(part_difference) != 1:
            raise AssertionError()
        part = part_difference[0]
        return part

    transition_to_part_name = [transitionToPart(c, n) for (c, n) in transition_vocabulary]
    transition_to_part_index = [part_names_to_idxs[name] for name in transition_to_part_name]
    transition_to_bin = part_idxs_to_bins[transition_to_part_index]
    assembly_scores = bin_scores[:, transition_to_bin]

    if initial_scores is not None:
        assembly_scores[0, :] += initial_scores

    if final_scores is not None:
        assembly_scores[-1, :] += final_scores

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

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    logger.info(f"Writing to: {out_dir}")

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
        # write_mode = 'w'
    else:
        results_file = os.path.expanduser(results_file)
        # write_mode = 'a'

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    part_names, part_names_to_idxs, part_idxs_to_bins = airplanecorpus.loadParts()

    trial_ids = utils.getUniqueIds(bin_scores_dir, prefix='trial=')
    bin_score_seqs = loadAll(trial_ids, 'score-seq.pkl', bin_scores_dir)

    transition_to_index = {}
    assembly_label_seqs = []
    for i, trial_id in enumerate(trial_ids):
        video_id = utils.stripExtension(trial_id).replace('-', '_')
        action_labels = airplanecorpus.loadLabels(
            video_id, dir_name=action_labels_dir,
            part_names_to_idxs=part_names_to_idxs
        )

        # Make assembly labels from action labels
        assembly_labels = makeAssemblyLabels(
            action_labels, part_names, bin_score_seqs[i].shape[0],
            vocabulary=transition_to_index
        )
        assembly_label_seqs.append(assembly_labels)

    transition_scores, initial_scores, final_scores = su.smoothCounts(
        *su.countSeqs(assembly_label_seqs), as_scores=True
    )
    fn = os.path.join(fig_dir, "transitions.png")
    su.plot_transitions(fn, transition_scores, initial_scores, final_scores)

    num_items = len(transition_to_index)
    transition_vocabulary = {v: k for k, v in transition_to_index.items()}
    transition_vocabulary = tuple(transition_vocabulary[i] for i in range(num_items))

    for i, trial_id in enumerate(trial_ids):
        video_id = utils.stripExtension(trial_id).replace('-', '_')

        bin_score_seq = bin_score_seqs[i]
        assembly_labels = assembly_label_seqs[i]

        assembly_scores = makeAssemblyScores(
            bin_score_seq, part_names_to_idxs, part_idxs_to_bins, transition_vocabulary,
            initial_scores=initial_scores, final_scores=final_scores
        )

        fig_fn = os.path.join(fig_dir, f"{video_id}.png")
        utils.plot_array(assembly_scores.T, (assembly_labels,), ('assembly',), fn=fig_fn)

        video_id = video_id.replace('_', '-')
        saveVariable(assembly_scores, f'trial={video_id}_score-seq')
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
