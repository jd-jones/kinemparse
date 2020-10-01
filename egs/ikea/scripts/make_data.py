import os
import logging
import glob

import yaml
import joblib
import numpy as np
import pandas as pd

from mathtools import utils, pose
from kinemparse import assembly as lib_assembly


logger = logging.getLogger(__name__)


def relativePose(poses_seq, lower_tri_only=True, magnitude_only=False):
    num_poses = poses_seq.shape[-1]
    rel_poses = np.stack(
        tuple(
            np.stack(
                tuple(
                    pose.relPose(
                        poses_seq[..., i], poses_seq[..., j],
                        magnitude_only=magnitude_only
                    )
                    for j in range(num_poses)
                ),
                axis=1
            )
            for i in range(num_poses)
        ), axis=1
    )

    if lower_tri_only:
        rows, cols = np.tril_indices(rel_poses.shape[1], k=-1)
        rel_poses = rel_poses[:, rows, cols]

    return rel_poses


def actionLabels(
        labels_arr, num_samples, action_name_to_index, part_name_to_index,
        lower_tri_only=True):
    num_parts = len(part_name_to_index)
    label_seq = np.zeros((num_samples, num_parts, num_parts), dtype=int)

    for i, (start_idx, end_idx, action, part1, part2) in labels_arr.iterrows():
        action_idx = action_name_to_index[action]
        part1_idx = part_name_to_index[part1]
        part2_idx = part_name_to_index[part2]

        label_seq[start_idx:end_idx, part1_idx, part2_idx] = action_idx + 1
        label_seq[start_idx:end_idx, part2_idx, part1_idx] = action_idx + 1

    if lower_tri_only:
        rows, cols = np.tril_indices(label_seq.shape[1], k=-1)
        label_seq = label_seq[:, rows, cols]

    return label_seq


def assemblyLabels(labels_arr, num_samples, assembly_vocab=[]):
    def getIndex(assembly):
        for i, a in enumerate(assembly_vocab):
            if a == assembly:
                return i
        else:
            assembly_vocab.append(assembly)
            return len(assembly_vocab) - 1

    label_seq = np.zeros(num_samples, dtype=int)

    assembly = lib_assembly.Assembly()
    assembly_index = getIndex(assembly)
    prev_end_idx = 0
    for i, (start_idx, end_idx, action, part1, part2) in labels_arr.iterrows():
        label_seq[prev_end_idx:end_idx] = assembly_index
        if action == 'connect':
            assembly = assembly.add_joint(part1, part2, in_place=False, directed=False)
        elif action == 'disconnect':
            assembly = assembly.remove_joint(part1, part2, in_place=False, directed=False)
        assembly_index = getIndex(assembly)
        prev_end_idx = end_idx
    label_seq[end_idx:] = assembly_index

    return label_seq


def makePairs(seq, lower_tri_only=True):
    pairs = tuple(tuple((token1, token2) for token2 in seq) for token1 in seq)

    if lower_tri_only:
        rows, cols = np.tril_indices(len(seq), k=-1)
        pairs = tuple(pairs[r][c] for r, c in zip(rows, cols))

    return pairs


def possibleConnections(part_pair_names):
    def holeInfo(part_name):
        part_name, hole_idx = part_name.split('_hole_')
        return part_name, hole_idx

    def connectionPossible(name_A, name_B):
        name_A, hole_A = holeInfo(name_A)
        name_B, hole_B = holeInfo(name_B)
        if name_A == name_B:
            return False
        return True

    return np.array([connectionPossible(a, b) for a, b in part_pair_names])


def main(
        out_dir=None, data_dir=None,
        plot_output=None, results_file=None, sweep_param_name=None, start_from=None):

    data_dir = os.path.expanduser(data_dir)

    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    debug_dir = os.path.join(out_dir, 'debug')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    logger.info(f"Reading from: {data_dir}")
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

    labels_dir = os.path.join(data_dir, 'labels')

    with open(os.path.join(labels_dir, 'action_and_part_names.yaml'), 'rt') as f:
        names = yaml.safe_load(f)
    action_names = names['action_names']
    action_name_to_index = {name: i for i, name in enumerate(action_names)}
    part_names = names['part_names']
    part_name_to_index = {name: i for i, name in enumerate(part_names)}

    video_ids = []
    all_label_arrs = []
    for label_fn in glob.glob(os.path.join(labels_dir, "*.csv")):
        video_id = utils.stripExtension(label_fn)
        labels_arr = pd.read_csv(label_fn)
        all_label_arrs.append(labels_arr)
        video_ids.append(video_id)

    pose_dir = os.path.join(data_dir, 'poses')
    pose_ids = tuple(
        video_id
        for video_id in video_ids
        if os.path.exists(os.path.join(pose_dir, video_id))
    )
    keep_ids = tuple(v_id in pose_ids for v_id in video_ids)
    logger.info(
        f"Ignoring {len(keep_ids) - sum(keep_ids)} video(s) with missing data: "
        f"{', '.join([v_id for v_id, keep in zip(video_ids, keep_ids) if not keep])}"
    )

    def filterSeq(seq):
        return tuple(x for x, keep_id in zip(seq, keep_ids) if keep_id)
    all_label_arrs = filterSeq(all_label_arrs)
    video_ids = filterSeq(video_ids)

    assembly_vocab = []
    label_seqs = []
    for i, video_id in enumerate(video_ids):
        if start_from is not None and i < start_from:
            continue

        logger.info("PROCESSING VIDEO {0}: {1}".format(i, video_id))

        labels_arr = all_label_arrs[i]

        video_dir = os.path.join(pose_dir, video_id)

        def loadFile(part_name):
            path = os.path.join(video_dir, f'{part_name}.csv')
            arr = pd.read_csv(path)
            return arr

        part_data = tuple(loadFile(part_name) for part_name in part_names)

        poses_seq = np.stack(tuple(arr.values for arr in part_data), axis=-1)

        feature_seq = relativePose(poses_seq, lower_tri_only=True, magnitude_only=True)
        label_seq = actionLabels(
            labels_arr, feature_seq.shape[0],
            action_name_to_index, part_name_to_index
        )

        part_pair_names = makePairs(part_names, lower_tri_only=True)
        is_possible = possibleConnections(part_pair_names)
        feature_seq = feature_seq[:, is_possible, :]
        label_seq = label_seq[:, is_possible]
        part_pair_names = tuple(n for (b, n) in zip(is_possible, part_pair_names) if b)

        utils.plot_multi(
            np.moveaxis(feature_seq, (0, 1, 2), (-1, 0, 1)), label_seq.T,
            axis_names=part_pair_names, label_name='action',
            feature_names=('translation_dist', 'rotation_dist'),
            tick_names=[''] + action_names,
            fn=os.path.join(fig_dir, f"{video_id}_actions.png")
        )

        label_seq = assemblyLabels(
            labels_arr, feature_seq.shape[0],
            assembly_vocab=assembly_vocab
        )
        utils.plot_array(
            feature_seq.sum(axis=-1).T, (label_seq,), ('assembly',),
            fn=os.path.join(fig_dir, f"{video_id}_assemblies.png")
        )
        label_seqs.append(label_seq)

        label_segments, __ = utils.computeSegments(label_seq)
        assembly_segments = [assembly_vocab[i] for i in label_segments]
        lib_assembly.writeAssemblies(
            os.path.join(debug_dir, f'trial={video_id}_assembly-seq.txt'),
            assembly_segments
        )

        video_id = video_id.replace('_', '-')
        saveVariable(feature_seq, f'trial={video_id}_feature-seq')
        saveVariable(label_seq, f'trial={video_id}_label-seq')

    if False:
        from seqtools import utils as su
        transition_probs, start_probs, end_probs = su.smoothCounts(
            *su.countSeqs(label_seqs)
        )
        # import pdb; pdb.set_trace()

    lib_assembly.writeAssemblies(
        os.path.join(debug_dir, 'assembly-vocab.txt'),
        assembly_vocab
    )

    saveVariable(assembly_vocab, 'assembly-vocab')
    with open(os.path.join(out_data_dir, 'action-vocab.yaml'), 'wt') as f:
        yaml.dump(action_names, f)
    with open(os.path.join(out_data_dir, 'part-vocab.yaml'), 'wt') as f:
        yaml.dump(part_names, f)


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
