import os
import logging
import glob

import yaml
import joblib
import numpy as np
import pandas as pd

from mathtools import utils
import _utils as ikea_utils


logger = logging.getLogger(__name__)


def relativePose(poses_seq, lower_tri_only=True, magnitude_only=False):
    num_poses = poses_seq.shape[-1]
    rel_poses = np.stack(
        tuple(
            np.stack(
                tuple(
                    ikea_utils.relPose(
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


def assemblyLabels(
        labels_arr, num_samples, action_name_to_index, part_name_to_index,
        lower_tri_only=True):
    num_parts = len(part_name_to_index)
    label_seq = np.zeros((num_samples, num_parts, num_parts), dtype=int)

    for i, (start_idx, end_idx, action, part1, part2) in labels_arr.iterrows():
        action_idx = action_name_to_index[action]
        part1_idx = part_name_to_index[part1]
        part2_idx = part_name_to_index[part2]

        label_seq[start_idx:end_idx, part1_idx, part2_idx] = action_idx + 1

    if lower_tri_only:
        rows, cols = np.tril_indices(label_seq.shape[1], k=-1)
        label_seq = label_seq[:, rows, cols]

    return label_seq


def makePairs(seq, lower_tri_only=True):
    pairs = tuple(tuple((token1, token2) for token2 in seq) for token1 in seq)

    if lower_tri_only:
        rows, cols = np.tril_indices(len(seq), k=-1)
        pairs = tuple(pairs[r][c] for r, c in zip(rows, cols))

    return pairs


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

    pose_ids = tuple(
        video_id
        for video_id in video_ids
        if os.path.exists(os.path.join(data_dir, video_id))
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

    for i, video_id in enumerate(video_ids):
        if start_from is not None and i < start_from:
            continue

        logger.info("PROCESSING VIDEO {0}: {1}".format(i, video_id))

        labels_arr = all_label_arrs[i]

        video_dir = os.path.join(data_dir, video_id)

        def loadFile(part_name):
            path = os.path.join(video_dir, f'{part_name}_poses.csv')
            arr = pd.read_csv(path)
            return arr

        part_data = tuple(loadFile(part_name) for part_name in part_names)

        poses_seq = np.stack(tuple(arr.values for arr in part_data), axis=-1)

        feature_seq = relativePose(poses_seq, lower_tri_only=True, magnitude_only=True)
        label_seq = assemblyLabels(
            labels_arr, feature_seq.shape[0],
            action_name_to_index, part_name_to_index
        )

        part_pair_names = makePairs(part_names, lower_tri_only=True)

        utils.plot_multi(
            np.moveaxis(feature_seq, (0, 1, 2), (-1, 0, 1)), label_seq.T,
            axis_names=part_pair_names, label_name='action',
            feature_names=('translation_dist', 'rotation_dist'),
            tick_names=[''] + action_names,
            fn=os.path.join(fig_dir, f"{video_id}.png")
        )

        # utils.plot_array(feature_seq.T, (label_seq.sum(axis=-1),), ('activity',), fn=fig_fn)

        saveVariable(feature_seq, f'trial={video_id}_feature-seq')
        saveVariable(label_seq, f'trial={video_id}_label-seq')


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
