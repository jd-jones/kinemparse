#!/usr/bin/env/python
import os
import glob
import logging
import collections

import numpy as np
import pandas as pd
import yaml

import _utils as ikea_utils
from mathtools import utils


CAMERA_NAMES = ('upper', 'lower')
POSE_VAR_NAMES = ['p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w']


logger = logging.getLogger(__name__)


def collectMarkerPoses(marker_sample_seqs, marker_keys):
    marker_sample_seqs = [marker_sample_seqs[k] for k in marker_keys]

    min_len = min(x.shape[0] for x in marker_sample_seqs)
    for i in range(len(marker_sample_seqs)):
        num_samples = marker_sample_seqs[i].shape[0]
        if min_len < num_samples:
            marker_sample_seqs[i] = marker_sample_seqs[i][:min_len, :]
            logger.info(
                f"Truncated seq of len {num_samples} "
                f"to {marker_sample_seqs[i].shape[0]}"
            )

    marker_index_seqs = tuple(seq[:, :1] for seq in marker_sample_seqs)
    marker_pose_seqs = tuple(seq[:, 1:] for seq in marker_sample_seqs)

    return marker_index_seqs, marker_pose_seqs


def cameraIndices(marker_pose_seqs, marker_keys, camera_name):
    frame_indices = np.hstack(tuple(
        seq for seq, k in zip(marker_pose_seqs, marker_keys)
        if k[0] == camera_name
    ))

    if not np.all(frame_indices == frame_indices[:, 0:1]):
        # import pdb; pdb.set_trace()
        raise AssertionError()

    return frame_indices[:, 0]


def readFrameFns(fn, names_as_int=False):
    frame_fns = pd.read_csv(fn, header=None).iloc[:, 0].tolist()

    if names_as_int:
        frame_fns = np.array([int(fn.strip('.png').strip('frame')) for fn in frame_fns])

    return frame_fns


def frameNamesToSampleIndices(labels, frame_fns, frame_idx_seq):
    def sampleIndexFromFrameName(label_frame_names):
        # import pdb; pdb.set_trace()
        label_frame_idxs = utils.firstMatchingIndex(
            frame_fns, label_frame_names,
            check_single=False
        )
        label_sample_idxs = utils.firstMatchingIndex(
            frame_idx_seq, label_frame_idxs,
            check_single=False
        )
        return label_sample_idxs

    for key in ('start', 'end'):
        labels[key] = sampleIndexFromFrameName(labels[key].to_numpy())

    return labels


def main(
        out_dir=None, marker_pose_dir=None, marker_bundles_dir=None, labels_dir=None,
        urdf_file=None, start_from=None, rename_parts={}):
    marker_pose_dir = os.path.expanduser(marker_pose_dir)
    marker_bundles_dir = os.path.expanduser(marker_bundles_dir)
    labels_dir = os.path.expanduser(labels_dir)
    urdf_file = os.path.expanduser(urdf_file)

    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    out_labels_dir = os.path.join(out_data_dir, 'labels')
    if not os.path.exists(out_labels_dir):
        os.makedirs(out_labels_dir)
    out_idxs_dir = os.path.join(out_data_dir, 'frame-indices')
    if not os.path.exists(out_idxs_dir):
        os.makedirs(out_idxs_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    diffs_dir = os.path.join(fig_dir, 'diffs')
    if not os.path.exists(diffs_dir):
        os.makedirs(diffs_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    marker_to_part = {}
    for bundle_fn in glob.glob(os.path.join(marker_bundles_dir, '*.xml')):
        part_name, bundle_data = ikea_utils.getBundleData(bundle_fn)
        for marker_index, pose in bundle_data.items():
            marker_to_part[marker_index] = part_name

    action_names = frozenset()
    part_names = frozenset()
    for i, label_fn in enumerate(glob.glob(os.path.join(labels_dir, "*.txt"))):
        # Load labels
        labels, video_id, camera_name = ikea_utils.readLabels(
            label_fn, rename_parts=rename_parts
        )

        if not os.path.exists(os.path.join(marker_pose_dir, video_id)):
            logger.warning(f"Skipping video {video_id}: no marker poses")
            continue

        # Load frame fns
        frame_fns = readFrameFns(
            os.path.join(marker_pose_dir, video_id, 'metadata', f"frame-fns_{camera_name}.csv"),
            names_as_int=True
        )

        logger.info("PROCESSING VIDEO {0}: {1}".format(i, video_id))

        out_video_dir = os.path.join(out_data_dir, video_id)
        if not os.path.exists(out_video_dir):
            os.makedirs(out_video_dir)

        marker_samples = {}
        part_to_keys = collections.defaultdict(list)
        for file_path in glob.glob(os.path.join(marker_pose_dir, video_id, '*.csv')):
            file_id = ikea_utils.basename(file_path)
            camera_name, marker_index = ikea_utils.metadataFromTopicName(file_id)
            part_name = marker_to_part[marker_index]
            marker_pose_seq = ikea_utils.readMarkerPoses(file_path, pose_and_index=True)
            if np.isnan(marker_pose_seq[:, 1:]).all():
                continue

            marker_samples[camera_name, marker_index] = marker_pose_seq
            part_to_keys[part_name].append((camera_name, marker_index))

        all_part_frame_seqs = []
        for part_name, marker_keys in part_to_keys.items():
            marker_index_seqs, marker_pose_seqs = collectMarkerPoses(marker_samples, marker_keys)
            part_pose_seq = ikea_utils.avgPose(*marker_pose_seqs)

            # VISUALIZE POSES
            ikea_utils.plotPoses(
                os.path.join(fig_dir, f"{video_id}_part={part_name}_all-markers.png"),
                *marker_pose_seqs,
                labels=tuple(', '.join(tuple(map(str, k))) for k in marker_keys)
            )
            ikea_utils.plotPoses(
                os.path.join(fig_dir, f"{video_id}_part={part_name}.png"),
                part_pose_seq
            )

            # SAVE POSES
            pd.DataFrame(data=part_pose_seq, columns=POSE_VAR_NAMES).to_csv(
                os.path.join(out_video_dir, '{0}_poses.csv'.format(part_name)),
                index=False
            )

            part_frame_seq = np.column_stack(tuple(
                cameraIndices(marker_index_seqs, marker_keys, name)
                for name in CAMERA_NAMES
            ))
            all_part_frame_seqs.append(part_frame_seq)

        all_part_frame_seqs = np.stack(tuple(all_part_frame_seqs), axis=-1)
        if not np.all(all_part_frame_seqs == all_part_frame_seqs[:, :, :1]):
            # import pdb; pdb.set_trace()
            raise AssertionError()
        frame_idx_seq = all_part_frame_seqs[..., 0]

        labels = frameNamesToSampleIndices(labels, frame_fns, frame_idx_seq)

        # save labels
        labels.to_csv(os.path.join(out_labels_dir, f"{video_id}.csv"), index=False)

        pd.DataFrame(data=frame_idx_seq, columns=CAMERA_NAMES).to_csv(
            os.path.join(out_idxs_dir, '{0}.csv'.format(part_name)),
            index=False
        )

        action_names = action_names | frozenset(labels['action'].tolist())
        part_names = part_names | frozenset(labels['arg1'].tolist())
        part_names = part_names | frozenset(labels['arg2'].tolist())

    with open(os.path.join(out_labels_dir, 'action_and_part_names.yaml'), 'wt') as f:
        yaml.dump({'action_names': list(action_names), 'part_names': list(part_names)}, f)


if __name__ == '__main__':
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
