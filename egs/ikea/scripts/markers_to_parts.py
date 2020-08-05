#!/usr/bin/env/python
import os
import glob
import collections

import numpy as np
from scipy.spatial.transform import Rotation

import _utils as ikea_utils
from mathtools import utils


def markerPosesToPartPoses(marker_pose_worldframe_seq, marker_pose_partframe):
    if np.isnan(marker_pose_worldframe_seq).any():
        row_is_valid = ~(np.isnan(marker_pose_worldframe_seq).any(axis=1))
        valid_marker_poses = marker_pose_worldframe_seq[row_is_valid, :]
        valid_part_poses = markerPosesToPartPoses(valid_marker_poses, marker_pose_partframe)

        part_poses = marker_pose_worldframe_seq.copy()
        part_poses[row_is_valid, 1:] = valid_part_poses

        return part_poses

    t_wm = marker_pose_worldframe_seq[:, 1:4]
    q_wm = marker_pose_worldframe_seq[:, 4:8]  # orientation as quaternion

    # orientation as abstract rotation
    r_wm = Rotation.from_quat(q_wm)

    t_pm, R_pm = marker_pose_partframe
    r_pm = Rotation.from_matrix(R_pm)

    # FOR COLUMN VECTORS:
    # x_p = R_pm x_m + t_pm
    # x_m = R_pm_inv ( x_p - t_pm )
    # R_pm_inv = R_pm.T
    # x_m = R_pm.T x_p - R_pm.T t_pm
    #     = R_mp x_p + t_mp
    # --> R_mp = R_pm.T    t_mp = - R_pm.T t_pm
    #
    # CONVERTING TO ROW VECTORS:
    # x_m.T = (R_mp x_p + t_mp).T
    #       = x_p.T R_mp.T + t_mp.T
    #       = x_p.T R_pm + t_mp.T
    r_mp = r_pm.inv()
    t_mp = r_mp.apply(-t_pm)

    t_wp = r_wm.apply(t_mp) + t_wm
    r_wp = r_wm * r_mp  # FIXME: crashes on error

    # orientation as quaternion
    q_wp = np.full(q_wm.shape, np.nan)
    q_wp = r_wp.as_quat()

    part_pose_worldframe_seq = np.hstack((t_wp, q_wp))
    return part_pose_worldframe_seq


def main(out_dir=None, marker_pose_dir=None, marker_bundles_dir=None, start_from=None):
    marker_pose_dir = os.path.expanduser(marker_pose_dir)
    marker_bundles_dir = os.path.expanduser(marker_bundles_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    marker_to_part = {}
    marker_poses_partframe = {}
    for bundle_fn in glob.glob(os.path.join(marker_bundles_dir, '*.xml')):
        part_name, bundle_data = ikea_utils.getBundleData(bundle_fn)
        for marker_index, pose in bundle_data.items():
            marker_poses_partframe[marker_index] = pose
            marker_to_part[marker_index] = part_name

    for i, video_dir in enumerate(glob.glob(os.path.join(marker_pose_dir, '*'))):
        if start_from is not None and i < start_from:
            continue

        video_id = os.path.basename(os.path.normpath(video_dir))

        video_out_dir = os.path.join(out_dir, video_id)
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)

        print("PROCESSING VIDEO {0}: {1}".format(i, video_id))

        out_video_dir = os.path.join(out_dir, video_id)
        if not os.path.exists(out_video_dir):
            os.makedirs(out_video_dir)

        all_part_poses = collections.defaultdict(list)
        for file_path in glob.glob(os.path.join(video_dir, '*.csv')):
            camera_name, marker_index = ikea_utils.metadataFromTopicName(
                ikea_utils.basename(file_path)
            )
            part_name = marker_to_part[marker_index]
            marker_pose_partframe = marker_poses_partframe[marker_index]

            # Convert marker pose to part pose
            marker_pose_seq = ikea_utils.readMarkerPoses(file_path)
            part_pose_seq = markerPosesToPartPoses(marker_pose_seq, marker_pose_partframe)
            all_part_poses[part_name].append(part_pose_seq)

        for part_name, part_pose_seqs in all_part_poses.items():
            part_pose_seqs = np.stack(part_pose_seqs, axis=2)
            part_pose_seq = part_pose_seqs.mean(axis=2)

            fn = os.path.join(video_out_dir, "{0}.csv".format(part_name))
            np.savetxt(fn, part_pose_seq)


if __name__ == '__main__':
    cl_args = utils.parse_args(main)
    main(**cl_args)
