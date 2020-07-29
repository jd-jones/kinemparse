import os
import logging
import glob
import collections

import xml.etree.ElementTree as ET
import yaml
import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from mathtools import utils


logger = logging.getLogger(__name__)


def parseBundleFile(bundle_fn):
    root = ET.parse(bundle_fn).getroot()

    # num_markers = root.get('markers')

    def getIndex(element):
        status = int(element.get('status'))
        if status != 1:
            raise AssertionError("Encountered a bad status value!")

        index = element.get('index')
        return index

    # When ar_track_alvar is tracking marker bundles, it chooses the first
    # marker in the file as the "principal" marker. All the other marker poses
    # are given with respect to this one.
    principal_marker = getIndex(root.find('marker'))
    all_markers = tuple(map(getIndex, root.findall('marker')))

    return principal_marker, all_markers


def stackPartPoses(part_pose_dict):
    def changeColNames(df, part_name, marker_size, camera_pos):
        df.columns = [
            f"{part_name}_{marker_size}_{camera_pos}_{col_name}"
            for col_name in df.columns
        ]
        return df

    dfs = (
        changeColNames(pose_seq, p_name, marker_size, camera_pos)
        for p_name, d1 in part_pose_dict.items()
        for marker_size, d2 in d1.items()
        for camera_pos, pose_seq in d2.items()
    )
    all_poses = pd.concat(dfs, axis=1)

    return all_poses


def vizPartPoses(part_pose_dict, fn=None):
    num_subfigs = len(part_pose_dict)
    fig_size = (12, 3 * num_subfigs)
    f, axes = plt.subplots(num_subfigs, figsize=fig_size, sharex=True)

    for i, (p_name, d1) in enumerate(part_pose_dict.items()):
        axis = axes[i]
        axis.set_ylabel(p_name)
        for marker_size, d2 in d1.items():
            for camera_pos, pose_seq in d2.items():
                pos_seq = pose_seq.filter(like='p_').values
                # pos_seq = np.ma.array(pos_seq, mask=np.isnan(pos_seq))
                pos_norms = np.linalg.norm(pos_seq, axis=1)
                axis.plot(pos_norms, label=f"{marker_size} {camera_pos}")
                # import pdb; pdb.set_trace()

    for axis in axes:
        axis.legend()

    plt.tight_layout()

    plt.savefig(fn)
    plt.close()


def main(
        out_dir=None,
        marker_pose_dir=None, marker_bundles_dir=None, urdf_dir=None,
        plot_output=None, results_file=None, sweep_param_name=None):

    marker_pose_dir = os.path.expanduser(marker_pose_dir)
    marker_bundles_dir = os.path.expanduser(marker_bundles_dir)
    urdf_dir = os.path.expanduser(urdf_dir)
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

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load parts and their marker bundles
    part_markers = collections.defaultdict(dict)
    for bundle_fn in glob.glob(os.path.join(marker_bundles_dir, "*.xml")):
        part_name, marker_size = utils.stripExtension(bundle_fn).split('_tags_')
        principal_marker, all_markers = parseBundleFile(bundle_fn)

        part_markers[part_name][marker_size] = principal_marker
    marker_to_part = {
        marker_id: (part_name, marker_size)
        for part_name, marker_size_dict in part_markers.items()
        for marker_size, marker_id in marker_size_dict.items()
    }

    video_ids = os.listdir(marker_pose_dir)
    for video_id in video_ids:
        logger.info(f"Processing video {video_id}")

        part_poses = collections.defaultdict(lambda: collections.defaultdict(dict))

        video_dir = os.path.join(marker_pose_dir, video_id)
        marker_pose_fns = glob.glob(os.path.join(video_dir, '*.csv'))
        for marker_pose_fn in marker_pose_fns:
            camera_pos, marker_id = utils.stripExtension(marker_pose_fn).split('_cam_marker_')

            try:
                part_name, marker_size = marker_to_part[marker_id]
            except KeyError:
                continue

            marker_pose_seq = pd.read_csv(marker_pose_fn)
            part_poses[part_name][marker_size][camera_pos] = marker_pose_seq

        vizPartPoses(part_poses, fn=os.path.join(fig_dir, f"{video_id}.png"))

        part_poses = stackPartPoses(part_poses)
        part_poses.to_csv(os.path.join(out_data_dir, f"{video_id}.csv"), index=False)


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
