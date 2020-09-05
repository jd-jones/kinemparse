import os
import logging
import glob

import yaml
import numpy as np
from scipy import optimize

from mathtools import utils
from visiontools import utils as vu
from kinemparse import airplanecorpus


logger = logging.getLogger(__name__)


def associate_detections(prev_feats, cur_feats):
    residuals = cur_feats[:, None, :] - prev_feats[None, :, :]
    costs = np.linalg.norm(residuals, axis=2)

    costs[np.isnan(costs)] = np.finfo(float).max

    cur_indices, prev_indices = optimize.linear_sum_assignment(costs)
    return prev_indices


def main(
        out_dir=None,
        videos_dir=None, hand_detections_dir=None,
        plot_output=None, results_file=None, sweep_param_name=None):

    videos_dir = os.path.expanduser(videos_dir)
    hand_detections_dir = os.path.expanduser(hand_detections_dir)

    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

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

    for i, video_fn in enumerate(glob.glob(os.path.join(videos_dir, '*.avi'))):
        video_id = utils.stripExtension(video_fn)

        logger.info(f"Processing video {video_id}")

        frames = vu.readVideo(video_fn)
        hand_detections = airplanecorpus.loadHandDetections(
            video_id, dir_name=hand_detections_dir, unflatten=True
        )

        num_frames = frames.shape[0]
        if hand_detections.shape[0] != num_frames:
            raise AssertionError()

        for frame_index in range(1, num_frames):
            cur_detections = hand_detections[frame_index]
            prev_detections = hand_detections[frame_index - 1]

            # Data association
            assignment = associate_detections(prev_detections, cur_detections)
            cur_detections = cur_detections[assignment]

            # Impute missing values
            imputed = prev_detections
            cur_is_nan = np.isnan(cur_detections)
            cur_detections[cur_is_nan] = imputed[cur_is_nan]

            hand_detections[frame_index] = cur_detections

        fn = os.path.join(out_data_dir, f"{video_id}.handsdetections.txt")
        hand_detections = hand_detections.reshape(hand_detections.shape[0], -1)
        np.savetxt(fn, hand_detections, delimiter=',', fmt='%.0f')

        video_fn = os.path.join(fig_dir, f'{video_id}.avi')


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
