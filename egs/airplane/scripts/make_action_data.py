import os
import logging
import glob

import yaml
import joblib
import numpy as np

from mathtools import utils
from kinemparse import airplanecorpus


logger = logging.getLogger(__name__)


def makeBinLabels(action_labels, part_idxs_to_bins, num_samples):
    no_bin = part_idxs_to_bins[0]  # 0 is the index of the null part
    bin_labels = np.full(num_samples, no_bin, dtype=int)

    for part_index, start_index, end_index in action_labels:
        bin_index = part_idxs_to_bins[part_index]
        bin_labels[start_index:end_index + 1] = bin_index

    return bin_labels


def main(
        out_dir=None, hand_detections_dir=None, labels_dir=None,
        plot_output=None, results_file=None, sweep_param_name=None):

    hand_detections_dir = os.path.expanduser(hand_detections_dir)
    labels_dir = os.path.expanduser(labels_dir)

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

    logger.info(f"Reading from: {hand_detections_dir}")
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

    for i, fn in enumerate(glob.glob(os.path.join(hand_detections_dir, '*.txt'))):
        video_id = utils.stripExtension(fn).split('.handsdetections')[0]

        logger.info(f"Processing video {video_id}")

        hand_detections = airplanecorpus.loadHandDetections(
            video_id, dir_name=hand_detections_dir, unflatten=True
        )
        hand_detections = hand_detections.reshape(hand_detections.shape[0], -1)
        mean_detection = np.nanmean(hand_detections)
        hand_detections[np.isnan(hand_detections)] = mean_detection

        action_labels = airplanecorpus.loadLabels(
            video_id, dir_name=labels_dir,
            part_names_to_idxs=part_names_to_idxs
        )

        bin_labels = makeBinLabels(action_labels, part_idxs_to_bins, hand_detections.shape[0])

        fig_fn = os.path.join(fig_dir, f"{video_id}.png")
        utils.plot_array(hand_detections.T, (bin_labels,), ('bin',), fn=fig_fn)

        video_id = video_id.replace('_', '-')
        saveVariable(hand_detections, f'trial={video_id}_feature-seq')
        saveVariable(bin_labels, f'trial={video_id}_label-seq')


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
