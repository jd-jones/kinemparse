import os
import logging

import numpy as np
import joblib
import yaml

from mathtools import utils


logger = logging.getLogger(__name__)


def makeActionLabels(action_seq, seq_len=None):
    """
    0: no action
    1: action
    """

    if seq_len is None:
        seq_len = action_seq['end'].max() + 1

    is_tag = action_seq['action'] > 7
    action_seq = action_seq[~is_tag]

    labels = np.zeros(seq_len, dtype=int)

    for action in action_seq:
        start = action['start']
        end = action['end']
        labels[start:end + 1] = 1

    return labels


def main(
        out_dir=None, video_data_dir=None, features_dir=None,
        activity_labels_dir=None, gt_keyframes_dir=None):
    out_dir = os.path.expanduser(out_dir)
    video_data_dir = os.path.expanduser(video_data_dir)
    features_dir = os.path.expanduser(features_dir)
    activity_labels_dir = os.path.expanduser(activity_labels_dir)
    if gt_keyframes_dir is not None:
        gt_keyframes_dir = os.path.expanduser(gt_keyframes_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = utils.getUniqueIds(features_dir, prefix='trial-', suffix='.pkl')

    for seq_idx, trial_id in enumerate(trial_ids):
        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        feature_seq = loadFromDir(f"trial={trial_id}_feature-seq", features_dir)
        raw_labels = loadFromDir(f"trial-{trial_id}_action-seq", video_data_dir)
        timestamp_seq = loadFromDir(f"trial-{trial_id}_rgb-frame-timestamp-seq", video_data_dir)

        action_labels = makeActionLabels(raw_labels, seq_len=feature_seq.shape[0])

        if timestamp_seq.shape != feature_seq.shape:
            logger.warning(
                f"Video dimensions don't match: "
                f"{feature_seq.shape} scores, {timestamp_seq.shape} timestamps"
            )
            continue

        logger.info(f"  Saving output...")
        trial_str = f"trial={trial_id}"
        saveToWorkingDir(feature_seq, f'{trial_str}_timestamp-seq')
        saveToWorkingDir(feature_seq, f'{trial_str}_feature-seq')
        saveToWorkingDir(action_labels, f'{trial_str}_label-seq')


if __name__ == '__main__':
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
