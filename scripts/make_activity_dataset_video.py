import os
import logging

import numpy as np
from matplotlib import pyplot as plt
import joblib
import yaml

from mathtools import utils


logger = logging.getLogger(__name__)


def expandLabels(action_seq, seq_len=None):
    """
    0: beginning of sequence
    1: no action
    2: action
    3: end of sequence
    """

    if seq_len is None:
        seq_len = action_seq['end'].max() + 1

    labels = np.ones(seq_len, dtype=int)
    first_start = action_seq[0]['start']
    labels[:first_start] = 0
    for action in action_seq:
        start = action['start']
        end = action['end']
        label = action['action']
        if label > 7:
            continue
        # seg_len = end - start
        # start = end - seg_len // 2
        labels[start:end + 1] = 2
    last_end = action_seq[-1]['end']
    labels[last_end + 1:] = 3

    return labels


def plotScores(
        timestamp_seq, score_seq,
        imu_timestamp_seq=None, imu_score_seq=None,
        keyframe_idxs=None, action_labels=None, raw_labels=None, fn=None):

    score_seqs = [score_seq]
    timestamp_seqs = [timestamp_seq]

    if imu_score_seq is not None:
        score_seqs.append(imu_score_seq)
        timestamp_seqs.append(imu_timestamp_seq)

    num_subplots = len(score_seqs)

    _, axes = plt.subplots(num_subplots, figsize=(12, 5 * num_subplots), sharex=True)
    if num_subplots == 1:
        axes = [axes]

    for i, axis in enumerate(axes):
        score_seq = score_seqs[i]
        timestamp_seq = timestamp_seqs[i]

        axis.set_title('Video frame scores')
        axis.set_xlabel('Frame index')
        axis.set_ylabel('Frame score')

        axis.axhline(np.nanmean(score_seq), color='k')
        axis.plot(timestamp_seq, score_seq)

        score_seq[np.isnan(score_seq)] = np.nanmean(score_seq)

        if action_labels is not None:
            axis = axis.twinx()
            axis.plot(timestamp_seqs[0], action_labels, color='tab:green')

    if raw_labels is not None:
        is_first_touch_label = raw_labels['action'] == 7
        first_touch_idxs = raw_labels['start'][is_first_touch_label]
        first_touch_times = timestamp_seqs[0][first_touch_idxs]
        first_touch_scores = score_seqs[0][first_touch_idxs]
        axes[0].scatter(first_touch_times, first_touch_scores, color='tab:red')

    if keyframe_idxs is not None:
        keyframe_times = timestamp_seqs[0][keyframe_idxs]
        keyframe_scores = score_seqs[0][keyframe_idxs]
        axes[0].scatter(keyframe_times, keyframe_scores, color='tab:orange')

    plt.tight_layout()

    if fn is None:
        pass
    else:
        plt.savefig(fn)
        plt.close()


def plotScoreHists(scores, labels, fn=None):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    f, axes = plt.subplots(num_labels, figsize=(12, 3 * num_labels))

    for i, label in enumerate(unique_labels):
        matches_label = labels == label
        matching_scores = scores[matches_label]
        axes[i].hist(matching_scores[~np.isnan(matching_scores)], bins=50)
        axes[i].set_xlabel(f'scores, label={label}')
        axes[i].set_ylabel('counts')

    plt.tight_layout()

    if fn is None:
        pass
    else:
        plt.savefig(fn)
        plt.close()


def main(
        out_dir=None, video_data_dir=None, imu_data_dir=None,
        video_seg_scores_dir=None, imu_seg_scores_dir=None, gt_keyframes_dir=None):

    out_dir = os.path.expanduser(out_dir)
    video_data_dir = os.path.expanduser(video_data_dir)
    imu_data_dir = os.path.expanduser(imu_data_dir)
    video_seg_scores_dir = os.path.expanduser(video_seg_scores_dir)
    if imu_seg_scores_dir is not None:
        imu_seg_scores_dir = os.path.expanduser(imu_seg_scores_dir)
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

    trial_ids = utils.getUniqueIds(video_seg_scores_dir, prefix='trial-', suffix='.pkl')

    all_score_seqs = []
    all_action_labels = []
    for seq_idx, trial_id in enumerate(trial_ids):
        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        score_seq = loadFromDir(f"trial-{trial_id}_frame-scores", video_seg_scores_dir)
        raw_labels = loadFromDir(f"trial-{trial_id}_action-seq", video_data_dir)
        action_labels = expandLabels(raw_labels, seq_len=score_seq.shape[0])
        timestamp_seq = loadFromDir(f"trial-{trial_id}_rgb-frame-timestamp-seq", video_data_dir)

        if timestamp_seq.shape != score_seq.shape:
            logger.warning(
                f"Video dimensions don't match: "
                f"{score_seq.shape} scores, {timestamp_seq.shape} timestamps"
            )
            continue

        if imu_seg_scores_dir is not None:
            try:
                imu_score_seq = loadFromDir(
                    f'trial={trial_id}_score-seq',
                    imu_seg_scores_dir
                )
                imu_score_seq = imu_score_seq[..., 2].swapaxes(0, 1).max(axis=1)
            except FileNotFoundError:
                logger.info(f"  IMU scores not found: trial {trial_id}")
                continue
            imu_timestamp_seq = loadFromDir(f"trial={trial_id}_timestamp-seq", imu_data_dir)
            if imu_timestamp_seq.shape != imu_score_seq.shape:
                logger.warning(
                    f"IMU dimensions don't match: "
                    f"{imu_score_seq.shape} scores, {imu_timestamp_seq.shape} timestamps"
                )
                continue
            # Downsample imu scores to match rgb scores
            imu_score_seq = utils.resampleSeq(imu_score_seq, imu_timestamp_seq, timestamp_seq)
            imu_timestamp_seq = timestamp_seq

        else:
            imu_score_seq = None
            imu_timestamp_seq = None

        logger.info(f"  Saving output...")

        gt_keyframe_fn = os.path.join(gt_keyframes_dir, f"trial-{trial_id}_gt-keyframe-seq.pkl")
        if os.path.exists(gt_keyframe_fn):
            gt_keyframes = joblib.load(gt_keyframe_fn)
        else:
            gt_keyframes = None

        trial_str = f"trial={trial_id}"
        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(
            timestamp_seq, score_seq,
            action_labels=action_labels, raw_labels=raw_labels,
            imu_timestamp_seq=imu_timestamp_seq, imu_score_seq=imu_score_seq,
            keyframe_idxs=gt_keyframes, fn=fn
        )

        all_score_seqs.append(score_seq)
        all_action_labels.append(action_labels)

        # Save intermediate results
        score_seq -= np.nanmean(score_seq)
        score_is_nan = np.isnan(score_seq)
        score_seq[score_is_nan] = 0

        features = (score_seq, score_is_nan.astype(float))
        if imu_score_seq is not None:
            features += (imu_score_seq,)
        feature_seq = np.column_stack(features)

        saveToWorkingDir(feature_seq, f'{trial_str}_feature-seq')
        saveToWorkingDir(action_labels, f'{trial_str}_label-seq')

    all_score_seqs = np.hstack(tuple(all_score_seqs))
    all_action_labels = np.hstack(tuple(all_action_labels))
    fn = os.path.join(fig_dir, f'score-hists.png')
    plotScoreHists(all_score_seqs, all_action_labels, fn=fn)


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
