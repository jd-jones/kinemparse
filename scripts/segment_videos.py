import os
import argparse
import inspect

import numpy as np
from matplotlib import pyplot as plt
import joblib
import yaml

from mathtools import utils


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


def plotScores(scores, keyframe_idxs=None, action_labels=None, fn=None):
    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.set_title('Video frame scores')
    axis.set_xlabel('Frame index')
    axis.set_ylabel('Frame score')

    axis.axhline(np.nanmean(scores), color='k')
    axis.plot(scores)

    if keyframe_idxs is not None:
        axis.scatter(keyframe_idxs, scores[keyframe_idxs])

    if action_labels is not None:
        axis = axis.twinx()
        axis.plot(action_labels, color='tab:red')

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


def main(out_dir=None, data_dir=None, video_seg_scores_dir=None, imu_seg_scores_dir=None):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    video_seg_scores_dir = os.path.expanduser(video_seg_scores_dir)
    if imu_seg_scores_dir is not None:
        imu_seg_scores_dir = os.path.expanduser(imu_seg_scores_dir)

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
        action_labels = expandLabels(
            loadFromDir(f"trial-{trial_id}_action-seq", data_dir),
            seq_len=score_seq.shape[0]
        )

        if imu_seg_scores_dir is not None:
            fn = f'trial={trial_id}_pred-segment-seq-rgb'
            try:
                score_seq = loadFromDir(fn, imu_seg_scores_dir)  # FIXME
            except FileNotFoundError:
                logger.info(f"  File not found: {fn}")
                continue

        logger.info(f"  Saving output...")

        trial_str = f"trial={trial_id}"
        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(score_seq, action_labels=action_labels, fn=fn)

        all_score_seqs.append(score_seq)
        all_action_labels.append(action_labels)

        # Save intermediate results
        score_seq -= np.nanmean(score_seq)
        score_is_nan = np.isnan(score_seq)
        score_seq[score_is_nan] = 0
        feature_seq = np.column_stack((score_seq, score_is_nan.astype(float) - 0.5))
        saveToWorkingDir(feature_seq, f'{trial_str}_feature-seq')
        saveToWorkingDir(action_labels, f'{trial_str}_label-seq')

    all_score_seqs = np.hstack(tuple(all_score_seqs))
    all_action_labels = np.hstack(tuple(all_action_labels))
    fn = os.path.join(fig_dir, f'score-hists.png')
    plotScoreHists(all_score_seqs, all_action_labels, fn=fn)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    for arg_name in inspect.getfullargspec(main).args:
        parser.add_argument(f'--{arg_name}')

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.expanduser(
            os.path.join(
                '~', 'repo', 'blocks', 'blocks', 'estimation', 'scripts', 'config',
                config_fn
            )
        )
    else:
        config_fn = os.path.basename(config_file_path)
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    if config is None:
        config = {}

    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
