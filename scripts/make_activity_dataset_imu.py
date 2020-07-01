import os
import logging

import yaml
import joblib
import numpy as np
from matplotlib import pyplot as plt

from mathtools import utils


logger = logging.getLogger(__name__)


def activityFeatures(feature_seq):
    activity_features = feature_seq[..., 2].swapaxes(0, 1)
    return activity_features


def plotScores(scores, keyframe_idxs=None, action_labels=None, fn=None):
    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.set_title('IMU activity scores')
    axis.set_xlabel('Sample index')
    axis.set_ylabel('Activity score')

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


def main(
        out_dir=None, predictions_dir=None, imu_data_dir=None, video_data_dir=None,
        use_gt_segments=None, model_name=None, model_params={},
        results_file=None, sweep_param_name=None,
        cv_params={}, viz_params={},
        plot_predictions=None):

    predictions_dir = os.path.expanduser(predictions_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, f'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(predictions_dir, prefix='trial=', suffix='.pkl')

    for seq_idx, trial_id in enumerate(trial_ids):
        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        feature_seq = loadFromDir(f"trial={trial_id}_score-seq", predictions_dir)
        label_seq = loadFromDir(f"trial={trial_id}_true-label-seq", predictions_dir)

        feature_seq = feature_seq[..., 2].swapaxes(0, 1)
        label_seq = (label_seq.swapaxes(0, 1) == 2).any(axis=1).astype(int)

        trial_str = f"trial={trial_id}"
        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(feature_seq.max(axis=1), action_labels=label_seq, fn=fn)

        # all_score_seqs.append(score_seq)
        # all_action_labels.append(label_seq)
        feature_seq -= feature_seq.mean()

        saveVariable(feature_seq, f'{trial_str}_feature-seq')
        saveVariable(label_seq, f'{trial_str}_label-seq')


if __name__ == "__main__":
    # Parse command line arguments
    args = utils.parse_args(main)
    config, config_fn = utils.parse_config(args, script_name=__file__)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
