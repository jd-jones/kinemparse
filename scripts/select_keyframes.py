import os

from matplotlib import pyplot as plt
import joblib
import yaml
import scipy
import numpy as np

from mathtools import utils
from kinemparse import videoprocessing
from visiontools import imageprocessing


def makeSegments(action_label_seq, activity_label_seq):
    is_active = activity_label_seq == 1
    is_active_seg = action_label_seq == 1

    is_seg = np.zeros(activity_label_seq.shape[0], dtype=bool)
    is_seg[is_active] = is_active_seg

    segment_labels, num_labels = scipy.ndimage.label(is_seg)
    return segment_labels


def evalKeyframes(pred_keyframe_idxs, gt_keyframe_idxs):
    cost_matrix = np.array([
        [np.abs(i - j) for i in pred_keyframe_idxs]
        for j in gt_keyframe_idxs
    ])
    assigned_gt_idxs, assigned_pred_idxs = scipy.optimize.linear_sum_assignment(cost_matrix)

    best_costs = cost_matrix[assigned_gt_idxs, assigned_pred_idxs]
    false_alarms = [i for i, __ in enumerate(pred_keyframe_idxs) if i not in assigned_pred_idxs]
    misses = [i for i, __ in enumerate(gt_keyframe_idxs) if i not in assigned_gt_idxs]

    return best_costs, false_alarms, misses


def plotKeyframeMetrics(best_costs, fn):
    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.set_xlabel('Distance (in frames)')
    axis.set_ylabel('Count')

    axis.hist(best_costs, bins=50)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def plotScores(frame_scores, keyframe_idxs, fn, segments_seq=None, gt_keyframe_idxs=None):
    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.set_title('Video frame scores')
    axis.set_xlabel('Frame index')
    axis.set_ylabel('Frame score')

    axis.plot(frame_scores, color='k')
    axis.scatter(keyframe_idxs, frame_scores[keyframe_idxs], label='pred kfs')
    if gt_keyframe_idxs is not None:
        axis.scatter(gt_keyframe_idxs, frame_scores[gt_keyframe_idxs], label='gt kfs')
    axis.legend()

    if segments_seq is not None:
        axis = axis.twinx()
        axis.plot((segments_seq > 0).astype(int), color='tab:green')

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def main(
        out_dir=None, video_data_dir=None, keyframe_scores_dir=None,
        activity_labels_dir=None, action_labels_dir=None,
        max_seqs=None, use_gt_activity=False, use_gt_actions=False, frame_selection_options={}):

    out_dir = os.path.expanduser(out_dir)
    video_data_dir = os.path.expanduser(video_data_dir)
    keyframe_scores_dir = os.path.expanduser(keyframe_scores_dir)
    activity_labels_dir = os.path.expanduser(activity_labels_dir)
    action_labels_dir = os.path.expanduser(action_labels_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    out_video_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_video_data_dir):
        os.makedirs(out_video_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_video_data_dir, f"{var_name}.pkl"))

    trial_ids = utils.getUniqueIds(activity_labels_dir, prefix='trial=', suffix='.pkl')

    if max_seqs is not None:
        trial_ids = trial_ids[:max_seqs]

    all_costs = []
    all_precisions = []
    all_recalls = []
    for seq_idx, trial_id in enumerate(trial_ids):
        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        trial_str = f"trial-{trial_id}"
        rgb_frame_seq = loadFromDir(f'{trial_str}_rgb-frame-seq', video_data_dir)
        frame_scores = loadFromDir(f"{trial_str}_frame-scores", keyframe_scores_dir)

        if use_gt_activity:
            fn = f'trial={trial_id}_true-label-seq'
        else:
            fn = f'trial={trial_id}_pred-label-seq'
        activity_label_seq = loadFromDir(fn, activity_labels_dir)

        if use_gt_actions:
            fn = f'trial={trial_id}_true-label-seq'
        else:
            fn = f'trial={trial_id}_pred-label-seq'
        action_label_seq = loadFromDir(fn, action_labels_dir)

        segments_seq = makeSegments(action_label_seq, activity_label_seq)

        pred_keyframe_idxs = videoprocessing.selectSegmentKeyframes(
            frame_scores, segment_labels=segments_seq, **frame_selection_options
        )

        # Measure performance
        try:
            fn = f'trial-{trial_id}_gt-keyframe-seq'
            gt_keyframe_idxs = loadFromDir(fn, video_data_dir)
        except FileNotFoundError:
            gt_keyframe_idxs = None

        # Save and visualize output
        logger.info(f"  Saving output...")
        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(
            frame_scores, pred_keyframe_idxs, fn,
            segments_seq=segments_seq, gt_keyframe_idxs=gt_keyframe_idxs
        )

        def saveFrames(indices, label):
            best_rgb = rgb_frame_seq[indices]
            imageprocessing.displayImages(
                *best_rgb, num_rows=1,
                file_path=os.path.join(fig_dir, f'{trial_str}_best-frames-{label}.png')
            )
        saveFrames(pred_keyframe_idxs, 'pred')
        if gt_keyframe_idxs is not None:
            saveFrames(gt_keyframe_idxs, 'gt')
            best_costs, false_alarms, misses = evalKeyframes(pred_keyframe_idxs, gt_keyframe_idxs)
            num_fps = len(false_alarms)
            num_fns = len(misses)
            num_tps = best_costs.shape[0]
            precision = num_tps / (num_tps + num_fps)
            recall = num_tps / (num_tps + num_fns)

            logger.info(f"  PRC: {precision * 100:03.1f}%")
            logger.info(f"  REC: {recall * 100:03.1f}%")

            all_costs.append(best_costs)
            all_precisions.append(precision)
            all_recalls.append(recall)

        # Save intermediate results
        saveToWorkingDir(pred_keyframe_idxs, f'{trial_str}_keyframe-idxs')

    avg_precision = np.array(all_precisions).mean()
    avg_recall = np.array(all_recalls).mean()
    logger.info(f"AVG PRC: {avg_precision * 100:03.1f}%  ({len(all_precisions)} seqs)")
    logger.info(f"AVG REC: {avg_recall * 100:03.1f}%  ({len(all_recalls)} seqs)")

    all_costs = np.hstack(all_costs)
    fn = os.path.join(fig_dir, "keyframe-hist.png")
    plotKeyframeMetrics(all_costs, fn)


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
