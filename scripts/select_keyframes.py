import os

from matplotlib import pyplot as plt
import joblib
import yaml
import scipy

from mathtools import utils
from kinemparse import videoprocessing
from visiontools import imageprocessing


def actionToSegments(action_label_seq):
    is_seg = (action_label_seq == 1) + (action_label_seq == 3)
    segment_labels, num_labels = scipy.ndimage.label(is_seg)
    return segment_labels


def plotScores(frame_scores, keyframe_idxs, fn, segments_seq=None):
    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.plot(frame_scores)
    axis.set_xlabel('Frame index')
    axis.set_ylabel('Frame score')
    axis.scatter(keyframe_idxs, frame_scores[keyframe_idxs])
    axis.set_title('Video frame scores')

    if segments_seq is not None:
        axis = axis.twinx()
        axis.plot(segments_seq, color='tab:orange')

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def main(
        out_dir=None, video_data_dir=None, keyframe_scores_dir=None, segments_dir=None,
        max_seqs=None, use_gt_segments=False, frame_selection_options={}):

    out_dir = os.path.expanduser(out_dir)
    video_data_dir = os.path.expanduser(video_data_dir)
    keyframe_scores_dir = os.path.expanduser(keyframe_scores_dir)
    if segments_dir is not None:
        segments_dir = os.path.expanduser(segments_dir)

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

    trial_ids = utils.getUniqueIds(keyframe_scores_dir, prefix='trial-', suffix='.pkl')

    if max_seqs is not None:
        trial_ids = trial_ids[:max_seqs]

    for seq_idx, trial_id in enumerate(trial_ids):
        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        trial_str = f"trial-{trial_id}"
        rgb_frame_seq = loadFromDir(f'{trial_str}_rgb-frame-seq', video_data_dir)
        frame_scores = loadFromDir(f"{trial_str}_frame-scores", keyframe_scores_dir)

        if segments_dir is not None:
            # fn = f'trial={trial_id}_pred-segment-seq-rgb'
            if use_gt_segments:
                fn = f'trial={trial_id}_true-label-seq'
            else:
                fn = f'trial={trial_id}_pred-label-seq'
            try:
                segments_seq = loadFromDir(fn, segments_dir)
                segments_seq = actionToSegments(segments_seq)
            except FileNotFoundError:
                logger.info(f"  File not found: {fn}")
                continue
        else:
            segments_seq = None

        segment_keyframe_idxs = videoprocessing.selectSegmentKeyframes(
            frame_scores, segment_labels=segments_seq, **frame_selection_options
        )

        logger.info(f"  Saving output...")

        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(frame_scores, segment_keyframe_idxs, fn, segments_seq=segments_seq)

        def saveFrames(indices, label):
            best_rgb = rgb_frame_seq[indices]
            imageprocessing.displayImages(
                *best_rgb, num_rows=1,
                file_path=os.path.join(fig_dir, f'{trial_str}_best-frames-{label}.png')
            )
        saveFrames(segment_keyframe_idxs, 'segment')

        # Save intermediate results
        saveToWorkingDir(segment_keyframe_idxs, f'{trial_str}_keyframe-idxs')


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
