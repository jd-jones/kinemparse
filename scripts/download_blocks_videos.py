import os
import argparse

import joblib
import yaml

from blocks.core import duplocorpus, labels
from blocks.estimation import rawdata
from mathtools import utils


def fixStartEndIndices(action_seq, rgb_frame_timestamp_seq, selected_frame_indices):
    new_times = rgb_frame_timestamp_seq[selected_frame_indices]
    start_times = rgb_frame_timestamp_seq[action_seq['start']]
    end_times = rgb_frame_timestamp_seq[action_seq['end']]

    action_seq['start'] = utils.nearestIndices(new_times, start_times)
    action_seq['end'] = utils.nearestIndices(new_times, end_times)

    return action_seq


def main(
        out_dir=None, corpus_name=None, default_annotator=None,
        start_from=None, stop_at=None,
        start_video_from_first_touch=None, load_video_frames=True,
        use_annotated_keyframes=None, subsample_period=None):

    out_dir = os.path.expanduser(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    selected_frame_indices = slice(None, None, subsample_period)

    trial_ids = utils.loadVariable('trial_ids', 'preprocess-all-data', corpus_name)

    corpus = duplocorpus.DuploCorpus(corpus_name)

    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        if start_video_from_first_touch:
            label_seq = corpus.readLabels(trial_id, default_annotator)[0]
            first_touch_idxs = label_seq['start'][label_seq['action'] == 7]
            if not len(first_touch_idxs):
                logger.info(f"  Skipping trial {trial_id}: no first touch annotated")
                continue
            first_touch_idx = int(first_touch_idxs[0])
            # Video is sampled at 30 FPS --> start one second before the first
            # touch was annotated (unless that frame would have happened before
            # the camera started recording)
            start_idx = max(0, first_touch_idx - 30)
            selected_frame_indices = slice(start_idx, None, subsample_period)

        if use_annotated_keyframes:
            annotator = 'Jonathan'
            keyframe_idxs = labels.loadKeyframeIndices(corpus, annotator, trial_id)
            if not keyframe_idxs:
                logger.info(f"  Skipping trial {trial_id}: no labeled keyframes")
                continue
            selected_frame_indices = keyframe_idxs

        logger.info(f"  Loading data...")
        rgb_frame_fn_seq = corpus.getRgbFrameFns(trial_id)
        rgb_frame_timestamp_seq = corpus.readRgbTimestamps(trial_id, times_only=True)

        depth_frame_fn_seq = corpus.getDepthFrameFns(trial_id)
        depth_frame_timestamp_seq = corpus.readDepthTimestamps(trial_id, times_only=True)

        action_seq, _ = corpus.readLabels(trial_id, default_annotator)

        action_seq = fixStartEndIndices(
            action_seq, rgb_frame_timestamp_seq, selected_frame_indices
        )
        rgb_frame_timestamp_seq = rgb_frame_timestamp_seq[selected_frame_indices]
        depth_frame_timestamp_seq = depth_frame_timestamp_seq[selected_frame_indices]

        logger.info(f"  Saving output...")
        trial_str = f"trial={trial_id}"
        saveToWorkingDir(action_seq, f'{trial_str}_action-seq')
        # saveToWorkingDir(rgb_frame_fn_seq, f'{trial_str}_rgb-frame-fn-seq')
        # saveToWorkingDir(rgb_frame_timestamp_seq, f'{trial_str}_rgb-frame-timestamp-seq')
        # saveToWorkingDir(depth_frame_fn_seq, f'{trial_str}_depth-frame-fn-seq')
        # saveToWorkingDir(depth_frame_timestamp_seq, f'{trial_str}_depth-frame-timestamp-seq')

        if load_video_frames:
            logger.info(f"  Loading and saving video data...")
            rgb_frame_seq = rawdata.loadRgbFrameSeq(
                rgb_frame_fn_seq, rgb_frame_timestamp_seq,
                stack_frames=True
            )[selected_frame_indices]
            depth_frame_seq = rawdata.loadDepthFrameSeq(
                depth_frame_fn_seq, depth_frame_timestamp_seq,
                stack_frames=True
            )[selected_frame_indices]
            saveToWorkingDir(rgb_frame_seq, f'{trial_str}_rgb-frame-seq')
            saveToWorkingDir(depth_frame_seq, f'{trial_str}_depth-frame-seq')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
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
    config.update(args)

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
