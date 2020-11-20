import os
import argparse

import joblib
import yaml
import numpy as np

from mathtools import utils
from blocks.estimation import videoprocessing, render, imageprocessing


def main(
        out_dir=None, data_dir=None, corpus_name=None,
        start_from=None, stop_at=None,
        display_summary_img=None,
        background_removal_options={}, segment_image_options={}):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromWorkingDir(var_name):
        return joblib.load(os.path.join(data_dir, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    # corpus = duplocorpus.DuploCorpus(corpus_name)

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    camera_pose = render.camera_pose
    camera_params = render.intrinsic_matrix

    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        trial_str = f"trial={trial_id}"

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")
        # task_id = corpus.getTaskIndex(trial_id)
        # goal_state = labels.constructGoalState(task_id)
        goal_state = None

        logger.info(f"  Loading data...")
        rgb_frame_seq = loadFromWorkingDir(f"{trial_str}_rgb-frame-seq")
        depth_frame_seq = loadFromWorkingDir(f"{trial_str}_depth-frame-seq")

        logger.info(f"  Removing background...")
        foreground_mask_seq, background_plane_seq = utils.batchProcess(
            videoprocessing.foregroundPixels,
            depth_frame_seq,
            static_args=(camera_params, camera_pose),
            static_kwargs=background_removal_options,
            unzip=True
        )
        foreground_mask_seq = np.stack(foreground_mask_seq)

        logger.info(f"  Segmenting foreground...")
        segment_frame_seq = utils.batchProcess(
            videoprocessing.segmentImage,
            rgb_frame_seq, depth_frame_seq, foreground_mask_seq,
            static_args=(goal_state,),
            static_kwargs=segment_image_options
        )
        segment_frame_seq = np.stack(segment_frame_seq)

        foreground_mask_seq_no_ref_model = segment_frame_seq != 0

        logger.info(f"  Saving output...")
        saveToWorkingDir(background_plane_seq, f'{trial_str}_background-plane-seq')
        saveToWorkingDir(foreground_mask_seq, f'{trial_str}_foreground-mask-seq')
        saveToWorkingDir(segment_frame_seq, f'{trial_str}_segment-frame-seq')
        saveToWorkingDir(
            foreground_mask_seq_no_ref_model,
            f'{trial_str}_foreground-mask-seq_no-ref-model'
        )

        if display_summary_img:
            if utils.in_ipython_console():
                file_path = None
            else:
                trial_str = f"trial={trial_id}"
                file_path = os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            imageprocessing.displayImages(
                *rgb_frame_seq, *depth_frame_seq, *segment_frame_seq,
                num_rows=3, file_path=file_path
            )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--start_from')
    parser.add_argument('--stop_at')
    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

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
