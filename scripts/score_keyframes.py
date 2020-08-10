import os

from matplotlib import pyplot as plt
import numpy as np
import joblib
import yaml

from mathtools import utils
# FIXME: remove dependency on blocks
from blocks.estimation import models
from kinemparse import videoprocessing
from visiontools import imageprocessing


def plotScores(frame_scores, keyframe_idxs, fn):
    _, axis = plt.subplots(1, figsize=(12, 8))
    axis.plot(frame_scores)
    axis.set_xlabel('Frame index')
    axis.set_ylabel('Frame score')
    axis.scatter(keyframe_idxs, frame_scores[keyframe_idxs])
    axis.set_title('Video frame scores')
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def main(
        out_dir=None, data_dir=None, preprocess_dir=None, segments_dir=None,
        keyframe_model_fn=None, max_seqs=None, subsample_period=None,
        frame_scoring_options={}, frame_selection_options={}):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    preprocess_dir = os.path.expanduser(preprocess_dir)
    keyframe_model_fn = os.path.expanduser(keyframe_model_fn)
    if segments_dir is not None:
        segments_dir = os.path.expanduser(segments_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def loadFromPreprocessDir(var_name):
        return joblib.load(os.path.join(preprocess_dir, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = utils.getUniqueIds(preprocess_dir, prefix='trial-', suffix='.pkl')
    keyframe_model = joblib.load(keyframe_model_fn)
    models.visualizeKeyframeModel(keyframe_model, fn=os.path.join(fig_dir, 'keyframe-model.png'))

    if max_seqs is not None:
        trial_ids = trial_ids[:max_seqs]

    for seq_idx, trial_id in enumerate(trial_ids):
        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        trial_str = f"trial-{trial_id}"
        rgb_frame_seq = loadFromDir(f'{trial_str}_rgb-frame-seq', data_dir)
        segment_frame_seq = loadFromDir(f'{trial_str}_segment-frame-seq', preprocess_dir)

        if segments_dir is not None:
            fn = f'trial={trial_id}_pred-segment-seq-rgb'
            try:
                segments_seq = loadFromDir(fn, segments_dir)
            except FileNotFoundError:
                logger.info(f"  File not found: {fn}")
                continue
        else:
            segments_seq = None

        logger.info(f"  Scoring frames...")
        frame_scores = videoprocessing.scoreFrames(
            keyframe_model,
            rgb_frame_seq, segment_frame_seq,
            score_kwargs=frame_scoring_options
        )

        segment_keyframe_idxs = videoprocessing.selectSegmentKeyframes(
            frame_scores, segment_labels=segments_seq, **frame_selection_options
        )

        logger.info(f"  Saving output...")

        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(frame_scores, segment_keyframe_idxs, fn)

        def saveFrames(indices, label):
            best_rgb = rgb_frame_seq[indices]
            best_seg = segment_frame_seq[indices]
            rgb_quantized = np.stack(
                tuple(
                    videoprocessing.quantizeImage(keyframe_model, rgb_img, segment_img)
                    for rgb_img, segment_img in zip(best_rgb, best_seg)
                )
            )
            imageprocessing.displayImages(
                *best_rgb, *best_seg, *rgb_quantized, num_rows=3,
                file_path=os.path.join(fig_dir, f'{trial_str}_best-frames-{label}.png')
            )

        saveFrames(segment_keyframe_idxs, 'segment')

        # Save intermediate results
        saveToWorkingDir(frame_scores, f'{trial_str}_frame-scores')
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
