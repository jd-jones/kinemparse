import os
import argparse
import glob

from matplotlib import pyplot as plt
import numpy as np
import joblib
import yaml

from blocks.core import utils, labels, duplocorpus
from blocks.estimation import imageprocessing, videoprocessing, models, rawdata


def monkeyPatchKeyframeModel(keyframe_model):
    # color_names = ('red', 'blue', 'green', 'yellow')
    colors = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0.75, 0]
    ])

    keyframe_model.is_noise_cluster[[0, 2, 4, 5, 9, 19, 21, 23]] = True
    keyframe_model.class_histograms[[0, 2, 4, 5, 9, 19, 21, 23], 0] = 1
    keyframe_model.class_histograms[[0, 2, 4, 5, 9, 19, 21, 23], 1] = 0
    keyframe_model.is_noise_cluster[8] = False
    keyframe_model = models.assignModelClusters(keyframe_model, colors=colors)

    return keyframe_model


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


def getUniqueTrialIds(dir_path):
    trial_ids = set(
        int(os.path.basename(fn).split('-')[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"trial-*.pkl"))
    )
    return sorted(tuple(trial_ids))


def main(
        out_dir=None, data_dir=None, preprocess_dir=None, keyframe_model_fn=None,
        corpus_name=None, max_seqs=None, subsample_period=None,
        frame_scoring_options={}, frame_selection_options={}):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    preprocess_dir = os.path.expanduser(preprocess_dir)
    keyframe_model_fn = os.path.expanduser(keyframe_model_fn)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    def loadFromDataDir(var_name):
        return joblib.load(os.path.join(data_dir, f"{var_name}.pkl"))

    def loadFromPreprocessDir(var_name):
        return joblib.load(os.path.join(preprocess_dir, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = getUniqueTrialIds(preprocess_dir)
    # keyframe_model = monkeyPatchKeyframeModel(joblib.load(keyframe_model_fn))
    keyframe_model = joblib.load(keyframe_model_fn)
    models.visualizeKeyframeModel(keyframe_model, fn=os.path.join(fig_dir, 'keyframe-model.png'))

    if max_seqs is not None:
        trial_ids = trial_ids[:max_seqs]

    corpus = duplocorpus.DuploCorpus(corpus_name)
    for seq_idx, trial_id in enumerate(trial_ids):

        logger.info(f"Processing video {seq_idx} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"Loading data...")
        trial_str = f"trial-{trial_id}"
        rgb_frame_seq = loadFromDataDir(f'{trial_str}_rgb-frame-seq')
        segment_seq = loadFromPreprocessDir(f'{trial_str}_segment-frame-seq')

        logger.info(f"Scoring frames...")
        frame_scores = videoprocessing.scoreFrames(
            keyframe_model,
            rgb_frame_seq, segment_seq,
            score_kwargs=frame_scoring_options
        )

        segment_keyframe_idxs = videoprocessing.selectSegmentKeyframes(
            frame_scores, **frame_selection_options
        )

        logger.info(f"Saving output...")

        fn = os.path.join(fig_dir, f'{trial_str}_scores-plot.png')
        plotScores(frame_scores, segment_keyframe_idxs, fn)

        def saveFrames(indices, label):
            best_rgb = rgb_frame_seq[indices]
            best_seg = segment_seq[indices]
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

        if corpus_name == 'child':
            annotator = 'Jonathan'
            gt_keyframe_idxs = labels.loadKeyframeIndices(corpus, annotator, trial_id)
            rgb_frame_fns = corpus.getRgbFrameFns(trial_id)
            rgb_timestamps = corpus.readRgbTimestamps(trial_id, times_only=True)
            rgb_frames = rawdata.loadRgbFrameSeq(rgb_frame_fns, rgb_timestamps)
            rgb_keyframes = tuple(rgb_frames[i] for i in gt_keyframe_idxs)
            imageprocessing.displayImages(
                *rgb_keyframes,
                file_path=os.path.join(out_dir, f'{trial_str}_best-frames-gt.png')
            )

        # Save intermediate results
        saveToWorkingDir(frame_scores, f'{trial_str}_frame-scores')
        saveToWorkingDir(segment_keyframe_idxs, f'{trial_str}_keyframe-idxs')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--preprocess_dir')
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
