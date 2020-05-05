import argparse
import os
import glob
import warnings

import yaml
import joblib
import numpy as np
from skimage import img_as_float, img_as_ubyte
import imageio

from mathtools import utils
from visiontools import imageprocessing


def getUniqueTrialIds(dir_path):
    trial_ids = set(
        int(os.path.basename(fn).split('-')[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"trial-*.pkl"))
    )
    return sorted(tuple(trial_ids))


def makeBasketLabels(action_label_seq, num_rgb_frames):
    placement_actions = action_label_seq[action_label_seq['action'] < 2]

    num_blocks = len(definitions.blocks)
    block_is_in_basket = np.ones((num_rgb_frames, num_blocks), dtype=int)
    for block_id in range(num_blocks):
        is_object = placement_actions['object'] == block_id
        is_target = placement_actions['target'] == block_id
        block_actions = placement_actions[is_object | is_target]
        if len(block_actions) == 0:
            continue
        first_placement_idx = block_actions['start'][0]

        block_is_in_basket[first_placement_idx:, block_id] = 0

    return block_is_in_basket


def makeDataLabels(study_name, cached_frames_fn=None):
    corpus = duplocorpus.DuploCorpus(study_name)
    trial_ids = utils.loadVariable('trial_ids', 'preprocess-all-data', study_name)

    annotations = tuple(corpus.readLabels(trial_id, 'Jonathan')[0] for trial_id in trial_ids)

    if cached_frames_fn is not None and os.path.exists(cached_frames_fn):
        rgb_frame_seqs = joblib.load(cached_frames_fn)
    else:
        rgb_frame_seqs = tuple(
            rawdata.loadRgbFrameSeq(corpus.getRgbFrameFns(trial_id))
            for trial_id in trial_ids
        )
        if cached_frames_fn is not None:
            joblib.dump(rgb_frame_seqs, cached_frames_fn)

    labels = tuple(
        makeBasketLabels(label_seq, len(rgb_frame_seq))
        for label_seq, rgb_frame_seq in zip(annotations, rgb_frame_seqs)
    )

    labels = np.vstack(labels)

    data = tuple(itertools.chain.from_iterable(rgb_frame_seqs))
    data = np.moveaxis(np.stack(data, axis=3), [0, 1, 2, 3], [2, 3, 1, 0])

    logger.info(f"data shape: {data.shape}")

    return data, labels


def main(
        out_dir=None, data_dir=None, preprocess_dir=None, classifier_fn=None,
        display_summary_img=None, write_video=None, start_from=None, stop_after=None):

    if start_from is None:
        start_from = 0

    if stop_after is None:
        stop_after = float("Inf")

    data_dir = os.path.expanduser(data_dir)
    preprocess_dir = os.path.expanduser(preprocess_dir)
    out_dir = os.path.expanduser(out_dir)
    classifier_fn = os.path.expanduser(classifier_fn)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromDataDir(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def loadFromPreprocessDir(var_name):
        return joblib.load(os.path.join(preprocess_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    classifier = joblib.load(classifier_fn)

    trial_ids = getUniqueTrialIds(preprocess_dir)
    for i, trial_id in enumerate(trial_ids):

        if i < start_from:
            continue

        if i > stop_after:
            break

        trial_str = f"trial-{trial_id}"

        logger.info(f"Processing video {i + 1} / {len(trial_ids)}  (trial {trial_id})")
        rgb_frame_seq = loadFromDataDir(f"{trial_str}_rgb-frame-seq")
        # depth_frame_seq = loadFromDataDir(f"{trial_str}_depth-frame-seq")
        # foreground_mask_seq = loadFromPreprocessDir(f'{trial_str}_foreground-mask-seq')
        segment_frame_seq = loadFromPreprocessDir(f'{trial_str}_segment-frame-seq')
        # block_segment_frame_seq = loadFromDetectionsDir(f'{trial_str}_block-segment-frame-seq')
        # skin_segment_frame_seq = loadFromDetectionsDir(f'{trial_str}_skin-segment-frame-seq')
        # color_label_frame_seq = loadFromDetectionsDir(f'{trial_str}_color-label-frame-seq')
        # class_label_frame_seq = loadFromDetectionsDir(f'{trial_str}_class-label-frame-seq')

        segment_features_seq, feature_frame_seq = utils.batchProcess(
            extractSegmentFeatures,
            rgb_frame_seq, segment_frame_seq,
            static_args=(classifier,),
            unzip=True
        )

        saveVariable(segment_features_seq, f'{trial_str}_segment-features-seq')

        if display_summary_img:
            if utils.in_ipython_console():
                file_path = None
            else:
                trial_str = f"trial-{trial_id}"
                file_path = os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            imageprocessing.displayImages(
                *rgb_frame_seq, *feature_frame_seq,
                num_rows=2, file_path=file_path
            )

        if write_video:
            video_dir = os.path.join(out_dir, 'detection-videos')
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            fn = os.path.join(video_dir, f"{trial_str}.gif")
            writer = imageio.get_writer(fn, mode='I')
            for rgb_frame, feature_frame in zip(rgb_frame_seq, feature_frame_seq):
                feature_frame = feature_frame.astype(float)
                max_val = feature_frame.max()
                if max_val:
                    feature_frame = feature_frame / max_val
                feature_frame = np.stack((feature_frame,) * 3, axis=-1)
                rgb_frame = img_as_float(rgb_frame)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    image = img_as_ubyte(np.hstack((rgb_frame, feature_frame)))
                writer.append_data(image)
            writer.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--preprocess_dir')
    parser.add_argument('--start_from')
    parser.add_argument('--stop_after')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
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
