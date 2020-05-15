import os
import glob
import argparse

import joblib
import yaml
import numpy as np
import imageio

from blocks.core import utils
from blocks.estimation import imageprocessing, videoprocessing, models


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


def labelImage(model, rgb_image, seg_image):
    foreground = imageprocessing.foregroundPixels(
        rgb_image, seg_image,
        image_transform=imageprocessing.color.rgb2hsv,
        background_class_index=0
    )
    px_class_labels, px_color_labels = model.predictClass(foreground, return_cluster_preds=True)
    px_quantized = model.quantize(px_color_labels, colorspace='rgb', input_is_clusters=True)

    # Don't forget that color labels are zero-indexed and don't have a
    # class --- we add one so that zero can represent the background class.
    color_label_image = imageprocessing.imageFromForegroundPixels(
        px_color_labels + 1, seg_image, background_class_index=0
    )

    class_label_image = imageprocessing.imageFromForegroundPixels(
        px_class_labels, seg_image, background_class_index=0
    )

    quantized_image = imageprocessing.imageFromForegroundPixels(
        px_quantized, seg_image, background_class_index=0
    )

    return color_label_image, class_label_image, quantized_image


def markDetections(rgb_image, blocks_bboxes, skin_bboxes):
    marked_image = rgb_image.copy()

    for bbox in blocks_bboxes:
        perim_r, perim_c = imageprocessing.rectangle_perimeter(*bbox)
        perim_r = np.hstack(
            (perim_r,) * 3 + (perim_r + 1,) * 3 + (perim_r - 1,) * 3
        )
        perim_c = np.hstack(
            (perim_c, perim_c + 1, perim_c - 1) * 3
        )
        marked_image[perim_r, perim_c] = [0, 255, 0]

    for bbox in skin_bboxes:
        perim_r, perim_c = imageprocessing.rectangle_perimeter(*bbox)
        perim_r = np.hstack(
            (perim_r,) * 3 + (perim_r + 1,) * 3 + (perim_r - 1,) * 3
        )
        perim_c = np.hstack(
            (perim_c, perim_c + 1, perim_c - 1) * 3
        )
        marked_image[perim_r, perim_c] = [255, 0, 0]

    return marked_image


def getUniqueTrialIds(dir_path):
    trial_ids = set(
        int(os.path.basename(fn).split('-')[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"trial-*.pkl"))
    )
    return sorted(tuple(trial_ids))


def main(
        out_dir=None, data_dir=None, preprocess_dir=None, keyframe_model_fn=None,
        start_from=None, stop_at=None,
        write_video=None, upsample_frames_by=1, display_summary_img=None,
        filter_options={}):

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

    if write_video:
        video_dir = os.path.join(out_dir, 'detection-videos')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

    def loadFromDataDir(var_name):
        return joblib.load(os.path.join(data_dir, f"{var_name}.pkl"))

    def loadFromPreprocessDir(var_name):
        return joblib.load(os.path.join(preprocess_dir, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    keyframe_model = monkeyPatchKeyframeModel(joblib.load(keyframe_model_fn))

    trial_ids = getUniqueTrialIds(preprocess_dir)
    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        trial_str = f"trial-{trial_id}"

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info(f"  Loading data...")
        rgb_frame_seq = loadFromDataDir(f"{trial_str}_rgb-frame-seq")
        depth_frame_seq = loadFromDataDir(f"{trial_str}_depth-frame-seq")
        foreground_mask_seq = loadFromPreprocessDir(f'{trial_str}_foreground-mask-seq')
        segment_frame_seq = loadFromPreprocessDir(f'{trial_str}_segment-frame-seq')

        logger.info(f"  Labeling foreground...")
        color_label_frame_seq, class_label_frame_seq, quantized_frame_seq = utils.batchProcess(
            labelImage, rgb_frame_seq, foreground_mask_seq,
            static_args=(keyframe_model,),
            unzip=True
        )
        depth_class_frame_seq = utils.batchProcess(videoprocessing.depthClasses, depth_frame_seq)

        logger.info(f"  Computing object detections...")
        block_segment_frame_seq = utils.batchProcess(
            videoprocessing.filterSegments,
            segment_frame_seq, class_label_frame_seq, depth_class_frame_seq,
            static_kwargs={'remove_skin': True, 'remove_white': True}
        )
        skin_segment_frame_seq = utils.batchProcess(
            videoprocessing.filterSegments,
            segment_frame_seq, class_label_frame_seq, depth_class_frame_seq,
            static_kwargs={'remove_blocks': True, 'remove_white': True}
        )

        block_bboxes_seq = utils.batchProcess(
            imageprocessing.segmentBoundingBoxes,
            rgb_frame_seq, block_segment_frame_seq
        )
        skin_bboxes_seq = utils.batchProcess(
            imageprocessing.segmentBoundingBoxes,
            rgb_frame_seq, skin_segment_frame_seq
        )

        marked_rgb_frame_seq = utils.batchProcess(
            markDetections,
            rgb_frame_seq, block_bboxes_seq, skin_bboxes_seq
        )

        logger.info(f"  Saving output...")
        saveToWorkingDir(np.stack(block_segment_frame_seq), f'{trial_str}_block-segment-frame-seq')
        saveToWorkingDir(np.stack(skin_segment_frame_seq), f'{trial_str}_skin-segment-frame-seq')
        saveToWorkingDir(np.stack(color_label_frame_seq), f'{trial_str}_color-label-frame-seq')
        saveToWorkingDir(np.stack(class_label_frame_seq), f'{trial_str}_class-label-frame-seq')

        if display_summary_img:
            if utils.in_ipython_console():
                file_path = None
            else:
                trial_str = f"trial-{trial_id}"
                file_path = os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            imageprocessing.displayImages(
                *marked_rgb_frame_seq, *class_label_frame_seq,
                num_rows=2, file_path=file_path
            )

        if write_video:
            fn = os.path.join(video_dir, f"{trial_str}.gif")
            writer = imageio.get_writer(fn, mode='I')
            for img in marked_rgb_frame_seq:
                for i in range(upsample_frames_by):
                    writer.append_data(img)
            writer.close()
    logger.info("Finished.")


if __name__ == "__main__":
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
