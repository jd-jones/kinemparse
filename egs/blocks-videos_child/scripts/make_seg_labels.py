import os
import warnings

import joblib
import yaml
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology, segmentation, measure, color, img_as_float

from mathtools import utils
from visiontools import imageprocessing


def removeTargetModel(seg_image, num):
    """ Find and remove the target model from a segment image. """

    seg_centroids = np.row_stack(tuple(
        np.column_stack(np.nonzero(seg_image == i)).mean(axis=0)
        for i in range(1, num + 1)
    ))

    direction = np.array([3, 4])
    seg_scores = seg_centroids @ direction

    # Segment labels are one-indexed
    best_idx = np.array(seg_scores).argmax() + 1
    seg_image[seg_image == best_idx] = 0
    seg_image = segmentation.relabel_sequential(seg_image)[0]

    return seg_image, num - 1


def makeCoarseSegmentLabels(mask, min_size=100):
    mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=1)
    labels, num = measure.label(mask.astype(int), return_num=True)
    if num < 2:
        return labels
    labels, num = removeTargetModel(labels, num)
    return labels


def makeFineSegmentLabels(coarse_seg_labels, bg_mask_sat, min_size=100):
    labels, num = measure.label(coarse_seg_labels, return_num=True)
    for i in range(1, num + 1):
        in_seg = labels == i
        bg_vals = bg_mask_sat[in_seg]
        class_counts = np.hstack((np.sum(bg_vals == 0), np.sum(bg_vals == 1)))
        is_bg = class_counts.argmax().astype(bool)
        if is_bg:
            labels[in_seg] = 0

    fg_mask = morphology.remove_small_objects(labels != 0, min_size=min_size, connectivity=1)
    labels, num = measure.label(fg_mask.astype(int), return_num=True)
    for i in range(1, num + 1):
        in_seg = labels == i
        labels[in_seg] = coarse_seg_labels[in_seg]

    labels = segmentation.relabel_sequential(labels)[0]
    return labels


def makeHsvFrame(rgb_image):
    rgb_image = img_as_float(rgb_image)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        hsv_image = color.rgb2hsv(rgb_image)

    return hsv_image


def plotHsvHist(hsv_frame_seq, seg_labels_seq, file_path=None):
    fg = hsv_frame_seq[seg_labels_seq != 0]

    names = ('hue', 'sat', 'val')
    fig, axes = plt.subplots(3)
    for i in range(3):
        axes[i].hist(fg[:, i], bins=100)
        axes[i].set_ylabel(names[i])

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def main(
        out_dir=None, data_dir=None, person_masks_dir=None, bg_masks_dir=None,
        sat_thresh=1, start_from=None, stop_at=None, num_disp_imgs=None):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    person_masks_dir = os.path.expanduser(person_masks_dir)
    bg_masks_dir = os.path.expanduser(bg_masks_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        trial_str = f"trial={trial_id}"

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info("  Loading data...")
        rgb_frame_seq = loadFromDir(f'{trial_str}_rgb-frame-seq', data_dir)
        person_mask_seq = loadFromDir(f'{trial_str}_person-mask-seq', person_masks_dir)
        bg_mask_seq_depth = loadFromDir(f'{trial_str}_bg-mask-seq-depth', bg_masks_dir)
        # bg_mask_seq_rgb = loadFromDir(f'{trial_str}_bg-mask-seq-rgb', bg_masks_dir)

        logger.info("  Making segment labels...")
        fg_mask_seq = ~bg_mask_seq_depth
        seg_labels_seq = np.stack(tuple(map(makeCoarseSegmentLabels, fg_mask_seq)), axis=0)

        hsv_frame_seq = np.stack(tuple(map(makeHsvFrame, rgb_frame_seq)), axis=0)
        sat_frame_seq = hsv_frame_seq[..., 1]
        bg_mask_seq_sat = sat_frame_seq < sat_thresh

        seg_labels_seq[person_mask_seq] = 0
        seg_labels_seq = np.stack(
            tuple(
                makeFineSegmentLabels(segs, sat)
                for segs, sat in zip(seg_labels_seq, bg_mask_seq_sat)
            ),
            axis=0
        )

        logger.info("  Saving output...")
        saveToWorkingDir(seg_labels_seq.astype(np.uint8), f'{trial_str}_seg-labels-seq')

        plotHsvHist(
            hsv_frame_seq, seg_labels_seq,
            file_path=os.path.join(fig_dir, f'{trial_str}_hsv-hists.png')
        )

        if num_disp_imgs is not None:
            if rgb_frame_seq.shape[0] > num_disp_imgs:
                idxs = np.arange(rgb_frame_seq.shape[0])
                np.random.shuffle(idxs)
                idxs = idxs[:num_disp_imgs]
            else:
                idxs = slice(None, None, None)
            imageprocessing.displayImages(
                *(rgb_frame_seq[idxs]),
                *(bg_mask_seq_sat[idxs]),
                *(bg_mask_seq_depth[idxs]),
                *(person_mask_seq[idxs]),
                *(seg_labels_seq[idxs]),
                num_rows=5, file_path=os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            )


if __name__ == "__main__":
    # Parse command-line args and config file
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
