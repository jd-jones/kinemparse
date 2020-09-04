import logging
import warnings

import numpy as np
import skimage
import scipy

from mathtools import utils
from visiontools import imageprocessing, render
from . import primesense


"""
This module contains functions that are used specifically to pre-process video
frames before decoding assembly sequences. It is meant to be separate from
``imageprocessing``, which should hold general-purpose image processing
utilities.

FIXME: Right now, ``imageprocessing`` still contains some non-general-purpose
    functions, which should be moved into this module.
"""


logger = logging.getLogger(__name__)


def loadVideo(
        rgb_frame_fn_seq, rgb_frame_timestamp_seq,
        depth_frame_fn_seq, depth_frame_timestamp_seq):

    num_rgb = len(rgb_frame_fn_seq)
    num_depth = len(depth_frame_fn_seq)
    if num_rgb != num_depth:
        err_str = f'{num_rgb} RGB frames != {num_depth} depth frames'
        raise ValueError(err_str)

    rgb_frame_seq = primesense.loadRgbFrameSeq(rgb_frame_fn_seq, rgb_frame_timestamp_seq)
    depth_frame_seq = primesense.loadDepthFrameSeq(depth_frame_fn_seq, depth_frame_timestamp_seq)

    return rgb_frame_seq, depth_frame_seq


def foregroundPixels(
        camera_params, camera_pose, depth_image,
        mask_left_side=0.4, plane_distance_thresh=10):
    """ Identify foreground pixels from a depth frame.

    The scene captured by this depth frame is assumed to be supported by a plane.

    Parameters
    ----------
    depth_image : numpy array of float, shape (img_height, img_width)
        Depth image. Each pixel stores that location's distance from the camera,
        in millimeters. Note that this must have type ``float`` to play nicely
        with the plane-fitting routine it calls.
    plane_distance_thresh : float, optional
    mask_left_side : float, optional

    Returns
    -------
    is_foreground : numpy array of bool, shape (img_height, img_width)
        A mask image. Each pixel is True if that location is assigned to the
        foreground, and False if not.
    background_plane : geometry.Plane
        The plane which best fits the depth image. This should match the tabletop
        that supports the foreground of the scene.
    """

    background_mask = imageprocessing.maskDepthArtifacts(depth_image)

    background_plane, plane_distance_image = imageprocessing.fitPlane(
        depth_image, ~background_mask, plane_distance_thresh,
        camera_params=camera_params, camera_pose=camera_pose,
        max_trials=50
    )
    background_mask |= plane_distance_image < plane_distance_thresh

    background_mask = imageprocessing.makeBackgroundMask(depth_image, background_mask)

    background_mask = imageprocessing.maskOutsideBuildArea(
        background_mask, mask_left_side=mask_left_side, mask_bottom=0
    )

    return ~background_mask, background_plane


def quantizeImage(pixel_classifier, rgb_img, segment_img):
    """ Use pixel_classifier to quantize the input image.

    Parameters
    ----------
    pixel_classifier : models.FrameScorer
        A FrameScorer mixture model. This is used to classify each pixel in the
        RGB frames as blocks or as nuisance (hands or specularity).
    rgb_img : numpy array of float, shape (img_height, img_width, 3)
        RGB frame to classify.
    segment_img : numpy array of int, shape (img_height, img_width)
        A segmentation of an RGBD video frame. The background is assigned label
        0, and foreground segments are enumerated from 1.

    Returns
    -------
    px_class_img : numpy array of int, shape (img_height, img_width)
        The model's predictions for the pixels of the input image.
    """

    if not segment_img.any():
        return np.zeros_like(rgb_img)

    pixels = imageprocessing.foregroundPixels(
        rgb_img, segment_img,
        image_transform=skimage.color.rgb2hsv, background_class_index=0
    )

    quantized = pixel_classifier.quantize(pixels, colorspace='rgb')

    pixel_class_img = imageprocessing.imageFromForegroundPixels(
        quantized, segment_img,
        background_class_index=0
    )

    return pixel_class_img


def pixelClasses(
        pixel_classifier, rgb_img, segment_img, predict_clusters=False, **classifier_kwargs):
    """ Classify each pixel of the input image.

    Parameters
    ----------
    pixel_classifier : models.FrameScorer
        A FrameScorer mixture model. This is used to classify each pixel in the
        RGB frames as blocks or as nuisance (hands or specularity).
    rgb_img : numpy array of float, shape (img_height, img_width, 3)
        RGB frame to classify.
    segment_img : numpy array of int, shape (img_height, img_width)
        A segmentation of an RGBD video frame. The background is assigned label
        0, and foreground segments are enumerated from 1.
    predict_clusters : bool
        If True, predict cluster indices instead of high-level classes.
    **classifier_kwargs : optional
        Any extra keyword arguments are passed to the pixel classifier.

    Returns
    -------
    px_class_img : numpy array of int, shape (img_height, img_width)
        The model's predictions for the pixels of the input image.
    """

    # Extract foreground pixels in HSV colorspace
    pixels = imageprocessing.foregroundPixels(
        rgb_img, segment_img,
        image_transform=skimage.color.rgb2hsv, background_class_index=0
    )

    # Predict pixel classes
    if predict_clusters:
        pixel_classes = pixel_classifier.predict(pixels)
    else:
        pixel_snrs = pixel_classifier.pixelwiseSnr(pixels, log_domain=True, **classifier_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered")
            pixel_classes = (pixel_snrs < 0).astype(int)

    # Convert back to image shape
    pixel_class_img = imageprocessing.imageFromForegroundPixels(
        pixel_classes, segment_img,
        background_class_index=0
    )

    return pixel_class_img


def depthClasses(depth_img, min_depth=500):
    """ Identify pixels in a depth image that are probably hands or arms.

    Parameters
    ----------
    depth_img : numpy array of float, shape (img_height, img_width)
    min_depth : float, optional

    Returns
    -------
    is_near_table : numpy array of bool, shape (img_height, img_width)
    """

    is_near_camera = depth_img < min_depth

    return is_near_camera


def removeTargetModel(
        goal_state, rgb_image, depth_image, seg_image,
        camera_params=None, camera_pose=None, block_colors=None,
        method='lower_right'):
    """ Find and remove the target model from a segment image.
    """

    if camera_params is None:
        camera_params = render.intrinsic_matrix
    if camera_pose is None:
        camera_pose = render.camera_pose
    if block_colors is None:
        block_colors = render.object_colors

    rgb_image = rgb_image.copy()
    depth_image = depth_image.copy()
    seg_image = seg_image.copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero")
        rgb_image = imageprocessing.color.rgb2hsv(rgb_image)
    rgb_image = imageprocessing.hueImgAsRgb(rgb_image)

    rgb_image[seg_image == 0] = 0
    depth_image[seg_image == 0] = 0

    goodness_arr = []
    for seg_idx in range(1, seg_image.max() + 1):
        object_mask = seg_image == seg_idx

        R, t = imageprocessing.estimateSegmentPose(
            camera_params, camera_pose, depth_image, object_mask
        )

        if method == 'match_template':
            rgb_render, depth_render, label_render = render.renderComponent(
                goal_state, 0, component_pose=(R, t), img_type=None,
                camera_pose=camera_pose, camera_params=camera_params,
                block_colors=render.object_colors,
                crop_rendered=False,
                obsv_std=1
            )
            rgb_residual = rgb_image - rgb_render
            goodness = - np.sum(rgb_residual ** 2)
        elif method == 'lower_right':
            direction = np.array([1, -1, 0])
            goodness = np.dot(direction, t)

        goodness_arr.append(goodness)

    if goodness_arr:
        # Segment labels are one-indexed
        # logger.info(f"goodness: {goodness_arr}")
        best_idx = np.array(goodness_arr).argmax() + 1
        seg_image[seg_image == best_idx] = 0
        seg_image = skimage.segmentation.relabel_sequential(seg_image)[0]

    return seg_image


def segmentImage(
        goal_state, rgb_image, depth_image, is_foreground,
        target_removal_method='lower_right', slic=True):
    """ Segment the foreground of an image to produce object proposals.

    Parameters
    ----------
    goal_state : blockassembly.BlockAssembly
    rgb : numpy array of float, shape (img_height, img_width)
    depth : numpy array of float, shape (img_height, img_width)
    background_model : geometry.Plane
    background_mask : numpy array of bool, shape (img_height, img_width)
    target_removal_method : {'match_template', 'lower_right'}, optional

    Returns
    -------
    filtered_segs : numpy array of int, shape (img_height, img_width)
    pixel_classes : numpy array of int, shape (img_height, img_width)
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only one label was provided")
        is_foreground = skimage.morphology.remove_small_objects(is_foreground, min_size=64)
    foreground_segs, n_labels = skimage.morphology.label(is_foreground, return_num=True)

    if target_removal_method is not None:
        foreground_segs = removeTargetModel(
            goal_state, rgb_image, depth_image, foreground_segs,
            method=target_removal_method
        )

    is_foreground = foreground_segs > 0

    if slic:
        superpixels = skimage.segmentation.slic(
            rgb_image,
            n_segments=100, compactness=10.0, max_iter=10, sigma=0,
            spacing=None, multichannel=True, convert2lab=None,
            enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3,
            slic_zero=False
        )
        superpixels[~is_foreground] = 0
        foreground_segs = skimage.segmentation.join_segmentations(foreground_segs, superpixels)

    return foreground_segs


def filterSegments(
        segments, pixel_classes, depth_classes, thresh=0.9,
        remove_skin=False, remove_white=False, remove_blocks=False):
    """ Remove small segments and segments that are mostly nuisance pixels.

    Parameters
    ----------
    segments :
        Each image is a segmentation of the corresponding (rgb, depth) pair.
        The background is assigned label 0, and foreground segments are
        enumerated sequentially from 1.
    pixel_classes :
    depth_classes :

    Returns
    -------
    filtered_segments :
        Each image is a segmentation of the corresponding (rgb, depth) pair.
        The background is assigned label 0, and foreground segments are
        enumerated sequentially from 1.
    """

    # Ignore small segments
    # filtered_segments = skimage.morphology.remove_small_objects(segments, min_size=64)

    pixel_is_bad = (pixel_classes == 1) | (depth_classes == 1)

    filtered_segments = segments.copy()
    for segment_idx in range(1, filtered_segments.max() + 1):
        segment_mask = filtered_segments == segment_idx
        num_pixels = np.sum(segment_mask)
        num_bad_pixels = np.sum(pixel_is_bad[segment_mask])

        if num_bad_pixels / num_pixels > thresh:
            filtered_segments[segment_mask] = 0

    # Ignore skin pixels and pixels too close to the camera
    if remove_skin:
        filtered_segments[pixel_classes == 1] = 0
        filtered_segments[depth_classes == 1] = 0

    if remove_white:
        filtered_segments[pixel_classes == 2] = 0

    if remove_blocks:
        filtered_segments[pixel_classes == 3] = 0

    # Create new segments out of the contiguous parts of the new foreground
    is_foreground = filtered_segments > 0
    filtered_segments = skimage.morphology.label(is_foreground)

    # Remove any of the new segments that are too small, then relabel so the
    # returned segments are enumerated sequentially
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Only one label was provided")
        filtered_segments = skimage.morphology.remove_small_objects(filtered_segments, min_size=64)
    filtered_segments = skimage.segmentation.relabel_sequential(filtered_segments)[0]

    return filtered_segments


def segmentVideo(
        pixel_classifier, goal_state, rgb_frame_seq, depth_frame_seq,
        filter_segments=True, **segmentimage_kwargs):
    """ Segment each frame in an RGB-D video and remove nuisance objects

    This pre-processing procedure consists of the following steps:
        1. Identify the scene background
            a. Fit a plane to the tabletop
            b. Mask some areas in the image
        2. Segment the foreground image
            a. Find connected components in the foreground
            b. Remove the segment that best fits the goal state template image
            c. Segment the remaining foreground into SLIC superpixels
            d. Intersect the foreground and SLIC segmentations
        3. Filter image segments (optional)
            a. Remove segments smaller than 64 pixels
            b. Classify nuisance pixels from RGB and depth images
            c. Remove segments that are mostly nuisance pixels

    Parameters
    ----------
    pixel_classifier : models.FrameScorer
        A FrameScorer mixture model. This is used to classify each pixel in the
        RGB frames as blocks or as nuisance (hands or specularity).
    goal_state : blockassembly.BlockAssembly
        The builder's goal state for this video. This is used to match and
        remove the reference model from the frame.
    rgb_frame_seq : iterable(numpy array of float, shape (img_height, img_width, 3))
        RGB frames to process.
    depth_frame_seq : iterable(numpy array of float, shape (img_height, img_width))
        Depth frames to process.
    filter_segments : bool, optional
        If True, segments are removed if they contain too many nuisance pixels
        or if they are too close to the camera. Default is True.
    **segmentimage_kwargs: optional
        Any extra keyword arguments are passed to ``videoprocessing.segmentImage``.

    Returns
    -------
    segment_seq : tuple(numpy array of int, shape (img_height, img_width))
        Each image is a segmentation of the corresponding (rgb, depth) pair.
        The background is assigned label 0, and foreground segments are
        enumerated sequentially from 1.
    px_class_seq : tuple(numpy array of int, shape (img_height, img_width))
        Each image contains `frame_scorer`'s predictions for the pixels in the
        corresponding RGB image.
    background_model_seq : tuple(geometry.Plane)
        Each element is the plane that best fits the corresponding depth image.
        This plane should represent the tabletop, which supports the planar
        scene recorded by the camera (if we're lucky).
    """

    # Detect the background
    foreground_mask_seq, background_plane_seq = utils.batchProcess(
        foregroundPixels,
        depth_frame_seq,
        unzip=True
    )

    # Create an initial image segmentation
    segment_seq = utils.batchProcess(
        segmentImage,
        rgb_frame_seq, depth_frame_seq, background_plane_seq, foreground_mask_seq,
        static_args=(goal_state.copy(),),
        static_kwargs=segmentimage_kwargs
    )

    # Identify nuisance or outlying objects in images
    pixel_class_seq = utils.batchProcess(
        pixelClasses, rgb_frame_seq, segment_seq,
        static_args=(pixel_classifier,)
    )
    depth_class_seq = utils.batchProcess(depthClasses, depth_frame_seq)

    # Remove segments that are mostly nuisance pixels
    if filter_segments:
        segment_seq = utils.batchProcess(
            filterSegments,
            segment_seq, pixel_class_seq, depth_class_seq
        )

    return segment_seq, pixel_class_seq, background_plane_seq


def scoreFrames(pixel_classifier, rgb_frame_seq, segment_seq, score_kwargs=None):
    """ Score each video frame. Higher scores means more blocks pixels.

    Parameters
    ----------
    pixel_classifier : models.FrameScorer
        A FrameScorer mixture model. This is used to classify each pixel in the
        RGB frames as blocks or as nuisance (hands or specularity).
    rgb_frame_seq : iterable(numpy array of float, shape (img_height, img_width, 3))
        RGB frames to process.
    segment_seq : iterable(numpy array of int, shape (img_height, img_width))

    Returns
    -------
    scores : numpy array of float, shape (num_frames / sample_rate,)
        Average (log) likelihood ratio of 'blocks' class vs. 'skin' class
    """

    if score_kwargs is None:
        score_kwargs = {'log_domain': True, 'hard_assign_clusters': True}

    fg_px_seq = utils.batchProcess(
        imageprocessing.foregroundPixels,
        rgb_frame_seq, segment_seq,
        static_kwargs={
            'image_transform': lambda x: skimage.color.rgb2hsv(skimage.img_as_float(x)),
            'background_class_index': 0
        }
    )

    scores = utils.batchProcess(
        pixel_classifier.averageSnr,
        fg_px_seq,
        static_kwargs=score_kwargs
    )

    return np.array(scores)


# FIXME: Move to rawdata or utils
def extractWindows(signal, window_size=10, return_window_indices=False):
    """ Reshape a signal into a series of non-overlapping windows.

    Parameters
    ----------
    signal : numpy array, shape (num_samples,)
    window_size : int, optional
    return_window_indices : bool, optional

    Returns
    -------
    windows : numpy array, shape (num_windows, window_size)
    window_indices : numpy array of int, shape (num_windows, window_size)
    """

    tail_len = signal.shape[0] % window_size
    pad_arr = np.full(window_size - tail_len, np.nan)
    signal_padded = np.concatenate((signal, pad_arr))
    windows = signal_padded.reshape((-1, window_size))

    if not return_window_indices:
        return windows

    indices = np.arange(signal_padded.shape[0])
    window_indices = indices.reshape((-1, window_size))

    return windows, window_indices


# FIXME: move to rawdata or utils
def nanargmax(signal, axis=1):
    """ Select the greatest non-Nan entry from each row.

    If a row does not contain at least one non-NaN entry, it does not have a
    corresponding output.

    Parameters
    ----------
    signal : numpy array, shape (num_rows, num_cols)
        The reference signal for argmax. This is usually an array constructed
        by windowing a univariate signal. Each row is a window of the signal.
    axis : int, optional
        Axis along which to compute the argmax. Not implemented yet.

    Returns
    -------
    non_nan_row_idxs : numpy array of int, shape (num_non_nan_rows,)
        An array of the row indices in `signal` that have at least one non-NaN
        entry, arranged in increasing order.
    non_nan_argmax : numpy array of int, shape (num_non_nan_rows,)
        An array of the greatest non-NaN entry in `signal` for each of the rows
        in `non_nan_row_idxs`.
    """

    if axis != 1:
        err_str = 'Only axis 1 is supported right now'
        raise NotImplementedError(err_str)

    # Determine which rows contain all NaN
    row_is_all_nan = np.isnan(signal).all(axis=1)
    non_nan_row_idxs = np.nonzero(~row_is_all_nan)[0]

    # Find the (non-NaN) argmax of each row with at least one non-NaN value
    non_nan_rows = signal[~row_is_all_nan, :]
    non_nan_argmax = np.nanargmax(non_nan_rows, axis=1)

    return non_nan_row_idxs, non_nan_argmax


def selectSegmentKeyframes(
        scores, segment_labels=None, score_thresh=0, min_seg_len=2, prepend_first=False):
    """ Identify segments in a score sequence, then choose the highest-scoring frame in each.

    Parameters
    ----------
    scores : numpy array of float, shape (num_frames,)
    segment_labels : numpy array of int, shape (num_frames,)
    score_thresh : float, optional
    prepend_first : bool, optional
        If True, every keyframe sequence begins with 0.

    Returns
    -------
    keyframe_idxs : numpy array of int, shape (num_segments,)
    """

    if segment_labels is None:
        # > throws a warning if any values are NaN. We can ignore it because the
        # default behavior works for us---NaN returns False for any comparison
        # other than == or !=.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered")
            is_peak = scores > score_thresh
        segment_labels, num_labels = scipy.ndimage.label(is_peak)

    unique_labels = np.unique(segment_labels)
    # 0 is the "background" class (ie no activity), so ignore it
    unique_labels = unique_labels[unique_labels != 0]
    num_unique_labels = unique_labels.shape[0]

    best_idxs = np.zeros(num_unique_labels, dtype=int)
    for i, label in enumerate(unique_labels):
        l_scores = scores.copy()
        if np.sum(segment_labels == label) < min_seg_len:
            best_idx = -1
        else:
            l_scores[segment_labels != label] = np.nan
            try:
                best_idx = np.nanargmax(l_scores)
            except ValueError:
                best_idx = 0
        best_idxs[i] = best_idx
    best_idxs = best_idxs[best_idxs > -1]

    if prepend_first:
        new_best_idxs = np.zeros(len(best_idxs) + 1, dtype=int)
        new_best_idxs[1:] = best_idxs
        best_idxs = new_best_idxs

    return best_idxs


def selectWindowKeyframes(scores, window_size=10):
    """ Choose the highest-scoring frame in each window.

    If all the scores in a window are NaN, that window is ignored.

    Parameters
    ----------
    scores :
    window_size : int, optional

    Returns
    -------
    window_idxs :
    """

    score_windows, window_indices = extractWindows(
        scores, window_size=window_size, return_window_indices=True
    )

    row_idxs, col_argmax = nanargmax(score_windows)

    keyframe_idxs = window_indices[row_idxs, col_argmax]

    return keyframe_idxs
