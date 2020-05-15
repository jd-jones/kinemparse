import logging
import functools

import numpy as np
from skimage import io, color, img_as_float

from blocks.core import definitions as defn
from mathtools import utils
from visiontools import imageprocessing


logger = logging.getLogger(__name__)


def checkSeqTime(frame_timestamp_seq):
    start_time = frame_timestamp_seq[0]
    return start_time < defn.time_of_unmirror_merge


def unmirrorStateSeqs(timestamps, state_seqs, path_ids):
    return utils.iterate(unmirrorStateSeq, timestamps, state_seqs, path_ids)


def unmirrorStateSeq(timestamp_seq, state_seq, path_id):
    """ Some videos were recorded and annotated as mirror-images of what
    actually happened. This function checks if a state sequence corresponds to
    a mirror-image video and un-mirrors the state sequence if it does.

    Parameters
    ----------
    bound_seq : []
      []
    state_seq : []
      []
    path_id : int
      []

    Returns
    -------
    state_seq : []
      []
    """

    # start_time = time.gmtime(bound_seq[1][0])
    # time_str = time.strftime("%Y-%m-%d %H:%M:%S (GMT)", start_time)
    # logger.debug('Video %s recorded ~ %s', str(path_id), time_str)
    mirror = checkSeqTime(timestamp_seq)

    if mirror:
        logger.info('Un-mirroring trial {}...'.format(path_id))
        for state in state_seq:
            state.mirror()

    return state_seq


def loadDepthFrameSeqs(frame_fn_seqs, frame_timestamp_seqs):
    return utils.iterate(loadDepthFrameSeq, frame_fn_seqs, frame_timestamp_seqs)


def loadDepthFrameSeq(frame_fn_seq, frame_timestamp_seq, stack_frames=False, **load_kwargs):
    if not any(frame_fn_seq):
        return tuple()

    mirror = checkSeqTime(frame_timestamp_seq)
    f = functools.partial(loadDepthFrame, mirror=mirror, **load_kwargs)
    depth_frame_seq = utils.iterate(f, frame_fn_seq)

    if stack_frames:
        depth_frame_seq = np.stack(depth_frame_seq)

    return depth_frame_seq


def loadDepthFrame(path, mirror=False, as_float=False, normalize=False):
    img = io.imread(path)
    if mirror:
        img = np.fliplr(img)
    if as_float:
        img = img.astype(float)  # img_as_float(img)
    if normalize:
        img /= img.max()
    return img


def loadRgbFrameSeqs(frame_fn_seqs, frame_timestamp_seqs):
    return utils.iterate(loadRgbFrameSeq, frame_fn_seqs, frame_timestamp_seqs)


def loadRgbFrameSeq(frame_fn_seq, frame_timestamp_seq=None, stack_frames=False):
    """ Load a sequence of video frames.

    NOTE: Since the first videos were recorded when the camera was in 'mirror'
        mode, this function un-mirrors any videos recorded before the time of
        the last mirrored video.

    Parameters
    ----------
    frame_fn_seq : iterable(string)
        The filename of each video frame.
    frame_timestamp_seq : numpy array of float, shape (num_frames,), optional
        The time each video frame was received by the data collection computer.
        This is used to decide if the video should be un-mirrored. If no
        value is passed, this function doesn't un-mirror the video.

    Returns
    -------
    frame_seq : tuple( numpy array of float, shape (img_height, img_width) )
    """

    if not any(frame_fn_seq):
        return tuple()

    if frame_timestamp_seq is None:
        mirror = False
    else:
        mirror = checkSeqTime(frame_timestamp_seq)

    frame_seq = utils.iterate(
        loadRgbFrame,
        frame_fn_seq,
        static_kwargs={'mirror': mirror}
    )

    if stack_frames:
        frame_seq = np.stack(frame_seq)

    return frame_seq


def loadRgbFrame(path, mirror=False, as_float=False):
    """ Load an RGB video frame.

    Parameters
    ----------
    path : string
        Path to the image file.
    mirror : bool, optional
        If True, the image is reflected along the horizontal axis (ie mirrored)
        before it is returned. Default is False.
    as_float : bool, optional
        If True, the image is converted to floating-point representation before
        being returned. If False, the image is loaded as-is (probably 256-color
        integer representation). Default is False.

    Returns
    -------
    rgb_frame : numpy array of int (or float), shape (img_height, img_width)
        The video frame.
    """

    img = io.imread(path)

    if mirror:
        img = np.fliplr(img)

    if as_float:
        img = img_as_float(img)

    return img


def resampleFrameFnSeqs(frame_fn_seqs, frame_timestamp_seqs, seq_bounds):
    resampled_pairs = utils.iterate(
        resampleFrameFnSeq,
        frame_fn_seqs, frame_timestamp_seqs, seq_bounds
    )
    return tuple(zip(*resampled_pairs))


def resampleFrameFnSeq(frame_fn_seq, frame_timestamp_seq, seq_bound):
    start_time, end_time = seq_bound
    sample_rate = defn.video_sample_rate
    sample_times = utils.computeSampleTimes(sample_rate, start_time, end_time)

    resampled_fns = utils.resampleSeq(frame_fn_seq, frame_timestamp_seq, sample_times)
    resampled_timestamps = utils.resampleSeq(frame_timestamp_seq, frame_timestamp_seq, sample_times)

    return resampled_fns, resampled_timestamps


def estimateKeyframeFnSeqs(frame_fn_seqs, frame_timestamp_seqs, keyframe_timestamp_seqs):
    keyframe_fn_seqs = utils.iterate(
        estimateKeyframeFnSeq,
        frame_fn_seqs, frame_timestamp_seqs, keyframe_timestamp_seqs
    )

    return keyframe_fn_seqs


def estimateKeyframeFnSeq(frame_fn_seq, frame_timestamp_seq, keyframe_timestamps):
    keyframe_fns = utils.resampleSeq(frame_fn_seq, frame_timestamp_seq, keyframe_timestamps)

    return keyframe_fns


def findBlobs(rgb_image, depth_image):
    # mag_image = color.rgb2gray(rgb_image)
    hsv_image = color.rgb2hsv(rgb_image)
    hue_image = hsv_image[:,:,0]

    background_mask = imageprocessing.makeBackgroundMask(depth_image)

    label_mask = imageprocessing.labelObjects(background_mask)

    num_seg_pixels = 200
    superpixels = imageprocessing.segmentObjects(
        label_mask, num_seg_pixels,
        depth_image,
        100 * imageprocessing.shift(hue_image)
    )

    if superpixels.max():
        superpixels = imageprocessing.mergeSuperpixels(superpixels, rgb_image, depth_image)

    superpixel_image = imageprocessing.makeSuperpixelImage(superpixels, hsv_image)

    return superpixels, superpixel_image


def markBuildAreas(image):
    image = image.copy()

    rows, cols, channels = image.shape
    row_bound = int(0.4 * rows)
    col_bound = cols // 3
    image[:, col_bound, :] = 1
    image[row_bound, col_bound:, :] = 1

    return image
