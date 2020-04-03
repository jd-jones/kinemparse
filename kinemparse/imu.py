import logging
import os
import functools

import numpy as np
from matplotlib import pyplot as plt

from mathtools import utils
from blocks.core import definitions as defn


logger = logging.getLogger(__name__)


# --=( DATA ACCESS )=----------------------------------------------------------
def getImuGlobalTimestamps(imu_sample_sequence):
    """ return a np vector with the global timestamps for this sample sequence. """

    index = defn.imu_sample_var_idxs['global time']
    return imu_sample_sequence[:,index]


def getImuLocalTimestamps(imu_sample_sequence):
    """ return a np vector with the local timestamps for this sample sequence. """

    index = defn.imu_sample_var_idxs['local time']
    return imu_sample_sequence[:,index]


def getImuSamples(imu_sample_sequence):
    """ return a np array with the samples for this sample sequence. """

    var_names = ('x', 'y', 'z')
    indices = [defn.imu_sample_var_idxs[n] for n in var_names]
    return imu_sample_sequence[:,indices]


def makeDummyImuSamples(num_samples):
    """ return an (immutable) np array with zeros for samples and NaN for timestamps """

    global_timestamps_index = defn.imu_sample_var_idxs['global time']
    local_timestamps_index = defn.imu_sample_var_idxs['local time']

    dummy = np.zeros((num_samples, defn.num_imu_sample_vars))
    dummy[:,global_timestamps_index] = np.NaN
    dummy[:,local_timestamps_index] = np.NaN

    return dummy


def resampleImuSeq(imu_seq, seq_bound):
    start_time, end_time = seq_bound
    sample_rate = defn.imu_sample_rate
    new_sample_times = utils.computeSampleTimes(sample_rate, start_time, end_time)
    num_new_samples = new_sample_times.shape[0]

    resampled_seq = {}
    for block_index in defn.block_ids.values():
        imu_sample_seq = imu_seq.get(block_index, None)
        if imu_sample_seq is None:
            msg_str = 'No samples for block %d -- replacing with zeros'
            logger.warning(msg_str, block_index)
            resampled = makeDummyImuSamples(num_new_samples)
        else:
            sample_times = getImuGlobalTimestamps(imu_sample_seq)
            num_samples = sample_times.shape[0]
            imu_samples = np.vsplit(imu_sample_seq, num_samples)
            resampled = utils.resampleSeq(imu_samples, sample_times, new_sample_times)
            resampled = np.vstack(resampled)
        resampled_seq[block_index] = resampled

    return resampled_seq


# --=( IMU LABELS )=-----------------------------------------------------------
def labelSeq2Slices(label_seq):
    """ Convert a binary signal to a sequence of slices.

    Each slice represents a region where the signal is 1.
    """

    starts_with_one = label_seq[0]
    ends_with_one = label_seq[-1]

    deriv = np.diff(label_seq.astype(int))

    # add 1 to start_idxs because deriv[t] = signal[t+1] - signal[t]
    change_to_one_idxs = (np.nonzero(deriv == 1)[0] + 1).tolist()
    change_to_zero_idxs = np.nonzero(deriv == -1)[0].tolist()

    start_idxs = change_to_one_idxs
    if starts_with_one:
        start_idxs = [0] + start_idxs
    end_idxs = change_to_zero_idxs
    if ends_with_one:
        last_idx = len(label_seq) - 1
        end_idxs = end_idxs + [last_idx]

    if len(start_idxs) != len(end_idxs):
        err_str = f'start idxs has length {len(start_idxs)} but end idxs has length {len(end_idxs)}'
        raise ValueError(err_str)

    seg_slices = (slice(s, e + 1) for s, e in zip(start_idxs, end_idxs))

    return tuple(seg_slices)


def labelSlices2Seq(seg_slices, ref_label_seq):
    label_seq = np.zeros_like(ref_label_seq)
    for sl in seg_slices:
        label_seq[sl] = 1

    return label_seq


def reviseLabels(activity_labels, gyro_mag_signal):
    seg_slices = labelSeq2Slices(activity_labels)

    filtered_slices = tuple(
        sl for sl in seg_slices if isActive(gyro_mag_signal[sl])
    )

    revised_labels = labelSlices2Seq(filtered_slices, activity_labels)

    return revised_labels


def makeImuActivityLabels(centered_imu_samples, imu_obj_label_seq, imu_tgt_label_seq):
    """ Filter and join IMU activity labels """

    revised_imu_obj_labels = {
        imu_id: reviseLabels(imu_obj_label_seq[imu_id], imu_samples)
        for imu_id, imu_samples in centered_imu_samples.items()
    }

    revised_imu_tgt_labels = {
        imu_id: reviseLabels(imu_tgt_label_seq[imu_id], imu_samples)
        for imu_id, imu_samples in centered_imu_samples.items()
    }

    final_imu_labels = {
        imu_id: np.logical_or(revised_imu_obj_labels[imu_id], revised_imu_tgt_labels[imu_id])
        for imu_id in revised_imu_obj_labels.keys()
    }

    return final_imu_labels


def makeImuLabelSeq(
        rgb_label_seq, rgb_label_timestamp_seq, imu_sample_timestamp_seq,
        as_array=False):
    """ Resample video frame indexed labels to match IMU sample rate. """

    imu_label_seq = {
        i: utils.resampleSeq(
            rgb_label_seq[i,:], rgb_label_timestamp_seq,
            imu_sample_timestamp_seq[i],
        )
        for i in range(rgb_label_seq.shape[0])
    }

    return imu_label_seq


def computeWindowMeans(signal, window_labels):
    """ Compute the mean activity label over a window.

    Parameters
    ----------
    signal : np.array, shape (NUM_SAMPLES, NUM_DIMS)
    window_labels : np.array, shape (NUM_SAMPLES,)

    Returns
    -------
    mean_signal : np.array, shape (NUM_SAMPLES, NUM_DIMS)
    """

    window_slices = labelSeq2Slices(window_labels)

    mean_signal = np.zeros_like(signal)
    for sl in window_slices:
        mean_signal[sl,:] = signal[sl,:].mean(axis=0)

    return mean_signal


# --=( SIGNAL FEATURES )=------------------------------------------------------
def makeImuSeq(accel_seq, gyro_seq, mag_only=False):
    """ Returns an array whose columns are IMU signals sampled from multiple devices.

    NOTE: Inputs sequences are assumed to be aligned and trimmed to the same
        length already.

    Parameters
    ----------
    accel_seq : dict{ int : np.array, shape (NUM_SAMPLES, NUM_DIMS) }
        Acceleration samples from multiple IMUs. NUM_DIMS in (1, 3)
    gyro_seq : dict{ int : np.array, shape (NUM_SAMPLES, NUM_DIMS) }
        Angular velocity samples from multiple IMUs. NUM_DIMS in (1, 3)

    Returns
    -------
    imu_sample_seq : dict{ int : np.array, shape (NUM_SAMPLES, 2 * NUM_DIMS) }
        Array containing all accel and gyro samples arranged along its columns.
    imu_timestamp_seq : dict{ int : np.array, shape (NUM_SAMPLES, 1) }
        Accelerometer timestamps for each IMU.
    """

    imu_sample_seq = {}
    imu_timestamp_seq = {}
    for k in accel_seq.keys():
        accel = getImuSamples(accel_seq[k])
        gyro = getImuSamples(gyro_seq[k])
        if mag_only:
            accel = np.expand_dims(np.linalg.norm(accel, axis=1), 1)
            gyro = np.expand_dims(np.linalg.norm(gyro, axis=1), 1)
        imu_sample_seq[k] = np.hstack((accel, gyro))
        imu_timestamp_seq[k] = getImuGlobalTimestamps(accel_seq[k])

    return imu_sample_seq, imu_timestamp_seq


def isActive(signal, min_thresh=0):
    return signal.mean() > min_thresh


def activeDevices(imu_signals, check_dim='gyro'):
    if check_dim == 'gyro':
        i = 1
        thresh = 1
    elif check_dim == 'accel':
        i = 0
        thresh = 0.05

    active_devices = {
        k: signal for k, signal in imu_signals.items()
        if not deviceRestingFromSignal(signal[:,i], std_thresh=thresh)}

    return active_devices


def deviceRestingFromSignal(imu_signal, std_thresh=1):
    """ Returns True if the standard deviation of `imu_signal` is below a threshold.

    If `imu_signal` has multi-dimensional observations, the threshold is compared
    against the square root of the determinant of the covariance matrix.

    Parameters
    ----------
    imu_signal : np.array, shape (NUM_SAMPLES, NUM_DIMS)
    std_thresh : float, optional

    Returns
    -------
    is_resting : bool
    """

    cov = np.cov(imu_signal, rowvar=False)

    if utils.isUnivariate(imu_signal):
        std_dev = cov ** 0.5
    else:
        std_dev = np.linalg.det(cov) ** 0.5

    return std_dev < std_thresh


def centerSignals(imu_sample_seq, imu_is_resting=None):
    """ Subtract the mean of the active signals. """

    if isinstance(imu_sample_seq, np.ndarray):
        if imu_is_resting is None:
            imu_is_resting = np.array([deviceRestingFromSignal(seq) for seq in imu_sample_seq.T])
        mean_imu_seq = imu_sample_seq[:, ~imu_is_resting].mean(axis=1)
        centered_imu_samples = imu_sample_seq - mean_imu_seq[:, None]
        return centered_imu_samples

    if imu_is_resting is None:
        imu_is_resting = {
            imu_id: deviceRestingFromSignal(signal[:,1])
            for imu_id, signal in imu_sample_seq.items()
        }

    active_imu_seqs = {
        imu_id: signal for imu_id, signal in imu_sample_seq.items()
        if not imu_is_resting[imu_id]
    }

    mean_imu_seq = np.dstack(tuple(active_imu_seqs.values())).mean(axis=2)

    centered_imu_samples = {
        imu_id: imu_samples.copy() - mean_imu_seq
        for imu_id, imu_samples in imu_sample_seq.items()
    }

    return centered_imu_samples


def imuCorr(signal, window_len=5, eps=1e-6, lower_tri_only=True):
    windows = utils.slidingWindow(signal, window_len, stride_len=1, padding='reflect', axis=0)
    windows = windows - windows.mean(axis=-1, keepdims=True)

    inner_prods = np.einsum('tiw,tjw->tij', windows, windows)
    norms = np.einsum('tiw,tiw->ti', windows, windows) ** 0.5
    norm_prods = np.einsum('ti,tj->tij', norms, norms)
    corrs = inner_prods / (norm_prods + eps)

    if lower_tri_only:
        rows, cols = np.tril_indices(corrs.shape[1], k=-1)
        corrs = corrs[:, rows, cols]

    return corrs


def imuCovs(imu_seq, imu_is_resting=None, lower_tri_only=True):
    centered_mags = centerSignals(imu_seq, imu_is_resting=imu_is_resting)

    covs = np.einsum('ni,nj->nij', centered_mags, centered_mags)

    if lower_tri_only:
        rows, cols = np.tril_indices(covs.shape[1], k=-1)
        covs = covs[:, rows, cols]
    return covs


# --=( VISUALIZATION: DEPRECATED? )=-------------------------------------------
def plotKinemNormSeqs(sample_dicts, trial_ids, base_path, label):
    f = functools.partial(plotKinemNormSeq, base_path=base_path, label=label)
    utils.evaluate(f, sample_dicts, trial_ids)


def plotKinemNormSeq(kinem_seq, trial_id, base_path=None, label=None):
    num_blocks = len(defn.blocks)
    norm_fig, norm_axes = plt.subplots(num_blocks, sharex=True, sharey=True, figsize=(8,12))

    for block_id, sample_seq in kinem_seq.items():
        block_name = defn.blocks[block_id]
        if not sample_seq.any():
            continue

        norm_axis = norm_axes[block_id]

        global_timestamps = sample_seq[:,0] - sample_seq[0,0]
        if np.isnan(global_timestamps).any():
            continue

        samples = sample_seq[:,2:5]
        sample_norms = np.linalg.norm(samples, axis=1)

        norm_axis.plot(global_timestamps, sample_norms, c='k')
        norm_axis.set_ylabel(block_name)

    # Format and set labels
    norm_axes[-1].set_xlabel('time (seconds)')
    fig_title = '2-norm, {}'.format(label)
    norm_axes[0].set_title(fig_title)
    norm_fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in norm_axes[:-1]], visible=False)

    # Save figure
    if base_path is not None:
        path = os.path.join(base_path, '{}'.format(trial_id))
        if not os.path.exists(path):
            os.makedirs(path)

        fn = '{}_trial-{:03d}.png'.format(label, trial_id)
        plt.figure(norm_fig.number)
        plt.savefig(os.path.join(path, fn))
        plt.close('all')


def plotKinemSeqs(sample_dicts, trial_ids, base_path):
    f = functools.partial(plotKinemSeq, base_path=base_path)
    utils.evaluate(f, sample_dicts, trial_ids)


def plotKinemSeq(kinem_seq, trial_id, base_path):
    path = os.path.join(base_path, str(trial_id))
    if not os.path.exists(path):
        os.makedirs(path)

    # num_blocks = len(defn.blocks)
    axis_labels = ('x', 'y', 'z')
    sample_fig_axes = [plt.subplots(len(axis_labels), sharex=True, sharey=True)
                       for block in defn.blocks]

    for block_id, sample_seq in kinem_seq.items():
        block_name = defn.blocks[block_id]
        if not sample_seq.any():
            continue

        sample_axes = sample_fig_axes[block_id][1]
        global_timestamps = sample_seq[:,0]
        samples = sample_seq[:,2:5]
        for i, axis in enumerate(sample_axes):
            axis.plot(global_timestamps, samples[:,i], c='k')

    for block_id, (fig, axes) in enumerate(sample_fig_axes):
        block_name = defn.blocks[block_id]
        # Format and set labels for figure
        for label, axis in zip(axis_labels, axes):
            axis.set_ylabel(label)
        axes[2].set_xlabel('time (seconds)')
        fig_title = 'Kinematic samples, {}'.format(block_name)
        axes[0].set_title(fig_title)
        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in axes[:-1]], visible=False)

        # Save figure
        fn = 'kinem-samples_{}.png'.format(block_name)
        plt.figure(fig.number)
        plt.savefig(os.path.join(path, fn))
        plt.close()
