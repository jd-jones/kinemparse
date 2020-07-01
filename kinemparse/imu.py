import logging
import os
import functools

import numpy as np
import torch
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


def resampleImuSeq(imu_seq, seq_bounds=None):
    if seq_bounds is None:
        raise NotImplementedError()

    start_time, end_time = seq_bounds
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


def loadImuSampleSeq(corpus, trial_id, sensor_name=None):
    if sensor_name == 'accel':
        loadImuSamples = corpus.readAccelSamples
    elif sensor_name == 'gyro':
        loadImuSamples = corpus.readGyroSamples
    else:
        raise AssertionError()

    metadata = corpus.meta_data[trial_id]
    name_pairs = (
        (imu_id, defn.full_names[metadata[imu_id]])
        for imu_id in defn.imu_ids
        if metadata[imu_id] not in ('UNUSED', 'MISSING')
    )
    id_pairs = ((imu_id, defn.block_ids[name]) for imu_id, name in name_pairs)

    sample_dict = {
        block_id: loadImuSamples(trial_id, imu_id)
        for imu_id, block_id in id_pairs
    }

    return {k: v for k, v in sample_dict.items() if v is not None}


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


def all_array(*args):
    is_array = map(lambda x: isinstance(x, np.ndarray), args)
    return all(is_array)


def makeImuActivityLabels(centered_imu_samples, imu_obj_label_seq, imu_tgt_label_seq):
    """ Filter and join IMU activity labels """

    if all_array(centered_imu_samples, imu_obj_label_seq, imu_tgt_label_seq):
        num_imus = centered_imu_samples.shape[1]

        def revise(labels):
            return np.column_stack(
                tuple(
                    reviseLabels(labels[:, i], centered_imu_samples[:, i])
                    for i in range(num_imus)
                )
            )
        return revise(imu_obj_label_seq) | revise(imu_tgt_label_seq)
    else:
        raise AssertionError()

    # DEPRECATED
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


def rgbIdxsToImuIdxs(assembly_seq, rgb_timestamp_seq, imu_timestamp_seq, action_idxs=False):
    if action_idxs:
        def getFrames(assembly):
            return assembly.getActionStartEndFrames()

        def setFrames(assembly, frames):
            return assembly.setActionStartEndFrames(*frames)
    else:
        def getFrames(assembly):
            return assembly.getStartEndFrames()

        def setFrames(assembly, frames):
            return assembly.setStartEndFrames(*frames)

    # Convert segment indices from RGB video frames to IMU samples
    segment_idxs_rgb = np.array(tuple(map(getFrames, assembly_seq)))
    segment_times = rgb_timestamp_seq[segment_idxs_rgb]
    segment_idxs_imu = np.column_stack(
        tuple(utils.nearestIndices(imu_timestamp_seq, times) for times in segment_times.T)
    )

    if action_idxs:
        segment_idxs_imu[0, :] = 0
    else:
        segment_idxs_imu[-1, 1] = len(imu_timestamp_seq) - 1

    # Overwrite old RGB start/end indices with new IMU start/end indices
    for frames, assembly in zip(segment_idxs_imu, assembly_seq):
        setFrames(assembly, frames)

    return segment_idxs_imu


# --=( ASSEMBLY PARSING )=-----------------------------------------------------
def componentMeans(imu_mag_seq, assembly_components):
    other_indices = {
        block_index: tuple(component - frozenset((block_index,)))
        for component in tuple(assembly_components)
        for i, block_index in enumerate(tuple(component))
    }

    def gen_columns(other_indices):
        for i in sorted(other_indices.keys()):
            if other_indices[i]:
                yield imu_mag_seq[:, other_indices[i]].mean(axis=1)
            else:
                yield np.zeros_like(imu_mag_seq[:, i])

    component_means = np.column_stack(tuple(gen_columns(other_indices)))
    return component_means


def error(imu_mag_seq, assembly_components):
    prediction_seq = componentMeans(imu_mag_seq, assembly_components)
    error = imu_mag_seq - prediction_seq
    return error


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


def restingDevices(action_label_seq):
    action_label_seq = np.hstack(action_label_seq)
    device_is_resting = np.ones(len(defn.blocks), dtype=bool)

    obj_idxs = np.unique(action_label_seq['object'])
    tgt_idxs = np.unique(action_label_seq['target'])
    device_is_resting[obj_idxs[obj_idxs != -1]] = False
    device_is_resting[tgt_idxs[tgt_idxs != -1]] = False

    return device_is_resting


def centerSignals(imu_sample_seq, imu_is_resting=None):
    """ Subtract the mean of the active signals. """

    if isinstance(imu_sample_seq, np.ndarray):
        if imu_is_resting is None:
            imu_is_resting = np.array([deviceRestingFromSignal(seq) for seq in imu_sample_seq.T])
            if imu_is_resting.all():
                raise AssertionError()
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


def imuDiff(signal, lower_tri_only=True):
    max_value = 200
    diffs = max_value / 2 - np.abs(signal[..., None, :] - signal[..., None])
    prods = np.einsum('ti,tj->tij', signal / max_value, signal / max_value)
    diffs = diffs * prods

    if lower_tri_only:
        rows, cols = np.tril_indices(diffs.shape[1], k=-1)
        diffs = diffs[:, rows, cols]

    return diffs


def imuCorr(signal, window_len=5, eps=1e-6, lower_tri_only=True, normalized=True):
    windows = utils.slidingWindow(signal, window_len, stride_len=1, padding='reflect', axis=0)
    # windows = windows - windows.mean(axis=-1, keepdims=True)

    inner_prods = np.einsum('tiw,tjw->tij', windows, windows)
    if normalized:
        norms = np.einsum('tiw,tiw->ti', windows, windows) ** 0.5
        norm_prods = np.einsum('ti,tj->tij', norms, norms)
    else:
        max_val = 200
        norm_prods = np.ones_like(inner_prods) * window_len * max_val
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


def pairwiseFeats(mag_seq):
    mag_pairs = utils.lower_tri(
        np.stack(
            (
                np.broadcast_to(mag_seq[:, :, None], mag_seq.shape + mag_seq.shape[-1:]),
                np.broadcast_to(mag_seq[:, None, :], mag_seq.shape + mag_seq.shape[-1:])
            ),
            axis=1
        )
    ).swapaxes(1, 2)

    mag_corrs = imuDiff(mag_seq, lower_tri_only=True)[..., None]
    pairwise_feats = np.concatenate((mag_corrs, mag_pairs), axis=-1)
    return pairwise_feats


# --=( VISUALIZATION )=-------------------------------------------
def unpack(io_sample, scores_as_inputs=False):
    def toNumpy(x):
        if isinstance(x, torch.Tensor):
            return x.squeeze().cpu().numpy()
        return x

    ret = tuple(map(toNumpy, io_sample))

    if len(ret) == 3:
        inputs, true_labels, ids = ret
        labels = (true_labels,)
        label_names = ('labels',)
    elif len(ret) == 5:
        preds, scores, inputs, true_labels, ids = ret
        labels = (preds, true_labels)
        label_names = ('preds', 'labels')
    else:
        raise AssertionError()

    if scores_as_inputs:
        inputs = scores.T
        labels = tuple(l.T for l in labels)

    return inputs, labels, label_names, ids


def plot_prediction_eg(*args, fig_type=None, **kwargs):
    if fig_type is None:
        return plot_prediction_eg_standard(*args, **kwargs)
    elif fig_type == 'array':
        return plot_prediction_eg_array(*args, **kwargs)
    elif fig_type == 'multi':
        return plot_prediction_eg_multi(*args, **kwargs)


def plot_prediction_eg_array(
        io_history, expt_out_path, output_data=None, tick_names=None, scores_as_inputs=False):
    subplot_width = 12
    subplot_height = 3

    for fig_idx, io_sample in enumerate(io_history):
        inputs, labels, label_names, ids = unpack(io_sample, scores_as_inputs=scores_as_inputs)

        num_axes = 1 + len(labels)

        figsize = (subplot_width, num_axes * subplot_height)
        fig, axes = plt.subplots(num_axes, figsize=figsize, sharex=True)

        inputs = inputs.reshape(-1, inputs.shape[-1])
        # axes[-1].imshow(inputs.T, interpolation='none', aspect='auto')
        axes[-1].imshow(inputs, interpolation='none', aspect='auto')
        axes[-1].set_ylabel('Input')

        for i, (label, label_name) in enumerate(zip(labels, label_names)):
            if label.ndim == 1:
                axes[i].plot(label, label=label_name)
                if tick_names is not None:
                    axes[i].set_yticks(range(len(tick_names)))
                    axes[i].set_yticklabels(tick_names)
            elif label.ndim == 2:
                axes[i].imshow(label, interpolation='none', aspect='auto')
                if tick_names is not None:
                    axes[i].set_yticks(range(len(tick_names)))
                    axes[i].set_yticklabels(tick_names)
            axes[i].set_ylabel(label_name)

        if isinstance(ids, int):
            trial_id = ids
        else:
            if ids.shape == ():
                trial_id = int(ids)
            else:
                if not all(i == ids[0] for i in ids):
                    raise AssertionError()
                trial_id = ids[0]

        plt.tight_layout()
        fig_name = f'trial-{trial_id:03}.png'
        plt.savefig(os.path.join(expt_out_path, fig_name))
        plt.close()


def plot_prediction_eg_multi(io_history, expt_out_path, output_data=None):
    subplot_width = 12
    subplot_height = 2
    num_blocks = 8

    if output_data == 'activity':
        # num_samples_per_fig = num_blocks
        axis_ylabels = tuple(f'mag({i})' for i in range(num_blocks))
    elif output_data == 'connections':
        indices = np.column_stack(np.tril_indices(num_blocks, k=-1))
        # num_samples_per_fig = indices.shape[0]
        axis_ylabels = tuple(f'corr({i}, {j})' for i, j in indices)

    for fig_idx, io_sample in enumerate(io_history):
        inputs, labels, label_names, ids = unpack(io_sample)
        num_seqs = labels[0].shape[1]
        figsize = (subplot_width, num_seqs * subplot_height)
        fig, axes = plt.subplots(num_seqs, figsize=figsize)
        if num_seqs == 1:
            axes = (axes,)
        for i in range(num_seqs):
            axis = axes[i]
            ylabel = axis_ylabels[i]
            input_seq = inputs[:, i]
            label_seqs = tuple(x[:, i] for x in labels)
            _ = plotImu((input_seq,), label_seqs, label_names=label_names, axis=axis)
            axis.set_ylabel(f'seq {ids}; ' + ylabel)
        plt.tight_layout()
        fig_name = f'{fig_idx:03}.png'
        plt.savefig(os.path.join(expt_out_path, fig_name))
        plt.close()


def plot_prediction_eg_standard(io_history, expt_out_path, fig_type=None, output_data=None):
    subplot_width = 12
    subplot_height = 2
    num_blocks = 8

    if output_data == 'activity':
        num_samples_per_fig = num_blocks
        axis_ylabels = tuple(f'mag({i})' for i in range(num_blocks))
    elif output_data == 'connections':
        indices = np.column_stack(np.tril_indices(num_blocks, k=-1))
        num_samples_per_fig = indices.shape[0]
        axis_ylabels = tuple(f'corr({i}, {j})' for i, j in indices)

    s_idxs = tuple(range(0, len(io_history), num_samples_per_fig))
    e_idxs = s_idxs[1:] + (len(io_history),)
    io_histories = tuple(io_history[s_idx:e_idx] for s_idx, e_idx in zip(s_idxs, e_idxs))

    for fig_idx, io_samples in enumerate(io_histories):
        num_seqs = len(io_samples)
        figsize = (subplot_width, num_seqs * subplot_height)
        fig, axes = plt.subplots(num_seqs, figsize=figsize)  # , sharex=True, sharey=True)
        if num_seqs == 1:
            axes = (axes,)
        for axis, io_sample, ylabel in zip(axes, io_samples, axis_ylabels):
            inputs, labels, label_names, ids = unpack(io_sample)
            _ = plotImu((inputs,), labels, label_names=label_names, axis=axis)
            axis.set_ylabel(f'seq {ids}; ' + ylabel)
        plt.tight_layout()

        fig_name = f'{fig_idx:03}.png'
        plt.savefig(os.path.join(expt_out_path, fig_name))
        plt.close()


def plotImu(
        imu_samples, imu_labels,
        imu_timestamps=None, label_timestamps=None,
        label_names=None, axis=None, measurement_name='gyro',
        scale_labels=True):
    """ Plot IMU samples and labels.

    Parameters
    ----------
    imu_samples : iterable(numpy array, dype float)
    imu_labels : iterable(iterable(int))
    imu_timestamps : iterable(numpy array, dype float)
    label_timestamps : iterable(numpy array, dype float)
    label_names : iterable(str)
    axis :
    measurement_name : {'accel', 'gyro'}, optional
        This argument is ignored for any element in imu_samples which has only
        a single column.
        If 'accel', plot first column of each element in `imu_samples`
        If 'gyro', plot second column of each element in `imu_samples`
    scale_labels : bool

    Returns
    -------
    axis :
    """

    if axis is None:
        axis = plt.gca()

    if measurement_name == 'accel':
        sl = slice(0, 1)
    elif measurement_name == 'gyro':
        sl = slice(1, 2)
    else:
        err_str = f'keyword argument measurement_name={measurement_name} is not supported'
        raise ValueError(err_str)

    scale_val = 1

    for s in imu_samples:
        if np.squeeze(imu_samples).ndim > 1:
            sample_norms = s[:, sl]
        else:
            sample_norms = s

        if imu_timestamps is not None:
            axis.plot(imu_timestamps, sample_norms)
        else:
            axis.plot(sample_norms)

        if scale_labels:
            scale_val = max(scale_val, sample_norms.max())

    label_axis = axis.twinx()
    for i, l in enumerate(imu_labels):
        if label_names is not None:
            label_name = label_names[i]
        else:
            label_name = ''
        if label_timestamps is not None:
            # axis.plot(label_timestamps, l * scale_val, ':', label=label_name)
            label_axis.plot(label_timestamps, l, ':', label=label_name)
        else:
            # axis.plot(l * scale_val, ':', label=label_name)
            label_axis.plot(l, ':', label=label_name)
        if label_names is not None:
            label_axis.legend()

    return axis


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
