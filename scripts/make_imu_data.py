import argparse
import os

import yaml
import numpy as np
import joblib

from mathtools import utils
from blocks.core import labels
from kinemparse import imu


def imuConnectionLabels(action_seq, rgb_timestamp_seq, imu_timestamp_seq):
    state_seq = labels.parseLabelSeq(None, timestamps=rgb_timestamp_seq, action_seq=action_seq)
    connections = labels.blockComponentSeq(state_seq, lower_tri_only=True)
    boundary_times = labels.boundaryTimestampSeq(
        labels.stateBoundarySeq(state_seq, from_action_start=True, as_numpy=True),
        rgb_timestamp_seq
    )
    imu_labels = labels.resampleLabels(connections, boundary_times, imu_timestamp_seq)
    return imu_labels


def imuActivityLabels(action_seq, rgb_timestamp_seq, imu_timestamp_seq, action_type=None):
    state_seq = labels.parseLabelSeq(
        None, timestamps=rgb_timestamp_seq, action_seq=action_seq,
        structure_change_only=True
    )
    state_seq[-1].end_idx = len(rgb_timestamp_seq)
    action_seq = np.hstack(action_seq)

    rgb_activity = labels.blockActionSeq(
        action_seq, rgb_timestamp_seq, state_seq=state_seq, action_type=None
    ).T
    imu_activity = utils.resampleSeq(rgb_activity, rgb_timestamp_seq, imu_timestamp_seq)
    return imu_activity


def imuComponentLabels(action_seq, rgb_timestamp_seq, imu_timestamp_seq, unique_components):
    assembly_seq = labels.parseLabelSeq(
        None, timestamps=rgb_timestamp_seq, action_seq=action_seq,
        structure_change_only=True
    )
    assembly_seq[-1].end_idx = len(rgb_timestamp_seq) - 1

    # Convert segment indices from RGB video frames to IMU samples
    segment_idxs_rgb = np.array(tuple(map(lambda x: x.getStartEndFrames(), assembly_seq)))
    segment_times = rgb_timestamp_seq[segment_idxs_rgb]
    segment_idxs_imu = np.column_stack(
        tuple(utils.nearestIndices(imu_timestamp_seq, times) for times in segment_times.T)
    )
    segment_idxs_imu[-1, 1] = len(imu_timestamp_seq) - 1

    # Overwrite old RGB start/end indices with new IMU start/end indices
    for (start_idx, end_idx), assembly in zip(segment_idxs_imu, assembly_seq):
        assembly.start_idx = start_idx
        assembly.end_idx = end_idx

    component_seq = tuple(x.getComponents(include_singleton_blocks=True) for x in assembly_seq)

    def gen_labels(component_seq):
        for component in component_seq:
            component_index = unique_components.get(component, None)
            if component_index is None:
                component_index = max(unique_components.values()) + 1
                unique_components[component] = component_index
            yield component_index

    label_seq = np.zeros(len(imu_timestamp_seq), dtype=int)
    for (s_idx, e_idx), c_idx in zip(segment_idxs_imu, gen_labels(component_seq)):
        label_seq[s_idx:e_idx + 1] = c_idx

    return label_seq, assembly_seq


def dictToArray(imu_seqs, transform=None):
    if transform is None:
        return np.hstack(tuple(imu_seqs[i] for i in range(len(imu_seqs))))
    return np.hstack(tuple(transform(imu_seqs[i]) for i in range(len(imu_seqs))))


def makeTimestamps(*imu_dicts):
    def transform(x):
        return imu.getImuGlobalTimestamps(x)[:, None]
    imu_timestamps = tuple(
        dictToArray(imu_dict, transform=transform)
        for imu_dict in imu_dicts
    )
    imu_timestamps = np.column_stack(imu_timestamps)
    return imu_timestamps.mean(axis=1)


def beforeFirstTouch(action_seq, rgb_timestamp_seq, imu_timestamp_seq):
    for action in action_seq:
        first_touch_indices = np.nonzero(action['action'] == 7)[0]
        if first_touch_indices.size:
            first_touch_idx = action['start'][first_touch_indices[0]]
            break
    else:
        logger.warning('No first touch annotation')
        return None

    before_first_touch = imu_timestamp_seq <= rgb_timestamp_seq[first_touch_idx]
    return before_first_touch


def main(
        out_dir=None, data_dir=None,
        output_data=None, magnitude_centering=None, resting_from_gt=None,
        remove_before_first_touch=None, include_signals=None, fig_type=None):
    logger.info(f"Reading from: {data_dir}")
    logger.info(f"Writing to: {out_dir}")

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    if fig_type is None:
        fig_type = 'multi'

    # Load data
    trial_ids = loadVariable('trial_ids')
    accel_seqs = loadVariable('accel_samples')
    gyro_seqs = loadVariable('gyro_samples')
    action_seqs = loadVariable('action_seqs')
    rgb_timestamp_seqs = loadVariable('orig_rgb_timestamps')

    def validate_imu(seqs):
        def is_valid(d):
            return not any(np.isnan(x).any() for x in d.values())
        return np.array([is_valid(d) for d in seqs])
    imu_is_valid = validate_imu(accel_seqs) & validate_imu(gyro_seqs)
    logger.info(
        f"Ignoring {(~imu_is_valid).sum()} invalid IMU sequences "
        f"(of {len(imu_is_valid)} total)"
    )

    def chooseValid(seq):
        return tuple(x for x, is_valid in zip(seq, imu_is_valid) if is_valid)
    trial_ids = chooseValid(trial_ids)
    accel_seqs = chooseValid(accel_seqs)
    gyro_seqs = chooseValid(gyro_seqs)
    action_seqs = chooseValid(action_seqs)
    rgb_timestamp_seqs = chooseValid(rgb_timestamp_seqs)

    def norm(x):
        norm = np.linalg.norm(imu.getImuSamples(x), axis=1)[:, None]
        return norm
    accel_mag_seqs = tuple(map(lambda x: dictToArray(x, transform=norm), accel_seqs))
    gyro_mag_seqs = tuple(map(lambda x: dictToArray(x, transform=norm), gyro_seqs))

    if magnitude_centering == 'across devices':
        if resting_from_gt:
            imu_unused = tuple(map(imu.restingDevices, action_seqs))
        else:
            def resting(gyro_mag_seq):
                device_is_resting = np.array(
                    [imu.deviceRestingFromSignal(seq) for seq in gyro_mag_seq.T]
                )
                if device_is_resting.all():
                    raise AssertionError()
                return device_is_resting
            imu_unused = tuple(map(resting, gyro_mag_seqs))
        accel_mag_seqs = utils.batchProcess(
            imu.centerSignals,
            accel_mag_seqs, imu_unused
        )
        gyro_mag_seqs = utils.batchProcess(
            imu.centerSignals,
            gyro_mag_seqs, imu_unused
        )
    elif magnitude_centering is not None:
        raise AssertionError()

    imu_timestamp_seqs = utils.batchProcess(makeTimestamps, accel_seqs, gyro_seqs)

    if remove_before_first_touch:
        before_first_touch_seqs = utils.batchProcess(
            beforeFirstTouch, action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs
        )

        def clip(signal, bool_array):
            return signal[~bool_array, ...]
        accel_mag_seqs = tuple(
            clip(signal, b) for signal, b in zip(accel_mag_seqs, before_first_touch_seqs)
            if b is not None
        )
        gyro_mag_seqs = tuple(
            clip(signal, b) for signal, b in zip(gyro_mag_seqs, before_first_touch_seqs)
            if b is not None
        )
        imu_timestamp_seqs = tuple(
            clip(signal, b) for signal, b in zip(imu_timestamp_seqs, before_first_touch_seqs)
            if b is not None
        )
        trial_ids = tuple(
            x for x, b in zip(trial_ids, before_first_touch_seqs)
            if b is not None
        )
        action_seqs = tuple(
            x for x, b in zip(action_seqs, before_first_touch_seqs)
            if b is not None
        )
        rgb_timestamp_seqs = tuple(
            x for x, b in zip(rgb_timestamp_seqs, before_first_touch_seqs)
            if b is not None
        )

    if output_data == 'activity':
        accel_feat_seqs = accel_mag_seqs
        gyro_feat_seqs = gyro_mag_seqs
        imu_label_seqs = utils.batchProcess(
            imuActivityLabels,
            action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs,
            static_kwargs={'action_type': None}
        )
    elif output_data == 'connections':
        accel_feat_seqs = tuple(
            imu.imuDiff(x, lower_tri_only=True)
            for x in accel_mag_seqs
        )
        gyro_feat_seqs = tuple(
            imu.imuDiff(x, lower_tri_only=True)
            for x in gyro_mag_seqs
        )
        imu_label_seqs = utils.batchProcess(
            imuConnectionLabels, action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs
        )
    elif output_data == 'components':
        accel_feat_seqs = accel_mag_seqs
        gyro_feat_seqs = gyro_mag_seqs
        unique_components = {frozenset(): 0}
        imu_label_seqs, assembly_seqs = zip(
            *tuple(
                imuComponentLabels(*args, unique_components)
                for args in zip(action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs)
            )
        )
        saveVariable(assembly_seqs, f'assembly_seqs')
        saveVariable(unique_components, f'unique_components')
    else:
        raise AssertionError()

    signals = {'accel': accel_feat_seqs, 'gyro': gyro_feat_seqs}
    if include_signals is None:
        include_signals = tuple(signals.keys())
    signals = tuple(signals[key] for key in include_signals)

    imu_sample_seqs = tuple(np.stack(x, axis=-1) for x in zip(*signals))

    imu.plot_prediction_eg(
        tuple(zip(imu_sample_seqs, imu_label_seqs, trial_ids)), fig_dir,
        fig_type=fig_type, output_data=output_data
    )

    saveVariable(imu_sample_seqs, f'imu_sample_seqs')
    saveVariable(imu_label_seqs, f'imu_label_seqs')
    saveVariable(trial_ids, f'trial_ids')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
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
