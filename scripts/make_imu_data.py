import argparse
import os
import itertools

import yaml
import numpy as np
import joblib

from mathtools import utils
from blocks.core import labels
from kinemparse import imu


def imuConnectionLabels(action_seq, rgb_timestamp_seq, imu_timestamp_seq):
    state_seq = labels.parseLabelSeq(None, timestamps=rgb_timestamp_seq, action_seq=action_seq)
    connections = labels.blockConnectionsSeq(state_seq, lower_tri_only=True)
    boundary_times = labels.boundaryTimestampSeq(
        labels.stateBoundarySeq(state_seq, as_numpy=True),
        rgb_timestamp_seq
    )
    imu_labels = labels.resampleLabels(connections, boundary_times, imu_timestamp_seq)
    return imu_labels


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


def main(out_dir=None, data_dir=None, independent_signals=None):
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
        return np.linalg.norm(imu.getImuSamples(x), axis=1)[:, None]
    accel_mag_seqs = tuple(dictToArray(x, transform=norm) for x in accel_seqs)
    gyro_mag_seqs = tuple(dictToArray(x, transform=norm) for x in gyro_seqs)
    accel_corr_seqs = tuple(imu.imuCorr(x, lower_tri_only=True) for x in accel_mag_seqs)
    gyro_corr_seqs = tuple(imu.imuCorr(x, lower_tri_only=True) for x in gyro_mag_seqs)
    imu_timestamp_seqs = utils.batchProcess(makeTimestamps, accel_seqs, gyro_seqs)

    # def blockResting(action_seq, rgb_timestamps):
    #     action_seq = tuple(event for action in action_seq for event in action)
    #     return labels.blockResting(
    #         labels.blockActionSeq(action_seq, rgb_timestamps, action_type='object'),
    #         labels.blockActionSeq(action_seq, rgb_timestamps, action_type='target'),
    #     )
    # imu_is_resting = utils.batchProcess(blockResting, action_seqs, rgb_timestamp_seqs)

    imu_label_seqs = utils.batchProcess(
        imuConnectionLabels, action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs
    )

    if independent_signals:
        num_signals = imu_label_seqs[0].shape[1]

        def validate(seqs):
            return all(seq.shape[1] == num_signals for seq in seqs)
        all_valid = all(validate(x) for x in (accel_corr_seqs, gyro_corr_seqs, imu_label_seqs))
        if not all_valid:
            raise AssertionError("IMU and labels don't all have the same number of sequences")

        trial_ids = tuple(itertools.chain(*((t_id,) * num_signals for t_id in trial_ids)))

        def split(arrays):
            return tuple(itertools.chain(*((col for col in array.T) for array in arrays)))
        accel_corr_seqs = split(accel_corr_seqs)
        gyro_corr_seqs = split(gyro_corr_seqs)
        imu_label_seqs = split(imu_label_seqs)

        imu_sample_seqs = tuple(
            np.column_stack(covs) for covs in zip(accel_corr_seqs, gyro_corr_seqs)
        )
    else:
        imu_sample_seqs = tuple(np.hstack(covs) for covs in zip(accel_corr_seqs, gyro_corr_seqs))

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
