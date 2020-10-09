import os
import logging

import yaml
import numpy as np
import joblib

from mathtools import utils
from blocks.core import labels
from kinemparse import imu


logger = logging.getLogger(__name__)


def parseActions(action_seq, rgb_timestamp_seq, imu_timestamp_seq):
    assembly_seq = labels.parseLabelSeq(
        # None, timestamps=rgb_timestamp_seq, action_seq=action_seq,
        action_seq, timestamps=rgb_timestamp_seq,
        structure_change_only=True
    )
    assembly_seq[-1].end_idx = len(rgb_timestamp_seq) - 1

    _ = imu.rgbIdxsToImuIdxs(
        assembly_seq, rgb_timestamp_seq, imu_timestamp_seq,
        action_idxs=False
    )

    _ = imu.rgbIdxsToImuIdxs(
        assembly_seq, rgb_timestamp_seq, imu_timestamp_seq,
        action_idxs=True
    )

    return assembly_seq


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
    first_touch_indices = np.nonzero(action_seq['action'] == 7)[0]
    if first_touch_indices.size:
        first_touch_idx = action_seq['start'][first_touch_indices[0]]
    else:
        logger.warning('No first touch annotation')
        return None

    before_first_touch = imu_timestamp_seq <= rgb_timestamp_seq[first_touch_idx]
    return before_first_touch


def main(
        out_dir=None, data_dir=None, use_vid_ids_from=None,
        output_data=None, magnitude_centering=None, resting_from_gt=None,
        remove_before_first_touch=None, include_signals=None, fig_type=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    logger.info(f"Reading from: {data_dir}")
    logger.info(f"Writing to: {out_dir}")

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadAll(seq_ids, var_name):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    if fig_type is None:
        fig_type = 'multi'

    # Load data
    if use_vid_ids_from is None:
        trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)
    else:
        use_vid_ids_from = os.path.expanduser(use_vid_ids_from)
        trial_ids = utils.getUniqueIds(use_vid_ids_from, prefix='trial-', to_array=True)
    if not trial_ids.any():
        logger.warning("No filenames matched the trial ID string!")

    accel_seqs = loadAll(trial_ids, 'accel-samples.pkl')
    gyro_seqs = loadAll(trial_ids, 'gyro-samples.pkl')
    action_seqs = loadAll(trial_ids, 'action-seq.pkl')
    rgb_timestamp_seqs = loadAll(trial_ids, 'rgb-frame-timestamp-seq.pkl')

    def validate_imu(seqs):
        def is_valid(d):
            return not any(np.isnan(x).any() for x in d.values())
        return np.array([is_valid(d) for d in seqs])

    imu_is_valid = validate_imu(accel_seqs) & validate_imu(gyro_seqs)
    logger.info(
        f"Ignoring {(~imu_is_valid).sum()} IMU sequences with NaN-valued samples "
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

    imu_timestamp_seqs = utils.batchProcess(makeTimestamps, accel_seqs, gyro_seqs)

    if remove_before_first_touch:
        before_first_touch_seqs = utils.batchProcess(
            beforeFirstTouch, action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs
        )

        num_ignored = sum(b is None for b in before_first_touch_seqs)
        logger.info(
            f"Ignoring {num_ignored} sequences without first-touch annotations "
            f"(of {len(before_first_touch_seqs)} total)"
        )
        trials_missing_first_touch = [
            i for b, i in zip(before_first_touch_seqs, trial_ids)
            if b is None
        ]
        logger.info(f"Trials without first touch: {trials_missing_first_touch}")

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

    assembly_seqs = utils.batchProcess(
        parseActions,
        action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs
    )

    if output_data == 'components':
        accel_feat_seqs = accel_mag_seqs
        gyro_feat_seqs = gyro_mag_seqs
        unique_components = {frozenset(): 0}
        imu_label_seqs = zip(
            *tuple(
                labels.componentLabels(*args, unique_components)
                for args in zip(action_seqs, rgb_timestamp_seqs, imu_timestamp_seqs)
            )
        )
        saveVariable(unique_components, 'unique_components')
    elif output_data == 'pairwise components':
        imu_label_seqs = utils.batchProcess(
            labels.pairwiseComponentLabels, assembly_seqs,
            static_kwargs={'lower_tri_only': True}
        )
        accel_feat_seqs = tuple(map(imu.pairwiseFeats, accel_mag_seqs))
        gyro_feat_seqs = tuple(map(imu.pairwiseFeats, gyro_mag_seqs))
    else:
        raise AssertionError()

    signals = {'accel': accel_feat_seqs, 'gyro': gyro_feat_seqs}
    if include_signals is None:
        include_signals = tuple(signals.keys())
    signals = tuple(signals[key] for key in include_signals)
    imu_feature_seqs = tuple(np.stack(x, axis=-1).squeeze(axis=-1) for x in zip(*signals))

    video_seqs = tuple(zip(imu_feature_seqs, imu_label_seqs, trial_ids))
    imu.plot_prediction_eg(video_seqs, fig_dir, fig_type=fig_type, output_data=output_data)

    video_seqs = tuple(
        zip(assembly_seqs, imu_feature_seqs, imu_timestamp_seqs, imu_label_seqs, trial_ids)
    )
    for assembly_seq, feature_seq, timestamp_seq, label_seq, trial_id in video_seqs:
        id_string = f"trial={trial_id}"
        saveVariable(assembly_seq, f'{id_string}_assembly-seq')
        saveVariable(feature_seq, f'{id_string}_feature-seq')
        saveVariable(timestamp_seq, f'{id_string}_timestamp-seq')
        saveVariable(label_seq, f'{id_string}_label-seq')


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
