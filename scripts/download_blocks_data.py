import os
import logging

import joblib
import yaml
import pandas as pd

from blocks.core import duplocorpus, labels
from mathtools import utils
from kinemparse import imu, primesense


logger = logging.getLogger(__name__)


def fixStartEndIndices(action_seq, rgb_frame_timestamp_seq, selected_frame_indices):
    new_times = rgb_frame_timestamp_seq[selected_frame_indices]
    start_times = rgb_frame_timestamp_seq[action_seq['start']]
    end_times = rgb_frame_timestamp_seq[action_seq['end']]

    action_seq['start'] = utils.nearestIndices(new_times, start_times)
    action_seq['end'] = utils.nearestIndices(new_times, end_times)

    return action_seq


def seqBounds(rgb_timestamp_seq, depth_timestamp_seq, accel_seq=[], gyro_seq=[]):
    def getTimestampBounds(timestamp_seq):
        lower = timestamp_seq.min()
        upper = timestamp_seq.max()
        return lower, upper

    start_times = []
    end_times = []

    if rgb_timestamp_seq is not None and not utils.isEmpty(rgb_timestamp_seq):
        lower, upper = getTimestampBounds(rgb_timestamp_seq)
        start_times.append(lower)
        end_times.append(upper)

    if depth_timestamp_seq is not None and not utils.isEmpty(depth_timestamp_seq):
        lower, upper = getTimestampBounds(depth_timestamp_seq)
        start_times.append(lower)
        end_times.append(upper)

    if accel_seq:
        for block_index, imu_sample_seq in accel_seq.items():
            sample_timestamps = imu.getImuGlobalTimestamps(imu_sample_seq)
            lower, upper = getTimestampBounds(sample_timestamps)
            start_times.append(lower)
            end_times.append(upper)

    if gyro_seq:
        for block_index, imu_sample_seq in gyro_seq.items():
            sample_timestamps = imu.getImuGlobalTimestamps(imu_sample_seq)
            lower, upper = getTimestampBounds(sample_timestamps)
            start_times.append(lower)
            end_times.append(upper)

    last_start_time = max(start_times)
    first_end_time = min(end_times)

    if last_start_time > first_end_time:
        warn_str = 'Skipping this sequence -- no overlapping data samples'
        logger.warning(warn_str)
        return None

    return max(start_times), min(end_times)


def loadMetadata(metadata_file, metadata_criteria={}):
    metadata = pd.read_excel(metadata_file, index_col=None)
    metadata = metadata.drop(columns=['Eyetrackingfilename', 'Notes'])
    metadata = metadata.loc[:, ~metadata.columns.str.contains('^Unnamed')]

    metadata = metadata.dropna(subset=['TaskID', 'VidID'])
    for key, value in metadata_criteria.items():
        in_corpus = metadata[key] == value
        metadata = metadata[in_corpus]

    metadata['VidID'] = metadata['VidID'].astype(int)
    metadata['TaskID'] = metadata['TaskID'].astype(int)

    return metadata


def main(
        out_dir=None, corpus_name=None, default_annotator=None,
        metadata_file=None, metadata_criteria={},
        start_from=None, stop_at=None,
        start_video_from_first_touch=None, subsample_period=None,
        modalities=None, use_annotated_keyframes=None, download_gt_keyframes=None):

    if modalities is None:
        modalities = ('video', 'imu')

    out_dir = os.path.expanduser(out_dir)
    metadata_file = os.path.expanduser(metadata_file)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    selected_frame_indices = slice(None, None, subsample_period)

    metadata = loadMetadata(metadata_file, metadata_criteria=metadata_criteria)

    corpus = duplocorpus.DuploCorpus(corpus_name)
    trial_ids = metadata['VidID']

    annotator_names = {}
    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        if start_video_from_first_touch:
            label_seq = corpus.readLabels(trial_id, default_annotator)[0]
            first_touch_idxs = label_seq['start'][label_seq['action'] == 7]
            if not len(first_touch_idxs):
                logger.info(f"  Skipping trial {trial_id}: no first touch annotated")
                continue
            first_touch_idx = int(first_touch_idxs[0])
            # Video is sampled at 30 FPS --> start one second before the first
            # touch was annotated (unless that frame would have happened before
            # the camera started recording)
            start_idx = max(0, first_touch_idx - 30)
            selected_frame_indices = slice(start_idx, None, subsample_period)

        if use_annotated_keyframes:
            annotator = 'Jonathan'
            keyframe_idxs = labels.loadKeyframeIndices(corpus, annotator, trial_id)
            if not keyframe_idxs:
                logger.info(f"  Skipping trial {trial_id}: no labeled keyframes")
                continue
            selected_frame_indices = keyframe_idxs

        logger.info(f"  Loading labels...")
        action_seq, annotator_name, is_valid = corpus.readLabels(trial_id, default_annotator)
        if not is_valid:
            logger.info(f"    Skipping trial {trial_id}: No labels")
            continue

        annotator_names[trial_id] = annotator_name

        rgb_frame_fn_seq = corpus.getRgbFrameFns(trial_id)
        if rgb_frame_fn_seq is None:
            logger.info(f"    Skipping trial {trial_id}: No RGB frames")
            continue

        rgb_frame_timestamp_seq = corpus.readRgbTimestamps(trial_id, times_only=True)
        if rgb_frame_timestamp_seq is None:
            logger.info(f"    Skipping trial {trial_id}: No RGB timestamps")
            continue

        depth_frame_fn_seq = corpus.getDepthFrameFns(trial_id)
        if depth_frame_fn_seq is None:
            logger.info(f"    Skipping trial {trial_id}: No depth frames")
            continue

        depth_frame_timestamp_seq = corpus.readDepthTimestamps(trial_id, times_only=True)
        if depth_frame_timestamp_seq is None:
            logger.info(f"    Skipping trial {trial_id}: No depth timestamps")
            continue

        if action_seq['start'].max() >= len(rgb_frame_fn_seq):
            logger.info(f"    Skipping trial {trial_id}: actions longer than #rgb frames")
            continue

        if action_seq['end'].max() >= len(rgb_frame_fn_seq):
            logger.info(f"    Skipping trial {trial_id}: actions longer than #rgb frames")
            continue

        action_seq = fixStartEndIndices(
            action_seq, rgb_frame_timestamp_seq, selected_frame_indices
        )
        assembly_seq, is_valid = labels.constructStateSeq(
            trial_id, labels.constructActionSeq(trial_id, action_seq)[0],
            structure_change_only=True, actions_as_states=False
        )
        if not is_valid:
            logger.info(f"    Skipping trial {trial_id}: Bad labels")
            continue

        rgb_frame_timestamp_seq = rgb_frame_timestamp_seq[selected_frame_indices]
        depth_frame_timestamp_seq = depth_frame_timestamp_seq[selected_frame_indices]

        # trial_str = f"trial={trial_id}"
        trial_str = f"trial-{trial_id}"
        if 'imu' in modalities:
            logger.info(f"  Loading and saving IMU data...")

            accel_seq = imu.loadImuSampleSeq(corpus, trial_id, sensor_name='accel')
            gyro_seq = imu.loadImuSampleSeq(corpus, trial_id, sensor_name='gyro')

            if not accel_seq:
                logger.info(f"    Skipping trial {trial_id}: Missing accel features")
                continue

            if not gyro_seq:
                logger.info(f"    Skipping trial {trial_id}: Missing gyro features")
                continue

            imu_bounds = seqBounds(None, None, accel_seq=accel_seq, gyro_seq=gyro_seq)
            if imu_bounds is None:
                logger.info(f"    Skipping trial {trial_id}: No overlapping data samples")
                continue

            accel_seq = imu.resampleImuSeq(accel_seq, seq_bounds=imu_bounds)
            gyro_seq = imu.resampleImuSeq(gyro_seq, seq_bounds=imu_bounds)

            saveToWorkingDir(accel_seq, f'{trial_str}_accel-samples')
            saveToWorkingDir(gyro_seq, f'{trial_str}_gyro-samples')

        if 'video' in modalities:
            logger.info(f"  Loading and saving video data...")

            rgb_frame_seq = primesense.loadRgbFrameSeq(
                rgb_frame_fn_seq, rgb_frame_timestamp_seq,
                stack_frames=True
            )[selected_frame_indices]
            saveToWorkingDir(rgb_frame_seq, f'{trial_str}_rgb-frame-seq')

            depth_frame_seq = primesense.loadDepthFrameSeq(
                depth_frame_fn_seq, depth_frame_timestamp_seq,
                stack_frames=True
            )[selected_frame_indices]
            saveToWorkingDir(depth_frame_seq, f'{trial_str}_depth-frame-seq')

        # saveToWorkingDir(rgb_frame_fn_seq, f'{trial_str}_rgb-frame-fn-seq')
        # saveToWorkingDir(rgb_frame_timestamp_seq, f'{trial_str}_rgb-frame-timestamp-seq')
        # saveToWorkingDir(depth_frame_fn_seq, f'{trial_str}_depth-frame-fn-seq')
        # saveToWorkingDir(depth_frame_timestamp_seq, f'{trial_str}_depth-frame-timestamp-seq')
        # saveToWorkingDir(action_seq, f'{trial_str}_action-seq')

        if download_gt_keyframes:
            annotator = 'Jonathan'
            keyframe_idxs = labels.loadKeyframeIndices(corpus, annotator, trial_id)
            if not keyframe_idxs.any():
                continue
            if start_video_from_first_touch:
                raise NotImplementedError()
            if subsample_period is not None:
                keyframe_idxs = utils.roundToInt(keyframe_idxs / subsample_period)
            saveToWorkingDir(keyframe_idxs, f'{trial_str}_gt-keyframe-seq')

    trial_ids = sorted(annotator_names.keys())
    annotator_names = [annotator_names[t_id] for t_id in trial_ids]
    annotator_names = pd.DataFrame({'trial_id': trial_ids, 'annotator_name': annotator_names})
    annotator_names.to_csv(os.path.join(out_data_dir, 'annotator_names.csv'), index=False)


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
