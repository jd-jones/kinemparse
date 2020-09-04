import os
import logging
import glob

import yaml
import cv2
import numpy as np

from mathtools import utils
from kinemparse import airplanecorpus


logger = logging.getLogger(__name__)


def genFrames(video_fn):
    cap = cv2.VideoCapture(video_fn)

    if not cap.isOpened():
        raise FileNotFoundError()

    frame_status, frame = cap.read()
    while(frame_status):
        yield frame
        frame_status, frame = cap.read()

    cap.release()


def readVideo(video_fn):
    frames = genFrames(video_fn)
    frames = np.stack(tuple(frames), axis=0)
    return frames


def writeVideo(frames, video_fn, frame_rate=30):
    frame_shape = tuple(int(i) for i in reversed(frames[0].shape[:2]))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_fn, fourcc, frame_rate, frame_shape, True)

    for frame in frames:
        out.write(frame)

    out.release()


def vizDetection(frame, detection, bounding_box_size=(10, 10), bounding_box_color=(0, 255, 0)):
    x, y = tuple(int(x) for x in detection)
    w, h = bounding_box_size
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def main(
        out_dir=None,
        videos_dir=None, hand_detections_dir=None,
        viz_params={},
        plot_output=None, results_file=None, sweep_param_name=None):

    videos_dir = os.path.expanduser(videos_dir)
    hand_detections_dir = os.path.expanduser(hand_detections_dir)

    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
        # write_mode = 'w'
    else:
        results_file = os.path.expanduser(results_file)
        # write_mode = 'a'

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    for i, video_fn in enumerate(glob.glob(os.path.join(videos_dir, '*.avi'))):
        if i:
            break

        video_id = utils.stripExtension(video_fn)

        logger.info(f"Processing video {video_id}")

        frames = readVideo(video_fn)
        hand_detections = airplanecorpus.loadHandDetections(
            video_id, dir_name=hand_detections_dir, unflatten=True
        )

        for detections, frame in zip(hand_detections, frames):
            for detection in detections:
                if np.isnan(detection).any():
                    continue
                vizDetection(frame, detection, **viz_params)

        video_fn = os.path.join(fig_dir, f'{video_id}.avi')
        writeVideo(frames, video_fn)


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
