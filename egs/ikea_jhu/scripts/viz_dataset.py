#!/usr/bin/env/python
import os
import glob
import logging

import numpy as np
import pandas as pd
import cv2
import yaml

import _utils as ikea_utils
from mathtools import utils, transformations


AXIS_LEN_WORLD = 0.1  # m

RGB = ((255, 0, 0),
       (0, 255, 0),
       (0, 0, 255))
CYM = ((0, 255, 255),
       (255, 0, 255),
       (255, 255, 0))


logger = logging.getLogger(__name__)


def readFrameFns(fn, names_as_int=False):
    frame_fns = pd.read_csv(fn, header=None).iloc[:, 0].tolist()

    if names_as_int:
        frame_fns = np.array([int(fn.strip('.png').strip('frame')) for fn in frame_fns])

    return frame_fns


def drawAxes(img, origin, axis_endpoints, axis_colors=None, label=None):
    if axis_colors is None:
        axis_colors = (
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255)
        )

    for i, endpoint in enumerate(axis_endpoints):
        img = cv2.line(img, tuple(origin.ravel()), tuple(endpoint.ravel()), axis_colors[i], 5)

    if label is not None:
        bottom_left_corner = (origin - np.array([-10, 10])).ravel()
        font_scale = 1
        white = (255, 255, 255)
        thickness = 1
        cv2.putText(
            img, label, tuple(bottom_left_corner),
            cv2.FONT_HERSHEY_DUPLEX, font_scale, white, thickness
        )

    return img


def markerFrameToWorldFrame(marker_position, marker_orientation_quat, marker_points=None):
    if marker_points is None:
        marker_points = np.array(
            [[0, 0, 0],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            dtype=float
        ) * AXIS_LEN_WORLD

    # quaternion_matrix returns a matrix in homogeneous coordinates
    marker_orientation_mat = transformations.quaternion_matrix(marker_orientation_quat)[:3, :3]

    # orientation_world operates on column vectors, so transpose when we apply
    # it to an array of row vectors
    world_points = np.dot(marker_points, marker_orientation_mat.T) + marker_position

    return world_points


def quantizePoints(points, image_shape):
    quantized = points.astype(int)

    image_height, image_width = image_shape
    quantized[quantized < 0] = 0
    quantized[quantized[:, 0] > image_height, 0] = image_height
    quantized[quantized[:, 1] > image_width, 1] = image_width

    return quantized


def cameraParamsToCv2(camera_params):
    intrinsics = camera_params['intrinsics']
    cv2_intrinsics = {
        'K': np.array(intrinsics['K']),
        'D': np.array(intrinsics['D']),
        'image_shape': (int(intrinsics['height']), int(intrinsics['width']))
    }

    extrinsics = camera_params['extrinsics']
    r_mat = transformations.quaternion_matrix(extrinsics['orientation'])[:3, :3]
    cv2_extrinsics = {
        't_vec': np.array(extrinsics['position']),
        'r_vec': cv2.Rodrigues(r_mat)[0]
    }

    cv2_params = {'intrinsics': cv2_intrinsics, 'extrinsics': cv2_extrinsics}
    return cv2_params


def getFrameFns(sample_frame_idxs, camera_frame_fns):
    camera_frame_index = {}
    for camera_name, cur_frame_index in sample_frame_idxs.iteritems():
        stored_frame_index = camera_frame_index.get(camera_name, None)
        if stored_frame_index is None:
            camera_frame_index[camera_name] = cur_frame_index
        elif stored_frame_index != cur_frame_index:
            raise AssertionError()

    camera_frame_fns = {
        camera_name: camera_frame_fns[camera_name][frame_index]
        for camera_name, frame_index in camera_frame_index.items()
    }

    return camera_frame_fns


def loadFrames(camera_frame_fns, frames_dir, video_id):
    def loadFrame(camera_name, frame_fn):
        cur_video_dir = os.path.join(frames_dir, "{0}_{1}_cam".format(video_id, camera_name))
        frame = cv2.imread(os.path.join(cur_video_dir, frame_fn))
        if frame is None:
            raise AssertionError()
        return frame

    camera_frames = {
        camera_name: loadFrame(camera_name, frame_fn)
        for camera_name, frame_fn in camera_frame_fns.items()
    }

    return camera_frames


def projectAxes(marker_position, marker_orientation, intrinsics, extrinsics):
    # Make origin and axis endpoints from p, q
    axis_points_world = markerFrameToWorldFrame(marker_position, marker_orientation)
    axis_points_px, _ = cv2.projectPoints(
        axis_points_world,
        extrinsics['r_vec'], extrinsics['t_vec'],
        intrinsics['K'], intrinsics['D']
    )
    # FIXME: I don't know why projectPoints returns an array with dimensions
    #   (num_points, 1, num_img_dims(==2)) --- maybe to represent
    #   each coord explicitly as a row vector?
    axis_points_px = axis_points_px.squeeze()

    # Cast to int and project into image dims
    axis_points_px = quantizePoints(axis_points_px, intrinsics['image_shape'])

    origin_px = axis_points_px[0, :]
    axis_endpoints_px = axis_points_px[1:, :]

    return origin_px, axis_endpoints_px


def main(out_dir=None, data_dir=None, frames_dir=None, metadata_parent_dir=None, start_from=None):
    data_dir = os.path.expanduser(data_dir)
    frames_dir = os.path.expanduser(frames_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if metadata_parent_dir is not None:
        metadata_parent_dir = os.path.join(metadata_parent_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    pose_dir = os.path.join(data_dir, 'poses')
    sample_frame_idx_dir = os.path.join(data_dir, 'frame-indices')

    for i, video_dir in enumerate(glob.glob(os.path.join(pose_dir, '*'))):
        if start_from is not None and i < start_from:
            continue

        video_id = os.path.basename(os.path.normpath(video_dir))

        logger.info("VISUALIZING VIDEO {0}: {1}".format(i, video_id))

        out_video_dir = os.path.join(out_dir, 'images', video_id)
        if not os.path.exists(out_video_dir):
            os.makedirs(out_video_dir)

        if metadata_parent_dir is None:
            metadata_dir = os.path.join(video_dir, 'metadata')
        else:
            metadata_dir = os.path.join(metadata_parent_dir, video_id, 'metadata')

        camera_params, bundle_topics, camera_frame_fns = ikea_utils.readMetadata(metadata_dir)

        sample_frame_idxs = pd.read_csv(
            os.path.join(sample_frame_idx_dir, f"{video_id}.csv")
        ).astype(int)

        camera_params = {
            camera_name: cameraParamsToCv2(params)
            for camera_name, params in camera_params.items()
        }

        # First, load all the marker poses into memory
        prev_shape = None
        camera_marker_poses = {}
        for file_path in glob.glob(os.path.join(video_dir, '*.csv')):
            part_name = ikea_utils.basename(file_path)
            pose_seq = pd.read_csv(file_path).to_numpy()
            camera_marker_poses[part_name] = pose_seq

            # Make sure all pose sequences have the same shape
            if prev_shape is not None and pose_seq.shape != prev_shape:
                warn_fmt_str = 'Pose shapes differ: {0} != {1} (truncating to shortest)'
                logger.warning(warn_fmt_str.format(pose_seq.shape, prev_shape))
            if prev_shape is None or pose_seq.shape[0] < prev_shape[0]:
                prev_shape = pose_seq.shape

        num_pose_samples = prev_shape[0]
        camera_marker_poses = {
            key: pose_seq[:num_pose_samples, :]
            for key, pose_seq in camera_marker_poses.items()
        }

        for sample_index in range(num_pose_samples):
            frame_idxs = sample_frame_idxs.iloc[sample_index, :]
            sample_frame_fns = getFrameFns(frame_idxs, camera_frame_fns)
            try:
                sample_frames = loadFrames(sample_frame_fns, frames_dir, video_id)
            except AssertionError:
                continue

            marker_samples = {
                key: pose_seq[sample_index, :]
                for key, pose_seq in camera_marker_poses.items()
            }
            for part_name, sample in marker_samples.items():
                if np.isnan(sample).any():
                    continue

                # Draw each marker's pose to every frame
                for camera_name in sample_frames.keys():
                    intrinsics = camera_params[camera_name]['intrinsics']
                    extrinsics = camera_params[camera_name]['extrinsics']

                    marker_position = sample[:3]
                    marker_orientation = sample[3:]
                    origin_px, axis_endpoints_px = projectAxes(
                        marker_position, marker_orientation,
                        intrinsics, extrinsics
                    )
                    sample_frames[camera_name] = drawAxes(
                        sample_frames[camera_name], origin_px, axis_endpoints_px,
                        axis_colors=RGB, label=part_name
                    )

            output_frame_fn = "{0:06d}.png".format(sample_index)
            output_frame = np.vstack(tuple(sample_frames.values()))
            cv2.imwrite(os.path.join(out_video_dir, output_frame_fn), output_frame)


if __name__ == '__main__':
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
