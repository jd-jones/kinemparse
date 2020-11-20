import os

import joblib
import yaml
import numpy as np
import torch
import torchgeometry
from skimage import measure, feature, transform

from mathtools import utils, torchutils
from visiontools import render, imageprocessing, geometry


def maskLowerLeftBorder(
        depth_image,
        background_mask,
        sigma=1, l_thresh=0, h_thresh=1000,
        axis_tol=0.1, hough_thresh_ratio=0.4,
        x_max_thresh=0.1, y_min_thresh=0.75, margin=10):
    """ Return a mask array that identifies the background of the input image.

    Parameters
    ----------
    depth_image : numpy array of float, shape (img_height, img_width)
    sigma : float, optional
    l_thresh : int, optional
    h_thresh : int, optional
    axis_tol : float, optional
    hough_thresh_ratio : float, optional
    x_max_thresh : float, optional
        If a vertical line is detected to the left of `x_max_thresh` in the
        image, this function masks the left side of the output up to that line.
        You can disable this masking by setting `x_max_thresh` < 0.
    y_min_thresh : float, optional
        If a horizontal line is detected below `y_min_thresh` in the
        image, this function masks the bottom of the output up to that line.
        You can disable this masking by setting `y_min_thresh` > 1.

    Returns
    -------
    background_mask : numpy array of bool, shape (img_height, img_width)
    """

    # Find rudimentary edges in the image
    masked_image = depth_image.copy()
    masked_image[background_mask] = 0
    edge_image = feature.canny(
        masked_image,
        sigma=sigma,
        low_threshold=l_thresh,
        high_threshold=h_thresh
    )

    num_rows, num_cols = edge_image.shape

    x_mid = num_cols / 2
    y_mid = num_rows / 2
    x_max = x_max_thresh * num_cols
    y_min = y_min_thresh * num_rows

    hough_mask = np.zeros_like(edge_image, dtype=bool)

    # Detect lines using the Hough transform
    h, theta, d = transform.hough_line(edge_image)
    __, angles, dists = transform.hough_line_peaks(
        h, theta, d,
        threshold=hough_thresh_ratio * h.max()
    )

    # Filter Hough lines and mask the border if appropriate
    for angle, dist in zip(angles, dists):
        if geometry.axisAligned(angle, tol=axis_tol, axis='horizontal'):
            y = geometry.solveLine(angle, dist, x=x_mid)
            if y > y_min:
                hough_mask[int(y) - margin:, :] = True
        elif geometry.axisAligned(angle, tol=axis_tol, axis='vertical'):
            x = geometry.solveLine(angle, dist, y=y_mid)
            if x < x_max:
                hough_mask[:, :int(x) + margin] = True
        else:
            continue

    return hough_mask


def fitBackgroundDepth(
        depth_image, camera_params=None, camera_pose=None,
        plane_distance_thresh=10, **ransac_kwargs):
    """ Fit a plane to a depth image using RANSAC.

    Parameters
    ----------
    camera_image : numpy array of float, shape (img_height, img_width)
    is_inlier : numpy array of bool, shape (img_height, img_width)
    residual_threshold : float
        Required by RANSAC. RANSAC classifies any point with residual error
        greater than this threshold as an outlier.
    **ransac_kwargs : optional
        Any extra keyword arguments are passed to `skimage.measure.ransac`.

    Returns
    -------
    plane : geometry.Plane
    plane_distance_image : numpy array of float, shape (img_height, img_width)
    """

    if camera_params is None:
        camera_params = render.intrinsic_matrix

    if camera_pose is None:
        camera_pose = render.camera_pose

    # Backproject the depth image to world coordinates
    is_inlier = ~imageprocessing.maskDepthArtifacts(depth_image)
    inlier_depth_coords = depth_image[is_inlier]
    inlier_px_coords = np.column_stack(np.nonzero(is_inlier)[1:])
    camera_coords = imageprocessing.backprojectPixels(
        camera_params, camera_pose, np.flip(inlier_px_coords, axis=1), inlier_depth_coords,
        in_camera_coords=True
    )

    # Fit plane
    plane, ransac_inliers = measure.ransac(
        camera_coords, geometry.Plane, 3, plane_distance_thresh,
        **ransac_kwargs
    )

    plane_distance = plane.residuals(camera_coords)
    plane_distance_image = imageprocessing.imageFromForegroundPixels(plane_distance, is_inlier)
    background_mask = plane_distance_image < plane_distance_thresh

    mean_image = depth_image.mean(axis=0)
    mean_bg = background_mask.astype(float).mean(axis=0) < 0.5
    llb_mask = maskLowerLeftBorder(mean_image, mean_bg)[None, ...]
    background_mask |= llb_mask

    return plane, background_mask


def detectBackgroundDepth(
        plane, depth_image, plane_distance_thresh=10,
        camera_params=None, camera_pose=None, **ransac_kwargs):
    if camera_params is None:
        camera_params = render.intrinsic_matrix

    if camera_pose is None:
        camera_pose = render.camera_pose

    # Backproject the depth image to world coordinates
    is_inlier = ~imageprocessing.maskDepthArtifacts(depth_image)
    inlier_depth_coords = depth_image[is_inlier]
    inlier_px_coords = np.column_stack(np.nonzero(is_inlier)[1:])
    camera_coords = imageprocessing.backprojectPixels(
        camera_params, camera_pose, np.flip(inlier_px_coords, axis=1), inlier_depth_coords,
        in_camera_coords=True
    )

    plane_distance = plane.residuals(camera_coords)
    plane_distance_image = imageprocessing.imageFromForegroundPixels(plane_distance, is_inlier)
    background_mask = plane_distance_image < plane_distance_thresh

    mean_image = depth_image.mean(axis=0)
    mean_bg = background_mask.astype(float).mean(axis=0) < 0.5
    llb_mask = maskLowerLeftBorder(mean_image, mean_bg)[None, ...]
    background_mask |= llb_mask

    return background_mask


def detectBackgroundDepth_deprecated(
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

    background_plane, plane_distance_image = fitBackgroundDepth(
        depth_image, ~background_mask, plane_distance_thresh,
        camera_params=camera_params, camera_pose=camera_pose,
        max_trials=50
    )
    background_mask |= plane_distance_image < plane_distance_thresh

    background_mask = imageprocessing.makeBackgroundMask(depth_image, background_mask)

    if mask_left_side:
        background_mask = imageprocessing.maskOutsideBuildArea(
            background_mask, mask_left_side=mask_left_side, mask_bottom=0
        )

    return background_mask, background_plane


def fitBackgroundRgb(rgb_frame_seq, bg_mask_seq_depth):
    background = np.ma.array(
        rgb_frame_seq,
        mask=np.broadcast_to((~bg_mask_seq_depth)[..., None], rgb_frame_seq.shape)
    ).mean(axis=0).data

    return background


def detectBackgroundRgb(rgb_frame_seq, background, px_distance_thresh=0.2):
    bg_dists = np.linalg.norm(rgb_frame_seq - background[None, ...], axis=-1)
    bg_mask_seq_rgb = bg_dists < px_distance_thresh

    return bg_mask_seq_rgb


class RgbBackgroundModel(torch.nn.Module):
    def __init__(
            self, background_image, kernel_size=1,
            update_bg=False, device=None, detect_mode=True):
        super().__init__()

        self.detect_mode = detect_mode

        self.device = device

        self._bg = torch.nn.Parameter(
            torch.tensor(
                background_image, dtype=torch.float, device=self.device
            ).permute(-1, 0, 1),
            requires_grad=update_bg
        )
        if self.detect_mode:
            self._conv = torch.nn.Conv2d(
                1, 1, kernel_size=kernel_size, padding=kernel_size // 2
            ).to(device=device)
            self._conv.weight = torch.nn.Parameter(
                torch.ones_like(self._conv.weight),
                requires_grad=True
            )
            self._conv.bias = torch.nn.Parameter(
                torch.zeros_like(self._conv.bias),
                requires_grad=True
            )
        else:
            self._conv = torch.nn.Conv2d(
                1, 2, kernel_size=kernel_size, padding=kernel_size // 2
            ).to(device=device)

    def forward(self, inputs):
        # diffs = 1 - torch.abs(inputs - self._bg[None, ...])
        dists = torch.norm(inputs - self._bg[None, ...], dim=1, keepdim=True)
        # smoothed = self._conv(dists)
        return dists  # smoothed

    def fit(self, inputs, targets, num_epochs=1):
        batch_size = 500
        inputs = inputs[:batch_size]
        targets = targets[:batch_size]

        if self.detect_mode:
            criterion = torch.nn.BCEWithLogitsLoss()
            targets_dtype = torch.float
        else:
            # criterion = torch.nn.CrossEntropyLoss()
            criterion = torchgeometry.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='sum')
            targets_dtype = torch.long

        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )

        inputs = torch.tensor(
            inputs, dtype=torch.float, device=self.device
        ).permute(0, -1, 1, 2)

        if self.detect_mode:
            targets = torch.tensor(
                targets, dtype=targets_dtype, device=self.device
            ).view(targets.shape[0], -1)
        else:
            targets = torch.tensor(
                targets, dtype=targets_dtype, device=self.device
            )

        losses = []
        metrics = []
        for i in range(num_epochs):
            outputs = self.forward(inputs)
            preds = self.predict(outputs)

            if self.detect_mode:
                outputs = outputs.view(outputs.shape[0], -1)
                preds = preds.view(preds.shape[0], -1)
            else:
                # outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
                # preds = preds.view(preds.shape[0], -1)
                pass

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.detect_mode:
                mean_acc = meanAccuracy(preds, targets.to(dtype=torch.uint8))
            else:
                mean_acc = meanAccuracy(preds, targets)

            losses.append(loss.item())
            metrics.append(mean_acc.item())

        return losses, metrics

    def predict(self, outputs, thresh=0.5):
        if self.detect_mode:
            probs = torch.sigmoid(outputs)
            preds = probs > thresh
        else:
            _, preds = torch.max(outputs, dim=1)
        return preds


def meanAccuracy(preds, targets):
    matches = preds == targets
    mean_acc = torch.sum(matches).float() / float(torch.numel(matches))
    return mean_acc


def main(
        out_dir=None, data_dir=None, background_data_dir=None, learn_bg_model=False,
        gpu_dev_id=None, start_from=None, stop_at=None, num_disp_imgs=None,
        depth_bg_detection_kwargs={}, rgb_bg_detection_kwargs={}):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    background_data_dir = os.path.expanduser(background_data_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    device = torchutils.selectDevice(gpu_dev_id)

    camera_pose = render.camera_pose
    camera_params = render.intrinsic_matrix

    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        trial_str = f"trial={trial_id}"

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info("  Loading data...")
        try:
            rgb_frame_seq = loadFromDir(f"{trial_str}_rgb-frame-seq", data_dir)
            depth_frame_seq = loadFromDir(f"{trial_str}_depth-frame-seq", data_dir)
            rgb_train = loadFromDir(
                f"{trial_str}_rgb-frame-seq-before-first-touch",
                background_data_dir
            )
            depth_train = loadFromDir(
                f"{trial_str}_depth-frame-seq-before-first-touch",
                background_data_dir
            )
        except FileNotFoundError as e:
            logger.info(e)
            continue

        logger.info("  Removing background...")

        try:
            bg_mask_depth_train = loadFromDir(f'{trial_str}_bg-mask-depth-train', out_data_dir)
            bg_mask_seq_depth = loadFromDir(f'{trial_str}_bg-mask-seq-depth', out_data_dir)
        except FileNotFoundError:
            bg_model_depth, bg_mask_depth_train = fitBackgroundDepth(
                depth_train, camera_params=camera_params, camera_pose=camera_pose,
                **depth_bg_detection_kwargs
            )

            bg_mask_seq_depth = detectBackgroundDepth(
                bg_model_depth, depth_frame_seq,
                camera_params=camera_params, camera_pose=camera_pose,
                **depth_bg_detection_kwargs
            )

            __, bg_model_depth_image, __ = render.renderPlane(
                bg_model_depth, camera_pose=camera_pose, camera_params=camera_params
            )
            imageprocessing.displayImage(
                bg_model_depth_image,
                file_path=os.path.join(fig_dir, f'{trial_str}_bg-image-depth.png')
            )

        bg_model_rgb = fitBackgroundRgb(
            np.vstack((rgb_train, rgb_frame_seq)),
            np.vstack((bg_mask_depth_train, bg_mask_seq_depth))
        )

        if learn_bg_model:
            model = RgbBackgroundModel(bg_model_rgb, update_bg=True, device=device)
            losses, metrics = model.fit(
                np.vstack((rgb_train, rgb_frame_seq)),
                np.vstack((bg_mask_depth_train, bg_mask_seq_depth)),
                num_epochs=100
            )

            outputs = model.forward(
                torch.tensor(rgb_frame_seq, dtype=torch.float, device=device).permute(0, -1, 1, 2)
            )
            bg_mask_seq_rgb = model.predict(outputs).cpu().numpy().squeeze()
            plot_dict = {'Loss': losses, 'Accuracy': metrics},
        else:
            def f1(preds, targets):
                true_positives = np.sum((targets == 1) * (preds == 1))
                false_positives = np.sum((targets == 0) * (preds == 1))
                false_negatives = np.sum((targets == 1) * (preds == 0))

                precision = true_positives / (true_positives + false_positives)
                recall = true_positives / (true_positives + false_negatives)

                f1 = 2 * (precision * recall) / (precision + recall)
                return f1

            def acc(preds, targets):
                matches = preds == targets
                return matches.mean()

            bg_dists = np.linalg.norm(rgb_frame_seq - bg_model_rgb[None, ...], axis=-1)
            thresh_vals = np.linspace(0, 1, num=50)
            scores = np.array([acc(bg_dists < t, bg_mask_seq_depth) for t in thresh_vals])
            best_index = scores.argmax()
            best_thresh = thresh_vals[best_index]
            bg_mask_seq_rgb = bg_dists < best_thresh
            plot_dict = {'Accuracy': scores}

        torchutils.plotEpochLog(
            plot_dict,
            subfig_size=(10, 2.5),
            title='Training performance',
            fn=os.path.join(fig_dir, f'{trial_str}_train-plot.png')
        )

        logger.info("  Saving output...")
        saveToWorkingDir(bg_mask_depth_train.astype(bool), f'{trial_str}_bg-mask-depth-train')
        saveToWorkingDir(bg_mask_seq_depth.astype(bool), f'{trial_str}_bg-mask-seq-depth')
        saveToWorkingDir(bg_mask_seq_rgb.astype(bool), f'{trial_str}_bg-mask-seq-rgb')

        if num_disp_imgs is not None:
            if rgb_frame_seq.shape[0] > num_disp_imgs:
                idxs = np.arange(rgb_frame_seq.shape[0])
                np.random.shuffle(idxs)
                idxs = idxs[:num_disp_imgs]
            else:
                idxs = slice(None, None, None)
            imageprocessing.displayImages(
                *(rgb_frame_seq[idxs]), *(bg_mask_seq_rgb[idxs]),
                *(depth_frame_seq[idxs]), *(bg_mask_seq_depth[idxs]),
                num_rows=4, file_path=os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            )
            imageprocessing.displayImage(
                bg_model_rgb, file_path=os.path.join(fig_dir, f'{trial_str}_bg-image-rgb.png')
            )


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
