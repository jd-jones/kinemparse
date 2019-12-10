import logging

import numpy as np
import torch
import scipy

from mathtools import utils, torchutils
from visiontools import geometry, render, imageprocessing
import neural_renderer as nr


logger = logging.getLogger(__name__)


class TorchSceneRenderer(nr.Renderer):
    def __init__(
            self, intrinsic_matrix=None, camera_pose=None, colors=None, image_size=None,
            **super_kwargs):

        if not isinstance(intrinsic_matrix, torch.Tensor):
            intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=torch.float)

        if not isinstance(camera_pose, torch.Tensor):
            camera_pose = torch.tensor(camera_pose, dtype=torch.float)

        if not isinstance(colors, torch.Tensor):
            colors = torch.tensor(colors, dtype=torch.float)

        K = intrinsic_matrix
        K = K[None, :, :].cuda()

        R, t = geometry.fromHomogeneous(camera_pose)
        R = R[None, :, :].float().cuda()
        t = t[None, None, :].float().cuda()

        self.colors = colors
        self.image_size = image_size

        super().__init__(
            camera_mode='projection', K=K, R=R, t=t, orig_size=max(self.image_size),
            near=0, far=1000, **super_kwargs
        )

    def render(self, vertices, faces, textures, intrinsic_matrix=None, camera_pose=None):
        """ Wrapper around a differentiable renderer implemented in pytorch.

        Parameters
        ----------

        Returns
        -------
        image
        """

        if intrinsic_matrix is None:
            K = None
        else:
            K = intrinsic_matrix
            K = K[None, :, :].cuda()

        if camera_pose is None:
            R = None
            t = None
        else:
            R, t = geometry.fromHomogeneous(camera_pose)
            R = R[None, :, :].float().cuda()
            t = t[None, None, :].float().cuda()

        if len(vertices.shape) == 2:
            # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
            vertices = vertices[None, ...]

        if len(faces.shape) == 2:
            # [num_faces, 3] -> [batch_size=1, num_faces, 3]
            faces = faces[None, ...]

        if len(textures.shape) == 5:
            textures = textures[None, ...]

        images_rgb, images_depth, images_alpha = super().render(vertices, faces, textures)

        # [batch_size, RGB, image_size, image_size] -> [batch_size, image_size, image_size, RGB]
        images_rgb = images_rgb.permute(0, 2, 3, 1)

        # Crop square image back to the original aspect ratio
        images_rgb = images_rgb[:, :self.image_size[0], :self.image_size[1]]
        images_depth = images_depth[:, :self.image_size[0], :self.image_size[1]]

        return images_rgb, images_depth

    def renderScene(
            self, assembly, component_poses,
            rgb_background=None, depth_background=None,
            camera_pose=None, camera_params=None,
            as_numpy=False):
        """ Render a scene consisting of a spatial assembly and a background plane.

        Parameters
        ----------

        Returns
        -------
        """

        if camera_pose is None:
            camera_pose = geometry.homogeneousMatrix(self.R[0], self.t[0][0])

        if camera_params is None:
            camera_params = self.K[0]

        if rgb_background is None:
            rgb_background = torch.zeros(*self.image_size, 3)

        if depth_background is None:
            depth_background = torch.full(self.image_size, float('inf'))

        if not assembly.blocks:
            return rgb_background, depth_background

        assembly = assembly.setPose(component_poses, in_place=False)

        vertices = torchutils.makeBatch(assembly.vertices, dtype=torch.float).cuda()
        faces = torchutils.makeBatch(assembly.faces, dtype=torch.int).cuda()
        textures = torchutils.makeBatch(assembly.textures, dtype=torch.float).cuda()

        rgb_images, depth_images = self.render(vertices, faces, textures)

        rgb_images = torch.cat((rgb_background, rgb_images), 0)
        depth_images = torch.cat((depth_background, depth_images), 0)

        rgb_image, depth_image, label_image = render.reduceByDepth(rgb_images, depth_images)

        if as_numpy:
            rgb_image = rgb_image.detach().cpu().numpy()
            depth_image = depth_image.detach().cpu().numpy()
            label_image = label_image.detach().cpu().numpy()

        return rgb_image, depth_image, label_image

    def renderPlane(self, plane, camera_pose=None, camera_params=None):
        if camera_pose is None:
            camera_pose = geometry.homogeneousMatrix(self.R[0], self.t[0][0])

        if camera_params is None:
            camera_params = self.K[0]

        vertices, faces = render.planeVertices(plane, camera_pose, camera_params)
        textures = render.makeTextures(faces, uniform_color=self.colors['black'])
        rgb_image, depth_image = self.render(vertices, faces, textures)

        return rgb_image, depth_image

    def renderComponent(
            self, assembly, component_index, component_pose,
            rgb_background=None, depth_background=None):
        """

        Parameters
        ----------
        rgb_background : array of float, shape (img_height, img_width, 3), optional
        depth_background : array of shape (img_height, img_width), optional

        Returns
        -------
        rgb_image :
        depth_image :
        label_image :
        """

        if (rgb_background is None) != (depth_background is None):
            err_str = (
                "Keyword arguments 'rgb_background' and 'depth_background'"
                "must be passed together --- one of the arguments passed is None,"
                "but the other is not."
            )
            raise AssertionError(err_str)

        assembly = assembly.recenter(component_index, in_place=False)

        vertices = torchutils.makeBatch(
            assembly.componentVertices(component_index), dtype=torch.float
        ).cuda()
        faces = torchutils.makeBatch(
            assembly.componentFaces(component_index), dtype=torch.int
        ).cuda()
        textures = torchutils.makeBatch(
            assembly.componentTextures(component_index), dtype=torch.float
        ).cuda()

        R, t = component_pose
        vertices = vertices @ R.T + t

        rgb_images, depth_images = self.render(vertices, faces, textures)

        if rgb_background is not None:
            rgb_images = torch.cat((rgb_background, rgb_images), 0)

        if depth_background is not None:
            depth_images = torch.cat((depth_background, depth_images), 0)

        rgb_image, depth_image, label_image = render.reduceByDepth(rgb_images, depth_images)

        return rgb_image, depth_image, label_image


class LegacySceneRenderer(object):
    def __init__(self, intrinsic_matrix=None, camera_pose=None, colors=None, **super_kwargs):
        self.intrinsic_matrix = intrinsic_matrix
        self.camera_pose = camera_pose
        self.colors = colors

    def renderScene(self, background_plane, assembly, component_poses):
        out = render.renderScene(
            background_plane, assembly, component_poses,
            camera_pose=self.camera_pose, camera_params=self.intrinsic_matrix,
            object_appearances=self.colors
        )
        return out

    def renderPlane(self, plane):
        out = render.renderPlane(
            plane, camera_pose=None, camera_params=None, plane_appearance=None,
            range_image=None, label_image=None, rgb_image=None
        )
        return out

    def renderComponent(self, assembly, component_idx):
        out = render.renderComponent(
            assembly, component_idx, component_pose=None, img_type=None,
            camera_pose=None, camera_params=None, block_colors=None,
            range_image=None, label_image=None, rgb_image=None,
            crop_rendered=False, in_place=True
        )
        return out


class RenderingSceneScorer(object):
    def forward(self, rgb_image, depth_image, segment_image, assembly, background_plane, **kwargs):
        error, __, __ = self.fitScene(
            rgb_image, depth_image, segment_image, assembly, background_plane,
            **kwargs
        )

        return error

    def fitScene(
            self, rgb_image, depth_image, segment_image,
            rgb_background, depth_background, assembly,
            camera_params=None, camera_pose=None, block_colors=None,
            W=None, error_func=None, bias=None, scale=None,
            ignore_background=False):
        """ Fit a spatial assembly and a background plane to an RGBD image.

        Parameters
        ----------

        Returns
        -------
        """

        if error_func is None:
            error_func = sse

        if W is None:
            W = np.ones(2)

        # Estimate initial poses from each detected image segment
        segment_labels = np.unique(segment_image[segment_image != 0])
        object_masks = tuple(segment_image == i for i in segment_labels)
        object_poses_est = utils.batchProcess(
            imageprocessing.estimateSegmentPose,
            object_masks,
            static_args=(camera_params, camera_pose, depth_image),
            static_kwargs={'estimate_orientation': False}
        )

        num_components = len(assembly.connected_components)
        num_segments = len(segment_labels)

        # Find the best pose for each component of the spatial assembly, assuming
        # we try to match it to a particular segment.
        errors = np.zeros((num_components, num_segments))
        poses = {}
        for component_index, component_key in enumerate(assembly.connected_components.keys()):
            for segment_index in range(num_segments):
                object_mask = object_masks[segment_index]
                init_pose = object_poses_est[segment_index]
                error, pose = self.refineComponentPose(
                    rgb_image, depth_image, segment_image, assembly,
                    rgb_background=rgb_background,
                    depth_background=depth_background,
                    component_index=component_key, init_pose=init_pose,
                    camera_params=camera_params, camera_pose=camera_pose,
                    block_colors=block_colors,
                    object_mask=object_mask, W=W, error_func=error_func,
                    bias=bias, scale=scale
                )
                errors[component_index, segment_index] = error
                poses[component_index, segment_index] = pose

        # Match components to segments by solving the linear sum assignment problem
        # (ie data association)
        _, component_poses, _ = matchComponentsToSegments(
            errors, poses, downstream_objectives=None, greedy_assignment=False
        )

        # Render the complete final scene
        # FIXME: Call renderComponents
        rgb_render, depth_render, label_render = self.renderScene(
            assembly, component_poses,
            camera_pose=camera_pose,
            camera_params=camera_params,
            object_appearances=block_colors
        )

        # Subtract background from all depth images. This gives distances relative
        # to the background plane instead of the camera, so RGB and depth models
        # are closer to the same scale.
        depth_render = depth_render - depth_background
        depth_image = depth_image - depth_background

        # Compute the total error of the final scene
        image_background = segment_image == 0
        render_background = label_render == 0

        rgb_error = error_func(
            rgb_image, rgb_render,
            bias=bias[0], scale=scale[0],
            true_mask=image_background, est_mask=render_background
        )

        depth_error = error_func(
            depth_image, depth_render,
            bias=bias[1], scale=scale[1],
            true_mask=image_background, est_mask=render_background
        )

        error = np.array([rgb_error, depth_error]) @ W

        return error, component_poses, (rgb_render, depth_render, label_render)

    def refineComponentPose(
            self, rgb_image, depth_image, segment_image, assembly,
            component_index=None, init_pose=None, theta_samples=None,
            object_mask=None, W=None, error_func=None, bias=None, scale=None,
            **render_kwargs):
        """ Refine a component's initial pose estimate using a simple registration routine.

        Parameters
        ----------

        Returns
        -------
        best_error : float
        best_pose : (R, t)
        """

        if error_func is None:
            error_func = sse

        if W is None:
            W = np.ones(2)

        if theta_samples is None:
            theta_samples = range(0, 360, 90)

        R_init, t_init = init_pose
        pose_candidates = tuple(
            (geometry.rotationMatrix(z_angle=theta, x_angle=0) @ R_init, t_init)
            for theta in theta_samples
        )

        rgb_renders, depth_renders, label_renders = utils.batchProcess(
            self.renderComponent, pose_candidates,
            static_args=(assembly, component_index),
            static_kwargs=render_kwargs,
            unzip=True
        )

        # Subtract background from all depth images. This gives distances relative
        # to the background plane instead of the camera, so RGB and depth models
        # are closer to the same scale.
        if 'depth_background' in render_kwargs:
            depth_renders = tuple(d - render_kwargs['depth_background'] for d in depth_renders)
            depth_image = depth_image - render_kwargs['depth_background']

        object_background_mask = ~object_mask
        label_background_masks = tuple(label_render == 0 for label_render in label_renders)

        rgb_errors = [
            error_func(
                rgb_image, rgb_render,
                true_mask=object_background_mask, est_mask=label_mask,
                bias=bias[0], scale=scale[0]
            )
            for rgb_render, label_mask in zip(rgb_renders, label_background_masks)
        ]

        depth_errors = [
            error_func(
                depth_image, depth_render,
                true_mask=object_background_mask, est_mask=label_mask,
                bias=bias[1], scale=scale[1]
            )
            for depth_render, label_mask in zip(depth_renders, label_background_masks)
        ]

        errors = np.column_stack((np.array(rgb_errors), np.array(depth_errors))) @ W

        best_idx = errors.argmin()
        best_error = errors[best_idx]
        best_pose = pose_candidates[best_idx]

        return best_error, best_pose


class LegacySceneScorer(RenderingSceneScorer, LegacySceneRenderer):
    pass


class TorchSceneScorer(RenderingSceneScorer, TorchSceneRenderer):
    pass


# -=( HELPER FUNCTIONS FOR SCENE SCORER)==-------------------------------------
def matchComponentsToSegments(
        objectives, poses, downstream_objectives=None, greedy_assignment=False):
    """

    Parameters
    ----------
    objectives : numpy array of float, shape (num_components, num_segments)
    poses :
    downstream_objectives : optional
    greedy_assignment : bool, optional

    Returns
    -------
    final_obj :
    best_poses :
    best_seg_idxs :
    """

    if not objectives.any():
        return None, (tuple(), tuple(), tuple()), tuple()

    # linear_sum_assignment can't take an infinty-valued matrix, so set those
    # greater than the max. That way they'll never be chosen by the routine.
    non_inf_max = objectives[~np.isinf(objectives)].max()
    objectives[np.isinf(objectives)] = non_inf_max + 1

    num_components, num_segments = objectives.shape
    if greedy_assignment:
        row_ind = np.arange(num_components)
        col_ind = objectives.argmin(axis=1)
    else:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(objectives)

    if downstream_objectives is None:
        final_obj = objectives[row_ind, col_ind].sum()
    else:
        final_obj = downstream_objectives[row_ind, col_ind].sum()

    best_poses = [poses[r, c] for r, c in zip(row_ind, col_ind)]
    best_seg_idxs = [c + 1 for c in col_ind]

    num_unassigned_components = num_components - num_segments
    for i in range(num_unassigned_components):
        R = geometry.rotationMatrix(z_angle=0, x_angle=0)
        t = np.zeros(3) + i * np.array([75, 0, 0])
        best_poses.append((R, t))
        best_seg_idxs.append(-1)

    # argmaxima = (theta_best, t_best, seg_best)
    if downstream_objectives is None:
        return final_obj, best_poses, best_seg_idxs


def residual(x_true, x_est, true_mask=None, est_mask=None):
    if true_mask is not None:
        x_true[true_mask] = 0

    if est_mask is not None:
        x_est[est_mask] = 0

    resid = x_true - x_est
    return resid


def standardize(x, bias=0, scale=1):
    x_standardized = (x - bias) / scale
    return x_standardized


def sse(x_true, x_est, true_mask=None, est_mask=None, bias=None, scale=None):
    x_true = standardize(x_true, bias=bias, scale=scale)
    x_est = standardize(x_est, bias=bias, scale=scale)

    resid = residual(x_true, x_est, true_mask=true_mask, est_mask=est_mask)
    sse = (resid ** 2).sum()

    return sse
