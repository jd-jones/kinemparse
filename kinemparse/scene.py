import logging

import torch
import scipy

import mathtools as m
from mathtools import utils, torchutils
from visiontools import geometry, render, imageprocessing
import neural_renderer as nr


logger = logging.getLogger(__name__)


class TorchSceneRenderer(nr.Renderer):
    def __init__(
            self, intrinsic_matrix=None, camera_pose=None, colors=None, image_shape=None,
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
        self.image_shape = image_shape

        super().__init__(
            camera_mode='projection', K=K, R=R, t=t,
            image_size=max(self.image_shape),
            near=0, far=1000, **super_kwargs
        )

    @property
    def camera_params(self):
        return self.K[0]

    @property
    def camera_pose(self):
        return geometry.homogeneousMatrix(self.R[0], self.t[0][0])

    @property
    def block_colors(self):
        return self.colors

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

        # [batch_size, RGB, image_shape, image_shape] -> [batch_size, image_shape, image_shape, RGB]
        images_rgb = images_rgb.permute(0, 2, 3, 1)

        # Crop square image back to the original aspect ratio
        images_rgb = images_rgb[:, :self.image_shape[0], :self.image_shape[1]]
        images_depth = images_depth[:, :self.image_shape[0], :self.image_shape[1]]

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
            rgb_background = torch.zeros(*self.image_shape, 3)

        if depth_background is None:
            depth_background = torch.full(self.image_shape, float('inf'))

        if not assembly.blocks:
            label_background = torch.zeros(*self.image_shape, dtype=torch.int)
            return rgb_background, depth_background, label_background

        assembly = assembly.setPose(component_poses, in_place=False)

        vertices = torchutils.makeBatch(assembly.vertices, dtype=torch.float).cuda()
        faces = torchutils.makeBatch(assembly.faces, dtype=torch.int).cuda()
        textures = torchutils.makeBatch(assembly.textures, dtype=torch.float).cuda()

        rgb_images, depth_images = self.render(vertices, faces, textures)

        rgb_images = torch.cat((rgb_background[None, ...], rgb_images), 0)
        depth_images = torch.cat((depth_background[None, ...], depth_images), 0)

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

        if not isinstance(plane._t, torch.Tensor):
            plane._t = m.np.array(plane._t)
        if not isinstance(plane._U, torch.Tensor):
            plane._U = m.np.array(plane._U)

        vertices, faces = render.planeVertices(
            plane, camera_params, camera_pose, image_shape=self.image_shape
        )
        faces = faces.int()
        textures = render.makeTextures(faces, uniform_color=self.colors[0])
        rgb_image_batch, depth_image_batch = self.render(vertices, faces, textures)

        rgb_image = rgb_image_batch[0, ...]
        depth_image = depth_image_batch[0, ...]

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

        vertices = torch.stack(tuple(assembly.componentVertices(component_index)))
        faces = torch.stack(tuple(assembly.componentFaces(component_index)))
        textures = torch.stack(tuple(assembly.componentTextures(component_index)))

        R, t = component_pose
        vertices = vertices @ m.np.transpose(R) + t

        rgb_images, depth_images = self.render(vertices, faces, textures)

        if rgb_background is not None:
            rgb_images = torch.cat((rgb_background[None, ...], rgb_images), 0)

        if depth_background is not None:
            depth_images = torch.cat((depth_background[None, ...], depth_images), 0)

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
    def forward(self, *args, **kwargs):
        return self._create_data_scores(*args, **kwargs)

    def _create_data_scores(self, sample, W=None, return_poses=False, **kwargs):
        """
        Parameters
        ----------
        sample :
            (rgb_image, depth_image, segment_image, rgb_background, depth_background)
        **kwargs : optional

        Returns
        -------
        error :
        """

        if W is None:
            W = m.np.ones(2)

        kwargs['W'] = W

        assemblies = tuple(self.integerizer[i] for i in range(self.num_states))

        errors, component_poses = utils.batchProcess(
            self.fitScene, assemblies,
            static_args=sample,
            static_kwargs=kwargs,
            unzip=True
        )

        error = (m.np.vstack(errors) @ W)[None, :]

        if not return_poses:
            return error

        return error, component_poses

    def fitScene(
            self, rgb_image, depth_image, segment_image,
            rgb_background, depth_background, assembly,
            camera_params=None, camera_pose=None, block_colors=None,
            W=None, error_func=None, bias=None, scale=None,
            ignore_background=False, legacy_mode=False):
        """ Fit a spatial assembly and a background plane to an RGBD image.

        Parameters
        ----------

        Returns
        -------
        """

        if camera_params is None:
            camera_params = self.camera_params

        if camera_pose is None:
            camera_pose = self.camera_pose

        if block_colors is None:
            block_colors = self.block_colors

        if W is None:
            W = m.np.ones(2)

        if error_func is None:
            error_func = sse

        if bias is None:
            bias = m.np.zeros(2)

        if scale is None:
            scale = m.np.ones(2)

        # Estimate initial poses from each detected image segment
        segment_labels = m.np.unique(segment_image[segment_image != 0])

        num_segments = len(segment_labels)
        num_components = len(assembly.connected_components.keys())

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
        errors = m.np.zeros((num_components, num_segments))
        poses = m.np.zeros((num_components, num_segments, 3, 4))
        for component_index, component_key in enumerate(assembly.connected_components.keys()):
            for segment_index in range(num_segments):
                object_mask = object_masks[segment_index]
                init_pose = object_poses_est[segment_index]
                error, pose = self.refineComponentPose(
                    rgb_image, depth_image, segment_image, assembly,
                    rgb_background=rgb_background,
                    depth_background=depth_background,
                    component_index=component_key, init_pose=init_pose,
                    object_mask=object_mask, W=W, error_func=error_func,
                    bias=bias, scale=scale
                )
                errors[component_index, segment_index] = error
                poses[component_index, segment_index] = pose

        # Match components to segments by solving the linear sum assignment problem
        # (ie data association)
        # FIXME: set greedy=False
        _, component_poses, _ = matchComponentsToSegments(errors, poses, greedy=True)

        # Render the complete final scene
        rgb_render, depth_render, label_render = self.renderScene(
            assembly, component_poses,
            depth_background=depth_background, rgb_background=rgb_background
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

        error_vec = m.np.array([rgb_error, depth_error])
        return error_vec, component_poses

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
        best_pose : m.np.array of float, shape (3, 4)
        """

        if error_func is None:
            error_func = sse

        if W is None:
            W = m.np.ones(2)

        if theta_samples is None:
            theta_samples = m.np.linspace(0, 1.5 * m.np.pi, 4)

        R_init, t_init = init_pose
        # pose_candidates = tuple(
        #     (geometry.rotationMatrix(z_angle=theta, x_angle=0) @ R_init, t_init)
        #     for theta in theta_samples
        # )

        rotation_candidates = geometry.zRotations(theta_samples) @ R_init
        pose_candidates = tuple((R, t_init) for R in rotation_candidates)

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

        errors = m.np.column_stack((m.np.array(rgb_errors), m.np.array(depth_errors))) @ W

        best_idx = errors.argmin()
        best_error = errors[best_idx]
        best_pose = pose_candidates[best_idx]
        best_pose = geometry.homogeneousMatrix(pose_candidates[best_idx])

        return best_error, best_pose


class LegacySceneScorer(RenderingSceneScorer, LegacySceneRenderer):
    pass


class TorchSceneScorer(RenderingSceneScorer, TorchSceneRenderer):
    pass


# -=( HELPER FUNCTIONS FOR SCENE SCORER)==-------------------------------------
def matchComponentsToSegments(objectives, poses, greedy=False):
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

    if greedy:
        def assign(objectives):
            row_ind = m.np.arange(objectives.shape[0])
            col_ind = objectives.argmin(1)
            return row_ind, col_ind
    else:
        def assign(objectives):
            # linear_sum_assignment can't take an infinty-valued matrix, so set those
            # greater than the max. That way they'll never be chosen by the routine.
            non_inf_max = objectives[~m.np.isinf(objectives)].max()
            objectives[m.np.isinf(objectives)] = non_inf_max + 1
            if isinstance(objectives, torch.Tensor):
                return scipy.optimize.linear_sum_assignment(objectives.cpu().numpy())
            else:
                return scipy.optimize.linear_sum_assignment(objectives)

    if not m.np.any(objectives):
        final_obj = 0
        best_poses = []
        best_seg_idxs = m.np.array([])
    else:
        row_ind, col_ind = assign(objectives)
        final_obj = objectives[row_ind, col_ind].sum()
        best_poses = list(zip(geometry.fromHomogeneous(poses[row_ind, col_ind])))
        best_seg_idxs = col_ind + 1

    # Give unassigned components a pose
    num_components, num_segments = objectives.shape
    num_unassigned = num_components - num_segments
    # FIXME: hstack breaks when best_seg_idxs is empty (ie objectives is empty)
    best_seg_idxs = m.np.hstack((best_seg_idxs, -m.np.ones(num_unassigned, dtype=m.np.long)))
    R = m.np.eye(3)
    t_space = m.np.array([75, 0, 0], dtype=m.np.float)
    for i in range(num_unassigned):
        best_poses.append((R, i * t_space))

    return final_obj, best_poses, best_seg_idxs


def sse(x_true, x_est, true_mask=None, est_mask=None, bias=None, scale=None):
    x_true = standardize(x_true, bias=bias, scale=scale)
    x_est = standardize(x_est, bias=bias, scale=scale)

    resid = residual(x_true, x_est, true_mask=true_mask, est_mask=est_mask)
    sse = (resid ** 2).sum()

    return sse


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
