import os
import logging
import math

import torch
import torchvision
import kornia
import numpy as np
from scipy.spatial.transform import Rotation

from mathtools import utils, torchutils, torch_future
from visiontools import render, imageprocessing


logger = logging.getLogger(__name__)


# -=( DATASETS )==-------------------------------------------------------------
class PickledVideoDataset(torch.utils.data.Dataset):
    """ A dataset wrapping sequences of numpy arrays stored in memory.

    Attributes
    ----------
    _data : tuple(np.ndarray, shape (num_samples, num_dims))
    _labels : tuple(np.ndarray, shape (num_samples,))
    _device : torch.Device
    """

    def __init__(
            self, data_loader, labels, device=None, labels_dtype=None,
            sliding_window_args=None, transpose_data=False, seq_ids=None,
            batch_size=None, batch_mode=None):
        """
        Parameters
        ----------
        data_loader : function
            data_loader should take a sequence ID and return the data sample
            corresponding to that ID --- ie an array_like of float with shape
            (sequence_len, num_dims)
        labels : iterable( array_like of int, shape (sequence_len,) )
        device :
        labels_dtype : torch data type
            If passed, labels will be converted to this type
        sliding_window_args : tuple(int, int, int), optional
            A tuple specifying parameters for extracting sliding windows from
            the data sequences. This should be ``(dimension, size, step)``---i.e.
            the input to ``torch.unfold``. The label of each sliding window is
            taken to be the median over the labels in that window.
        """

        if seq_ids is None:
            raise ValueError("This class must be initialized with seq_ids")

        if len(labels[0].shape) == 2:
            self.num_label_types = labels[0].shape[1]
        elif len(labels[0].shape) < 2:
            # self.num_label_types = np.unique(np.hstack(labels)).max() + 1
            self.num_label_types = len(np.unique(np.hstack(labels)))
        else:
            err_str = f"Labels have a weird shape: {labels[0].shape}"
            raise ValueError(err_str)

        self.sliding_window_args = sliding_window_args
        self.transpose_data = transpose_data
        self.batch_size = batch_size
        self.batch_mode = batch_mode

        self._load_data = data_loader

        self._device = device
        self._seq_ids = seq_ids

        self._labels = tuple(
            map(lambda x: torch.tensor(x, device=device, dtype=labels_dtype), labels)
        )

        self._seq_lens = tuple(x.shape[0] for x in self._labels)

        if self.batch_size is not None and self.batch_mode == 'flatten':
            self.unflatten = tuple(
                (seq_index, win_index)
                for seq_index, seq_len in enumerate(self._seq_lens)
                for win_index in range(0, seq_len, self.batch_size)
            )

        logger.info('Initialized PickledVideoDataset.')
        logger.info(f"{self.num_label_types} unique labels")

    def __len__(self):
        if self.batch_mode == 'flatten':
            return len(self.unflatten)
        return len(self._seq_ids)

    def __getitem__(self, i):
        if self.batch_size is not None:
            seq_idx, win_idx = self.unflatten[i]

            seq_id = self._seq_ids[seq_idx]
            label_seq = self._labels[seq_idx]
            data_seq = self._load_data(seq_id)

            start_idx = win_idx
            end_idx = start_idx + self.batch_size
            data_seq = data_seq[start_idx:end_idx]
            label_seq = label_seq[start_idx:end_idx]
        else:
            seq_id = self._seq_ids[i]
            label_seq = self._labels[i]
            data_seq = self._load_data(seq_id)

        data_seq = torch.tensor(data_seq, device=self._device, dtype=torch.float)

        # shape (sequence_len, num_dims) --> (num_dims, sequence_len)
        if self.transpose_data:
            data_seq = data_seq.transpose(0, 1)

        if self.sliding_window_args is not None:
            # Unfold gives shape (sequence_len, window_len);
            # after transpose, data_seq has shape (window_len, sequence_len)
            data_seq = data_seq.unfold(*self.sliding_window_args).transpose(-1, -2)
            label_seq = label_seq.unfold(*self.sliding_window_args).median(dim=-1).values

        return data_seq, label_seq, i


class BlocksVideoDataset(PickledVideoDataset):
    def __init__(self, *args, background=None, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(self, data_seqs):
        rgb_seq, bg_masks, person_masks = data_seqs

        rgb_saturated = torch.tensor(
            np.moveaxis(
                np.stack(tuple(
                    imageprocessing.saturateImage(rgb, to_float=False)
                    for rgb in rgb_seq
                )), 3, 1
            ), device=self._device, dtype=torch.float
        )

        depth_bg_masks = torch.tensor(
            np.moveaxis(bg_masks[..., None], 3, 1),
            dtype=torch.uint8, device=self._device
        )

        person_masks = torch.tensor(
            np.moveaxis(person_masks[..., None], 3, 1),
            dtype=torch.uint8, device=self._device
        )

        blocks_masks = ((depth_bg_masks + person_masks) < 1).float()
        rgb_saturated *= blocks_masks

        return rgb_saturated

    def _load(self, i):
        if self.batch_size is not None and self.batch_mode == 'flatten':
            seq_idx, win_idx = self.unflatten[i]

            seq_id = self._seq_ids[seq_idx]
            label_seq = self._labels[seq_idx]
            data_seqs = self._load_data(seq_id)

            start_idx = win_idx
            end_idx = start_idx + self.batch_size
            data_seqs = tuple(data_seq[start_idx:end_idx] for data_seq in data_seqs)
            label_seq = label_seq[start_idx:end_idx]
        elif self.batch_size is not None and self.batch_mode == 'sample':
            seq_id = self._seq_ids[i]
            label_seq = self._labels[i]
            data_seqs = self._load_data(seq_id)

            batch_idxs = utils.sampleWithoutReplacement(
                label_seq, num_samples=self.batch_size,
                return_indices=True
            )
            data_seqs = tuple(x[batch_idxs] for x in data_seqs)
            label_seq = label_seq[batch_idxs]
        else:
            seq_id = self._seq_ids[i]
            label_seq = self._labels[i]
            data_seqs = self._load_data(seq_id)

        return data_seqs, label_seq, seq_id

    def __getitem__(self, i):
        data_seqs, label_seq, seq_id = self._load(i)
        data_seq = self._transform(data_seqs)

        # shape (sequence_len, num_dims) --> (num_dims, sequence_len)
        if self.transpose_data:
            data_seq = data_seq.transpose(0, 1)

        if self.sliding_window_args is not None:
            # Unfold gives shape (sequence_len, window_len);
            # after transpose, data_seq has shape (window_len, sequence_len)
            data_seq = data_seq.unfold(*self.sliding_window_args).transpose(-1, -2)
            label_seq = label_seq.unfold(*self.sliding_window_args).median(dim=-1).values

        return data_seq, label_seq, i


class BlocksConnectionDataset(BlocksVideoDataset):
    def __init__(
            self, vocabulary, edge_labels, label_dtype, *args,
            background=None, debug_fig_dir=None, crop_images=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.crop_images = crop_images
        self.vocab = vocabulary
        self.edge_labels = torch.tensor(edge_labels, device=self._device, dtype=label_dtype)

        self._debug_fig_dir = debug_fig_dir

        all_labels = torch.cat(self._labels)
        all_edges = self.edge_labels[all_labels]
        num_classes = all_edges.max() + 1
        self.class_freqs = torch.stack(tuple(
            torch.tensor(
                utils.makeHistogram(num_classes, column.cpu().numpy(), normalize=True),
                device=self.edge_labels.device, dtype=torch.float
            )
            for column in all_edges.transpose(0, 1)
        ), dim=1)

    def __getitem__(self, i):
        data_seqs, label_seq, seq_id = self._load(i)
        data_seq, label_seq = self._transform(data_seqs, label_seq)

        # shape (sequence_len, num_dims) --> (num_dims, sequence_len)
        if self.transpose_data:
            data_seq = data_seq.transpose(0, 1)

        if self.sliding_window_args is not None:
            # Unfold gives shape (sequence_len, window_len);
            # after transpose, data_seq has shape (window_len, sequence_len)
            data_seq = data_seq.unfold(*self.sliding_window_args).transpose(-1, -2)
            label_seq = label_seq.unfold(*self.sliding_window_args).median(dim=-1).values

        return data_seq, label_seq, i

    def _transform(self, data_seqs, label_seq):
        def padCrops(crops, max_num_crops):
            num_crops = crops.shape[0]
            num_pad = max_num_crops - num_crops
            pad_crops = np.zeros((num_pad, *crops.shape[1:]))
            return np.concatenate((crops, pad_crops), axis=0)

        rgb_seq, segs_seq = data_seqs

        label_seq = self.edge_labels[label_seq]

        rgb_seq = np.stack(tuple(
            imageprocessing.saturateImage(rgb, to_float=True)
            for rgb in rgb_seq
        ))

        rgb_seq[segs_seq == 0] = 0

        if self.crop_images:
            rgb_seqs = tuple(
                makeCrops(rgb, seg_labels)
                for rgb, seg_labels in zip(rgb_seq, segs_seq)
            )
            max_num_crops = max(x.shape[0] for x in rgb_seqs)
            rgb_seq = np.stack(
                tuple(padCrops(x, max_num_crops) for x in rgb_seqs),
                axis=0
            )

        rgb_seq = torch.tensor(
            np.moveaxis(rgb_seq, -1, -3),
            device=label_seq.device, dtype=torch.float
        )

        return rgb_seq, label_seq

    @property
    def target_shape(self):
        assembly = self.vocab[0]
        return assembly.connections.shape


class RenderDataset(torch.utils.data.Dataset):
    def __init__(
            self, vocab, device=None, batch_size=None, occlusion_masks=None,
            intrinsic_matrix=None, camera_pose=None, colors=None, crop_size=128,
            kornia_tfs={}):
        if intrinsic_matrix is None:
            intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()

        if camera_pose is None:
            camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()

        if colors is None:
            colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

        self.occlusion_masks = occlusion_masks
        self.crop_size = crop_size
        self.image_size = min(render.IMAGE_WIDTH, render.IMAGE_HEIGHT)

        # FIXME: camera intrisics are calibrated for 240x320 camera,
        #    but we render at 240x240 resolution

        self.renderer = render.TorchSceneRenderer(
            intrinsic_matrix=intrinsic_matrix,
            camera_pose=camera_pose,
            colors=colors,
            light_intensity_ambient=1,
            image_size=self.image_size,
            orig_size=self.image_size
        )

        # FIXME: handle multi-component assemblies properly
        self.vocab = tuple(a for a in vocab if len(a.connected_components) == 1)

        self.device = device
        self.batch_size = batch_size

        self._kornia_tfs = self._init_kornia_tfs(kornia_tfs=kornia_tfs)

    def _init_kornia_tfs(self, kornia_tfs={}):
        tfs = torch.nn.Sequential(*(
            getattr(kornia.augmentation, name)(**params)
            for name, params in kornia_tfs.items()
        ))
        return tfs

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, i):
        image = self._getImage(i)
        target = self._getTarget(i)
        return image, target, -1

    def _getTarget(self, i):
        return i

    def _getImage(self, i, pose=None):
        assembly = self.vocab[i]

        if pose is None:
            R, t = self.samplePose()
        else:
            R, t = pose

        imgs_batch = renderTemplates(self.renderer, assembly, t, R)
        imgs_batch = self._pad_images(*imgs_batch)
        imgs_batch = self._occlude_images(*imgs_batch)

        rgb_batch, depth_batch, label_batch = imgs_batch
        rgb_crop = _crop(rgb_batch[0], label_batch[0], self.crop_size)

        rgb_crop = rgb_crop.permute(-1, 0, 1)
        # rgb_crop = self._augment_image(rgb_crop)[0]

        return rgb_crop

    def _pad_images(self, rgb, depth, label, target_shape=None):
        """ Expand images from 240x240 to 240x320 """

        if target_shape is None:
            target_shape = (render.IMAGE_HEIGHT, render.IMAGE_WIDTH)

        target_height, target_width = target_shape
        batch_size, img_height, img_width, num_channels = rgb.shape

        if img_height != target_height:
            raise AssertionError()

        pad_shape = (batch_size, target_height, target_width - img_width)

        padding = torch.zeros((*pad_shape, num_channels), dtype=rgb.dtype, device=rgb.device)
        rgb = torch.cat((padding, rgb), dim=2)

        padding = torch.zeros(pad_shape, dtype=label.dtype, device=label.device)
        label = torch.cat((padding, label), dim=2)

        padding = torch.full(
            pad_shape, self.renderer.far,
            dtype=depth.dtype, device=depth.device
        )
        depth = torch.cat((padding, depth), dim=2)

        return rgb, depth, label

    def _occlude_images(self, rgb, depth, label, in_place=True):
        if not in_place:
            raise NotImplementedError()

        is_occluded = self.sampleOcclusion()

        rgb[is_occluded] = 0
        depth[is_occluded] = self.renderer.far
        label[is_occluded] = 0

        return rgb, depth, label

    def _augment_image(self, rgb):
        return self._kornia_tfs(rgb.cpu()).cuda()

    def sampleOcclusion(self, num_samples=1):
        if self.occlusion_masks is None:
            raise NotImplementedError()
            occlusion_masks = None
            return occlusion_masks

        occlusion_masks = utils.sampleWithoutReplacement(
            self.occlusion_masks, num_samples=num_samples
        )
        return torch.tensor(occlusion_masks, dtype=torch.uint8, device=self.device)

    def sampleOrientation(self, num_samples=1):
        """
        Returns
        -------
        R : torch.Tensor of shape (num_samples, 3, 3)
        """

        z_angles = _sample_uniform(num_samples, 0, 2 * np.pi).cuda()
        y_angles = torch.full((num_samples, 1), 0).cuda()
        x_angles = torch.full((num_samples, 1), 0).cuda()
        angles = torch.cat((z_angles, y_angles, x_angles), dim=1)

        rotations = Rotation.from_euler('ZYX', angles.cpu().numpy())
        R = torch.tensor(rotations.as_matrix()).float()
        return R.cuda()

    def samplePosition(self, num_samples=1):
        """
        Returns
        -------
        t : torch.Tensor of shape (num_samples, 3)
        """

        camera_dist = self.renderer.t[0, 0, 2]

        px_bounds = torch.tensor(
            [[0, 0],
             [self.image_size, self.image_size]],
            dtype=torch.float
        ).cuda()
        depth_coords = torch.full(px_bounds.shape[0:1], camera_dist)[..., None].cuda()
        xy_bounds = kornia.unproject_points(
            px_bounds, depth_coords, self.renderer.K
        )[:, :-1]

        xy_samples = _sample_uniform(num_samples, xy_bounds[0], xy_bounds[1])
        z_samples = torch.full((num_samples, 1), 0).cuda()

        t = torch.cat((xy_samples, z_samples), dim=1)

        return t.cuda()

    def samplePose(self, num_samples=1):
        R = self.sampleOrientation(num_samples=num_samples)
        t = self.samplePosition(num_samples=num_samples)
        return R, t

    @property
    def data_shape(self):
        image, label, _ = self[0]
        return image.shape

    @property
    def label_shape(self):
        image, label, _ = self[0]
        return label.shape

    @property
    def num_classes(self):
        return len(self.vocab)


class DenoisingDataset(RenderDataset):
    def _getTarget(self, i):
        target_image = self._getImage(i, pose=self.canonicalPose())
        return target_image

    def canonicalPose(self, num_samples=1):
        angles = torch.zeros(num_samples, dtype=torch.float)
        rotations = Rotation.from_euler('Z', angles)
        R = torch.tensor(rotations.as_matrix()).float().cuda()
        t = torch.stack((torch.zeros_like(angles),) * 3, dim=1).float().cuda()
        return R, t


class ConnectionDataset(RenderDataset):
    def _getTarget(self, i):
        assembly = self.vocab[i]
        connections = torch.tensor(assembly.connections, dtype=torch.float)
        return connections.view(-1).contiguous()

    @property
    def target_shape(self):
        assembly = self.vocab[0]
        return assembly.connections.shape


class LabeledConnectionDataset(ConnectionDataset):
    def __init__(
            self, parts_vocab, part_labels, vocab,
            device=None, batch_size=None, occlusion_masks=None,
            intrinsic_matrix=None, camera_pose=None, colors=None, crop_size=128,
            kornia_tfs={}):
        if intrinsic_matrix is None:
            intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()

        if camera_pose is None:
            camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()

        if colors is None:
            colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

        self.device = device
        self.batch_size = batch_size

        self.occlusion_masks = occlusion_masks
        self.crop_size = crop_size
        self.image_size = min(render.IMAGE_WIDTH, render.IMAGE_HEIGHT)

        # FIXME: camera intrisics are calibrated for 240x320 camera,
        #    but we render at 240x240 resolution

        self.renderer = render.TorchSceneRenderer(
            intrinsic_matrix=intrinsic_matrix,
            camera_pose=camera_pose,
            colors=colors,
            light_intensity_ambient=1,
            image_size=self.image_size,
            orig_size=self.image_size
        )

        self.parts_vocab = parts_vocab
        self.part_labels = torch.tensor(part_labels, dtype=torch.long, device=self.device)

        # FIXME: handle multi-component assemblies properly
        has_one_component = [len(a.connected_components) == 1 for a in vocab]
        self.vocab = tuple(a for a, b in zip(vocab, has_one_component) if b)
        self.part_labels = self.part_labels[has_one_component]

        num_classes = self.part_labels.max() + 1
        self.class_freqs = torch.stack(tuple(
            torch.tensor(
                utils.makeHistogram(num_classes, column.cpu().numpy(), normalize=True),
                device=self.part_labels.device, dtype=torch.float
            )
            for column in self.part_labels.transpose(0, 1)
        ), dim=1)

        self._kornia_tfs = self._init_kornia_tfs(kornia_tfs=kornia_tfs)

    def _getTarget(self, i):
        return self.part_labels[i]


# -=( MODELS )==---------------------------------------------------------------
class SceneClassifier(torch.nn.Module):
    def __init__(self, pretrained_model, pred_thresh=0.5, finetune_extractor=True):
        super().__init__()
        self._model = pretrained_model
        self.pred_thresh = pred_thresh

        if not finetune_extractor:
            for param in self._model.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        in_shape = inputs.shape

        inputs = inputs.view(-1, *in_shape[2:])
        outputs = self._model(inputs)
        outputs = outputs.view(*in_shape[:2], *outputs.shape[1:])

        crop_is_padding = (inputs == 0).view(*in_shape[:2], -1).all(dim=2)
        outputs[crop_is_padding] = 0

        outputs = outputs.sum(dim=1)
        return outputs

    def predict(self, outputs):
        return self._model.predict(outputs)

    def plotBatches(self, io_batches, fig_dir, dataset=None, images_per_fig=None):
        for i, batch in enumerate(io_batches):
            preds, scores, inputs, labels, seq_id = batch

            batch_size = preds.shape[0]

            inputs = np.moveaxis(inputs.cpu().numpy(), -3, -1)
            inputs[inputs > 1] = 1
            flat_inputs = np.stack(
                tuple(np.hstack(tuple(c for c in crops)) for crops in inputs),
                axis=0
            )

            scores = scores.view(batch_size, *dataset.target_shape).cpu().numpy()
            preds = preds.view(batch_size, *dataset.target_shape).cpu().numpy()
            labels = labels.view(batch_size, *dataset.target_shape).cpu().numpy()

            imageprocessing.displayImages(
                *flat_inputs, *scores, *preds, *labels, num_rows=4,
                file_path=os.path.join(fig_dir, f"batch={i}.png")
            )


class ConnectionClassifier(torch.nn.Module):
    def __init__(self, out_dim, feature_dim=None, feature_extractor=None):
        super().__init__()

        if feature_extractor is None:
            pretrained_model = torchvision.models.resnet18(pretrained=True, progress=True)
            layers = list(pretrained_model.children())[:-1]
            feature_extractor = torch.nn.Sequential(*layers)
            feature_dim = 512

        self.feature_extractor = feature_extractor
        self.fc = torch.nn.Linear(feature_dim, out_dim)

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        outputs = self.fc(features.squeeze(dim=-1).squeeze(dim=-1))
        return outputs

    def predict(self, outputs):
        return (torch.sigmoid(outputs) > 0.5).float()

    def plotBatches(self, io_batches, fig_dir, dataset=None, images_per_fig=None):
        for i, batch in enumerate(io_batches):
            preds, scores, inputs, labels, seq_id = batch
            num_batch = preds.shape[0]

            if images_per_fig is None:
                images_per_fig = num_batch

            num_batches = math.ceil(num_batch / images_per_fig)
            for j in range(num_batches):
                start = j * num_batches
                end = start + images_per_fig

                b_scores = scores[start:end]
                b_preds = preds[start:end]
                b_labels = labels[start:end]
                b_inputs = inputs[start:end]

                b_size = b_scores.shape[0]
                if not b_size:
                    continue

                b_inputs = np.moveaxis(b_inputs.cpu().numpy(), 1, -1)
                b_inputs[b_inputs > 1] = 1

                b_scores = b_scores.view(b_size, *dataset.target_shape).cpu().numpy()
                b_preds = b_preds.view(b_size, *dataset.target_shape).cpu().numpy()
                b_labels = b_labels.view(b_size, *dataset.target_shape).cpu().numpy()

                imageprocessing.displayImages(
                    *b_inputs, *b_scores, *b_preds, *b_labels, num_rows=4,
                    file_path=os.path.join(fig_dir, f"batch({i},{j}).png")
                )


class LabeledConnectionClassifier(torch.nn.Module):
    def __init__(
            self, out_dim, num_vertices, edges,
            feature_dim=None, feature_extractor=None, finetune_extractor=True,
            feature_extractor_name='resnet50', feature_extractor_layer=-1):
        super().__init__()

        self.out_dim = out_dim
        self.num_vertices = num_vertices
        self.edges = edges

        if feature_extractor is None:
            # pretrained_model = torchvision.models.resnet18(pretrained=True, progress=True)
            Extractor = getattr(torchvision.models, feature_extractor_name)
            pretrained_model = Extractor(pretrained=True, progress=True)
            layers = list(pretrained_model.children())[:feature_extractor_layer]
            feature_extractor = torch.nn.Sequential(*layers)
            feature_dim = 512

        if not finetune_extractor:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        self.feature_extractor = feature_extractor
        self.fc = torch.nn.Linear(feature_dim, out_dim * self.edges.shape[0])

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        outputs = self.fc(features.squeeze(dim=-1).squeeze(dim=-1))
        outputs = outputs.view(outputs.shape[0], self.out_dim, self.edges.shape[0])
        return outputs

    def predict(self, outputs):
        preds = outputs.argmax(dim=1)
        return preds

    def edgeLabelsAsArray(self, batch):
        arr = torch.zeros(
            batch.shape[0], self.num_vertices, self.num_vertices,
            dtype=batch.dtype, device=batch.device
        )
        arr[:, self.edges[:, 0], self.edges[:, 1]] = batch
        arr[:, self.edges[:, 1], self.edges[:, 0]] = batch
        return arr

    def plotBatches(self, io_batches, fig_dir, dataset=None, images_per_fig=None):
        for i, batch in enumerate(io_batches):
            preds, scores, inputs, labels, seq_id = batch
            num_batch = preds.shape[0]

            if images_per_fig is None:
                images_per_fig = num_batch

            num_batches = math.ceil(num_batch / images_per_fig)
            for j in range(num_batches):
                start = j * num_batches
                end = start + images_per_fig

                b_scores = scores[start:end]
                b_preds = preds[start:end]
                b_labels = labels[start:end]
                b_inputs = inputs[start:end]

                b_size = b_scores.shape[0]
                if not b_size:
                    continue

                b_inputs = np.moveaxis(b_inputs.cpu().numpy(), 1, -1)
                b_inputs[b_inputs > 1] = 1

                # b_scores = self.edgeLabelsAsArray(b_scores).cpu().numpy()
                b_preds = self.edgeLabelsAsArray(b_preds).cpu().numpy()
                b_labels = self.edgeLabelsAsArray(b_labels).cpu().numpy()

                imageprocessing.displayImages(
                    *b_inputs,
                    # *b_scores,
                    *b_preds, *b_labels, num_rows=3,
                    file_path=os.path.join(fig_dir, f"batch({i},{j}).png")
                )


class AugmentedAutoEncoder(torch.nn.Module):
    def __init__(
            self, in_shape, num_classes, kernel_size=5, stride=2, latent_dim=128,
            encoder_channels=(3, 128, 256, 256, 512), decoder_channels=None,
            debug_fig_dir=None):
        super().__init__()

        if decoder_channels is None:
            decoder_channels = tuple(reversed(encoder_channels))

        self.encoder, conv_output_shape = self._makeEncoder(
            in_shape, kernel_size=kernel_size, stride=stride,
            encoder_channels=encoder_channels, latent_dim=latent_dim
        )

        self.decoder = self._makeDecoder(
            conv_output_shape, kernel_size=kernel_size, stride=stride,
            decoder_channels=decoder_channels, latent_dim=latent_dim
        )

        self._debug_fig_dir = debug_fig_dir
        self._index = 0

    def _makeEncoder(
            self, in_shape, kernel_size=5, stride=2,
            encoder_channels=(3, 128, 256, 256, 512), latent_dim=128):

        def makeConvBlock(in_channels, out_channels, kernel_size, stride):
            conv = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
            )
            activation = torch.nn.ReLU()
            block = torch.nn.Sequential(conv, activation)
            return block

        def makeLinearBlock(latent_dim, conv_output_shape):
            flatten = torch_future.Flatten()
            input_dim = torch.tensor(conv_output_shape).cumprod(0)[-1]
            fc = torch.nn.Linear(input_dim, latent_dim)
            block = torch.nn.Sequential(flatten, fc)
            return block

        conv_input_shape = in_shape
        for out_channels in encoder_channels[1:]:
            conv_output_shape = torchutils.conv2dOutputShape(
                conv_input_shape, out_channels, kernel_size,
                stride=stride, padding=(kernel_size // 2)
            )
            conv_input_shape = conv_output_shape

        encoder_blocks = [
            makeConvBlock(c_in, c_out, kernel_size, stride)
            for c_in, c_out in zip(encoder_channels[:-1], encoder_channels[1:])
        ] + [makeLinearBlock(latent_dim, conv_output_shape)]

        return torch.nn.Sequential(*encoder_blocks), conv_output_shape

    def _makeDecoder(
            self, conv_input_shape, kernel_size=5, stride=2,
            decoder_channels=(512, 256, 256, 128, 3), latent_dim=128):

        def makeDeconvBlock(in_channels, out_channels, kernel_size, stride, activation=None):
            if activation is None:
                activation = torch.nn.ReLU

            upsample = torch.nn.Upsample(scale_factor=stride)
            conv = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=1, padding=kernel_size // 2
            )
            block = torch.nn.Sequential(upsample, conv, activation())
            return block

        def makeLinearBlock(latent_dim, conv_input_shape):
            output_dim = torch.tensor(conv_input_shape).cumprod(0)[-1].item()
            fc = torch.nn.Linear(latent_dim, output_dim)
            unflatten = torch_future.Unflatten(1, conv_input_shape)
            block = torch.nn.Sequential(fc, unflatten)
            return block

        decoder_blocks = [makeLinearBlock(latent_dim, conv_input_shape)] + [
            makeDeconvBlock(
                c_in, c_out, kernel_size, stride,
                activation=(torch.nn.Sigmoid if c_out == decoder_channels[-1] else torch.nn.ReLU)
            )
            for c_in, c_out in zip(decoder_channels[:-1], decoder_channels[1:])
        ]

        return torch.nn.Sequential(*decoder_blocks)

    def forward(self, inputs):
        features = self.encoder(inputs)
        outputs = self.decoder(features)

        if self._debug_fig_dir is not None:
            self.viz(inputs, outputs)

        return outputs

    def viz(self, inputs, outputs):
        self._index += 1
        i = self._index

        inputs = np.moveaxis(inputs.cpu().numpy(), 1, -1)
        inputs[inputs > 1] = 1

        outputs = np.moveaxis(outputs.detach().cpu().numpy(), 1, -1)
        outputs[outputs > 1] = 1

        imageprocessing.displayImages(
            *inputs, *outputs, num_rows=2,
            file_path=os.path.join(self._debug_fig_dir, f"{i}.png")
        )

    def forward_debug(self, inputs):
        features = inputs
        for i, layer in enumerate(self.encoder):
            logger.info(f"encoder layer {i}, input: {features.shape}")
            features = layer(features)
            logger.info(f"encoder layer {i}, output: {features.shape}")

        outputs = features
        for i, layer in enumerate(self.decoder):
            logger.info(f"decoder layer {i}, input: {outputs.shape}")
            outputs = layer(outputs)
            logger.info(f"decoder layer {i}, output: {outputs.shape}")

        return outputs

    def plotBatches(self, io_batches, fig_dir, dataset=None):
        for i, batch in enumerate(io_batches):
            preds, scores, inputs, labels, seq_id = batch

            inputs = np.moveaxis(inputs.cpu().numpy(), 1, -1)
            inputs[inputs > 1] = 1

            scores = np.moveaxis(scores.cpu().numpy(), 1, -1)
            scores[scores > 1] = 1

            imageprocessing.displayImages(
                *inputs, *scores, num_rows=2,
                file_path=os.path.join(fig_dir, f"{i}.png")
            )


class ImageClassifier(torch.nn.Module):
    def __init__(self, out_set_size, feature_layer=-1, pretrained_model=None):
        super(ImageClassifier, self).__init__()

        if pretrained_model is None:
            pretrained_model = torchvision.models.resnet18(pretrained=True, progress=True)
            feature_dim = 512

        self.layers = list(pretrained_model.children())[:feature_layer]
        self.feature_extractor = torch.nn.Sequential(*self.layers)

        self.classifier = torch.nn.Linear(feature_dim, out_set_size)

    def forward(self, images):
        features = self.feature_extractor(images)
        features = torch.flatten(features, start_dim=1)
        output = self.classifier(features)
        return output

    def predict(self, output):
        __, preds = torch.max(output, -1)
        return preds


class AssemblyClassifier(torch.nn.Module):
    def __init__(self, vocab, shrink_by=3, num_rotation_samples=36, template_batch_size=None):
        super().__init__()

        intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()
        camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()
        colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

        intrinsic_matrix[:2, 2] /= shrink_by
        self.renderer = render.TorchSceneRenderer(
            intrinsic_matrix=intrinsic_matrix,
            camera_pose=camera_pose,
            colors=colors,
            light_intensity_ambient=1,
            image_size=render.IMAGE_WIDTH // shrink_by,
            orig_size=render.IMAGE_WIDTH // shrink_by
        )

        self.vocab = vocab

        angles = torch.arange(num_rotation_samples).float() * (2 * np.pi / num_rotation_samples)
        rotations = Rotation.from_euler('Z', angles)
        t = torch.stack((torch.zeros_like(angles),) * 3, dim=0).float().cuda()
        R = torch.tensor(rotations.as_dcm()).permute(1, 2, 0).float().cuda()

        self.template_vocab = tuple(a for a in vocab if len(a.connected_components) == 1)

        # FIXME: SOME RENDERED IMAGES HAVE CHANNEL VALUES > 1.0
        self.templates = torch.stack(
            tuple(self.renderTemplates(self.renderer, a, t, R)[0] for a in self.template_vocab),
            dim=0
        )

        # assemblies, poses, height, width, channels -->
        # assemblies, poses, channels, height, width
        self.templates = self.templates.permute(0, 1, 4, 2, 3).contiguous().cpu()

        if template_batch_size is None:
            template_batch_size = self.templates.shape[0]
        self.template_batch_size = template_batch_size
        self.num_template_batches = math.ceil(self.templates.shape[0] / self.template_batch_size)

        self.dummy = torch.nn.Parameter(torch.tensor([]), requires_grad=True)

        self._index = 0

    def renderTemplates(self, renderer, assembly, t, R):
        if R.shape[-1] != t.shape[-1]:
            err_str = f"R shape {R.shape} doesn't match t shape {t.shape}"
            raise AssertionError(err_str)

        num_templates = R.shape[-1]

        component_poses = ((np.eye(3), np.zeros(3)),)
        assembly = assembly.setPose(component_poses, in_place=False)

        init_vertices = render.makeBatch(assembly.vertices, dtype=torch.float).cuda()
        faces = render.makeBatch(assembly.faces, dtype=torch.int).cuda()
        textures = render.makeBatch(assembly.textures, dtype=torch.float).cuda()

        vertices = torch.einsum('nvj,jit->nvit', [init_vertices, R]) + t
        vertices = vertices.permute(-1, 0, 1, 2)

        faces = faces.expand(num_templates, *faces.shape)
        textures = textures.expand(num_templates, *textures.shape)

        rgb_images_obj, depth_images_obj = renderer.render(
            torch.reshape(vertices, (-1, *vertices.shape[2:])),
            torch.reshape(faces, (-1, *faces.shape[2:])),
            torch.reshape(textures, (-1, *textures.shape[2:]))
        )
        rgb_images_scene, depth_images_scene, label_images_scene = render.reduceByDepth(
            torch.reshape(rgb_images_obj, vertices.shape[:2] + rgb_images_obj.shape[1:]),
            torch.reshape(depth_images_obj, vertices.shape[:2] + depth_images_obj.shape[1:]),
        )

        return rgb_images_scene, depth_images_scene

    def predict(self, outputs):
        __, preds = torch.max(outputs, -1)
        return preds

    def forward(self, inputs):
        """
        inputs : shape (1, batch, in_channels, img_height, img_width)
        templates: shape (num_states, num_poses, in_channels, kernel_height, kernel_width)
        outputs : shape (1, batch, num_states)
        """

        inputs = inputs[0]

        def conv_template_batch(batch_index):
            start = batch_index * self.template_batch_size
            end = start + self.template_batch_size
            templates = self.templates[start:end].cuda()
            outputs = torch.nn.functional.conv2d(
                inputs,
                torch.reshape(templates, (-1, *templates.shape[2:]))
            )
            outputs = torch.reshape(
                outputs,
                (inputs.shape[:1] + templates.shape[0:2] + outputs.shape[2:])
            )
            # Pick the best template for each state
            outputs, argmaxima = torch.max(outputs, 2)
            return outputs

        outputs = torch.cat(
            tuple(conv_template_batch(i) for i in range(self.num_template_batches)),
            dim=1
        )

        # Pick the best location for each template
        outputs, _ = torch.max(outputs, -1)
        outputs, _ = torch.max(outputs, -1)

        outputs = outputs[None, ...]

        return outputs


class PretrainedClassifier(torch.nn.Module):
    def __init__(self, pretrained_model, debug_fig_dir=None):
        super().__init__()
        self.model = pretrained_model
        self.debug_fig_dir = debug_fig_dir

    def forward(self, inputs):
        inputs = inputs[0]

        outputs = self.model(inputs)

        if self.debug_fig_dir is not None:
            self.model.plotBatches(
                [(None, outputs[-10:], inputs[-10:], None, None)],
                self.debug_fig_dir
            )

        outputs = outputs[None, ...]
        return outputs


# -=( HELPER FUNCTIONS )==-----------------------------------------------------
def renderTemplates(renderer, assembly, t, R):
    t = t.permute(1, 0)
    R = R.permute(2, 1, 0)

    if R.shape[-1] != t.shape[-1]:
        err_str = f"R shape {R.shape} doesn't match t shape {t.shape}"
        raise AssertionError(err_str)

    num_templates = R.shape[-1]

    component_poses = tuple(
        (np.eye(3), np.zeros(3))
        for k in assembly.connected_components.keys()
    )
    assembly = assembly.setPose(component_poses, in_place=False)

    init_vertices = render.makeBatch(assembly.vertices, dtype=torch.float).cuda()
    faces = render.makeBatch(assembly.faces, dtype=torch.int).cuda()
    textures = render.makeBatch(assembly.textures, dtype=torch.float).cuda()

    vertices = torch.einsum('nvj,jit->nvit', [init_vertices, R]) + t
    vertices = vertices.permute(-1, 0, 1, 2)

    faces = faces.expand(num_templates, *faces.shape)
    textures = textures.expand(num_templates, *textures.shape)

    rgb_images_obj, depth_images_obj = renderer.render(
        torch.reshape(vertices, (-1, *vertices.shape[2:])),
        torch.reshape(faces, (-1, *faces.shape[2:])),
        torch.reshape(textures, (-1, *textures.shape[2:]))
    )
    rgb_images_scene, depth_images_scene, label_images_scene = render.reduceByDepth(
        torch.reshape(rgb_images_obj, vertices.shape[:2] + rgb_images_obj.shape[1:]),
        torch.reshape(depth_images_obj, vertices.shape[:2] + depth_images_obj.shape[1:]),
        max_depth=renderer.far
    )

    return rgb_images_scene, depth_images_scene, label_images_scene


def makeCrops(rgb_image, label_image):
    def makeCrop(i):
        crop = np.zeros((128, 128, 3))

        img_coords = np.column_stack(np.where(label_image == i))

        seg_center = img_coords.mean(axis=0).astype(int)
        crop_center = np.array([x // 2 for x in crop.shape[:2]])

        crop_coords = img_coords - seg_center + crop_center
        coords_ok = np.all((crop_coords > 0) * (crop_coords < 128), axis=1)
        crop_coords = crop_coords[coords_ok, :]
        img_coords = img_coords[coords_ok, :]

        c_rows = crop_coords[:, 0]
        c_cols = crop_coords[:, 1]
        i_rows = img_coords[:, 0]
        i_cols = img_coords[:, 1]
        crop[c_rows, c_cols] = rgb_image[i_rows, i_cols]

        return crop

    if not label_image.any():
        return np.zeros((1, 128, 128, 3))

    fg_label_image = np.unique(label_image[label_image != 0])
    rgb_image_crops = np.stack(tuple(makeCrop(i) for i in fg_label_image))
    return rgb_image_crops


def _crop(rgb_image, label_image, image_size):
    crop = torch.zeros((image_size, image_size, 3), device=rgb_image.device)

    img_coords = torch.nonzero(label_image)
    seg_center = img_coords.float().mean(dim=0).to(dtype=torch.long)
    crop_center = torch.tensor(
        [x // 2 for x in crop.shape[:2]],
        dtype=torch.long, device=seg_center.device
    )

    crop_coords = img_coords - seg_center + crop_center
    coords_ok = torch.all((crop_coords > 0) * (crop_coords < image_size), dim=1)
    crop_coords = crop_coords[coords_ok, :]
    img_coords = img_coords[coords_ok, :]

    c_rows = crop_coords[:, 0]
    c_cols = crop_coords[:, 1]
    i_rows = img_coords[:, 0]
    i_cols = img_coords[:, 1]
    crop[c_rows, c_cols] = rgb_image[i_rows, i_cols]
    return crop


def _sample_uniform(num_samples, lower, upper):
    if isinstance(lower, int) or isinstance(lower, float):
        lower = torch.tensor([lower], dtype=torch.float)

    if isinstance(upper, int) or isinstance(upper, float):
        upper = torch.tensor([upper], dtype=torch.float)

    num_dims = lower.shape[0]
    shift = lower
    scale = upper - lower
    samples = torch.rand(num_samples, num_dims, device=lower.device) * scale + shift
    return samples


def _sample_normal(num_samples, mean, std):
    if isinstance(mean, int) or isinstance(mean, float):
        mean = torch.tensor([mean], dtype=torch.float)

    if isinstance(std, int) or isinstance(std, float):
        std = torch.tensor([std], dtype=torch.float)

    num_dims = mean.shape[0]
    samples = torch.randn(num_samples, num_dims, device=mean.device) * std + mean
    return samples


def viz_model_params(model, templates_dir):
    templates = model.templates.cpu().numpy()
    # FIXME: SOME RENDERED IMAGES HAVE CHANNEL VALUES > 1.0
    templates[templates > 1] = 1

    for i, assembly_templates in enumerate(templates):
        imageprocessing.displayImages(
            *assembly_templates, num_rows=6, figsize=(15, 15),
            file_path=os.path.join(templates_dir, f"{i}.png")
        )
