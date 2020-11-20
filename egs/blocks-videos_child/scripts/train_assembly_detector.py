import os
import collections
import logging

import yaml
import torch
import torchvision
import kornia
import joblib
import numpy as np
from scipy.spatial.transform import Rotation

from mathtools import utils, metrics, torchutils, torch_future
from visiontools import render, imageprocessing


logger = logging.getLogger(__name__)


class ConnectionClassifier(torch.nn.Module):
    def __init__(self, out_dim, feature_dim, feature_extractor):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.fc = torch.nn.Linear(feature_dim, out_dim)

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        outputs = self.fc(features)
        return outputs

    def predict(self, outputs):
        return (torch.sigmoid(outputs) > 0.5).float()

    def plotBatches(self, io_batches, fig_dir, dataset=None):
        for i, batch in enumerate(io_batches):
            preds, scores, inputs, labels, seq_id = batch

            num_batch = preds.shape[0]
            scores = scores.view(num_batch, *dataset.target_shape).cpu().numpy()
            preds = preds.view(num_batch, *dataset.target_shape).cpu().numpy()
            labels = labels.view(num_batch, *dataset.target_shape).cpu().numpy()
            inputs = np.moveaxis(inputs.cpu().numpy(), 1, -1)
            inputs[inputs > 1] = 1

            imageprocessing.displayImages(
                *inputs, *scores, *preds, *labels, num_rows=4,
                file_path=os.path.join(fig_dir, f"{i}.png")
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


class RenderDataset(torch.utils.data.Dataset):
    def __init__(
            self, vocab, device=None, batch_size=None,
            intrinsic_matrix=None, camera_pose=None, colors=None, image_size=128):
        if intrinsic_matrix is None:
            intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()

        if camera_pose is None:
            camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()

        if colors is None:
            colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()

        self.image_size = image_size

        self.renderer = render.TorchSceneRenderer(
            intrinsic_matrix=intrinsic_matrix,
            camera_pose=camera_pose,
            colors=colors,
            light_intensity_ambient=1,
            image_size=max(render.IMAGE_WIDTH, render.IMAGE_HEIGHT),
            orig_size=image_size
        )

        # FIXME: handle multi-component assemblies properly
        self.vocab = tuple(a for a in vocab if len(a.connected_components) == 1)

        self.device = device
        self.batch_size = batch_size

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, i):
        assembly = self.vocab[i]
        R, t = self.samplePose()
        rgb_batch, depth_batch, label_batch = renderTemplates(self.renderer, assembly, t, R)
        image = _crop(rgb_batch[0], label_batch[0], self.image_size)
        return image, i, -1

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
             [render.IMAGE_WIDTH, render.IMAGE_HEIGHT]],
            dtype=torch.float
        ).cuda()
        depth_coords = torch.full(px_bounds.shape[0:1], camera_dist)[..., None].cuda()
        # FIXME: kornia expects K of shape (B, 4, 4), but input has shape (3, 3)
        xy_bounds = kornia.unproject_points(
            px_bounds, depth_coords, self.renderer.K
        ).transpose(0, 1)

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


class DenoisingDataset(RenderDataset):
    def __getitem__(self, i):
        assembly = self.vocab[i]

        R, t = self.samplePose()
        rgb_batch, depth_batch, label_batch = renderTemplates(self.renderer, assembly, t, R)
        randomized_image = _crop(rgb_batch[0], label_batch[0], self.image_size).permute(-1, 0, 1)

        R, t = self.canonicalPose()
        rgb_batch, depth_batch, label_batch = renderTemplates(self.renderer, assembly, t, R)
        canonical_image = _crop(rgb_batch[0], label_batch[0], self.image_size).permute(-1, 0, 1)

        return randomized_image, canonical_image, -1

    def canonicalPose(self, num_samples=1):
        angles = torch.zeros(num_samples, dtype=torch.float)
        rotations = Rotation.from_euler('Z', angles)
        R = torch.tensor(rotations.as_matrix()).float().cuda()
        t = torch.stack((torch.zeros_like(angles),) * 3, dim=1).float().cuda()
        return R, t


class ConnectionDataset(RenderDataset):
    def __getitem__(self, i):
        assembly = self.vocab[i]

        R, t = self.samplePose()
        rgb_batch, depth_batch = renderTemplates(self.renderer, assembly, t, R)
        randomized_image = rgb_batch[0].permute(-1, 0, 1)

        connections = torch.tensor(assembly.connections, dtype=torch.float).view(-1).contiguous()

        return randomized_image, connections, -1

    @property
    def target_shape(self):
        assembly = self.vocab[0]
        return assembly.connections.shape


def renderTemplates(renderer, assembly, t, R):
    t = t.permute(1, 0)
    R = R.permute(1, 2, 0)

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

    return rgb_images_scene, depth_images_scene, label_images_scene


def makeMetric(name):
    if name == 'Reciprocal Loss':
        return metrics.ReciprocalAverageLoss()
    elif name == 'Loss':
        return metrics.AverageLoss()
    elif name == 'Accuracy':
        return metrics.Accuracy()
    else:
        raise AssertionError()


def main(
        out_dir=None, data_dir=None, pretrain_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        num_disp_imgs=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    if pretrain_dir is not None:
        pretrain_dir = os.path.expanduser(pretrain_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    io_dir = os.path.join(fig_dir, 'model-io')
    if not os.path.exists(io_dir):
        os.makedirs(io_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadAssemblies(seq_id, vocab):
        assembly_seq = joblib.load(os.path.join(data_dir, f"trial={seq_id}_assembly-seq.pkl"))
        labels = np.array([utils.getIndex(assembly, vocab) for assembly in assembly_seq])
        return labels

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadVariable(var_name):
        return joblib.load(os.path.join(pretrain_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    vocab = []
    label_seqs = tuple(loadAssemblies(t_id, vocab) for t_id in trial_ids)

    logger.info(f"Loaded {len(trial_ids)} sequences; {len(vocab)} unique assemblies")

    trial_ids = np.array([
        trial_id for trial_id, l_seq in zip(trial_ids, label_seqs)
        if l_seq is not None
    ])
    label_seqs = tuple(
        l_seq for l_seq in label_seqs
        if l_seq is not None
    )

    device = torchutils.selectDevice(gpu_dev_id)

    if model_name == 'AAE':
        dataset = DenoisingDataset
    elif model_name == 'Resnet':
        dataset = RenderDataset
    elif model_name == 'Connections':
        dataset = ConnectionDataset

    for cv_index, cv_splits in enumerate(range(1)):
        train_set = dataset(vocab, device=device)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_set = dataset(vocab, device=device)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

        val_set = dataset(vocab, device=device)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        if model_name == 'AAE':
            model = AugmentedAutoEncoder(train_set.data_shape, train_set.num_classes)
            criterion = torchutils.BootstrappedCriterion(
                0.25, base_criterion=torch.nn.functional.mse_loss
            )
            metric_names = ('Reciprocal Loss',)
        elif model_name == 'Resnet':
            model = ImageClassifier(train_set.num_classes)
            criterion = torch.nn.CrossEntropy()
            metric_names = ('Loss', 'Accuracy')
        elif model_name == 'Connections':
            feature_dim = 128
            aae = loadVariable('cvfold=0_AAE-best')
            model = ConnectionClassifier(train_set.label_shape[0], feature_dim, aae.encoder)
            criterion = torch.nn.BCEWithLogitsLoss()
            metric_names = ('Loss', 'Accuracy')

        model = model.to(device=device)

        train_epoch_log = collections.defaultdict(list)
        val_epoch_log = collections.defaultdict(list)
        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

        metric_dict = {name: makeMetric(name) for name in metric_names}
        model, last_model_wts = torchutils.trainModel(
            model, criterion, optimizer_ft, lr_scheduler,
            train_loader, val_loader,
            device=device,
            metrics=metric_dict,
            train_epoch_log=train_epoch_log,
            val_epoch_log=val_epoch_log,
            **train_params
        )

        # Test model
        metric_dict = {name: makeMetric(name) for name in metric_names}
        test_io_batches = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )
        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq.cpu().numpy(), f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq.cpu().numpy(), f'trial={trial_id}_score-seq')
            saveVariable(label_seq.cpu().numpy(), f'trial={trial_id}_true-label-seq')
        for batch in test_io_batches:
            saveTrialData(*batch)

        cv_str = f"cvfold={cv_index}"

        saveVariable(train_epoch_log, f'{cv_str}_model-train-epoch-log')
        saveVariable(val_epoch_log, f'{cv_str}_model-val-epoch-log')
        saveVariable(metric_dict, f'{cv_str}_model-metric-dict')
        saveVariable(model, f'{cv_str}_model-best')

        model.load_state_dict(last_model_wts)
        saveVariable(model, f'{cv_str}_model-last')

        if train_epoch_log:
            torchutils.plotEpochLog(
                train_epoch_log,
                subfig_size=(10, 2.5),
                title='Training performance',
                fn=os.path.join(fig_dir, f'{cv_str}_train-plot.png')
            )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(fig_dir, f'{cv_str}_val-plot.png')
            )

        if num_disp_imgs is not None:
            model.plotBatches(test_io_batches, io_dir, dataset=test_set)


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
