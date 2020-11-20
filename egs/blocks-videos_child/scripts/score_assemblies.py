import os
import collections
import logging
import math

import yaml
import torch
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from skimage import measure

from mathtools import utils, torchutils, metrics
from visiontools import render, imageprocessing

# FIXME: hack for loading pickled pretrained model
from train_assembly_detector import AugmentedAutoEncoder


logger = logging.getLogger(__name__)


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


class BlocksVideoDataset(torchutils.PickledVideoDataset):
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
        if self.batch_size is not None:
            seq_idx, win_idx = self.unflatten[i]

            seq_id = self._seq_ids[seq_idx]
            label_seq = self._labels[seq_idx]
            data_seqs = self._load_data(seq_id)

            start_idx = win_idx
            end_idx = start_idx + self.batch_size
            data_seqs = tuple(data_seq[start_idx:end_idx] for data_seq in data_seqs)
            label_seq = label_seq[start_idx:end_idx]
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

        return data_seq, label_seq, seq_id


class PretrainDataset(BlocksVideoDataset):
    def _transform(self, data_seqs):
        def makeCrops(rgb, mask):
            labels, num = measure.label(mask.astype(int), return_num=True)

            if not num:
                return np.zeros((1, 128, 128, 3))

            def makeCrop(i):
                crop = np.zeros((128, 128, 3))

                img_coords = np.column_stack(np.where(labels == i))

                seg_center = img_coords.mean(axis=0).astype(int)
                crop_center = np.array([x // 2 for x in crop.shape[:2]])

                crop_coords = img_coords - seg_center + crop_center
                coords_ok = np.all((crop_coords > 0) * (crop_coords < 128), axis=1)
                crop_coords = crop_coords[coords_ok, :]
                img_coords = img_coords[coords_ok, :]

                crop[crop_coords[:, 0], crop_coords[:, 1]] = rgb[img_coords[:, 0], img_coords[:, 1]]

                return crop

            return np.stack(tuple(makeCrop(i) for i in range(1, num + 1)))

        rgb_seq, bg_masks, person_masks = data_seqs

        blocks_masks = ~(bg_masks.astype(bool) + person_masks.astype(bool))
        blocks_masks[:, :, :110] = False

        rgb_seq = np.stack(tuple(
            imageprocessing.saturateImage(rgb, to_float=False)
            for rgb in rgb_seq
        ))

        rgb_seq[~blocks_masks] = 0

        rgb_seq = np.vstack(
            tuple(makeCrops(rgb, mask) for rgb, mask in zip(rgb_seq, blocks_masks))
        )

        rgb_seq = torch.tensor(
            np.moveaxis(rgb_seq, 3, 1),
            device=self._device, dtype=torch.float
        )

        return rgb_seq


def viz_model_params(model, templates_dir):
    templates = model.templates.cpu().numpy()
    # FIXME: SOME RENDERED IMAGES HAVE CHANNEL VALUES > 1.0
    templates[templates > 1] = 1

    for i, assembly_templates in enumerate(templates):
        imageprocessing.displayImages(
            *assembly_templates, num_rows=6, figsize=(15, 15),
            file_path=os.path.join(templates_dir, f"{i}.png")
        )


def main(
        out_dir=None, data_dir=None, background_dir=None, detections_dir=None,
        pretrained_model_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        num_disp_imgs=None, viz_templates=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    background_dir = os.path.expanduser(background_dir)
    detections_dir = os.path.expanduser(detections_dir)
    pretrained_model_dir = os.path.expanduser(pretrained_model_dir)
    out_dir = os.path.expanduser(out_dir)
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

    def loadData(seq_id):
        rgb_frames = joblib.load(os.path.join(data_dir, f"trial={seq_id}_rgb-frame-seq.pkl"))
        bg_masks = joblib.load(
            os.path.join(background_dir, f"trial={seq_id}_bg-mask-seq-depth.pkl")
        )
        person_masks = joblib.load(
            os.path.join(detections_dir, f"trial={seq_id}_person-mask-seq.pkl")
        )
        return rgb_frames, bg_masks, person_masks

    def loadAssemblies(seq_id, vocab):
        assembly_seq = joblib.load(os.path.join(data_dir, f"trial={seq_id}_assembly-seq.pkl"))
        labels = np.zeros(assembly_seq[-1].end_idx, dtype=int)
        for assembly in assembly_seq:
            i = utils.getIndex(assembly, vocab)
            labels[assembly.start_idx:assembly.end_idx] = i
        return labels

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    vocab = []
    label_seqs = tuple(loadAssemblies(t_id, vocab) for t_id in trial_ids)

    trial_ids = np.array([
        trial_id for trial_id, l_seq in zip(trial_ids, label_seqs)
        if l_seq is not None
    ])
    label_seqs = tuple(
        l_seq for l_seq in label_seqs
        if l_seq is not None
    )

    device = torchutils.selectDevice(gpu_dev_id)

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (label_seqs, trial_ids)
        )
        return split_data

    dataset = PretrainDataset
    # dataset = BlocksVideoDataset

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        criterion = torch.nn.CrossEntropyLoss()
        labels_dtype = torch.long

        train_labels, train_ids = train_data
        train_set = dataset(
            loadData, train_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=train_ids,
            batch_size=batch_size,
        )
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

        test_labels, test_ids = test_data
        test_set = dataset(
            loadData, test_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=test_ids,
            batch_size=batch_size,
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

        val_labels, val_ids = val_data
        val_set = dataset(
            loadData, val_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=val_ids,
            batch_size=batch_size,
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        if model_name == 'template':
            model = AssemblyClassifier(vocab, **model_params)
        elif model_name == 'pretrained':
            pretrained_model = joblib.load(
                os.path.join(pretrained_model_dir, "cvfold=0_AAE-best.pkl")
            )
            model = PretrainedClassifier(pretrained_model, debug_fig_dir=io_dir)
        else:
            raise AssertionError()

        train_epoch_log = collections.defaultdict(list)
        val_epoch_log = collections.defaultdict(list)
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy(),
            'Precision': metrics.Precision(),
            'Recall': metrics.Recall(),
            'F1': metrics.Fmeasure()
        }

        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

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
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy(),
            'Precision': metrics.Precision(),
            'Recall': metrics.Recall(),
            'F1': metrics.Fmeasure()
        }
        test_io_history = torchutils.predictSamples(
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
        for io in test_io_history:
            saveTrialData(*io)

        saveVariable(test_set.unflatten, f'cvfold={cv_index}_test-set-unflatten')

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        model.load_state_dict(last_model_wts)
        saveVariable(model, f'cvfold={cv_index}_{model_name}-last')

        if train_epoch_log:
            torchutils.plotEpochLog(
                train_epoch_log,
                subfig_size=(10, 2.5),
                title='Training performance',
                fn=os.path.join(fig_dir, f'cvfold={cv_index}_train-plot.png')
            )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(fig_dir, f'cvfold={cv_index}_val-plot.png')
            )

        if model_name == 'pretrained' and num_disp_imgs is not None:
            model.model.plotBatches(test_io_history, io_dir)

        if model_name == 'template' and num_disp_imgs is not None:
            io_dir = os.path.join(fig_dir, 'model-io')
            if not os.path.exists(io_dir):
                os.makedirs(io_dir)
            file_path = os.path.join(io_dir, f"cvfold={cv_index}.png")

            inputs = np.moveaxis(
                torch.cat(tuple(batches[2][0] for batches in test_io_history)).numpy(),
                1, -1
            )
            outputs = torch.cat(tuple(batches[1][0] for batches in test_io_history))

            if inputs.shape[0] > num_disp_imgs:
                idxs = np.arange(inputs.shape[0])
                np.random.shuffle(idxs)
                idxs = np.sort(idxs[:num_disp_imgs])
            else:
                idxs = slice(None, None, None)

            inputs = inputs[idxs]
            outputs = outputs[idxs]

            def make_templates(preds):
                pred_templates = np.moveaxis(model.templates[preds, 0].numpy(), 1, -1)
                pred_templates[pred_templates > 1] = 1
                return pred_templates

            k = 5
            __, topk_preds = torch.topk(outputs, k, dim=-1)
            topk_preds = topk_preds.transpose(0, 1).contiguous().view(-1)
            topk_templates = make_templates(topk_preds)

            imageprocessing.displayImages(
                *inputs, *topk_templates, num_rows=1 + k,
                file_path=file_path
            )

        if viz_templates:
            viz_model_params(model, templates_dir=None)


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
