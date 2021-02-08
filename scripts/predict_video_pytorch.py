import os
import collections
import logging

import yaml
import torch
import torchvision
import numpy as np
from skimage import io

from mathtools import utils, torchutils, metrics


logger = logging.getLogger(__name__)


class ImageClassifier(torch.nn.Module):
    def __init__(
            self, out_dim,
            feature_dim=None, feature_extractor=None, finetune_extractor=True,
            feature_extractor_name='resnet50', feature_extractor_layer=-1):
        super().__init__()

        self.out_dim = out_dim

        if feature_extractor is None:
            Extractor = getattr(torchvision.models, feature_extractor_name)
            pretrained_model = Extractor(pretrained=True, progress=True)
            layers = list(pretrained_model.children())[:feature_extractor_layer]
            feature_extractor = torch.nn.Sequential(*layers)
            feature_dim = 512

        if not finetune_extractor:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        self.feature_extractor = feature_extractor
        self.classifier = torch.nn.Linear(feature_dim, out_dim)

    def forward(self, inputs):
        features = self.feature_extractor(inputs).squeeze(-1).squeeze(-1)
        outputs = self.classifier(features)
        return outputs

    def predict(self, outputs):
        preds = outputs.argmax(dim=1)
        return preds


class VideoDataset(torch.utils.data.Dataset):
    """ A dataset wrapping images stored on disk.

    Attributes
    ----------
    _data : tuple(np.ndarray, shape (num_samples, num_dims))
    _labels : tuple(np.ndarray, shape (num_samples,))
    _device : torch.Device
    """

    def __init__(
            self, frame_fns, labels, device=None, labels_dtype=None,
            transpose_data=False, seq_ids=None, batch_size=None, batch_mode=None):
        """
        Parameters
        ----------
        frame_fns : iterable( array_like of string, len (sequence_len) )
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

        self.transpose_data = transpose_data
        self.batch_size = batch_size
        self.batch_mode = batch_mode

        self._device = device
        self._seq_ids = seq_ids

        self._frame_fns = frame_fns
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

        logger.info('Initialized VideoDataset.')
        logger.info(f"{self.num_label_types} unique labels")

    def __len__(self):
        if self.batch_mode == 'flatten':
            return len(self.unflatten)
        return len(self._seq_ids)

    def __getitem__(self, i):
        if self.batch_size is not None:
            seq_idx, win_idx = self.unflatten[i]

            label_seq = self._labels[seq_idx]
            frame_fns = self._frame_fns[seq_idx]

            start_idx = win_idx
            end_idx = start_idx + self.batch_size
            frame_fns = frame_fns[start_idx:end_idx]
            label_seq = label_seq[start_idx:end_idx]
        else:
            label_seq = self._labels[i]
            frame_fns = self._frame_fns[i]

        data_seq = torch.tensor(
            self._load_frames(frame_fns),
            device=self._device, dtype=torch.float
        )

        # shape (batch_size, num_rows, num_cols, num_channels) -->
        #       (batch_size, num_channels, num_rows, num_cols)
        data_seq = data_seq.permute(0, 3, 1, 2)

        return data_seq, label_seq, i

    def _load_frames(self, frame_fns):
        frames = np.stack(tuple(io.imread(fn) for fn in frame_fns), axis=0)
        return frames


def main(
        out_dir=None, data_dir=None, prefix='trial=',
        model_name=None, gpu_dev_id=None, batch_size=None, learning_rate=None,
        file_fn_format=None, label_fn_format=None,
        start_from=None, stop_at=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        num_disp_imgs=None, viz_templates=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
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

    def saveVariable(var, var_name, to_dir=out_data_dir):
        utils.saveVariable(var, var_name, to_dir)

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix=prefix, to_array=True)
    vocab = utils.loadVariable('vocab', data_dir)
    saveVariable(vocab, 'vocab')

    # Define cross-validation folds
    data_loader = utils.CvDataset(
        trial_ids, data_dir, vocab=vocab, prefix=prefix,
        feature_fn_format=file_fn_format, label_fn_format=label_fn_format
    )
    cv_folds = utils.makeDataSplits(len(data_loader.trial_ids), **cv_params)

    device = torchutils.selectDevice(gpu_dev_id)
    labels_dtype = torch.long
    criterion = torch.nn.CrossEntropyLoss()
    metric_names = ('Loss', 'Accuracy')

    def make_dataset(fns, labels, ids, batch_mode='sample', shuffle=True):
        dataset = VideoDataset(
            fns, labels,
            device=device, labels_dtype=labels_dtype, seq_ids=ids,
            batch_size=batch_size, batch_mode=batch_mode
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle)
        return dataset, loader

    for cv_index, cv_fold in enumerate(cv_folds):
        if start_from is not None and cv_index < start_from:
            continue

        if stop_at is not None and cv_index > stop_at:
            break

        train_data, val_data, test_data = data_loader.getFold(cv_fold)
        train_set, train_loader = make_dataset(*train_data, batch_mode='flatten', shuffle=True)
        test_set, test_loader = make_dataset(*test_data, batch_mode='flatten', shuffle=False)
        val_set, val_loader = make_dataset(*val_data, batch_mode='flatten', shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(data_loader.trial_ids)} total '
            f'({len(train_set)} train, {len(val_set)} val, {len(test_set)} test)'
        )

        model = ImageClassifier(len(vocab), **model_params)

        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

        train_epoch_log = collections.defaultdict(list)
        val_epoch_log = collections.defaultdict(list)
        metric_dict = {name: metrics.makeMetric(name) for name in metric_names}
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
        metric_dict = {name: metrics.makeMetric(name) for name in metric_names}
        test_io_history = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )
        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        utils.writeResults(
            results_file, {name: m.value for name, m in metric_dict.items()},
            sweep_param_name, model_params
        )

        for pred_seq, score_seq, feat_seq, label_seq, batch_id in test_io_history:
            prefix = f'cvfold={cv_index}_batch={batch_id}'
            saveVariable(pred_seq.cpu().numpy(), f'{prefix}_pred-label-seq')
            saveVariable(score_seq.cpu().numpy(), f'{prefix}_score-seq')
            saveVariable(label_seq.cpu().numpy(), f'{prefix}_true-label-seq')
        saveVariable(test_set.unflatten, f'cvfold={cv_index}_test-set-unflatten')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

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
