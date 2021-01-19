import os
import collections
import logging

import yaml
import torch
import numpy as np

from mathtools import utils, torchutils, metrics
from visiontools import imageprocessing
from kinemparse import sim2real


logger = logging.getLogger(__name__)


class VideoLoader(utils.FeaturelessCvDataset):
    def __init__(self, trial_ids, data_dir, seg_dir, **kwargs):
        super().__init__(trial_ids, data_dir, **kwargs)

        self.seg_dir = seg_dir

        self.trial_ids = np.array([
            trial_id for trial_id, l_seq in zip(self.trial_ids, self.label_seqs)
            if l_seq is not None
        ])

        self.label_seqs = tuple(
            l_seq for l_seq in self.label_seqs
            if l_seq is not None
        )

    def _load_labels(self, label_fn_format="assembly-seq"):
        labels = tuple(
            self.loadAssemblies(t_id, label_fn_format, self.vocab)
            for t_id in self.trial_ids
        )
        return labels

    def loadAssemblies(self, seq_id, var_name, vocab, prefix='trial='):
        assembly_seq = utils.loadVariable(f"{prefix}{seq_id}_{var_name}", self.data_dir)
        labels = np.zeros(assembly_seq[-1].end_idx, dtype=int)
        for assembly in assembly_seq:
            i = utils.getIndex(assembly, vocab)
            labels[assembly.start_idx:assembly.end_idx] = i
        return labels

    def loadData(self, seq_id):
        rgb_frames = utils.loadVariable(f"trial={seq_id}_rgb-frame-seq.", self.data_dir)
        seg_frames = utils.loadVariable(f"trial={seq_id}_seg-labels-seq", self.seg_dir).astype(int)
        return rgb_frames, seg_frames


def plot_topk(model, test_io_history, num_disp_imgs, file_path):
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


def main(
        out_dir=None, data_dir=None, segs_dir=None, pretrained_model_dir=None,
        model_name=None, gpu_dev_id=None, batch_size=None, learning_rate=None,
        start_from=None, stop_at=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        num_disp_imgs=None, viz_templates=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    segs_dir = os.path.expanduser(segs_dir)
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

    def saveVariable(var, var_name, to_dir=out_data_dir):
        utils.saveVariable(var, var_name, to_dir)

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)
    vocab = utils.loadVariable('vocab', pretrained_model_dir)
    parts_vocab = utils.loadVariable('parts-vocab', pretrained_model_dir)
    edge_labels = utils.loadVariable('part-labels', pretrained_model_dir)
    saveVariable(vocab, 'vocab')
    saveVariable(parts_vocab, 'parts-vocab')
    saveVariable(edge_labels, 'part-labels')

    # Define cross-validation folds
    data_loader = VideoLoader(
        trial_ids, data_dir, segs_dir, vocab=vocab,
        label_fn_format='assembly-seq'
    )
    cv_folds = utils.makeDataSplits(len(data_loader.trial_ids), **cv_params)

    Dataset = sim2real.BlocksConnectionDataset
    device = torchutils.selectDevice(gpu_dev_id)
    label_dtype = torch.long
    labels_dtype = torch.long  # FIXME
    criterion = torch.nn.CrossEntropyLoss()

    def make_dataset(labels, ids, batch_mode='sample', shuffle=True):
        dataset = Dataset(
            vocab, edge_labels, label_dtype, data_loader.loadData, labels,
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
        train_set, train_loader = make_dataset(*train_data, batch_mode='sample', shuffle=True)
        test_set, test_loader = make_dataset(*test_data, batch_mode='flatten', shuffle=False)
        val_set, val_loader = make_dataset(*val_data, batch_mode='sample', shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(data_loader.trial_ids)} total '
            f'({len(train_set)} train, {len(val_set)} val, {len(test_set)} test)'
        )

        if model_name == 'template':
            model = sim2real.AssemblyClassifier(vocab, **model_params)
        elif model_name == 'pretrained':
            pretrained_model = utils.loadVariable("cvfold=0_model-best", pretrained_model_dir)
            model = sim2real.SceneClassifier(pretrained_model)
            metric_names = ('Loss', 'Accuracy', 'Precision', 'Recall', 'F1')
            # criterion = torch.nn.BCEWithLogitsLoss()
            # criterion = torchutils.BootstrappedCriterion(
            #     0.25, base_criterion=torch.nn.functional.binary_cross_entropy_with_logits,
            # )
            # criterion = torch.nn.CrossEntropyLoss()
            criterion = torchutils.BootstrappedCriterion(
                0.25, base_criterion=torch.nn.functional.cross_entropy,
            )
        else:
            raise AssertionError()

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

        if model_name == 'pretrained' and num_disp_imgs is not None:
            cvfold_dir = os.path.join(io_dir, f'cvfold={cv_index}')
            if not os.path.exists(cvfold_dir):
                os.makedirs(cvfold_dir)
            model.plotBatches(
                test_io_history, cvfold_dir,
                images_per_fig=num_disp_imgs, dataset=test_set
            )

        if model_name == 'template' and num_disp_imgs is not None:
            io_dir = os.path.join(fig_dir, 'model-io')
            if not os.path.exists(io_dir):
                os.makedirs(io_dir)
            plot_topk(
                model, test_io_history, num_disp_imgs,
                os.path.join(io_dir, f"cvfold={cv_index}.png")
            )

        if viz_templates:
            sim2real.viz_model_params(model, templates_dir=None)


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
