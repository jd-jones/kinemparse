import os
import collections
import logging

import yaml
import torch
import joblib
import numpy as np

from mathtools import utils, torchutils, metrics
from visiontools import imageprocessing
from kinemparse import sim2real


logger = logging.getLogger(__name__)


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

    def loadData(seq_id):
        rgb_frames = joblib.load(os.path.join(data_dir, f"trial={seq_id}_rgb-frame-seq.pkl"))
        seg_frames = joblib.load(
            os.path.join(segs_dir, f"trial={seq_id}_seg-labels-seq.pkl")
        ).astype(int)
        seg_frames[:, :, :110] = 0
        return rgb_frames, seg_frames

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

    dataset = sim2real.BlocksConnectionDataset

    for cv_index, cv_splits in enumerate(cv_folds):
        if start_from is not None and cv_index < start_from:
            continue

        if stop_at is not None and cv_index > stop_at:
            break

        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        criterion = torch.nn.CrossEntropyLoss()
        labels_dtype = torch.long

        train_labels, train_ids = train_data
        train_set = dataset(
            vocab, loadData, train_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=train_ids,
            batch_size=batch_size
        )
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

        test_labels, test_ids = test_data
        test_set = dataset(
            vocab, loadData, test_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=test_ids,
            batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

        val_labels, val_ids = val_data
        val_set = dataset(
            vocab, loadData, val_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=val_ids,
            batch_size=batch_size
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        if model_name == 'template':
            model = sim2real.AssemblyClassifier(vocab, **model_params)
        elif model_name == 'pretrained':
            pretrained_model = joblib.load(
                os.path.join(pretrained_model_dir, "cvfold=0_model-best.pkl")
            )
            model = sim2real.SceneClassifier(pretrained_model)
            metric_names = ('Loss', 'Accuracy', 'Precision', 'Recall', 'F1')
            criterion = torch.nn.BCEWithLogitsLoss()
            criterion = torchutils.BootstrappedCriterion(
                0.25, base_criterion=torch.nn.functional.binary_cross_entropy_with_logits,
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

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, batch_id):
            saveVariable(
                pred_seq.cpu().numpy(),
                f'cvfold={cv_index}_batch={batch_id}_pred-label-seq'
            )
            saveVariable(
                score_seq.cpu().numpy(),
                f'cvfold={cv_index}_batch={batch_id}_score-seq'
            )
            saveVariable(
                label_seq.cpu().numpy(),
                f'cvfold={cv_index}_batch={batch_id}_true-label-seq'
            )
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
