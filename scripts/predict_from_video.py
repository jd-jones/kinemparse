import argparse
import os
import collections
import logging

import yaml
import torch
import joblib
import torchvision.models

from mathtools import utils, torchutils, metrics
from kinemparse import imu


logger = logging.getLogger(__name__)


class ImageClassifier(torch.nn.Module):
    def __init__(
            self, out_set_size, feature_layer=-1, pretrained_model=None, multiscale=False,
            binary_labels=False, seq_as_batch=False):
        super(ImageClassifier, self).__init__()

        self.multiscale = multiscale
        self.seq_as_batch = seq_as_batch
        self.binary_labels = binary_labels

        if pretrained_model is None:
            pretrained_model = torchvision.models.resnet18(pretrained=True, progress=True)
            feature_dim = 512

        self.layers = list(pretrained_model.children())[:feature_layer]
        self.feature_extractor = torch.nn.Sequential(*self.layers)

        self.classifier = torch.nn.Linear(feature_dim, out_set_size)

    def forward(self, images, seq_as_batch=None):
        # If the input has shape (batch_size, 3, 240, 320), then features will
        # have shape (batch_size, 512, 8, 10)

        if seq_as_batch is None:
            seq_as_batch = self.seq_as_batch

        if seq_as_batch:
            return self.forward(images[0], seq_as_batch=False)[None, ...]

        if self.multiscale:
            features = []
            feature = images
            for layer in self.layers:
                feature = layer(feature)
                features.append(feature)
        else:
            features = self.feature_extractor(images)

        features = torch.flatten(features, start_dim=1)
        output = self.classifier(features)
        return output

    def predict(self, output):
        if self.binary_labels:
            return (output > 0.5).float()
        __, preds = torch.max(output, -1)
        return preds


def main(
        out_dir=None, data_dir=None, attr_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        plot_predictions=None, results_file=None, sweep_param_name=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    attr_dir = os.path.expanduser(attr_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, f'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadData(seq_id):
        var_name = f"trial-{seq_id}_rgb-frame-seq"
        data = joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))
        return data.swapaxes(1, 3)

    def loadLabels(seq_id):
        var_name = f"trial-{seq_id}_label-seq"
        return joblib.load(os.path.join(attr_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir)
    label_seqs = tuple(map(loadLabels, trial_ids))

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

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        criterion = torch.nn.BCEWithLogitsLoss()
        labels_dtype = torch.float

        train_labels, train_ids = train_data
        train_set = torchutils.PickledVideoDataset(
            loadData, train_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=train_ids,
            batch_size=batch_size
        )
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

        test_labels, test_ids = test_data
        test_set = torchutils.PickledVideoDataset(
            loadData, test_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=test_ids,
            batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

        val_labels, val_ids = val_data
        val_set = torchutils.PickledVideoDataset(
            loadData, val_labels,
            device=device, labels_dtype=labels_dtype, seq_ids=val_ids,
            batch_size=batch_size
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        if model_name == 'resnet':
            # input_dim = train_set.num_obsv_dims
            output_dim = train_set.num_label_types
            model = ImageClassifier(output_dim, **model_params).to(device=device)
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

        if plot_predictions:
            # imu.plot_prediction_eg(test_io_history, fig_dir, fig_type=fig_type, **viz_params)
            imu.plot_prediction_eg(test_io_history, fig_dir, **viz_params)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq.cpu().numpy(), f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq.cpu().numpy(), f'trial={trial_id}_score-seq')
            saveVariable(label_seq.cpu().numpy(), f'trial={trial_id}_true-label-seq')
        for io in test_io_history:
            saveTrialData(*io)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        model.load_state_dict(last_model_wts)
        saveVariable(model, f'cvfold={cv_index}_{model_name}-last')

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
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--model_params')
    parser.add_argument('--results_file')
    parser.add_argument('--sweep_param_name')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
