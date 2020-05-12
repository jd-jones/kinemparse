import argparse
import os
import logging

import yaml
import joblib
import numpy as np
from sklearn import metrics

from mathtools import utils
# from seqtools import fsm


logger = logging.getLogger(__name__)


def removeSmallSegments(labels, min_seg_len=10):
    seg_labels, seg_lens = utils.computeSegments(labels)

    seg_label_seq = np.zeros_like(labels)

    prev_start_index = 0
    prev_seg_len = 0
    prev_seg_label = -1
    for seg_label, seg_len in zip(seg_labels, seg_lens):

        if seg_len < min_seg_len:
            prev_seg_len += seg_len
            continue

        if seg_label == prev_seg_label:
            prev_seg_len += seg_len
            continue

        prev_end_index = prev_start_index + prev_seg_len
        seg_label_seq[prev_start_index:prev_end_index] = prev_seg_label

        prev_start_index = prev_end_index
        prev_seg_len = seg_len
        prev_seg_label = seg_label
    prev_end_index = prev_start_index + prev_seg_len
    seg_label_seq[prev_start_index:prev_end_index] = prev_seg_label

    return seg_label_seq


def filterSegments(label_seq, **filter_args):
    label_seq = label_seq.copy()
    label_seq[label_seq == 2] = 0
    label_seq[label_seq == 3] = 0

    label_seq = label_seq.copy()
    for i in range(label_seq.shape[0]):
        label_seq[i] = removeSmallSegments(label_seq[i], **filter_args)
    return label_seq


def segmentFromLabels(label_seq, num_vals=2, min_seg_len=10):
    row_nums = num_vals ** np.arange(label_seq.shape[0])
    reduced_label_seq = np.dot(row_nums, label_seq)

    reduced_label_seq = removeSmallSegments(reduced_label_seq, min_seg_len=30)

    seg_labels, seg_lens = utils.computeSegments(reduced_label_seq)

    seg_label_seq = np.zeros_like(reduced_label_seq)
    start_index = 0
    for i, seg_len in enumerate(seg_lens):
        end_index = start_index + seg_len
        seg_label_seq[start_index:end_index] = i
        start_index = end_index

    return seg_label_seq


def main(
        out_dir=None, data_dir=None,
        model_name=None, model_params={},
        results_file=None, sweep_param_name=None,
        cv_params={}, viz_params={},
        plot_predictions=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

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

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=')
    feature_seqs = tuple(
        joblib.load(
            os.path.join(data_dir, f'trial={trial_id}_pred-label-seq.pkl')
        )
        for trial_id in trial_ids
    )
    label_seqs = tuple(
        joblib.load(
            os.path.join(data_dir, f'trial={trial_id}_true-label-seq.pkl')
        )
        for trial_id in trial_ids
    )

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, label_seqs, trial_ids)
        )
        return split_data

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        train_ids = train_data[-1]
        test_ids = test_data[-1]
        val_ids = val_data[-1]
        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        metric_dict = {
            'adjusted_rand_score': [],
        }

        for feature_seq, label_seq, trial_id in zip(*test_data):
            label_seq = filterSegments(label_seq)
            pred_label_seq = filterSegments(feature_seq)
            gt_seg_label_seq = segmentFromLabels(label_seq, num_vals=4)
            pred_seg_label_seq = segmentFromLabels(pred_label_seq, num_vals=4)
            for name in metric_dict.keys():
                value = getattr(metrics, name)(gt_seg_label_seq, pred_seg_label_seq)
                metric_dict[name].append(value)

            if plot_predictions:
                fn = os.path.join(fig_dir, f'trial-{trial_id:03}.png')
                labels = (gt_seg_label_seq, label_seq, pred_seg_label_seq, pred_label_seq)
                label_names = ('gt segments', 'labels', 'pred segments', 'pred labels')
                utils.plot_array(feature_seq, labels, label_names, fn=fn)

            # saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
            # saveVariable(score_seq, f'trial={trial_id}_score-seq')
            # saveVariable(label_seq, f'trial={trial_id}_true-label-seq')

        for name in metric_dict.keys():
            metric_dict[name] = np.array(value).mean()
        metric_str = '  '.join(f"{k}: {v * 100:.1f}%" for k, v in metric_dict.items())
        logger.info('[TST]  ' + metric_str)
        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        # saveVariable(model, f'cvfold={cv_index}_{model_name}-best')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--scores_dir')
    parser.add_argument('--model_params')
    parser.add_argument('--results_file')
    parser.add_argument('--sweep_param_name')

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    else:
        config_fn = os.path.basename(config_file_path)
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
