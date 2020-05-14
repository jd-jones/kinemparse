import argparse
import os
import logging

import yaml
import joblib
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

from mathtools import utils


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


def plot_array(
        gt_seg_label_seq, pred_seg_label_seq, imu_timestamp_seq, keyframe_timestamp_seq,
        fn=None):
    subplot_width = 12
    subplot_height = 3

    num_axes = 1

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axis = plt.subplots(num_axes, figsize=figsize, sharex=True)

    axis.plot(imu_timestamp_seq, gt_seg_label_seq)
    axis.plot(imu_timestamp_seq, pred_seg_label_seq)
    axis.scatter(keyframe_timestamp_seq, np.zeros_like(keyframe_timestamp_seq))

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def retrievalMetrics(seg_keyframe_counts):
    centered_counts = seg_keyframe_counts - 1
    num_true_positives = np.sum(centered_counts == 0)
    num_false_positives = np.sum(centered_counts < 0)
    num_false_negatives = np.sum(centered_counts > 0)

    precision = num_true_positives / (num_true_positives + num_false_positives)
    recall = num_true_positives / (num_true_positives + num_false_negatives)
    F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, F1


def main(
        out_dir=None, predictions_dir=None, imu_data_dir=None, video_data_dir=None,
        model_name=None, model_params={},
        results_file=None, sweep_param_name=None,
        cv_params={}, viz_params={},
        plot_predictions=None):

    predictions_dir = os.path.expanduser(predictions_dir)
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

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    trial_ids = utils.getUniqueIds(predictions_dir, prefix='trial=')
    feature_seqs = loadAll(trial_ids, 'pred-label-seq.pkl', predictions_dir)
    label_seqs = loadAll(trial_ids, 'true-label-seq.pkl', predictions_dir)

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
            'ARI': [],
            'kf_prec': [],
            'kf_rec': [],
            'kf_f1': []
        }

        for feature_seq, label_seq, trial_id in zip(*test_data):
            label_seq = filterSegments(label_seq)
            pred_label_seq = filterSegments(feature_seq)
            gt_seg_label_seq = segmentFromLabels(label_seq, num_vals=4)
            pred_seg_label_seq = segmentFromLabels(pred_label_seq, num_vals=4)
            if imu_data_dir is not None and video_data_dir is not None:
                imu_data_dir = os.path.expanduser(imu_data_dir)
                imu_timestamp_seq = joblib.load(
                    os.path.join(imu_data_dir, f'trial={trial_id}_timestamp-seq.pkl')
                )

                try:
                    video_data_dir = os.path.expanduser(video_data_dir)
                    timestamp_fn = f'trial={trial_id}_rgb-frame-timestamp-seq.pkl'
                    keyframe_timestamp_seq = joblib.load(os.path.join(video_data_dir, timestamp_fn))
                    # keyframe_fn = f'trial={trial_id}_rgb-frame-seq.pkl'
                    # keyframe_seq = joblib.load(os.path.join(video_data_dir, keyframe_fn))
                    # import pdb; pdb.set_trace()
                except FileNotFoundError:
                    continue

                # find imu indices closes to keyframe timestamps
                imu_keyframe_idxs = utils.nearestIndices(imu_timestamp_seq, keyframe_timestamp_seq)

                keyframe_segments = pred_seg_label_seq[imu_keyframe_idxs]
                num_segments = np.unique(pred_seg_label_seq).max() + 1
                seg_keyframe_counts = utils.makeHistogram(num_segments, keyframe_segments)
                prec, rec, f1 = retrievalMetrics(seg_keyframe_counts)
                metric_dict['kf_prec'].append(prec)
                metric_dict['kf_rec'].append(rec)
                metric_dict['kf_f1'].append(f1)

                if plot_predictions:
                    fn = os.path.join(fig_dir, f'trial-{trial_id:03}_segs.png')
                    plot_array(
                        gt_seg_label_seq, pred_seg_label_seq,
                        imu_timestamp_seq, keyframe_timestamp_seq, fn=fn
                    )
                # fn = os.path.join(fig_dir, f'trial-{trial_id:03}.png')
                # labels = (gt_seg_label_seq, label_seq, pred_seg_label_seq, pred_label_seq)
                # label_names = ('gt segments', 'labels', 'pred segments', 'pred labels')
                # utils.plot_array(feature_seq, labels, label_names, fn=fn)

            metric_dict['ARI'] = metrics.adjusted_rand_score(gt_seg_label_seq, pred_seg_label_seq)
            # for name in metric_dict.keys():
            #     value = getattr(metrics, name)(gt_seg_label_seq, pred_seg_label_seq)
            #     metric_dict[name].append(value)

            # saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
            # saveVariable(score_seq, f'trial={trial_id}_score-seq')
            # saveVariable(label_seq, f'trial={trial_id}_true-label-seq')

        for name, value in metric_dict.items():
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
    parser.add_argument('--predictions_dir')
    parser.add_argument('--imu_data_dir')
    parser.add_argument('--video_data_dir')
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
