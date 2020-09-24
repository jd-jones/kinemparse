import os
import logging
import csv
import warnings

import yaml
import joblib
from scipy import io
import numpy as np
from matplotlib import pyplot as plt

# Stop numba from throwing a bunch of warnings when it compiles LCTM
from numba import NumbaWarning; warnings.filterwarnings('ignore', category=NumbaWarning)
import LCTM.metrics

from mathtools import utils
from kinemparse import airplanecorpus


logger = logging.getLogger(__name__)


def writeLabels(fn, label_seq, header=None):
    with open(fn, 'wt') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if header is not None:
            writer.writerow(header)
        for label in label_seq:
            writer.writerow(label)


def toStringSeq(string_array):
    return [arr[0] for arr in string_array[0]]


def plotDetections(fn, detection_seq, pred_seq, label_seq):
    num_samples, num_detections = detection_seq.shape
    f, axes = plt.subplots(num_detections + 1, sharex=True, sharey=True)

    for i in range(num_detections):
        detection_label = (label_seq == i).astype(int)
        axes[i].set_ylabel(f'bin {i}')
        axes[i].plot(detection_seq[:, i])
        axes[i].twinx().plot(detection_label, color='tab:orange')

    axes[-1].plot(pred_seq, label='pred')
    axes[-1].plot(label_seq, label='true')
    axes[-1].legend()

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def main(
        out_dir=None, preds_dir=None, data_dir=None, metric_names=None,
        detection_threshold=None,
        plot_output=None, results_file=None, sweep_param_name=None):

    if metric_names is None:
        metric_names = ('accuracy', 'edit_score', 'overlap_score')

    preds_dir = os.path.expanduser(preds_dir)
    data_dir = os.path.expanduser(data_dir)

    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    logger.info(f"Writing to: {out_dir}")

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
        if os.path.exists(results_file):
            os.remove(results_file)
    else:
        results_file = os.path.expanduser(results_file)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir, convert=None):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            key = os.path.splitext(var_name)[0].replace('-', '_')
            if var_name.endswith('.mat'):
                data = io.loadmat(fn)[key]
            elif var_name.endswith('.pkl'):
                data = joblib.load(fn)
            if convert is not None:
                data = convert(data)
            return data
        return tuple(map(loadOne, seq_ids))

    part_names, part_names_to_idxs, part_idxs_to_bins = airplanecorpus.loadParts()
    transition_vocabulary = joblib.load(os.path.join(data_dir, 'transition-vocabulary.pkl'))

    trial_ids = utils.getUniqueIds(preds_dir, prefix='trial=', suffix='.mat')
    pred_seqs = loadAll(trial_ids, 'pred-state-seq.mat', preds_dir, convert=toStringSeq)
    # true_seqs = loadAll(trial_ids, 'true-state-seq.mat', preds_dir, convert=toStringSeq)
    true_seqs = loadAll(trial_ids, 'label-seq.pkl', data_dir)
    detection_scores = loadAll(trial_ids, 'detection-scores.mat', preds_dir)

    for i, trial_id in enumerate(trial_ids):
        logger.info(f"VIDEO {trial_id}:")

        pred_action_seq = pred_seqs[i]
        true_seq = true_seqs[i]
        detection_score_seq = detection_scores[i]
        seq_len = min(len(pred_action_seq), true_seq.shape[0], detection_score_seq.shape[0])
        pred_action_seq = pred_action_seq[:seq_len]
        true_seq = true_seq[:seq_len]
        detection_score_seq = detection_score_seq[:seq_len, :]

        true_transition_seq = tuple(transition_vocabulary[i] for i in true_seq)
        # true_assembly_seq = tuple(n for c, n in true_transition_seq)
        true_action_seq = tuple(
            airplanecorpus.actionFromTransition(c, n)
            for c, n in true_transition_seq
        )
        true_action_index_seq = np.array([part_names_to_idxs[i] for i in true_action_seq])
        true_bin_index_seq = np.array([part_idxs_to_bins[i] for i in true_action_index_seq])

        pred_action_index_seq = np.array([part_names_to_idxs[i] for i in pred_action_seq])
        pred_bin_index_seq = detection_score_seq.argmax(axis=1)

        if detection_threshold is not None:
            above_thresh = detection_score_seq.max(axis=1) > detection_threshold
            true_bin_index_seq = true_bin_index_seq[above_thresh]
            pred_bin_index_seq = pred_bin_index_seq[above_thresh]
            detection_score_seq = detection_score_seq[above_thresh, :]

        fn = os.path.join(fig_dir, f"trial={trial_id}_baseline-detections.png")
        plotDetections(fn, detection_score_seq, pred_bin_index_seq, true_bin_index_seq)

        writeLabels(
            os.path.join(fig_dir, f"trial={trial_id}_action-seqs"),
            zip(true_action_seq, pred_action_seq),
            header=('true', 'pred')
        )

        writeLabels(
            os.path.join(fig_dir, f"trial={trial_id}_bin-seqs"),
            zip(true_bin_index_seq, pred_bin_index_seq),
            header=('true', 'pred')
        )

        metric_dict = {}
        for name in metric_names:
            key = f"{name}_action"
            value = getattr(LCTM.metrics, name)(pred_action_index_seq, true_action_index_seq) / 100
            metric_dict[key] = value
            logger.info(f"  {key}: {value * 100:.1f}%")

            key = f"{name}_bin"
            value = getattr(LCTM.metrics, name)(pred_bin_index_seq, true_bin_index_seq) / 100
            metric_dict[key] = value
            logger.info(f"  {key}: {value * 100:.1f}%")

        utils.writeResults(results_file, metric_dict, sweep_param_name, {})


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
