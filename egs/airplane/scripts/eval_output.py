import os
import logging
import csv
import warnings

import yaml
import joblib
import numpy as np

# Stop numba from throwing a bunch of warnings when it compiles LCTM
from numba import NumbaWarning; warnings.filterwarnings('ignore', category=NumbaWarning)
import LCTM.metrics

from mathtools import utils
from kinemparse import airplanecorpus


logger = logging.getLogger(__name__)


def equivalent(pred_assembly, true_assembly):
    residual = pred_assembly ^ true_assembly
    residual = residual - frozenset(['wheel1', 'wheel2', 'wheel3', 'wheel4'])
    return len(residual) == 0


def writeLabels(fn, label_seq, header=None):
    with open(fn, 'wt') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if header is not None:
            writer.writerow(header)
        for label in label_seq:
            writer.writerow(label)


def main(
        out_dir=None, preds_dir=None, data_dir=None, metric_names=None,
        plot_output=None, results_file=None, sweep_param_name=None):

    if metric_names is None:
        metric_names = ('accuracy', 'edit_score', 'overlap_score')

    preds_dir = os.path.expanduser(preds_dir)

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

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    part_names, part_names_to_idxs, part_idxs_to_bins = airplanecorpus.loadParts()
    transition_vocabulary = joblib.load(os.path.join(data_dir, 'transition-vocabulary.pkl'))

    trial_ids = utils.getUniqueIds(preds_dir, prefix='trial=')
    pred_seqs = loadAll(trial_ids, 'pred-label-seq.pkl', preds_dir)
    true_seqs = loadAll(trial_ids, 'true-label-seq.pkl', preds_dir)

    for i, trial_id in enumerate(trial_ids):
        logger.info(f"VIDEO {trial_id}:")

        pred_transition_index_seq = pred_seqs[i]
        pred_transition_seq = tuple(transition_vocabulary[i] for i in pred_transition_index_seq)
        pred_action_seq = tuple(
            airplanecorpus.actionFromTransition(c, n)
            for c, n in pred_transition_seq
        )
        pred_action_index_seq = np.array([part_names_to_idxs[i] for i in pred_action_seq])
        pred_bin_index_seq = np.array([part_idxs_to_bins[i] for i in pred_action_index_seq])
        pred_assembly_seq = tuple(n for c, n in pred_transition_seq)
        saveVariable(pred_assembly_seq, f'trial={trial_id}_pred-assembly-seq')
        saveVariable(pred_action_seq, f'trial={trial_id}_pred-action-seq')

        true_transition_index_seq = true_seqs[i]
        true_transition_seq = tuple(transition_vocabulary[i] for i in true_transition_index_seq)
        true_assembly_seq = tuple(n for c, n in true_transition_seq)
        true_action_seq = tuple(
            airplanecorpus.actionFromTransition(c, n)
            for c, n in true_transition_seq
        )
        true_action_index_seq = np.array([part_names_to_idxs[i] for i in true_action_seq])
        true_bin_index_seq = np.array([part_idxs_to_bins[i] for i in true_action_index_seq])
        saveVariable(true_assembly_seq, f'trial={trial_id}_true-assembly-seq')
        saveVariable(true_action_seq, f'trial={trial_id}_true-action-seq')

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

        final_pred = pred_assembly_seq[-1]
        final_true = true_assembly_seq[-1]
        final_equivalent = equivalent(final_pred, final_true)
        metric_dict['accuracy_model'] = float(final_equivalent)
        logger.info(f"  FINAL (PRED): {final_pred}")
        logger.info(f"  FINAL (TRUE): {final_true}")
        logger.info(f"  RESIDUAL: {final_pred ^ final_true}")
        logger.info(f"  EQUIVALENT:   {final_equivalent}")

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
