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
from kinemparse import assembly as lib_assembly


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


def actionsFromAssemblies(assembly_index_seq, assembly_vocab, action_vocab):
    assembly_index_segs, seg_lens = utils.computeSegments(assembly_index_seq)
    assembly_segs = tuple(assembly_vocab[i] for i in assembly_index_segs)
    action_segs = (
        (lib_assembly.AssemblyAction(),)
        + tuple(n - c for c, n in zip(assembly_segs[:-1], assembly_segs[1:]))
    )
    action_index_segs = tuple(utils.getIndex(a, action_vocab) for a in action_segs)
    action_index_seq = np.array(utils.fromSegments(action_index_segs, seg_lens))

    return action_segs, action_index_seq


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

    assembly_vocab = joblib.load(os.path.join(data_dir, 'assembly-vocab.pkl'))

    trial_ids = utils.getUniqueIds(preds_dir, prefix='trial=')
    pred_seqs = loadAll(trial_ids, 'pred-label-seq.pkl', preds_dir)
    true_seqs = loadAll(trial_ids, 'true-label-seq.pkl', preds_dir)
    input_seqs = loadAll(trial_ids, 'score-seq.pkl', preds_dir)

    action_vocab = []
    batch = []
    for i, trial_id in enumerate(trial_ids):
        logger.info(f"VIDEO {trial_id}:")

        pred_action_segs, pred_action_index_seq = actionsFromAssemblies(
            pred_seqs[i], assembly_vocab, action_vocab
        )

        lib_assembly.writeAssemblies(
            os.path.join(out_data_dir, f'trial={trial_id}_pred-actions.txt'),
            pred_action_segs
        )

        true_action_segs, true_action_index_seq = actionsFromAssemblies(
            true_seqs[i], assembly_vocab, action_vocab
        )

        lib_assembly.writeAssemblies(
            os.path.join(out_data_dir, f'trial={trial_id}_true-actions.txt'),
            true_action_segs
        )

        metric_dict = {}
        for name in metric_names:
            key = f"{name}_action"
            value = getattr(LCTM.metrics, name)(pred_action_index_seq, true_action_index_seq) / 100
            metric_dict[key] = value
            logger.info(f"  {key}: {value * 100:.1f}%")

        utils.writeResults(results_file, metric_dict, sweep_param_name, {})

        input_seq = input_seqs[i]
        batch.append((pred_action_index_seq, input_seq, true_action_index_seq, trial_id))

    lib_assembly.writeAssemblies(os.path.join(out_data_dir, 'action-vocab.txt'), action_vocab)

    label_names = ('gt', 'pred')
    for preds, inputs, gt_labels, seq_id in batch:
        fn = os.path.join(fig_dir, f"trial={seq_id}_model-io.png")
        utils.plot_array(inputs, (gt_labels, preds), label_names, fn=fn)


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
