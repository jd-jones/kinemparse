import os
import logging
import collections

import yaml
import numpy as np

from mathtools import utils


logger = logging.getLogger(__name__)


def main(
        out_dir=None, data_dir=None, scores_dir=None, start_from=None, stop_at=None,
        results_file=None, sweep_param_name=None, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
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

    def loadVariable(var_name, from_dir=scores_dir):
        var = utils.loadVariable(var_name, from_dir)
        return var

    def saveVariable(var, var_name, to_dir=out_data_dir):
        utils.saveVariable(var, var_name, out_data_dir)

    def makeSeqBatches(unflatten, seq_ids):
        d = collections.defaultdict(list)
        for batch_index, (seq_index, win_index) in enumerate(unflatten):
            seq_id = seq_ids[seq_index]
            d[seq_id].append(batch_index)
        return d

    def loadBatchData(cv_index, batch_index):
        prefix = f"cvfold={cv_index}_batch={batch_index}"
        batch_score = loadVariable(f"{prefix}_score-seq")
        batch_pred = loadVariable(f"{prefix}_pred-label-seq").astype(int)
        batch_true = loadVariable(f"{prefix}_true-label-seq").astype(int)
        return batch_score, batch_pred, batch_true

    vocab = loadVariable('vocab')
    parts_vocab = loadVariable('parts-vocab')
    edge_labels = loadVariable('part-labels')
    saveVariable(vocab, 'vocab')
    saveVariable(parts_vocab, 'parts-vocab')
    saveVariable(edge_labels, 'part-labels')

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', suffix='', to_array=True)
    cv_folds = utils.makeDataSplits(len(trial_ids), **cv_params)

    for cv_index, (__, __, test_indices) in enumerate(cv_folds):
        logger.info(f"CV FOLD {cv_index + 1} / {len(cv_folds)}")
        test_ids = trial_ids[test_indices]
        unflatten = loadVariable(f"cvfold={cv_index}_test-set-unflatten")
        flatten = makeSeqBatches(unflatten, test_ids)
        for seq_id in test_ids:
            logger.info(f"  Processing sequence {seq_id}...")
            batch_idxs = flatten[seq_id]
            score_seq, pred_seq, true_seq = map(
                np.vstack,
                zip(*tuple(loadBatchData(cv_index, i) for i in batch_idxs))
            )

            trial_prefix = f"trial={seq_id}"
            rgb_seq = loadVariable(f"{trial_prefix}_rgb-frame-seq.", from_dir=data_dir)
            if score_seq.shape[0] != rgb_seq.shape[0]:
                err_str = f"scores shape {score_seq.shape} != data shape {rgb_seq.shape}"
                raise AssertionError(err_str)

            saveVariable(score_seq, f"{trial_prefix}_score-seq")
            saveVariable(pred_seq, f"{trial_prefix}_pred-label-seq")
            saveVariable(true_seq, f"{trial_prefix}_true-label-seq")


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
