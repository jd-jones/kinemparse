import os
import logging
import collections

import yaml
import joblib
import numpy as np

from mathtools import utils


logger = logging.getLogger(__name__)


def main(
        out_dir=None, data_dir=None, scores_dir=None, start_from=None, stop_at=None,
        results_file=None, sweep_param_name=None):

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
        var = joblib.load(os.path.join(from_dir, f"{var_name}.pkl"))
        return var

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

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

    cv_fold_indices = utils.getUniqueIds(scores_dir, prefix='cvfold=', to_array=True)
    num_cv_folds = len(cv_fold_indices)

    for cv_index in cv_fold_indices:
        logger.info(f"CV FOLD {cv_index + 1} / {num_cv_folds}")
        seq_ids = loadVariable(f"cvfold={cv_index}_test-ids")
        unflatten = loadVariable(f"cvfold={cv_index}_test-set-unflatten")
        flatten = makeSeqBatches(unflatten, seq_ids)
        for seq_id in seq_ids:
            logger.info(f"  Processing sequence {seq_id}...")
            batch_idxs = flatten[seq_id]
            score_seq, pred_seq, true_seq = map(
                np.vstack,
                zip(*tuple(loadBatchData(cv_index, i) for i in batch_idxs))
            )

            trial_prefix = f"trial={seq_id}"
            rgb_seq = loadVariable(f"{trial_prefix}_rgb-frame-seq", from_dir=data_dir)
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
