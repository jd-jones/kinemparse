import os
import logging

import yaml
import numpy as np

import LCTM.metrics

from mathtools import utils, metrics


logger = logging.getLogger(__name__)


def eval_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
    state_acc = (pred_seq == true_seq).astype(float).mean()

    metric_dict = {
        'State Accuracy' + name_suffix: state_acc,
        'State Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'State Overlap Score' + name_suffix: LCTM.metrics.overlap_score(pred_seq, true_seq) / 100
    }

    append_to.update(metric_dict)
    return append_to


def oov_rate(state_seq, state_vocab):
    state_is_oov = ~np.array([s in state_vocab for s in state_seq], dtype=bool)
    prop_state_oov = state_is_oov.sum() / state_is_oov.size
    return prop_state_oov


def confusionMatrix(all_pred_seqs, all_true_seqs, vocab_size):
    """
    Returns
    -------
    confusions: np.ndarray of int, shape (vocab_size, vocab_size)
        Rows represent predicted labels; columns represent true labels.
    """

    confusions = np.zeros((vocab_size, vocab_size), dtype=int)

    for pred_seq, true_seq in zip(all_pred_seqs, all_true_seqs):
        for i_pred, i_true in zip(pred_seq, true_seq):
            confusions[i_pred, i_true] += 1

    return confusions


def main(
        out_dir=None, data_dir=None, scores_dirs={},
        vocab_from_scores_dir=None, only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dirs = {name: os.path.expanduser(dir_) for name, dir_ in scores_dirs.items()}
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if len(scores_dirs) != 2:
        err_str = (
            f"scores_dirs has {len(scores_dirs)} entries, but this script "
            "compare exactly 2"
        )
        raise NotImplementedError(err_str)

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix='labels.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {data_dir}")

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    vocabs = {}
    confusions = {}
    accs = {}
    counts = {}
    for expt_name, scores_dir in scores_dirs.items():
        if vocab_from_scores_dir:
            vocabs[expt_name] = utils.loadVariable('vocab', scores_dir)
        else:
            vocabs[expt_name] = utils.loadVariable('vocab', data_dir)

        confusions[expt_name] = utils.loadVariable("confusions", scores_dir)
        per_class_accs, class_counts = metrics.perClassAcc(
            confusions[expt_name],
            return_counts=True
        )

        accs[expt_name] = per_class_accs
        counts[expt_name] = class_counts

    vocab = utils.reduce_all_equal(tuple(vocabs.values()))
    class_counts = utils.reduce_all_equal(tuple(counts.values()))

    first_name, second_name = scores_dirs.keys()
    confusions_diff = confusions[first_name] - confusions[second_name]
    acc_diff = accs[first_name] - accs[second_name]

    metrics.plotConfusions(
        os.path.join(fig_dir, f'confusions_{first_name}-minus-{second_name}.png'),
        confusions_diff, vocab
    )
    metrics.plotPerClassAcc(
        os.path.join(fig_dir, f'accs_{first_name}-minus-{second_name}.png'),
        vocab, acc_diff, confusions_diff.sum(axis=1), class_counts
    )


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
