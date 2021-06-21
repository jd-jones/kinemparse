import os
import logging

import yaml
import numpy as np
import scipy

from mathtools import utils


logger = logging.getLogger(__name__)


def drawVocab(fig_dir, vocab):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for i, x in enumerate(vocab):
        x.draw(fig_dir, i)


def main(
        out_dir=None, data_dirs=None,
        prefix='seq=', feature_fn_format='score-seq', label_fn_format='true-label-seq',
        stride=None,
        only_fold=None, stop_after=None, take_log=None,
        modalities=('assembly', 'event'),
        plot_io=None, draw_vocab=False,
        results_file=None, sweep_param_name=None):

    data_dirs = {
        name: os.path.expanduser(dir_)
        for name, dir_ in data_dirs.items()
    }
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    dir_seq_ids = {
        name: utils.getUniqueIds(
            data_dirs[name], prefix=prefix[name], suffix=f'{label_fn_format[name]}.*',
            to_array=True
        )
        for name in modalities
    }

    ids_sets = [set(ids) for ids in dir_seq_ids.values()]
    seq_ids = np.array(sorted(ids_sets[0].intersection(*ids_sets[1:])))
    substr = '; '.join([f'{len(ids)} seqs in {name}' for name, ids in dir_seq_ids.items()])
    logger.info(f"Found {substr}; {len(seq_ids)} shared")

    for name in modalities:
        data_dir = data_dirs[name]
        out_data_dir = os.path.join(out_dir, f'{name}-data')
        if not os.path.exists(out_data_dir):
            os.makedirs(out_data_dir)

        fig_dir = os.path.join(out_dir, 'figures', name)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        misc_dir = os.path.join(out_dir, 'misc', name)
        if not os.path.exists(misc_dir):
            os.makedirs(misc_dir)

        vocab = utils.loadVariable('vocab', data_dir)
        metadata = utils.loadMetadata(data_dir, rows=seq_ids)

        if name == 'event':
            for i in range(len(vocab)):
                if isinstance(vocab[i].sign, np.ndarray):
                    vocab[i].sign = np.sign(vocab[i].sign.sum())

        all_labels = tuple(
            utils.loadVariable(f"{prefix[name]}{seq_id}_{label_fn_format[name]}", data_dir)
            for seq_id in seq_ids
        )

        # Remove labels that don't occur in the dataset
        if name == 'assembly':
            unique_labels = np.sort(np.unique(np.hstack(all_labels)))
            OOV_INDEX = unique_labels.shape[0]
            old_idxs_to_new = np.full(len(vocab), OOV_INDEX, dtype=int)
            for new, old in enumerate(unique_labels):
                old_idxs_to_new[old] = new
            vocab = [vocab[i] for i in unique_labels]
            num_removed = np.sum(old_idxs_to_new == OOV_INDEX)
            logger.info(f'Removing {num_removed} labels that do not occur in dataset')

        utils.saveVariable(vocab, 'vocab', out_data_dir)
        utils.saveMetadata(metadata, out_data_dir)

        if draw_vocab:
            drawVocab(os.path.join(fig_dir, 'vocab'), vocab)

        for i, seq_id in enumerate(seq_ids):
            if stop_after is not None and i >= stop_after:
                break

            trial_prefix = f"{prefix[name]}{seq_id}"
            logger.info(f"Processing sequence {seq_id}...")

            true_label_seq = all_labels[i]
            score_seq = utils.loadVariable(
                f"{trial_prefix}_{feature_fn_format[name]}",
                data_dir
            )

            true_label_seq = true_label_seq[::stride[name]]
            if name == 'assembly':
                true_label_seq = old_idxs_to_new[true_label_seq]

            if take_log[name]:
                score_seq = np.log(score_seq)

            score_seq = score_seq[::stride[name]]
            if name == 'assembly':
                score_seq = score_seq[:, unique_labels]

            score_seq = scipy.special.log_softmax(score_seq, axis=1)

            pred_label_seq = score_seq.argmax(axis=1)

            trial_prefix = f"seq={seq_id}"
            utils.saveVariable(score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_label_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            utils.saveVariable(true_label_seq, f'{trial_prefix}_true-label-seq', out_data_dir)

            if plot_io:
                utils.plot_array(
                    score_seq.T, (true_label_seq.T,), ('gt',),
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}.png")
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
