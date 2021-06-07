import os
import logging

import yaml
import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt

from mathtools import utils
from blocks.core import labels as labels_lib


logger = logging.getLogger(__name__)


def make_attribute_features(score_seq):
    return 2 * score_seq - 1


class FusionDataset(object):
    def __init__(
            self, trial_ids, labels_dir, actions_dir, parts_dir,
            vocab=None,
            modalities=('actions', 'parts'),
            prefix='seq=', feature_fn_format='score-seq', label_fn_format='true-label-seq'):
        self.trial_ids = trial_ids
        self.modalities = modalities

        if vocab is None:
            vocab = utils.loadVariable('vocab', labels_dir)
        self.vocab = vocab
        self.metadata = utils.loadMetadata(labels_dir, rows=trial_ids)

        self.labels_dir = labels_dir
        self.actions_dir = actions_dir
        self.parts_dir = parts_dir

        self.prefix = prefix
        self.feature_fn_format = feature_fn_format
        self.label_fn_format = label_fn_format

        self.vocab = list(abs(x) for x in self.vocab)
        self.parts_vocab, self.part_labels = labels_lib.make_parts_vocab(
            self.vocab, lower_tri_only=True, append_to_vocab=False
        )

    def loadInputs(self, seq_id):
        trial_prefix = f"{self.prefix}{seq_id}"
        actions_seq = utils.loadVariable(
            f"{trial_prefix}_{self.feature_fn_format}",
            self.actions_dir
        )
        parts_seq = utils.loadVariable(
            f"{trial_prefix}_{self.feature_fn_format}",
            self.parts_dir
        )

        attribute_feats = {
            'actions': make_attribute_features(actions_seq),
            'parts': make_attribute_features(parts_seq)
        }

        attribute_feats = np.concatenate(
            tuple(attribute_feats[name] for name in self.modalities),
            axis=1
        )

        return attribute_feats

    def loadTargets(self, seq_id, win_size=None, stride=None):
        trial_prefix = f"{self.prefix}{seq_id}"
        true_label_seq = utils.loadVariable(
            f'{trial_prefix}_{self.label_fn_format}',
            self.labels_dir
        )

        true_label_seq = self.part_labels[true_label_seq]
        true_label_seq = true_label_seq[::stride]

        return true_label_seq


def main(
        out_dir=None, data_dir=None,
        actions_dir=None, parts_dir=None, labels_dir=None,
        prefix='seq=', feature_fn_format='score-seq', label_fn_format='true-label-seq',
        stop_after=None, only_fold=None, plot_io=None,
        win_params={}, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    actions_dir = os.path.expanduser(actions_dir)
    parts_dir = os.path.expanduser(parts_dir)
    labels_dir = os.path.expanduser(labels_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    seq_ids = utils.getUniqueIds(
        labels_dir, prefix=prefix, suffix=f'{label_fn_format}.*',
        to_array=True
    )

    vocab = utils.loadVariable(
        'assembly-action-vocab',
        os.path.join(data_dir, 'event-dataset')
    )
    dataset = FusionDataset(
        seq_ids, labels_dir, actions_dir, parts_dir,
        prefix=prefix, vocab=vocab,
        feature_fn_format=feature_fn_format, label_fn_format=label_fn_format
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)
    utils.saveVariable(dataset.vocab, 'vocab', out_data_dir)

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {data_dir}")

    # Define cross-validation folds
    # cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    # utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    for i, seq_id in enumerate(seq_ids):
        logger.info(f"Processing sequence {seq_id}...")

        labels = dataset.loadTargets(seq_id, **win_params)
        features = dataset.loadInputs(seq_id)

        if labels.shape[0] != features.shape[0]:
            message = f'Label shape {labels.shape} != feature shape {features.shape}'
            raise AssertionError(message)

        seq_prefix = f"seq={seq_id}"
        utils.saveVariable(features, f'{seq_prefix}_feature-seq', out_data_dir)
        utils.saveVariable(labels, f'{seq_prefix}_label-seq', out_data_dir)

        if plot_io:
            fn = os.path.join(fig_dir, f'{seq_prefix}.png')
            utils.plot_array(
                features.T,
                (labels.T,),
                ('gt',),
                fn=fn
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
