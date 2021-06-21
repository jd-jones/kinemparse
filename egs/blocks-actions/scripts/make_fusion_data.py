import os
import logging

import yaml
import numpy as np
import scipy
# import pandas as pd
# from matplotlib import pyplot as plt

from mathtools import utils
from blocks.core import labels as labels_lib
from blocks.core import blockassembly


logger = logging.getLogger(__name__)


def make_attribute_features(score_seq):
    return 2 * score_seq - 1


class FusionDataset(object):
    def __init__(
            self, actions_dir, parts_dir, events_dir, edges_dir,
            vocab=None,
            modalities=('actions', 'parts', 'edges'), labels='edges',
            prefix='seq=', feature_fn_format='score-seq', label_fn_format='true-label-seq'):
        self.modalities = modalities
        self.labels = labels

        self.actions_dir = actions_dir
        self.parts_dir = parts_dir
        self.events_dir = events_dir
        self.edges_dir = edges_dir

        self.data_dirs = {
            'actions': self.actions_dir,
            'parts': self.parts_dir,
            'events': self.events_dir,
            'edges': self.edges_dir,
            'edge diffs': self.events_dir,
            'edge diffs binary': self.events_dir
        }

        labels_dir = self.data_dirs[self.labels]
        self.trial_ids = utils.getUniqueIds(
            labels_dir,
            prefix='trial=' if self.labels == 'edges' else prefix,
            suffix=f'{label_fn_format}.*',
            to_array=True
        )
        self.prefix = prefix
        self.feature_fn_format = feature_fn_format
        self.label_fn_format = label_fn_format

        if self.labels in ('edge diffs', 'edge diffs binary'):
            vocab = (
                [blockassembly.AssemblyAction()]
                + utils.loadVariable('assembly-action-vocab', self.events_dir)
            )
            for a in vocab:
                if isinstance(a.sign, np.ndarray):
                    a.sign = np.sign(a.sign.sum())
                    logger.info('Replaced sign')
            self.parts_vocab, self.edge_diffs = labels_lib.make_parts_vocab(
                vocab, lower_tri_only=True, append_to_vocab=False
            )
            signs = np.array([a.sign for a in vocab], dtype=int)
            signs[signs == -1] = 2
            self.edge_diffs = np.concatenate((self.edge_diffs, signs[:, None]), axis=1)

            if self.labels == 'edge diffs binary':
                self.edge_diffs = (self.edge_diffs > 0).astype(int)

        if vocab is None:
            vocab = utils.loadVariable('vocab', labels_dir)
        self.vocab = vocab
        self.metadata = utils.loadMetadata(labels_dir, rows=self.trial_ids)

    def loadInputs(self, seq_id, prefix=None, stride=None):
        if prefix is None:
            prefix = self.prefix

        actions_seq = utils.loadVariable(
            f"{prefix}{seq_id}_{self.feature_fn_format}",
            self.actions_dir
        )
        actions_seq = scipy.special.softmax(actions_seq, axis=1)

        events_seq = utils.loadVariable(
            f"{prefix}{seq_id}_{self.feature_fn_format}",
            self.events_dir
        )
        # events_seq = scipy.special.softmax(events_seq, axis=1)

        parts_seq = utils.loadVariable(
            f"{prefix}{seq_id}_{self.feature_fn_format}",
            self.parts_dir
        )
        parts_seq = scipy.special.expit(parts_seq)

        prefix = 'trial='
        edges_seq = utils.loadVariable(
            f"{prefix}{seq_id}_{self.feature_fn_format}",
            self.edges_dir
        )
        edges_seq = scipy.special.softmax(edges_seq, axis=1)
        edges_seq = np.reshape(edges_seq, (edges_seq.shape[0], -1))
        edges_seq = edges_seq[::5]

        attribute_feats = {
            'actions': make_attribute_features(actions_seq),
            'parts': make_attribute_features(parts_seq),
            'events': events_seq,
            'edges': make_attribute_features(edges_seq)
        }

        attribute_feats = np.concatenate(
            tuple(attribute_feats[name] for name in self.modalities),
            axis=1
        )

        return attribute_feats

    def loadTargets(self, seq_id, prefix=None):
        if prefix is None:
            prefix = self.prefix

        if self.labels == 'edges':
            prefix = 'trial='
            true_label_seq = utils.loadVariable(
                f'{prefix}{seq_id}_{self.label_fn_format}',
                self.data_dirs[self.labels]
            )
            true_label_seq = true_label_seq[::5]
        else:
            true_label_seq = utils.loadVariable(
                f'{prefix}{seq_id}_{self.label_fn_format}',
                self.data_dirs[self.labels]
            )

        if self.labels in ('edge diffs', 'edge diffs binary'):
            true_label_seq = self.edge_diffs[true_label_seq]

        return true_label_seq


def main(
        out_dir=None, data_dir=None,
        actions_dir=None, parts_dir=None, events_dir=None, edges_dir=None,
        prefix='seq=', feature_fn_format='score-seq', label_fn_format='true-label-seq',
        stop_after=None, only_fold=None, plot_io=None,
        dataset_params={}, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    actions_dir = os.path.expanduser(actions_dir)
    parts_dir = os.path.expanduser(parts_dir)
    events_dir = os.path.expanduser(events_dir)
    edges_dir = os.path.expanduser(edges_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    # vocab = utils.loadVariable(
    #     'assembly-action-vocab',
    #     os.path.join(data_dir, 'event-dataset')
    # )
    # vocab = [BlockAssembly()] + list(abs(x) for x in vocab)
    dataset = FusionDataset(
        actions_dir, parts_dir, events_dir, edges_dir,
        prefix=prefix,
        **dataset_params,
        # vocab=vocab,
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)
    utils.saveVariable(dataset.vocab, 'vocab', out_data_dir)

    seq_ids = dataset.trial_ids
    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {data_dir}")

    # Define cross-validation folds
    # cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    # utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    for i, seq_id in enumerate(seq_ids):
        try:
            labels = dataset.loadTargets(seq_id)
            features = dataset.loadInputs(seq_id)
        except AssertionError as e:
            logger.warning(f'Skipping sequence {seq_id}: {e}')
            continue

        logger.info(f"Processing sequence {seq_id}")

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
