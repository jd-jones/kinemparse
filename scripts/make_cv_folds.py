import os
import logging
import json
import glob

import yaml
import numpy as np
import pandas as pd

from mathtools import utils


logger = logging.getLogger(__name__)


def save_cv_folds(cv_folds, fn):
    def to_list(indices):
        if isinstance(indices, np.ndarray):
            return indices.tolist()
        return list(indices)
    cv_folds = [
        [to_list(indices) for indices in split]
        for split in cv_folds
    ]
    with open(fn, 'wt') as file_:
        json.dump(cv_folds, file_)


def main(
        out_dir=None, data_dir=None, prefix='trial=',
        feature_fn_format='feature-seq.pkl', label_fn_format='label_seq.pkl',
        slowfast_labels_path=None, cv_params={}, slowfast_csv_params={}):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    # Load data
    trial_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix=feature_fn_format,
        to_array=True
    )
    dataset = utils.CvDataset(
        trial_ids, data_dir,
        feature_fn_format=feature_fn_format, label_fn_format=label_fn_format,
        vocab=[], prefix=prefix
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)

    # Make folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, metadata=dataset.metadata, **cv_params)
    save_cv_folds(cv_folds, os.path.join(out_data_dir, 'cv-folds.json'))

    split_names = ('train', 'val', 'test')

    # Check folds
    for cv_index, cv_fold in enumerate(cv_folds):
        train_data, val_data, test_data = dataset.getFold(cv_fold)
        train_feats, train_labels, train_ids = train_data
        test_feats, test_labels, test_ids = test_data
        val_feats, val_labels, val_ids = val_data

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        slowfast_labels_pattern = os.path.join(data_dir, 'slowfast-labels*.csv')
        for slowfast_labels_path in glob.glob(slowfast_labels_pattern):
            cv_str = f"cvfold={cv_index}"
            fn = os.path.basename(slowfast_labels_path)
            slowfast_labels = pd.read_csv(
                slowfast_labels_path, index_col=0, keep_default_na=False,
                **slowfast_csv_params
            )
            for split_indices, split_name in zip(cv_fold, split_names):
                matches = tuple(
                    slowfast_labels.loc[slowfast_labels['video_name'] == vid_id]
                    for vid_id in dataset.metadata.iloc[split_indices]['dir_name'].to_list()
                )
                if matches:
                    split = pd.concat(matches, axis=0)
                    split.to_csv(
                        os.path.join(out_data_dir, f"{cv_str}_{split_name}_{fn}"),
                        **slowfast_csv_params
                    )
                else:
                    logger.info(f'  Skipping empty slowfast split: {split_name}')


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
