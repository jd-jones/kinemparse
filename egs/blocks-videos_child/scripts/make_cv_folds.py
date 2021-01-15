import os
import logging
import json

import yaml

from mathtools import utils


logger = logging.getLogger(__name__)


def save_cv_folds(cv_folds, fn):
    cv_folds = [
        [[indices.tolist() for indices in split] for split in splits]
        for splits in cv_folds
    ]
    with open(fn, 'wt') as file_:
        json.dump(cv_folds, file_)


def main(
        out_dir=None, data_dir=None,
        feature_fn_format='feature-seq.pkl', label_fn_format='label_seq.pkl',
        cv_params={}):

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
        data_dir, prefix='trial=', suffix=feature_fn_format,
        to_array=True
    )
    dataset = utils.CvDataset(
        trial_ids, data_dir,
        feature_fn_format=feature_fn_format, label_fn_format=label_fn_format
    )

    utils.saveMetadata(dataset.metadata, out_data_dir)

    # Make folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)
    save_cv_folds(cv_folds, os.path.join(out_data_dir, 'cv-folds.json'))

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
