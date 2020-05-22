import inspect
import argparse
import os

import yaml
import joblib
import numpy as np
# from matplotlib import pyplot as plt

from mathtools import utils
from blocks.core import labels


def make_signatures(unique_assemblies):
    signatures = np.stack([
        labels.inSameComponent(a, lower_tri_only=True)
        for a in unique_assemblies
    ])
    signatures = 2 * signatures.astype(float) - 1

    return signatures


def main(out_dir=None, data_dir=None):
    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadVariable(var_name):
        fn = os.path.join(data_dir, f'{var_name}.pkl')
        return joblib.load(fn)

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial-')
    cv_folds = loadVariable(f'cv-folds')
    assembly_seqs = tuple(
        loadVariable(f"trial-{seq_id}_true-state-seq-orig")
        for seq_id in trial_ids
    )

    unique_assemblies = []
    label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(seq, unique_assemblies, equivalent=None)))
        for seq in assembly_seqs
    )

    saveVariable(unique_assemblies, f'unique-assemblies')

    for cv_index, (train_idxs, test_idxs) in enumerate(cv_folds):
        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_idxs)} train, {len(test_idxs)} test)'
        )

        # train_ids = trial_ids[np.array(train_idxs)]
        train_label_seqs = tuple(label_seqs[i] for i in train_idxs)

        unique_train_labels = np.unique(np.hstack(train_label_seqs))
        unique_train_assemblies = tuple(unique_assemblies[i] for i in unique_train_labels)
        signatures = make_signatures(unique_train_assemblies)

        test_ids = trial_ids[np.array(test_idxs)]
        for trial_id in test_ids:
            saveVariable(signatures, f'trial={trial_id}_signatures')
            saveVariable(unique_train_labels, f'trial={trial_id}_unique-train-labels')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    for arg_name in inspect.getfullargspec(main).args:
        parser.add_argument(f'--{arg_name}')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    else:
        config_fn = os.path.basename(config_file_path)
    if not os.path.exists(config_file_path):
        config = {}
    else:
        with open(config_file_path, 'rt') as config_file:
            config = yaml.safe_load(config_file)
    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
