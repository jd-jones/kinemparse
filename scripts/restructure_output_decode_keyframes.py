import argparse
import os
import inspect

import yaml
import joblib
import numpy as np

from mathtools import utils


def main(out_dir=None, data_dir=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def loadAll(var_name, trial_ids):
        def loadOne(trial_id):
            return loadVariable(f"trial-{trial_id}_{var_name}")
        return tuple(map(loadOne, trial_ids))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial-')

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(
        dataset_size,
        precomputed_fn=os.path.join(data_dir, 'cv-folds.pkl')
    )
    cv_folds = tuple(tuple(map(np.array, splits)) for splits in cv_folds)

    # Validate CV folds by checking that each split covers all trial ids
    for train_idxs, test_idxs in cv_folds:
        num_train = len(train_idxs)
        num_test = len(test_idxs)
        num_total = len(trial_ids)
        if num_train + num_test != num_total:
            err_str = f"{num_train} train + {num_test} test != {num_total} total"
            raise AssertionError(err_str)

    cv_fold_trial_ids = tuple(
        tuple(map(lambda x: trial_ids[x], splits))
        for splits in cv_folds
    )
    saveVariable(cv_fold_trial_ids, f"cv-fold-trial-ids")

    for cv_index, (train_idxs, test_idxs) in enumerate(cv_folds):
        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_idxs)} train, {len(test_idxs)} test)'
        )

        hmm = loadVariable(f"hmm-fold{cv_index}")
        train_assemblies = hmm.states

        saveVariable(train_assemblies, f"cvfold={cv_index}_train-assemblies")

        for trial_id in trial_ids[test_idxs]:
            # true_state_seqs = loadVariable('trial-{trial_id}_true-state-seq-orig', trial_ids)
            saved_predictions = loadVariable(f'trial-{trial_id}_pred-state-seq')
            data_scores = loadVariable(f'trial-{trial_id}_data-scores')

            # Validate the loaded data
            pred_assembly_idxs = data_scores.argmax(axis=0)
            pred_assembly_seq = tuple(train_assemblies[i] for i in pred_assembly_idxs)

            preds_same = all(p1 == p1 for p1, p2 in zip(pred_assembly_seq, saved_predictions))
            if not preds_same:
                raise AssertionError('Computed predictions differ from saved predictions')

            saveVariable(data_scores, f"trial={trial_id}_data-scores")


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
