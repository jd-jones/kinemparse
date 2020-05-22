import argparse
import os
import inspect

import yaml
import numpy as np
import joblib
# from matplotlib import pyplot as plt

from mathtools import utils, metrics
from blocks.core import labels


def main(
        out_dir=None, data_dir=None, cv_data_dir=None, score_dirs=[],
        plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    score_dirs = tuple(map(os.path.expanduser, score_dirs))
    if cv_data_dir is not None:
        cv_data_dir = os.path.expanduser(cv_data_dir)

    if results_file is None:
        results_file = os.path.join(out_dir, f'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    dir_trial_ids = tuple(set(utils.getUniqueIds(d, prefix='trial=')) for d in score_dirs)
    dir_trial_ids += (set(utils.getUniqueIds(data_dir, prefix='trial=')),)
    trial_ids = np.array(list(sorted(set.intersection(*dir_trial_ids))))

    assembly_seqs = loadAll(trial_ids, 'assembly-seq.pkl', data_dir)
    feature_seqs = tuple(loadAll(trial_ids, 'data-scores.pkl', d) for d in score_dirs)

    # Combine feature seqs
    ret = tuple(
        (i, np.stack(feats))
        for i, feats in enumerate(zip(*feature_seqs))
        if all(f.shape == feats[0].shape for f in feats)
    )
    idxs, feature_seqs = tuple(zip(*ret))
    trial_ids = trial_ids[list(idxs)]
    assembly_seqs = tuple(assembly_seqs[i] for i in idxs)

    # Define cross-validation folds
    if cv_data_dir is None:
        dataset_size = len(trial_ids)
        cv_folds = utils.makeDataSplits(dataset_size, **cv_params)
        cv_fold_trial_ids = tuple(
            tuple(map(lambda x: trial_ids[x], splits))
            for splits in cv_folds
        )
    else:
        fn = os.path.join(cv_data_dir, f'cv-fold-trial-ids.pkl')
        cv_fold_trial_ids = joblib.load(fn)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, assembly_seqs, trial_ids)
        )
        return split_data

    for cv_index, (train_ids, test_ids) in enumerate(cv_fold_trial_ids):
        # train_data, test_data = tuple(map(getSplit, cv_splits))
        # train_ids = train_data[-1]
        # test_ids = test_data[-1]
        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(test_ids)} test)'
        )

        try:
            test_idxs = np.array([trial_ids.tolist().index(i) for i in test_ids])
        except ValueError:
            logger.info(f"  Skipping fold: missing test data")
            continue

        # saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        # saveVariable(test_ids, f'cvfold={cv_index}_test-ids')

        # TRAIN PHASE
        if cv_data_dir is None:
            train_idxs = np.array([trial_ids.index(i) for i in train_ids])
            train_assembly_seqs = tuple(assembly_seqs[i] for i in train_idxs)
            train_assemblies = []
            for seq in train_assembly_seqs:
                list(labels.gen_eq_classes(seq, train_assemblies, equivalent=None))
        else:
            fn = os.path.join(cv_data_dir, f'cvfold={cv_index}_train-assemblies.pkl')
            train_assemblies = joblib.load(fn)

        # TEST PHASE
        accuracies = []
        for feature_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):

            test_assemblies = train_assemblies.copy()
            gt_seq = np.array(list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            ))

            score_seq = feature_seq.sum(axis=0)
            pred_seq = score_seq.argmax(axis=0)

            pred_assemblies = [train_assemblies[i] for i in pred_seq]
            gt_assemblies = [test_assemblies[i] for i in gt_seq]
            acc = metrics.accuracy_upto(pred_assemblies, gt_assemblies)
            accuracies.append(acc)

            logger.info(f"  trial {trial_id}: {acc * 100:.1f}%")

            saveVariable(score_seq, f'trial={trial_id}_attr-score-seq')

            if plot_predictions:
                fn = os.path.join(fig_dir, f'trial-{trial_id:03}.png')
                utils.plot_array(
                    feature_seq,
                    (gt_seq, pred_seq, score_seq),
                    ('gt', 'pred', 'scores'),
                    fn=fn
                )

        fold_accuracy = float(np.array(accuracies).mean())
        logger.info(f'  acc: {fold_accuracy * 100:.1f}%')
        metric_dict = {'Accuracy': fold_accuracy}
        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)


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
