import argparse
import os
import logging

import yaml
import joblib
# import numpy as np
# from matplotlib import pyplot as plt

from mathtools import utils
# from seqtools import fsm
from kinemparse import imu


logger = logging.getLogger(__name__)


def main(
        out_dir=None, data_dir=None, scores_dir=None,
        model_name=None, model_params={},
        results_file=None, sweep_param_name=None,
        cv_params={}, viz_params={},
        plot_predictions=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

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

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = loadVariable('trial_ids')
    feature_seqs = loadVariable('imu_sample_seqs')
    label_seqs = loadVariable('imu_label_seqs')

    if scores_dir is not None:
        scores_dir = os.path.expanduser(scores_dir)
        feature_seqs = tuple(
            joblib.load(
                os.path.join(scores_dir, f'trial={trial_id}_score-seq.pkl')
            ).swapaxes(0, 1)
            for trial_id in trial_ids
        )

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    metric_dict = {
        'accuracy': [],
        'edit_score': [],
        'overlap_score': []
    }

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, label_seqs, trial_ids)
        )
        return split_data

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        train_ids = train_data[-1]
        test_ids = test_data[-1]
        val_ids = val_data[-1]

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        for name in metric_dict.keys():
            value = None  # FIXME
            metric_dict[name] += [value]
        metric_str = '  '.join(f"{k}: {v[-1]:.1f}%" for k, v in metric_dict.items())
        logger.info('[TST]  ' + metric_str)

        d = {k: v[-1] for k, v in metric_dict.items()}
        utils.writeResults(results_file, d, sweep_param_name, model_params)

        test_io_history = None  # FIXME

        if plot_predictions:
            imu.plot_prediction_eg(test_io_history, fig_dir, **viz_params)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq, f'trial={trial_id}_score-seq')
            saveVariable(label_seq, f'trial={trial_id}_true-label-seq')
        for io in test_io_history:
            saveTrialData(*io)

        # saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        # saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        # saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        # saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        # saveVariable(model, f'cvfold={cv_index}_{model_name}-best')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--scores_dir')
    parser.add_argument('--model_params')
    parser.add_argument('--results_file')
    parser.add_argument('--sweep_param_name')

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
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
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
