import argparse
import os

import yaml
import torch
import joblib
from matplotlib import pyplot as plt

from mathtools import utils, metrics
from blocks.core import labels
from kinemparse import imu


def makeAssemblyLabels(assembly_id_seq, assembly_seq):
    label_seq = torch.zeros(assembly_seq[-1].end_idx + 1, dtype=torch.long)
    for assembly_id, assembly in zip(assembly_id_seq, assembly_seq):
        label_seq[assembly.start_idx:assembly.end_idx + 1] = assembly_id

    return label_seq


def make_features(feature_seq):
    feature_seq = torch.tensor(feature_seq)
    feature_seq = 2 * torch.nn.functional.softmax(feature_seq, dim=-1) - 1
    feature_seq = feature_seq[..., 1]

    return feature_seq


def make_signatures(unique_assemblies):
    signatures = torch.stack([
        torch.tensor(labels.inSameComponent(a, lower_tri_only=True))
        for a in unique_assemblies
    ])
    signatures = 2 * signatures.float() - 1

    return signatures


def score(feature_seq, signatures):
    score_seq = signatures @ feature_seq
    return score_seq


def predict(score_seq):
    idxs = score_seq.argmax(dim=0)
    return idxs


def main(
        out_dir=None, data_dir=None, attributes_dir=None, gpu_dev_id=None,
        plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    attributes_dir = os.path.expanduser(attributes_dir)

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
    assembly_seqs = loadVariable('assembly_seqs')

    feature_seqs = tuple(
        make_features(
            joblib.load(
                os.path.join(attributes_dir, f'trial={trial_id}_score-seq.pkl')
            )
        )
        for trial_id in trial_ids
    )

    def equivalent(x, y):
        return (labels.inSameComponent(x) == labels.inSameComponent(y)).all()
    unique_assemblies = []
    assembly_id_seqs = tuple(
        labels.gen_eq_classes(seq, unique_assemblies, equivalent=equivalent)
        for seq in assembly_seqs
    )
    label_seqs = utils.batchProcess(makeAssemblyLabels, assembly_id_seqs, assembly_seqs)
    signatures = make_signatures(unique_assemblies)

    score_seqs = tuple(score(feature_seq, signatures) for feature_seq in feature_seqs)
    pred_seqs = tuple(predict(score_seq) for score_seq in score_seqs)

    acc = metrics.Accuracy()
    for pred_seq, gt_seq in zip(pred_seqs, label_seqs):
        acc.accumulate(pred_seq, gt_seq, None)
    logger.info(str(acc))

    # Define cross-validation folds
    # dataset_size = len(trial_ids)
    # cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    # def getSplit(split_idxs):
    #     split_data = tuple(
    #         tuple(s[i] for i in split_idxs)
    #         for s in (feature_seqs, label_seqs, trial_ids)
    #     )
    #     return split_data

    # for cv_index, cv_splits in enumerate(cv_folds):
    #     train_data, val_data, test_data = tuple(map(getSplit, cv_splits))
    figsize = (12, 3)
    fig, axis = plt.subplots(1, figsize=figsize)
    axis.imshow(signatures.numpy().T, interpolation='none', aspect='auto')
    plt.savefig(os.path.join(fig_dir, f"signatures.png"))
    plt.close()

    def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
        saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
        saveVariable(score_seq, f'trial={trial_id}_score-seq')
        saveVariable(label_seq, f'trial={trial_id}_true-label-seq')

    test_io_history = tuple(zip(pred_seqs, score_seqs, feature_seqs, label_seqs, trial_ids))
    if plot_predictions:
        imu.plot_prediction_eg(test_io_history, fig_dir, **viz_params)
    # for io in test_io_history:
    #     saveTrialData(*io)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--model_params')
    parser.add_argument('--results_file')
    parser.add_argument('--sweep_param_name')

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
