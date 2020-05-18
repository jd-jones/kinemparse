import argparse
import os

import yaml
import torch
import joblib
from matplotlib import pyplot as plt

from mathtools import utils, metrics
from blocks.core import labels


def makeAssemblyLabels(assembly_id_seq, assembly_seq):
    label_seq = torch.zeros(assembly_seq[-1].end_idx + 1, dtype=torch.long)
    for assembly_id, assembly in zip(assembly_id_seq, assembly_seq):
        # label_seq[assembly.action_start_idx:assembly.action_end_idx + 1] = 0
        label_seq[assembly.start_idx:assembly.end_idx + 1] = assembly_id
        # label_seq[assembly.start_idx:assembly.end_idx + 1] = 1

    return label_seq


def make_features(feature_seq, raw_scores=False):
    feature_seq = feature_seq.swapaxes(0, 1)
    feature_seq = torch.tensor(feature_seq)

    if not raw_scores:
        feature_seq = 2 * torch.nn.functional.softmax(feature_seq[..., 0:2], dim=-1) - 1
        feature_seq = feature_seq[..., 1]

    feature_seq = feature_seq.reshape(feature_seq.shape[0], -1)
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


def components_equivalent(x, y):
    return (labels.inSameComponent(x) == labels.inSameComponent(y)).all()


def main(
        out_dir=None, data_dir=None, attributes_dir=None, segments_dir=None,
        gpu_dev_id=None,
        plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    attributes_dir = os.path.expanduser(attributes_dir)
    if segments_dir is not None:
        segments_dir = os.path.expanduser(segments_dir)

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
    trial_ids = utils.getUniqueIds(attributes_dir, prefix='trial=')
    assembly_seqs = loadAll(trial_ids, 'assembly-seq.pkl', data_dir)
    attr_score_seqs = loadAll(trial_ids, 'score-seq.pkl', attributes_dir)

    # Extract features, etc
    feature_seqs = tuple(
        # FIXME: implement consistent array shapes across scripts
        #   (samples along rows)
        make_features(scores.swapaxes(0, 1), raw_scores=False)
        for scores in attr_score_seqs
    )

    unique_assemblies = []
    assembly_id_seqs = tuple(
        labels.gen_eq_classes(seq, unique_assemblies, equivalent=components_equivalent)
        for seq in assembly_seqs
    )
    label_seqs = utils.batchProcess(makeAssemblyLabels, assembly_id_seqs, assembly_seqs)

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, label_seqs, trial_ids)
        )
        return split_data

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, test_data = tuple(map(getSplit, cv_splits))
        train_ids = train_data[-1]
        test_ids = test_data[-1]
        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(test_ids)} test)'
        )
        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')

        # TRAIN PHASE
        _, train_labels, _ = train_data
        unique_train_labels = torch.cat(train_labels).unique()
        unique_train_assemblies = tuple(unique_assemblies[i] for i in unique_train_labels)
        signatures = make_signatures(unique_train_assemblies)

        if plot_predictions:
            figsize = (12, 3)
            fig, axis = plt.subplots(1, figsize=figsize)
            axis.imshow(signatures.numpy().T, interpolation='none', aspect='auto')
            plt.savefig(os.path.join(fig_dir, f"signatures.png"))
            plt.close()

        # TEST PHASE
        acc = metrics.Accuracy()
        for feature_seq, gt_seq, trial_id in zip(*test_data):
            score_seq = score(feature_seq, signatures)
            pred_seq = unique_train_labels[predict(score_seq)]

            acc.accumulate(pred_seq, gt_seq, None)

            if plot_predictions:
                fn = os.path.join(fig_dir, f'trial-{trial_id:03}.png')
                utils.plot_array(
                    feature_seq, (pred_seq.numpy(), gt_seq.numpy()), ('pred', 'gt'),
                    fn=fn
                )

            score_seq = score_seq.numpy().T
            if segments_dir is not None:
                var_name = 'segment-seq-imu.pkl'
                fn = os.path.join(segments_dir, f'trial={trial_id}_{var_name}')
                try:
                    segments = joblib.load(fn)
                except FileNotFoundError:
                    logger.info(f"File not found: {fn}")
                    continue
                score_seq, _ = utils.reduce_over_segments(
                    score_seq, segments,
                    reduction=lambda x: x.mean(axis=0)
                )
            saveVariable(score_seq, f'trial={trial_id}_attr-score-seq-rgb')

        logger.info('  ' + str(acc))

        metric_dict = {'Accuracy': acc.value}
        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--attributes_dir')
    parser.add_argument('--segments_dir')
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
