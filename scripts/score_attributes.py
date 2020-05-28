import inspect
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
        label_seq[assembly.start_idx:assembly.end_idx + 1] = assembly_id

    return label_seq


def make_features(feature_seq, raw_scores=False):
    if raw_scores:
        raise NotImplementedError()

    feature_seq = 2 * torch.nn.functional.softmax(feature_seq[..., 0:2], dim=-1) - 1
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
    score_seq = torch.nn.functional.softmax(signatures @ feature_seq, dim=0).log()
    return score_seq


def predict(score_seq):
    idxs = score_seq.argmax(dim=0)
    return idxs


def components_equivalent(x, y):
    return (labels.inSameComponent(x) == labels.inSameComponent(y)).all()


def main(
        out_dir=None, data_dir=None, attributes_dir=None,
        use_gt_segments=None, segments_dir=None, cv_data_dir=None,
        gpu_dev_id=None,
        plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    attributes_dir = os.path.expanduser(attributes_dir)
    if segments_dir is not None:
        segments_dir = os.path.expanduser(segments_dir)
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

    def loadAll(seq_ids, var_name, data_dir, prefix='trial='):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'{prefix}{seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=')
    assembly_seqs = loadAll(trial_ids, 'assembly-seq.pkl', data_dir, prefix='trial=')

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
            for s in (assembly_seqs, trial_ids)
        )
        return split_data

    for cv_index, (train_ids, test_ids) in enumerate(cv_fold_trial_ids):
        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(test_ids)} test)'
        )

        try:
            test_idxs = torch.tensor([trial_ids.tolist().index(i) for i in test_ids])
        except ValueError:
            logger.info(f"  Skipping fold: missing test data")
            continue

        # TRAIN PHASE
        if cv_data_dir is None:
            train_idxs = torch.tensor([trial_ids.index(i) for i in train_ids])
            train_assembly_seqs = tuple(assembly_seqs[i] for i in train_idxs)
            train_assemblies = []
            for seq in train_assembly_seqs:
                list(labels.gen_eq_classes(seq, train_assemblies, equivalent=None))
        else:
            fn = os.path.join(cv_data_dir, f'cvfold={cv_index}_train-assemblies.pkl')
            train_assemblies = joblib.load(fn)

        signatures = make_signatures(train_assemblies)

        if plot_predictions:
            figsize = (12, 3)
            fig, axis = plt.subplots(1, figsize=figsize)
            axis.imshow(signatures.numpy().T, interpolation='none', aspect='auto')
            plt.savefig(os.path.join(fig_dir, f"cvfold={cv_index}_signatures.png"))
            plt.close()

        # TEST PHASE
        accuracies = []
        for gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
            # FIXME: implement consistent data dimensions during serialization
            #   (ie samples along rows)
            # feature_seq shape is (pairs, samples, classes)
            # should be (samples, pairs, classes)
            try:
                fn = os.path.join(attributes_dir, f'trial={trial_id}_score-seq.pkl')
                feature_seq = joblib.load(fn)
            except FileNotFoundError:
                logger.info(f"  File not found: {fn}")
                continue

            test_assemblies = train_assemblies.copy()
            gt_assembly_id_seq = list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            )
            gt_seq = makeAssemblyLabels(gt_assembly_id_seq, gt_assembly_seq)

            if use_gt_segments:
                segments = utils.makeSegmentLabels(gt_seq)
            elif segments_dir is not None:
                var_name = 'segment-seq-imu.pkl'
                fn = os.path.join(segments_dir, f'trial={trial_id}_{var_name}')
                try:
                    segments = joblib.load(fn)
                except FileNotFoundError:
                    logger.info(f"  File not found: {fn}")
                    continue
            else:
                segments = None

            feature_seq = torch.tensor(feature_seq, dtype=torch.float)

            if segments is not None:
                feature_seq = feature_seq.transpose(0, 1)

                feature_seq, _ = utils.reduce_over_segments(
                    feature_seq.numpy(), segments,
                    reduction=lambda x: x.mean(axis=0)
                )
                feature_seq = torch.tensor(feature_seq.swapaxes(0, 1), dtype=torch.float)

                gt_seq, _ = utils.reduce_over_segments(gt_seq.numpy(), segments)
                gt_seq = torch.tensor(gt_seq, dtype=torch.long)

            feature_seq = make_features(feature_seq)
            score_seq = score(feature_seq, signatures)
            pred_seq = predict(score_seq)

            pred_assemblies = [train_assemblies[i] for i in pred_seq]
            gt_assemblies = [test_assemblies[i] for i in gt_seq]
            acc = metrics.accuracy_upto(
                pred_assemblies, gt_assemblies,
                equivalence=components_equivalent
            )
            accuracies.append(acc)

            # FIXME: Improve data naming convention in decode_keyframes.py
            saveVariable(score_seq, f'trial={trial_id}_data-scores')

            if plot_predictions:
                fn = os.path.join(fig_dir, f'trial-{trial_id:03}.png')
                utils.plot_array(
                    feature_seq,
                    (gt_seq.numpy(), pred_seq.numpy(), score_seq.numpy()),
                    ('gt', 'pred', 'scores'),
                    fn=fn
                )

        if not accuracies:
            continue
        fold_accuracy = float(torch.tensor(accuracies).mean())
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
