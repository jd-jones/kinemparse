import os
import logging

import yaml
import numpy as np
import torch
import joblib

from mathtools import utils, metrics
from blocks.core import labels
from seqtools import utils as su


logger = logging.getLogger(__name__)


def trial_ids_from_dirs(*paths):
    dir_trial_ids = tuple(
        set(utils.getUniqueIds(path, prefix='trial=', to_array=True))
        for path in paths
    )
    trial_ids = np.array(list(sorted(set.intersection(*dir_trial_ids))))

    for dir_name, t_ids in zip(paths, dir_trial_ids):
        logger.info(f"{len(t_ids)} trial ids from {dir_name}:")
        logger.info(f"  {t_ids}")
    logger.info(f"{len(trial_ids)} trials in intersection: {trial_ids}")

    return trial_ids


def idxs_of_matching_shapes(feature_seqs, trial_ids):
    include_indices = []
    for i, seq_feats in enumerate(feature_seqs):
        feat_shapes = tuple(f.shape for f in seq_feats)
        include_seq = all(f == feat_shapes[0] for f in feat_shapes)
        if include_seq:
            include_indices.append(i)
        else:
            warn_str = (
                f'Excluding trial {trial_ids[i]} with mismatched feature shapes: '
                f'{feat_shapes}'
            )
            logger.warning(warn_str)
    return include_indices


def main(
        out_dir=None, data_dir=None, cv_data_dir=None, score_dirs=[],
        fusion_method='sum', decode=None,
        plot_predictions=None, results_file=None, sweep_param_name=None,
        gpu_dev_id=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    if cv_data_dir is None:
        raise AssertionError()

    def score_features(feature_seq):
        if fusion_method == 'sum':
            return feature_seq.sum(axis=0)
        elif fusion_method == 'rgb_only':
            return feature_seq[1]
        elif fusion_method == 'imu_only':
            return feature_seq[0]
        else:
            raise NotImplementedError()

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    score_dirs = tuple(map(os.path.expanduser, score_dirs))
    if cv_data_dir is not None:
        cv_data_dir = os.path.expanduser(cv_data_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
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

    dirs = score_dirs + (data_dir,)
    trial_ids = trial_ids_from_dirs(*dirs)

    assembly_seqs = loadAll(trial_ids, 'assembly-seq.pkl', data_dir)
    feature_seqs = tuple(loadAll(trial_ids, 'data-scores.pkl', d) for d in score_dirs)
    feature_seqs = tuple(zip(*feature_seqs))

    # Filter out sequences where feature shape don't match
    include_indices = idxs_of_matching_shapes(feature_seqs, trial_ids)
    trial_ids = trial_ids[include_indices]
    assembly_seqs = tuple(assembly_seqs[i] for i in include_indices)
    feature_seqs = tuple(feature_seqs[i] for i in include_indices)

    feature_seqs = tuple(np.stack(f) for f in feature_seqs)

    # Define cross-validation folds
    fn = os.path.join(cv_data_dir, 'cv-fold-trial-ids.pkl')
    cv_fold_trial_ids = joblib.load(fn)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, assembly_seqs, trial_ids)
        )
        return split_data

    new_cv_fold_trial_ids = []
    for cv_index, (train_ids, test_ids) in enumerate(cv_fold_trial_ids):
        new_test_ids = np.array([i for i in test_ids if np.any(trial_ids == i)])
        new_train_ids = np.array([i for i in train_ids if np.any(trial_ids == i)])
        new_cv_fold_trial_ids.append((new_train_ids, new_test_ids))
    cv_fold_trial_ids = new_cv_fold_trial_ids

    for cv_index, (train_ids, test_ids) in enumerate(cv_fold_trial_ids):
        if not test_ids.any():
            logger.info(f"Skipping CV fold {cv_index + 1}: no test seqs")
            continue

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(test_ids)} test)'
        )

        # TRAIN PHASE
        model = joblib.load(os.path.join(cv_data_dir, f'cvfold={cv_index}_model.pkl'))
        train_assemblies = joblib.load(
            os.path.join(cv_data_dir, f'cvfold={cv_index}_train-assemblies.pkl')
        )
        test_idxs = np.array([trial_ids.tolist().index(i) for i in test_ids])
        train_idxs = np.array([trial_ids.tolist().index(i) for i in train_ids])
        # train_idxs = np.array([i for i in range(len(trial_ids)) if i not in test_idxs])

        train_features, train_assembly_seqs, train_ids = getSplit(train_idxs)

        test_assemblies = train_assemblies.copy()
        for feature_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
            gt_seq = np.array(list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            ))

        vocab = train_assemblies.copy()
        train_gt_seqs = tuple(
            np.array(list(labels.gen_eq_classes(seq, vocab, equivalent=None)))
            for seq in train_assembly_seqs
        )
        train_vocab_idxs = np.unique(np.hstack(train_gt_seqs))
        if np.any(train_vocab_idxs > len(train_assemblies)):
            raise AssertionError()
        train_assembly_is_oov = np.array([
            not np.any(train_vocab_idxs == i)
            for i in range(len(train_assemblies))
        ])
        # train_vocab = [
        #     a for i, a in enumerate(train_assemblies)
        #     if not train_assembly_is_oov[i]
        # ]

        # model.states = train_vocab
        # tx_probs, _, _, = su.smoothCounts(*su.countSeqs(train_gt_seqs))
        # model.psi[~train_assembly_is_oov, ~train_assembly_is_oov] = tx_probs
        # model.log_psi[~train_assembly_is_oov, ~train_assembly_is_oov] = np.log(tx_probs)
        model.psi[train_assembly_is_oov, train_assembly_is_oov] = 0
        model.log_psi[train_assembly_is_oov, train_assembly_is_oov] = -np.inf

        # TEST PHASE
        accuracies = []
        for feature_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
            logger.info(f"  testing on trial {trial_id}")
            gt_seq = np.array(list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            ))

            num_labels = gt_seq.shape[0]
            num_features = feature_seq.shape[-1]
            if num_labels != num_features:
                err_str = (
                    f"Skipping trial {trial_id}: "
                    f"{num_labels} labels != {num_features} features"
                )
                logger.info(err_str)
                continue

            score_seq = score_features(feature_seq)
            score_seq[train_assembly_is_oov, :] = -np.inf

            if not decode:
                # pred_seq = score_seq.argmax(axis=0)
                pred_seq = torch.tensor(score_seq).argmax(dim=0).numpy()
            else:
                dummy_samples = np.arange(score_seq.shape[1])
                pred_seq, _, _, _ = model.viterbi(
                    dummy_samples, log_likelihoods=score_seq, ml_decode=False
                )

            pred_assemblies = [train_assemblies[i] for i in pred_seq]
            gt_assemblies = [test_assemblies[i] for i in gt_seq]

            acc = metrics.accuracy_upto(pred_assemblies, gt_assemblies, equivalence=None)
            accuracies.append(acc)

            saveVariable(score_seq, f'trial={trial_id}_data-scores')
            saveVariable(pred_assemblies, f'trial={trial_id}_pred-assembly-seq')
            saveVariable(gt_assemblies, f'trial={trial_id}_gt-assembly-seq')

        if accuracies:
            fold_accuracy = float(np.array(accuracies).mean())
            logger.info(f'  acc: {fold_accuracy * 100:.1f}%')
            metric_dict = {'Accuracy': fold_accuracy}
            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)


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
