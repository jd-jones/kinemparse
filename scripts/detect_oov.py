import argparse
import os
import inspect
import collections

import yaml
import numpy as np
import joblib
from matplotlib import pyplot as plt

from mathtools import utils, metrics
from blocks.core import labels


def OOVrate(assembly_seqs, eq_class=None):
    all_assemblies = []
    label_seqs = tuple(
        np.array(list(labels.gen_eq_classes(seq, all_assemblies, equivalent=None)))
        for seq in assembly_seqs
    )

    num_oov = collections.defaultdict(int)
    num_counts = collections.defaultdict(int)
    for i, heldout_seq in enumerate(label_seqs):
        train_seqs = label_seqs[:i] + label_seqs[i + 1:]
        train_seqs = np.hstack(train_seqs)
        train_labels = np.unique(train_seqs)
        for label, prev_label in zip(heldout_seq[1:], heldout_seq[:-1]):
            if eq_class == 'state index':
                prev_eq_class = prev_label
            elif eq_class == 'is oov':
                prev_eq_class = int(prev_label not in train_labels)
            else:
                raise NotImplementedError()

            if label not in train_labels:
                num_oov[prev_eq_class] += 1
            num_counts[prev_eq_class] += 1

    num_labels = max(num_counts.keys()) + 1
    oov_counts = np.zeros(num_labels)
    for label, count in num_oov.items():
        oov_counts[label] = count
    total_counts = np.zeros(num_labels)
    for label, count in num_counts.items():
        total_counts[label] = count

    contextual_oov_rate = oov_counts / total_counts
    oov_rate = oov_counts.sum() / total_counts.sum()
    return oov_rate, contextual_oov_rate, total_counts


def plotOOV(oov_rates, state_counts, fn=None, eq_class=None, subplot_width=12, subplot_height=3):

    num_subplots = 2
    figsize = (subplot_width, num_subplots * subplot_height)
    fig, axes = plt.subplots(num_subplots, figsize=figsize, sharex=True)

    axes[0].stem(oov_rates, use_line_collection=True)
    axes[0].set_ylabel("OOV rate")
    axes[0].set_xlabel(eq_class)

    axes[1].stem(state_counts, use_line_collection=True)
    axes[1].set_ylabel("count")
    axes[1].set_xlabel(eq_class)

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def scatterOOV(
        oov_rates, state_counts, fn=None, eq_class=None, subplot_width=12, subplot_height=12):
    num_subplots = 1
    figsize = (subplot_width, num_subplots * subplot_height)
    fig, axis = plt.subplots(num_subplots, figsize=figsize, sharex=True)
    axes = [axis]

    axes[0].plot(state_counts, oov_rates, 'o')
    axes[0].set_ylabel("OOV rate")
    axes[0].set_xlabel("state count")

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def main(
        out_dir=None, data_dir=None, cv_data_dir=None, scores_dir=None, eq_class='state index',
        plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    scores_dir = os.path.expanduser(scores_dir)
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
    trial_ids = utils.getUniqueIds(scores_dir, prefix='trial=')
    assembly_seqs = loadAll(trial_ids, 'assembly-seq.pkl', data_dir)
    score_seqs = loadAll(trial_ids, 'data-scores.pkl', scores_dir)

    oov_rate, contextual_oov_rate, state_counts = OOVrate(assembly_seqs, eq_class=eq_class)
    logger.info(f"OOV RATE: {oov_rate * 100:.1f}%")
    plotOOV(
        contextual_oov_rate, state_counts,
        fn=os.path.join(fig_dir, "oovs.png"), eq_class=eq_class
    )
    scatterOOV(
        contextual_oov_rate, state_counts,
        fn=os.path.join(fig_dir, "oovs_scatter.png"), eq_class=eq_class
    )

    import sys; sys.exit()

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
            for s in (score_seqs, assembly_seqs, trial_ids)
        )
        return split_data

    for cv_index, (train_ids, test_ids) in enumerate(cv_fold_trial_ids):

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(test_ids)} test)'
        )

        try:
            test_idxs = np.array([trial_ids.tolist().index(i) for i in test_ids])
        except ValueError:
            logger.info(f"  Skipping fold: missing test data")
            continue

        # TRAIN PHASE
        if cv_data_dir is None:
            train_idxs = np.array([trial_ids.index(i) for i in train_ids])
            train_assembly_seqs = tuple(assembly_seqs[i] for i in train_idxs)
            train_assemblies = []
            for seq in train_assembly_seqs:
                list(labels.gen_eq_classes(seq, train_assemblies, equivalent=None))
            model = None
        else:
            fn = f'cvfold={cv_index}_train-assemblies.pkl'
            train_assemblies = joblib.load(os.path.join(cv_data_dir, fn))
            train_idxs = [i for i in range(len(trial_ids)) if i not in test_idxs]

            fn = f'cvfold={cv_index}_model.pkl'
            # model = joblib.load(os.path.join(cv_data_dir, fn))
            model = None

        train_features, train_assembly_seqs, train_ids = getSplit(train_idxs)
        test_assemblies = train_assemblies.copy()
        for score_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
            gt_seq = np.array(list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            ))

        # oov_rate = OOVrate(train_assembly_seqs)
        # logger.info(f"  OOV RATE: {oov_rate * 100:.1f}%")

        if plot_predictions:
            assembly_fig_dir = os.path.join(fig_dir, 'assembly-imgs')
            if not os.path.exists(assembly_fig_dir):
                os.makedirs(assembly_fig_dir)
            for i, assembly in enumerate(test_assemblies):
                assembly.draw(assembly_fig_dir, i)

        # TEST PHASE
        accuracies = []
        for score_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
            gt_seq = np.array(list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            ))

            num_labels = gt_seq.shape[0]
            num_scores = score_seq.shape[-1]
            if num_labels != num_scores:
                err_str = f"Skipping trial {trial_id}: {num_labels} labels != {num_scores} scores"
                logger.info(err_str)
                continue

            if model is None:
                pred_seq = score_seq.argmax(axis=0)
            else:
                raise AssertionError()
            pred_seq = score_seq.argmax(axis=0)

            pred_assemblies = [train_assemblies[i] for i in pred_seq]
            gt_assemblies = [test_assemblies[i] for i in gt_seq]

            acc = metrics.accuracy_upto(pred_assemblies, gt_assemblies, equivalence=None)
            accuracies.append(acc)

            # num_states = len(gt_seq)
            # logger.info(f"  trial {trial_id}: {num_states} keyframes")
            # logger.info(f"    accuracy (fused): {acc * 100:.1f}%")

            saveVariable(score_seq, f'trial={trial_id}_data-scores')
            saveVariable(pred_assemblies, f'trial={trial_id}_pred-assembly-seq')
            saveVariable(gt_assemblies, f'trial={trial_id}_gt-assembly-seq')

        if accuracies:
            fold_accuracy = float(np.array(accuracies).mean())
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
