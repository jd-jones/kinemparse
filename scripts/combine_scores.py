import argparse
import os
import inspect
import collections

import yaml
import numpy as np
import joblib
from matplotlib import pyplot as plt
import graphviz as gv
# import scipy
import torch
import pandas as pd

from mathtools import utils, metrics, torchutils
from seqtools import torchutils as sequtils
from blocks.core import labels
from blocks.analysis import assemblystats


class FusionClassifier(torch.nn.Module):
    def __init__(self, num_sources=None):
        super().__init__()

        self.num_sources = num_sources

        self.linear = torch.nn.Linear(self.num_sources, 1)

        logger.info(
            f'Initialized fusion classifier. '
            f'Sources: {self.num_sources}'
        )

    def forward(self, input_seq):
        """
        Parameters
        ----------
        input_seq : torch.tensor of float, shape (batch, num_sources, num_classes, num_samples)

        Returns
        -------
        output_seq : torch.tensor of float, shape ()
        """
        output_seq = self.linear(input_seq.transpose(1, 3)).squeeze(dim=-1)
        return output_seq

    def predict(self, outputs):
        __, preds = torch.max(outputs, -1)
        return preds


class FusionCRF(FusionClassifier, sequtils.LinearChainScorer):
    pass


def drawPaths(paths, fig_fn, base_path, state_img_dir, path_scores=None, img_ext='png'):
    """ Draw a sequence of `BlockAssembly` states using graphviz.

    Parameters
    ----------
    path : iterable( int )
        A path is a list of state indices.
    fig_fn : str
        Filename of the figure
    base_path : str
        Path to the directory where figure will be saved
    state_img_dir : str
        Path to the directory containing source images of the states that make
        up this path. State filenames are assumed to have the format
        `state<state_index>.<img_ext>`
    img_ext : str, optional
        Extension specifying the image file type. Can be 'svg', 'png', etc.
    """

    path_graph = gv.Digraph(name=fig_fn, format=img_ext, directory=base_path)

    for j, path in enumerate(paths):
        for i, state_index in enumerate(path):
            image_fn = 'state{}.{}'.format(state_index, img_ext)
            image_path = os.path.join(state_img_dir, image_fn)

            if path_scores is not None:
                label = f"{path_scores[j, i]:.2f}"
            else:
                label = None

            path_graph.node(
                f"{j}, {i}", image=image_path,
                fixedsize='true', width='1', height='0.5', imagescale='true',
                pad='1', fontsize='12', label=label
            )
            if i > 0:
                path_graph.edge(f"{j}, {i - 1}", f"{j}, {i}", fontsize='12')

    path_graph.render(filename=fig_fn, directory=base_path, cleanup=True)


def plot_scores(score_seq, k=None, fn=None):
    subplot_width = 12
    subplot_height = 3
    num_axes = score_seq.shape[0] + 1

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize)
    if num_axes == 1:
        axes = [axes]

    score_seq = score_seq.copy()

    for i, scores in enumerate(score_seq):
        if k is not None:
            bottom_k = (-scores).argsort(axis=0)[k:, :]
            for j in range(scores.shape[1]):
                scores[bottom_k[:, j], j] = -np.inf
        axes[i].imshow(scores, interpolation='none', aspect='auto')
    axes[-1].imshow(score_seq.sum(axis=0), interpolation='none', aspect='auto')

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def plot_hists(scores, axis_labels=None, fn=None):
    subplot_width = 12
    subplot_height = 3
    num_axes = scores.shape[0]

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize)
    if num_axes == 1:
        axes = [axes]

    for i, s in enumerate(scores):
        s[np.isinf(s)] = s[~np.isinf(s)].min() - 1
        axes[i].hist(s, bins=50, density=True)
        if axis_labels is not None:
            axes[i].set_ylabel(axis_labels[i])

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def prune(scores, k=1):
    scores = scores.copy()
    for i in range(scores.shape[1]):
        k_best_score = -np.sort(-np.unique(scores[:, i]), axis=0)[k]
        scores[scores[:, i] < k_best_score, i] = -np.inf
    return scores


def main(
        out_dir=None, data_dir=None, cv_data_dir=None, score_dirs=[],
        metadata_file=None,
        fusion_method='sum', prune_imu=None, standardize=None, decode=None,
        plot_predictions=None, results_file=None, sweep_param_name=None,
        gpu_dev_id=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    score_dirs = tuple(map(os.path.expanduser, score_dirs))
    if cv_data_dir is not None:
        cv_data_dir = os.path.expanduser(cv_data_dir)

    if metadata_file is not None:
        metadata_file = os.path.expanduser(metadata_file)
        metadata = pd.read_csv(metadata_file, index_col=0)

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

    device = torchutils.selectDevice(gpu_dev_id)

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

    SMALL_NUMBER = -1e9
    for feat in feature_seqs:
        feat[np.isinf(feat)] = SMALL_NUMBER

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

    gt_scores = []
    all_scores = []
    num_keyframes_total = 0
    num_rgb_errors_total = 0
    num_correctable_errors_total = 0
    num_oov_total = 0
    num_changed_total = 0
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
            model = joblib.load(os.path.join(cv_data_dir, fn))

        train_features, train_assembly_seqs, train_ids = getSplit(train_idxs)
        train_labels = tuple(
            np.array(
                list(labels.gen_eq_classes(assembly_seq, train_assemblies, equivalent=None)),
            )
            for assembly_seq in train_assembly_seqs
        )

        train_set = torchutils.SequenceDataset(
            train_features, train_labels, seq_ids=train_ids,
            device=device
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=1, shuffle=True
        )

        model = FusionClassifier(num_sources=train_features[0].shape[0])

        train_epoch_log = collections.defaultdict(list)
        # val_epoch_log = collections.defaultdict(list)
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy()
        }

        criterion = torch.nn.CrossEntropyLoss()
        optimizer_ft = torch.optim.Adam(
            model.parameters(), lr=1e-3,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=0, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

        model, last_model_wts = torchutils.trainModel(
            model, criterion, optimizer_ft, lr_scheduler,
            train_loader,  # val_loader,
            device=device,
            metrics=metric_dict,
            train_epoch_log=train_epoch_log,
            # val_epoch_log=val_epoch_log,
            **train_params
        )

        test_assemblies = train_assemblies.copy()
        for feature_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
            gt_seq = np.array(list(
                labels.gen_eq_classes(gt_assembly_seq, test_assemblies, equivalent=None)
            ))

        if plot_predictions:
            assembly_fig_dir = os.path.join(fig_dir, 'assembly-imgs')
            if not os.path.exists(assembly_fig_dir):
                os.makedirs(assembly_fig_dir)
            for i, assembly in enumerate(test_assemblies):
                assembly.draw(assembly_fig_dir, i)

        # TEST PHASE
        accuracies = []
        for feature_seq, gt_assembly_seq, trial_id in zip(*getSplit(test_idxs)):
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

            # Ignore OOV states in ground-truth
            sample_idxs = np.arange(feature_seq.shape[-1])
            score_idxs = gt_seq[gt_seq < feature_seq.shape[1]]
            sample_idxs = sample_idxs[gt_seq < feature_seq.shape[1]]
            gt_scores.append(feature_seq[:, score_idxs, sample_idxs])
            all_scores.append(feature_seq.reshape(feature_seq.shape[0], -1))

            if fusion_method == 'sum':
                score_seq = feature_seq.sum(axis=0)
            elif fusion_method == 'rgb_only':
                score_seq = feature_seq[1]
            elif fusion_method == 'imu_only':
                score_seq = feature_seq[0]
            else:
                raise NotImplementedError()

            # if not decode:
            #     model = None

            if model is None:
                pred_seq = score_seq.argmax(axis=0)
            elif isinstance(model, torch.nn.Module):
                inputs = torch.tensor(feature_seq[None, ...], dtype=torch.float, device=device)
                outputs = model.forward(inputs)
                pred_seq = model.predict(outputs)[0].cpu().numpy()
            else:
                dummy_samples = np.arange(score_seq.shape[1])
                pred_seq, _, _, _ = model.viterbi(
                    dummy_samples, log_likelihoods=score_seq, ml_decode=(not decode)
                )

            pred_assemblies = [train_assemblies[i] for i in pred_seq]
            gt_assemblies = [test_assemblies[i] for i in gt_seq]

            task = int(metadata.iloc[trial_id]['task id'])
            goal = labels.constructGoalState(task)

            def equivalence(pred, true):
                return (pred <= goal) == (true <= goal)
            acc = metrics.accuracy_upto(pred_assemblies, gt_assemblies, equivalence=None)
            accuracies.append(acc)

            rgb_pred_seq = feature_seq[1].argmax(axis=0)
            num_changed = np.sum(rgb_pred_seq != pred_seq)
            rgb_is_wrong = rgb_pred_seq != gt_seq
            num_rgb_errors = np.sum(rgb_is_wrong)
            imu_scores = feature_seq[0]
            imu_scores_gt = np.array([
                imu_scores[s_idx, t] if s_idx < imu_scores.shape[0] else -np.inf
                for t, s_idx in enumerate(gt_seq)
            ])
            imu_scores_rgb = np.array([
                imu_scores[s_idx, t] if s_idx < imu_scores.shape[0] else -np.inf
                for t, s_idx in enumerate(rgb_pred_seq)
            ])
            # imu_scores_gt = imu_scores[gt_seq, range(len(rgb_pred_seq))]
            best_imu_scores = imu_scores.max(axis=0)
            imu_is_right = imu_scores_gt >= best_imu_scores
            rgb_pred_score_is_lower = imu_scores_gt > imu_scores_rgb
            is_correctable_error = rgb_is_wrong & imu_is_right & rgb_pred_score_is_lower
            num_correctable_errors = np.sum(is_correctable_error)
            prop_correctable = num_correctable_errors / num_rgb_errors

            num_oov = np.sum(gt_seq >= len(train_assemblies))
            num_states = len(gt_seq)

            num_keyframes_total += num_states
            num_rgb_errors_total += num_rgb_errors
            num_correctable_errors_total += num_correctable_errors
            num_oov_total += num_oov
            num_changed_total += num_changed

            logger.info(f"  trial {trial_id}: {num_states} keyframes")
            logger.info(f"    accuracy (fused): {acc * 100:.1f}%")
            logger.info(f"    {num_oov} OOV states ({num_oov / num_states * 100:.1f}%)")
            logger.info(
                f"    {num_rgb_errors} RGB errors; "
                f"{num_correctable_errors} correctable from IMU ({prop_correctable * 100:.1f}%)"
            )

            saveVariable(score_seq, f'trial={trial_id}_data-scores')

            torchutils.plotEpochLog(
                train_epoch_log,
                subfig_size=(10, 2.5),
                title='Training performance',
                fn=os.path.join(fig_dir, f'cvfold={cv_index}_train-plot.png')
            )

            if plot_predictions:
                io_figs_dir = os.path.join(fig_dir, 'system-io')
                if not os.path.exists(io_figs_dir):
                    os.makedirs(io_figs_dir)
                fn = os.path.join(io_figs_dir, f'trial={trial_id:03}.png')
                utils.plot_array(
                    feature_seq,
                    (gt_seq, pred_seq, score_seq),
                    ('gt', 'pred', 'scores'),
                    fn=fn
                )

                score_figs_dir = os.path.join(fig_dir, 'modality-scores')
                if not os.path.exists(score_figs_dir):
                    os.makedirs(score_figs_dir)
                plot_scores(
                    feature_seq, k=25,
                    fn=os.path.join(score_figs_dir, f"trial={trial_id:03}.png")
                )

                paths_dir = os.path.join(fig_dir, 'path-imgs')
                if not os.path.exists(paths_dir):
                    os.makedirs(paths_dir)
                assemblystats.drawPath(
                    pred_seq, trial_id,
                    f"trial={trial_id}_pred-seq", paths_dir, assembly_fig_dir
                )
                assemblystats.drawPath(
                    gt_seq, trial_id,
                    f"trial={trial_id}_gt-seq", paths_dir, assembly_fig_dir
                )

                label_seqs = (gt_seq,) + tuple(scores.argmax(axis=0) for scores in feature_seq)
                label_seqs = np.row_stack(label_seqs)
                k = 10
                for i, scores in enumerate(feature_seq):
                    label_score_seqs = tuple(
                        np.array([
                            scores[s_idx, t] if s_idx < scores.shape[0] else -np.inf
                            for t, s_idx in enumerate(label_seq)
                        ])
                        for label_seq in label_seqs
                    )
                    label_score_seqs = np.row_stack(label_score_seqs)
                    drawPaths(
                        label_seqs, f"trial={trial_id}_pred-scores_modality={i}",
                        paths_dir, assembly_fig_dir,
                        path_scores=label_score_seqs
                    )

                    topk_seq = (-scores).argsort(axis=0)[:k, :]
                    path_scores = np.column_stack(
                        tuple(scores[idxs, i] for i, idxs in enumerate(topk_seq.T))
                    )
                    drawPaths(
                        topk_seq, f"trial={trial_id}_topk_modality={i}",
                        paths_dir, assembly_fig_dir,
                        path_scores=path_scores
                    )
                label_score_seqs = tuple(
                    np.array([
                        score_seq[s_idx, t] if s_idx < score_seq.shape[0] else -np.inf
                        for t, s_idx in enumerate(label_seq)
                    ])
                    for label_seq in label_seqs
                )
                label_score_seqs = np.row_stack(label_score_seqs)
                drawPaths(
                    label_seqs, f"trial={trial_id}_pred-scores_fused",
                    paths_dir, assembly_fig_dir,
                    path_scores=label_score_seqs
                )
                topk_seq = (-score_seq).argsort(axis=0)[:k, :]
                path_scores = np.column_stack(
                    tuple(score_seq[idxs, i] for i, idxs in enumerate(topk_seq.T))
                )
                drawPaths(
                    topk_seq, f"trial={trial_id}_topk_fused",
                    paths_dir, assembly_fig_dir,
                    path_scores=path_scores
                )

        if accuracies:
            fold_accuracy = float(np.array(accuracies).mean())
            # logger.info(f'  acc: {fold_accuracy * 100:.1f}%')
            metric_dict = {'Accuracy': fold_accuracy}
            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

    num_unexplained_errors = num_rgb_errors_total - (num_oov_total + num_correctable_errors_total)
    prop_correctable = num_correctable_errors_total / num_rgb_errors_total
    prop_oov = num_oov_total / num_rgb_errors_total
    prop_unexplained = num_unexplained_errors / num_rgb_errors_total
    prop_changed = num_changed_total / num_keyframes_total
    logger.info("PERFORMANCE ANALYSIS")
    logger.info(
        f"  {num_rgb_errors_total} / {num_keyframes_total} "
        f"RGB errors ({num_rgb_errors_total / num_keyframes_total * 100:.1f}%)"
    )
    logger.info(
        f"  {num_oov_total} / {num_rgb_errors_total} "
        f"RGB errors are OOV ({prop_oov * 100:.1f}%)"
    )
    logger.info(
        f"  {num_correctable_errors_total} / {num_rgb_errors_total} "
        f"RGB errors are correctable ({prop_correctable * 100:.1f}%)"
    )
    logger.info(
        f"  {num_unexplained_errors} / {num_rgb_errors_total} "
        f"RGB errors are unexplained ({prop_unexplained * 100:.1f}%)"
    )
    logger.info(
        f"  {num_changed_total} / {num_keyframes_total} "
        f"Predictions changed after fusion ({prop_changed * 100:.1f}%)"
    )

    gt_scores = np.hstack(tuple(gt_scores))
    plot_hists(np.exp(gt_scores), fn=os.path.join(fig_dir, f"score-hists_gt.png"))
    all_scores = np.hstack(tuple(all_scores))
    plot_hists(np.exp(all_scores), fn=os.path.join(fig_dir, f"score-hists_all.png"))


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
