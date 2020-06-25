import argparse
import os
import warnings
import logging

import yaml
import joblib
import numpy as np
from matplotlib import pyplot as plt

# Stop numba from throwing a bunch of warnings when it compiles LCTM
from numba import NumbaWarning; warnings.filterwarnings('ignore', category=NumbaWarning)

import LCTM.models
import LCTM.metrics
import LCTM.utils
import LCTM.learn

from mathtools import utils


logger = logging.getLogger(__name__)


def preprocess(imu_feature_seqs):
    def preprocessSeq(feature_seq):
        return feature_seq.swapaxes(0, 1)

    preprocessed_feature_seqs = tuple(preprocessSeq(x) for x in imu_feature_seqs)

    return preprocessed_feature_seqs


def pre_init(model, train_samples, train_labels, pretrain=True, transitions=None):
    n_samples = len(train_samples)
    model.n_features = train_samples[0].shape[0]
    model.n_classes = np.max(list(map(np.max, train_labels))) + 1
    model.max_segs = LCTM.utils.max_seg_count(train_labels)

    model.ws.init_weights(model)

    if pretrain:
        if model.is_latent:
            Z = [
                utils.partition_latent_labels(train_labels[i], model.n_latent)
                for i in range(n_samples)
            ]
            LCTM.learn.pretrain_weights(model, train_samples, Z)
        else:
            LCTM.learn.pretrain_weights(model, train_samples, train_labels)

    # Overwrite pairwise weights with negative log transition probs
    # train_labels = tuple(map(LCTM.utils.segment_labels, train_labels))
    # transition_probs, initial_probs, final_probs = fsm.smoothCounts(
    #     *fsm.countSeqs(train_labels), structure_only=True, as_numpy=True
    # )

    transition_probs = np.zeros((model.n_classes, model.n_classes), dtype=float)
    for cur_state, next_states in transitions.items():
        for next_state in next_states:
            transition_probs[cur_state, next_state] = 1

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='divide by zero')
        model.ws['seg_pw'] = np.log(transition_probs)
    # model.ws['pw'][transition_probs == 0] = -np.inf

    return model


def plot_weights(model, fn=None):
    subplot_width = 12
    subplot_height = 3

    ws = {name: weights for name, weights in model.ws.items() if name != 'pre'}

    num_axes = len(ws)

    if not num_axes:
        return

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize)

    if num_axes == 1:
        axes = [axes]

    for axis, (name, weights) in zip(axes, ws.items()):
        if weights.ndim > 2:
            logger.info(f"Flattening {name} weights with shape {weights.shape}")
            weights = weights.reshape(weights.shape[0], -1)
        axis.imshow(weights, interpolation='None')
        axis.set_title(f"{name}:  min {weights.min():.02f}  max {weights.max():.02f}")

    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def plot_train(objectives, fn=None):
    """ Save or display a figure summarizing the model's training behavior.

    Parameters
    ----------
    objectives : dict(int -> float)
        Maps iteration indices to objective values.
    fn : string, optional
    """

    figsize = (12, 3)
    fig, axis = plt.subplots(1, figsize=figsize)

    indices = np.array(list(objectives.keys()))
    values = np.array(list(objectives.values()))

    axis.plot(indices, values)

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def main(
        out_dir=None, data_dir=None, scores_dir=None, model_name=None,
        results_file=None, sweep_param_name=None,
        pre_init_pw=None, transitions=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        plot_predictions=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, f'results.csv')
        write_mode = 'w'
    else:
        results_file = os.path.expanduser(results_file)
        write_mode = 'a'

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
    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=')
    feature_seqs = loadAll(trial_ids, 'feature-seq.pkl', data_dir)
    label_seqs = loadAll(trial_ids, 'label-seq.pkl', data_dir)

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

        train_samples, train_labels, train_ids = train_data
        val_samples, val_labels, val_ids = val_data
        test_samples, test_labels, test_ids = test_data

        # Transpose input data so they have shape (num_features, num_samples),
        # to conform with LCTM interface
        # train_samples = preprocess(train_samples)
        # test_samples = preprocess(test_samples)
        # val_samples = preprocess(val_samples)

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        model = getattr(LCTM.models, model_name)(**model_params)

        if pre_init_pw:
            pretrain = train_params.get('pretrain', True)
            model = pre_init(
                model, train_samples, train_labels,
                pretrain=pretrain, transitions=transitions
            )
        else:
            model.fit(train_samples, train_labels, **train_params)
            # FIXME: Is this even necessary?
            if model_params.get('inference', None) == 'segmental':
                model.max_segs = LCTM.utils.max_seg_count(train_labels)

        train_fig_dir = os.path.join(fig_dir, 'train-plots')
        if not os.path.exists(train_fig_dir):
            os.makedirs(train_fig_dir)

        plot_weights(
            model, fn=os.path.join(train_fig_dir, f"cvfold={cv_index}_model-weights-trained.png")
        )
        plot_train(
            model.logger.objectives,
            fn=os.path.join(train_fig_dir, f"cvfold={cv_index}_train-loss.png")
        )

        # Test model
        pred_labels = model.predict(test_samples)
        # test_samples = tuple(map(lambda x: x.swapaxes(0, 1), test_samples))
        test_io_history = tuple(
            zip([pred_labels], [test_samples], [test_samples], [test_labels], [test_ids])
        )

        for name in metric_dict.keys():
            value = getattr(LCTM.metrics, name)(pred_labels, test_labels)
            metric_dict[name] += [value]
        metric_str = '  '.join(f"{k}: {v[-1]:.1f}%" for k, v in metric_dict.items())
        logger.info('[TST]  ' + metric_str)

        all_labels = np.hstack(test_labels)
        label_hist = utils.makeHistogram(len(np.unique(all_labels)), all_labels, normalize=True)
        logger.info(f'Label distribution: {label_hist}')

        d = {k: v[-1] / 100 for k, v in metric_dict.items()}
        utils.writeResults(
            results_file, d, sweep_param_name, model_params,
            write_mode=write_mode
        )

        if plot_predictions:
            io_fig_dir = os.path.join(fig_dir, 'model-io')
            if not os.path.exists(io_fig_dir):
                os.makedirs(io_fig_dir)

            label_names = ('gt', 'pred')
            preds, scores, inputs, gt_labels, ids = zip(*test_io_history)
            for batch in test_io_history:
                for preds, _, inputs, gt_labels, seq_id in zip(*batch):
                    fn = os.path.join(io_fig_dir, f"trial={seq_id}_model-io.png")
                    utils.plot_array(inputs, (gt_labels, preds), label_names, fn=fn)

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq, f'trial={trial_id}_score-seq')
            saveVariable(label_seq, f'trial={trial_id}_true-label-seq')
        for io in test_io_history:
            saveTrialData(*io)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')


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
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
