import argparse
import os
import itertools
import functools

import yaml
import joblib
import numpy as np
from matplotlib import pyplot as plt

import LCTM.models
import LCTM.metrics
import LCTM.utils
import LCTM.learn

from mathtools import utils
from seqtools import fsm
from kinemparse import imu


def split(imu_feature_seqs, imu_label_seqs, trial_ids, active_only=False):
    num_signals = imu_label_seqs[0].shape[1]

    def validate(seqs):
        return all(seq.shape[1] == num_signals for seq in seqs)
    all_valid = all(validate(x) for x in (imu_feature_seqs, imu_label_seqs))
    if not all_valid:
        raise AssertionError("IMU and labels don't all have the same number of sequences")

    trial_ids = tuple(
        itertools.chain(
            *(
                tuple(t_id + 0.01 * (i + 1) for i in range(num_signals))
                for t_id in trial_ids
            )
        )
    )

    def splitSeq(arrays):
        return tuple(row for array in arrays for row in array)
    imu_feature_seqs = splitSeq(map(lambda x: x.swapaxes(0, 1), imu_feature_seqs))
    imu_label_seqs = splitSeq(map(lambda x: x.T, imu_label_seqs))

    if active_only:
        is_active = tuple(map(np.any, imu_label_seqs))

        def filterInactive(arrays):
            return tuple(arr for arr, act in zip(arrays, is_active) if act)
        return tuple(map(filterInactive, (imu_feature_seqs, imu_label_seqs, trial_ids)))

    return imu_feature_seqs, imu_label_seqs, trial_ids


def preprocess(imu_feature_seqs):
    def preprocessSeq(feature_seq):
        return feature_seq.swapaxes(0, 1)

    preprocessed_feature_seqs = tuple(preprocessSeq(x) for x in imu_feature_seqs)

    return preprocessed_feature_seqs


def pre_init(model, train_samples, train_labels, pretrain=True):
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
    train_labels = tuple(map(LCTM.utils.segment_labels, train_labels))
    transition_probs, initial_probs, final_probs = fsm.smoothCounts(
        *fsm.countSeqs(train_labels), structure_only=False, as_numpy=True
    )

    # model.ws['pw'] = np.log(transition_probs)
    model.ws['pw'][transition_probs == 0] = -np.inf

    return model


def plot_weights(model, fn=None):
    subplot_width = 12
    subplot_height = 3

    num_axes = len(model.ws)

    if not num_axes:
        return

    figsize = (subplot_width, num_axes * subplot_height)
    fig, axes = plt.subplots(num_axes, figsize=figsize)

    for axis, (name, weights) in zip(axes, model.ws.items()):
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
        out_dir=None, data_dir=None, model_name=None,
        independent_signals=None, active_only=None,
        label_mapping=None, eval_label_mapping=None, pre_init_pw=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        viz_output=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

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
    imu_sample_seqs = loadVariable('imu_sample_seqs')
    imu_label_seqs = loadVariable('imu_label_seqs')

    if label_mapping is not None:
        def map_labels(labels):
            for i, j in label_mapping.items():
                labels[labels == i] = j
            return labels
        imu_label_seqs = tuple(map(map_labels, imu_label_seqs))

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(
        imu_sample_seqs, imu_label_seqs, trial_ids,
        **cv_params
    )

    metric_dict = {
        'accuracy': [],
        'edit_score': [],
        'overlap_score': []
    }

    for cv_index, (train_data, val_data, test_data) in enumerate(cv_folds):
        if independent_signals:
            split_ = functools.partial(split, active_only=active_only)
            train_samples, train_labels, train_ids = split_(*train_data)
            test_samples, test_labels, test_ids = split_(*test_data)
            val_samples, val_labels, val_ids = split_(*val_data)

            # Transpose input data so they have shape (num_features, num_samples),
            # to conform with LCTM interface
            train_samples = preprocess(train_samples)
            test_samples = preprocess(test_samples)
            val_samples = preprocess(val_samples)

        else:
            raise NotImplementedError()

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        model = getattr(LCTM.models, model_name)(**model_params)

        if pre_init_pw:
            pretrain = train_params.get('pretrain', True)
            model = pre_init(model, train_samples, train_labels, pretrain=pretrain)
        plot_weights(model, fn=os.path.join(fig_dir, f"cvfold={cv_index}_model-weights-init.png"))

        model.fit(train_samples, train_labels, **train_params)
        if model_params.get('inference', None) == 'segmental':
            model.max_segs = LCTM.utils.max_seg_count(train_labels)

        plot_weights(
            model, fn=os.path.join(fig_dir, f"cvfold={cv_index}_model-weights-trained.png")
        )
        plot_train(
            model.logger.objectives,
            fn=os.path.join(fig_dir, f"cvfold={cv_index}_train-loss.png")
        )

        # Test model
        pred_labels = model.predict(test_samples)
        for name in metric_dict.keys():
            value = getattr(LCTM.metrics, name)(pred_labels, test_labels)
            metric_dict[name] += [value]
        metric_str = '  '.join(f"{k}: {v[-1]}" for k, v in metric_dict.items())
        logger.info('[TST]  ' + metric_str)

        all_labels = np.hstack(test_labels)
        label_hist = utils.makeHistogram(len(np.unique(all_labels)), all_labels, normalize=True)
        logger.info(f'Label distribution: {label_hist}')

        if viz_output:
            # imu.plot_prediction_eg(test_io_history, fig_dir, fig_type=fig_type, **viz_params)
            test_samples = tuple(map(lambda x: x.swapaxes(0, 1), test_samples))
            test_io_history = tuple(zip(pred_labels, test_samples, test_labels, test_ids))
            imu.plot_prediction_eg(test_io_history, fig_dir, **viz_params)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        if eval_label_mapping is not None:
            def map_labels(labels):
                for i, j in eval_label_mapping.items():
                    labels[labels == i] = j
                return labels
            pred_labels = tuple(map(map_labels, pred_labels))
            test_labels = tuple(map(map_labels, test_labels))
            for name in metric_dict.keys():
                value = getattr(LCTM.metrics, name)(pred_labels, test_labels)
                metric_dict[name] += [value]
            metric_str = '  '.join(f"{k}: {v[-1]}" for k, v in metric_dict.items())
            logger.info('[TST]  ' + metric_str)

            all_labels = np.hstack(test_labels)
            label_hist = utils.makeHistogram(len(np.unique(all_labels)), all_labels, normalize=True)
            logger.info(f'Label distribution: {label_hist}')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
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
    config.update(args)

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
