import os
import warnings
import logging

import yaml
import numpy as np
from matplotlib import pyplot as plt

# Stop numba from throwing a bunch of warnings when it compiles LCTM
from numba import NumbaWarning; warnings.filterwarnings('ignore', category=NumbaWarning)

import LCTM.models
import LCTM.metrics
import LCTM.utils
import LCTM.learn

from mathtools import utils
from seqtools import utils as su


logger = logging.getLogger(__name__)


def preprocess(imu_feature_seqs):
    def preprocessSeq(feature_seq):
        return feature_seq.swapaxes(0, 1)

    preprocessed_feature_seqs = tuple(preprocessSeq(x) for x in imu_feature_seqs)

    return preprocessed_feature_seqs


def pre_init(
        model, train_samples, train_labels, pretrain=True, num_states=None,
        transition_scores=None, start_scores=None, end_scores=None):
    if num_states is None:
        num_states = np.max(list(map(np.max, train_labels))) + 1

    n_samples = len(train_samples)
    model.n_features = train_samples[0].shape[0]
    model.n_classes = num_states
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

    if transition_scores is not None:
        model.ws['seg_pw'] = transition_scores

    if start_scores is not None:
        model.ws['start'] = start_scores

    if end_scores is not None:
        model.ws['end'] = end_scores

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
        if weights.ndim == 1:
            weights = weights[None, :]
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
        feature_fn_format='feature-seq.pkl', label_fn_format='label_seq.pkl',
        pre_init_pw=None, transitions=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        metric_names=['accuracy', 'edit_score', 'overlap_score'],
        plot_predictions=None):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
        if os.path.exists(results_file):
            os.remove(results_file)
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    io_fig_dir = os.path.join(fig_dir, 'model-io')
    if not os.path.exists(io_fig_dir):
        os.makedirs(io_fig_dir)

    train_fig_dir = os.path.join(fig_dir, 'train-plots')
    if not os.path.exists(train_fig_dir):
        os.makedirs(train_fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name, to_dir=out_data_dir):
        return utils.saveVariable(var, var_name, to_dir)

    # Load data
    trial_ids = utils.getUniqueIds(
        data_dir, prefix='trial=', suffix=feature_fn_format,
        to_array=True
    )
    dataset = utils.CvDataset(
        trial_ids, data_dir,
        feature_fn_format=feature_fn_format, label_fn_format=label_fn_format,
        feat_transform=lambda x: np.swapaxes(x, 0, 1)
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)
    utils.saveVariable(dataset.vocab, 'vocab', out_data_dir)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(trial_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    metric_dict = {name: [] for name in metric_names}
    for cv_index, cv_fold in enumerate(cv_folds):
        train_data, val_data, test_data = dataset.getFold(cv_fold)
        train_samples, train_labels, train_ids = train_data
        val_samples, val_labels, val_ids = val_data
        test_samples, test_labels, test_ids = test_data

        logger.info(
            f'CV fold {cv_index + 1}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        model = getattr(LCTM.models, model_name)(**model_params)

        # -=( TRAIN PHASE )==--------------------------------------------------
        if pre_init_pw:
            if transitions is None:
                transition_probs, start_probs, end_probs = su.smoothCounts(
                    *su.countSeqs(train_labels),
                    num_states=dataset.num_states
                )
                # Overwrite transitions by loading from file, if it exists
                if os.path.exists(os.path.join(data_dir, 'transition-probs.npy')):
                    logger.info("Overriding transition probs from file")
                    transition_probs = utils.loadVariable('transition-probs', from_dir=data_dir)
            else:
                transition_probs = np.zeros((model.n_classes, model.n_classes), dtype=float)
                for cur_state, next_states in transitions.items():
                    for next_state in next_states:
                        transition_probs[cur_state, next_state] = 1
                start_probs = np.ones(model.n_classes)
                end_probs = np.ones(model.n_classes)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='divide by zero')
                transition_scores = np.log(transition_probs)
                start_scores = np.log(start_probs)
                end_scores = np.log(end_probs)

            pretrain = train_params.get('pretrain', True)
            model = pre_init(
                model, train_samples, train_labels,
                pretrain=pretrain, transition_scores=transition_scores,
                start_scores=start_scores, end_scores=end_scores,
                num_states=dataset.num_states
            )
        else:
            model.fit(train_samples, train_labels, **train_params)
            if model_params.get('inference', None) == 'segmental':
                model.max_segs = LCTM.utils.max_seg_count(train_labels)

        plot_weights(
            model, fn=os.path.join(train_fig_dir, f"cvfold={cv_index}_model-weights-trained.png")
        )
        plot_train(
            model.logger.objectives,
            fn=os.path.join(train_fig_dir, f"cvfold={cv_index}_train-loss.png")
        )

        # -=( TEST PHASE )==---------------------------------------------------
        pred_labels = model.predict(test_samples)
        test_io_history = tuple(
            zip([pred_labels], [test_samples], [test_samples], [test_labels], [test_ids])
        )

        for name in metric_dict.keys():
            value = getattr(LCTM.metrics, name)(pred_labels, test_labels)
            metric_dict[name] += [value]
        metric_str = '  '.join(f"{k}: {v[-1]:.1f}%" for k, v in metric_dict.items())
        logger.info('[TST]  ' + metric_str)

        utils.writeResults(
            results_file, {k: v[-1] / 100 for k, v in metric_dict.items()},
            sweep_param_name, model_params
        )

        # -=( MAKE OUTPUT )==--------------------------------------------------
        if plot_predictions:
            label_names = ('gt', 'pred')
            preds, scores, inputs, gt_labels, ids = zip(*test_io_history)
            for batch in test_io_history:
                for preds, _, inputs, gt_labels, seq_id in zip(*batch):
                    fn = os.path.join(io_fig_dir, f"trial={seq_id}_model-io.png")
                    utils.plot_array(inputs, (gt_labels, preds), label_names, fn=fn, **viz_params)

        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')
        for batch in test_io_history:
            for pred_seq, score_seq, feat_seq, label_seq, trial_id in zip(*batch):
                saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
                saveVariable(score_seq.T, f'trial={trial_id}_score-seq')
                saveVariable(label_seq, f'trial={trial_id}_true-label-seq')


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
