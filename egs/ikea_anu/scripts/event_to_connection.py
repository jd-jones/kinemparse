import os
import logging

import yaml
import numpy as np
# from matplotlib import pyplot as plt

import LCTM.metrics

from mathtools import utils


logger = logging.getLogger(__name__)


class AttributeClassifier(object):
    def __init__(self, event_attrs, connection_attrs):
        """
        Parameters
        ----------
        action_part_counts : np.ndarray of int with shape (NUM_ACTIONS, NUM_PARTS)
        """
        self.event_attrs = event_attrs
        self.connection_attrs = connection_attrs

    def forward(self, action_scores, part_scores):
        """ Combine action and part scores into event scores.

        Parameters
        ----------
        action_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS)
        part_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_PARTS)

        Returns
        -------
        event_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS, NUM_PARTS)
        """

        # FIXME
        event_scores = (
            action_scores[:, :, None]
            + part_scores[:, None, :]
            + self.feat_ap[None, :, :]
        )
        return event_scores

    def predict(self, outputs):
        """ Choose the best labels from an array of output activations.

        Parameters
        ----------
        outputs : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS, NUM_PARTS)

        Returns
        -------
        preds : np.ndarray of int with shape (NUM_SAMPLES, 2)
            preds[:, 0] contains the index of the action predicted for each sample.
            preds[:, 1] contains the index of the part predicted for each sample.
        """

        # FIXME
        pair_scores = outputs.reshape(outputs.shape[0], -1)
        pair_preds = pair_scores.argmax(axis=-1)
        # preds = np.column_stack(np.unravel_index(pair_preds, outputs.shape[1:]))
        preds = np.unravel_index(pair_preds, outputs.shape[1:])
        return preds


def eval_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
    state_acc = (pred_seq == true_seq).astype(float).mean()

    metric_dict = {
        'State Accuracy' + name_suffix: state_acc,
        'State Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'State Overlap Score' + name_suffix: LCTM.metrics.overlap_score(pred_seq, true_seq) / 100
    }

    append_to.update(metric_dict)
    return append_to


def main(
        out_dir=None, data_dir=None, scores_dir=None,
        only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    io_dir_images = os.path.join(fig_dir, 'model-io_images')
    if not os.path.exists(io_dir_images):
        os.makedirs(io_dir_images)

    io_dir_plots = os.path.join(fig_dir, 'model-io_plots')
    if not os.path.exists(io_dir_plots):
        os.makedirs(io_dir_plots)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix='labels.*',
        to_array=True
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {scores_dir}")

    # vocab = utils.loadVariable('vocab', data_dir)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        # TODO: LOAD EVENT, CONNECTION ATTRIBUTES
        model = AttributeClassifier(event_attrs, connection_attrs)

        for i in test_indices:
            seq_id = seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"{prefix}{seq_id}"
            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            # true_seq = utils.loadVariable(f"{trial_prefix}_true-label-seq", scores_dir)

            connection_score_seq = model.forward(event_score_seq)
            pred_connection_seq = model.predict(connection_score_seq)

            # metric_dict = eval_metrics(pred_seq, true_seq)
            # for name, value in metric_dict.items():
            #     logger.info(f"    {name}: {value * 100:.2f}%")

            utils.saveVariable(event_score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_connection_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            # utils.saveVariable(true_event_seq, f'{seq_id_str}_true-label-seq', out_data_dir)
            # utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if plot_io:
                utils.plot_array(
                    # connection_score_seq.T, (true_seq, pred_seq), ('true', 'pred'),
                    connection_score_seq.T, (pred_connection_seq,), ('pred',),
                    fn=os.path.join(io_dir_plots, f"seq={seq_id:03d}.png")
                )


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