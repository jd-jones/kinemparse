import os
import logging

import yaml
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import graphviz as gv

import LCTM.metrics
import pywrapfst as openfst

from mathtools import utils
from seqtools import fstutils_openfst as libfst


logger = logging.getLogger(__name__)


def transducer_from_connection_attrs(
        transition_weights, transition_vocab, num_states,
        input_table=None, output_table=None,
        arc_type='standard'):
    num_inputs, num_edges = transition_weights.shape

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_table)
    fst.set_output_symbols(output_table)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    fst.set_start(fst.add_state())

    def makeState(i):
        state = fst.add_state()
        arc = openfst.Arc(libfst.EPSILON, libfst.EPSILON, one, state)
        fst.add_arc(fst.start(), arc)
        fst.set_final(state, one)
        return state

    states = tuple(makeState(i) for i in range(num_states))

    for i_action, row in enumerate(transition_weights):
        for i_edge, tx_weight in enumerate(row):
            i_cur, i_next = transition_vocab[i_edge]
            cur_state = states[i_cur]
            next_state = states[i_next]
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            transition_id = i_edge + 1
            action_id = i_action + 1
            if weight != zero:
                arc = openfst.Arc(action_id, transition_id, weight, next_state)
                fst.add_arc(cur_state, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


class AttributeClassifier(object):
    def __init__(self, event_attrs, connection_attrs, vocab):
        """
        Parameters
        ----------
        event_attrs : pd.DataFrame
        connection_attrs : pd.DataFrame
        vocab : list( string )
        """

        self.event_attrs = event_attrs
        self.connection_attrs = connection_attrs
        self.event_vocab = vocab

        # map event index to action index
        # map action name to index: action_ids
        # map transition name to index: transition_ids

        transition_vocab = [
            tuple(int(x) for x in col_name.split('->'))
            for col_name in connection_attrs.columns if col_name != 'action'
        ]
        # transition_integerizer = {x: i for i, x in enumerate(transition_vocab)}
        num_states = max(max(tx) for tx in transition_vocab) + 1

        # event index --> all data
        action_vocab = connection_attrs['action'].to_list()
        # action_integerizer = {name: i for i, name in enumerate(action_vocab)}
        # event_idx_to_action_idx = tuple(
        #     action_integerizer[name]
        #     for name in event_attrs.set_index('event').loc[self.event_vocab]['action']
        # )

        self.input_table = libfst.makeSymbolTable(action_vocab)
        self.output_table = libfst.makeSymbolTable(transition_vocab)

        # FIXME: Should take the log of transition weights
        transition_weights = self.connection_attrs.drop(['action'], axis=1).to_numpy()
        self.lattice = transducer_from_connection_attrs(
            transition_weights, transition_vocab, num_states,
            input_table=self.input_table, output_table=self.output_table,
            arc_type='standard'
        )

    def forward(self, event_scores):
        """ Combine action and part scores into event scores.

        Parameters
        ----------
        event_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_ACTIONS, NUM_PARTS)

        Returns
        -------
        connection_scores : np.ndarray of float with shape (NUM_SAMPLES, 2, NUM_EDGES)
        """

        # FIXME
        connection_scores = None
        return connection_scores

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
        event_attr_fn=None, connection_attr_fn=None,
        only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
    event_attr_fn = os.path.expanduser(event_attr_fn)
    connection_attr_fn = os.path.expanduser(connection_attr_fn)
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

    vocab = utils.loadVariable('vocab', data_dir)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    # Load event, connection attributes
    event_attrs = pd.read_csv(event_attr_fn, index_col=False, keep_default_na=False)
    connection_attrs = pd.read_csv(connection_attr_fn, index_col=False, keep_default_na=False)

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        model = AttributeClassifier(event_attrs, connection_attrs, vocab)
        # Draw model.lattice
        fn = os.path.join(fig_dir, f'cvfold={cv_index}_lattice')
        model.lattice.draw(
            fn, isymbols=model.input_table, osymbols=model.output_table,
            vertical=True,
            width=50, height=50,
            portrait=True
        )
        gv.render('dot', 'png', fn)
        import pdb; pdb.set_trace()

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
