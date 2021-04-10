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


def draw_fst(fn, fst, extension='png', **draw_kwargs):
    fst.draw(fn, **draw_kwargs)
    gv.render('dot', extension, fn)
    os.remove(fn)


def transducer_from_connection_attrs(
        transition_weights, transition_vocab, num_states,
        input_table=None, output_table=None,
        arc_type='standard'):

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


def durationFst(label, num_states, final_weights=None, arc_type='standard', symbol_table=None):
    """ Construct a left-to-right WFST from an input sequence.

    Parameters
    ----------
    input_seq : iterable(int or string)

    Returns
    -------
    fst : openfst.Fst
    """

    if num_states < 1:
        raise AssertionError(f"num_states = {num_states}, but should be >= 1)")

    fst = openfst.VectorFst(arc_type=arc_type)
    one = openfst.Weight.one(fst.weight_type())
    zero = openfst.Weight.zero(fst.weight_type())

    if final_weights is None:
        final_weights = [one for __ in range(num_states)]

    if symbol_table is not None:
        fst.set_input_symbols(symbol_table)
        fst.set_output_symbols(symbol_table)

    init_state = fst.add_state()
    fst.set_start(init_state)

    cur_state = fst.add_state()
    arc = openfst.Arc(libfst.EPSILON, label + 1, one, cur_state)
    fst.add_arc(init_state, arc)

    for i in range(num_states):
        next_state = fst.add_state()

        arc = openfst.Arc(label + 1, libfst.EPSILON, one, next_state)
        fst.add_arc(cur_state, arc)

        final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
        if final_weight != zero:
            arc = openfst.Arc(label + 1, libfst.EPSILON, final_weight, cur_state)
            fst.add_arc(cur_state, arc)
            fst.set_final(cur_state, final_weight)

        cur_state = next_state

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


def make_duration_fst(final_weights, arc_type='standard', symbol_table=None):
    num_classes, num_states = final_weights.shape

    dur_fsts = [
        durationFst(
            i, num_states, final_weights=final_weights[i],
            arc_type=arc_type, symbol_table=symbol_table
        )
        for i in range(num_classes)
    ]

    dur_fst = dur_fsts[0].union(*dur_fsts[1:]).closure(closure_plus=True)

    return dur_fst


def single_state_transducer(
        transition_weights, input_table=None, output_table=None,
        arc_type='standard'):

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_table)
    fst.set_output_symbols(output_table)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    state = fst.add_state()
    fst.set_start(state)
    fst.set_final(state, one)

    for i_input, row in enumerate(transition_weights):
        for i_output, tx_weight in enumerate(row):
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            input_id = i_input + 1
            output_id = i_output + 1
            if weight != zero:
                arc = openfst.Arc(input_id, output_id, weight, state)
                fst.add_arc(state, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


class AttributeClassifier(object):
    def __init__(self, event_attrs, connection_attrs, vocab, event_duration_weights=None):
        """
        Parameters
        ----------
        event_attrs : pd.DataFrame
        connection_attrs : pd.DataFrame
        vocab : list( string )
        """

        self.event_attrs = event_attrs
        self.connection_attrs = connection_attrs

        # map event index to action index
        # map action name to index: action_ids
        # map transition name to index: transition_ids

        self.event_vocab = vocab
        self.action_vocab = connection_attrs['action'].to_list()
        self.transition_vocab = [
            tuple(int(x) for x in col_name.split('->'))
            for col_name in connection_attrs.columns if col_name != 'action'
        ]

        self.num_events = len(self.event_vocab)
        self.num_actions = len(self.action_vocab)
        self.num_transitions = len(self.transition_vocab)

        self.event_symbols = libfst.makeSymbolTable(self.event_vocab)
        self.action_symbols = libfst.makeSymbolTable(self.action_vocab)
        self.transition_symbols = libfst.makeSymbolTable(self.transition_vocab)

        self.event_duration_weights = event_duration_weights
        self.event_duration_fst = make_duration_fst(
            -np.log(self.event_duration_weights),
            symbol_table=self.event_symbols,
            arc_type='standard',
        )

        # event index --> all data
        action_integerizer = {name: i for i, name in enumerate(self.action_vocab)}
        event_action_weights = np.zeros((self.num_events, self.num_actions), dtype=float)
        for i, name in enumerate(event_attrs.set_index('event').loc[self.event_vocab]['action']):
            event_action_weights[i, action_integerizer[name]] = 1
        self.event_to_action = single_state_transducer(
            -np.log(event_action_weights),
            input_table=self.event_symbols, output_table=self.action_symbols,
            arc_type='standard'
        )

        self.action_to_connection = transducer_from_connection_attrs(
            -np.log(self.connection_attrs.drop(['action'], axis=1).to_numpy()),
            self.transition_vocab,
            max(max(tx) for tx in self.transition_vocab) + 1,
            input_table=self.action_symbols, output_table=self.transition_symbols,
            arc_type='standard'
        )

        self.seq_model = libfst.easyCompose(
            *[self.event_duration_fst, self.event_to_action, self.action_to_connection],
            # determinize=True,
            # minimize=True
            # *[self.event_to_action, self.action_to_connection],
            determinize=False,
            minimize=False

        )

    def forward(self, event_scores):
        """ Combine action and part scores into event scores.

        Parameters
        ----------
        event_scores : np.ndarray of float with shape (NUM_SAMPLES, NUM_EVENTS)

        Returns
        -------
        connection_scores : np.ndarray of float with shape (NUM_SAMPLES, 2, NUM_EDGES)
        """

        # Convert event scores to lattice
        lattice = libfst.fromArray(
            -event_scores,
            input_symbols=None,
            output_symbols=self.event_symbols,
            arc_type='standard'
        )

        # Compose event scores with event --> connection map
        connection_scores = openfst.compose(lattice, self.seq_model)
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

        event_duration_weights = np.ones((event_attrs.shape[0], 10), dtype=float)
        model = AttributeClassifier(
            event_attrs, connection_attrs, vocab,
            event_duration_weights=event_duration_weights
        )

        draw_fst(
            os.path.join(fig_dir, f'cvfold={cv_index}_event-duration'),
            model.event_duration_fst,
            vertical=True, width=50, height=50, portrait=True
        )

        draw_fst(
            os.path.join(fig_dir, f'cvfold={cv_index}_event-to-action'),
            model.event_to_action,
            vertical=True, width=50, height=50, portrait=True
        )

        draw_fst(
            os.path.join(fig_dir, f'cvfold={cv_index}_action-to-connection'),
            model.action_to_connection,
            vertical=True, width=50, height=50, portrait=True
        )

        draw_fst(
            os.path.join(fig_dir, f'cvfold={cv_index}_seq-model'),
            model.seq_model,
            vertical=True, width=50, height=50, portrait=True
        )

        for i in test_indices:
            seq_id = seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"{prefix}{seq_id}"
            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            # true_seq = utils.loadVariable(f"{trial_prefix}_true-label-seq", scores_dir)

            connection_score_seq = model.forward(event_score_seq)
            # draw_fst(
            #     os.path.join(fig_dir, f'seq={seq_id}_connection-scores'),
            #     connection_score_seq,
            #     vertical=True, width=50, height=50, portrait=True
            # )
            print(connection_score_seq)
            import pdb; pdb.set_trace()
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
                    fn=os.path.join(fig_dir, f"seq={seq_id:03d}.png")
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
