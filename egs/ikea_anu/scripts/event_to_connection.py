import os
import logging
import json
import itertools
import time

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
        init_weights=None, final_weights=None,
        input_table=None, output_table=None,
        arc_type='standard'):

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_table)
    fst.set_output_symbols(output_table)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    if init_weights is None:
        init_weights = tuple(float(one) for __ in range(num_states))

    if final_weights is None:
        final_weights = tuple(float(one) for __ in range(num_states))

    fst.set_start(fst.add_state())

    def makeState(i):
        state = fst.add_state()

        init_weight = openfst.Weight(fst.weight_type(), init_weights[i])
        if init_weight != zero:
            connection_id = i + 1
            action_id = libfst.EPSILON
            arc = openfst.Arc(action_id, connection_id, init_weight, state)
            fst.add_arc(fst.start(), arc)

        final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
        if final_weight != zero:
            fst.set_final(state, final_weight)

        return state

    states = tuple(makeState(i) for i in range(num_states))

    for i_action, row in enumerate(transition_weights):
        for i_edge, tx_weight in enumerate(row):
            i_cur, i_next = transition_vocab[i_edge]
            cur_state = states[i_cur]
            next_state = states[i_next]
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            connection_id = i_next + 1
            action_id = i_action + 1
            if weight != zero:
                arc = openfst.Arc(action_id, connection_id, weight, next_state)
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
            fst.set_final(next_state, final_weight)

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


def toArray(lattice):
    zero = 0
    # zero = openfst.Weight.zero(lattice.weight_type())
    # one = openfst.Weight.one(lattice.weight_type())

    num_inputs = lattice.input_symbols().num_symbols()
    num_outputs = lattice.output_symbols().num_symbols()
    weights = np.full((num_inputs, num_outputs), float(zero))

    for state in lattice.states():
        for arc in lattice.arcs(state):
            weights[arc.ilabel, arc.olabel] += float(arc.weight)

    return weights, lattice.weight_type()


def loadPartInfo(fn):
    with open(fn, 'rt') as file_:
        data = json.load(file_)

    part_vocab = data['part_vocab']
    part_categories = data['part_categories']
    part_connections = data['part_connections']

    return part_vocab, part_categories, part_connections


def events_to_actions(event_attrs, action_vocab, event_vocab, edge_vocab, part_categories):
    def makeActiveParts(active_part_categories):
        parts = tuple(
            part_categories.get(part_category, [part_category])
            for part_category in active_part_categories
        )
        return frozenset([frozenset(prod) for prod in itertools.product(*parts)])

    num_events = len(event_vocab)
    num_actions = len(action_vocab)
    num_edges = len(edge_vocab)

    # event index --> all data
    action_integerizer = {name: i for i, name in enumerate(action_vocab)}
    edge_integerizer = {name: i for i, name in enumerate(edge_vocab)}
    event_attrs = event_attrs.set_index('event')

    part_cols = [name for name in event_attrs.columns if name.endswith('_active')]

    event_action_weights = np.zeros((num_edges, num_events, num_actions), dtype=float)
    for i_event, event_name in enumerate(event_vocab):
        row = event_attrs.loc[event_name]
        action_name = row['action']
        active_part_categories = tuple(
            name.split('_active')[0] for name in part_cols if row[name]
        )
        all_active_parts = makeActiveParts(active_part_categories)
        for i_edge, _ in enumerate(edge_vocab):
            for active_parts in all_active_parts:
                active_edge_index = edge_integerizer.get(active_parts, None)
                if active_edge_index == i_edge:
                    i_action = action_integerizer[action_name]
                    break
            else:
                # FIXME: Implement consistent background class
                i_action = action_integerizer['NA']
            event_action_weights[i_edge, i_event, i_action] = 1

    return event_action_weights


class AttributeClassifier(object):
    def __init__(
            self, event_attrs, connection_attrs, event_vocab,
            part_vocab, part_categories, part_connections,
            event_duration_weights=None):
        """
        Parameters
        ----------
        event_attrs : pd.DataFrame
        connection_attrs : pd.DataFrame
        vocab : list( string )
        """

        self.event_attrs = event_attrs
        self.connection_attrs = connection_attrs

        self.part_categories = part_categories
        self.part_connections = part_connections

        # map event index to action index
        # map action name to index: action_ids
        # map transition name to index: transition_ids

        self.event_vocab = event_vocab
        self.action_vocab = connection_attrs['action'].to_list()
        self.part_vocab = part_vocab
        self.transition_vocab = tuple(
            tuple(int(x) for x in col_name.split('->'))
            for col_name in connection_attrs.columns if col_name != 'action'
        )
        self.edge_vocab = tuple(frozenset([
            frozenset([part_1, part_2])
            for part_1, neighbors in self.part_connections.items()
            for part_2 in neighbors
        ]))
        self.connection_vocab = tuple(sorted(
            frozenset().union(*[frozenset(t) for t in self.transition_vocab])
        ))

        self.event_symbols = libfst.makeSymbolTable(self.event_vocab)
        self.action_symbols = libfst.makeSymbolTable(self.action_vocab)
        self.transition_symbols = libfst.makeSymbolTable(self.transition_vocab)
        self.edge_symbols = libfst.makeSymbolTable(self.edge_vocab)
        self.connection_symbols = libfst.makeSymbolTable(self.connection_vocab)

        self.event_duration_weights = event_duration_weights
        self.event_duration_fst = make_duration_fst(
            -np.log(self.event_duration_weights),
            symbol_table=self.event_symbols,
            arc_type='standard',
        ).arcsort(sort_type='ilabel')

        self.action_to_connection = transducer_from_connection_attrs(
            -np.log(self.connection_attrs.drop(['action'], axis=1).to_numpy()),
            self.transition_vocab,
            len(self.connection_vocab),
            init_weights=-np.log(np.array([1, 0])),
            input_table=self.action_symbols, output_table=self.connection_symbols,
            arc_type='standard'
        ).arcsort(sort_type='ilabel')

        # event index --> all data
        action_integerizer = {name: i for i, name in enumerate(self.action_vocab)}
        event_action_weights = np.zeros((self.num_events, self.num_actions), dtype=float)
        for i, name in enumerate(event_attrs.set_index('event').loc[self.event_vocab]['action']):
            event_action_weights[i, action_integerizer[name]] = 1
        event_action_weights = events_to_actions(
            self.event_attrs,
            self.action_vocab, self.event_vocab, self.edge_vocab,
            self.part_categories
        )

        self.event_to_action = []
        self.seq_models = []
        for i, edge_event_action_weights in enumerate(event_action_weights):
            event_to_action = single_state_transducer(
                -np.log(edge_event_action_weights),
                input_table=self.event_symbols, output_table=self.action_symbols,
                arc_type='standard'
            ).arcsort(sort_type='ilabel')

            seq_model = libfst.easyCompose(
                *[self.event_duration_fst, event_to_action, self.action_to_connection],
                # *[event_to_action, self.action_to_connection],
                # determinize=True,
                # minimize=True
                determinize=False,
                minimize=False

            ).arcsort(sort_type='ilabel')

            self.event_to_action.append(event_to_action)
            self.seq_models.append(seq_model)

    @property
    def num_events(self):
        return len(self.event_vocab)

    @property
    def num_parts(self):
        return len(self.part_vocab)

    @property
    def num_actions(self):
        return len(self.action_vocab)

    @property
    def num_transitions(self):
        return len(self.transition_vocab)

    @property
    def num_edges(self):
        return len(self.edge_vocab)

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
        sample_symbols = libfst.makeSymbolTable(
            tuple(f"{i}" for i in range(event_scores.shape[0]))
        )
        lattice = libfst.fromArray(
            -event_scores,
            input_symbols=sample_symbols,
            output_symbols=self.event_symbols,
            arc_type='standard'
        ).arcsort(sort_type='ilabel')

        def getScores(seq_model):
            # Compose event scores with event --> connection map
            init_time = time.time()
            decode_lattice = openfst.compose(lattice, seq_model)
            time_str = makeTimeString(time.time() - init_time)
            logger.info(f"compose finished in {time_str}")

            num_states = decode_lattice.num_states()
            num_arcs = sum(
                decode_lattice.num_arcs(i) for i in range(num_states)
            )
            logger.info(f"Decode lattice has {num_arcs} arcs; {num_states} states")

            # Compute edge posterior marginals
            init_time = time.time()
            grad_lattice = libfst.fstArcGradient(decode_lattice)
            time_str = makeTimeString(time.time() - init_time)
            logger.info(f"fstArcGradient finished in {time_str}")

            init_time = time.time()
            connection_scores, weight_type = toArray(grad_lattice)
            time_str = makeTimeString(time.time() - init_time)
            logger.info(f"toArray finished in {time_str}")

            # import pdb; pdb.set_trace()
            return connection_scores

        # connection_scores = np.stack(
        #     tuple(getScores(seq_model) for seq_model in self.seq_models),
        #     axis=-1
        # )

        lattices = tuple(openfst.compose(lattice, seq_model) for seq_model in self.seq_models)
        return lattices  # connection_scores

    def predict(self, outputs):
        """ Choose the best labels from an array of output activations.

        Parameters
        ----------
        outputs : np.ndarray of float with shape (NUM_SAMPLES, NUM_CLASSES, ...)

        Returns
        -------
        preds : np.ndarray of int with shape (NUM_SAMPLES, ...)
        """

        preds = tuple(np.array(libfst.viterbi(lattice)) for lattice in outputs),
        import pdb; pdb.set_trace()
        edge_preds = np.stack(preds, axis=-1)
        # preds = outputs.argmax(axis=1)
        return edge_preds


def makeTimeString(time_elapsed):
    mins_elapsed = time_elapsed // 60
    secs_elapsed = time_elapsed % 60
    time_str = f'{mins_elapsed:.0f}m {secs_elapsed:.0f}s'
    return time_str


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
        event_attr_fn=None, connection_attr_fn=None, part_info_fn=None,
        only_fold=None, plot_io=None, prefix='seq=',
        results_file=None, sweep_param_name=None, model_params={}, cv_params={}):

    data_dir = os.path.expanduser(data_dir)
    scores_dir = os.path.expanduser(scores_dir)
    event_attr_fn = os.path.expanduser(event_attr_fn)
    connection_attr_fn = os.path.expanduser(connection_attr_fn)
    part_info_fn = os.path.expanduser(part_info_fn)
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

    event_vocab = utils.loadVariable('vocab', data_dir)

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    # Load event, connection attributes
    event_attrs = pd.read_csv(event_attr_fn, index_col=False, keep_default_na=False)
    connection_attrs = pd.read_csv(connection_attr_fn, index_col=False, keep_default_na=False)
    part_vocab, part_categories, part_connections = loadPartInfo(part_info_fn)

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
            event_attrs, connection_attrs, event_vocab,
            part_vocab, part_categories, part_connections,
            event_duration_weights=event_duration_weights
        )

        draw_fst(
            os.path.join(fig_dir, f'cvfold={cv_index}_event-duration'),
            model.event_duration_fst,
            vertical=True, width=50, height=50, portrait=True
        )

        for i, fst in enumerate(model.event_to_action):
            draw_fst(
                os.path.join(fig_dir, f'cvfold={cv_index}_event-to-action_edge={i}'),
                fst,
                vertical=True, width=50, height=50, portrait=True
            )

        draw_fst(
            os.path.join(fig_dir, f'cvfold={cv_index}_action-to-connection'),
            model.action_to_connection,
            vertical=True, width=50, height=50, portrait=True
        )

        # draw_fst(
        #     os.path.join(fig_dir, f'cvfold={cv_index}_seq-model'),
        #     model.seq_model,
        #     vertical=True, width=50, height=50, portrait=True
        # )

        for i in test_indices:
            seq_id = seq_ids[i]
            logger.info(f"  Processing sequence {seq_id}...")

            trial_prefix = f"{prefix}{seq_id}"
            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", scores_dir)
            # FIXME: the serialized variables are probs, not log-probs
            event_score_seq = np.log(event_score_seq)
            # true_seq = utils.loadVariable(f"{trial_prefix}_true-label-seq", scores_dir)

            logger.info(f"    event scores shape: {event_score_seq.shape}")

            connection_score_seq = model.forward(event_score_seq)
            # logger.info(f"    connection scores shape: {connection_score_seq.shape}")
            # draw_fst(
            #     os.path.join(fig_dir, f'seq={seq_id}_connection-scores'),
            #     connection_score_seq,
            #     vertical=True, width=50, height=50, portrait=True
            # )
            # print(connection_score_seq)
            # import pdb; pdb.set_trace()
            pred_connection_seq = model.predict(connection_score_seq)

            # metric_dict = eval_metrics(pred_seq, true_seq)
            # for name, value in metric_dict.items():
            #     logger.info(f"    {name}: {value * 100:.2f}%")

            # utils.saveVariable(connection_score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_connection_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            # utils.saveVariable(true_event_seq, f'{seq_id_str}_true-label-seq', out_data_dir)
            # utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if plot_io:
                utils.plot_array(
                    # connection_score_seq.T, (true_seq, pred_seq), ('true', 'pred'),
                    event_score_seq.T, (pred_connection_seq,), ('pred',),
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
