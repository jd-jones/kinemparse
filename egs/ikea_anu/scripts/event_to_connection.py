import os
import logging
import json
import itertools

import yaml
import numpy as np
import scipy
import pandas as pd
# from matplotlib import pyplot as plt
import graphviz as gv

import LCTM.metrics
import pywrapfst as openfst

from mathtools import utils
from seqtools import fstutils_openfst as libfst


logger = logging.getLogger(__name__)


def draw_fst(fn, fst, extension='pdf', **draw_kwargs):
    fst.draw(fn, **draw_kwargs)
    gv.render('dot', extension, fn)
    os.remove(fn)


def transducer_from_connection_attrs(
        transition_weights, transition_vocab, num_states,
        input_vocab, output_vocab,
        input_seg_to_str, output_seg_to_str,
        init_weights=None, final_weights=None,
        input_table=None, output_table=None,
        eps_str='ε', bos_str='<BOS>', eos_str='<EOS>',
        seg_internal_str='I', seg_final_str='F',
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
    final_state = fst.add_state()
    fst.set_final(final_state, one)

    def makeState(i):
        state = fst.add_state()

        init_weight = openfst.Weight(fst.weight_type(), init_weights[i])
        if init_weight != zero:
            input_id = fst.input_symbols().find(bos_str)
            output_id = fst.output_symbols().find(eps_str)
            arc = openfst.Arc(input_id, output_id, init_weight, state)
            fst.add_arc(fst.start(), arc)

        final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
        if final_weight != zero:
            output_str = output_seg_to_str[output_vocab[i], seg_final_str]
            input_id = fst.input_symbols().find(eos_str)
            output_id = fst.output_symbols().find(output_str)
            arc = openfst.Arc(input_id, output_id, final_weight, final_state)
            fst.add_arc(state, arc)

        return state

    states = tuple(makeState(i) for i in range(num_states))

    for i_action, row in enumerate(transition_weights):
        state_has_action_segment_internal_loop = set([])
        for i_edge, tx_weight in enumerate(row):
            i_cur, i_next = transition_vocab[i_edge]
            cur_state = states[i_cur]
            next_state = states[i_next]
            weight = openfst.Weight(fst.weight_type(), tx_weight)

            # CASE 1: (a, I) : (c_cur, I), self-loop, weight one
            input_str = input_seg_to_str[input_vocab[i_action], seg_internal_str]
            output_str = output_seg_to_str[output_vocab[i_cur], seg_internal_str]
            input_id = fst.input_symbols().find(input_str)
            output_id = fst.output_symbols().find(output_str)
            if weight != zero and not (i_cur in state_has_action_segment_internal_loop):
                arc = openfst.Arc(input_id, output_id, one, cur_state)
                fst.add_arc(cur_state, arc)
                state_has_action_segment_internal_loop.add(i_cur)

            # CASE 2: (a, F) : (c_cur, I), transition arc, weight tx_weight
            input_str = input_seg_to_str[input_vocab[i_action], seg_final_str]
            output_str = output_seg_to_str[output_vocab[i_cur], seg_final_str]
            input_id = fst.input_symbols().find(input_str)
            output_id = fst.output_symbols().find(output_str)
            if weight != zero:
                arc = openfst.Arc(input_id, output_id, weight, next_state)
                fst.add_arc(cur_state, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


def add_endpoints(fst, bos_str='<BOS>', eos_str='<EOS>'):
    one = openfst.Weight.one(fst.weight_type())
    zero = openfst.Weight.zero(fst.weight_type())

    # add pre-initial state accepting BOS
    i_bos_in = fst.input_symbols().find(bos_str)
    i_bos_out = fst.output_symbols().find(bos_str)
    old_start = fst.start()
    new_start = fst.add_state()
    fst.set_start(new_start)
    init_arc = openfst.Arc(i_bos_in, i_bos_out, one, old_start)
    fst.add_arc(new_start, init_arc)

    # add superfinal state accepting EOS
    i_eos_in = fst.input_symbols().find(eos_str)
    i_eos_out = fst.output_symbols().find(eos_str)
    new_final = fst.add_state()
    for state in fst.states():
        w_final = fst.final(state)
        if w_final != zero:
            fst.set_final(state, zero)
            final_arc = openfst.Arc(i_eos_in, i_eos_out, w_final, new_final)
            fst.add_arc(state, final_arc)
    fst.set_final(new_final, one)

    return fst


def durationFst_BACKUP(
        label, num_states, final_weights=None, arc_type='standard',
        symbol_table=None):
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


def durationFst(
        label_str, seg_internal_str, seg_final_str, max_dur,
        final_weights=None, input_symbols=None, output_symbols=None,
        arc_type='standard'):
    """ Construct a left-to-right WFST from an input sequence.

    Parameters
    ----------
    input_seq : iterable(int or string)

    Returns
    -------
    fst : openfst.Fst
    """

    input_label = input_symbols.find(label_str)
    output_label_int = output_symbols.find(seg_internal_str)
    output_label_ext = output_symbols.find(seg_final_str)

    if max_dur < 1:
        raise AssertionError(f"max_dur = {max_dur}, but should be >= 1)")

    fst = openfst.VectorFst(arc_type=arc_type)
    one = openfst.Weight.one(fst.weight_type())
    zero = openfst.Weight.zero(fst.weight_type())

    if final_weights is None:
        final_weights = [one for __ in range(max_dur)]

    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    states = tuple(fst.add_state() for __ in range(max_dur))
    final_state = fst.add_state()
    fst.set_start(states[0])
    fst.set_final(final_state, one)

    for i, cur_state in enumerate(states):
        cur_state = states[i]

        final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
        if final_weight != zero:
            arc = openfst.Arc(input_label, output_label_ext, one, final_state)
            fst.add_arc(cur_state, arc)

        if i + 1 < len(states):
            next_state = states[i + 1]
            arc = openfst.Arc(input_label, output_label_int, one, next_state)
            fst.add_arc(cur_state, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


def make_duration_fst(
        final_weights, class_vocab, class_seg_to_str,
        seg_internal_str='I', seg_final_str='F',
        input_symbols=None, output_symbols=None,
        arc_type='standard'):
    num_classes, num_states = final_weights.shape

    dur_fsts = [
        durationFst(
            class_vocab[i],
            class_seg_to_str[class_vocab[i], seg_internal_str],
            class_seg_to_str[class_vocab[i], seg_final_str],
            num_states, final_weights=final_weights[i],
            arc_type=arc_type,
            input_symbols=input_symbols,
            output_symbols=output_symbols
        )
        for i in range(num_classes)
    ]

    empty = openfst.VectorFst(arc_type=arc_type)
    empty.set_input_symbols(input_symbols)
    empty.set_output_symbols(output_symbols)
    fst = empty.union(*dur_fsts).closure(closure_plus=True)
    return fst


def single_state_transducer(
        transition_weights, row_vocab, col_vocab,
        input_symbols=None, output_symbols=None,
        arc_type='standard'):

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    state = fst.add_state()
    fst.set_start(state)
    fst.set_final(state, one)

    for i_input, row in enumerate(transition_weights):
        for i_output, tx_weight in enumerate(row):
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            input_id = fst.input_symbols().find(row_vocab[i_input])
            output_id = fst.output_symbols().find(col_vocab[i_output])
            if weight != zero:
                arc = openfst.Arc(input_id, output_id, weight, state)
                fst.add_arc(state, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


def single_seg_transducer(
        weights, input_vocab, output_vocab,
        input_seg_to_str, output_seg_to_str,
        input_symbols=None, output_symbols=None,
        eps_str='ε', bos_str='<BOS>', eos_str='<EOS>',
        seg_internal_str='I', seg_final_str='F',
        arc_type='standard'):

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    state = fst.add_state()
    fst.set_start(state)
    fst.set_final(state, one)

    def make_state(i_input, i_output, weight):
        io_state = fst.add_state()

        # CASE 1: (in, I) : (out, I), weight one, transition into io state
        input_str = input_seg_to_str[input_vocab[i_input], seg_internal_str]
        output_str = output_seg_to_str[output_vocab[i_output], seg_internal_str]
        arc = openfst.Arc(
            fst.input_symbols().find(input_str),
            fst.output_symbols().find(output_str),
            one,
            io_state
        )
        fst.add_arc(state, arc)
        fst.add_arc(io_state, arc.copy())

        # CASE 2: (in, F) : (out, F), weight tx_weight
        input_str = input_seg_to_str[input_vocab[i_input], seg_final_str]
        output_str = output_seg_to_str[output_vocab[i_output], seg_final_str]
        arc = openfst.Arc(
            fst.input_symbols().find(input_str),
            fst.output_symbols().find(output_str),
            weight,
            state
        )
        fst.add_arc(io_state, arc)

    for i_input, row in enumerate(weights):
        for i_output, tx_weight in enumerate(row):
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            if weight != zero:
                make_state(i_input, i_output, weight)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


def toArray(lattice, row_integerizer, col_integerizer):
    zero = openfst.Weight.zero(lattice.weight_type())
    # one = openfst.Weight.one(lattice.weight_type())

    semiring_weights = {}
    for state in lattice.states():
        for arc in lattice.arcs(state):
            io_key = (
                lattice.input_symbols().find(arc.ilabel),
                lattice.output_symbols().find(arc.olabel)
            )
            cur_weight = semiring_weights.get(io_key, zero)
            semiring_weights[io_key] = openfst.plus(cur_weight, arc.weight)

    num_inputs = len(row_integerizer)
    num_outputs = len(col_integerizer)
    weights = np.full((num_inputs, num_outputs), float(zero))
    for (i_name, o_name), weight in semiring_weights.items():
        row = row_integerizer.get(i_name, None)
        col = col_integerizer.get(o_name, None)
        if row is None or col is None:
            continue
        weights[row, col] = float(weight)

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


def fromTransitions(
        transition_weights, row_vocab, col_vocab,
        init_weights=None, final_weights=None,
        input_symbols=None, output_symbols=None,
        bos_str='<BOS>', eos_str='<EOS>', eps_str=libfst.EPSILON_STRING,
        arc_type='standard', transition_ids=None):
    """ Instantiate a state machine from state transitions.

    Parameters
    ----------

    Returns
    -------
    """

    num_states = transition_weights.shape[0]

    if transition_ids is None:
        transition_ids = {}
        for s_cur in range(num_states):
            for s_next in range(num_states):
                transition_ids[(s_cur, s_next)] = len(transition_ids)
        for s in range(num_states):
            transition_ids[(-1, s)] = len(transition_ids)

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    if init_weights is None:
        init_weights = tuple(float(one) for __ in range(num_states))

    if final_weights is None:
        final_weights = tuple(float(one) for __ in range(num_states))

    fst.set_start(fst.add_state())
    final_state = fst.add_state()
    fst.set_final(final_state, one)

    bos_index = fst.input_symbols().find(bos_str)
    eos_index = fst.input_symbols().find(eos_str)
    eps_index = fst.output_symbols().find(eps_str)

    def makeState(i):
        state = fst.add_state()

        initial_weight = openfst.Weight(fst.weight_type(), init_weights[i])
        if initial_weight != zero:
            next_state_str = col_vocab[i]
            next_state_index = fst.output_symbols().find(next_state_str)
            arc = openfst.Arc(bos_index, next_state_index, initial_weight, state)
            fst.add_arc(fst.start(), arc)

        final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
        if final_weight != zero:
            arc = openfst.Arc(eos_index, eps_index, final_weight, final_state)
            fst.add_arc(state, arc)

        return state

    states = tuple(makeState(i) for i in range(num_states))
    for i_cur, row in enumerate(transition_weights):
        for i_next, tx_weight in enumerate(row):
            cur_state = states[i_cur]
            next_state = states[i_next]
            weight = openfst.Weight(fst.weight_type(), tx_weight)
            if weight != zero:
                next_state_str = col_vocab[i_next]
                next_state_index = fst.output_symbols().find(next_state_str)
                arc = openfst.Arc(next_state_index, next_state_index, weight, next_state)
                fst.add_arc(cur_state, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


def fromArray(
        weights, row_vocab, col_vocab,
        final_weight=None, arc_type=None,
        input_symbols=None, output_symbols=None):
    """ Instantiate a state machine from an array of weights.

    Parameters
    ----------
    weights : array_like, shape (num_inputs, num_outputs)
        Needs to implement `.shape`, so it should be a numpy array or a torch
        tensor.
    final_weight : arc_types.AbstractSemiringWeight, optional
        Should have the same type as `arc_type`. Default is `arc_type.zero`
    arc_type : {'standard', 'log'}, optional
        Default is 'standard' (ie the tropical arc_type)
    input_labels :
    output_labels :

    Returns
    -------
    fst : fsm.FST
        The transducer's arcs have input labels corresponding to the state
        they left, and output labels corresponding to the state they entered.
    """

    if weights.ndim != 2:
        raise AssertionError(f"weights have unrecognized shape {weights.shape}")

    if arc_type is None:
        arc_type = 'standard'

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    if final_weight is None:
        final_weight = one
    else:
        final_weight = openfst.Weight(fst.weight_type(), final_weight)

    init_state = fst.add_state()
    fst.set_start(init_state)

    prev_state = init_state
    for sample_index, row in enumerate(weights):
        cur_state = fst.add_state()
        for i, weight in enumerate(row):
            input_label = row_vocab[sample_index]
            output_label = col_vocab[i]
            input_label_index = fst.input_symbols().find(input_label)
            output_label_index = fst.output_symbols().find(output_label)
            weight = openfst.Weight(fst.weight_type(), weight)
            if weight != zero:
                arc = openfst.Arc(
                    input_label_index, output_label_index,
                    weight, cur_state
                )
                fst.add_arc(prev_state, arc)
        prev_state = cur_state
    fst.set_final(cur_state, final_weight)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


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


def to_strings(vocab):
    return tuple(map(str, vocab))


def to_integerizer(vocab):
    return {item: i for i, item in enumerate(vocab)}


class Vocabulary(object):
    def __init__(self, vocab, aux_symbols=('ε',)):
        # Object and string representations of the input vocabulary
        self.vocab = tuple(vocab)
        self.str_vocab = tuple(map(str, vocab))

        # OpenFST symbol tables
        self.aux_symbols = tuple(aux_symbols)
        self.symbol_table = libfst.makeSymbolTable(
            self.aux_symbols + self.str_vocab,
            prepend_epsilon=False
        )

        # Map vocab items to their integer values
        self.integerizer = {item: i for i, item in enumerate(self.vocab)}
        self.str_integerizer = {item: i for i, item in enumerate(self.str_vocab)}

    def to_str(self, obj):
        i = self.integerizer[obj]
        return self.str_vocab[i]

    def to_obj(self, string):
        i = self.str_integerizer[string]
        return self.vocab[i]

    def as_str(self):
        return self.str_vocab

    def as_raw(self):
        return self.vocab


class AttributeClassifier(object):
    def __init__(
            self, event_attrs, connection_attrs, event_vocab,
            part_vocab, part_categories, part_connections,
            event_duration_weights=None,
            eps_str='ε', bos_str='<BOS>', eos_str='<EOS>',
            seg_internal_str='I', seg_final_str='F'):
        """
        Parameters
        ----------
        event_attrs : pd.DataFrame
        connection_attrs : pd.DataFrame
        vocab : list( string )
        """

        self.eps_str = eps_str
        self.bos_str = bos_str
        self.eos_str = eos_str
        self.seg_internal_str = seg_internal_str
        self.seg_final_str = seg_final_str

        self.event_attrs = event_attrs
        self.connection_attrs = connection_attrs

        self.part_categories = part_categories
        self.part_connections = part_connections

        # VOCABULARIES: OBJECT VERSIONS
        self.aux_symbols = (eps_str, bos_str, eos_str)
        seg_vocab = (seg_internal_str, seg_final_str)
        action_vocab = connection_attrs['action'].to_list()
        transition_vocab = tuple(
            tuple(int(x) for x in col_name.split('->'))
            for col_name in connection_attrs.columns if col_name != 'action'
        )
        edge_vocab = tuple(
            frozenset([
                frozenset([part_1, part_2])
                for part_1, neighbors in self.part_connections.items()
                for part_2 in neighbors
            ])
        )
        connection_vocab = tuple(
            sorted(frozenset().union(*[frozenset(t) for t in transition_vocab]))
        )

        event_action_weights = events_to_actions(
            self.event_attrs,
            action_vocab, event_vocab, edge_vocab,
            self.part_categories
        )

        # FIXME: ONLY USED TO DEBUG
        def make_dummy(vocab, prefix):
            return tuple(f"{prefix}{i}" for i, _ in enumerate(vocab))
        event_vocab = make_dummy(event_vocab, 'e')
        action_vocab = make_dummy(action_vocab, 'a')

        def make_vocab(vocab):
            return Vocabulary(vocab, aux_symbols=self.aux_symbols)
        self.seg_vocab = make_vocab(seg_vocab)
        self.event_vocab = make_vocab(event_vocab)
        self.action_vocab = make_vocab(action_vocab)
        self.part_vocab = make_vocab(part_vocab)
        self.transition_vocab = make_vocab(transition_vocab)
        self.edge_vocab = make_vocab(edge_vocab)
        self.connection_vocab = make_vocab(connection_vocab)

        def convert_vocab_to_seg(vocab):
            return tuple(
                (x, s)
                for x in vocab.as_raw()
                for s in self.seg_vocab.as_raw()
            )
        self.event_seg_vocab = make_vocab(convert_vocab_to_seg(self.event_vocab))
        self.action_seg_vocab = make_vocab(convert_vocab_to_seg(self.action_vocab))
        self.connection_seg_vocab = make_vocab(convert_vocab_to_seg(self.connection_vocab))

        def get_parts(class_seg_key, class_vocab, class_seg_vocab):
            c, s = class_seg_vocab.to_obj(class_seg_key)
            c_str = class_vocab.to_str(c)
            s_str = self.seg_vocab.to_str(s)
            return c_str, s_str
        self.event_seg_to_str = {
            get_parts(name, self.event_vocab, self.event_seg_vocab): name
            for name in self.event_seg_vocab.as_str()
        }
        self.action_seg_to_str = {
            get_parts(name, self.action_vocab, self.action_seg_vocab): name
            for name in self.action_seg_vocab.as_str()
        }
        self.connection_seg_to_str = {
            get_parts(name, self.connection_vocab, self.connection_seg_vocab): name
            for name in self.connection_seg_vocab.as_str()
        }

        self.event_duration_fst = add_endpoints(
            make_duration_fst(
                -np.log(event_duration_weights),
                self.event_vocab.as_str(), self.event_seg_to_str,
                seg_internal_str=seg_internal_str, seg_final_str=seg_final_str,
                input_symbols=self.event_vocab.symbol_table,
                output_symbols=self.event_seg_vocab.symbol_table,
                arc_type='standard',
            ),
            bos_str=bos_str, eos_str=eos_str
        ).arcsort(sort_type='ilabel')

        self.action_to_connection = transducer_from_connection_attrs(
            -np.log(self.connection_attrs.drop(['action'], axis=1).to_numpy()),
            self.transition_vocab.as_raw(), len(self.connection_vocab.as_raw()),
            self.action_vocab.as_str(), self.connection_vocab.as_str(),
            self.action_seg_to_str, self.connection_seg_to_str,
            init_weights=-np.log(np.array([1, 0])),
            input_table=self.action_seg_vocab.symbol_table,
            output_table=self.connection_seg_vocab.symbol_table,
            arc_type='standard'
        ).arcsort(sort_type='ilabel')

        connection_seg_to_connection_scores = [
            [0 if c == c_other else np.inf for c_other in self.connection_vocab.as_raw()]
            for c, s in self.connection_seg_vocab.as_raw()
        ]
        self.connection_seg_to_connection = single_state_transducer(
            np.array(connection_seg_to_connection_scores, dtype=float),
            self.connection_seg_vocab.as_str(), self.connection_vocab.as_str(),
            input_symbols=self.connection_seg_vocab.symbol_table,
            output_symbols=self.connection_vocab.symbol_table,
            arc_type='standard'
        ).arcsort(sort_type='ilabel')

        self.event_to_action = []
        self.seq_models = []
        for i, edge_event_action_weights in enumerate(event_action_weights):
            event_to_action = add_endpoints(
                single_seg_transducer(
                    -np.log(edge_event_action_weights),
                    self.event_vocab.as_str(), self.action_vocab.as_str(),
                    self.event_seg_to_str, self.action_seg_to_str,
                    input_symbols=self.event_seg_vocab.symbol_table,
                    output_symbols=self.action_seg_vocab.symbol_table,
                    arc_type='standard'
                )
            ).arcsort(sort_type='ilabel')

            seq_model = libfst.easyCompose(
                *[self.event_duration_fst, event_to_action, self.action_to_connection],
                determinize=False,
                minimize=False
            ).arcsort(sort_type='ilabel')

            self.event_to_action.append(event_to_action)
            self.seq_models.append(seq_model)

    def viz_params(self, fig_dir):
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        draw_fst(
            os.path.join(fig_dir, 'event-duration'),
            self.event_duration_fst,
            vertical=True, width=50, height=50, portrait=True
        )

        for i, fst in enumerate(self.event_to_action):
            draw_fst(
                os.path.join(fig_dir, f'event-to-action_edge={i}'),
                fst,
                vertical=True, width=50, height=50, portrait=True
            )

        draw_fst(
            os.path.join(fig_dir, 'action-to-connection'),
            self.action_to_connection,
            vertical=True, width=50, height=50, portrait=True
        )

        for i, fst in enumerate(self.seq_models):
            draw_fst(
                os.path.join(fig_dir, f'seq-model_edge={i}'),
                fst,
                vertical=True, width=50, height=50, portrait=True
            )

    @property
    def num_events(self):
        return len(self.event_vocab.as_raw())

    @property
    def num_parts(self):
        return len(self.part_vocab.as_raw())

    @property
    def num_actions(self):
        return len(self.action_vocab.as_raw())

    @property
    def num_transitions(self):
        return len(self.transition_vocab.as_raw())

    @property
    def num_edges(self):
        return len(self.edge_vocab.as_raw())

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
        sample_vocab = Vocabulary(
            tuple(f"{i}" for i in range(event_scores.shape[0])),
            aux_symbols=self.aux_symbols
        )

        lattice = add_endpoints(
            fromArray(
                -event_scores,
                sample_vocab.as_str(), self.event_vocab.as_str(),
                input_symbols=sample_vocab.symbol_table,
                output_symbols=self.event_vocab.symbol_table,
                arc_type='standard'
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        def getScores(seq_model):
            # Compose event scores with event --> connection map
            # Compute edge posterior marginals
            decode_lattice = openfst.compose(lattice, seq_model)
            arc_score_lattice = libfst.fstArcGradient(decode_lattice)
            state_score_lattice = openfst.compose(
                arc_score_lattice,
                openfst.arcmap(self.connection_seg_to_connection, map_type='to_log')
            )

            connection_scores, weight_type = toArray(
                state_score_lattice,
                sample_vocab.str_integerizer,
                self.connection_vocab.str_integerizer
                # self.connection_seg_vocab.str_integerizer
            )
            return connection_scores

        connection_neglogprobs = np.stack(
            tuple(getScores(seq_model) for seq_model in self.seq_models),
            axis=-1
        )

        # lattices = tuple(
        #     openfst.compose(lattice, seq_model)
        #     for seq_model in self.seq_models
        # )
        # return lattices
        return -connection_neglogprobs

    def predict(self, outputs):
        """ Choose the best labels from an array of output activations.

        Parameters
        ----------
        outputs : np.ndarray of float with shape (NUM_SAMPLES, NUM_CLASSES, ...)

        Returns
        -------
        preds : np.ndarray of int with shape (NUM_SAMPLES, ...)
        """

        # preds = tuple(np.array(libfst.viterbi(lattice)) for lattice in outputs),
        # edge_preds = np.stack(preds, axis=-1)
        edge_preds = outputs.argmax(axis=1)
        return edge_preds


def __test(fig_dir, num_classes=2, num_samples=10, min_dur=2, max_dur=4):
    eps_str = 'ε'
    bos_str = '<BOS>'
    eos_str = '<EOS>'
    seg_internal_str = 'I'
    seg_final_str = 'F'

    aux_symbols = (eps_str, bos_str, eos_str)
    seg_vocab = (seg_internal_str, seg_final_str)
    sample_vocab = tuple(i for i in range(num_samples))
    dur_vocab = tuple(i for i in range(1, max_dur + 1))
    class_vocab = tuple(i for i in range(num_classes))
    class_seg_vocab = tuple((c, s) for c in class_vocab for s in seg_vocab)

    def to_strings(vocab):
        return tuple(map(str, vocab))

    def to_integerizer(vocab):
        return {item: i for i, item in enumerate(vocab)}

    sample_vocab_str = to_strings(sample_vocab)
    seg_vocab_str = to_strings(seg_vocab)
    class_vocab_str = to_strings(class_vocab)
    class_seg_vocab_str = to_strings(class_seg_vocab)

    # sample_integerizer = to_integerizer(sample_vocab)
    seg_integerizer = to_integerizer(seg_vocab)
    class_integerizer = to_integerizer(class_vocab)
    # class_seg_integerizer = to_integerizer(class_seg_vocab)

    sample_str_integerizer = to_integerizer(sample_vocab_str)
    class_str_integerizer = to_integerizer(class_vocab_str)
    class_seg_str_integerizer = to_integerizer(class_seg_vocab_str)

    def get_parts(class_seg_key):
        c, s = class_seg_vocab[class_seg_str_integerizer[class_seg_key]]
        c_str = class_vocab_str[class_integerizer[c]]
        s_str = seg_vocab_str[seg_integerizer[s]]
        return c_str, s_str

    class_seg_to_str = {get_parts(name): name for name in class_seg_vocab_str}

    sample_symbols = libfst.makeSymbolTable(aux_symbols + sample_vocab, prepend_epsilon=False)
    # dur_symbols = libfst.makeSymbolTable(aux_symbols + dur_vocab, prepend_epsilon=False)
    class_symbols = libfst.makeSymbolTable(aux_symbols + class_vocab, prepend_epsilon=False)
    # seg_symbols = libfst.makeSymbolTable(aux_symbols + seg_vocab, prepend_epsilon=False)
    class_seg_symbols = libfst.makeSymbolTable(aux_symbols + class_seg_vocab, prepend_epsilon=False)

    obs_scores = np.zeros((num_samples, num_classes))

    dur_scores = np.array(
        [[0 if d >= min_dur else np.inf for d in dur_vocab] for c in class_vocab],
        dtype=float
    )

    def score_transition(c_prev, s_prev, c_cur, s_cur):
        if c_prev != c_cur:
            if s_prev == seg_final_str and s_cur == seg_internal_str:
                score = 0
            else:
                score = np.inf
        else:
            if s_prev == seg_internal_str and s_cur == seg_final_str:
                score = 0
            elif s_prev == seg_internal_str and s_cur == seg_internal_str:
                score = 0
            else:
                score = np.inf
        return score

    transition_scores = np.array(
        [
            [
                score_transition(c_prev, s_prev, c_cur, s_cur)
                for (c_cur, s_cur) in class_seg_vocab
            ]
            for (c_prev, s_prev) in class_seg_vocab
        ],
        dtype=float
    )
    init_scores = np.array(
        [0 if c == 0 and s == seg_internal_str else np.inf for (c, s) in class_seg_vocab],
        dtype=float
    )
    final_scores = np.array(
        [0 if c == 1 and s == seg_final_str else np.inf for (c, s) in class_seg_vocab],
        dtype=float
    )

    def score_arc_state(class_seg_key, class_key):
        c, s = class_seg_vocab[class_seg_str_integerizer[class_seg_key]]
        c_prime = class_str_integerizer[class_key]

        if c == c_prime:
            score = 0
        else:
            score = np.inf

        return score

    class_seg_to_class_scores = np.array(
        [
            [
                score_arc_state(class_seg_key, class_key)
                for class_key in class_vocab_str
            ]
            for class_seg_key in class_seg_vocab_str
        ],
        dtype=float
    )

    def log_normalize(arr, axis=1):
        denom = -scipy.special.logsumexp(-arr, axis=axis, keepdims=True)
        return arr - denom

    obs_scores = log_normalize(obs_scores)
    dur_scores = log_normalize(dur_scores)
    transition_scores = log_normalize(transition_scores)
    init_scores = log_normalize(init_scores, axis=None)
    final_scores = log_normalize(final_scores, axis=None)

    obs_fst = add_endpoints(
        fromArray(
            obs_scores, sample_vocab_str, class_vocab_str,
            input_symbols=sample_symbols,
            output_symbols=class_symbols,
            arc_type='standard'
        ),
        bos_str=bos_str, eos_str=eos_str
    ).arcsort(sort_type='ilabel')

    dur_fst = add_endpoints(
        make_duration_fst(
            dur_scores, class_vocab_str, class_seg_to_str,
            seg_internal_str=seg_internal_str, seg_final_str=seg_final_str,
            input_symbols=class_symbols,
            output_symbols=class_seg_symbols,
            arc_type='standard',
        ),
        bos_str=bos_str, eos_str=eos_str
    ).arcsort(sort_type='ilabel')

    transition_fst = fromTransitions(
        transition_scores, class_seg_vocab_str, class_seg_vocab_str,
        init_weights=init_scores,
        final_weights=final_scores,
        input_symbols=class_seg_symbols,
        output_symbols=class_seg_symbols
    ).arcsort(sort_type='ilabel')

    class_seg_to_class_fst = single_state_transducer(
        class_seg_to_class_scores, class_seg_vocab_str, class_vocab_str,
        input_symbols=class_seg_symbols, output_symbols=class_symbols,
        arc_type='standard'
    ).arcsort(sort_type='ilabel')

    seq_model = openfst.compose(dur_fst, transition_fst)
    decode_lattice = openfst.compose(obs_fst, seq_model).rmepsilon()

    # Result is in the log semiring (ie weights are negative log probs)
    arc_scores = libfst.fstArcGradient(decode_lattice).arcsort(sort_type='ilabel')
    best_arcs = openfst.shortestpath(decode_lattice).arcsort(sort_type='ilabel')

    state_scores = openfst.compose(
        arc_scores,
        openfst.arcmap(class_seg_to_class_fst, map_type='to_log')
    )
    best_states = openfst.compose(best_arcs, class_seg_to_class_fst)

    state_scores_arr, weight_type = toArray(
        state_scores,
        sample_str_integerizer,
        class_str_integerizer
    )

    draw_fst(
        os.path.join(fig_dir, 'obs_fst'), obs_fst,
        vertical=True, width=50, height=50, portrait=True
    )

    draw_fst(
        os.path.join(fig_dir, 'dur_fst'), dur_fst,
        vertical=True, width=50, height=50, portrait=True
    )

    draw_fst(
        os.path.join(fig_dir, 'transition_fst'), transition_fst,
        vertical=True, width=50, height=50, portrait=True
    )

    draw_fst(
        os.path.join(fig_dir, 'seq_model'), seq_model,
        vertical=True, width=50, height=50, portrait=True
    )

    draw_fst(
        os.path.join(fig_dir, 'decode_lattice'), decode_lattice,
        vertical=True, width=50, height=50, portrait=True
    )

    draw_fst(
        os.path.join(fig_dir, 'state_scores'), state_scores,
        vertical=True, width=50, height=50, portrait=True
    )

    draw_fst(
        os.path.join(fig_dir, 'best_states'), best_states,
        vertical=True, width=50, height=50, portrait=True
    )

    utils.plot_array(
        obs_scores.T, (state_scores_arr.T,), ('-logprobs',),
        fn=os.path.join(fig_dir, "test_io.png")
    )


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

    # __test(fig_dir)
    # return

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

        cv_str = f'cvfold={cv_index}'

        event_duration_weights = np.ones((event_attrs.shape[0], 10), dtype=float)
        model = AttributeClassifier(
            event_attrs, connection_attrs, event_vocab,
            part_vocab, part_categories, part_connections,
            event_duration_weights=event_duration_weights
        )

        model.viz_params(os.path.join(fig_dir, f'{cv_str}_model-params'))

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
            pred_connection_seq = model.predict(connection_score_seq)

            # metric_dict = eval_metrics(pred_seq, true_seq)
            # for name, value in metric_dict.items():
            #     logger.info(f"    {name}: {value * 100:.2f}%")

            # utils.saveVariable(connection_score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_connection_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            # utils.saveVariable(true_event_seq, f'{seq_id_str}_true-label-seq', out_data_dir)
            # utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            if plot_io:
                pred_event_seq = event_score_seq.argmax(axis=1)
                for i in range(connection_score_seq.shape[-1]):
                    utils.plot_array(
                        connection_score_seq[..., i].T,
                        (pred_connection_seq[..., i], pred_event_seq),
                        ('pred_connection', 'pred_event'),
                        fn=os.path.join(fig_dir, f"seq={seq_id:03d}_edge={i:02d}.png")
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
