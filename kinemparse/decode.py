import os
import logging

import numpy as np
import scipy
import graphviz as gv

import pywrapfst as openfst

from mathtools import utils  # , metrics
from seqtools import fstutils_openfst as libfst


logger = logging.getLogger(__name__)


# -=( VOCABULARY )==-----------------------------------------------------------
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


def to_strings(vocab):
    return tuple(map(str, vocab))


def to_integerizer(vocab):
    return {item: i for i, item in enumerate(vocab)}


# -=( FST MODELS FOR ACTION -> ASSEMBLY PARSING )==----------------------------
class AssemblyActionRecognizer(object):
    def __init__(self, scores, vocabs, params):
        """
        Parameters
        ----------
        event_attrs : pd.DataFrame
        connection_attrs : pd.DataFrame
        vocab : list( string )
        """

        self.vocabs = vocabs
        self.scores = scores
        self.params = params

        self._init_vocabs(**vocabs)
        self._init_models(*scores, **params)

    def make_vocab(self, vocab):
        return Vocabulary(vocab, aux_symbols=self.aux_symbols)

    def product_vocab(self, *vocabs):
        product = ((tuple(),))
        for vocab in vocabs:
            product = tuple((*prod_item, item) for item in vocab.as_raw() for prod_item in product)
        return self.make_vocab(product)

    def map_parts_to_str(self, product_vocab, *part_vocabs):
        def get_parts(class_dur_key):
            parts = product_vocab.to_obj(class_dur_key)
            parts_as_str = tuple(part_vocabs[i].to_str(part) for i, part in enumerate(parts))
            return parts_as_str

        parts_to_str = {
            get_parts(name): name
            for name in product_vocab.as_str()
        }

        return parts_to_str

    def _init_vocabs(
            self, event_vocab=None, action_vocab=None, part_vocab=None,
            joint_vocab=None, connection_vocab=None, assembly_vocab=None,
            eps_str='ε', bos_str='<BOS>', eos_str='<EOS>',
            dur_internal_str='I', dur_final_str='F'):

        # VOCABULARIES: OBJECT VERSIONS
        self.eps_str = eps_str
        self.bos_str = bos_str
        self.eos_str = eos_str
        self.dur_internal_str = dur_internal_str
        self.dur_final_str = dur_final_str
        self.aux_symbols = (eps_str, bos_str, eos_str)
        dur_vocab = (dur_internal_str, dur_final_str)

        self.vocabs['aux_symbols'] = self.aux_symbols
        self.vocabs['dur_vocab'] = dur_vocab

        self.dur_vocab = self.make_vocab(dur_vocab)
        self.event_vocab = self.make_vocab(event_vocab)
        # self.action_vocab = self.make_vocab(action_vocab)
        # self.part_vocab = self.make_vocab(part_vocab)
        # self.joint_vocab = self.make_vocab(joint_vocab)
        # self.connection_vocab = self.make_vocab(connection_vocab)
        self.assembly_vocab = self.make_vocab(assembly_vocab)

    def _init_models(
            self, *stage_scores,
            arc_type='log', decode_type='marginal', return_label='output',
            reduce_order='pre', output_stage=3):

        self.reduce_order = reduce_order
        self.return_label = return_label
        self.arc_type = arc_type
        self.output_stage = output_stage
        self.decode_type = decode_type

        model_builders = [
            self.make_event_dur_model,
            self.make_event_to_assembly_model,
            self.make_assembly_transition_model
        ]
        self.submodels, self.submodel_outputs, self.reducers = zip(
            *[builder(-scores) for builder, scores in zip(model_builders, stage_scores)]
        )

        self.seq_model = libfst.easyCompose(
            *self.submodels[:self.output_stage],
            determinize=False, minimize=False
        ).arcsort(sort_type='ilabel')
        if self.reduce_order == 'pre':
            self.seq_model = openfst.compose(
                self.seq_model,
                self.reducers[self.output_stage - 1]
            )

    def make_event_dur_model(self, duration_weights):
        self.event_dur_vocab = self.product_vocab(self.event_vocab, self.dur_vocab)
        self.event_dur_to_str = self.map_parts_to_str(
            self.event_dur_vocab,
            self.event_vocab, self.dur_vocab
        )

        event_duration_fst = add_endpoints(
            make_duration_fst(
                duration_weights,
                self.event_vocab.as_str(), self.event_dur_to_str,
                dur_internal_str=self.dur_internal_str, dur_final_str=self.dur_final_str,
                input_symbols=self.event_vocab.symbol_table,
                output_symbols=self.event_dur_vocab.symbol_table,
                arc_type=self.arc_type,
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        reducer = add_endpoints(
            make_reduce_fst(
                self.event_dur_vocab, self.event_vocab,
                keep_dim=0,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')
        return event_duration_fst, self.event_vocab, reducer

    def make_event_to_assembly_model(self, event_to_assembly_weights):
        self.event_assembly_tx_dur_vocab = self.product_vocab(
            self.event_vocab, self.assembly_vocab, self.assembly_vocab, self.dur_vocab
        )
        self.event_assembly_tx_dur_to_str = self.map_parts_to_str(
            self.event_assembly_tx_dur_vocab,
            self.event_vocab, self.assembly_vocab, self.assembly_vocab, self.dur_vocab
        )

        event_to_assembly_fst = make_event_to_assembly_fst(
            event_to_assembly_weights,
            self.event_vocab.as_str(), self.assembly_vocab.as_str(),
            self.event_dur_to_str, self.event_assembly_tx_dur_to_str,
            input_symbols=self.event_dur_vocab.symbol_table,
            output_symbols=self.event_assembly_tx_dur_vocab.symbol_table,
            eps_str=self.eps_str,
            dur_internal_str=self.dur_internal_str, dur_final_str=self.dur_final_str,
            bos_str=self.bos_str, eos_str=self.eos_str,
            arc_type=self.arc_type
        ).arcsort(sort_type='ilabel')

        if self.return_label == 'output':
            keep_dim = 1
            output_vocab = self.assembly_vocab
        elif self.return_label == 'input':
            keep_dim = 0
            output_vocab = self.event_vocab
        else:
            err_str = f"Unrecognized value for self.return_label: {self.return_label}"
            raise AssertionError(err_str)
        reducer = add_endpoints(
            make_reduce_fst(
                self.event_assembly_tx_dur_vocab, output_vocab,
                keep_dim=keep_dim,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        return event_to_assembly_fst, output_vocab, reducer

    def make_assembly_transition_model(self, transition_weights):
        init_weights = transition_weights[0, :-1]
        final_weights = transition_weights[1:, -1]
        transition_weights = transition_weights[1:, :-1]

        transition_weights = np.broadcast_to(
            transition_weights,
            (len(self.event_vocab.as_raw()), *transition_weights.shape)
        )
        init_weights = np.broadcast_to(
            init_weights,
            (len(self.event_vocab.as_raw()), *init_weights.shape)
        )
        final_weights = np.broadcast_to(
            final_weights,
            (len(self.event_vocab.as_raw()), *final_weights.shape)
        )

        assembly_transition_fst = make_event_to_assembly_fst(
            transition_weights,
            self.event_vocab.as_str(), self.assembly_vocab.as_str(),
            self.event_assembly_tx_dur_to_str, self.event_assembly_tx_dur_to_str,
            init_weights=init_weights, final_weights=final_weights,
            input_symbols=self.event_assembly_tx_dur_vocab.symbol_table,
            output_symbols=self.event_assembly_tx_dur_vocab.symbol_table,
            eps_str=self.eps_str,
            dur_internal_str=self.dur_internal_str, dur_final_str=self.dur_final_str,
            bos_str=self.bos_str, eos_str=self.eos_str,
            state_tx_in_input=True,
            arc_type=self.arc_type
        ).arcsort(sort_type='ilabel')

        if self.return_label == 'output':
            keep_dim = 1
            output_vocab = self.assembly_vocab
        elif self.return_label == 'input':
            keep_dim = 0
            output_vocab = self.event_vocab
        else:
            err_str = f"Unrecognized value for self.return_label: {self.return_label}"
            raise AssertionError(err_str)
        reducer = add_endpoints(
            make_reduce_fst(
                self.event_assembly_tx_dur_vocab, output_vocab,
                keep_dim=keep_dim,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        return assembly_transition_fst, output_vocab, reducer

    @property
    def input_vocab(self):
        return self.event_vocab

    @property
    def output_vocab(self):
        if self.return_label == 'input':
            return self.input_vocab

        return self.submodel_outputs[self.output_stage - 1]

    def save_vocabs(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for name, vocab in self.vocabs.items():
            utils.saveVariable(vocab, name.replace('_', '-'), out_dir)

    def draw_fsts(self, fig_dir):
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        for i, fst in enumerate(self.submodels):
            draw_fst(
                os.path.join(fig_dir, f'seq-model_stage={i}'),
                fst,
                vertical=True, width=50, height=50, portrait=True
            )

        draw_fst(
            os.path.join(fig_dir, 'seq-model'),
            self.seq_model,
            vertical=True, width=50, height=50, portrait=True
        )

    def write_fsts(self, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i, fst in enumerate(self.submodels):
            write_fst(os.path.join(out_dir, f'seq-model_stage={i}'), fst)

        write_fst(os.path.join(out_dir, 'seq-model'), self.seq_model)

    def forward(self, input_scores):
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
            tuple(f"{i}" for i in range(input_scores.shape[0])),
            aux_symbols=self.aux_symbols
        )

        input_lattice = add_endpoints(
            fromArray(
                -input_scores,
                sample_vocab.as_str(), self.event_vocab.as_str(),
                input_symbols=sample_vocab.symbol_table,
                output_symbols=self.event_vocab.symbol_table,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        # Compose event scores with event --> connection map
        decode_lattice = openfst.compose(input_lattice, self.seq_model)
        output_lattice = self._decode(decode_lattice)
        output_weights, weight_type = toArray(
            output_lattice,
            sample_vocab.str_integerizer,
            self.output_vocab.str_integerizer
        )

        return -output_weights

    def _decode(self, lattice):
        if self.decode_type == 'marginal':
            fst = libfst.fstArcGradient(lattice)
            if self.arc_type == 'standard':
                fst = openfst.arcmap(fst, map_type='to_std')
        elif self.decode_type == 'joint':
            fst = libfst.viterbi(lattice)
            if self.arc_type == 'log':
                fst = openfst.arcmap(fst, map_type='to_log')
        else:
            err_str = f"Unrecognized value: self.decode_type={self.decode_type}"
            raise AssertionError(err_str)

        if self.reduce_order == 'post':
            fst = openfst.compose(fst, self.reducers[self.output_stage - 1])
        return fst

    def predict(self, outputs):
        """ Choose the best labels from an array of output activations.

        Parameters
        ----------
        outputs : np.ndarray of float with shape (NUM_SAMPLES, NUM_CLASSES, ...)

        Returns
        -------
        preds : np.ndarray of int with shape (NUM_SAMPLES, ...)
        """

        return outputs.argmax(axis=1)


# -=( FST UTILS )==------------------------------------------------------------
def write_fst(fn, fst):
    with open(fn, 'wt') as file_:
        file_.write(str(fst))


def draw_fst(fn, fst, extension='pdf', **draw_kwargs):
    fst.draw(fn, **draw_kwargs)
    gv.render('dot', extension, fn)
    os.remove(fn)


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
        input_parts_to_str, output_parts_to_str,
        input_symbols=None, output_symbols=None,
        eps_str='ε', bos_str='<BOS>', eos_str='<EOS>',
        dur_internal_str='I', dur_final_str='F',
        arc_type='standard', pass_input=False):

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

        state_istr = input_vocab[i_input]
        state_ostr = output_vocab[i_output]

        # CASE 1: (in, I) : (out, I), weight one, transition into io state
        arc_istr = input_parts_to_str[state_istr, dur_internal_str]
        if pass_input:
            arc_ostr = output_parts_to_str[state_istr, state_ostr, dur_internal_str]
        else:
            arc_ostr = output_parts_to_str[state_ostr, dur_internal_str]
        arc = openfst.Arc(
            fst.input_symbols().find(arc_istr),
            fst.output_symbols().find(arc_ostr),
            one,
            io_state
        )
        fst.add_arc(state, arc)
        fst.add_arc(io_state, arc.copy())

        # CASE 2: (in, F) : (out, F), weight tx_weight
        arc_istr = input_parts_to_str[state_istr, dur_final_str]
        if pass_input:
            arc_ostr = output_parts_to_str[state_istr, state_ostr, dur_final_str]
        else:
            arc_ostr = output_parts_to_str[state_ostr, dur_final_str]
        arc = openfst.Arc(
            fst.input_symbols().find(arc_istr),
            fst.output_symbols().find(arc_ostr),
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


def make_reduce_fst(input_vocab, output_vocab, keep_dim=0, arc_type='standard'):
    reduce_weights = [
        [0 if tup[keep_dim] == x else np.inf for x in output_vocab.as_raw()]
        for tup in input_vocab.as_raw()
    ]

    reduce_fst = single_state_transducer(
        np.array(reduce_weights, dtype=float),
        input_vocab.as_str(), output_vocab.as_str(),
        input_symbols=input_vocab.symbol_table,
        output_symbols=output_vocab.symbol_table,
        arc_type=arc_type
    ).arcsort(sort_type='ilabel')

    return reduce_fst


def make_duration_fst(
        final_weights, class_vocab, class_dur_to_str,
        dur_internal_str='I', dur_final_str='F',
        input_symbols=None, output_symbols=None,
        arc_type='standard'):
    num_classes, num_states = final_weights.shape

    def durationFst(
            label_str, dur_internal_str, dur_final_str, final_weights,
            input_symbols=None, output_symbols=None,
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
        output_label_int = output_symbols.find(dur_internal_str)
        output_label_ext = output_symbols.find(dur_final_str)

        fst = openfst.VectorFst(arc_type=arc_type)
        one = openfst.Weight.one(fst.weight_type())
        zero = openfst.Weight.zero(fst.weight_type())

        max_dur = np.nonzero(final_weights != float(zero))[0].max()
        if max_dur < 1:
            raise AssertionError(f"max_dur = {max_dur}, but should be >= 1)")

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

    dur_fsts = [
        durationFst(
            class_vocab[i],
            class_dur_to_str[class_vocab[i], dur_internal_str],
            class_dur_to_str[class_vocab[i], dur_final_str],
            final_weights[i],
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


def make_event_to_assembly_fst(
        weights,
        input_vocab, output_vocab,
        input_parts_to_str, output_parts_to_str,
        init_weights=None, final_weights=None,
        input_symbols=None, output_symbols=None,
        state_tx_in_input=False,
        eps_str='ε', dur_internal_str='I', dur_final_str='F',
        bos_str='<BOS>', eos_str='<EOS>',
        arc_type='standard'):

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    zero = openfst.Weight.zero(fst.weight_type())
    one = openfst.Weight.one(fst.weight_type())

    init_state = fst.add_state()
    final_state = fst.add_state()
    fst.set_start(init_state)
    fst.set_final(final_state, one)

    def make_states(i_input, i_output):
        seg_internal_state = fst.add_state()
        seg_final_state = fst.add_state()

        state_istr = input_vocab[i_input]
        state_ostr = output_vocab[i_output]

        # Initial state -> seg internal state
        if init_weights is None:
            init_weight = one
        else:
            init_weight = openfst.Weight(fst.weight_type(), init_weights[i_input, i_output])
        if init_weight != zero:
            arc_istr = bos_str
            arc_ostr = bos_str
            arc = openfst.Arc(
                fst.input_symbols().find(arc_istr),
                fst.output_symbols().find(arc_ostr),
                init_weight,
                seg_internal_state
            )
            fst.add_arc(init_state, arc)

        # (in, I) : (out, I), weight one, self transition
        if state_tx_in_input:
            arc_istr = input_parts_to_str[state_istr, state_ostr, state_ostr, dur_internal_str]
        else:
            arc_istr = input_parts_to_str[state_istr, dur_internal_str]
        arc_ostr = output_parts_to_str[state_istr, state_ostr, state_ostr, dur_internal_str]
        arc = openfst.Arc(
            fst.input_symbols().find(arc_istr),
            fst.output_symbols().find(arc_ostr),
            one,
            seg_internal_state
        )
        fst.add_arc(seg_internal_state, arc)

        # (in, F) : (out, F), weight one, transition into final state
        if state_tx_in_input:
            arc_istr = input_parts_to_str[state_istr, state_ostr, state_ostr, dur_final_str]
        else:
            arc_istr = input_parts_to_str[state_istr, dur_final_str]
        arc_ostr = output_parts_to_str[state_istr, state_ostr, state_ostr, dur_final_str]
        arc = openfst.Arc(
            fst.input_symbols().find(arc_istr),
            fst.output_symbols().find(arc_ostr),
            one,
            seg_final_state
        )
        fst.add_arc(seg_internal_state, arc)

        # seg final state -> final_state
        if final_weights is None:
            final_weight = one
        else:
            final_weight = openfst.Weight(fst.weight_type(), final_weights[i_input, i_output])
        if final_weight != zero:
            arc_istr = eos_str
            arc_ostr = eos_str
            arc = openfst.Arc(
                fst.input_symbols().find(arc_istr),
                fst.output_symbols().find(arc_ostr),
                final_weight,
                final_state
            )
            fst.add_arc(seg_final_state, arc)

        return [seg_internal_state, seg_final_state]

    # Build segmental backbone
    states = [
        [make_states(i_input, i_output) for i_output, _ in enumerate(output_vocab)]
        for i_input, _ in enumerate(input_vocab)
    ]

    # Add transitions from final (action, assembly) to initial (action, assembly)
    for i_input, arr in enumerate(weights):
        for i_cur, row in enumerate(arr):
            for i_next, tx_weight in enumerate(row):
                weight = openfst.Weight(fst.weight_type(), tx_weight)
                if weight != zero:
                    from_state = states[i_input][i_cur][1]
                    to_state = states[i_input][i_next][0]

                    istr = input_vocab[i_input]
                    cur_ostr = output_vocab[i_cur]
                    next_ostr = output_vocab[i_next]
                    if state_tx_in_input:
                        arc_istr = input_parts_to_str[istr, cur_ostr, next_ostr, dur_internal_str]
                    else:
                        arc_istr = input_parts_to_str[istr, dur_internal_str]
                    arc_ostr = output_parts_to_str[istr, cur_ostr, next_ostr, dur_internal_str]

                    arc = openfst.Arc(
                        fst.input_symbols().find(arc_istr),
                        fst.output_symbols().find(arc_ostr),
                        one,
                        to_state
                    )
                    fst.add_arc(from_state, arc)

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


def __test(fig_dir, num_classes=2, num_samples=10, min_dur=2, max_dur=4):
    eps_str = 'ε'
    bos_str = '<BOS>'
    eos_str = '<EOS>'
    dur_internal_str = 'I'
    dur_final_str = 'F'

    aux_symbols = (eps_str, bos_str, eos_str)
    dur_vocab = (dur_internal_str, dur_final_str)
    sample_vocab = tuple(i for i in range(num_samples))
    dur_vocab = tuple(i for i in range(1, max_dur + 1))
    class_vocab = tuple(i for i in range(num_classes))
    class_dur_vocab = tuple((c, s) for c in class_vocab for s in dur_vocab)

    def to_strings(vocab):
        return tuple(map(str, vocab))

    def to_integerizer(vocab):
        return {item: i for i, item in enumerate(vocab)}

    sample_vocab_str = to_strings(sample_vocab)
    dur_vocab_str = to_strings(dur_vocab)
    class_vocab_str = to_strings(class_vocab)
    class_dur_vocab_str = to_strings(class_dur_vocab)

    # sample_integerizer = to_integerizer(sample_vocab)
    dur_integerizer = to_integerizer(dur_vocab)
    class_integerizer = to_integerizer(class_vocab)
    # class_dur_integerizer = to_integerizer(class_dur_vocab)

    sample_str_integerizer = to_integerizer(sample_vocab_str)
    class_str_integerizer = to_integerizer(class_vocab_str)
    class_dur_str_integerizer = to_integerizer(class_dur_vocab_str)

    def get_parts(class_dur_key):
        c, s = class_dur_vocab[class_dur_str_integerizer[class_dur_key]]
        c_str = class_vocab_str[class_integerizer[c]]
        s_str = dur_vocab_str[dur_integerizer[s]]
        return c_str, s_str

    class_dur_to_str = {get_parts(name): name for name in class_dur_vocab_str}

    sample_symbols = libfst.makeSymbolTable(aux_symbols + sample_vocab, prepend_epsilon=False)
    # dur_symbols = libfst.makeSymbolTable(aux_symbols + dur_vocab, prepend_epsilon=False)
    class_symbols = libfst.makeSymbolTable(aux_symbols + class_vocab, prepend_epsilon=False)
    # dur_symbols = libfst.makeSymbolTable(aux_symbols + dur_vocab, prepend_epsilon=False)
    class_dur_symbols = libfst.makeSymbolTable(aux_symbols + class_dur_vocab, prepend_epsilon=False)

    obs_scores = np.zeros((num_samples, num_classes))

    dur_scores = np.array(
        [[0 if d >= min_dur else np.inf for d in dur_vocab] for c in class_vocab],
        dtype=float
    )

    def score_transition(c_prev, s_prev, c_cur, s_cur):
        if c_prev != c_cur:
            if s_prev == dur_final_str and s_cur == dur_internal_str:
                score = 0
            else:
                score = np.inf
        else:
            if s_prev == dur_internal_str and s_cur == dur_final_str:
                score = 0
            elif s_prev == dur_internal_str and s_cur == dur_internal_str:
                score = 0
            else:
                score = np.inf
        return score

    transition_scores = np.array(
        [
            [
                score_transition(c_prev, s_prev, c_cur, s_cur)
                for (c_cur, s_cur) in class_dur_vocab
            ]
            for (c_prev, s_prev) in class_dur_vocab
        ],
        dtype=float
    )
    init_scores = np.array(
        [0 if c == 0 and s == dur_internal_str else np.inf for (c, s) in class_dur_vocab],
        dtype=float
    )
    final_scores = np.array(
        [0 if c == 1 and s == dur_final_str else np.inf for (c, s) in class_dur_vocab],
        dtype=float
    )

    def score_arc_state(class_dur_key, class_key):
        c, s = class_dur_vocab[class_dur_str_integerizer[class_dur_key]]
        c_prime = class_str_integerizer[class_key]

        if c == c_prime:
            score = 0
        else:
            score = np.inf

        return score

    class_dur_to_class_scores = np.array(
        [
            [
                score_arc_state(class_dur_key, class_key)
                for class_key in class_vocab_str
            ]
            for class_dur_key in class_dur_vocab_str
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
            dur_scores, class_vocab_str, class_dur_to_str,
            dur_internal_str=dur_internal_str, dur_final_str=dur_final_str,
            input_symbols=class_symbols,
            output_symbols=class_dur_symbols,
            arc_type='standard',
        ),
        bos_str=bos_str, eos_str=eos_str
    ).arcsort(sort_type='ilabel')

    transition_fst = fromTransitions(
        transition_scores, class_dur_vocab_str, class_dur_vocab_str,
        init_weights=init_scores,
        final_weights=final_scores,
        input_symbols=class_dur_symbols,
        output_symbols=class_dur_symbols
    ).arcsort(sort_type='ilabel')

    class_dur_to_class_fst = single_state_transducer(
        class_dur_to_class_scores, class_dur_vocab_str, class_vocab_str,
        input_symbols=class_dur_symbols, output_symbols=class_symbols,
        arc_type='standard'
    ).arcsort(sort_type='ilabel')

    seq_model = openfst.compose(dur_fst, transition_fst)
    decode_lattice = openfst.compose(obs_fst, seq_model).rmepsilon()

    # Result is in the log semiring (ie weights are negative log probs)
    arc_scores = libfst.fstArcGradient(decode_lattice).arcsort(sort_type='ilabel')
    best_arcs = openfst.shortestpath(decode_lattice).arcsort(sort_type='ilabel')

    state_scores = openfst.compose(
        arc_scores,
        openfst.arcmap(class_dur_to_class_fst, map_type='to_log')
    )
    best_states = openfst.compose(best_arcs, class_dur_to_class_fst)

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
