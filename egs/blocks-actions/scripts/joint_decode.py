import os
import logging

import yaml
import numpy as np
import scipy
from matplotlib import pyplot as plt

import LCTM.metrics

import pywrapfst as openfst

from kinemparse import decode
from mathtools import utils
from seqtools import fstutils_openfst as libfst


logger = logging.getLogger(__name__)


def eval_metrics(pred_seq, true_seq, name_suffix='', append_to={}):
    state_acc = (pred_seq == true_seq).astype(float).mean()

    metric_dict = {
        'State Accuracy' + name_suffix: state_acc,
        'State Edit Score' + name_suffix: LCTM.metrics.edit_score(pred_seq, true_seq) / 100,
        'State Overlap Score' + name_suffix: LCTM.metrics.overlap_score(pred_seq, true_seq) / 100
    }

    append_to.update(metric_dict)
    return append_to


def suppress_nonmax(scores):
    col_idxs = scores.argmax(axis=1)
    new_scores = np.zeros_like(scores)
    row_idxs = np.arange(scores.shape[0])
    new_scores[row_idxs, col_idxs] = scores[row_idxs, col_idxs]
    return new_scores


def make_event_assembly_transition_priors(event_vocab, assembly_vocab):
    def isValid(event, cur_assembly, next_assembly):
        is_valid = diff == event
        return is_valid

    num_events = len(event_vocab)
    num_assemblies = len(assembly_vocab)
    priors = np.zeros((num_events, num_assemblies, num_assemblies), dtype=bool)

    for j, cur_assembly in enumerate(assembly_vocab):
        for k, next_assembly in enumerate(assembly_vocab):
            try:
                diff = next_assembly - cur_assembly
            except ValueError:
                continue

            for i, event in enumerate(event_vocab):
                priors[i, j, k] = diff == event

    return priors


def make_assembly_transition_priors(assembly_vocab):
    def isValid(diff):
        for i in range(diff.connections.shape[0]):
            c = diff.connections.copy()
            c[i, :] = 0
            c[:, i] = 0
            if not c.any():
                return True
        return False

    num_assemblies = len(assembly_vocab)
    priors = np.zeros((num_assemblies, num_assemblies), dtype=bool)

    for j, cur_assembly in enumerate(assembly_vocab):
        for k, next_assembly in enumerate(assembly_vocab):
            if cur_assembly == next_assembly:
                continue

            try:
                diff = next_assembly - cur_assembly
            except ValueError:
                continue

            priors[j, k] = isValid(diff)

    return priors


def count_transitions(label_seqs, num_classes, support_only=False):
    start_counts = np.zeros(num_classes, dtype=float)
    end_counts = np.zeros(num_classes, dtype=float)
    for label_seq in label_seqs:
        start_counts[label_seq[0]] += 1
        end_counts[label_seq[-1]] += 1

    start_probs = start_counts / start_counts.sum()
    end_probs = end_counts / end_counts.sum()

    if support_only:
        start_probs = (start_probs > 0).astype(float)
        end_probs = (end_probs > 0).astype(float)

    return start_probs, end_probs


def count_priors(label_seqs, num_classes, stride=None, approx_upto=None, support_only=False):
    dur_counts = {}
    class_counts = {}
    for label_seq in label_seqs:
        for label, dur in zip(*utils.computeSegments(label_seq[::stride])):
            class_counts[label] = class_counts.get(label, 0) + 1
            dur_counts[label, dur] = dur_counts.get((label, dur), 0) + 1

    class_priors = np.zeros((num_classes))
    for label, count in class_counts.items():
        class_priors[label] = count
    class_priors /= class_priors.sum()

    max_dur = max(dur for label, dur in dur_counts.keys())
    dur_priors = np.zeros((num_classes, max_dur))
    for (label, dur), count in dur_counts.items():
        assert dur
        dur_priors[label, dur - 1] = count
    dur_priors /= dur_priors.sum(axis=1, keepdims=True)

    if approx_upto is not None:
        cdf = dur_priors.cumsum(axis=1)
        approx_bounds = (cdf >= approx_upto).argmax(axis=1)
        dur_priors = dur_priors[:, :approx_bounds.max()]

    if support_only:
        dur_priors = (dur_priors > 0).astype(float)

    return class_priors, dur_priors


def viz_priors(fn, class_priors, dur_priors):
    fig, axes = plt.subplots(3)
    axes[0].matshow(dur_priors)
    axes[1].stem(class_priors)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def viz_transition_probs(fig_dir, transitions):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for i, transition_arr in enumerate(transitions):
        plt.matshow(transition_arr)
        plt.savefig(os.path.join(fig_dir, f"action={i:03d}"))
        plt.close()


def pack_scores(transitions, start, end):
    num_assemblies = transitions.shape[0]
    packed = np.zeros((num_assemblies + 1, num_assemblies + 1), dtype=float)
    packed[0, :-1] = start
    packed[1:, -1] = end
    packed[1:, :-1] = transitions
    return packed


def computeMoments(feature_seqs):
    features = np.concatenate(feature_seqs, axis=0)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return mean, std


def make_joint_scores(event_score_seq, assembly_score_seq, nonzero_indices, weights=[1, 1]):
    event_indices, assembly_indices = nonzero_indices.T
    score_seq = (
        weights[0] * event_score_seq[:, event_indices]
        + weights[1] * assembly_score_seq[:, assembly_indices]
    )

    return score_seq


def make_joint_labels(e, a, ea_mapper):
    labels = np.array(
        [utils.getIndex((i, j), ea_mapper) for i, j in zip(e, a)],
        dtype=int
    )
    return labels


def make_duration_fst(
        final_weights, class_vocab, class_dur_to_str, input_str_to_parts,
        dur_internal_str='I', dur_final_str='F',
        input_symbols=None, output_symbols=None,
        allow_self_transitions=True,
        arc_type='standard'):
    num_classes, num_states = final_weights.shape

    fst = openfst.VectorFst(arc_type=arc_type)
    fst.set_input_symbols(input_symbols)
    fst.set_output_symbols(output_symbols)

    one = openfst.Weight.one(fst.weight_type())
    zero = openfst.Weight.zero(fst.weight_type())

    init_state = fst.add_state()
    final_state = fst.add_state()
    fst.set_start(init_state)
    fst.set_final(final_state, one)

    def durationFst(label_str, dur_internal_str, dur_final_str, final_weights):
        """ Construct a left-to-right WFST from an input sequence.

        Parameters
        ----------
        input_seq : iterable(int or string)

        Returns
        -------
        fst : openfst.Fst
        """

        input_label = fst.input_symbols().find(label_str)
        output_label_int = fst.output_symbols().find(dur_internal_str)
        output_label_ext = fst.output_symbols().find(dur_final_str)

        max_dur = np.nonzero(final_weights != float(zero))[0].max()
        if max_dur < 1:
            raise AssertionError(f"max_dur = {max_dur}, but should be >= 1)")

        states = tuple(fst.add_state() for __ in range(max_dur))
        seg_final_state = fst.add_state()
        fst.add_arc(init_state, openfst.Arc(0, 0, one, states[0]))
        fst.add_arc(seg_final_state, openfst.Arc(0, 0, one, final_state))

        for i, cur_state in enumerate(states):
            cur_state = states[i]

            final_weight = openfst.Weight(fst.weight_type(), final_weights[i])
            if final_weight != zero:
                arc = openfst.Arc(input_label, output_label_ext, one, seg_final_state)
                fst.add_arc(cur_state, arc)

            if i + 1 < len(states):
                next_state = states[i + 1]
                arc = openfst.Arc(input_label, output_label_int, one, next_state)
                fst.add_arc(cur_state, arc)

        return states[0], seg_final_state

    endpoints = tuple(
        durationFst(
            class_vocab[i],
            class_dur_to_str[(*input_str_to_parts[class_vocab[i]], dur_internal_str)],
            class_dur_to_str[(*input_str_to_parts[class_vocab[i]], dur_final_str)],
            final_weights[i],
        )
        for i in range(num_classes)
    )

    for i, (s_cur_first, s_cur_last) in enumerate(endpoints):
        for j, (s_next_first, s_next_last) in enumerate(endpoints):
            if not allow_self_transitions and i == j:
                continue
            arc = openfst.Arc(0, 0, one, s_next_first)
            fst.add_arc(s_cur_last, arc)

    if not fst.verify():
        raise openfst.FstError("fst.verify() returned False")

    return fst


class AssemblyActionRecognizer(decode.AssemblyActionRecognizer):
    def _init_models(
            self, *stage_scores,
            arc_type='log', decode_type='marginal', return_label='output',
            reduce_order='pre', output_stage=3, allow_self_transitions=True):

        self.reduce_order = reduce_order
        self.return_label = return_label
        self.arc_type = arc_type
        self.output_stage = output_stage
        self.decode_type = decode_type
        self.allow_self_transitions = allow_self_transitions

        self.event_assembly_tx_weights = -stage_scores[1]
        self.event_assembly_weights = -scipy.special.logsumexp(
            -self.event_assembly_tx_weights,
            axis=-1
        )
        self.ead_weights = np.stack(
            (self.event_assembly_weights, self.event_assembly_weights),
            axis=-1
        )
        self.nonzero_indices_ea = np.column_stack(
            np.nonzero(~np.isinf(self.event_assembly_weights))
        )
        self.nonzero_indices_ead = np.column_stack(
            np.nonzero(~np.isinf(self.ead_weights))
        )

        model_builders = [
            self.make_joint_dur_model,
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

    def make_joint_dur_model(self, duration_weights):
        self.event_assembly_tx_vocab = self.product_vocab(
            self.event_vocab, self.assembly_vocab,
            nonzero_indices=self.nonzero_indices_ea
        )
        self.event_assembly_tx_to_str = self.map_parts_to_str(
            self.event_assembly_tx_vocab,
            self.event_vocab, self.assembly_vocab
        )
        self.event_assembly_tx_dur_vocab = self.product_vocab(
            self.event_vocab, self.assembly_vocab, self.dur_vocab,
            nonzero_indices=self.nonzero_indices_ead
        )
        self.event_assembly_tx_dur_to_str = self.map_parts_to_str(
            self.event_assembly_tx_dur_vocab,
            self.event_vocab, self.assembly_vocab, self.dur_vocab
        )

        input_str_to_parts = {v: k for k, v in self.event_assembly_tx_to_str.items()}

        event_duration_fst = decode.add_endpoints(
            make_duration_fst(
                duration_weights,
                self.event_assembly_tx_vocab.as_str(),
                self.event_assembly_tx_dur_to_str,
                input_str_to_parts,
                dur_internal_str=self.dur_internal_str, dur_final_str=self.dur_final_str,
                input_symbols=self.event_assembly_tx_vocab.symbol_table,
                output_symbols=self.event_assembly_tx_dur_vocab.symbol_table,
                allow_self_transitions=self.allow_self_transitions,
                arc_type=self.arc_type,
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
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
        reducer = decode.add_endpoints(
            decode.make_reduce_fst(
                self.event_assembly_tx_dur_vocab, output_vocab,
                keep_dim=keep_dim,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')
        return event_duration_fst, self.event_vocab, reducer

    def make_event_to_assembly_model(self, event_to_assembly_weights):
        event_to_assembly_fst = decode.make_event_to_assembly_fst(
            event_to_assembly_weights,
            self.event_vocab.as_str(), self.assembly_vocab.as_str(),
            self.event_assembly_tx_dur_to_str, self.event_assembly_tx_dur_to_str,
            input_symbols=self.event_assembly_tx_dur_vocab.symbol_table,
            output_symbols=self.event_assembly_tx_dur_vocab.symbol_table,
            eps_str=self.eps_str,
            dur_internal_str=self.dur_internal_str, dur_final_str=self.dur_final_str,
            bos_str=self.bos_str, eos_str=self.eos_str,
            state_in_input=True,
            arc_type=self.arc_type,
            sparsity_ref=self.event_assembly_tx_weights
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
        reducer = decode.add_endpoints(
            decode.make_reduce_fst(
                self.event_assembly_tx_dur_vocab, output_vocab,
                keep_dim=keep_dim,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        return event_to_assembly_fst, output_vocab, reducer

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
        sample_vocab = decode.Vocabulary(
            tuple(f"{i}" for i in range(input_scores.shape[0])),
            aux_symbols=self.aux_symbols
        )

        input_lattice = decode.add_endpoints(
            decode.fromArray(
                -input_scores,
                sample_vocab.as_str(), self.event_assembly_tx_vocab.as_str(),
                input_symbols=sample_vocab.symbol_table,
                output_symbols=self.event_assembly_tx_vocab.symbol_table,
                arc_type=self.arc_type
            ),
            bos_str=self.bos_str, eos_str=self.eos_str
        ).arcsort(sort_type='ilabel')

        # Compose event scores with event --> connection map
        decode_lattice = openfst.compose(input_lattice, self.seq_model)
        if not decode_lattice.num_states():
            warn_str = "Empty decode lattice: Input scores aren't compatible with seq model"
            logger.warning(warn_str)
        output_lattice = self._decode(decode_lattice)
        output_weights, weight_type = decode.toArray(
            output_lattice,
            sample_vocab.str_integerizer,
            self.output_vocab.str_integerizer
        )

        return -output_weights


def main(
        out_dir=None, assembly_scores_dir=None, event_scores_dir=None,
        labels_from='assemblies',
        feature_fn_format='score-seq', label_fn_format='true-label-seq',
        only_fold=None, plot_io=None, prefix='seq=', stop_after=None,
        background_action='', stride=None, standardize_inputs=False,
        model_params={}, cv_params={},
        results_file=None, sweep_param_name=None):

    event_scores_dir = os.path.expanduser(event_scores_dir)
    assembly_scores_dir = os.path.expanduser(assembly_scores_dir)
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

    misc_dir = os.path.join(out_dir, 'misc')
    if not os.path.exists(misc_dir):
        os.makedirs(misc_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    scores_dirs = {
        'events': event_scores_dir,
        'assemblies': assembly_scores_dir
    }
    data_dir = scores_dirs[labels_from]
    seq_ids = utils.getUniqueIds(
        data_dir, prefix=prefix, suffix=f'{label_fn_format}.*',
        to_array=True
    )

    event_dataset = utils.FeaturelessCvDataset(
        seq_ids, event_scores_dir,
        prefix=prefix,
        label_fn_format=label_fn_format
    )

    assembly_dataset = utils.FeaturelessCvDataset(
        seq_ids, assembly_scores_dir,
        prefix=prefix,
        label_fn_format=label_fn_format
    )

    logger.info(f"Loaded scores for {len(seq_ids)} sequences from {data_dir}")

    # Define cross-validation folds
    cv_folds = utils.makeDataSplits(len(seq_ids), **cv_params)
    utils.saveVariable(cv_folds, 'cv-folds', out_data_dir)

    # Load vocabs; create priors
    event_vocab = utils.loadVariable('vocab', event_scores_dir)
    assembly_vocab = utils.loadVariable('vocab', assembly_scores_dir)

    vocabs = {
        'event_vocab': tuple(range(len(event_vocab))),
        'assembly_vocab': tuple(range(len(assembly_vocab)))
    }

    try:
        event_priors = utils.loadVariable('event-priors', out_data_dir)
    except AssertionError:
        event_priors = make_event_assembly_transition_priors(event_vocab, assembly_vocab)
        utils.saveVariable(event_priors, 'event-priors', out_data_dir)
        viz_transition_probs(os.path.join(fig_dir, 'event-priors'), event_priors)
        np.savetxt(
            os.path.join(misc_dir, "event-transitions.csv"),
            np.column_stack(event_priors.nonzero()),
            delimiter=",", fmt='%d'
        )

    try:
        assembly_priors = utils.loadVariable('assembly-priors', out_data_dir)
    except AssertionError:
        assembly_priors = make_assembly_transition_priors(assembly_vocab)
        utils.saveVariable(assembly_priors, 'assembly-priors', out_data_dir)
        viz_transition_probs(os.path.join(fig_dir, 'assembly-priors'), assembly_priors[None, ...])
        np.savetxt(
            os.path.join(misc_dir, "assembly-transitions.csv"),
            np.column_stack(assembly_priors.nonzero()),
            delimiter=",", fmt='%d'
        )

    event_assembly_scores = np.log(event_priors)
    assembly_scores = np.log(assembly_priors)
    assembly_scores = np.zeros_like(assembly_scores)
    ea_scores = scipy.special.logsumexp(event_assembly_scores, axis=-1)
    nonzero_indices_ea = np.column_stack(np.nonzero(~np.isinf(ea_scores)))
    _vocabs = (event_vocab, assembly_vocab)
    ea_vocab = tuple(
        tuple(_vocabs[i][j] for i, j in enumerate(indices))
        for indices in nonzero_indices_ea
    )
    ea_mapper = {
        tuple(indices.tolist()): i
        for i, indices in enumerate(nonzero_indices_ea)
    }
    pair_vocab_size = len(ea_mapper)

    for cv_index, cv_fold in enumerate(cv_folds):
        if only_fold is not None and cv_index != only_fold:
            continue

        train_indices, val_indices, test_indices = cv_fold
        logger.info(
            f"CV FOLD {cv_index + 1} / {len(cv_folds)}: "
            f"{len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test"
        )

        cv_str = f'cvfold={cv_index}'

        (train_event_labels, train_seq_ids), _, (_, test_seq_ids) = event_dataset.getFold(cv_fold)
        (train_assembly_labels, _), _, _ = assembly_dataset.getFold(cv_fold)

        event_score_seqs = tuple(
            utils.loadVariable(f"{prefix}{seq_id}_score-seq", event_scores_dir)
            for seq_id in train_seq_ids
        )
        event_mean, event_std = computeMoments(event_score_seqs)

        assembly_start_probs, assembly_end_probs = count_transitions(
            train_assembly_labels, len(assembly_vocab),
            support_only=True
        )
        assembly_start_scores = np.log(assembly_start_probs)
        assembly_end_scores = np.log(assembly_end_probs)
        assembly_transition_scores = pack_scores(
            assembly_scores, assembly_start_scores, assembly_end_scores
        )

        train_labels = tuple(
            make_joint_labels(e, a, ea_mapper)
            for e, a in zip(train_event_labels, train_assembly_labels)
        )

        class_priors, dur_probs = count_priors(
            train_labels, len(ea_mapper),
            approx_upto=0.95, support_only=True
        )
        dur_scores = np.log(dur_probs)
        dur_scores = np.zeros_like(dur_scores)[:pair_vocab_size]
        scores = (dur_scores, event_assembly_scores, assembly_transition_scores)

        model = AssemblyActionRecognizer(scores, vocabs, model_params)

        viz_priors(
            os.path.join(fig_dir, f'{cv_str}_priors'),
            class_priors, dur_probs
        )
        model.write_fsts(os.path.join(misc_dir, f'{cv_str}_fsts'))
        model.save_vocabs(os.path.join(out_data_dir, f'{cv_str}_model-vocabs'))

        for i, seq_id in enumerate(test_seq_ids):
            if stop_after is not None and i >= stop_after:
                break

            trial_prefix = f"{prefix}{seq_id}"
            logger.info(f"  Processing sequence {seq_id}...")

            true_label_seq = utils.loadVariable(
                f"{trial_prefix}_true-label-seq",
                data_dir
            )

            event_score_seq = utils.loadVariable(f"{trial_prefix}_score-seq", event_scores_dir)
            if standardize_inputs:
                event_score_seq = (event_score_seq - event_mean) / event_std
            assembly_score_seq = utils.loadVariable(
                f"{trial_prefix}_score-seq",
                assembly_scores_dir
            )

            joint_score_seq = make_joint_scores(
                event_score_seq, assembly_score_seq,
                model.nonzero_indices_ea
            )

            score_seq = model.forward(joint_score_seq)
            pred_label_seq = model.predict(score_seq)

            metric_dict = eval_metrics(pred_label_seq, true_label_seq)
            for name, value in metric_dict.items():
                logger.info(f"    {name}: {value * 100:.2f}%")
            utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)

            utils.saveVariable(score_seq, f'{trial_prefix}_score-seq', out_data_dir)
            utils.saveVariable(pred_label_seq, f'{trial_prefix}_pred-label-seq', out_data_dir)
            utils.saveVariable(true_label_seq, f'{trial_prefix}_true-label-seq', out_data_dir)

            if plot_io:
                utils.plot_array(
                    joint_score_seq.T, (pred_label_seq.T, true_label_seq.T), ('pred', 'true'),
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
