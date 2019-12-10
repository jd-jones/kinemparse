import logging

import numpy as np
import torch

from seqtools import fsm, torchutils

from .. import scene


logger = logging.getLogger(__name__)


class LegacyHmmInterface(object):
    """ This class allows newer pytorch models to conform to the sklearn-style API. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, transition_weights=None)

    def fit(self, label_seqs, *feat_seqs,
            uniform_regularizer=0, diag_regularizer=0, empty_regularizer=0,
            zero_transition_regularizer=0,
            override_transitions=False):
        """ Fit this model to observed data-label pairs.

        This method estimates the state-to-state transition parameters.

        Parameters
        ----------
        label_seqs : list(np array), shape [n_samples, n_edges]
            Observed labels.
        *feat_seqs : list(np array), shape [n_samples, n_dim]
            Observed data.
        uniform_regularizer : float, optional
            Number of extra counts added to all entries of transition matrix.
        diag_regularizer : float, optional
            Number of extra counts added to diagonal entries of transition
            matrix (ie self-transitions).
        empty_regularizer : float, optional
            Number of extra counts added to the self-transition of the 0
            (presumed empty) state.
        zero_transition_regularizer : float, optional
            Number of extra counts added to all transitions that lead to the
            0 (presumed empty) state.
        override_transitions : bool, optional
            If True, set the state transition parameters to a constant array of
            ones instead of the observed relative frequencies.

        NOTE: state_seqs can actually contain anything iterable over the
            number of samples. If it contains 2D numpy arrays, the array rows
            must represent samples because numpy iterates over rows by default.
        """

        # Instantiate an integerizer and convert labels to integer values
        self.integerizer = fsm.UnhashableFstIntegerizer(prepend_epsilon=False)
        self.integerizer.updateFromSequences(label_seqs)

        num_tokens = sum(len(s) for s in label_seqs)
        num_types = len(self.integerizer._objects)
        logger.info(f"  {num_tokens} tokens -> {num_types} types")

        label_seqs = tuple(self.integerizer.integerizeSequence(s) for s in label_seqs)

        edge_counts, state_counts, init_states, final_states = fsm.countSeqs(label_seqs)

        self.num_states = len(state_counts)

        bigram_counts = np.zeros((self.num_states, self.num_states))
        for (i, j), count in edge_counts.items():
            bigram_counts[i, j] = count

        unigram_counts = np.zeros(self.num_states)
        for i, count in state_counts.items():
            unigram_counts[i] = count

        initial_counts = np.zeros(self.num_states)
        for i, count in init_states.items():
            initial_counts[i] = count

        final_counts = np.zeros(self.num_states)
        for i, count in final_states.items():
            final_counts[i] = count

        # Regularize the heck out of these counts
        bigram_counts[0, 0] += empty_regularizer
        bigram_counts[:, 0] += zero_transition_regularizer
        bigram_counts[0, :] += zero_transition_regularizer
        bigram_counts += uniform_regularizer
        diag_indices = np.diag_indices_from(bigram_counts)
        bigram_counts[diag_indices] += diag_regularizer

        if override_transitions:
            logger.info('Overriding bigram_counts with an array of all ones')
            bigram_counts = np.ones_like(bigram_counts)

        denominator = bigram_counts.sum(axis=1)
        transition_probs = bigram_counts / denominator
        initial_probs = initial_counts / initial_counts.sum()
        final_probs = final_counts / final_counts.sum()

        self.transition_weights = np.log(transition_probs)
        self.initial_weights = np.log(initial_probs)
        self.final_weights = np.log(final_probs)

    def predictSeq(self, *feat_seqs, decode_method='MAP', viz_predictions=False, **kwargs):

        # ((num samples,), ..., (num_samples,)) -> (batch size, ..., num samples)
        input_seq = torch.stack(
            tuple(
                torchutils.tensorFromSequence(seq)
                for seq in feat_seqs
            ),
            dim=0
        )[None, ...]

        # import pdb; pdb.set_trace()

        outputs = super().forward(input_seq)
        pred_idxs = super().predict(outputs)

        pred_states = self.integerizer.deintegerizeSequence(pred_idxs)

        return pred_states, pred_idxs, None, None, None


class RenderingCrf(LegacyHmmInterface, torchutils.LinearChainScorer, scene.TorchSceneScorer):
    pass
