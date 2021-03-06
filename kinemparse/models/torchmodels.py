import collections
import logging
import math
import pdb

import torch
import numpy as np
from matplotlib import pyplot as plt
import graphviz

import mathtools as m
from mathtools import utils
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

        bigram_counts = m.np.zeros((self.num_states, self.num_states))
        for (i, j), count in edge_counts.items():
            bigram_counts[i, j] = count

        unigram_counts = m.np.zeros(self.num_states)
        for i, count in state_counts.items():
            unigram_counts[i] = count

        initial_counts = m.np.zeros(self.num_states)
        for i, count in init_states.items():
            initial_counts[i] = count

        final_counts = m.np.zeros(self.num_states)
        for i, count in final_states.items():
            final_counts[i] = count

        # Regularize the heck out of these counts
        bigram_counts[0, 0] += empty_regularizer
        bigram_counts[:, 0] += zero_transition_regularizer
        bigram_counts[0, :] += zero_transition_regularizer
        bigram_counts += uniform_regularizer
        diag_indices = m.np.diag_indices_from(bigram_counts)
        bigram_counts[diag_indices] += diag_regularizer

        if override_transitions:
            logger.info('Overriding bigram_counts with an array of all ones')
            bigram_counts = m.np.ones_like(bigram_counts)

        denominator = bigram_counts.sum(1)
        transition_probs = bigram_counts / denominator
        initial_probs = initial_counts / initial_counts.sum()
        final_probs = final_counts / final_counts.sum()

        self.transition_weights = m.np.log(transition_probs)
        self.initial_weights = m.np.log(initial_probs)
        self.final_weights = m.np.log(final_probs)

    def predictSeq(self, *feat_seqs, decode_method='MAP', viz_predictions=False, **kwargs):

        input_seq = zip(*feat_seqs)
        scores, component_poses = utils.batchProcess(
            self.obsv_model.forward,
            input_seq,
            static_kwargs={'return_poses':True},
            unzip=True
        )
        scores = torch.stack(scores, dim=1)

        outputs = super().forward(scores, scores_as_input=True)
        pred_idxs = super().predict(outputs)[0]

        pred_states = self.integerizer.deintegerizeSequence(pred_idxs)
        pred_poses = tuple(component_poses[t][i] for t, i in enumerate(pred_idxs))

        return pred_states, pred_idxs, None, scores, pred_poses


def gaussian_logprob(x, mu, K_inv_sqrt, log_Z=None, viz=False, ignore_cov=False):
    """ Compute log-probability under a set of Gaussians.

    Parameters
    ----------
    x : torch.Tensor, shape (num_samples, num_dims)
        Input data that we want to score.
    mu : torch.Tensor, shape (num_classes, num_dims)
        Mean of each Gaussian.
    K_inv_sqrt : torch.Tensor, shape (num_classes, num_dims, num_dims)
        The square root of each Gaussian's precision matrix;
        i.e. K ** (-1/2).
    log_Z : torch.Tensor, shape (num_classes,), optional
        Each Gaussian's log-normalizing constant. This quantity will be computed
        from ``K_inv_sqrt`` if it is not supplied.

    Returns
    -------
    score : torch.Tensor, shape (num_samples, num_classes)
        The (t,n)-th element is the log-probability of sample x[t] under Gaussian n.
    """

    if log_Z is None:
        raise NotImplementedError()

    # Compute un-normalized score
    # log p(x ; mu, K) = -score(x; mu, K) - log Z(K)
    centered = (x[..., None] - mu.transpose(0, 1))

    if ignore_cov:
        scaled = centered
    else:
        scaled = torch.einsum("nij,bjn->bin", K_inv_sqrt, centered)

    score = torch.einsum("bin,bin->bn", scaled, scaled)

    # Subtract off the log-normalizing constant Z(K)
    if ignore_cov:
        log_prob = -score
    else:
        log_prob = -score - log_Z

    if viz:
        dists = torch.einsum("bin,bin->bn", centered, centered) ** 0.5

        f, axes = plt.subplots(dists.shape[1], figsize=(12, 12))
        for i, axis in enumerate(axes):
            axis.plot(dists[:, i].numpy())
        axes[0].set_title("dist")
        plt.show()

        f, axes = plt.subplots(score.shape[1], figsize=(12, 12))
        for i, axis in enumerate(axes):
            axis.plot(score[:, i].numpy())
        axes[0].set_title("score")
        plt.show()

        f, axes = plt.subplots(log_prob.shape[1], figsize=(12, 12))
        for i, axis in enumerate(axes):
            axis.plot(log_prob[:, i].numpy())
        axes[0].set_title("log_prob")
        plt.show()

    return log_prob


def logmean(X, dim=None):
    """ Compute the mean along a dimension, in the log-domain. """
    num_items = X.shape[dim]
    X_sum = torch.logsumexp(X, dim=dim)
    mean = X_sum - num_items

    return mean


def fitGaussianParams(samples, num_classes=None):
    """ Fit parameters for a set of Gaussians.

    Parameters
    ----------
    samples : dict{int -> torch.tensor, shape (NUM_SAMPLES, k)}
        Maps class indices to samples.
    num_classes : int, optional
        If not provided, `num_classes` is taken to be the maximum index in `samples`.

    Returns
    -------
    means : torch.tensor, shape (num_classes, k)
    precs_sqrt : torch.tensor, shape (num_classes, k, k)
    logZs : torch.tensor, shape (num_classes)
    """

    k = next(iter(samples.values())).shape[1]
    k_log_2pi = k * torch.tensor(2 * math.pi).log()

    means = torch.zeros(num_classes, k)
    precs_sqrt = torch.zeros(num_classes, k, k)
    logZs = torch.zeros(num_classes)
    eigenvalues = torch.zeros(num_classes, k)
    eigenvectors = torch.zeros(num_classes, k, k)

    for idx, detections in samples.items():
        has_nan = torch.isnan(detections).any(dim=1)
        non_nan_detections = detections[~has_nan, :]
        mean = torch.mean(non_nan_detections, 0)

        centered = non_nan_detections - mean
        U, S, Vt = torch.svd(centered)
        V = Vt.transpose(0, 1)
        K_inv_sqrt = V @ torch.diag(1 / S)

        # normalizing constant is sqrt((2pi) ** k * det(K))
        #   --> log is 0.5 * (k log(2pi) + log(det(K)))
        # log(det(K)) = log(det(S ** 2)) = 2 log(det(S)) = 2 trace(log(S))
        log_det_K = 2 * S.log().sum()
        log_Z = 0.5 * (k_log_2pi + log_det_K)

        means[idx, :] = mean
        precs_sqrt[idx, :, :] = K_inv_sqrt
        logZs[idx] = log_Z
        eigenvalues[idx, :] = S ** 2
        eigenvectors[idx, :, :] = V

    return means, precs_sqrt, logZs, eigenvalues, eigenvectors


def stateToString(names):
    if names == '<BOS>':
        return names
    if not names:
        return '()'
    combined_names = [names[0]]
    for name in names[1:]:
        if combined_names[-1].startswith(name[:-1]):
            if name == 'null':
                combined_names.append(name)
            else:
                combined_names[-1] += name[-1]
        else:
            combined_names.append(name)
    string = '(' + ', '.join(combined_names) + ')'
    return string


class ReachDetector(torch.nn.Module):
    def __init__(
            self, part_names, part_names_to_idxs, part_idxs_to_bins,
            bias_params=None, nan_score=None):
        """
        Parameters
        ----------
        part_names : tuple(string)
        part_names_to_idxs : dict(string -> int)
        part_idxs_to_bins : torch.tensor(int)
        bias_params : (float, float)
            0 -- Amount of probability to add when hand detection is present
            1 -- Amount of probability to add when hand detection is missing
        nan_score : float
            ???
        """

        super().__init__()

        if bias_params is None:
            # Consistent with settings in Vo et al.
            bias_params = (0.01, 0.1)

        if nan_score is None:
            nan_score = torch.tensor(-math.inf)
        self.nan_score = nan_score

        self._part_names = part_names
        self._part_names_to_idxs = part_names_to_idxs
        self._part_to_bin = torch.tensor(part_idxs_to_bins)

        self._num_parts = len(self._part_names)
        self._num_bins = len(self._part_to_bin.unique())

        self._bin_means = None
        self._bin_precs_sqrt = None
        self._bin_logZs = None
        self._mean_detection_scores = None
        self._log_biases = torch.tensor(bias_params).log()

    @property
    def num_states(self):
        return self._num_parts

    def fit(self, action_seqs, hand_detection_seqs):
        """ Fit parameters for the duration and bin detector models.

        Parameters
        ----------
        action_seqs : tuple(numpy.ndarray of int, shape (num_actions, 3))
        hand_detection_seqs : tuple(numpy.ndarray of float, shape (num_samples, 2))
        """

        # action_seqs = tuple(torch.tensor(a) for a in action_seqs)
        # hand_detection_seqs = tuple(torch.tensor(d).float() for d in hand_detection_seqs)

        self._fitBinDetectors(hand_detection_seqs, action_seqs)
        self._fitDurations(action_seqs)

    def _collectSegments(self, hand_detection_seq, action_seq, win_len=17, segments=None):
        half_win_len = (win_len - 1) // 2
        min_val = torch.zeros_like(action_seq[:, 1])
        max_val = torch.ones_like(action_seq[:, 1]) * (hand_detection_seq.shape[0] - 1)
        start_idxs = torch.max(action_seq[:, 1] - half_win_len, min_val)
        end_idxs = torch.min(action_seq[:, 1] + half_win_len, max_val)

        for action_idx, start_idx, end_idx in zip(action_seq[:,0], start_idxs, end_idxs):
            bin_idx = self._part_to_bin[int(action_idx)]
            segments[int(bin_idx)].append(hand_detection_seq[start_idx:end_idx, :])

        return segments

    def _collectDurations(self, action_seq, durations=None):
        for action_idx, start_idx, end_idx in action_seq:
            durations[int(action_idx)].append(end_idx - start_idx)
        return durations

    def _fitDurations(self, action_seqs):
        """ """

        # Get segments from action labels
        durations = collections.defaultdict(list)
        for action_seq in action_seqs:
            self._collectDurations(action_seq, durations=durations)
        durations = {key: torch.tensor(value).float()[:, None] for key, value in durations.items()}
        self._max_dur = max(float(ds.max()) for ds in durations.values())

        self._dur_means, self._dur_precs_sqrt, self._dur_logZs, self._dur_evals, self._dur_evecs = \
            fitGaussianParams(durations, num_classes=self._num_parts)

    def _fitBinDetectors(self, hand_detection_seqs, action_seqs, outlier_ratio=0.05):
        """ Fit a Gaussian location model for each bin.

        Parameters
        ----------
        hand_detection_seqs : torch.Tensor, shape (num_samples, 4)
        action_seqs : torch.Tensor of int, shape (num_actions, 3)
            Columns are (action_id, start_idx, end_idx).
        """
        #   Detect reaching hands
        # reaching_hand = airplanecorpus.getReachingHand(
        #     hand_detection_seqs, reaching_threshold=250)

        # Get segments from action labels
        segments = collections.defaultdict(list)
        for detections, action_seq in zip(hand_detection_seqs, action_seqs):
            self._collectSegments(detections, action_seq, segments=segments)
        segments = {key: torch.cat(value) for key, value in segments.items()}
        self._bin_means, self._bin_precs_sqrt, self._bin_logZs, self._bin_evals, self._bin_evecs = \
            fitGaussianParams(segments, num_classes=self._num_bins)
        # self.visualize(segments, train=True)

        # Remove bottom 5% of detections and re-fit model
        if outlier_ratio is not None:
            inlier_segments = {
                i: self._removeOutliers(detections, i, outlier_ratio=outlier_ratio)
                for i, detections in segments.items()
            }
            self._bin_means, self._bin_precs_sqrt, self._bin_logZs, self._bin_evals, self._bin_evecs = \
                fitGaussianParams(inlier_segments, num_classes=self._num_bins)
            # self.visualize(segments, train=True)

        # Compute mean detection scores for each bin detector (in log-domain)
        subsample_period = 5
        samples = torch.cat(tuple(v[::subsample_period] for v in segments.values()), dim=0)
        scores = self._binScores(samples, normalize=False, add_bias=True, viz=False)
        self._mean_detection_scores = logmean(scores, dim=0)

    def _removeOutliers(self, detections, model_index, outlier_ratio=None):
        """ Return the proportion of detections with highest score under a model.
        """

        if not outlier_ratio:
            return detections

        bin_means = self._bin_means[model_index:model_index + 1]
        bin_precs_sqrt = self._bin_precs_sqrt[model_index:model_index + 1]
        bin_logZs = self._bin_logZs[model_index:model_index + 1]

        num_samples = detections.shape[0]
        num_outliers = int(outlier_ratio * num_samples)

        scores = self._binScores(
            detections,
            bin_means=bin_means, bin_precs_sqrt=bin_precs_sqrt, bin_logZs=bin_logZs,
            normalize=False, add_bias=True, viz=False
        )
        # argsort sorts in ascending order and scores are log-likelihoods, so
        # we want to remove the samples with LOWEST indices
        sort_idxs = scores.argsort(0)[:, 0]
        inliers = detections[sort_idxs > num_outliers, :]

        return inliers

    def _scoreDurations(self, segment_lens):
        """ Compute log-prob of segment durations for each part.

        Parameters
        ----------
        segment_lens : torch.tensor, shape (num_lens)

        Returns
        -------
        dur_scores : torch.tensor, shape (num_lens, num_parts)
        """

        dur_scores = gaussian_logprob(
            segment_lens, self._dur_means, self._dur_precs_sqrt,
            log_Z=self._dur_logZs
        )

        return dur_scores

    def maxDuration(self, num_stds=1):
        upper_bound = self._dur_means + num_stds * self._dur_evals ** 0.5
        thresh_dur, _ = upper_bound.max(dim=0)
        # logger.info(f"max_dur: {self._max_dur}, thresh: {thresh_dur.squeeze()}")
        return self._max_dur

    def predictSeq(self, *args, scores=None, viz=True, **kwargs):
        args = tuple(torch.tensor(a).float() for a in args)

        if scores is None:
            scores = self.forward(*args, **kwargs, normalize=True, viz=viz, add_bias=False)
        else:
            scores = scores[:, self._part_to_bin]

        pred_idxs = scores.argmax(-1)
        pred_actions = tuple(self._part_names[i] for i in pred_idxs)

        return pred_actions, pred_idxs, scores

    def visualize(self, hand_detections, train=False):
        bin_means = self._bin_means.numpy()
        bin_diag_errs = torch.einsum('bij,bj->bi', self._bin_evecs, self._bin_evals ** 0.5)

        # logger.info(f"stds: {self._bin_evals ** 0.5}")

        _ = plt.figure()
        if train:
            for index, d in hand_detections.items():
                d = d.numpy()
                plt.scatter(d[:, 0], d[:, 1])
        else:
            hand_detections = hand_detections.numpy()
            plt.scatter(hand_detections[:, 0], hand_detections[:, 1])
        plt.errorbar(
            bin_means[:, 0], bin_means[:, 1],
            yerr=bin_diag_errs[:, 0], xerr=bin_diag_errs[:, 1]
        )
        plt.scatter(bin_means[:, 0], bin_means[:, 1])
        plt.show()

    def _binScores(
            self, hand_detections,
            bin_means=None, bin_precs_sqrt=None, bin_logZs=None,
            normalize=True, add_bias=True, viz=False):

        if bin_means is None:
            bin_means = self._bin_means

        if bin_precs_sqrt is None:
            bin_precs_sqrt = self._bin_precs_sqrt

        if bin_logZs is None:
            bin_logZs = self._bin_logZs

        row_has_nan = torch.isnan(hand_detections).any(1)
        non_nan_detections = hand_detections[~row_has_nan, :]

        if viz:
            _ = plt.figure(figsize=(12, 12))
            plt.plot(hand_detections[:, 0].numpy(), hand_detections[:, 1].numpy())
            plt.scatter(hand_detections[:, 0].numpy(), hand_detections[:, 1].numpy())
            plt.show()

        # Compute log-likelihood for each bin
        detection_scores = gaussian_logprob(
            non_nan_detections, bin_means, bin_precs_sqrt, log_Z=bin_logZs,
            ignore_cov=False,
            viz=viz
        )

        num_samples = hand_detections.shape[0]
        num_classes = detection_scores.shape[-1]
        bin_scores = torch.full((num_samples, num_classes), -math.inf)
        bin_scores[~row_has_nan, :] = detection_scores

        # Add in the bias term (in log-domain)
        if add_bias:
            bias_param_idx = torch.isnan(hand_detections).any(1).long()
            log_biases = self._log_biases[bias_param_idx][:, None].expand(-1, bin_scores.shape[1])
            bin_scores = torch.logsumexp(torch.stack((bin_scores, log_biases), -1), -1)

        if viz:
            f, axes = plt.subplots(bin_scores.shape[1], figsize=(12, 12))
            for i, axis in enumerate(axes):
                axis.plot(bin_scores[:, i].numpy())
            axes[0].set_title("bin scores")
            plt.show()

        if normalize:
            # Divide by the average detector score (in log-domain)
            bin_scores = bin_scores - self._mean_detection_scores

        if viz:
            _ = plt.figure(figsize=(16, 1))
            plt.matshow(bin_scores.transpose(0, 1).numpy()[:, ::10])
            # f, axes = plt.subplots(3, figsize=(16, 4))
            # axes[0].matshow(bin_scores.transpose(0, 1).numpy()[:, ::10])
            # axes[1].matshow(log_biases.transpose(0, 1).numpy()[:, ::10])
            # axes[2].matshow(bin_scores.transpose(0, 1).numpy()[:, ::10])
            plt.show()

        return bin_scores

    def forward(self, hand_detections, **bin_kwargs):
        """ Compute (almost) log-likelihood of hand detections for each part.

        Parameters
        ---------
        hand_detection : torch.tensor, shape (num_samples, 2)
        normalize : bool, optional
            If True, divide by the average detector score (in log domain)

        Returns
        -------
        part_scores : torch.tensor, shape (num_samples, num_parts)
        """

        bin_scores = self._binScores(hand_detections, **bin_kwargs)

        # Get part scores from part -> bin associations
        part_scores = bin_scores[:, self._part_to_bin]

        if True:
            f, axes = plt.subplots(part_scores.shape[1], figsize=(12, 12))
            for i, axis in enumerate(axes):
                axis.plot(part_scores[:, i].numpy())
                axis.set_ylabel(self._part_names[i])
            axes[0].set_title("part scores")
            plt.show()

        return bin_scores


def smoothCounts(
        edge_counts, state_counts, init_states, final_states,
        # empty_regularizer=0, zero_transition_regularizer=0,
        init_regularizer=0, final_regularizer=0,
        uniform_regularizer=0, diag_regularizer=0,
        override_transitions=False, structure_only=False):

    num_states = len(state_counts)

    bigram_counts = torch.zeros((num_states, num_states))
    for (i, j), count in edge_counts.items():
        bigram_counts[i, j] = count

    unigram_counts = torch.zeros(num_states)
    for i, count in state_counts.items():
        unigram_counts[i] = count

    initial_counts = torch.zeros(num_states)
    for i, count in init_states.items():
        initial_counts[i] = count

    final_counts = torch.zeros(num_states)
    for i, count in final_states.items():
        final_counts[i] = count

    # Regularize the heck out of these counts
    initial_states = initial_counts.nonzero()[:, 0]
    for i in initial_states:
        bigram_counts[i, i] += init_regularizer

    final_states = final_counts.nonzero()[:, 0]
    for i in final_states:
        bigram_counts[i, i] += final_regularizer
    # bigram_counts[:, 0] += zero_transition_regularizer
    # bigram_counts[0, :] += zero_transition_regularizer
    bigram_counts += uniform_regularizer
    diag_indices = np.diag_indices(bigram_counts.shape[0])
    bigram_counts[diag_indices] += diag_regularizer

    if override_transitions:
        logger.info('Overriding bigram_counts with an array of all ones')
        bigram_counts = torch.ones_like(bigram_counts)

    if structure_only:
        bigram_counts = (bigram_counts > 0).float()
        initial_counts = (initial_counts > 0).float()
        final_counts = (final_counts > 0).float()

    denominator = bigram_counts.sum(1)
    transition_probs = bigram_counts / denominator[:, None]
    transition_probs[torch.isnan(transition_probs)] = 0
    initial_probs = initial_counts / initial_counts.sum()
    final_probs = (final_counts > 0).float()  # final_counts / final_counts.sum()

    return transition_probs, initial_probs, final_probs

# class AirplaneParser(LegacyHmmInterface, torchutils.SemiMarkovScorer, ReachDetector):
#     pass


class AirplaneParser(torch.nn.Module):
    def __init__(self, *args, device=None, subsample_period=1):
        super().__init__()

        self._obsv_model = ReachDetector(*args)
        self._seq_model = torchutils.SemiMarkovScorer()
        self._seq_model.scoreSegment = self._scoreSegment
        self._seq_model.scoreDurations = self._scoreDurations

        self.device = device

        self._subsample_period = subsample_period

    def _scoreDurations(self, durations, dur_scores=None):
        if dur_scores is None:
            durations = durations.float() * self._subsample_period
            dur_scores = self._obsv_model._scoreDurations(durations[:, None])
        else:
            dur_scores = dur_scores[durations, :]
        return self._expandActionScores(dur_scores[None, ...])

    def _scoreSegment(self, action_scores):
        action_seg_scores = action_scores.sum(dim=-1)
        # pdb.set_trace()
        return self._expandActionScores(action_seg_scores)

    def _expandActionScores(self, action_scores):
        shape = action_scores.shape[:-1] + (self.num_states, self.num_states)
        transition_scores = torch.full(shape, -float('Inf'))
        transition_scores[..., self._edges] = action_scores[..., self._edge_labels[self._edges]]
        # pdb.set_trace()
        return transition_scores

    def fit(self, action_seqs, observations, **seq_kwargs):
        action_seqs = tuple(torch.tensor(s).long() for s in action_seqs)
        observations = tuple(torch.tensor(s).float() for s in observations)

        self._fitObsvModel(action_seqs, observations)

        action_id_seqs = tuple(action_seq[:, 0] for action_seq in action_seqs)
        action_seqs = tuple(
            tuple(self._obsv_model._part_names[a] for a in action_id_seq)
            for action_id_seq in action_id_seqs
        )

        action_id_seqs = tuple(
            tuple(self._obsv_model._part_names_to_idxs[a] for a in a_seq)
            for a_seq in action_seqs
        )

        def disambiguate(action_seq):
            if 'wheel2' not in action_seq:
                i = action_seq.index('body4')
                return action_seq[:i] + ('body4_',) + action_seq[i + 1:]
            return action_seq

        action_seqs = tuple(disambiguate(a_seq) for a_seq in action_seqs)

        state_seqs = tuple(
            tuple(action_seq[:i] for i in range(len(action_seq) + 1))
            for action_seq in action_seqs
        )
        self.integerizer = fsm.HashableFstIntegerizer(prepend_epsilon=False)

        def make_equivalent(state):
            if 'body4_' in state:
                i = state.index('body4_')
                return state[:i] + ('body4', 'wheel1', 'wheel2') + state[i + 1:]
            return state

        state_seqs = tuple(
            tuple(frozenset(make_equivalent(state)) for state in state_seq)
            for state_seq in state_seqs
        )
        self.integerizer.updateFromSequences(state_seqs)
        state_id_seqs = tuple(self.integerizer.integerizeSequence(ss) for ss in state_seqs)

        self._fitSeqModel(state_id_seqs, **seq_kwargs)

        # Record edge IDs/labels for every state transition
        self._edge_labels = -torch.ones(self.num_states, self.num_states, dtype=torch.long)
        self._edge_labels[self._edges] = 0
        for action_ids, state_ids in zip(action_id_seqs, state_id_seqs):
            if len(action_ids) != len(state_ids[1:]):
                err_str = f"{len(action_ids)} actions != {len(state_ids)} states"
                raise AssertionError(err_str)
            for i in range(len(action_ids)):
                s_cur = state_ids[i]
                s_next = state_ids[i + 1]
                a = action_ids[i]
                stored_label = self._edge_labels[s_next, s_cur]
                if stored_label > 0 and stored_label != a:
                    err_str = f"Transition ({s_cur} -> {s_next}) has multiple labels!"
                    raise AssertionError(err_str)
                self._edge_labels[s_next, s_cur] = a

        # self.printFST()

    @property
    def _edges(self):
        # return self._edge_labels >= 0
        return self._seq_model.transition_probs > 0

    def printFST(self):
        nz = self._edges.nonzero()
        s_cur = nz[:, 1]
        s_next = nz[:, 0]
        edge_labels = self._edge_labels[s_next, s_cur]
        for i, j, k in zip(s_cur, s_next, edge_labels):
            a_name = self._obsv_model._part_names[k]
            tx_prob = self._seq_model.transition_probs[j, i]
            prev_state = stateToString(self.integerizer[i])
            cur_state = stateToString(self.integerizer[j])
            logger.info(f"{prev_state} -> {cur_state}: {a_name}, {tx_prob}")

        nz = self._seq_model.start_probs.nonzero()
        s_i = nz[:, 0]
        for i in s_i:
            prob = self._seq_model.start_probs[i]
            cur_state = stateToString(self.integerizer[i])
            logger.info(f"(s) -> {cur_state}: <eps>, {prob}")

        nz = self._seq_model.end_probs.nonzero()
        s_f = nz[:, 0]
        for i in s_f:
            prob = self._seq_model.end_probs[i]
            prev_state = stateToString(self.integerizer[i])
            logger.info(f"{prev_state} -> (e): <eps>, {prob}")

    def showFST(self, action_dict=None, state_dict=None, display_dir='LR'):
        """ When returned from an ipython cell, this will generate the FST visualization. """

        def make_label(x):
            if x < 0:
                return '<eps>'
            return str(action_dict[x])

        fst = graphviz.Digraph("finite state machine", filename="fsm.gv")
        fst.attr(rankdir=display_dir)

        zero = 0

        initial_state = int(self._seq_model.start_probs.nonzero()[0][0])

        for s_idx in range(self.num_states):
            s_label = str(s_idx)
            final_prob = self._seq_model.end_probs[s_idx]
            if final_prob == zero:
                fst.node(str(s_idx), label=s_label, shape='circle')
            else:
                s_label += f'\n({final_prob:.2f})'
                fst.node(str(s_idx), label=s_label, shape='doublecircle')

        fst.attr('node', shape='circle')
        edges = self._edges.nonzero()
        for next_state, cur_state in edges.tolist():
            edge_label = self._edge_labels[next_state, cur_state]
            edge_prob = self._seq_model.transition_probs[next_state, cur_state]

            label = make_label(edge_label) + f'/{edge_prob:.2f}'
            fst.edge(str(cur_state), str(next_state), label=label)

        # mark the start state
        fst.node('', shape='point')
        fst.edge('', str(initial_state))

        return fst

    def _fitSeqModel(self, label_seqs, **kwargs):
        p_tx, p_i, p_f = smoothCounts(*fsm.countSeqs(label_seqs), **kwargs)
        w_tx = p_tx.log()
        w_i = p_i.log()
        w_f = p_f.log()

        self._seq_model.start_probs = p_i
        self._seq_model.end_probs = p_f
        self._seq_model.transition_probs = p_tx.transpose(0, 1)
        self._seq_model.start_weights = torch.nn.Parameter(w_i, requires_grad=False)
        self._seq_model.end_weights = torch.nn.Parameter(w_f, requires_grad=False)
        self._seq_model.transition_weights = torch.nn.Parameter(
            w_tx.transpose(0, 1), requires_grad=False
        )

        self.num_states = w_tx.shape[0]
        self._seq_model.max_duration = int(self._obsv_model.maxDuration() / self._subsample_period)

    def _fitObsvModel(self, *args, **kwargs):
        self._obsv_model.fit(*args, **kwargs)

    def predictSeq(self, hand_detections, obsv_scores=None, dur_scores=None, **kwargs):
        hand_detections = torch.tensor(hand_detections[::self._subsample_period, :]).float()

        if obsv_scores is None:
            obsv_scores = self._obsv_model.forward(hand_detections)
        else:
            obsv_scores = obsv_scores[:, self._obsv_model._part_to_bin]
            num_detections = hand_detections.shape[0]
            num_scores = obsv_scores.shape[0]
            if num_detections != num_scores:
                err_str = f"{num_detections} detections != {num_scores} scores"
                if abs(num_detections - num_scores) == 1:
                    logger.warning(err_str)
                else:
                    raise AssertionError(err_str)

        obsv_scores = obsv_scores.transpose(0, 1)[None, ...]
        log_potentials = self._seq_model.forward(obsv_scores, dur_scores=dur_scores)

        log_potentials = log_potentials.to(self.device)
        pred_action_idxs, pred_scores = self._seq_model.predict(
            log_potentials, arc_labels=self._edge_labels, **kwargs
        )

        pred_action_idxs, pred_scores = self._seq_model.predict(
            log_potentials, arc_labels=self._edge_labels, **kwargs
        )

        return pred_action_idxs, pred_scores.detach()


class FstAirplaneParser(AirplaneParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override sequence model
        self._seq_model = torchutils.MarkovScorer()
        self._seq_model.scoreSegment = self._scoreSegment

    def predictSeq(self, hand_detections, obsv_scores=None, dur_scores=None, **kwargs):
        hand_detections = torch.tensor(hand_detections[::self._subsample_period, :]).float()

        if obsv_scores is None:
            obsv_scores = self._obsv_model.forward(hand_detections)
        else:
            obsv_scores = obsv_scores[:, self._obsv_model._part_to_bin]

        obsv_scores = obsv_scores.transpose(0, 1)[None, ...]
        log_potentials = self._seq_model.forward(obsv_scores)

        log_potentials = log_potentials.to(self.device)
        pred_action_idxs, pred_scores = self._seq_model.predict(
            log_potentials, arc_labels=self._edge_labels, **kwargs
        )

        return pred_action_idxs, pred_scores.detach()


class RenderingCrf(LegacyHmmInterface, torchutils.LinearChainScorer, scene.TorchSceneScorer):
    pass
