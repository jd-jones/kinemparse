import sys
import functools
import itertools
import os
import logging
import warnings
import collections

import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
from skimage import img_as_float
# from LCTM import models as lctm_models

from blocks.core import utils, geometry, labels
from blocks.core.mixins import LoggerMixin
from blocks.estimation import imageprocessing, render


this_module = sys.modules[__name__]
logger = logging.getLogger(__name__)


# -=( PROBABILITY MODEL HELPER FUNCTIONS )=------------------------------------
def makeHistogram(
        num_classes, samples, normalize=False, ignore_empty=True, backoff_counts=0):
    """ Make a histogram representing the empirical distribution of the input.

    Parameters
    ----------
    num_classes : int
        The length of the histogram.
    samples : numpy array of int, shape (num_samples,)
        Array of outcome indices.
    normalize : bool, optional
        If True, the histogram is normalized to it sums to one (ie this
        method returns a probability distribution). If False, this method
        returns the class counts. Default is False.
    ignore_empty : bool, optional
        Can be useful when ``normalize == False`` and ``backoff_counts == 0``.
        If True, this function doesn't try to normalize the histogram of an
        empty input. Trying to normalize that histogram would lead to a
        divide-by-zero error and an output of all ``NaN``.
    backoff_counts : int or float, optional
        This amount is added to all bins in the histogram before normalization
        (if `normalize` is True). Non-integer backoff counts are allowed.

    Returns
    -------
    class_histogram : numpy array of float, shape (num_clusters,)
        Histogram representing the input data. If `normalize` is True, this is
        a probability distribution (possibly smoothed via backoff). If not,
        this is the number of times each cluster was encountered in the input,
        plus any additional backoff counts.
    """

    hist = np.zeros(num_classes)

    # Convert input to array, to be safe
    samples = np.array(samples)

    # Sometimes the input can be empty. We would rather return a vector of
    # zeros than trying to normalize and get NaN.
    if ignore_empty and not np.any(samples):
        return hist

    # Count occurrences of each class in the histogram
    for index in range(num_classes):
        class_count = np.sum(samples == index)
        hist[index] = class_count

    # Add in the backoff counts
    hist += backoff_counts

    if normalize:
        hist /= hist.sum()

    return hist


def assignToModes(probs, mode_subset=None, log_domain=False):
    """ Hard-assign distributions to their modes.

    Parameters
    ----------
    probs : numpy array of float, shape (num_samples, num_outcomes)
        A distribution over possible outcomes for a sample of data. This is
        usually the probability score of some latent class.
    mode_subset : iterable(int), optional
        If a value is passed for this argument, distributions are only hard-assigned
        if their modes are in this set. Otherwise, this function returns the
        input distribution.
    log_domain : bool, optional
        If True, the input is treated as log-probabilities. Default is False.

    Returns
    -------
    assigned_probs : numpy array of float, shape (num_samples, num_outcomes)
        A copy of the input with new distributions along the rows. Each row is
        a delta distribution with probability 1 at the mode of the corresponding
        row of the input.
    """

    if log_domain:
        unit = 0
        zero = -np.inf
    else:
        unit = 1
        zero = 0

    modes = probs.argmax(axis=1)
    assigned_probs = np.full(probs.shape, zero)
    assigned_probs[range(assigned_probs.shape[0]), modes] = unit

    if mode_subset is not None:
        mode_in_subset = utils.arrayMatchesAny(modes, mode_subset)
        assigned_probs[~mode_in_subset, :] = probs[~mode_in_subset, :]

    return assigned_probs


def gaussianSample(mean=0, std=1):
    return mean + float(np.random.randn(1)) * std


def viterbiForward(self, unary_scores, pairwise_scores, minimize=False):
    """ Forward pass of Viterbi search.

    Parameters
    ----------
    unary_scores : iterable( np.array of float, shape (num_states_t,) )
    pairwise_scores : iterable( np.array of float, shape (num_states_t, num_states_tprev) )
    minimize : bool, optional
        If True, Viterbi minimizes the score instead of maximizing.

    Returns
    -------
    final_scores : np.array of float, shape (num_states_final,)
    best_state_idxs : iterable( np.array of int, shape (num_states_t,) )
    """

    if minimize:
        unary_scores = -unary_scores
        pairwise_scores = -pairwise_scores

    # Initialize scores with zero cost
    best_state_idxs = []
    prev_max_scores = 0

    # Forward pass (max-sum)
    scores = zip(unary_scores, pairwise_scores)
    for t, scores_t in enumerate(scores):

        def totalScore(unary_score, pairwise_scores):
            return prev_max_scores + pairwise_scores + unary_score

        # Compute scores for each of the current states
        total_scores = utils.batchProcess(totalScore, zip(*scores_t))

        # For each of the current states, find the best previous state
        best_state_idxs.append(utils.batchProcess(np.argmax, total_scores))

        # Save the total score so far for each of the current states
        prev_max_scores = utils.batchProcess(np.max, total_scores)

    final_scores = prev_max_scores

    return final_scores, best_state_idxs


def viterbiBackward(final_scores, best_state_idxs):
    """ Viterbi backward pass (backtrace).

    Parameters
    ----------
    final_scores :
    best_state_idxs :

    Returns
    -------
    pred_idxs :
    """

    # Find the best final state
    pred_idxs = [final_scores.argmax()]

    # Trace back the best path from the final state
    for best_idxs in reversed(best_state_idxs):
        prev_best_idx = pred_idxs[-1]
        pred_idxs.append(best_idxs[prev_best_idx])

    # Reverse the output because we constructed it right-to-left
    pred_idxs = reversed(pred_idxs)

    return pred_idxs


def sparsifyApprox(probs, err_thresh):
    """ Sparsify a probability vector. """

    # FIXME
    # sorted_indices = None
    return probs


def sparsifyThresh(log_vec, log_thresh):
    """ Threshold-based beam pruning. """
    # num_elem = len(log_vec)
    # non_inf_entries = log_vec[~np.isinf(log_vec)]

    # Prune state space using beam search
    # Keep in mind that G is in the range [-inf, 0]
    dynamic_thresh = log_vec.max() + log_thresh
    in_beam_support = log_vec >= dynamic_thresh

    return in_beam_support


def sparsifyBest(vec, K):
    num_elem = len(vec)

    k_best_indices = np.argpartition(vec, -K)[-K:]
    in_beam_support = np.zeros(num_elem, dtype=bool)
    in_beam_support[k_best_indices] = True

    return in_beam_support


def estimateGaussianParams(X, cov_structure=None):
    """ Estimate the parameters of a normal distribution, ignoring NaN values.

    Parameters
    ----------
    X : numpy array of float, shape (num_samples, num_dims)
    cov_structure : {'diag'}, optional

    Returns
    -------
    mean : numpy array of float, shape (num_dims,)
    cov : numpy array of float, shape (num_dims,)

    Raises
    ------
    ValueError
        When the return values contain NaN entries
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice")
        mean = np.nanmean(X, axis=0)

    if cov_structure == 'diag':
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Degrees of freedom")
            cov = np.nanstd(X, axis=0) ** 2

    if np.isnan(mean).any():
        err_str = f"NaN-valued entries in mean vector: {mean}"
        logger.warning(err_str)
        # mean[np.isnan(mean)] = 0

    if np.isnan(cov).any():
        err_str = f"NaN-valued entries in cov vector: {cov}"
        logger.warning(err_str)
        # cov[np.isnan(cov)] = 0

    return mean, cov


def gaussianScore(x, mean, cov):
    standardized = (x - mean) / cov
    score = np.sum(standardized ** 2, axis=1)
    return score


def computeErrorCovariance(
        model_error_seqs, true_state_idx_seqs, is_log_likelihood=False,
        max_error_thresh=2e4):
    """
    Compute the mean model error.

    Parameters
    ----------
    model_error_seqs : iterable( numpy array of float, shape (num_samples, num_states) )
    true_state_idx_seqs : iterable( numpy array of int, shape (num_samples) )
    is_log_likelihood : bool, optional
        If True, each entry in `model_error_seqs` is actually the *negative*
        model error. This is useful if you have Gaussian log-likelihoods.
        Default is False.
    max_error_thresh : float, optional
        Model errors above this value are considered outliers and ignored.
        Default is 20,000.

    Returns
    -------
    error_covariance : float
    """

    def computeErrorCovarianceSeq(
            error_seq, true_state_idx_seq, max_error_thresh=None):
        """
        Parameters
        ----------
        error_seq : numpy array of float, shape (num_states, num_samples)
        true_state_idx_seq : numpy array of int, shape (num_samples,)
        max_error_thresh : float

        Returns
        -------
        numerator : float
        denominator : float
        """

        true_error_seq = error_seq[true_state_idx_seq, range(error_seq.shape[1])]

        is_finite = np.isfinite(true_error_seq)
        is_below_max_thresh = np.abs(true_error_seq) <= max_error_thresh
        is_inlier = np.logical_and(is_finite, is_below_max_thresh)
        true_error_seq = true_error_seq[is_inlier]

        numerator = true_error_seq.sum()
        denominator = true_error_seq.shape[0]

        return numerator, denominator

    sum_errors, num_items = utils.batchProcess(
        computeErrorCovarianceSeq,
        model_error_seqs, true_state_idx_seqs,
        static_kwargs={'max_error_thresh': max_error_thresh},
        unzip=True
    )

    error_covariance = sum(sum_errors) / sum(num_items)

    if is_log_likelihood:
        error_covariance *= -1

    return error_covariance


# -=( PROBABILITY MODEL MIXINS )=----------------------------------------------
class ExponentialDist(object):
    def __init__(self):
        self.beta = None
        self.log_beta = None

    def fit(self, x):
        self.beta = x.mean()
        self.log_beta = np.log(self.beta)

    def logl(self, x):
        return np.sum(-self.log_beta - x / self.beta)

    def likelihood(self, x):
        return np.exp(self.logl(x))


class FrameScorer(object):
    """ Mixture model for scoring video frames.

    This model is meant to extend a base mixture model like k-means or GMM
    using multiple inheritance (see `models.KMeansFrameScorer` for an example).

    This model assumes there are multiple high-level classes with different
    distributions over the same set of low-level classes. Concretely, it is used
    to distinguish between hand and blocks pixels in a distantly-supervised way.
    In this application, the base mixture model first segments the color space
    into a set of `k` distinct regions. Then, the 'hands' and 'blocks' classes
    each have their own prior distribution over the same `k` regions in color
    space.
    """

    def __init__(
            self,
            n_clusters=32, sat_thresh=0.1, ignore_low_sat=True,
            one_indexed=False, **super_kwargs):
        """

        Parameters
        ----------
        n_clusters : int, optional
            Number of shared clusters (i.e., segments in color space).
            Default is 32.
        sat_thresh : float, optional
            Minimum saturation threshold. If the average saturation of a
            color-space segment is below this value, that segment is assigned
            to a special low-saturation (background) class. Default is 0.2.
        ignore_low_sat : bool, optional
            If True, this model discards clusters with mean saturation below
            `sat_thresh` in ``self.fit`` and ``self.is_bad_cluster``.
        one_indexed : bool, optional
            If True, the color-space segments are one-indexed. Default is False.
        **super_kwargs : optional
            Any extra keyword arguments get passed to ``super().__init()__``.
        """

        if n_clusters % 2:
            err_str = 'n_clusters must be an even number!'
            raise ValueError(err_str)

        self.class_histograms = None
        self.n_clusters = n_clusters
        self.one_indexed = one_indexed
        self.sat_thresh = sat_thresh

        self.ignore_low_sat = ignore_low_sat

        super().__init__(n_clusters=n_clusters, **super_kwargs)

    def fit(self, X, Y, backoff_counts=0):
        """ Fit this model to a collection of (data, label) pairs.

        This model fits itself in two stages:
            1. (Unsupervised) Segment the colorspace by fitting a mixture model to `X`.
            2. (Supervised) For each class in `Y`, fit a categorical distribution
                :math:`P(segment | class)`.

        Parameters
        ----------
        X : numpy array of float, shape (num_pixels, 3) )
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.
        Y : iterable(int)
            Category labels in the training dataset. The i-th element in Y is
            the label for the i-th element in X. If a pixel came from the child
            dataset its label is 0, and if it came from the controlled dataset
            its labels is 1.
        backoff_counts : int or float, optional
            Each class in `Y` has its own distribution over the colorspace
            segments. Passing a nonzero value for this argument allows you to
            smooth each of those distributions by adding that amount to all bins
            in the histogram before normalizing. Non-integer backoff counts are
            allowed.
        """

        # Quantize color space into clusters
        super().fit(X)

        # Identify low-saturation clusters. These probably belong to the
        # background, which is white.
        cluster_center_sats = self.cluster_centers_[:,1]
        self.is_low_sat_cluster = cluster_center_sats < self.sat_thresh
        self.low_sat_clusters = self.is_low_sat_cluster.nonzero()[0]

        # Set class attributes
        Y = Y.astype(int)
        self.class_indices = np.unique(Y)
        self.num_classes = len(self.class_indices)

        if self.num_classes != 2:
            err_str = (
                'FrameScorer only supports 2 classes! '
                f'({self.num_classes} classes were passed)'
            )
            raise ValueError(err_str)

        # Fit class histograms---ie, each class's prior distribution over the
        # color space clusters.
        self.class_histograms = np.zeros((self.n_clusters, self.num_classes))
        for class_index in self.class_indices:
            in_class = Y == class_index
            class_samples = X[in_class, :]
            sample_clusters = self.predict(class_samples)

            if self.ignore_low_sat:
                if self.one_indexed:
                    is_low_sat = utils.arrayMatchesAny(sample_clusters - 1, self.low_sat_clusters)
                else:
                    is_low_sat = utils.arrayMatchesAny(sample_clusters, self.low_sat_clusters)
                sample_clusters = sample_clusters[~is_low_sat]

            class_histogram = self.constructHist(
                sample_clusters, normalize=True, backoff_counts=backoff_counts
            )
            self.class_histograms[:, class_index] = class_histogram

        # Infer cluster labels. Each cluster is assigned to the class with
        # highest prior probability.
        noise = self.class_histograms[:, 0]
        clean = self.class_histograms[:, 1]
        diff_hist = noise - clean
        self.is_noise_cluster = np.zeros(self.n_clusters, dtype=bool)
        self.is_noise_cluster[diff_hist > 0] = 1

        # Compute a signal-to-noise ratio for each class
        self.class_snrs = []
        for class_index in self.class_indices:
            class_histogram = self.class_histograms[:, class_index]
            class_snr = self.computeClassSnr(class_histogram)
            self.class_snrs.append(class_snr)

        # Set hsv and rgb means for convenience
        self.hsv_means = self.cluster_centers_
        hsv_mean_img = self.hsv_means.reshape(2, self.n_clusters // 2, 3)
        rgb_mean_img = imageprocessing.color.hsv2rgb(hsv_mean_img)
        self.rgb_means = rgb_mean_img.reshape(self.n_clusters, 3)

    def pixelwiseSnr(self, sample, log_domain=False, hard_assign_clusters=False):
        """ Compute a signal-to-noise ratio (SNR) for each element in the input.

        This SNR is really a likelihood ratio. If :math:`C_s` is the set of
        signal classes and :math:`C_n` is the set of noise classes,

        ..math:
            SNR(x) = \frac{
                \sum_{c \in C_s} P(x | c)
            }{
                \sum_{c' \in C_n} P(x | c')
            }

        Parameters
        ----------
        sample : numpy array of float, shape (num_pixels, 3) )
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.
        log_domain : bool, optional
            If True, :math:`\log(SNR)` is returned instead of the actual SNR.
            Default is False.
        hard_assign_clusters : bool, optional
            If True, each pixel is hard-assigned to its nearest cluster instead
            of marginalizing over cluster assignments. This means the pixel
            likelihood :math:`P(x | c)` is really class-conditional prior
            :math:`P(cluster | c)`.

        Returns
        -------
        snr : numpy array of float, shape (num_samples,)
            The sample's elementwise signal-to-noise ratio. If `log_domain` is
            True, the log of this value is returned.
        """

        if not sample.any():
            return np.array([])

        signal_class = 1
        noise_class = 0

        log_probs = self.logLikelihood(
            (noise_class, signal_class), sample,
            hard_assign_clusters=hard_assign_clusters
        )

        with warnings.catch_warnings():
            # If a pixel has probability zero under both classes, log_snr
            # tries to do infinity - infinity (= NaN) and throws a warning
            warnings.filterwarnings("ignore", "invalid value encountered")
            log_snr = np.diff(log_probs).squeeze()

        if log_domain:
            return log_snr

        return np.exp(log_snr)

    def averageSnr(self, sample, **snr_kwargs):
        """ Compute the average signal-to-noise ratio of the input.

        For documentation on the type of SNR used, see ``self.pixelwiseSnr``.
        This method ignores any pixels whose SNR is NaN, which can happen if
        they have zero probability under both models.

        Parameters
        ----------
        sample : numpy array of float, shape (num_pixels, 3) )
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.
        **snr_kwargs : optional
            Any extra keyword arguments get passed to ``self.pixelwiseSnr``.

        Returns
        -------
        avg_snr : float
            The average pixelwise SNR, ignoring NaN values.
        """

        px_snrs = self.pixelwiseSnr(sample, **snr_kwargs)

        with warnings.catch_warnings():
            # It's ok if the input is empty; don't display the warning
            warnings.filterwarnings("ignore", "Mean of empty slice")
            avg_snr = np.nanmean(px_snrs)

        return avg_snr

    def clusterPrior(self, class_index):
        """ Return a class's prior distribution over color-space segments.

        Parameters
        ----------
        class_index : int or iterable(int)

        Returns
        -------
        cluster_prior : numpt array of float, shape (num_clusters, len(class_index))
        """
        return self.class_histograms[:, class_index]

    def quantize(self, X, colorspace='rgb'):
        """ Quantize the input pixels.

        Parameters
        ----------
        X : numpy array of float, shape (num_pixels, 3)
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.
        colorspace : {'rgb', 'hsv'}, optional
            Colorspace representation of the return value. Default is ``'rgb'``.

        Returns
        -------
        quantized : numpy array of float, shape (num_pixels, 3)
            The i-th row is this this model's closest match to the i-th row of
            the input.
        """

        if not X.any():
            return np.zeros(X.shape, dtype=int)

        clusters = super().predict(X)

        if colorspace == 'rgb':
            quantized = self.rgb_means[clusters, :]
        elif colorspace == 'hsv':
            quantized = self.hsv_means[clusters, :]
        else:
            err_str = (
                f"Received bad argument colorspace={colorspace} "
                "(must be one of {'rgb', 'hsv'})"
            )
            raise ValueError(err_str)

        return quantized

    def predict(self, X):
        """ Map each pixel in the given input to its closest colorspace segment.

        Parameters
        ----------
        X : numpy array of float, shape (num_pixels, 3)
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.

        Returns
        -------
        predicted : numpy array of int, shape (num_pixels,)
            The i-th element is this model's best guess about the category
            label of the i-th row in `X`.
        """

        if not X.any():
            return np.zeros(X.shape, dtype=int)

        predicted = super().predict(X)
        if self.one_indexed:
            predicted += 1

        return predicted

    def predictClassFromCluster(self):
        """ Map each colorspace segment to the higher-level class it belongs to.

        Returns
        -------
        mapping : numpy array of int, shape (num_clusters,)
            Each entry in this array contains the higher-level class of the
            corresponding segment in colorspace. Classes are enumerated as
            follows:
                0 -- background
                1 -- skin
                2 -- low-saturation (specular or white background)
                3 -- blocks
        """

        class_idxs = {
            'background': 0,
            'skin': 1,
            'losat': 2,
            'blocks': 3
        }

        mapping = np.zeros_like(self.is_low_sat_cluster, dtype=int)
        mapping[self.is_low_sat_cluster] = class_idxs['losat']
        mapping[self.is_noise_cluster] = class_idxs['skin']

        is_signal_cluster = ~self.is_low_sat_cluster * ~self.is_noise_cluster
        mapping[is_signal_cluster] = class_idxs['blocks']

        return mapping

    def predictClass(self, X):
        """ Predict the class of each pixel in the given input.

        Parameters
        ----------
        X : numpy array of float, shape (num_pixels, 3)
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.

        Returns
        -------
        class_preds : numpy.array of int, shape (img_height, img_width)
            Each pixel contains its class label. Classes are enumerated as
            follows:
                0 -- background
                1 -- skin
                2 -- low-saturation (specular or white background)
                3 -- blocks
        """

        if not X.any():
            return np.zeros(X.shape, dtype=int)

        # First, assign each pixel to the closest segment
        cluster_preds = super().predict(X)

        # Then, assign to the best class based on the segment prediction
        cluster_to_class_map = self.predictClassFromCluster()
        class_preds = cluster_to_class_map[cluster_preds]

        return class_preds

    def logLikelihood(self, class_index, sample, hard_assign_clusters=None):
        """ Return the log probability that a sample belongs to a particular class.

        ..math:
            P(sample | class) = \sum_{cluster} P( sample | cluster ) P(cluster | class)

        Parameters
        ----------
        class_index : int or iterable(int)
            Class or classes for which the log likelihood will be computed. The
            input can be a single integer or a collection of integers.
        sample : numpy array of float, shape (num_pixels, 3) )
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.
        hard_assign_clusters : {'all', 'losat'}, optional
            If ``'all'``, each pixel is hard-assigned to its nearest cluster
            instead of marginalizing over cluster assignments. This means the
            pixel likelihood :math:`P(x | c)` is really class-conditional prior
            :math:`P(cluster | c)`. If ``'losat'``, pixels are only hard-assigned
            if their nearest cluster is in ``self.low_sat_clusters`` (ie their
            mean saturation is below a set threshold).

        Returns
        -------
        class_logprobs : numpy array of float, shape (num_samples, num_classes)
            The i-th column of `class_logprobs` is the probability that `sample`
            belongs to the class in ``class_index[i]``.
        """

        arg_set = ('all', 'losat')
        if hard_assign_clusters is not None and hard_assign_clusters not in arg_set:
            err_str = (
                f"Keyword argument hard_assign_clusters={hard_assign_clusters} "
                f"not recognized---must be one of {arg_set}"
            )
            raise ValueError(err_str)

        if type(class_index) is int:
            class_index = (class_index,)

        # We can compute probs more efficiently if all clusters are hard-assigned
        if hard_assign_clusters == 'all':
            cluster_assignments = super().predict(sample)
            with warnings.catch_warnings():
                # If there are any structural zeros in the cluster prior, log
                # will automatically throw a warning we don't want to see
                warnings.filterwarnings("ignore", "divide by zero")
                cluster_logprobs = np.log(self.clusterPrior(class_index))
            class_logprobs = cluster_logprobs[cluster_assignments, :]
            return class_logprobs

        cluster_logprobs = self.clusterLogProbs(sample)
        if hard_assign_clusters == 'losat':
            cluster_logprobs = assignToModes(
                cluster_logprobs, mode_subset=self.low_sat_clusters,
                log_domain=True
            )

        class_logprobs = tuple(
            scipy.misc.logsumexp(cluster_logprobs, axis=1, b=self.clusterPrior(i))
            for i in class_index
        )

        return np.column_stack(class_logprobs)

    def clusterScores(self, sample):
        """ Computer the parent model's score for each item in the input.

        Parameters
        ----------
        sample : numpy array of float, shape (num_pixels, 3) )
            Each row is a pixel from an image in the training dataset,
            represented in the HSV colorspace.

        Returns
        -------
        cluster_scores :
        """
        return super().transform(sample)

    def clusterProbsUnnormalized(self, sample):
        return np.exp(-self.clusterScores(sample))

    @property
    def cluster_normalizer(self):
        return (2 * np.pi) ** 0.5

    @property
    def cluster_log_normalizer(self):
        return 0.5 * np.log(2 * np.pi)

    def clusterLogProbs(self, sample):
        return -self.clusterScores(sample) - self.cluster_log_normalizer

    def clusterProbs(self, sample):
        """
        Parameters
        ----------
        sample :

        Returns
        -------
        cluster_probs :
        """
        return self.clusterProbsUnnormalized(sample) / self.cluster_normalizer

    @property
    def is_bad_cluster(self):
        ret_val = self.is_noise_cluster
        if self.ignore_low_sat:
            ret_val = np.logical_or(ret_val, self.is_low_sat_cluster)

        return ret_val

    def constructHist(self, pixel_labels, **hist_kwargs):
        """ Make a histogram representing the empirical distribution of the input.

        Parameters
        ----------
        pixel_labels : numpy array of int, shape (num_samples,)
            Array of indices identifying the closest segments in colorspace for
            a set of pixels.
        **hist_kwargs : optional
            Any extra keyword arguments are passed to `models.makeHistogram`.

        Returns
        -------
        class_histogram : numpy array of float, shape (num_clusters,)
            Cluster histogram. If `normalize` is True, this is a distribution
            over clusters. If not, this is the number of times each cluster
            was encountered in the input.
        """

        if self.one_indexed:
            # FIXME: This could be modifying the argument in-place
            pixel_labels -= 1

        class_histogram = makeHistogram(self.n_clusters, pixel_labels, **hist_kwargs)
        return class_histogram

    def computeClassSnr(self, class_histogram):
        """ Compute a signal-to-noise ratio (SNR) for a particular class.

        This SNR is really a probability ratio. If :math:`C_s` is the set of
        colorspace clusters assigned to the signal class and :math:`C_n` is the
        set of colorspace clusters assigned to the noise class,

        ..math:
            SNR = \frac{
                \sum_{c \in C_s} P(c)
            }{
                \sum_{c' \in C_n} P(c')
            }

        So this SNR compares the proportion of this class's probability mass
        that is distributed among "signal" classes vs. "noise" classes.

        Parameters
        ----------
        class_histogram : numpy array of float, shape (num_clusters,)
            A class's prior distribution over colorspace segments, ie
            :math:`P(cluster | class)`.

        Returns
        -------
        class_snr : float
            The signal-to-noise ratio of this class.
        """

        num_signal_samples = class_histogram[~self.is_bad_cluster].sum()
        num_noise_samples = class_histogram[self.is_bad_cluster].sum()

        class_snr = num_signal_samples / num_noise_samples
        return class_snr

    def predictSeq(self, pixel_seq):
        pixel_label_seq = utils.iterate(self.predict, pixel_seq, obj=tuple)
        return pixel_label_seq

    def constructHistSeq(self, pixel_seq):
        hist_tup = utils.iterate(self.constructHist, pixel_seq, obj=tuple)
        if not hist_tup:
            return np.array([])
        return np.row_stack(hist_tup)

    def computeClassSnrSeq(self, pixel_seq):
        hist_arr = self.constructHistSeq(pixel_seq)
        snr_arr = np.array([self.computeClassSnr(hist) for hist in hist_arr])
        return snr_arr

    def bestFrame(self, pixel_seq):
        snr_arr = self.computeClassSnrSeq(pixel_seq)
        best_idx = snr_arr.argmax()
        return best_idx


class EmpiricalLatentVariables(object):
    """ Variable and potential function names follow Murphy """

    def __init__(self):
        self.states = None
        self.state_index_map = None

    def getStateIndex(self, state):
        """ []

        Parameters
        ----------
        state : integer-valued symmetric numpy array

        Returns
        -------
        index : int
          Label index of the provided argument
        """

        state_str = ''.join([str(elem) for elem in state.astype(int)])

        index = self.state_index_map.get(state_str, None)
        if index is None:
            index = len(self.states)
            self.state_index_map[state_str] = index
            self.states.append(state)
        return index

    def getState(self, index):
        return self.states[index]

    def toFlattenedLabelIndexArray(self, label_seqs):
        labels = [self.getStateIndex(l) for ls in label_seqs for l in ls]
        return np.array(labels)

    def toFlattenedLabelArray(self, label_seqs):
        raise NotImplementedError

    def toLabelIndexArray(self, label_seq):
        labels = [self.getStateIndex(l) for l in label_seq]
        return np.array(labels)

    def toLabelIndexArrays(self, label_seqs):
        return utils.iterate(self.toLabelIndexArray, label_seqs)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def state_indices(self):
        return tuple(range(self.num_states))

    @property
    def num_vertex_vals(self):
        return 1

    @property
    def num_vertices(self):
        return len(self.states)

    def fit(self, state_seqs, *feat_seqs, model_segments=False, memory_len=1):
        """ Identify the support of the input's empirical distribution.

        Parameters
        ----------
        state_seqs : iterable( iterable( object ) )
            Each element is a sequence of 'state' objects. These objects can be
            almost anything---the only requirement is that they must implement
            ``__eq__()``.
        *feat_seqs : iterable(numpy array of float, shape (num_samples, num_features))
            Ignored---Exists for API compatibility.
        memory_len : int, optional
            The length of n-gram samples to record.
            For example:
                If 1, this function records all states encountered in the input.
                If 2, it records all state pairs encountered.
                Don't try anything more than two though, because it isn't
                implemented.
        """

        self.states = []
        self.state_index_map = {}

        if memory_len > 1:
            self.transitions = set()

        # Populate the state list & index map so we can deal with labels that
        # aren't naturally integer-valued (such as graph-like objects)
        for seq in state_seqs:
            if model_segments:
                seq, segment_len_seq = labels.computeSegments(seq)
            prev_state_index = -1
            for state in seq:
                state_index = self.getStateIndex(state)
                if memory_len > 1 and prev_state_index >= 0:
                    transition = (prev_state_index, state_index)
                    self.transitions.add(transition)
                prev_state_index = state_index


class EmpiricalStateVariable(EmpiricalLatentVariables):
    def getStateIndex(self, state):
        """ Override inherited method """
        try:
            state_index = self.states.index(state)
        except ValueError:
            self.states.append(state)
            state_index = self.num_states - 1
        return state_index


class DummyLikelihood(object):
    """
    Likelihood model that ignores the input.
    This is useful for creating baseline models that only use the prior to
    decode.
    """

    def __init__(self):
        super().__init__()

    def fit(self, state_seqs, data_seqs):
        super().fit(state_seqs, data_seqs)

    def computeStateLogLikelihood(self, state_idx, *data):
        return 0, None


class ImuLikelihood(object):
    """ Compute the likelihood of an IMU sample. """

    def __init__(self, num_imu_vars=1):
        super().__init__()

        self.num_imu_vars = num_imu_vars
        self.lms = tuple(
            (ExponentialDist(), ExponentialDist())
            for i in range(self.num_imu_vars))

    def fit(self, state_seqs, *all_corr_seqs):
        super().fit(state_seqs, *all_corr_seqs)

        for i, corr_seqs in enumerate(all_corr_seqs):
            data = countMagCorrSeqs(corr_seqs, state_seqs)
            for dist, sample in zip(self.lms[i], data):
                dist.fit(sample)

    def computeStateLogLikelihood(
            self, state_idx, *all_corr_samples):
        state = self.states[state_idx]

        logl = 0
        for i, corr_sample in enumerate(all_corr_samples):
            data = countMagCorrs(corr_sample, state)
            if len(data) != len(self.lms[i]):
                err_str = 'Number of data categories does not match number of models'
                raise ValueError(err_str)

            for samples, lm in zip(data, self.lms[i]):
                logl += lm.logl(samples)

        return logl, None


class ImageLikelihood(object):
    """ Compute the likelihood of an image. """

    def __init__(self, structured=True, pixel_classifier=None):
        """
        Parameters
        ----------
        structured : bool, optional
            This flag selects between two image likelihood models.
            If ``structured == True``, the image model incorporates global image
            structure using template rendering and registration.
            If ``structured == False``, the image model assumes image pixels are
            independent and uses a Gaussian mixture model.
            The default value is False.
        pixel_classifier : ???, optional
            This is an optional model that detects (and scores) nuisance pixels.
        """

        super().__init__()
        self.debug = False

        self.pixel_classifier = pixel_classifier

        self.num_seqs = None
        self.cur_seq_idx = None

        self.structured = structured

        if self.structured:
            self.computeStateLogLikelihood = self.structuredStateLogl
        else:
            self.base_gmm = initBaseGmm(self.block_colors)
            self.template_priors = {}
            self.computeStateLogLikelihood = self.unstructuredStateLogl

        self.templates = {}

    def fit(self, label_seqs, *feat_seqs, model_responses=None, err_cov=None, model_segments=False):
        """ Estimate the parameters of an `ImageLikelihood` model from data.

        Parameters
        ----------
        label_seqs : iterable( numpy array of int, shape (n_samples,) )
            Each item in `label_seqs` is a sequence of labels for each of the
            corresponding items in `feat_seqs`.
        *feat_seqs : iterable( iterable( numpy array of float, shape (n_samples, n_dim) ) )
            Feature sequences. Each item in `feat_seqs` is a separate set of
            feature sequences.
        model_responses : iterable( numpy array of float, shape (n_samples, n_states) ), optional
            This model's responses to `feat_seqs`, cached from a previous run.
            If this argument is provided, `fit` skips evaluating `feat_seqs`
            estimates its parameters using `model_responses` instead.
        err_cov : float, optional
            The model's error covariance, :math:`sigma^2`.
        """

        super().fit(label_seqs, *feat_seqs)

        """
        if err_cov is None:
            # Compute model responses for the correct labels
            if model_responses is None:
                feat_seqs = zip(*feat_seqs)
                model_responses = tuple(
                    utils.batchProcess(
                        self.computeLogLikelihoods,
                        *feat_seq_tup,
                        state_idx=label_seq
                    )
                    for feat_seq_tup, label_seq in zip(feat_seqs, label_seqs)
                )
            err_cov = computeErrorCovariance(
                model_responses, label_seqs,
                is_log_likelihood=True,
                max_error_thresh=2e4
            )

        self.error_covariance = err_cov
        """

    def structuredStateLogl(
            self, rgb_image, depth_image, segment_image,
            background_plane, state_idx=None, state=None,
            # include_hand_pixel_logprob=False,
            **fitscene_kwargs):
        """
        Score an observed image against a hypothesis state.

        Parameters
        ----------
        rgb_foreground_image : numpy array of float, shape (img_height, img_width, 3)
        depth_foreground_image : numpy array of float, shape (img_height, img_width)
        label_img : numpy array of int, shape (img_height, img_width)
            DEPRECATED FOR NOW
            This array gives a class assignment for each pixel in `observed`.
            Classes are from the following set:
                0 -- Hands
                1 -- Blocks
                2 -- Specularity
        segment_img : numpy array of int, shape (img_height, img_width)
            This array gives a segment assignment for each pixel in `observed`.
            Segments are (mostly) contiguous regions of the image.
        background_plane : geometry.Plane
        state_idx : int, optional
            Index of the hypothesis state to score. One of `state_idx` and `state`
            must be passed to this method. If `state_idx` is not passed, this
            method will retrieve the index of `state`.
        state : blockassembly.BlockAssembly, optional
            The hypothesis state to score. One of `state_idx` and `state` must
            be passed to this method. If `state` is not passed, this method will
            retrieve the state corresponding to `state_idx`.
        include_hand_pixel_logprob : bool, optional
            DEPRECATED FOR NOW
            If True, add in the log-probability score from this object's pixel
            classifier. Default is False.
        **fitscene_kwargs : optional
            Any extra keyword arguments get passed to `models.fitScene`.

        Returns
        -------
        log_likelihoods : numpy array of float, shape (len(state_idxs),)
            (Proportional to ) log likelihood score for each hypothesis in `state_idxs`
        argmaxima : iterable( ??? )
            DEPRECATED FOR NOW
            arg-maximizers of any sub-models that were optimized to produce
            `log_likelihoods`. The i-th element if `argmaxima` corresponds to
            the i-th element of `log_likelihoods`.
        component_poses : tuple( (np array of float, shape (3, 3), np array of float, shape(3,)) )
            Collection of (R, t) pairs specifying the orientation and position
            of each sub-part in the spatial assembly.
        """

        if state is None:
            state = self.getState(state_idx)

        if state_idx is None:
            state_idx = self.getStateIndex(state)

        error, component_poses, rendered_images = fitScene(
            rgb_image, depth_image, segment_image,
            state, background_plane,
            camera_params=render.intrinsic_matrix,
            camera_pose=render.camera_pose,
            block_colors=render.object_colors,
            **fitscene_kwargs
        )

        # is_hand = label_img == 1
        # if include_hand_pixel_logprob and is_hand.any():
        #     sample = rgb_foreground_image[is_hand]
        #     logprobs = self.pixel_classifier.logLikelihood(0, sample)
        #     state_ll += np.sum(logprobs)

        # state_ll /= self.err_cov

        # return state_ll, argmaxima
        return -error, component_poses

    def unstructuredStateLogl(
            self, observed, pixel_classes, segment_labels, background_model,
            state_idx=None, state=None, viz_pose=False, is_depth=False,
            obsv_std=1, greedy_assignment=False, **optim_kwargs):
        """
        Score an observed image against a hypothesis state.

        Parameters
        ----------
        observed : numpy array of float, shape (img_height, img_width)
        pixel_classes : numpy array of int, shape (img_height, img_width)
            This array gives a class assignment for each pixel in `observed`.
            Classes are from the following set:
                0 -- Hands
                1 -- Blocks
                2 -- Specularity
        segment_labels : numpy array of int, shape (img_height, img_width)
            This array gives a segment assignment for each pixel in `observed`.
            Segments are (mostly) contiguous regions of the image.
        background_model :
        state_idx : int, optional
            Index of the hypothesis state to score. One of `state_idx` and `state`
            must be passed to this method. If `state_idx` is not passed, this
            method will retrieve the index of `state`.
        state : blockassembly.BlockAssembly, optional
            The hypothesis state to score. One of `state_idx` and `state` must
            be passed to this method. If `state` is not passed, this method will
            retrieve the state corresponding to `state_idx`.
        viz_pose : bool, optional
        is_depth : bool, optional
        obsv_std : bool, optional
        greedy_assignment : bool, optional
        **optim_kwargs : optional
            Additional keyword arguments that get passed to
            `self.computeStateLogLikelihood`.

        Returns
        -------
        log_likelihoods : numpy array of float, shape (len(state_idxs),)
            Log likelihood score for each hypothesis in `state_idxs`
        argmaxima : iterable( ??? )
            arg-maximizers of any sub-models that were optimized to produce
            `log_likelihoods`. The i-th element if `argmaxima` corresponds to
            the i-th element of `log_likelihoods`.
        """

        if state is None:
            state = self.states[state_idx]

        if state_idx is None:
            state_idx = self.getStateIndex(state)

        if not state.connected_components:
            state.connected_components[0] = set()

        background_plane_img = None
        if is_depth:
            background_plane_img = render.renderPlane(background_model, observed.shape)
            background_plane_img /= obsv_std
            observed = observed.astype(float) / obsv_std
            observed[segment_labels == 0] = background_plane_img[segment_labels == 0]

        bboxes = imageprocessing.segmentBoundingBoxes(observed, segment_labels)
        num_segments = segment_labels.max()

        if viz_pose:
            obsv_bboxes = observed.copy()

        num_components = len(state.connected_components)
        nll_upper_bound = 1e14
        OBJECTIVES = np.full((num_components, num_segments), nll_upper_bound)
        NEG_LOGLS = np.full((num_components, num_segments), nll_upper_bound)

        visibility_ratio_lower_bound = 0.35

        TS = []
        THETAS = []
        TEMPLATES = []
        for comp_arr_idx, comp_idx in enumerate(state.connected_components.keys()):
            template_model = self.makeTemplateGmm(state_idx, comp_idx)

            if viz_pose:
                plt.figure()
                plt.stem(template_model.weights_)
                plt.title(f'GMM prior, state {state_idx} component {comp_idx}')
                plt.ylabel('p(color)')
                plt.xlabel('color')
                plt.show()

            # TEMPLATES.append(rendered)
            if viz_pose:
                rgb_rendered = self.getCanonicalTemplate(state_idx, comp_idx, img_type='rgb')
                TEMPLATES.append(rgb_rendered)

            TS.append([])
            THETAS.append([])
            for seg_idx in range(1, num_segments + 1):
                bbox = bboxes[seg_idx - 1]
                t, __ = geometry.extremePoints(*bbox)

                TS[-1].append(t)
                THETAS[-1].append(0)

                in_seg = segment_labels == seg_idx
                is_blocks = pixel_classes == 3
                is_blocks_in_seg = in_seg & is_blocks

                visibility_ratio = is_blocks_in_seg.sum() / in_seg.sum()
                prune_segment = visibility_ratio < visibility_ratio_lower_bound
                # logger.info(f'visibility ratio: {visibility_ratio:.2f}')

                if not prune_segment:
                    seg_pixels = observed[in_seg, :]
                    avg_log_prob = template_model.score(seg_pixels)
                    log_prob = template_model.score_samples(seg_pixels).sum()
                    OBJECTIVES[comp_arr_idx, seg_idx - 1] = -avg_log_prob
                    NEG_LOGLS[comp_arr_idx, seg_idx - 1] = -log_prob

                if viz_pose:
                    perimeter = imageprocessing.rectangle_perimeter(*bbox)
                    obsv_bboxes[perimeter] = 1
                    if prune_segment:
                        obsv_bboxes[in_seg] = 1

        final_obj, argmaxima = matchComponentsToSegments(
            OBJECTIVES, TS, THETAS,
            # downstream_objectives=NEG_LOGLS,
            greedy_assignment=greedy_assignment
        )

        if viz_pose:
            final_render = render.makeFinalRender(
                TEMPLATES, observed, argmaxima[0], argmaxima[1],
                copy_observed=True)

            imageprocessing.displayImages(obsv_bboxes, final_render, pixel_classes, segment_labels)
            if is_depth:
                f, axes = plt.subplots(2, 1, figsize=(16, 8))
                # axes[0].hist(residual_img.ravel(), bins=100)
                axes[1].hist(final_render.ravel(), bins=100)
                plt.show()

        return -final_obj, argmaxima

    def computeLogLikelihoods(
            self, rgb_image, depth_image, segment_image, background_plane,
            state_idxs=None, viz_logls=False, **fitscene_kwargs):
        """
        Score an observed image against a set of hypothesis states.

        Parameters
        ----------
        rgb_image : numpy array of float, shape (img_height, img_width, 3)
        depth_image : numpy array of float, shape (img_height, img_width)
        segment_image : numpy array of int, shape (img_height, img_width)
            This array gives a segment assignment for each pixel in `observed`.
            Segments are (mostly) contiguous regions of the image.
        background_plane : geometry.Plane
        state_idxs : iterable(int), optional
            Indices of the hypothesis states to score. By default this method
            will score all the states in the model's vocabulary.
        viz_logls : bool, optional
        **fitscene_kwargs : optional
            Additional keyword arguments that get passed to `models.fitScene`.

        Returns
        -------
        log_likelihoods : numpy array of float, shape (len(state_idxs),)
            Log likelihood score for each hypothesis in `state_idxs`
        argmaxima : iterable( ??? )
            arg-maximizers of any sub-models that were optimized to produce
            `log_likelihoods`. The i-th element if `argmaxima` corresponds to
            the i-th element of `log_likelihoods`.
        """

        if state_idxs is None:
            state_idxs = range(self.num_states)

        logls, argmaxima = utils.batchProcess(
            self.computeStateLogLikelihood,
            state_idxs,
            static_args=(rgb_image, depth_image, segment_image, background_plane),
            static_kwargs=fitscene_kwargs,
            unzip=True
        )

        if viz_logls:
            plt.figure()
            plt.stem(state_idxs, logls)
            plt.xlabel('State indices')
            plt.ylabel('Log probability')
            plt.title('Frame log likelihoods')
            plt.show()

        return np.array(logls), argmaxima

    def computeLikelihoods(self, *feat_seqs, **kwargs):
        """ Calls `self.computeLogLikelihoods` and exponentiates to produce likelihoods.

        See `self.computeLogLikelihoods`.
        """

        logl, argmaxima = self.computeLogLikelihoods(*feat_seqs, **kwargs)
        likelihood = np.exp(logl)

        return likelihood, argmaxima

    def getCanonicalTemplate(self, state_idx, comp_idx, img_type='label'):
        """
        Retrieve an image of a spatial assembly component in a canonical pose.

        If the template is already cached in ``self.templates``, return it.
        If not, render the template, cache it, and return it.

        Parameters
        ----------
        state_idx : int
            Index of the spatial assembly state.
        comp_idx : int
            The componenent's index in its spatial assembly.
        img_type : str, optional
            Type of template to retrieve. Can be one of `label`, `rgb`, or
            `depth`. Default is `label`.

        Returns
        -------
        template : numpy array of float, shape (template_height, template_width)
        """

        rendered = self.templates.get((state_idx, comp_idx, img_type), None)
        if rendered is not None:
            return rendered

        state = self.getState(state_idx)
        template = self.renderCanonicalTemplate(state, comp_idx, img_type=img_type)
        self.templates[(state_idx, comp_idx, img_type)] = template

        return template

    def makeTemplateGmm(self, state_idx, comp_idx):
        """
        Create an appearance model for a component of an assembly state.

        The appearance model is a Gaussian mixture model, and assumes pixels
        are independent (so it is unstructured). Each Gaussian component in the
        GMM corresponds to an object in the connected component of the spatial
        assembly. This method renders a label image and computes a histogram
        to estimate each object's prior probability in the GMM.

        Parameters
        ----------
        state_idx : int
            Index of the spatial assembly state.
        comp_idx : int
            The componenent's index in its spatial assembly.

        Returns
        -------
        gmm : sklearn.mixture.GaussianMixture
            GMM ppearance model. It has `num_objects + 1` components, whose
            means are the colors of the background and each object in the
            spatial assembly. Covariances are all the identity matrix.
            Component priors are histograms computed from a template image
            rendered by this object.
        """

        key = (state_idx, comp_idx)
        prior = self.template_priors.get(key, None)
        if prior is None:
            num_components = self.base_gmm.means_.shape[0]
            rendered = self.getCanonicalTemplate(state_idx, comp_idx, img_type='label')
            prior = makeHistogram(
                num_components, rendered,
                ignore_empty=False, normalize=True
            )

        # FIXME: return a copy (GaussianMixture doesn't have a copy method)
        gmm = self.base_gmm
        gmm.weights_ = prior
        return gmm

    def predictSample(self, state_logls, state_idxs=None):
        """
        Return the assembly state (and its state index) corresponding to the
        arg-max of `state_logls`.

        Parameters
        ----------
        state_logls : numpy array, shape (num_states)
        state_idxs : iterable(int), optional

        Returns
        -------
        best_state : blockassembly.BlockAssembly
        best_state_idx : int
        """

        if state_idxs is None:
            state_idxs = tuple(range(self.num_states))

        best_idx = state_logls.argmax()
        best_state_idx = state_idxs[best_idx]
        best_state = self.getState(best_state_idx)

        return best_state, best_state_idx

    def visualize(self, base_path):
        """
        Vizualize this model's state space.

        This method saves visualizations of each state in the model's state
        space in a directory named ``states``.

        Parameters
        ----------
        base_path : str
            Path to the directory that visualizations should be saved in.
            ``states`` will be created in this directory.
        """

        fig_path = os.path.join(base_path, 'states')
        os.makedirs(fig_path)
        for i, state in enumerate(self.states):
            state.draw(fig_path, i)

    def visualizePredictionInfo(self, rgb_seq, state_index_seq, pred_args):
        """
        Visualize model predictions using matplotlib.

        Parameters
        ----------
        rgb_seq : iterable( numpy array, shape (img_height, img_width) )
            Observed image from which the predictions were made
        state_index_seq : iterable( int )
            Predicted assembly state indices.
        pred_args : iterable( ??? )
            Values of any nuisance arguments that were maximized along with the
            assembly state values. Right now this means the pose of each
            component in the assembly state.
        """

        # observed_eq = len(rgb_seq) == len(depth_seq)
        cross_eq = len(rgb_seq) == len(state_index_seq)
        if not cross_eq:
            err_str = 'Arg sequences are not all of equal length'
            raise ValueError(err_str)

        N = len(rgb_seq)
        for i in range(N):
            rgb_image = img_as_float(rgb_seq[i])
            # depth_image = depth_seq[i]
            state_index = state_index_seq[i]
            predicted_arg = pred_args[i][state_index]

            # displayImages(rgb_image, depth_image)
            resid_img = rgb_image - predicted_arg
            imageprocessing.displayImages(predicted_arg, np.abs(resid_img))


class MultimodalLikelihood(ImuLikelihood, ImageLikelihood):
    """ Container class wrapping IMU and Image likelihood models. """

    def __init__(self, **imu_kwargs):
        ImuLikelihood.__init__(self, **imu_kwargs)
        ImageLikelihood.__init__(self)

    def fit(self, state_seqs, *feat_seqs):
        imu_seqs = feat_seqs[:self.num_imu_vars]
        image_seqs = feat_seqs[self.num_imu_vars:]

        ImuLikelihood.fit(self, state_seqs, *imu_seqs)
        ImageLikelihood.fit(self, state_seqs, *image_seqs)

    def computeStateLogLikelihood(
            self, state_idx, *samples, imu_weight=1, image_weight=1,
            **optim_kwargs):

        imu_samples = samples[:self.num_imu_vars]
        image_samples = samples[self.num_imu_vars:]

        imu_ll, __ = ImuLikelihood.computeStateLogLikelihood(self, state_idx, *imu_samples)
        # self.logger.info(f'IMU LOGL: {imu_ll}')

        image_ll, argmaxima = ImageLikelihood.computeStateLogLikelihood(
            self, state_idx, *image_samples, **optim_kwargs)
        # self.logger.info(f'IMAGE LOGL: {image_ll}')

        logl = imu_weight * imu_ll + image_weight * image_ll
        return logl, argmaxima


class Hmm(object):
    """ Timeseries model with a discrete, Markov-distributed latent state """

    def __init__(self, **likelihood_kwargs):
        super().__init__(**likelihood_kwargs)
        self.psi = None
        self.log_psi = None

        self.sparse_uniform_transitions_ = None
        self.uniform_transitions_ = None

        self.cur_seq_idx = None

    @property
    def sparse_uniform_transitions(self):
        if self.sparse_uniform_transitions_ is None:
            self.sparse_uniform_transitions_ = self.makeUniformTransitions(sparse=True)
        return self.sparse_uniform_transitions_

    @property
    def uniform_transitions(self):
        if self.uniform_transitions_ is None:
            self.uniform_transitions_ = self.makeUniformTransitions(sparse=False)
        return self.uniform_transitions_

    def makeUniformTransitions(self, sparse=False):
        transition_probs = np.ones_like(self.psi)
        if sparse:
            transition_probs[self.psi == 0] = 0
        transition_probs /= transition_probs.sum(axis=1)
        return transition_probs

    def fit(self, label_seqs, *feat_seqs,
            uniform_regularizer=0, diag_regularizer=0, empty_regularizer=0,
            zero_transition_regularizer=0, model_segments=False,
            estimate_final_state_scores=False,
            override_transitions=False,
            **super_kwargs):
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

        super().fit(label_seqs, *feat_seqs, model_segments=model_segments, **super_kwargs)

        psi = np.zeros((self.num_states, self.num_states))
        psi += uniform_regularizer
        for i in range(self.num_states):
            psi[i, i] += diag_regularizer

        psi[0, 0] += empty_regularizer
        psi[:, 0] += zero_transition_regularizer
        psi[0, :] += zero_transition_regularizer

        unigram_counts = np.zeros(self.num_states)

        if model_segments:
            segment_lens = collections.defaultdict(list)
        else:
            segment_lens = None

        for l in label_seqs:
            seq_bigram_counts, seq_unigram_counts = self.fitSeq(l, segment_lens=segment_lens)
            psi += seq_bigram_counts
            unigram_counts += seq_unigram_counts

        if estimate_final_state_scores:
            final_states = np.array([self.getStateIndex(label_seq[-1]) for label_seq in label_seqs])
            self.final_state_probs = makeHistogram(self.num_states, final_states, normalize=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "divide by zero encountered")
                self.final_state_scores = np.log(self.final_state_probs)
            if np.any(np.isnan(self.final_state_scores)):
                raise ValueError()
        else:
            self.final_state_probs = np.ones(self.num_states)
            self.final_state_scores = np.zeros(self.num_states)

        plt.figure()
        plt.stem(self.final_state_probs)

        if segment_lens is not None:
            self.max_duration = max(itertools.chain(*segment_lens.values()))
            self.durations = np.arange(self.max_duration)
            self.duration_probs = np.array(
                tuple(
                    makeHistogram(self.max_duration, segment_lens[state_idx], normalize=True)
                    for state_idx in range(self.num_states)
                )
            )
            plt.matshow(self.duration_probs)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "divide by zero encountered")
                self.duration_scores = np.log(self.duration_probs)
            state_has_no_counts = ~self.duration_probs.any(axis=1)
            if state_has_no_counts.any():
                err_str = (
                    f"States {state_has_no_counts.nonzero()[0]} "
                    "have zero counts for every duration!"
                )
                raise ValueError(err_str)

        if override_transitions:
            logger.info('Overriding pairwise counts with array of all ones')
            psi = np.ones_like(psi)

        # Normalize rows of transition matrix
        for index in range(self.num_states):
            # Define 0 / 0 as 0
            if not psi[index, :].any():  # and not unigram_counts[index]:
                continue
            # psi[index,:] /= unigram_counts[index]
            psi[index, :] /= psi[index, :].sum()

        self.psi = psi
        plt.matshow(self.psi)
        plt.show()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero")
            self.log_psi = np.log(psi)

    def fitSeq(self, label_seq, segment_lens=None):
        """ Count max-likelihood transition parameters for a single sequence.

        Parameters
        ----------
        label_seq : iterable( object )
            Sequence of label states---not state integer IDs.
        segment_lens : collections.defaultdict(list), optional
            If provided, this dictionary is updated with the duration of each
            segment in the label sequence.

        Returns
        -------
        pairwise_counts : numpy array of int, shape (self.num_states, self.num_states)
        unigram_counts : numpy array of int, shape (self.num_states,)
        """

        label_idx_seq = self.toLabelIndexArray(label_seq)

        # Convert sample labels to segment labels and compute segment duration
        # counts if segment_lens was passed.
        if segment_lens is not None:
            label_idx_seq, segment_len_seq = labels.computeSegments(label_idx_seq)
            for state_id, segment_len in zip(label_idx_seq, segment_len_seq):
                segment_lens[state_id].append(segment_len)

        # Compute unigram counts
        unigram_counts = np.zeros(self.num_states)
        for index in label_idx_seq:
            unigram_counts[index] += 1

        # Compute bigram counts
        pairwise_counts = np.zeros((self.num_states, self.num_states))
        for prev_index, cur_index in zip(label_idx_seq[:-1], label_idx_seq[1:]):
            pairwise_counts[prev_index, cur_index] += 1

        return pairwise_counts, unigram_counts

    def forward(self, local_evidence):
        """
        HMM forward algorithm for computing data likelihood.

        NOTE: This implementation follows the pseudocode in Murphy Ch. 17.4.2
            (Algorithm 17.1)

        Parameters
        ----------
        local_evidence :

        Returns
        -------
        alpha :
        Z :
        """

        def normalize(u):
            u_sum = u.sum()
            if u.any():
                u /= u_sum
            return u, u_sum

        # initialization
        num_states, num_samples = local_evidence.shape
        alpha = np.zeros_like(local_evidence)
        Z = np.zeros(num_samples)

        # We always start in state 0, so pi is the standard basis vector e_0
        # NOTE: pi is the marginal distribution on the state at time t = 1,
        #   p(z_1 = j)
        pi = np.zeros(num_states)
        pi[0] = 1

        alpha[:, 0], Z[0] = normalize(local_evidence[:, 0] * pi)
        for t in range(1, num_samples):
            alpha[:, t], Z[t] = normalize(
                local_evidence[:, t] * (self.psi.T @ alpha[:, t - 1]))

        return alpha, Z.sum()

    def backward(self, local_evidence):
        """
        HMM backward algorithm for computing data likelihood.

        NOTE: This implementation follows the pseudocode in Murphy Ch. 17.4.2
            (Algorithm 17.1)

        Parameters
        ----------
        local_evidence :

        Returns
        -------
        beta :
        """

        num_states, num_samples = local_evidence.shape
        beta = np.zeros_like(local_evidence)

        beta[:, -1] = 1
        for t in reversed(range(1, num_samples)):
            beta[:, t - 1] = self.psi @ (local_evidence[:, t] * beta[:, t])

        return beta

    def prune(
            self, prev_max_lps, transition_probs,
            greed_coeff=None, sparsity_level=None,
            transition_thresh=0, verbose=0):
        """
        Produces a set of predictions (hypotheses) for time t, given a set of
        scores for the model's predictions at time t-1 and a transition model.

        Parameters
        ----------
        prev_max_lps : numpy.ndarray, shape (NUM_STATES,)
            Scores for the model's predictions at time t-1.
        transition_probs : numpy.ndarray, shape (NUM_STATES, NUM_STATES)
            Pairwise scores. Usually interpreted as state-to-state transition
            probabilities.
        greed_coeff : float, optional
            Controls the "greediness" of the pruner by thresholding against the
            highest-scoring hypothesis. Must be in the range [0, infinity].
            If 0, nothing is pruned. If infinity, everything but the one-best
            hypothesis is pruned.
        sparsity_level : int, optional
            Controls the "greediness" of the pruner by keeping a k-best list.
            Let `sparsity_level = k`. Everything but the k highest-scoring
            hypotheses is pruned.
        transition_thresh : float, optional
            Threshold for sparsifying pairwise scores. Any state transition with
            score less than `transition_thresh` is pruned.
        verbose : bool, optional
            If True, sends a message summarizing the pruning operation to
            logger.info.

        Returns
        -------
        beam_image : list(int)
            New predictions (hypothesis set). Each element is the index of a
            hypothesized state.
        """

        if greed_coeff is not None and sparsity_level is not None:
            err_str = 'Only one of (greed_coeff, sparsity_level) may be passed!'
            raise ValueError(err_str)

        transition_prob_nonzero = transition_probs > transition_thresh

        if greed_coeff is not None:
            in_beam_support = sparsifyThresh(prev_max_lps, greed_coeff)
        elif sparsity_level is not None:
            in_beam_support = sparsifyBest(prev_max_lps, sparsity_level)
        else:
            in_beam_support = np.ones(self.num_states, dtype=bool)

        in_beam_image = transition_prob_nonzero.T @ in_beam_support
        # in_beam_image = in_beam_support
        if not in_beam_image.any():
            err_str = 'Viterbi beam is empty!'
            raise ValueError(err_str)

        beam_image = np.where(in_beam_image)[0]

        if verbose:
            support_size = in_beam_support.sum()
            image_size = in_beam_image.sum()
            debug_str = f'Beam size {support_size:2} -> {image_size:2}'
            self.logger.info(debug_str)

        return beam_image

    def transitionProbs(self, ml_decode=None, sparse_ml_decode=None):
        """ Return pairwise scores for a decode run.

        By default, this method returns the model's pairwise parameters.

        Parameters:
        -----------
        ml_decode : bool, optional
            If True, this method emulates a maximum-likelihood decode by
            ignoring the prior distribution over states. In other words, it
            returns uniform score arrays instead of the model parameters.
        sparse_ml_decode: bool, optional
            If True, this method returns score arrays that are uniform over the
            domain of the model's pairwise parameters (ie any state transition
            that has zero probability in the model's parameters also has zero
            probability in these scores).

        Returns
        -------
        transition_probs : numpy.ndarray, shape (NUM_STATES, NUM_STATES)
            Pairwise potentials (probabities) for states.
        log_transition_probs : numpy.ndarray, shape (NUM_STATES, NUM_STATES)
            Pairwise scores (log probabilities) for states.
        """

        if ml_decode:
            transition_probs = self.uniform_transitions
            log_transition_probs = np.log(transition_probs)
        elif sparse_ml_decode:
            transition_probs = self.sparse_uniform_transitions
            log_transition_probs = np.log(transition_probs)
        else:
            transition_probs = self.psi
            log_transition_probs = self.log_psi

        return transition_probs, log_transition_probs

    def viterbi(
            self, *samples, prior=None,
            greed_coeff=None, sparsity_level=None, verbose_level=False,
            transition_thresh=0, ml_decode=False, sparse_ml_decode=False,
            state_log_likelihoods=None, edge_logls=None,
            score_samples_as_batch=False, no_state_logl=False,
            **ll_kwargs):
        """ Viterbi search with threshold and sparsity-based beam pruning.

        Parameters
        ----------
        samples :
        prior :
        greed_coeff :
        sparsity_level :
        verbose_level :
        transition_thresh :
        ml_decode :
        sparse_ml_decode :
        log_likelihoods :
        edge_logls :
        ll_kwargs :

        Returns
        -------
        pred_idxs :
        max_log_probs :
        log-likelihoods :
        final_argmaxima :
        """

        transition_probs, log_transition_probs = self.transitionProbs(
            ml_decode=ml_decode,
            sparse_ml_decode=sparse_ml_decode
        )

        if prior is None:
            prior = np.zeros(self.num_states)
            prior[0] = 1

        # You need to call this before unzipping samples below
        if score_samples_as_batch:
            edge_logls = self.computeLogLikelihoods(*samples, as_array=True)

        # Convert tuple of sequences to sequence of tuples for easier
        # iteration
        samples = tuple(zip(*samples))

        # Initialization---you need to call this AFTER unzipping samples above
        all_argmaxima = {}
        num_samples = len(samples)
        array_dims = (self.num_states, num_samples)
        max_log_probs = np.full(array_dims, -np.inf)
        best_state_idxs = np.full(array_dims, np.nan, dtype=int)

        # Forward pass (max-sum)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero")
            prev_max_lps = np.log(prior)

        for sample_idx, sample in enumerate(samples):

            if sample_idx == 0:
                state_lps = prev_max_lps
                max_log_probs[:, sample_idx] = state_lps
                prev_max_lps = max_log_probs[:, sample_idx]
                all_argmaxima[0, sample_idx] = tuple()
                continue

            # Prune hypotheses
            beam_image = self.prune(
                prev_max_lps, transition_probs,
                greed_coeff=greed_coeff, sparsity_level=sparsity_level,
                transition_thresh=transition_thresh,
                verbose=verbose_level
            )

            # Compute likelihoods / data scores
            if no_state_logl:
                state_logls = None
            elif state_log_likelihoods is None:
                state_logls, argmaxima = self.computeLogLikelihoods(
                    *sample, **ll_kwargs, state_idxs=beam_image
                )
                state_log_likelihoods = np.full(array_dims, -np.inf)
            else:
                state_logls = state_log_likelihoods[beam_image, sample_idx]
                argmaxima = [None] * len(beam_image)

            # Add in state transition scores
            for i, state_idx in enumerate(beam_image):
                state_lps = (prev_max_lps + log_transition_probs[:, state_idx])

                if state_logls is not None:
                    state_logl = state_logls[i]
                    argm = argmaxima[i]
                    state_lps += state_logl
                    state_log_likelihoods[state_idx, sample_idx] = state_logl

                if edge_logls is not None:
                    state_lps += edge_logls[sample_idx, :, state_idx]
                    argm = None

                max_log_probs[state_idx, sample_idx] = state_lps.max()
                best_state_idxs[state_idx, sample_idx] = state_lps.argmax()
                all_argmaxima[state_idx, sample_idx] = argm
            prev_max_lps = max_log_probs[:, sample_idx]

        plt.stem(state_logls)
        plt.matshow(max_log_probs)
        plt.show()

        # Backward pass (backtrace)
        pred_idxs = np.zeros(num_samples, dtype=int)
        pred_idxs[-1] = max_log_probs[:, -1].argmax()
        for sample_idx in reversed(range(1, num_samples)):
            prev_best_state = pred_idxs[sample_idx]
            best_state_idx = best_state_idxs[prev_best_state, sample_idx]
            pred_idxs[sample_idx - 1] = best_state_idx

        final_argmaxima = tuple(
            all_argmaxima[state_idx, sample_idx]
            for sample_idx, state_idx in enumerate(pred_idxs)
        )

        return pred_idxs, max_log_probs, state_log_likelihoods, final_argmaxima

    def viterbiSegmental(
            self, *samples, prior=None,
            greed_coeff=None, sparsity_level=None, verbose_level=False,
            transition_thresh=0, ml_decode=False, sparse_ml_decode=False,
            state_log_likelihoods=None, edge_logls=None,
            score_samples_as_batch=False, no_state_logl=False,
            **ll_kwargs):
        """ Viterbi search with threshold and sparsity-based beam pruning.

        Parameters
        ----------
        samples :
        prior :
        greed_coeff :
        sparsity_level :
        verbose_level :
        transition_thresh :
        ml_decode :
        sparse_ml_decode :
        log_likelihoods :
        edge_logls :
        ll_kwargs :

        Returns
        -------
        pred_idxs :
        max_log_probs :
        log-likelihoods :
        final_argmaxima :
        """

        transition_probs, log_transition_probs = self.transitionProbs(
            ml_decode=ml_decode,
            sparse_ml_decode=sparse_ml_decode
        )

        if prior is None:
            prior = np.zeros(self.num_states)
            prior[0] = 1

        # You need to call this before unzipping samples below
        if score_samples_as_batch:
            edge_logls = self.computeLogLikelihoods(*samples, as_array=True)
            for sample in edge_logls:
                plt.matshow(sample)
                plt.show()

        # Convert tuple of sequences to sequence of tuples for easier
        # iteration
        samples = tuple(zip(*samples))

        # Initialization---you need to call this AFTER unzipping samples above
        all_argmaxima = {}
        num_samples = len(samples)
        array_dims = (self.num_states, num_samples)
        best_scores = np.full(array_dims, -np.inf)
        # Seed packpointers so we can decode the first segment when backtracing
        backpointers = {}  # (0, 0): (0, -1)}

        if state_log_likelihoods is None and not no_state_logl:
            state_log_likelihoods = np.full(array_dims, -np.inf)

        # Forward pass (max-sum)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero")
            log_prior = np.log(prior)
            prev_best_scores = log_prior

        for sample_idx, sample in enumerate(samples):
            # if sample_idx == 0:
            #     state_lps = prev_best_scores
            #     best_scores[:, sample_idx] = state_lps
            #     prev_best_scores = best_scores[:, sample_idx]
            #     all_argmaxima[0, sample_idx] = tuple()
            #     continue

            # Prune hypotheses
            beam_image = self.prune(
                prev_best_scores, transition_probs,
                greed_coeff=greed_coeff, sparsity_level=sparsity_level,
                transition_thresh=transition_thresh,
                verbose=verbose_level
            )

            # logger.info(f"Sample {sample_idx}: {beam_image}")

            # Compute likelihoods / data scores
            if no_state_logl:
                state_logls = None
            elif state_logls is None:
                state_logls, argmaxima = self.computeLogLikelihoods(
                    *sample, **ll_kwargs, state_idxs=beam_image
                )
            else:
                state_logls = state_log_likelihoods[beam_image, sample_idx]
                argmaxima = [None] * len(beam_image)

            for i, seg_idx in enumerate(beam_image):
                score_table = np.full(
                    (self.duration_scores.shape[0], self.duration_scores.shape[1]),
                    -np.inf
                )
                for j, prev_seg_idx in enumerate(self.state_indices):
                    duration_prob_is_pos = self.duration_probs[seg_idx, :] > 0
                    segment_begins_in_seq = (sample_idx - self.durations) + 1 >= 0
                    duration_candidate_idxs = np.nonzero(
                        duration_prob_is_pos * segment_begins_in_seq
                    )[0]
                    duration_candidates = self.durations[duration_candidate_idxs]
                    # logger.info(f"duration candidates: {duration_candidates}")
                    duration_scores = self.duration_scores[seg_idx, duration_candidates]
                    # if not duration_candidates.any():
                    #     raise ValueError()
                    # for duration, dur_score in enumerate(self.duration_scores[prev_seg_idx,:])
                    for duration, dur_score in zip(duration_candidates, duration_scores):
                        prev_seg_end_idx = sample_idx - duration
                        seg_start_idx = prev_seg_end_idx + 1
                        if seg_start_idx < 0:
                            raise ValueError()

                        if sample_idx == -1:
                            prev_seg_prior = log_prior[seg_idx]
                            seg_score = prev_seg_prior
                        else:
                            prev_seg_prior = best_scores[prev_seg_idx, prev_seg_end_idx]
                            seg_transition_score = log_transition_probs[prev_seg_idx, seg_idx]
                            seg_score = prev_seg_prior + seg_transition_score
                        seg_score += dur_score

                        # Compute data score for this segment
                        seg = slice(seg_start_idx, sample_idx + 1)

                        # FIXME: hasn't been properly converted to segmental decode
                        if state_logls is not None:
                            state_logl = state_logls[seg]
                            argm = argmaxima[i]
                            # state_lps += state_logl
                            state_log_likelihoods[seg_idx, sample_idx] = state_logl

                        if edge_logls is not None and prev_seg_idx >= 0:
                            seg_score += edge_logls[seg, prev_seg_idx, seg_idx].sum()
                            argm = None

                        score_table[j, duration] = seg_score
                # if sample_idx == 1:
                # plt.matshow(score_table.T)
                # plt.show()
                # assert(True == False)
                j, best_duration = utils.argmaxNd(score_table)
                prev_best_seg = self.state_indices[j]
                prev_best_seg_end_idx = sample_idx - best_duration
                backpointers[seg_idx, sample_idx] = (prev_best_seg, prev_best_seg_end_idx)
                best_scores[seg_idx, sample_idx] = score_table[j, best_duration]
                all_argmaxima[seg_idx, sample_idx] = argm
            prev_best_scores = best_scores[:, sample_idx]

        plt.matshow(best_scores.T)
        plt.show()
        # Add in final state scores to eliminate sequences that aren't accepted
        # by this machine
        best_scores[:, -1] += self.final_state_scores

        # Backward pass (backtrace)
        best_seg = best_scores[:, -1].argmax()
        best_seg_end_idx = num_samples - 1
        pred_idxs = np.full(num_samples, -1, dtype=int)
        sample_idx = best_seg_end_idx
        # logger.info(f"backpointer: {best_seg}, @ {best_seg_end_idx}")
        while sample_idx > -1:
            prev_best_seg, prev_best_seg_end_idx = backpointers[best_seg, sample_idx]
            best_seg_start_idx = prev_best_seg_end_idx + 1
            pred_idxs[best_seg_start_idx:best_seg_end_idx + 1] = best_seg

            best_seg = prev_best_seg
            best_seg_end_idx = prev_best_seg_end_idx
            sample_idx = best_seg_end_idx
            # logger.info(f"backpointer: {best_seg}, @ {best_seg_end_idx}")

        sample_was_skipped = pred_idxs < 0
        if np.any(sample_was_skipped):
            err_str = f"skipped {sample_was_skipped.sum()} samples when backtracing"
            raise ValueError(err_str)

        final_argmaxima = tuple(
            all_argmaxima[state_idx, sample_idx]
            for sample_idx, state_idx in enumerate(pred_idxs)
        )

        return pred_idxs, best_scores, state_log_likelihoods, final_argmaxima

    def mpeDecode(self, logl, **kwargs):
        """
        Marginal posterior estimation using the forward-backward algorithm.

        Parameters
        ----------
        logl :
        kwargs :

        Returns
        -------
        predicted_idxs :
        """

        local_evidence = np.exp(logl)

        # Forward, backward passes
        alpha, __ = self.forward(local_evidence)
        beta = self.backward(local_evidence)

        # Compute gamma, proportional to the marginal posterior up to a
        # dividing term which is constant in z (so we can just argmax over
        # gamma)
        gamma = alpha * beta
        predicted_idxs = gamma.argmax(axis=0)

        return predicted_idxs

    def predict(self, *feat_seqs, **predict_kwargs):
        self.num_seqs = len(feat_seqs[0])
        self.cur_seq_idx = 0

        f = functools.partial(self.predictSeq, **predict_kwargs)
        prediction_tups = utils.iterate(f, *feat_seqs)

        pred_state_seqs, pred_idx_seqs, unary_info = tuple(zip(*prediction_tups))

        return pred_state_seqs, pred_idx_seqs, unary_info

    def predictSeq(
            self, *feat_seqs, decode_method='MAP', viz_predictions=False,
            **kwargs):

        if decode_method == 'MPE':
            decode = self.mpeDecode
        elif decode_method == 'MAP':
            decode = self.viterbi
        elif decode_method == 'ML':
            decode = functools.partial(self.viterbi, ml_decode=True)
        elif decode_method == 'MLSPARSE':
            decode = functools.partial(self.viterbi, sparse_ml_decode=True)
        elif decode_method == 'segmental':
            decode = self.viterbiSegmental
        else:
            err_str = (
                'Invalid argument "decode_method={}" '
                'decode_method must be one of: '
                'MPE (marginal posterior decoding) '
                'MAP (viterbi decoding) '
                'ML (maximum likelihood decoding)'
            )
            raise ValueError(err_str.format(decode_method))

        if self.cur_seq_idx is not None:
            fmt_str = 'Decoding sequence {} / {}'
            self.logger.info(fmt_str.format(self.cur_seq_idx + 1, self.num_seqs))

        # logl, argmaxima = self.computeLogUnaryPotential(
        #         *feat_seqs, **kwargs)
        # pred_states, pred_idxs = self.predictSeqFromUnary(
        #         *feat_seqs, decode_method=decode_method)
        # if viz_predictions:
        #     self.visualizePredictionInfo(*feat_seqs, pred_idxs, argmaxima)
        # unary_info = (logl, argmaxima)

        pred_idxs, log_probs, log_likelihoods, argmaxima = decode(*feat_seqs, **kwargs)
        pred_states = [self.getState(i) for i in pred_idxs]

        if self.cur_seq_idx is not None:
            self.cur_seq_idx += 1

        return pred_states, pred_idxs, log_probs, log_likelihoods, argmaxima

    def predictSeqFromUnary(self, logl, decode_method='MAP'):
        if decode_method == 'MPE':
            decode = self.mpeDecode
        elif decode_method == 'MAP':
            decode = self.mapDecode
        elif decode_method == 'ML':
            decode = self.mlDecode
        else:
            err_str = (
                'Invalid argument "decode_method={}" '
                'decode_method must be one of: '
                'MPE (marginal posterior decoding) '
                'MAP (viterbi decoding) '
                'ML (maximum likelihood decoding)')
            raise ValueError(err_str.format(decode_method))

        pred_idxs = decode(logl)
        pred_states = [self.getState(i) for i in pred_idxs]

        return pred_states, pred_idxs

    def predictFromUnary(self, logl_seqs, **predict_kwargs):
        f = functools.partial(self.predictSeqFromUnary, **predict_kwargs)
        prediction_tups = utils.iterate(f, logl_seqs)

        pred_state_seqs, pred_idx_seqs = tuple(zip(*prediction_tups))
        return pred_state_seqs, pred_idx_seqs

    def visualize(self, base_path):
        super().visualize(base_path)
        fn = os.path.join(base_path, 'transition-probs.png')
        label = 'Transition probabilities'
        utils.plotArray(self.psi, fn, label)


class Crf(object):
    """
    Straightforward application of some models in Lea et al.
    (ECCV, ICRA)
    """

    def __init__(self, base_model='ChainModel', **base_kwargs):
        super().__init__()
        # self.model = getattr(lctm_models, base_model)(**base_kwargs)

    def visualize(self, base_path):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        def plotLoss(model, base_path):
            fn = os.path.join(base_path, 'objective.png')

            iterate_idxs = np.array(list(model.logger.objectives.keys()))
            iterate_vals = np.array(list(model.logger.objectives.values()))

            plt.figure()
            plt.plot(iterate_idxs, iterate_vals)
            plt.title('Objective function')
            plt.ylabel('f(i)')
            plt.xlabel('i')
            plt.savefig(fn)
            plt.close()

        def plotWeights(model, base_path):
            for name, weights in model.ws.items():
                fn = os.path.join(base_path, 'weights-{}.png'.format(name))
                plt.matshow(weights)
                plt.colorbar(orientation='horizontal', pad=0.1)
                plt.title(name)
                plt.savefig(fn)
                plt.close()

        plotLoss(self.model, base_path)
        plotWeights(self.model, base_path)

    def fit(self, label_seqs, *feat_seqs, **kwargs):
        super().fit(label_seqs, *feat_seqs)

        labels = self.toLabelIndexArrays(label_seqs)

        self.preprocessor = preprocessing.StandardScaler()
        self.preprocessor.fit(utils.toFlattenedFeatureArray(*feat_seqs))

        features = utils.toFeatureArrays(*feat_seqs)
        features = [self.preprocessor.transform(f) for f in features]

        self.model.fit(utils.transposeSeq(features), labels, **kwargs)

    def predictSeq(self, *feat_seqs):
        features = utils.toFeatureArray(*feat_seqs)
        features = self.preprocessor.transform(features)

        predicted_idxs = self.model.predict(features.T)
        predicted_seq = [self.getState(index) for index in predicted_idxs]

        return predicted_seq

    def computeUnaryPotential(self, *feat_seqs):
        raise NotImplementedError

    def computeLogUnaryPotential(self, *feat_seqs):
        features = utils.toFeatureArray(*feat_seqs)
        features = self.preprocessor.transform(features)

        num_samples = features.shape[0]
        score = np.zeros([self.num_states, num_samples], np.float64)
        score = self.model.potentials['unary'].compute(
            self.model, features.T, score)

        return score


class BaseModel(object):
    """ [] """

    def localEvidence(self, *feat_seqs):
        return utils.iterate(self.computeLogUnaryPotential, *feat_seqs)

    def predict(self, *feat_seqs, **predict_kwargs):
        f = functools.partial(self.predictSeq, **predict_kwargs)
        return utils.iterate(f, *feat_seqs)

    def fit(self, label_seqs, *feat_seqs, **kwargs):
        raise NotImplementedError

    def visualize(self, base_path):
        raise NotImplementedError


# -=( COMPOSITE PROBABILITIC MODELS )=-----------------------------------------
class KMeansFrameScorer(FrameScorer, KMeans, LoggerMixin):
    pass


class MiniBatchKMeansFrameScorer(FrameScorer, MiniBatchKMeans, LoggerMixin):
    pass


class EmpiricalImageLikelihood(ImageLikelihood, EmpiricalStateVariable, LoggerMixin):
    pass


class EmpiricalImageHmm(Hmm, ImageLikelihood, EmpiricalStateVariable, LoggerMixin):
    pass


class EmpiricalImuHmm(Hmm, ImuLikelihood, EmpiricalStateVariable, LoggerMixin):
    pass


class EmpiricalMultimodalHmm(Hmm, MultimodalLikelihood, EmpiricalStateVariable, LoggerMixin):
    pass


class EmpiricalDummyHmm(Hmm, DummyLikelihood, EmpiricalStateVariable, LoggerMixin):
    pass


class HandDetectionHmm(Hmm, HandDetectionLikelihood, EmpiricalStateVariable, LoggerMixin):
    pass


class EmpiricalCrf(Crf, EmpiricalLatentVariables, LoggerMixin):
    pass


# -=( VISUALIZATION )----------------------------------------------------------
def plotModel(model, base_path, label):
    path = os.path.join(base_path, 'models')
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, label)
    model.visualize(model_path)


def plotLocalEvidenceSeqs(le_seqs, trial_ids, base_path, label):
    f = functools.partial(plotLocalEvidenceSeq, base_path=base_path, label=label)
    utils.evaluate(f, le_seqs, trial_ids)


def plotLocalEvidenceSeq(le_seq, trial_id, base_path, label):
    path = os.path.join(base_path, '{}'.format(trial_id))
    if not os.path.exists(path):
        os.makedirs(path)
    fn = os.path.join(path, '{}_trial-{:03d}.png'.format(label, trial_id))
    utils.plotArray(le_seq, fn, label)


def visualizeKeyframeModel(model):
    """ Draw a figure visualizing a keyframe model.

    Parameters
    ----------
    model : FrameScorer
    """

    noise_hist = model.class_histograms[:,0]
    clean_hist = model.class_histograms[:,1]
    indices = np.arange(model.n_clusters)
    snrs = model.class_snrs

    is_noise_cluster = model.is_bad_cluster
    rgb_means = model.rgb_means

    f, axes = plt.subplots(2, figsize=(10, 10))

    axes[0].bar(
        indices[~is_noise_cluster],
        clean_hist[~is_noise_cluster],
        color=rgb_means[~is_noise_cluster,:],
        edgecolor='k'
    )
    axes[0].set_title('Clean data')

    axes[0].bar(
        indices[is_noise_cluster],
        clean_hist[is_noise_cluster],
        color=rgb_means[is_noise_cluster,:],
        alpha=0.5
    )

    axes[1].bar(
        indices[~is_noise_cluster],
        noise_hist[~is_noise_cluster],
        color=rgb_means[~is_noise_cluster,:],
        alpha=0.5
    )

    axes[1].bar(
        indices[is_noise_cluster],
        noise_hist[is_noise_cluster],
        color=rgb_means[is_noise_cluster,:],
        edgecolor='k'
    )
    axes[1].set_title('Noisy data')

    plt.tight_layout()
    plt.show()

    logger.info(f'Noise SNR: {snrs[0]:.2f}')
    logger.info(f'Clean SNR: {snrs[1]:.2f}')


# -=( IMAGE MODEL HELPER FUNCTIONS )-------------------------------------------
def initBaseGmm(block_colors):
    """
    Partially-initialize a GMM appearance model.

    Parameters
    ----------
    block_colors : shape (1 + num_objects, 3)
        Numpy array whose rows represent the colors of each object in a scene.
        The first row (i.e. row index zero) corresponds to the background. Each
        subsequent row corresponds to an object in the spatial assembly.
        These block colors serve as a rudimentary appearance model.

    Returns
    -------
    gmm : sklearn.mixture.GaussianMixture
        Partially-initialized Gaussian mixture model. It has `num_objects + 1`
        components, whose means are the corresponding rows of `block_colors`
        and whose covariances are all the identity matrix. The component priors
        are all zero, and must be initialized later.
    """

    # class_means = block_colors
    # num_nonzero_components, num_dims = block_colors.shape
    # zero = np.zeros(num_dims)
    # class_means = np.vstack((zero, block_colors))
    class_means = block_colors
    num_components, num_dims = class_means.shape

    cov = np.eye(num_dims)
    prec = np.linalg.inv(cov)

    gmm = GaussianMixture(
        n_components=num_components,
        # weights_init=class_priors,
        means_init=class_means,
        precisions_init=prec,
        covariance_type='tied'
    )
    gmm._initialize(np.zeros((0, num_dims)), np.zeros((0, num_components)))
    gmm.covariances_ = cov

    return gmm


def registerTemplateToSegment(
        observed, rendered, seg_bounding_box, rendered_center, Tr, Tc,
        background_img=None, **optim_kwargs):
    """

    Parameters
    ----------

    Returns
    -------
    """

    V = geometry.sampleInteriorUniform(*seg_bounding_box)
    max_min = geometry.extremePoints(*seg_bounding_box)

    residual = functools.partial(
        templateMatchingResidual, observed, rendered, V,
        background_img=background_img
    )

    jacobian = functools.partial(templateMatchingJacobian, observed, rendered, Tr, Tc, V)

    bound = functools.partial(translationBoundConstraints, rendered_center, *max_min)

    t, theta, __ = optimizePose(residual, jacobian, bound, **optim_kwargs)

    return t, theta


def residual(x_true, x_est, true_mask=None, est_mask=None):
    if true_mask is not None:
        x_true[true_mask] = 0

    if est_mask is not None:
        x_est[est_mask] = 0

    resid = x_true - x_est
    return resid


def standardize(x, bias=0, scale=1):
    x_standardized = (x - bias) / scale
    return x_standardized


def mse(x_true, x_est, true_mask=None, est_mask=None, bias=None, scale=None):
    x_true = standardize(x_true, bias=bias, scale=scale)
    x_est = standardize(x_est, bias=bias, scale=scale)

    resid = residual(x_true, x_est, true_mask=true_mask, est_mask=est_mask)
    mse = np.mean(resid ** 2)

    return mse


def sse(x_true, x_est, true_mask=None, est_mask=None, bias=None, scale=None):
    x_true = standardize(x_true, bias=bias, scale=scale)
    x_est = standardize(x_est, bias=bias, scale=scale)

    resid = residual(x_true, x_est, true_mask=true_mask, est_mask=est_mask)
    sse = np.sum(resid ** 2)

    return sse


def refineComponentPose(
        rgb_image, depth_image, segment_image, assembly,
        rgb_background=None, depth_background=None, label_background=None,
        component_index=None, init_pose=None, theta_samples=None,
        camera_params=None, camera_pose=None, block_colors=None,
        object_mask=None, W=None, error_func=None, bias=None, scale=None):
    """ Refine a component's initial pose estimate using a simple registration routine.

    Parameters
    ----------

    Returns
    -------
    best_error : float
    best_pose : (R, t)
    """

    if error_func is None:
        error_func = sse

    if W is None:
        W = np.ones(2)

    if theta_samples is None:
        theta_samples = range(0, 360, 90)

    R_init, t_init = init_pose
    pose_candidates = tuple(
        (geometry.rotationMatrix(z_angle=theta, x_angle=0) @ R_init, t_init)
        for theta in theta_samples
    )

    rgb_renders, depth_renders, label_renders = utils.batchProcess(
        render.renderComponent, pose_candidates,
        static_args=(assembly, component_index),
        static_kwargs={
            'camera_pose': camera_pose,
            'camera_params': camera_params,
            'block_colors': block_colors,
            'rgb_image': rgb_background,
            'range_image': depth_background,
            'label_image': label_background,
            'in_place': False
        },
        unzip=True
    )

    # Subtract background from all depth images. This gives distances relative
    # to the background plane instead of the camera, so RGB and depth models
    # are closer to the same scale.
    depth_renders = tuple(d - depth_background for d in depth_renders)
    depth_image = depth_image - depth_background

    # imageprocessing.displayImages(*rgb_renders, *depth_renders, *label_renders, num_rows=3)
    object_background_mask = ~object_mask
    label_background_masks = tuple(label_render == 0 for label_render in label_renders)
    rgb_errors = np.array([
        error_func(
            rgb_image, rgb_render,
            true_mask=object_background_mask, est_mask=label_mask,
            bias=bias[0], scale=scale[0]
        )
        for rgb_render, label_mask in zip(rgb_renders, label_background_masks)
    ])
    depth_errors = np.array([
        error_func(
            depth_image, depth_render,
            true_mask=object_background_mask, est_mask=label_mask,
            bias=bias[1], scale=scale[1]
        )
        for depth_render, label_mask in zip(depth_renders, label_background_masks)
    ])

    errors = np.column_stack((rgb_errors, depth_errors)) @ W

    best_idx = errors.argmin()
    best_error = errors[best_idx]
    best_pose = pose_candidates[best_idx]

    return best_error, best_pose


def fitScene(
        rgb_image, depth_image, segment_image,
        assembly, background_plane,
        camera_params=None, camera_pose=None, block_colors=None,
        W=None, error_func=None, bias=None, scale=None,
        ignore_background=False):
    """ Fit a spatial assembly and a background plane to an RGBD image.

    Parameters
    ----------

    Returns
    -------
    """

    # Make copies so we don't accidentally modify the input
    rgb_image = rgb_image.copy()
    segment_image = segment_image.copy()
    depth_image = depth_image.copy()

    if error_func is None:
        error_func = sse

    if W is None:
        W = np.ones(2)

    rgb_background, depth_background, label_background = render.renderPlane(
        background_plane, camera_pose, camera_params,
        plane_appearance=block_colors[0, :],
    )

    # Estimate initial poses from each detected image segment
    segment_labels = np.unique(segment_image[segment_image != 0])
    object_masks = tuple(segment_image == i for i in segment_labels)
    object_poses_est = utils.batchProcess(
        imageprocessing.estimateSegmentPose,
        object_masks,
        static_args=(camera_params, camera_pose, depth_image),
        static_kwargs={'estimate_orientation': False}
    )

    num_components = len(assembly.connected_components)
    num_segments = len(segment_labels)

    # Find the best pose for each component of the spatial assembly, assuming
    # we try to match it to a particular segment.
    errors = np.zeros((num_components, num_segments))
    poses = {}
    for component_index, component_key in enumerate(assembly.connected_components.keys()):
        for segment_index in range(num_segments):
            object_mask = object_masks[segment_index]
            init_pose = object_poses_est[segment_index]
            error, pose = refineComponentPose(
                rgb_image, depth_image, segment_image, assembly,
                rgb_background=rgb_background,
                depth_background=depth_background,
                label_background=label_background,
                component_index=component_key, init_pose=init_pose,
                camera_params=camera_params, camera_pose=camera_pose,
                block_colors=block_colors,
                object_mask=object_mask, W=W, error_func=error_func,
                bias=bias, scale=scale
            )
            errors[component_index, segment_index] = error
            poses[component_index, segment_index] = pose

    # Match components to segments by solving the linear sum assignment problem
    # (ie data association)
    _, component_poses, _ = matchComponentsToSegments(
        errors, poses, downstream_objectives=None, greedy_assignment=False
    )

    # Render the complete final scene
    rgb_render, depth_render, label_render = render.renderScene(
        background_plane, assembly, component_poses,
        camera_pose=camera_pose,
        camera_params=camera_params,
        object_appearances=block_colors
    )

    # Subtract background from all depth images. This gives distances relative
    # to the background plane instead of the camera, so RGB and depth models
    # are closer to the same scale.
    depth_render = depth_render - depth_background
    depth_image = depth_image - depth_background

    # f, axes = plt.subplots(2, figsize=(10, 6), sharex=True)
    # axes[0].hist(depth_image[depth_image != 0], bins=100)
    # axes[0].set_ylabel('observed')
    # axes[1].hist(depth_render[depth_render != 0], bins=100)
    # axes[1].set_ylabel('rendered')
    # plt.tight_layout()
    # plt.show()

    # Compute the total error of the final scene
    image_background = segment_image == 0
    render_background = label_render == 0

    rgb_error = error_func(
        rgb_image, rgb_render,
        bias=bias[0], scale=scale[0],
        true_mask=image_background, est_mask=render_background
    )
    depth_error = error_func(
        depth_image, depth_render,
        bias=bias[1], scale=scale[1],
        true_mask=image_background, est_mask=render_background
    )
    error = np.array([rgb_error, depth_error]) @ W

    # imageprocessing.displayImages(
    #     rgb_image, depth_image, segment_image,
    #     rgb_render, depth_render, label_render,
    #     num_rows=2
    # )

    return error, component_poses, (rgb_render, depth_render, label_render)


def matchComponentsToSegments(
        objectives, poses, downstream_objectives=None, greedy_assignment=False):
    """

    Parameters
    ----------
    objectives : numpy array of float, shape (num_components, num_segments)
    poses :
    downstream_objectives : optional
    greedy_assignment : bool, optional

    Returns
    -------
    final_obj :
    best_poses :
    best_seg_idxs :
    """

    if not objectives.any():
        return None, (tuple(), tuple(), tuple()), tuple()

    # linear_sum_assignment can't take an infinty-valued matrix, so set those
    # greater than the max. That way they'll never be chosen by the routine.
    non_inf_max = objectives[~np.isinf(objectives)].max()
    objectives[np.isinf(objectives)] = non_inf_max + 1

    num_components, num_segments = objectives.shape
    if greedy_assignment:
        row_ind = np.arange(num_components)
        col_ind = objectives.argmin(axis=1)
    else:
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(objectives)

    if downstream_objectives is None:
        final_obj = objectives[row_ind, col_ind].sum()
    else:
        final_obj = downstream_objectives[row_ind, col_ind].sum()

    best_poses = [poses[r, c] for r, c in zip(row_ind, col_ind)]
    best_seg_idxs = [c + 1 for c in col_ind]

    num_unassigned_components = num_components - num_segments
    for i in range(num_unassigned_components):
        R = geometry.rotationMatrix(z_angle=0, x_angle=0)
        t = np.zeros(3) + i * np.array([75, 0, 0])
        best_poses.append((R, t))
        best_seg_idxs.append(-1)

    # argmaxima = (theta_best, t_best, seg_best)
    if downstream_objectives is None:
        return final_obj, best_poses, best_seg_idxs


# -=( PROBABLY GOING TO BE DEPRECATED )=---------------------------------------
def initModel(model_name, **model_kwargs):
    model = getattr(this_module, model_name)
    return model(**model_kwargs)


def train(model, labels, *feat_seqs, to_array=False, **train_kwargs):
    """
    """
    # feat_seqs = tuple(utils.transposeSeq(fs)
    #                  for fs in feat_seqs if fs is not None)

    if to_array:
        if labels is not None:
            err_str = 'Not yet implemented for supervised models!'
            raise ValueError(err_str)
        feat_array = utils.toFlattenedFeatureArray(*feat_seqs)
        model.fit(feat_array, **train_kwargs)
        return model

    model.fit(labels, *feat_seqs, **train_kwargs)

    return model


def predict(model, *feat_seqs, to_array=False, **decode_kwargs):
    """
    """

    # feat_seqs = tuple(
    #         utils.transposeSeq(fs) for fs in feat_seqs if fs is not None)

    if to_array:
        feat_array = utils.toFlattenedFeatureArray(*feat_seqs)
        pred_seqs = model.predict(feat_array, **decode_kwargs)
        return pred_seqs, None

    pred_seqs, pred_idxs, unary_info = model.predict(
        *feat_seqs, **decode_kwargs)

    return pred_seqs, unary_info


class CliqueLatentVariables(object):
    """ [] """

    def __init__(self):
        self.num_vertices = None
        self.num_edges = None

    def getStateIndex(self, state):
        """ []

        Parameters
        ----------
        state : binary-valued numpy vector
          []

        Returns
        -------
        index : int
          Label index of the provided argument
        """
        return utils.boolarray2int(state)

    def getState(self, index):
        return utils.int2boolarray(index, self.num_vertices)

    def toFlattenedLabelIndexArray(self, label_seqs):
        labels = [self.getStateIndex(l) for ls in label_seqs for l in ls]
        return np.array(labels)

    def toFlattenedLabelArray(self, label_seqs):
        """
        Returns
        -------
        labels : np array, size [n_features, n_classes]
          []
        """

        labels = tuple(np.squeeze(l) for ls in label_seqs for l in ls)
        return np.row_stack(labels)

    def toLabelIndexArray(self, label_seq):
        labels = [self.getStateIndex(l) for l in label_seq]
        return np.array(labels)

    def toLabelIndexArrays(self, label_seqs):
        return utils.iterate(self.toLabelIndexArray, label_seqs)

    @property
    def num_states(self):
        return self.num_vertex_vals ** self.num_vertices

    @property
    def num_vertex_vals(self):
        return 2

    @property
    def num_edge_vals(self):
        return 2

    def fit(self, state_seqs, *feat_seqs):
        """ []

        Parameters
        ----------
        state_seqs : list of list of numpy array of shape [n_edges, n_edges]
          []
        *feat_seqs : list of numpy array of shape [n_samples, n_dim]
          []

        NOTE: state_seqs can actually contain anything iterable over the
          number of samples. If it contains 2D numpy arrays, the array rows
          must represent samples because numpy iterates over rows by default.
        """

        self.num_vertices = state_seqs[0][0].shape[0]
        self.num_edges = self.num_vertices * (self.num_vertices - 1) // 2


def countMagCorrs(corr, state, on=None, off=None):
    if on is None:
        on = []
    if off is None:
        off = []

    edges = state.symmetrized_connections
    idx_tups = itertools.combinations(range(8), 2)
    for idx, (i, j) in enumerate(idx_tups):
        if edges[i,j]:
            on.append(corr[idx])
        else:
            off.append(corr[idx])
    return off, on


def countMagCorrSeqs(corr_seqs, state_seqs):
    on = []
    off = []
    for cseq, sseq in zip(corr_seqs, state_seqs):
        for c, s in zip(cseq, sseq):
            _ = countMagCorrs(c, s, on=on, off=off)
    return np.array(off), np.array(on)


def stateLogLikelihood(
        camera_params, camera_pose, block_colors,
        observed, pixel_classes, segment_labels, state,
        background_model,
        viz_pose=False, hue_only=False,
        mask_left_side=False, mask_bottom=False,
        depth_image=None,
        is_depth=False,
        obsv_std=1,
        greedy_assignment=False,
        normalize_logl=False,
        render_segment_template=False,
        **optim_kwargs):
    """
    Compute the best pose and its corresponding score.

    Specifically,
        best_pose = argmax_{pose} P( observed, pose | state )
        best_score = log P( observed, best_pose | state )

    Parameters
    ----------
    camera_params :
    camera_pose :
    block_colors :
    observed :
    pixel_classes :
    segment_labels :
    state :
    background_model :
    viz_pose :
    hue_only :
    mask_left_side :
    mask_bottom :
    is_depth :
    obsv_std :
    greedy_assignment :
    normalize_logl :
    render_segment_template : bool, optional
        If True, a template is rendered individually for each segment in
        `segment_labels`. If False, a single template is used for all segments.
        Default is False.

    Returns
    -------
    final_logprob :
    argmaxima :
    """

    # Make copies so we don't modify the input
    observed = observed.copy()
    pixel_classes = pixel_classes.copy()
    segment_labels = segment_labels.copy()
    if depth_image is not None:
        depth_image = depth_image.copy()

    if hue_only:
        # Colorspace conversion transforms can throw a divide-by-zero warning
        # for images with zero-valued pixels, but we don't want to see them.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "divide by zero")
            observed_hsv = imageprocessing.color.rgb2hsv(observed)
        observed = imageprocessing.hueImgAsRgb(observed_hsv)
        observed[segment_labels == 0] = 0

    if mask_left_side or mask_bottom:
        mask = np.zeros(pixel_classes.shape, dtype=bool)
        mask = imageprocessing.maskOutsideBuildArea(
            mask, mask_left_side=mask_left_side, mask_bottom=mask_bottom
        )
        observed[mask] = 0
        pixel_classes[mask] = 0
        segment_labels[mask] = 0

    background_plane_img = None
    if is_depth:
        background_plane_img = render.renderPlane(background_model, observed.shape)
        background_plane_img /= obsv_std
        observed = observed.astype(float) / obsv_std
        observed[segment_labels == 0] = background_plane_img[segment_labels == 0]

    bboxes = imageprocessing.segmentBoundingBoxes(observed, segment_labels)
    seg_arr_idxs = tuple(range(1, segment_labels.max() + 1))

    if viz_pose:
        obsv_bboxes = observed.copy()

    num_components = len(state.connected_components)
    num_segments = len(bboxes)
    OBJECTIVES = np.zeros((num_components, num_segments))

    unoccluded_pixels = pixel_classes != 1

    TS = []
    THETAS = []
    TEMPLATES = []

    if is_depth:
        final_render = background_plane_img.copy()
        img_type = 'depth'
    else:
        final_render = np.zeros_like(observed)
        img_type = 'rgb'

    for comp_arr_idx, comp_idx in enumerate(state.connected_components.keys()):
        # Render a template
        if not render_segment_template:
            rendered = render.renderComponent(
                state, comp_idx, img_type=img_type,
                camera_pose=camera_pose, camera_params=camera_params,
                block_colors=block_colors, obsv_std=obsv_std
            )
            TEMPLATES.append(rendered)

            # Compute the template's gradient and center
            rendered_grad_rows, rendered_grad_cols = imageprocessing.imgGradient(
                rendered, sigma=1
            )
            rendered_center = imageprocessing.imageMidpoint(rendered.shape[0:2])

        TS.append([])
        THETAS.append([])
        TEMPLATES.append([])

        # Register the component template to each object proposal
        for seg_arr_idx, seg_bounding_box in zip(seg_arr_idxs, bboxes):
            if np.sum(segment_labels == seg_arr_idx) < 2:
                TEMPLATES[-1].append(None)
                TS[-1].append(None)
                THETAS[-1].append(None)
                OBJECTIVES[comp_arr_idx, seg_arr_idx - 1] = np.inf
                continue
            if render_segment_template:
                pose_est = imageprocessing.estimateSegmentPose(
                    camera_params, camera_pose, depth_image, segment_labels == seg_arr_idx
                )
                rendered = render.renderComponent(
                    state, comp_idx, img_type=img_type,
                    component_pose=pose_est,
                    camera_pose=camera_pose, camera_params=camera_params,
                    block_colors=block_colors, obsv_std=obsv_std
                )
                # Compute the template's gradient and center
                rendered_grad_rows, rendered_grad_cols = imageprocessing.imgGradient(
                    rendered, sigma=1
                )
                rendered_center = imageprocessing.imageMidpoint(rendered.shape[0:2])

            if not rendered.any():
                TEMPLATES[-1].append(None)
                TS[-1].append(None)
                THETAS[-1].append(None)
                OBJECTIVES[comp_arr_idx, seg_arr_idx - 1] = np.inf
                continue

            TEMPLATES[-1].append(rendered)

            t, theta = registerTemplateToSegment(
                observed, rendered, seg_bounding_box, rendered_center,
                rendered_grad_rows, rendered_grad_cols,
                background_img=background_plane_img, **optim_kwargs
            )
            TS[-1].append(t)
            THETAS[-1].append(theta)

            obj = sse(
                observed, rendered, theta, t, background_plane_img=None,
                is_depth=False, unoccluded_pixels=None
            )
            OBJECTIVES[comp_arr_idx, seg_arr_idx - 1] = obj

            if viz_pose:
                perimeter = imageprocessing.rectangle_perimeter(*seg_bounding_box)
                obsv_bboxes[perimeter] = 1

    final_obj, argmaxima, TEMPLATES = matchComponentsToSegments(
        OBJECTIVES, TS, THETAS, TEMPLATES,
        greedy_assignment=greedy_assignment
    )

    final_render = render.makeFinalRender(
        TEMPLATES, observed, argmaxima[0], argmaxima[1],
        is_depth=is_depth, background_plane_img=background_plane_img
    )

    # FIXME: Turn this into a function call
    residual_img = observed - final_render
    final_obj = np.sum(residual_img[unoccluded_pixels] ** 2)

    if normalize_logl:
        num_samples = unoccluded_pixels.sum()
        final_obj /= num_samples

    if viz_pose:
        imageprocessing.displayImages(obsv_bboxes, np.abs(residual_img), final_render)
        if is_depth:
            f, axes = plt.subplots(2, 1, figsize=(16, 8))
            axes[0].hist(residual_img.ravel(), bins=100)
            axes[1].hist(final_render.ravel(), bins=100)
            plt.show()

    return -final_obj, argmaxima


def templateMatchingResidual(observed, rendered, V, x, theta=None, background_img=None):
    """

    Parameters
    ----------

    Returns
    -------
    """

    if theta is None:
        theta = int(x[:1])
        t = x[1:]
    else:
        t = x

    R = geometry.rotationMatrix(theta)
    U = geometry.computePreimage(V, R, t)
    U = imageprocessing.projectIntoImage(U, rendered.shape[0:2])

    residue = residueVector(observed, rendered, V, U, background_image=background_img)

    return residue.ravel()


def residueVector(
        observed, rendered, pixel_coords_observed, pixel_coords_rendered,
        background_image=None):
    """

    Parameters
    ----------
    observed :
    rendered :
    pixel_coords_observed :
    pixel_coords_rendered :

    Returns
    -------
    residue_vector :
    """

    o_rows, o_cols = utils.splitColumns(pixel_coords_observed)
    r_rows, r_cols = utils.splitColumns(pixel_coords_rendered)

    observed_pixels = observed[o_rows, o_cols].copy()

    rendered_pixels = rendered[r_rows, r_cols].copy()
    if background_image is not None:
        rendered_pixels += background_image[o_rows, o_cols]

    residue_vector = observed_pixels.squeeze(axis=1) - rendered_pixels.squeeze(axis=1)
    return residue_vector


def templateMatchingJacobian(I, T, grad_rendered_rows, grad_rendered_cols, V, x, theta=None):
    """

    Parameters
    ----------

    Returns
    -------
    """

    if theta is None:
        theta = int(x[:1])
        t = x[1:]
    else:
        t = x

    R = geometry.rotationMatrix(theta)
    U = geometry.computePreimage(V, R, t)
    U = imageprocessing.projectIntoImage(U, T.shape[0:2])

    Ur, Uc = utils.splitColumns(U)

    dT_dtr = np.squeeze(grad_rendered_rows[Ur, Uc], axis=1)
    dT_dtc = np.squeeze(grad_rendered_cols[Ur, Uc], axis=1)

    if theta is not None:
        return np.column_stack((dT_dtr.ravel(), dT_dtc.ravel()))

    Ux, Uy = geometry.rc2xy(Ur, Uc, T.shape)
    # Ttheta = -Uy * dT_dtc - Ux * dT_dtr
    dR_dtheta = geometry.rDot(theta)

    # Uxy = np.column_stack((Ux, Uy))

    dT_dt = np.column_stack((grad_rendered_rows[Ur, Uc], grad_rendered_cols[Ur, Uc]))

    stacked = np.stack((dT_dtr, dT_dtc), axis=2)
    dT_dt = np.swapaxes(stacked, 1, 2)

    dT_dtheta = np.zeros_like(dT_dtr)
    for i in range(dT_dt.shape[2]):
        dT_dtheta[:,i] = np.sum(dT_dt[:,:,i] * (V @ dR_dtheta.T), axis=1)

    jacobian = np.column_stack((dT_dtheta.ravel(), dT_dtr.ravel(), dT_dtc.ravel()))
    return jacobian


def optimizePose(
        resid_func, jacobian_func, bound_func,
        max_iters=25, min_theta=0, max_theta=360,
        **lsq_kwargs):
    """ Find the best pose (R, t) using a nonlinear least-squares objective.

    Parameters
    ----------
    resid_func : function
        Residual function inside the least-square objective.
    jacobian_func : function
        Jacobian of the residual.
    bound_func : function
        Function that produces bound constraints for the translation parameter.
    max_iters: int, optional
        Number of samples to test when optimizing `theta`. Default is 25.
    min_theta : float, optional
        Minimum value of `theta` samples. Default is 0.
    max_theta : float, optional
        Maximum value of `theta` samples. Default is 360.
    **lsq_kwargs : optional
        Any extra arguments get passed through optimizeTranslation to
        `scipy.optimize.least_squares`

    Returns
    -------
    t_best :
    theta_best :
    obj_best :
    """

    obj_best = np.inf
    t_best = np.zeros(2)
    theta_best = 0
    obj_evals = 0

    for theta in np.linspace(min_theta, max_theta, max_iters):
        resid_func = functools.partial(resid_func, theta=theta)
        jacobian_func = functools.partial(jacobian_func, theta=theta)

        R = geometry.rotationMatrix(theta)
        t_intervals = bound_func(R)
        t_init = geometry.midpoint(*t_intervals)

        t, obj, obj_evals = optimizeTranslation(
            resid_func, jacobian_func, t_init, obj_evals,
            **lsq_kwargs
        )

        t_in_bounds = geometry.sampleLiesInVolume(t, *t_intervals)
        obj_decreased = obj < obj_best

        if obj_decreased and t_in_bounds:
            obj_best = obj
            t_best = t
            theta_best = theta

    return t_best, theta_best, obj_best


def optimizeTranslation(resid_func, jacobian_func, t_init, obj_evals, **lsq_kwargs):
    """ Find the best translation `t` using a nonlinear least-squares objective.

    Parameters
    ----------
    resid_func : function
        Residual function inside the least-square objective.
    jacobian_func : function
        Jacobian of the residual.
    t_init : numpy array of float, shape (3,)
        Initial guess for the translation vector.
    obj_evals : int
        Number of objective evaluations made before calling this function.
    **lsq_kwargs : optional
        Any extra arguments get passed to `scipy.optimize.least_squares`.

    Returns
    -------
    t_new :
    obj_new :
    obj_evals :
    """

    result = scipy.optimize.least_squares(resid_func, t_init, jac=jacobian_func, **lsq_kwargs)

    t_new = result.x
    obj_new = result.cost
    obj_evals += result.nfev

    return t_new, obj_new, obj_evals


def makeInterval(bound1, bound2):
    dim_min = min(bound1, bound2)
    dim_max = max(bound1, bound2)

    return (dim_min, dim_max)


def translationBoundConstraints(rendered_center, obs_ul, obs_lr, R):
    """ Compute bound constraints for the translation vector.

    This function is used while registering a template to an observed image
    segment. While optimizing we want to add a constraint on the translation so
    that, when the template is transformed, its center is guaranteed to fall
    within the bounding box of the image segment. Otherwise the routine could
    translate the template completely outsize the observed image, and end up at
    a degenerate solution.

    This function computes bound constraints that define the set of translations
    which keep the template center within the segment's bounding box.

    Parameters
    ----------
    rendered_center :
    obs_ul :
    obs_lr :
    R :

    Returns
    -------
    dim_bounds :
    """

    if obs_ul.shape != obs_lr.shape:
        err_str = 'obs_ul shape {} != obs_lr shape {}'
        raise ValueError(err_str.format(obs_ul.shape, obs_lr.shape))

    t_min = obs_ul - R @ rendered_center
    t_max = obs_lr - R @ rendered_center

    num_dims = len(t_min)
    dim_bounds = tuple(makeInterval(t_min[i], t_max[i]) for i in range(num_dims))

    return dim_bounds
