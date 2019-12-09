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
from skimage import img_as_float
# from LCTM import models as lctm_models

from mathtools import utils
from visiontools import imageprocessing, render, geometry


logger = logging.getLogger(__name__)


# -=( HELPER FUNCTIONS )==-----------------------------------------------------
def gaussianScore(x, mean, cov):
    standardized = (x - mean) ** 2 / cov
    score = np.sum(standardized, axis=1) / 2 + np.sum(np.log(cov)) / 2
    return score


def makeHistogram(
        num_classes, samples, normalize=False, ignore_empty=True, backoff_counts=0):
    """ Make a histogram representing the empirical distribution of the input.

    Parameters
    ----------
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


# -=( INTEGERIZERS )==---------------------------------------------------------
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
          []

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
    def num_vertex_vals(self):
        return 1

    @property
    def num_vertices(self):
        return len(self.states)

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

        self.states = []
        self.state_index_map = {}

        # Populate the state list & index map so we can deal with labels that
        # aren't naturally integer-valued (such as vectors)
        for seq in state_seqs:
            for state in seq:
                self.getStateIndex(state)


class EmpiricalStateVariable(EmpiricalLatentVariables):
    def getStateIndex(self, state):
        try:
            state_index = self.states.index(state)
        except ValueError:
            self.states.append(state)
            state_index = self.num_states - 1
        return state_index


# -=( FRAME SCORING MODELS )==-------------------------------------------------
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

    def predict(self, X, predict_super=False):
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

        if predict_super:
            return predicted

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


class HandDetectionLikelihood(object):
    def __init__(self, bias_params=None, nan_score=None):
        """
        Parameters
        ----------
        bias_params : (float, float)
            0 -- Amount of probability to add when hand detection is present
            1 -- Amount of probability to add when hand detection is missing
        """

        if bias_params is None:
            # Consistent with settings in Vo et al.
            bias_params = (0.01, 0.1)

        if nan_score is None:
            nan_score = -np.inf
        self.nan_score = nan_score

        self._means = None
        self._covs = None
        self._mean_detection_scores = None
        self._log_biases = np.log(np.array(bias_params))

        pass

    def fit(
            self, label_seqs, hand_detection_seqs,
            action_means=None, action_covs=None,
            **super_kwargs):
        """ Estimate this model's parameters from data.

        Parameters
        ----------
        label_seqs : iterable( iterable( airplanecorpus.AirplaneAssembly ) )
        hand_detection_seqs : iterable( numpy array of float, shape (num_samples, 4) )
        detection_window_len : int (odd), optional
            Parameters are estimated around a window centered at the start of
            the action. This argument gives the width of that window.
        outlier_proportion : float (<= 1), optional
            After fitting an initial model, treat the least-likely
            `outlier_proportion` proportion of samples as outliers and re-estimate
            parameters.
        min_y : int
            Hand detections with y-value lower than this are discarded before training.
        **super_kwargs : optional
            Any extra keyword arguments are passed to ``super().fit()``.
        """

        super().fit(label_seqs, **super_kwargs)

        label_seqs = tuple(self.toLabelIndexArray(label_seq) for label_seq in label_seqs)
        hand_detection_seqs = np.vstack(hand_detection_seqs)
        # label_seqs contains row vectors
        label_seqs = np.hstack(label_seqs)

        num_data_samples = hand_detection_seqs.shape[0]
        num_label_samples = label_seqs.shape[0]
        if num_data_samples != num_label_samples and not self.segment_labels:
            err_str = f"{num_data_samples} data samples != {num_label_samples} label samples"
            raise ValueError(err_str)

        self._means = action_means
        self._covs = action_covs

    def computeActionScores(self, hand_detection_seq):
        scores = np.column_stack(
            tuple(
                self.computeEdgeScore(hand_detection_seq, action_key=key)
                for key in self._means.keys()
            )
        )

        return scores

    def computeEdgeScore(self, hand_detection_seq, transition=None, action_key=None):
        if action_key is None:
            action_key = self.makeActionKey(transition)

        # Compute Gaussian component of log prob
        mean = self._means[action_key]
        cov = self._covs[action_key]
        # mean_detection_score = self._mean_detection_scores[action_key]
        score = -gaussianScore(hand_detection_seq, mean, cov)  # - mean_detection_score
        # FIXME: add in Gaussian normalization constant?

        # Compute correction for missed hand detections
        score_is_nan = np.isnan(score)
        score[score_is_nan] = self.nan_score
        score = np.logaddexp(score, self._log_biases[score_is_nan.astype(int)])

        return score

    def computeLogLikelihoods(self, hand_detection_seq, transitions=None, as_array=False):
        """ Compute log-likelihood scores for a set of hypotheses.

        Parameters
        ----------
        hand_detection_seqs : iterable( numpy array of float, shape (num_samples, 4) )
        transitions : iterable( tuple(int, int) ), optional
            Each element is a pair of state indices, and represents the transition
            ``first --> second``. If `None`, this function defaults to evaluating
            all observed transitions.
        as_array : bool, optional
            If True, results is compiled into a numpy array before is it returned.
            This array has shape ``(num_samples, num_states, num_states)``.

        Returns
        -------
        transition_logprobs : dict{(int, int) -> np.array of float, shape (num_samples,)}
            See argument `as_array` for alternative return type.
        """

        if transitions is None:
            transitions = self.transitions

        transition_logprobs = {
            transition: self.computeEdgeScore(hand_detection_seq, transition)
            for transition in transitions
        }

        if as_array:
            num_samples = hand_detection_seq.shape[0]
            array = np.full((num_samples, self.num_states, self.num_states), -np.inf)
            for (prev_state_idx, state_idx), seq_probs in transition_logprobs.items():
                array[:, prev_state_idx, state_idx] = seq_probs
            transition_logprobs = array

        return transition_logprobs

    @property
    def transitions(self):
        prev_states, cur_states = self.psi.nonzero()
        transitions = zip(prev_states, cur_states)
        return transitions

    def makeActionKey(self, transition):
        start_state_idx, end_state_idx = transition

        start_objects = self.getState(start_state_idx).assembly_state
        end_objects = self.getState(end_state_idx).assembly_state
        difference = end_objects - start_objects

        if not difference:
            return ''

        if len(difference) > 1:
            logger.warning(
                f"transition {transition[0]} -> {transition[1]}"
                f"adds more than one object: {difference}"
            )

        action_key = difference.pop()
        return action_key


class BinDetectionLikelihood(object):
    def __init__(self, match_prob=0.9):
        self._scores = np.log(np.array([1 - match_prob, match_prob]))

    def fit(
            self, label_seqs, hand_detection_seqs,
            action_means=None, action_covs=None, bin_contents=None,
            **super_kwargs):
        """ Estimate this model's parameters from data.

        Parameters
        ----------
        label_seqs : iterable( iterable( airplanecorpus.AirplaneAssembly ) )
        hand_detection_seqs : iterable( numpy array of float, shape (num_samples, 4) )
        detection_window_len : int (odd), optional
            Parameters are estimated around a window centered at the start of
            the action. This argument gives the width of that window.
        outlier_proportion : float (<= 1), optional
            After fitting an initial model, treat the least-likely
            `outlier_proportion` proportion of samples as outliers and re-estimate
            parameters.
        min_y : int
            Hand detections with y-value lower than this are discarded before training.
        **super_kwargs : optional
            Any extra keyword arguments are passed to ``super().fit()``.
        """

        super().fit(label_seqs, **super_kwargs)

        label_seqs = tuple(self.toLabelIndexArray(label_seq) for label_seq in label_seqs)
        hand_detection_seqs = np.vstack(hand_detection_seqs)
        # label_seqs contains row vectors
        label_seqs = np.hstack(label_seqs)

        num_data_samples = hand_detection_seqs.shape[0]
        num_label_samples = label_seqs.shape[0]
        if num_data_samples != num_label_samples and not self.segment_labels:
            err_str = f"{num_data_samples} data samples != {num_label_samples} label samples"
            raise ValueError(err_str)

        self._means = action_means
        self._covs = action_covs
        self._bin_contents = bin_contents

    def computeEdgeScore(self, bin_detection_seq, transition=None, action_key=None):
        if action_key is None:
            action_key = self.makeActionKey(transition)

        scores = []
        for bin_index in bin_detection_seq:
            bin_objects = self._bin_contents[bin_index]
            transition_matches_bin = action_key in bin_objects
            scores.append(self._scores[int(transition_matches_bin)])
        scores = np.array(scores)

        return scores

    def computeLogLikelihoods(self, bin_detection_seq, transitions=None, as_array=False):
        """ Compute log-likelihood scores for a set of hypotheses.

        Parameters
        ----------
        hand_detection_seqs : iterable( numpy array of float, shape (num_samples, 4) )
        transitions : iterable( tuple(int, int) ), optional
            Each element is a pair of state indices, and represents the transition
            ``first --> second``. If `None`, this function defaults to evaluating
            all observed transitions.
        as_array : bool, optional
            If True, results is compiled into a numpy array before is it returned.
            This array has shape ``(num_samples, num_states, num_states)``.

        Returns
        -------
        transition_logprobs : dict{(int, int) -> np.array of float, shape (num_samples,)}
            See argument `as_array` for alternative return type.
        """

        if transitions is None:
            transitions = self.transitions

        transition_logprobs = {
            transition: self.computeEdgeScore(bin_detection_seq, transition)
            for transition in transitions
        }

        if as_array:
            num_samples = bin_detection_seq.shape[0]
            array = np.full((num_samples, self.num_states, self.num_states), -np.inf)
            for (prev_state_idx, state_idx), seq_probs in transition_logprobs.items():
                array[:, prev_state_idx, state_idx] = seq_probs
            transition_logprobs = array

        return transition_logprobs

    @property
    def transitions(self):
        prev_states, cur_states = self.psi.nonzero()
        transitions = zip(prev_states, cur_states)
        return transitions

    def makeActionKey(self, transition):
        start_state_idx, end_state_idx = transition

        start_objects = self.getState(start_state_idx).assembly_state
        end_objects = self.getState(end_state_idx).assembly_state
        difference = end_objects - start_objects

        if not difference:
            return ''

        if len(difference) > 1:
            logger.warning(
                f"transition {transition[0]} -> {transition[1]}"
                f"adds more than one object: {difference}"
            )

        action_key = difference.pop()
        return action_key


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


def assignModelClusters(keyframe_model, colors):
    """ Map each colorspace segment to the closest color in the input.

    Parameters
    ----------
    keyframe_model : FrameScorer
    colors : numpy array of int, shape (num_colors, 3)
    """

    hsv_mean_img = keyframe_model.hsv_means.copy().reshape(1, keyframe_model.n_clusters, 3)
    hsv_mean_img_saturated = hsv_mean_img.copy()
    hsv_mean_img_saturated[:, :, 1] = 1
    hsv_mean_img_saturated[:, :, 2] = 1
    rgb_mean_img = imageprocessing.color.hsv2rgb(hsv_mean_img)
    rgb_mean_img_saturated = imageprocessing.color.hsv2rgb(hsv_mean_img_saturated)
    imageprocessing.displayImage(rgb_mean_img)
    imageprocessing.displayImage(rgb_mean_img_saturated)

    rgb_means_saturated = rgb_mean_img_saturated.reshape(keyframe_model.n_clusters, 3)

    distances = np.array(tuple(
        np.linalg.norm(rgb_means_saturated - np.array(rgb_color), axis=1)
        for rgb_color in colors
    )).T
    best_idxs = distances.argmin(axis=1)

    keyframe_model.color_mappings = best_idxs
    return keyframe_model


def countColors(keyframe_model, pixels, color_names):
    px_clusters = keyframe_model.predict(pixels)
    if keyframe_model.one_indexed:
        px_clusters -= 1

    color_counts = collections.defaultdict(int)
    for px_cluster in px_clusters:
        if keyframe_model.is_low_sat_cluster[px_cluster]:
            color_name = 'background'
        elif keyframe_model.is_noise_cluster[px_cluster]:
            color_name = 'skin'
        else:
            color_name = color_names[keyframe_model.color_mappings[px_cluster]]
        color_counts[color_name] += 1

    return color_counts


class KMeansFrameScorer(FrameScorer, KMeans):
    pass


class MiniBatchKMeansFrameScorer(FrameScorer, MiniBatchKMeans):
    pass


# -=( IMU MODELS )==-----------------------------------------------------------
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


# -=( IMAGE MODELS )==---------------------------------------------------------
class ImageLikelihood(object):
    """ Compute the likelihood of an image. """

    def __init__(self, structured=False, pixel_classifier=None):
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

    def fit(self, label_seqs, *feat_seqs, model_responses=None, err_cov=None):
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
        """ Score an observed image against a hypothesis state.

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

        return -error, component_poses

    def unstructuredStateLogl(
            self, observed, pixel_classes, segment_labels, background_model,
            state_idx=None, state=None, viz_pose=False, is_depth=False,
            obsv_std=1, greedy_assignment=False, **optim_kwargs):
        """ Score an observed image against a hypothesis state.

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
            self, observed, pixel_classes, segment_labels, background_model,
            state_idxs=None, hue_only=False, mask_left_side=False,
            mask_bottom=False, viz_logls=False, **state_logl_kwargs):
        """ Score an observed image against a set of hypothesis states.

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
        state_idxs : iterable(int), optional
            Indices of the hypothesis states to score. By default this method
            will score all the states in the model's vocabulary.
        hue_only : bool, optional
            If True, pre-process `observed` by converting to HSV space,
            maxizing the saturation and value channels, and converting back to
            RGB space.
        mask_left_side : bool, optional
            If True, ignore the left side of the image by masking it. Default value is False.
        mask_bottom : bool, optional
            If True, ignore the bottom of the image by masking it. Default value is False.
        viz_logls : bool, optional
        **state_logl_kwargs : optional
            Additional keyword arguments that get passed to `self.computeStateLogLikelihood`.

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

        observed = imageprocessing.img_as_float(observed)

        if hue_only:
            observed_hsv = imageprocessing.color.rgb2hsv(observed)
            observed = imageprocessing.hueImgAsRgb(observed_hsv)
            observed[segment_labels == 0] = 0

        mask = np.zeros(pixel_classes.shape, dtype=bool)
        mask = imageprocessing.maskOutsideBuildArea(
            mask,
            mask_left_side=mask_left_side, mask_bottom=mask_bottom
        )
        if mask.any():
            observed = observed.copy()
            observed[mask] = 0
            pixel_classes = pixel_classes.copy()
            pixel_classes[mask] = 0
            segment_labels = segment_labels.copy()
            segment_labels[mask] = 0

        logls, argmaxima = utils.batchProcess(
            self.computeStateLogLikelihood,
            state_idxs,
            static_args=(observed, pixel_classes, segment_labels, background_model),
            static_kwargs=state_logl_kwargs,
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
        """ Retrieve an image of a spatial assembly component in a canonical pose.

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
        """ Create an appearance model for a component of an assembly state.

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
        """ Return the assembly state corresponding to the arg-max of `state_logls`.

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
        """ Vizualize this model's state space.

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
        """ Visualize model predictions using matplotlib.

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


class EmpiricalImageLikelihood(ImageLikelihood, EmpiricalStateVariable):
    pass


# -=( SEQUENCE MODELS )==------------------------------------------------------
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
            self.sparse_uniform_transitions_ = self.makeUniformTransitions(
                sparse=True)
        return self.sparse_uniform_transitions_

    @property
    def uniform_transitions(self):
        if self.uniform_transitions_ is None:
            self.uniform_transitions_ = self.makeUniformTransitions(
                sparse=False)
        return self.uniform_transitions_

    def makeUniformTransitions(self, sparse=False):
        transition_probs = np.ones_like(self.psi)
        if sparse:
            transition_probs[self.psi == 0] = 0
        transition_probs /= transition_probs.sum(axis=1)
        return transition_probs

    def fit(self, label_seqs, *feat_seqs,
            uniform_regularizer=0, diag_regularizer=0, empty_regularizer=0,
            zero_transition_regularizer=0,
            override_transitions=False, **super_kwargs):
        """
        Fit this model to observed data-label pairs.

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

        super().fit(label_seqs, *feat_seqs, **super_kwargs)

        psi = np.zeros((self.num_states, self.num_states))
        psi += uniform_regularizer
        for i in range(self.num_states):
            psi[i, i] += diag_regularizer

        psi[0, 0] += empty_regularizer

        psi[:, 0] += zero_transition_regularizer
        psi[0, :] += zero_transition_regularizer

        unigram_counts = np.zeros(self.num_states)

        for l in label_seqs:
            seq_bigram_counts, seq_unigram_counts = self.fitSeq(l)
            psi += seq_bigram_counts
            unigram_counts += seq_unigram_counts

        if override_transitions:
            logger.info('overriding psi')
            psi = np.ones_like(psi)

        # normalize rows of psi
        for index in range(self.num_states):
            # Define 0 / 0 as 0
            if not psi[index,:].any():  # and not unigram_counts[index]:
                continue

            # psi[index,:] /= unigram_counts[index]
            psi[index,:] /= psi[index,:].sum()

        self.psi = psi
        self.log_psi = np.log(psi)

    def fitSeq(self, label_seq):
        """
        Count max-likelihood transition parameters for a single sequence.

        Parameters
        ----------
        label_seq :

        Returns
        -------
        psi :
        unigram_counts :
        """

        labels = self.toLabelIndexArray(label_seq)

        # compute bigram and unigram counts
        unigram_counts = np.zeros(self.num_states)
        for index in labels:
            unigram_counts[index] += 1
        psi = np.zeros((self.num_states, self.num_states))
        for prev_index, cur_index in zip(labels[:-1], labels[1:]):
            psi[prev_index, cur_index] += 1

        return psi, unigram_counts

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
        if not in_beam_image.any():
            err_str = 'Viterbi beam is empty!'
            raise ValueError(err_str)

        beam_image = np.where(in_beam_image)[0]

        if verbose:
            support_size = in_beam_support.sum()
            image_size = in_beam_image.sum()
            debug_str = f'Beam size {support_size:2} -> {image_size:2}'
            logger.info(debug_str)

        return beam_image

    def transitionProbs(self, ml_decode=None, sparse_ml_decode=None):
        """
        Return pairwise scores for a decode run.

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
            log_likelihoods=None, edge_logls=None, **ll_kwargs):
        """
        Viterbi search with threshold and sparsity-based beam pruning.

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

        # Convert tuple of sequences to sequence of tuples for easier
        # iteration
        samples = tuple(zip(*samples))

        transition_probs, log_transition_probs = self.transitionProbs(
            ml_decode=ml_decode,
            sparse_ml_decode=sparse_ml_decode
        )

        if prior is None:
            prior = np.zeros(self.num_states)
            prior[0] = 1

        # Initialization
        all_argmaxima = {}
        num_samples = len(samples)
        array_dims = (self.num_states, num_samples)
        max_log_probs = np.full(array_dims, -np.inf)
        best_state_idxs = np.full(array_dims, np.nan, dtype=int)

        logl_precomputed = True
        if log_likelihoods is None:
            logl_precomputed = False
            log_likelihoods = np.full(array_dims, -np.inf)

        # Forward pass (max-sum)
        prev_max_lps = np.log(prior)
        for sample_idx, sample in enumerate(samples):

            if sample_idx == 0:
                state_lps = prev_max_lps
                max_log_probs[:, sample_idx] = state_lps
                prev_max_lps = max_log_probs[:, sample_idx]
                all_argmaxima[(0, sample_idx)] = tuple()
                continue

            # Prune hypotheses
            beam_image = self.prune(
                prev_max_lps, transition_probs,
                greed_coeff=greed_coeff, sparsity_level=sparsity_level,
                transition_thresh=transition_thresh,
                verbose=verbose_level
            )

            # Compute likelihoods / data scores
            if logl_precomputed:
                logls = log_likelihoods[beam_image, sample_idx]
                argmaxima = [None] * len(beam_image)
            else:
                logls, argmaxima = self.computeLogLikelihoods(
                    *sample, **ll_kwargs, state_idxs=beam_image
                )

            # Add in state transition scores
            for state_idx, logl, argm in zip(beam_image, logls, argmaxima):
                log_prior = (prev_max_lps + log_transition_probs[:, state_idx])
                state_lps = log_prior + logl

                if edge_logls is not None:
                    # state_lps += edge_logls[sample_idx][:-1, state_idx]
                    state_lps += edge_logls[sample_idx][:, state_idx]

                log_likelihoods[state_idx, sample_idx] = logl
                max_log_probs[state_idx, sample_idx] = state_lps.max()
                best_state_idxs[state_idx, sample_idx] = state_lps.argmax()
                all_argmaxima[(state_idx, sample_idx)] = argm
            prev_max_lps = max_log_probs[:, sample_idx]

        # Backward pass (backtrace)
        pred_idxs = np.zeros(num_samples, dtype=int)
        pred_idxs[-1] = max_log_probs[:, -1].argmax()
        for sample_idx in reversed(range(1, num_samples)):
            prev_best_state = pred_idxs[sample_idx]
            best_state_idx = best_state_idxs[prev_best_state, sample_idx]
            pred_idxs[sample_idx - 1] = best_state_idx

        final_argmaxima = tuple(
            all_argmaxima[(state_idx, sample_idx)]
            for sample_idx, state_idx in enumerate(pred_idxs))

        return pred_idxs, max_log_probs, log_likelihoods, final_argmaxima

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

        pred_state_seqs, pred_idx_seqs, unary_info = tuple(
            zip(*prediction_tups))

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
        else:
            err_str = (
                'Invalid argument "decode_method={}" '
                'decode_method must be one of: '
                'MPE (marginal posterior decoding) '
                'MAP (viterbi decoding) '
                'ML (maximum likelihood decoding)')
            raise ValueError(err_str.format(decode_method))

        if self.cur_seq_idx is not None:
            fmt_str = 'Decoding sequence {} / {}'
            logger.info(fmt_str.format(self.cur_seq_idx + 1, self.num_seqs))

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


class EdgeHmm(Hmm):
    def fit(
            self, label_seqs, *feat_seqs, final_scores=False,
            seg_level_transitions=False, **super_kwargs):
        if seg_level_transitions:
            self.segment_labels = True
            label_seqs = tuple(utils.computeSegments(label_seq)[0] for label_seq in label_seqs)

        super().fit(label_seqs, *feat_seqs, **super_kwargs)

        if final_scores:
            end_state_idxs = np.array(
                tuple(self.getStateIndex(labels[-1]) for labels in label_seqs)
            )
            self.final_probs = makeHistogram(self.num_states, end_state_idxs, normalize=True)
            self.final_scores = np.log(self.final_probs)

            plt.figure()
            plt.stem(self.final_probs)
            plt.show()
        else:
            self.final_probs = np.ones(self.num_states)
            self.final_scores = np.zeros(self.num_states)

    def viterbi(
            self, samples, prior=None,
            greed_coeff=None, sparsity_level=None, verbose_level=False,
            transition_thresh=0, ml_decode=False, sparse_ml_decode=False,
            edge_scores=None, score_samples_as_batch=False,
            **ll_kwargs):
        """
        Viterbi search with threshold and sparsity-based beam pruning.

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

        if score_samples_as_batch:
            edge_scores = self.computeLogLikelihoods(samples, as_array=True)

        # Initialization
        all_argmaxima = {}
        num_samples = samples.shape[0]
        array_dims = (self.num_states, num_samples)
        max_log_probs = np.full(array_dims, -np.inf)
        best_state_idxs = np.full(array_dims, np.nan, dtype=int)

        # Forward pass (max-sum)
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

            # Add in state transition scores
            for state_idx in beam_image:
                state_lps = (prev_max_lps + log_transition_probs[:, state_idx])

                # Compute likelihoods / data scores
                # Don't use this
                if edge_scores is None:
                    scores, argmaxima = self.computeLogLikelihoods(
                        sample, **ll_kwargs, state_idxs=beam_image
                    )
                else:
                    scores = edge_scores[sample_idx, :, state_idx]
                    argm = None
                state_lps += scores

                max_log_probs[state_idx, sample_idx] = state_lps.max()
                best_state_idxs[state_idx, sample_idx] = state_lps.argmax()
                all_argmaxima[state_idx, sample_idx] = argm
            prev_max_lps = max_log_probs[:, sample_idx]

        max_log_probs[:, -1] += self.final_scores
        plt.figure()
        plt.hist(edge_scores[~np.isinf(edge_scores)], bins=100)
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

        return pred_idxs, max_log_probs, None, final_argmaxima


def sparsifyThresh(log_vec, log_thresh):
    """

    Parameters
    ----------

    Returns
    -------
    """

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


class EmpiricalImageHmm(Hmm, ImageLikelihood, EmpiricalStateVariable):
    pass


class EmpiricalImuHmm(Hmm, ImuLikelihood, EmpiricalStateVariable):
    pass


class EmpiricalMultimodalHmm(Hmm, MultimodalLikelihood, EmpiricalStateVariable):
    pass


class EmpiricalDummyHmm(Hmm, DummyLikelihood, EmpiricalStateVariable):
    pass


class HandDetectionHmm(EdgeHmm, HandDetectionLikelihood, EmpiricalStateVariable):
    pass


class BinDetectionHmm(EdgeHmm, BinDetectionLikelihood, EmpiricalStateVariable):
    pass


# -=( DEPRECATED )==-----------------------------------------------------------
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
