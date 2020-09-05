import os
import functools
import glob
import logging

from sklearn import cluster
import skimage
from matplotlib import pyplot as plt
import numpy as np
import yaml

from mathtools import utils
try:
    from seqtools import fsm
except ImportError:
    fsm = None


LABEL_DTYPE = [('object', 'U10'), ('start_idx', 'i4'), ('end_idx', 'i4')]


logger = logging.getLogger(__name__)


def loadModelAssignments(part_to_model_fn=None):

    if part_to_model_fn is None:
        part_to_model_fn = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'kinemparse', 'resources',
            'airplane-parts-to-models.yaml'
        )

    with open(part_to_model_fn, 'rt') as f:
        part_names_to_models = yaml.safe_load(f)

    part_names = tuple(part_names_to_models.keys())
    part_names_to_idxs = {name: i for i, name in enumerate(part_names)}
    model_names = tuple(name for names in part_names_to_models.values() for name in names)
    model_names = sorted(tuple(set(model_names)))
    model_names_to_idxs = {name: i for i, name in enumerate(model_names)}
    num_parts = len(part_names)
    num_models = len(model_names)

    part_idxs_to_models = np.zeros((num_parts, num_models), dtype=bool)
    for part_name, models in part_names_to_models.items():
        part_idx = part_names_to_idxs[part_name]
        model_idxs = [n in models for n in model_names]
        part_idxs_to_models[part_idx, :] = model_idxs

    return part_idxs_to_models, model_names, model_names_to_idxs


def loadParts(part_to_bin_fn=None):
    if part_to_bin_fn is None:
        part_to_bin_fn = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'kinemparse', 'resources',
            'airplane-parts-to-bins.yaml'
        )

    with open(part_to_bin_fn, 'rt') as f:
        part_names_to_bins = yaml.safe_load(f)

    part_names = tuple(part_names_to_bins.keys())
    part_names_to_idxs = {name: i for i, name in enumerate(part_names)}
    num_parts = len(part_names)

    part_idxs_to_bins = np.zeros((num_parts,), dtype=int)
    for part_name, bin_idx in part_names_to_bins.items():
        part_idx = part_names_to_idxs[part_name]
        part_idxs_to_bins[part_idx] = bin_idx
    part_idxs_to_bins -= 1

    return part_names, part_names_to_idxs, part_idxs_to_bins


def segmentDetections(detections, kmeans, return_means=False):
    nan_rows = np.isnan(detections[:, 0]) + np.isnan(detections[:, 1])
    non_nan_detections = detections[~nan_rows, :]
    closest_bins = kmeans.predict(non_nan_detections)
    bin_segs, durations = utils.computeSegments(closest_bins)

    if not return_means:
        return bin_segs

    i = 0
    mean_detections = []
    for seg_idx, duration in zip(bin_segs, durations):
        mean_detection = np.mean(non_nan_detections[i:i + duration + 1, :], axis=0)
        i += duration + 1
        mean_detections.append(mean_detection)
    mean_detections = np.row_stack(tuple(mean_detections))

    return bin_segs, mean_detections


def fitBins(action_means, action_covs, viz=False):
    kmeans = cluster.KMeans(n_clusters=6)
    mean_arr = np.row_stack(tuple(action_means.values()))
    kmeans.fit(mean_arr)
    bin_centers = kmeans.cluster_centers_

    bin_ids = kmeans.predict(mean_arr)

    bin_contents = {i: [] for i in np.unique(bin_ids)}
    for action_name, bin_id in zip(action_means.keys(), bin_ids):
        bin_contents[bin_id].append(action_name)

    if viz:
        plt.figure()
        plt.scatter(bin_centers[:,0], bin_centers[:,1])
        plt.figure()
        for action_name, bin_id in zip(action_means.keys(), bin_ids):
            mean = action_means[action_name]
            std = action_covs[action_name] ** 0.5
            plt.errorbar(mean[0], mean[1], yerr=std[1], xerr=std[0])
            # logger.info(f"{action_name}:  bin {bin_id}  ({mean} +/- {std})")
        plt.title("Mean hand detection +/- std")
        plt.show()

        for k, v in bin_contents.items():
            logger.info(f"bin {k}: {v}")

    return kmeans, bin_contents


def estimateActionParams(hand_detection_seqs, action_seqs, win_len=17, min_y=250):
    action_segments = {}
    for detections, actions in zip(hand_detection_seqs, action_seqs):
        _ = segmentDetectionSeq(
            detections, actions, action_segments=action_segments,
            win_len=win_len, min_y=min_y
        )

    action_means = {}
    action_covs = {}
    for action_name, segments in action_segments.items():
        action_means[action_name] = np.nanmean(segments, axis=0)
        action_covs[action_name] = np.nanstd(segments, axis=0) ** 2
        # logger.info(f"{action_name}: {mean} +/- {std}")
        # axes[1].errorbar(mean[0], mean[1], yerr=std[1], xerr=std[0])

    return action_means, action_covs


def segmentDetectionSeq(
        hand_detection_seq, action_label_seq, action_segments=None,
        win_len=17, min_y=250):
    half_win_len = (win_len - 1) // 2

    for action_name, start_idx, end_idx in action_label_seq:
        # State transition observation model
        lower = max(start_idx - half_win_len, 0)
        upper = min(start_idx + half_win_len + 1, hand_detection_seq.shape[0] - 1)
        segment = hand_detection_seq[lower:upper, :]
        if min_y:
            detected_above_y_thresh = segment[:, 1] > min_y
            segment = segment[detected_above_y_thresh, :]
        prev_segments = action_segments.get(action_name, None)
        if prev_segments is None:
            action_segments[action_name] = segment
        else:
            action_segments[action_name] = np.vstack((prev_segments, segment))
        # Self-transition observation model
        lower = min(start_idx + half_win_len + 1, end_idx - half_win_len)
        upper = max(end_idx - half_win_len, start_idx + half_win_len + 1)
        segment = hand_detection_seq[lower:upper, :]
        if min_y:
            detected_below_y_thresh = segment[:, 1] < min_y
            segment = segment[detected_below_y_thresh, :]
        prev_segments = action_segments.get('', None)
        if prev_segments is None:
            action_segments[''] = segment
        else:
            action_segments[''] = np.vstack((prev_segments, segment))

    return action_segments


def getReachingHand(hand_detections, reaching_threshold=-1):
    """ Return the detected location of the hand reaching into the basket.

    Parameters
    ----------
    hand_detections : numpy array of float, shape (num_samples, 4)
    reaching_threshold : float, optional

    Returns
    -------
    reaching_hand_detections : numpy array of float, shape (num_samples, 2)
    detection_is_missing : numpy array of bool, shape (num_samples,)
    """
    detection_is_missing = np.isnan(hand_detections[:,0]) + np.isnan(hand_detections[:,2])

    first_is_reaching = hand_detections[:, 1] > reaching_threshold
    second_is_reaching = hand_detections[:, 3] > reaching_threshold

    return_first = first_is_reaching
    return_second = ~first_is_reaching * second_is_reaching

    reaching_hand_detections = np.full((hand_detections.shape[0], 2), np.nan)
    reaching_hand_detections[return_first, :] = hand_detections[return_first, :2]
    reaching_hand_detections[return_second, :] = hand_detections[return_second, 2:]

    return reaching_hand_detections, detection_is_missing


def loadCorpus(
        corpus_dir, file_id=None, subsample_period=None, reaching_only=True,
        ignore_objects_in_comparisons=None, parse_actions=False):

    corpus_dir = os.path.expanduser(corpus_dir)
    loadCorpusHandDetections = functools.partial(loadHandDetections, dir_name=corpus_dir)
    loadCorpusLabels = functools.partial(loadLabels, dir_name=corpus_dir)

    if file_id is None:
        matching_files = glob.glob(os.path.join(corpus_dir, '*.avi'))
        file_ids = tuple(utils.stripExtension(s) for s in matching_files)
    else:
        file_ids = (file_id,)

    sl = slice(None, None, subsample_period)
    hand_detection_seqs = tuple(loadCorpusHandDetections(file_id)[sl,:] for file_id in file_ids)
    if reaching_only:
        hand_detection_seqs = tuple(
            getReachingHand(detections, reaching_threshold=250)[0]
            for detections in hand_detection_seqs
        )

    part_names, part_names_to_idxs, part_idxs_to_bins = loadParts()
    action_seqs = tuple(
        loadCorpusLabels(file_id, part_names_to_idxs=part_names_to_idxs)
        for file_id in file_ids
    )

    if not parse_actions:
        part_info = (part_names, part_names_to_idxs, part_idxs_to_bins)
        return hand_detection_seqs, action_seqs, file_ids, part_info

    state_seqs = tuple(
        parseActions(actions, ignore_objects_in_comparisons=ignore_objects_in_comparisons)
        for actions in action_seqs
    )

    return hand_detection_seqs, action_seqs, state_seqs, file_ids


def loadVideoFrames(video_id, dir_name=None, as_float=False, subsample_period=None):
    if dir_name is not None:
        frames_dir = os.path.join(dir_name, f"{video_id}")

    if as_float:
        def loadFile(fn):
            return skimage.img_as_float(skimage.io.imread(fn))
    else:
        loadFile = skimage.io.imread

    subsample_slice = slice(None, None, subsample_period)

    video_frames_pattern = os.path.join(frames_dir, "*.png")
    video_fns = sorted(glob.glob(video_frames_pattern))[subsample_slice]
    video_frames = np.stack(tuple(loadFile(fn)) for fn in video_fns)

    return video_frames, video_fns


def loadHandDetections(video_id, dir_name=None, unflatten=False):
    """ Load pre-computed hand detections from CSV file.

    Parameters
    ----------
    video_id : str

    Returns
    -------
    hand_detections : numpy array of int, shape (num_video_frames, 4)
        Each row contains the location of detected hands, in pixel coordinates.
        If a hand is not detected, its corresponding array entry will be NaN.
        Column names are as follows:
        0 -- x coord, hand 1
        1 -- y coord, hand 1
        2 -- x coord, hand 2
        3 -- y coord, hand 2
    """

    if dir_name is not None:
        filename = os.path.join(dir_name, f"{video_id}.handsdetections.txt")

    def NanStrToNan(in_string):
        if in_string == 'NaN' or in_string == 'nan':
            return np.nan
        return in_string

    converters = {i: NanStrToNan for i in range(4)}
    hand_detections = np.loadtxt(filename, delimiter=",", converters=converters)

    if unflatten:
        hand_detections = np.stack((hand_detections[:, :2], hand_detections[:, 2:]), axis=1)

    return hand_detections


def loadLabels(filename, dir_name=None, part_names_to_idxs=None):
    """ Load action labels from CSV file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    actions : numpy structured array, shape (num_actions, 3)
        Each row contains an action that occurred in the corresponding video,
        along with its start and end video frame indices.
        Column names are as follows:
        0 -- object : str
        1 -- start_idx : int
        2 -- end_idx : int
    """

    if dir_name is not None:
        filename = os.path.join(dir_name, f"{filename}.txt")

    actions = np.loadtxt(filename, dtype=LABEL_DTYPE)

    end_idx_is_zero = actions['end_idx'] == 0

    # The last index is never zero and the first index never follows a zero,
    # so we can safely roll the array to index into the row following each
    # element of end_idx_is_zero
    is_next_start_idx = np.roll(end_idx_is_zero, 1)

    # The labels file uses the zero index as a shorthand to mean "the start
    # index of the next action".
    actions['end_idx'][end_idx_is_zero] = actions['start_idx'][is_next_start_idx]

    if part_names_to_idxs is not None:
        action_idxs = np.zeros_like(actions['start_idx'])
        for i, action_name in enumerate(actions['object']):
            action_idxs[i] = part_names_to_idxs[action_name]
        actions = np.column_stack((action_idxs, actions['start_idx'], actions['end_idx']))

    return actions


def parseActions(actions, ignore_objects_in_comparisons=None):
    """ Parse an action sequences into a sequence of simple assembly states.

    Parameters
    ----------
    actions : numpy structured array, shape (num_actions, 3)
        Each row contains an action that occurred in the corresponding video,
        along with its start and end video frame indices.
        Column names are as follows:
        0 -- object : str
        1 -- start_idx : int
        2 -- end_idx : int

    Returns
    -------
    states : tuple( airplanecorpus.AirplaneAssembly )
    """

    # We use this dummy row to set end_idx of the last state to -1
    dummy_row = np.array([('', -1, -1)], dtype=actions.dtype)
    actions = np.append(actions, dummy_row)

    states = [
        AirplaneAssembly(
            start_index=0, end_index=actions[0]['end_idx'],
            ignore_objects_in_comparisons=ignore_objects_in_comparisons
        )
    ]
    for action, next_action in zip(actions[:-1], actions[1:]):
        object_added = action['object']
        # action_start_idx = action['start_idx']
        action_end_idx = action['end_idx']
        next_action_end_idx = next_action['end_idx']
        new_state = states[-1].update(
            object_added, action_end_idx, next_action_end_idx,
            in_place=False
        )
        states.append(new_state)

    return states


def generateAssemblies(action_seq):
    """ Generator that consumes action strings and produces assemblies.

    Parameters
    ----------
    action_seq : iterable(string)

    Yields
    ------
    assembly : AirplaneAssembly
    """

    assembly = AirplaneAssembly()
    yield assembly

    for action in action_seq:
        assembly = assembly.update(action)
        yield assembly


class AirplaneAssembly(object):
    def __init__(
            self, start_index=None, end_index=None, assembly_state=None,
            ignore_objects_in_comparisons=None):
        self.start_index = start_index
        self.end_index = end_index

        if assembly_state is None:
            assembly_state = frozenset()
        self.assembly_state = frozenset(assembly_state)

        if ignore_objects_in_comparisons is None:
            ignore_objects_in_comparisons = []
        self.ignore_objects_in_comparisons = set(ignore_objects_in_comparisons)

    def update(self, object_added, start_index=None, end_index=None, in_place=False):
        new_state = self.assembly_state | frozenset([object_added])

        if in_place:
            self.assembly_state = new_state
            return self

        new_assembly = AirplaneAssembly(
            start_index=start_index, end_index=end_index, assembly_state=new_state,
            ignore_objects_in_comparisons=self.ignore_objects_in_comparisons
        )
        return new_assembly

    def getStartEndFrames(self):
        return self.start_index, self.end_index

    def __str__(self):
        return '\n'.join(list(self.assembly_state))

    def __eq__(self, other):
        residual = self.assembly_state ^ other.assembly_state
        eq_upto_ignored_objects = residual <= self.ignore_objects_in_comparisons
        return eq_upto_ignored_objects

    def __hash__(self):
        return hash(self.assembly_state)

    def any(self):
        """ Check if this assembly contains any edges.

        Returns
        -------
        any_edges : bool
            False if no objects are connected in this assembly.
        """
        return any(self.assembly_state)


if fsm is not None:
    class BinAssemblyModel(fsm.AbstractFstSequenceModel):
        """ Sequence model that decodes assembly actions by recognizing objects in bins. """

        def _initialize(self):
            self._integerizer = fsm.HashableFstIntegerizer()

        def _preprocessLabelSeqs(self, action_seqs):
            return tuple(action_seq['object'].tolist() for action_seq in action_seqs)

        def _fitObservationModel(self, action_segs, hand_detection_seqs, win_len=17, min_y=250):
            action_means, action_covs = estimateActionParams(
                hand_detection_seqs, action_segs, win_len=win_len, min_y=min_y
            )
            bin_locations, bin_contents = fitBins(action_means, action_covs)

            self._integerizer.update(tuple(f"bin {i}" for i in bin_contents.keys()))
            self._bin_locations = bin_locations
            self._bin_contents = bin_contents

        def _fitProcessModel(self, assembly_seqs):
            bin_to_obj = fsm.binToObjFst(self._bin_contents, self._integerizer)
            action_acceptor = fsm.stateTransitionFsa(assembly_seqs, self._integerizer)
            bin_to_action = bin_to_obj.compose(action_acceptor)
            return bin_to_action

        def _scoreObservations(self, hand_detection_seq):
            bin_segs = segmentDetections(
                hand_detection_seq, self._bin_locations, return_means=False
            )
            bin_detector = fsm.binDetectorFst(bin_segs, self._bin_contents, self._integerizer)
            return bin_detector

        def predictSeq(self, hand_detection_seq):
            action_seq = filter(None, super().predictSeq(hand_detection_seq))
            state_seq = generateAssemblies(action_seq)
            return tuple(state_seq)
