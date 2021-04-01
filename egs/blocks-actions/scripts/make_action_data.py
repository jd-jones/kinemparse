import os
import logging
import collections

import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from mathtools import utils
from blocks.core import duplocorpus, definitions

# Disable chained assignment warnings: see https://stackoverflow.com/a/20627316/3829959
pd.options.mode.chained_assignment = None


logger = logging.getLogger(__name__)


def actions_to_events(actions_arr, use_coarse_actions=False):
    def event_from_parts(df_row):
        args = sorted([df_row.object, df_row.target])
        args = [arg for arg in args if arg]
        arg_str = ','.join(args)
        event_name = f"{df_row.action}({arg_str})"
        return event_name

    def coarsen_actions(actions_df, part_is_active):
        action_names = actions_df['action'].to_list()
        is_connection = [a in definitions.constructive_actions for a in action_names]
        # is_disconnection = [a in definitions.deconstructive_actions for a in action_names]
        is_rotation = [a in definitions.rotation_actions for a in action_names]
        is_tag = [a in definitions.annotation_tags for a in action_names]

        actions_df['action'].loc[is_connection] = 'connect'
        actions_df['action'].loc[is_rotation] = 'rotate'

        is_not_tag = ~np.array(is_tag, dtype=bool)
        actions_df = actions_df.loc[is_not_tag]
        part_is_active = part_is_active.loc[is_not_tag]
        return actions_df, part_is_active

    actions_df = pd.DataFrame(actions_arr)

    # Make part-activity dataframe
    is_active = np.zeros((actions_df.shape[0], len(definitions.blocks)), dtype=bool)
    for col_name in ('object', 'target'):
        col = actions_df[col_name].to_numpy()
        rows = np.nonzero(col >= 0)[0]
        is_active[rows, col[rows]] = True
    col_names = [f"{name}_active" for name in definitions.blocks]
    part_is_active = pd.DataFrame(is_active, columns=col_names)

    # Make event and action-name dataframe
    for key in ('action', 'object', 'target'):
        if key == 'action':
            vocab = definitions.actions
        else:
            vocab = definitions.blocks
        actions_df[key] = ['' if i == -1 else vocab[i] for i in actions_df[key]]
    if use_coarse_actions:
        actions_df, part_is_active = coarsen_actions(actions_df, part_is_active)
    actions_df['event'] = actions_df.apply(event_from_parts, axis=1)

    # Combine event/action, part dataframes
    col_names = ['start', 'end', 'event', 'action']
    events_df = pd.concat((actions_df[col_names], part_is_active), axis=1)
    return events_df


def fixStartEndIndices(action_seq, rgb_frame_timestamp_seq, selected_frame_indices):
    new_times = rgb_frame_timestamp_seq[selected_frame_indices]
    start_times = rgb_frame_timestamp_seq[action_seq['start']]
    end_times = rgb_frame_timestamp_seq[action_seq['end']]

    action_seq['start'] = utils.nearestIndices(new_times, start_times)
    action_seq['end'] = utils.nearestIndices(new_times, end_times)

    return action_seq


def load_all_labels(
        corpus_name, default_annotator, metadata_file, metadata_criteria,
        start_video_from_first_touch=True, subsample_period=None,
        use_coarse_actions=False, frames_dir=None):
    def load_one(trial_id):
        if start_video_from_first_touch:
            label_seq = corpus.readLabels(trial_id, default_annotator)[0]
            first_touch_idxs = label_seq['start'][label_seq['action'] == 7]
            if not len(first_touch_idxs):
                raise AssertionError("No first touch annotated")
            first_touch_idx = int(first_touch_idxs[0])
            # Video is sampled at 30 FPS --> start one second before the first
            # touch was annotated (unless that frame would have happened before
            # the camera started recording)
            start_idx = max(0, first_touch_idx - 30)
            selected_frame_indices = slice(start_idx, None, subsample_period)

        logger.info("  Loading labels...")
        action_seq, annotator_name, is_valid = corpus.readLabels(trial_id, default_annotator)
        if not is_valid:
            raise AssertionError("No labels")

        if not (action_seq['action'] == 7).sum():
            logger.warning(f'    trial {trial_id}: missing first touch labels')

        rgb_frame_fn_seq = corpus.getRgbFrameFns(trial_id)
        if rgb_frame_fn_seq is None:
            raise AssertionError("No RGB frames")

        rgb_frame_timestamp_seq = corpus.readRgbTimestamps(trial_id, times_only=True)
        if rgb_frame_timestamp_seq is None:
            raise AssertionError("No RGB timestamps")

        depth_frame_fn_seq = corpus.getDepthFrameFns(trial_id)
        if depth_frame_fn_seq is None:
            raise AssertionError("No depth frames")

        depth_frame_timestamp_seq = corpus.readDepthTimestamps(trial_id, times_only=True)
        if depth_frame_timestamp_seq is None:
            raise AssertionError("No depth timestamps")

        if action_seq['start'].max() >= len(rgb_frame_fn_seq):
            raise AssertionError("Actions longer than rgb frames")

        if action_seq['end'].max() >= len(rgb_frame_fn_seq):
            raise AssertionError("Actions longer than rgb frames")

        action_seq = fixStartEndIndices(
            action_seq, rgb_frame_timestamp_seq, selected_frame_indices
        )
        action_seq = actions_to_events(action_seq, use_coarse_actions=use_coarse_actions)

        rgb_frame_timestamp_seq = rgb_frame_timestamp_seq[selected_frame_indices]
        depth_frame_timestamp_seq = depth_frame_timestamp_seq[selected_frame_indices]
        rgb_frame_fn_seq = rgb_frame_fn_seq[selected_frame_indices]
        depth_frame_fn_seq = depth_frame_fn_seq[selected_frame_indices]

        return action_seq, rgb_frame_fn_seq, annotator_name

    metadata = loadMetadata(metadata_file, metadata_criteria=metadata_criteria)
    corpus = duplocorpus.DuploCorpus(corpus_name)
    seq_ids = metadata.index.to_numpy()

    event_labels = []
    frame_fn_seqs = []
    annotator_names = []
    unique_events = frozenset()
    unique_actions = frozenset()
    processed_seq_ids = []
    for i, seq_id in enumerate(seq_ids):
        try:
            event_df, rgb_frame_fn_seq, annotator_name = load_one(seq_id)
        except AssertionError as e:
            warn_str = f"Skipping video {seq_id}: {e}"
            logger.warning(warn_str)
            continue

        event_labels.append(event_df)
        frame_fn_seqs.append(rgb_frame_fn_seq)
        annotator_names.append(annotator_name)
        unique_events |= frozenset(event_df['event'].to_list())
        unique_actions |= frozenset(event_df['action'].to_list())
        processed_seq_ids.append(seq_id)

    metadata = metadata.loc[processed_seq_ids]
    metadata['annotator'] = annotator_names
    seq_ids = metadata.index.to_numpy()

    event_vocab = ('',) + tuple(sorted(unique_events))
    action_vocab = ('',) + tuple(sorted(unique_actions))
    part_vocab = ('',) + tuple(definitions.blocks)
    all_vocabs = {
        'event': event_vocab,
        'action': action_vocab,
        'part': part_vocab
    }
    return seq_ids, event_labels, frame_fn_seqs, all_vocabs, metadata


def plot_event_labels(
        fn, event_index_seq, action_index_seq, part_activity_seq,
        event_vocab, action_vocab, part_vocab):

    f, axes = plt.subplots(3, sharex=True, figsize=(12, 24))

    axes[0].plot(event_index_seq)
    axes[0].set_yticks(range(len(event_vocab)))
    axes[0].set_yticklabels(event_vocab)
    axes[1].plot(action_index_seq)
    axes[1].set_yticks(range(len(action_vocab)))
    axes[1].set_yticklabels(action_vocab)
    axes[2].imshow(part_activity_seq.T, interpolation='none', aspect='auto')
    axes[2].set_yticks(range(len(part_vocab)))
    axes[2].set_yticklabels(part_vocab)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def make_labels(seg_bounds, seg_labels, default_val, num_samples=None):
    if num_samples is None:
        num_samples = seg_bounds.max() + 1

    label_shape = (num_samples,) + seg_labels.shape[1:]
    labels = np.full(label_shape, default_val, dtype=seg_labels.dtype)
    for (start, end), l in zip(seg_bounds, seg_labels):
        labels[start:end + 1] = l

    return labels


def make_event_data(
        event_seq, filenames, event_to_index, action_to_index, part_vocab,
        event_default, action_default, part_default):
    event_indices = np.array([event_to_index[name] for name in event_seq['event']])
    action_indices = np.array([action_to_index[name] for name in event_seq['action']])

    part_names = [name for name in part_vocab if name != '']
    col_names = [f"{name}_active" for name in part_names]
    part_is_active = event_seq[col_names].values

    seg_bounds = event_seq[['start', 'end']].values
    event_index_seq = make_labels(
        seg_bounds, event_indices, event_default,
        num_samples=len(filenames)
    )
    action_index_seq = make_labels(
        seg_bounds, action_indices, action_default,
        num_samples=len(filenames)
    )
    part_activity_seq = make_labels(
        seg_bounds, part_is_active, part_default,
        num_samples=len(filenames)
    )
    data_and_labels = pd.DataFrame({
        'fn': filenames,
        'event': event_index_seq,
        'action': action_index_seq
    })
    data_and_labels = pd.concat(
        (data_and_labels, pd.DataFrame(part_activity_seq, columns=col_names)),
        axis=1
    )
    return data_and_labels


def make_clips(event_data, event_vocab, action_vocab, clip_type='window', stride=1, win_size=1):
    if clip_type == 'window':
        samples = range(0, event_data.shape[0], stride)
        clip_slices = utils.slidingWindowSlices(
            event_data, stride=stride, win_size=win_size,
            samples=samples
        )

        def get_clip_labels(arr):
            return [arr[i] for i in samples]
    elif clip_type == 'segment':
        _, seg_lens = utils.computeSegments(event_data['event'].to_numpy())
        clip_slices = tuple(utils.genSegSlices(seg_lens))

        def get_clip_labels(arr):
            return [utils.majorityVote(arr[sl]) for sl in clip_slices]
    else:
        err_str = (
            f"Unrecognized argument: clip_type={clip_type} "
            "(accepted values are 'window' or 'segment')"
        )
        raise ValueError(err_str)

    d = {
        name: get_clip_labels(event_data[name].to_numpy())
        for name in event_data.columns if name != 'fn'
    }
    d['event_id'] = d['event']
    d['event'] = [event_vocab[i] for i in d['event']]
    d['action_id'] = d['action']
    d['action'] = [action_vocab[i] for i in d['action']]
    d['start'] = [sl.start for sl in clip_slices]
    d['end'] = [min(sl.stop, event_data.shape[0]) - 1 for sl in clip_slices]

    window_clips = pd.DataFrame(d)
    return window_clips


def make_slowfast_labels(segment_bounds, label_indices, fns):
    col_dict = {
        'video_name': fns[segment_bounds['start']].apply(
            lambda x: os.path.dirname(x).split('/')[-1]
        ).to_list(),
        'start_index': segment_bounds['start'].to_list(),
        'end_index': segment_bounds['end'].to_list(),
    }
    for name in label_indices.columns:
        col_dict[name] = label_indices[name].to_list()

    slowfast_labels = pd.DataFrame(col_dict)
    return slowfast_labels


def loadMetadata(metadata_file, metadata_criteria={}):
    metadata = pd.read_excel(metadata_file, index_col=None)
    metadata = metadata.drop(columns=['Eyetrackingfilename', 'Notes'])
    metadata = metadata.loc[:, ~metadata.columns.str.contains('^Unnamed')]

    metadata = metadata.dropna(subset=['TaskID', 'VidID'])
    for key, value in metadata_criteria.items():
        in_corpus = metadata[key] == value
        metadata = metadata[in_corpus]

    metadata['VidID'] = metadata['VidID'].astype(int)
    metadata['TaskID'] = metadata['TaskID'].astype(int)

    metadata = metadata.set_index('VidID', verify_integrity=True)
    return metadata


def main(
        out_dir=None, metadata_file=None, corpus_name=None, default_annotator=None,
        metadata_criteria={}, win_params={}, slowfast_csv_params={},
        label_types=('event', 'action', 'part')):
    out_dir = os.path.expanduser(out_dir)
    metadata_file = os.path.expanduser(metadata_file)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_labels_dir = os.path.join(out_dir, 'labels')
    if not os.path.exists(out_labels_dir):
        os.makedirs(out_labels_dir)

    data_dirs = {name: os.path.join(out_dir, f"{name}-dataset") for name in label_types}
    for name, dir_ in data_dirs.items():
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    seq_ids, event_labels, frame_fn_seqs, vocabs, metadata = load_all_labels(
        corpus_name, default_annotator, metadata_file, metadata_criteria,
        start_video_from_first_touch=True, subsample_period=None,
        use_coarse_actions=True
    )
    vocabs = {label_name: vocabs[label_name] for label_name in label_types}
    for name, vocab in vocabs.items():
        utils.saveVariable(vocab, 'vocab', data_dirs[name])

    utils.saveMetadata(metadata, out_labels_dir)
    for name, dir_ in data_dirs.items():
        utils.saveMetadata(metadata, dir_)

    logger.info(f"Loaded {len(seq_ids)} sequence labels from {corpus_name} dataset")

    part_names = [name for name in vocabs['part'] if name != '']
    col_names = [f"{name}_active" for name in part_names]
    integerizers = {
        label_name: {name: i for i, name in enumerate(label_vocab)}
        for label_name, label_vocab in vocabs.items()
    }

    all_slowfast_labels_seg = collections.defaultdict(list)
    all_slowfast_labels_win = collections.defaultdict(list)
    counts = np.zeros((len(vocabs['action']), len(vocabs['part'])), dtype=int)
    for i, seq_id in enumerate(seq_ids):
        logger.info(f"Processing sequence {i + 1} / {len(seq_ids)}")

        seq_id_str = f"seq={seq_id}"

        event_segs = event_labels[i]
        frame_fns = frame_fn_seqs[i]

        # Ignore 'check booklet' events because they don't have an impact on construction
        event_segs = event_segs.loc[event_segs['event'] != 'check_booklet']

        event_data = make_event_data(
            event_segs, frame_fns,
            integerizers['event'], integerizers['action'], integerizers['part'],
            vocabs['event'].index(''), vocabs['action'].index(''), False
        )

        # Redefining event segments from the sequence catches background segments
        # that are not annotated in the source labels
        event_segs = make_clips(
            event_data, vocabs['event'], vocabs['action'],
            clip_type='segment'
        )
        event_wins = make_clips(
            event_data, vocabs['event'], vocabs['action'],
            clip_type='window', **win_params
        )

        for name in ('event', 'action'):
            event_segs[f'{name}_id'] = [integerizers[name][n] for n in event_segs[name]]
            event_wins[f'{name}_id'] = [integerizers[name][n] for n in event_wins[name]]

        event_data.to_csv(os.path.join(out_labels_dir, f"{seq_id_str}_data.csv"), index=False)
        event_segs.to_csv(os.path.join(out_labels_dir, f"{seq_id_str}_segs.csv"), index=False)
        event_wins.to_csv(os.path.join(out_labels_dir, f"{seq_id_str}_wins.csv"), index=False)

        filenames = event_data['fn'].to_list()
        label_indices = {}
        bound_keys = ['start', 'end']
        for name in label_types:
            if name == 'part':
                label_indices[name] = event_data[col_names].to_numpy()
                label_keys = col_names
            else:
                label_indices[name] = event_data[name].to_numpy()
                label_keys = [f'{name}_id']

            seg_labels_slowfast = make_slowfast_labels(
                event_segs[bound_keys], event_segs[label_keys], event_data['fn']
            )
            win_labels_slowfast = make_slowfast_labels(
                event_wins[bound_keys], event_wins[label_keys], event_data['fn']
            )

            utils.saveVariable(filenames, f'{seq_id_str}_frame-fns', data_dirs[name])
            utils.saveVariable(label_indices[name], f'{seq_id_str}_labels', data_dirs[name])
            seg_labels_slowfast.to_csv(
                os.path.join(data_dirs[name], f'{seq_id_str}_slowfast-labels.csv'),
                index=False, **slowfast_csv_params
            )
            win_labels_slowfast.to_csv(
                os.path.join(data_dirs[name], f'{seq_id_str}_slowfast-labels.csv'),
                index=False, **slowfast_csv_params
            )

            all_slowfast_labels_seg[name].append(seg_labels_slowfast)
            all_slowfast_labels_win[name].append(win_labels_slowfast)

        plot_event_labels(
            os.path.join(fig_dir, f"{seq_id_str}.png"),
            label_indices['event'], label_indices['action'], label_indices['part'],
            vocabs['event'], vocabs['action'], part_names
        )

        for part_activity_row, action_index in zip(label_indices['part'], label_indices['action']):
            for i, is_active in enumerate(part_activity_row):
                part_index = integerizers['part'][part_names[i]]
                counts[action_index, part_index] += int(is_active)

    for name, labels in all_slowfast_labels_seg.items():
        pd.concat(labels, axis=0).to_csv(
            os.path.join(data_dirs[name], 'slowfast-labels_seg.csv'),
            **slowfast_csv_params
        )
    for name, labels in all_slowfast_labels_win.items():
        pd.concat(labels, axis=0).to_csv(
            os.path.join(data_dirs[name], 'slowfast-labels_win.csv'),
            **slowfast_csv_params
        )

    utils.saveVariable(counts, 'action-part-counts', out_labels_dir)

    plt.matshow(counts)
    plt.xticks(ticks=range(len(vocabs['part'])), labels=vocabs['part'], rotation='vertical')
    plt.yticks(ticks=range(len(vocabs['action'])), labels=vocabs['action'])
    plt.savefig(os.path.join(fig_dir, 'action-part-coocurrence.png'), bbox_inches='tight')


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
