import os
import logging
import json
import glob
import collections

import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from mathtools import utils


logger = logging.getLogger(__name__)


def load_vocabs(vocab_fn):
    def get_part_name(event_name):
        return utils.remove_prefix(event_name, 'pick up ')

    def get_action_name(event_name, part_vocab):
        if event_name.startswith('align'):
            return 'align'
        if event_name.startswith('attach'):
            return 'attach'
        if event_name.startswith('position'):
            return 'position'
        if event_name.startswith('slide'):
            return 'slide'
        if event_name.startswith('insert'):
            return 'insert'
        if event_name == '':
            return 'NA'
        if event_name == 'other':
            return event_name
        for part_name in part_vocab:
            if part_name != '' and event_name.endswith(part_name):
                return utils.remove_suffix(event_name, f" {part_name}")
        else:
            raise AssertionError(f"No part in vocab matching {event_name}")

    def get_event_tuple(event_name, part_vocab, action_vocab):
        if event_name == '':
            return ('NA', 'NA') + tuple(False for name in part_vocab if name != '')

        for name in action_vocab:
            if event_name.startswith(name):
                action_name = name
                break
        else:
            raise AssertionError(f"No action in vocab matching {event_name}")

        if event_name == 'align leg screw with table thread':
            part_name = 'leg'
        elif event_name == 'align side panel holes with front panel dowels':
            part_name = 'side panel'
        elif event_name == 'attach shelf to table':
            part_name = 'shelf'
        elif event_name == 'position the drawer right side up':
            part_name = 'drawer'
        elif event_name == 'slide bottom of drawer':
            part_name = 'bottom panel'
        elif event_name in ('NA', 'other'):
            part_name = ''
        else:
            for name in part_vocab:
                if name != '' and event_name.endswith(name):
                    part_name = name
                    break
            else:
                raise AssertionError(f"No part in vocab matching {event_name}")
        part_is_active = tuple(part_name == name for name in part_vocab if name != '')

        return (event_name, action_name) + part_is_active

    with open(vocab_fn, 'rt') as file_:
        event_vocab = file_.read().split('\n')

    part_vocab = ('',) + tuple(
        get_part_name(event_label) for event_label in event_vocab
        if event_label.startswith('pick up ')
    ) + ('table', 'drawer')

    action_vocab = tuple(set(
        get_action_name(event_label, part_vocab) for event_label in event_vocab
    ))

    event_df = pd.DataFrame(
        tuple(get_event_tuple(name, part_vocab, action_vocab) for name in event_vocab),
        columns=['event', 'action'] + [f"{name}_active" for name in part_vocab if name != '']
    )
    event_df = event_df.set_index('event')

    return event_df, part_vocab, action_vocab


def load_action_labels(label_fn, event_vocab):
    with open(label_fn, 'r') as _file:
        gt_segments = json.load(_file)

    def get_metadata(seq_name, ann_seq):
        furn_name, string = seq_name.split('/')
        person, color, place = string.split('_')[:3]
        dir_name = seq_name.replace('/', '_')
        split_name = ann_seq['subset']['subset']
        return (furn_name, person, color, place, split_name, dir_name)

    ignore_seqs = (
        'Lack_Side_Table_Special_Test',
    )

    metadata = pd.DataFrame(
        tuple(
            get_metadata(seq_name, ann_seq)
            for seq_name, ann_seq in gt_segments['database'].items()
            if seq_name not in ignore_seqs
        ),
        columns=['furn_name', 'person', 'color', 'place', 'split_name', 'dir_name']
    )

    ann_seqs = {
        seq_name.replace('/', '_'): [ann for ann in ann_seq['annotation']]
        for seq_name, ann_seq in gt_segments['database'].items()
        if seq_name not in ignore_seqs
    }

    def make_action_labels(ann_seq):
        action_names = [d['label'] for d in ann_seq]
        action_bounds = [d['segment'] for d in ann_seq]
        action_labels = pd.concat(
            (
                event_vocab.loc[action_names].reset_index(),
                pd.DataFrame(action_bounds, columns=['start', 'end'])
            ), axis=1
        )
        return action_labels

    seq_names = tuple(ann_seqs.keys())
    action_labels = tuple(make_action_labels(ann_seqs[seq_name]) for seq_name in seq_names)

    return metadata.index.to_numpy(), action_labels, metadata


def plot_event_labels(
        fn, event_index_seq, action_index_seq, part_activity_seq,
        event_vocab, action_vocab, part_vocab):
    f, axes = plt.subplots(3, sharex=True, figsize=(12, 12))

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


def make_window_clips(event_data, event_vocab, action_vocab, stride=1, **win_params):
    num_samples = event_data.shape[0]
    win_indices = utils.slidingWindowSlices(event_data, stride=stride, **win_params)
    d = {
        # name: [utils.majorityVote(event_data.loc[indices][name]) for indices in win_indices]
        name: [event_data.iloc[i][name] for i in range(0, num_samples, stride)]
        for name in event_data.columns if name != 'fn'
    }
    d['event'] = [event_vocab[i] for i in d['event']]
    d['action'] = [action_vocab[i] for i in d['action']]
    d['start'] = [sl.start for sl in win_indices]
    d['end'] = [min(sl.stop, event_data.shape[0]) - 1 for sl in win_indices]

    window_clips = pd.DataFrame(d)

    return window_clips


def make_slowfast_labels(segment_bounds, labels, fns, integerizer, col_format='standard'):
    if col_format == 'standard':
        col_dict = {
            'video_name': fns[segment_bounds['start']].apply(
                lambda x: os.path.dirname(x).split('/')[-1]
            ).to_list(),
            'label_id': [integerizer[name] for name in labels.to_list()],
            'label_name': labels.to_list(),
            'start_frame': fns[segment_bounds['start']].apply(
                lambda x: os.path.basename(x)
            ).to_list(),
            'end_frame': fns[segment_bounds['end']].apply(
                lambda x: os.path.basename(x)
            ).to_list()
        }
    elif col_format == 'ikea_tk':
        col_dict = {
            'segment_id': [i for i, name in enumerate(labels.to_list())],
            'label_id': [integerizer[name] for name in labels.to_list()],
            'start_frame': fns[segment_bounds['start']].apply(
                lambda x: int(os.path.splitext(os.path.basename(x))[0])
            ).to_list(),
            'end_frame': fns[segment_bounds['end']].apply(
                lambda x: int(os.path.splitext(os.path.basename(x))[0])
            ).to_list(),
            'label_name': labels.to_list(),
            'video_name': fns[segment_bounds['start']].apply(
                lambda x: os.path.dirname(x).split('/')[-1]
            ).to_list(),
        }
    else:
        accepted_args = ('standard', 'ikea_tk')
        err_str = f"Unrecognized argument col_format={col_format}; expected one of {accepted_args}"
        raise ValueError(err_str)

    slowfast_labels = pd.DataFrame(col_dict)
    return slowfast_labels


def getActivePart(part_activity_segs, part_labels):
    is_active = part_activity_segs.to_numpy()
    if (is_active.sum(axis=1) > 1).any():
        raise AssertionError('Some columns have more than one active object!')

    active_parts = [''] * len(part_activity_segs)
    for row, col in zip(*is_active.nonzero()):
        active_parts[row] = part_labels[col]
    active_parts = pd.DataFrame({'part': active_parts})['part']

    return active_parts


def main(
        out_dir=None, data_dir=None, annotation_dir=None, frames_dir=None,
        col_format='standard', win_params={}, slowfast_csv_params={},
        label_types=('event', 'action', 'part')):
    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    annotation_dir = os.path.expanduser(annotation_dir)
    frames_dir = os.path.expanduser(frames_dir)

    annotation_dir = os.path.join(annotation_dir, 'action_annotations')

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

    event_vocab_df, part_vocab, action_vocab = load_vocabs(
        os.path.join(data_dir, 'ANU_ikea_dataset', 'indexing_files', 'atomic_action_list.txt')
    )
    event_vocab_df.to_csv(os.path.join(out_labels_dir, 'event-vocab.csv'))
    event_vocab = event_vocab_df.index.tolist()
    vocabs = {
        'event': event_vocab,
        'action': action_vocab,
        'part': part_vocab
    }
    vocabs = {label_name: vocabs[label_name] for label_name in label_types}
    for name, vocab in vocabs.items():
        utils.saveVariable(vocab, 'vocab', data_dirs[name])

    label_fn = os.path.join(annotation_dir, 'gt_segments.json')
    seq_ids, event_labels, metadata = load_action_labels(label_fn, event_vocab_df)
    utils.saveMetadata(metadata, out_labels_dir)
    for name, dir_ in data_dirs.items():
        utils.saveMetadata(metadata, dir_)

    logger.info(f"Loaded {len(seq_ids)} sequences from {label_fn}")

    part_names = [name for name in part_vocab if name != '']
    col_names = [f"{name}_active" for name in part_names]
    integerizers = {
        label_name: {name: i for i, name in enumerate(label_vocab)}
        for label_name, label_vocab in vocabs.items()
    }

    all_slowfast_labels_seg = collections.defaultdict(list)
    all_slowfast_labels_win = collections.defaultdict(list)
    counts = np.zeros((len(action_vocab), len(part_vocab)), dtype=int)
    for i, seq_id in enumerate(seq_ids):
        seq_id_str = f"seq={seq_id}"
        seq_dir_name = metadata['dir_name'].loc[seq_id]

        event_segs = event_labels[i]
        if not event_segs.any(axis=None):
            logger.warning(f"No event labels for sequence {seq_id}")
            continue

        event_data = make_event_data(
            event_segs, sorted(glob.glob(os.path.join(frames_dir, seq_dir_name, '*.jpg'))),
            integerizers['event'], integerizers['action'], integerizers['part'],
            event_vocab.index('NA'), action_vocab.index('NA'), False
        )

        event_wins = make_window_clips(
            event_data, vocabs['event'], vocabs['action'],
            **win_params
        )

        event_data.to_csv(os.path.join(out_labels_dir, f"{seq_id_str}_data.csv"), index=False)
        event_segs.to_csv(os.path.join(out_labels_dir, f"{seq_id_str}_segs.csv"), index=False)

        filenames = event_data['fn'].to_list()
        label_indices = {}
        for name in label_types:
            if name == 'part':
                label_indices[name] = event_data[col_names].to_numpy()
                seg_labels_slowfast = make_slowfast_labels(
                    event_segs[['start', 'end']], getActivePart(event_segs[col_names], part_names),
                    event_data['fn'], integerizers[name],
                    col_format=col_format
                )
                win_labels_slowfast = make_slowfast_labels(
                    event_wins[['start', 'end']], getActivePart(event_wins[col_names], part_names),
                    event_data['fn'], integerizers[name],
                    col_format=col_format
                )
            else:
                label_indices[name] = event_data[name].to_numpy()
                seg_labels_slowfast = make_slowfast_labels(
                    event_segs[['start', 'end']], event_segs[name],
                    event_data['fn'], integerizers[name],
                    col_format=col_format
                )
                win_labels_slowfast = make_slowfast_labels(
                    event_wins[['start', 'end']], event_wins[name],
                    event_data['fn'], integerizers[name],
                    col_format=col_format
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
            event_vocab, action_vocab, part_names
        )

        for part_activity_row, action_index in zip(label_indices['part'], label_indices['action']):
            for i, is_active in enumerate(part_activity_row):
                part_index = integerizers['part'][part_names[i]]
                counts[action_index, part_index] += int(is_active)

    for name, labels in all_slowfast_labels_seg.items():
        pd.concat(labels, axis=0).to_csv(
            os.path.join(data_dirs[name], 'slowfast-labels_seg.csv'),
            index=False, **slowfast_csv_params
        )
    for name, labels in all_slowfast_labels_win.items():
        pd.concat(labels, axis=0).to_csv(
            os.path.join(data_dirs[name], 'slowfast-labels_win.csv'),
            index=False, **slowfast_csv_params
        )

    utils.saveVariable(counts, 'action-part-counts', out_labels_dir)

    plt.matshow(counts)
    plt.xticks(ticks=range(len(part_vocab)), labels=part_vocab, rotation='vertical')
    plt.yticks(ticks=range(len(action_vocab)), labels=action_vocab)
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
