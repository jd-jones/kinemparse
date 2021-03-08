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


def load_pyslowfast_labels(ann_dir):
    def load_fold_labels(fn):
        col_names = ['video_id', 'action_id', 'action_name', 'start_frame', 'end_frame']
        fold_labels = pd.read_csv(fn, names=col_names)
        return fold_labels

    fold_fns = ('test.csv', 'train.csv', 'val.csv')
    labels = pd.concat(
        tuple(load_fold_labels(os.path.join(ann_dir, fn)) for fn in fold_fns),
        axis=0
    )
    vocab = tuple(
        labels.loc[labels['action_id'] == i]['action_name'].iloc[0]
        for i in range(labels['action_id'].max() + 1)
    )
    seq_ids = labels['video_id'].unique()
    labels = tuple(
        labels.loc[labels['video_id'] == i].drop('video_id', axis=1).sort_values(by='start_frame')
        for i in seq_ids
    )

    sort_indices = seq_ids.argsort()
    seq_ids = seq_ids[sort_indices]
    labels = tuple(labels[i] for i in sort_indices)

    return seq_ids, labels, vocab


def load_coco_labels(ann_dir):
    def load_fold_labels(fn):
        with open(fn, 'rt') as file_:
            fold_labels = json.load(file_)

        image_metadata = pd.DataFrame(fold_labels['images']).set_index('id')
        vocab = pd.DataFrame(fold_labels['categories'])
        labels = pd.DataFrame(fold_labels['annotations'])
        labels['file_name'] = image_metadata.loc[labels['image_id']].reset_index()['file_name']
        labels.drop('image_id', axis=1)

        return vocab, labels

    fold_vocabs, fold_labels = zip(*tuple(
        load_fold_labels(os.path.join(ann_dir, f"instances_meccano_{fn}"))
        for fn in ('test.json', 'train.json', 'val.json')
    ))

    vocab = utils.selectSingleFromEq(fold_vocabs).set_index('id')
    labels = pd.concat(fold_labels, axis=0)
    labels['category_name'] = vocab.loc[labels['category_id']].reset_index()['name']
    return labels, vocab


def load_all_labels(annotation_dir):
    def get_objects(obj_labels, event_label):
        i_s = event_label.start_frame_index
        i_e = event_label.end_frame_index
        in_seg = (obj_labels.index >= i_s) * (obj_labels.index <= i_e)
        objects = obj_labels.iloc[in_seg]['category_name'].unique()
        return objects

    def make_event_labels(l_part, l_act, l_event, part_vocab):
        cols = ['start_frame_index', 'end_frame_index']
        if not (l_event[cols] == l_act[cols]).all(axis=None):
            raise AssertionError('Actions do not match events')

        event_objs = tuple(
            get_objects(l_part, tup)
            for tup in l_event.itertuples()
        )

        max_num_args = max(len(x) for x in event_objs)
        event_objs = pd.DataFrame(
            {
                f'arg{i}': tuple(objs[i] if i < len(objs) else '' for objs in event_objs)
                for i in range(max_num_args)
            },
        )

        l_event = l_event.rename(mapper={'action_name': 'event_name'}, axis=1)

        part_is_active = {}
        event_names = l_event['event_name'].copy()
        for name in part_vocab:
            if name == '':
                continue
            part_is_active[f"{name}_active"] = event_names.str.contains(name).to_numpy()
            event_names = event_names.str.replace(name, '')

        event_cols = ['start_frame_index', 'end_frame_index', 'event_name']
        act_cols = ['action_name']
        event_labels = pd.concat(
            (
                pd.DataFrame(l_event[event_cols].values, columns=event_cols),
                pd.DataFrame(l_act[act_cols].values, columns=act_cols),
                pd.DataFrame(part_is_active)
            ), axis=1
        )

        event_labels = event_labels.rename(
            mapper={
                'start_frame_index': 'start',
                'end_frame_index': 'end',
                'action_name': 'action',
                'event_name': 'event'
            }, axis=1
        )

        return event_labels

    event_seq_ids, event_labels, event_vocab = load_pyslowfast_labels(
        os.path.join(annotation_dir, 'MECCANO_action_temporal_annotations')
    )

    action_seq_ids, action_labels, action_vocab = load_pyslowfast_labels(
        os.path.join(annotation_dir, 'MECCANO_verb_temporal_annotations')
    )

    def fix_action_name(label_seq):
        label_seq['action_name'] = label_seq['action_name'].str.replace('takedriver', 'take')
        return label_seq

    action_labels = tuple(fix_action_name(label) for label in action_labels)

    part_labels, part_vocab = load_coco_labels(
        os.path.join(annotation_dir, 'MECCANO_object_annotations')
    )

    event_vocab = ('',) + event_vocab
    action_vocab = ('',) + action_vocab
    part_vocab = ('',) + tuple(part_vocab['name'].unique().tolist())

    part_labels['seq_id'] = part_labels['file_name'].str.split('_').str.get(0).astype(int)
    part_labels['file_name'] = part_labels['file_name'].str.split('_').str.get(1)
    part_seq_ids = part_labels['seq_id'].unique()
    part_seq_ids.sort()
    part_labels = tuple(
        part_labels[part_labels['seq_id'] == i].drop('seq_id', axis=1)
        for i in part_seq_ids
    )

    for label in part_labels:
        label['frame_index'] = label['file_name'].str.strip('.jpg').astype(int)
        label.set_index('frame_index', inplace=True)
    for label in action_labels:
        label['start_frame_index'] = label['start_frame'].str.strip('.jpg').astype(int)
        label['end_frame_index'] = label['end_frame'].str.strip('.jpg').astype(int)
    for label in event_labels:
        label['start_frame_index'] = label['start_frame'].str.strip('.jpg').astype(int)
        label['end_frame_index'] = label['end_frame'].str.strip('.jpg').astype(int)

    event_labels = tuple(
        make_event_labels(*tup, part_vocab)
        for tup in zip(part_labels, action_labels, event_labels)
    )

    seq_ids = utils.selectSingleFromEq((event_seq_ids, action_seq_ids))

    all_labels = (event_labels, action_labels, part_labels)
    all_vocabs = {
        'event': event_vocab,
        'action': action_vocab,
        'part': part_vocab
    }

    return seq_ids, all_labels, all_vocabs


def plot_event_labels(
        fn, event_index_seq, action_index_seq, part_activity_seq, seg_bounds,
        event_vocab, action_vocab, part_vocab):
    def make_labels(seg_bounds, seg_labels):
        label_shape = (seg_bounds.max() + 1,) + seg_labels.shape[1:]
        labels = np.zeros(label_shape, dtype=seg_labels.dtype)
        for (start, end), l in zip(seg_bounds, seg_labels):
            labels[start:end + 1] = l
        return labels

    f, axes = plt.subplots(3, sharex=True, figsize=(12, 12))

    axes[0].plot(make_labels(seg_bounds, event_index_seq))
    axes[0].set_yticks(range(len(event_vocab)))
    axes[0].set_yticklabels(event_vocab)
    axes[1].plot(make_labels(seg_bounds, action_index_seq))
    axes[1].set_yticks(range(len(action_vocab)))
    axes[1].set_yticklabels(action_vocab)
    axes[2].imshow(
        make_labels(seg_bounds, part_activity_seq).T,
        interpolation='none', aspect='auto'
    )
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
        out_dir=None, annotation_dir=None, frames_dir=None,
        col_format='standard', win_params={}, slowfast_csv_params={},
        label_types=('event', 'action', 'part')):
    out_dir = os.path.expanduser(out_dir)
    annotation_dir = os.path.expanduser(annotation_dir)
    frames_dir = os.path.expanduser(frames_dir)

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

    seq_ids, all_labels, vocabs = load_all_labels(annotation_dir)
    event_labels, action_labels, part_labels = all_labels
    vocabs = {label_name: vocabs[label_name] for label_name in label_types}

    logger.info(f"Loaded {len(seq_ids)} sequence labels from {annotation_dir}")

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
        seq_id_str = f"seq={seq_id}"

        event_segs = event_labels[i]

        # Ignore 'check booklet' events because they don't have an impact on construction
        event_segs = event_segs.loc[event_segs['event'] != 'check_booklet']

        event_segs.to_csv(os.path.join(out_labels_dir, f"{seq_id}.csv"), index=False)

        event_data = make_event_data(
            event_segs, sorted(glob.glob(os.path.join(frames_dir, f'{seq_id:02d}_*.jpg'))),
            integerizers['event'], integerizers['action'], integerizers['part'],
            vocabs['event'].index(''), vocabs['action'].index(''), False
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
            vocabs['event'], vocabs['action'], part_names
        )

        for part_activity_row, action_index in zip(label_indices['part'], label_indices['action']):
            for i, is_active in enumerate(part_activity_row):
                part_index = integerizers['part'][part_names[i]]
                counts[action_index, part_index] += int(is_active)

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
