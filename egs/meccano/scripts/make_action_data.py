import os
import logging
import json

import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import graphviz as gv
import pywrapfst as libfst

from mathtools import utils
from seqtools import fstutils_openfst as fstutils


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
            },
            axis=1
        )

        return event_labels

    event_seq_ids, event_labels, event_vocab = load_pyslowfast_labels(
        os.path.join(annotation_dir, 'action')
    )

    action_seq_ids, action_labels, action_vocab = load_pyslowfast_labels(
        os.path.join(annotation_dir, 'verb')
    )

    part_labels, part_vocab = load_coco_labels(
        os.path.join(annotation_dir, 'object')
    )

    action_vocab = ('',) + action_vocab + ('takedriver',)
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
    all_vocabs = (event_vocab, action_vocab, part_vocab)
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


def main(out_dir=None, data_dir=None, annotation_dir=None):
    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    annotation_dir = os.path.expanduser(annotation_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_labels_dir = os.path.join(out_dir, 'labels')
    if not os.path.exists(out_labels_dir):
        os.makedirs(out_labels_dir)

    seq_ids, all_labels, all_vocabs = load_all_labels(annotation_dir)
    event_labels, action_labels, part_labels = all_labels
    event_vocab, action_vocab, part_vocab = all_vocabs

    logger.info(f"Loaded {len(seq_ids)} sequence labels from {annotation_dir}")

    event_to_index = {name: i for i, name in enumerate(event_vocab)}
    action_to_index = {name: i for i, name in enumerate(action_vocab)}
    part_to_index = {name: i for i, name in enumerate(part_vocab)}

    label_fsts = []
    counts = np.zeros((len(action_vocab), len(part_vocab)), dtype=int)
    symbol_table = fstutils.makeSymbolTable(event_vocab)
    for i, seq_id in enumerate(seq_ids):
        # Ignore 'check booklet' events because they don't have an impact on construction
        event_seq = event_labels[i]
        event_seq = event_seq.loc[event_seq['event'] != 'check_booklet']
        event_seq.to_csv(os.path.join(out_labels_dir, f"{seq_id}.csv"), index=False)

        event_indices = np.array([event_to_index[name] for name in event_seq['event']])
        action_indices = np.array([action_to_index[name] for name in event_seq['action']])

        part_names = [name for name in part_vocab if name != '']
        col_names = [f"{name}_active" for name in part_names]
        part_is_active = event_seq[col_names].values

        plot_event_labels(
            os.path.join(fig_dir, f"{seq_id}.png"),
            event_indices, action_indices, part_is_active, event_seq[['start', 'end']].values,
            event_vocab, action_vocab, part_names
        )

        label_fst = fstutils.fromSequence(event_indices, symbol_table=symbol_table)
        label_fsts.append(label_fst)

        for part_activity_row, action_index in zip(part_is_active, action_indices):
            for i, is_active in enumerate(part_activity_row):
                part_index = part_to_index[part_names[i]]
                counts[action_index, part_index] += int(is_active)

    plt.matshow(counts)
    plt.xticks(ticks=range(len(part_vocab)), labels=part_vocab, rotation='vertical')
    plt.yticks(ticks=range(len(action_vocab)), labels=action_vocab)
    plt.savefig(os.path.join(fig_dir, 'action-part-coocurrence.png'), bbox_inches='tight')

    union_fst = libfst.determinize(fstutils.easyUnion(*label_fsts))
    union_fst.minimize()
    fn = os.path.join(fig_dir, "all-event-labels")
    union_fst.draw(
        fn,
        # isymbols=symbol_table, osymbols=symbol_table,
        # vertical=True,
        portrait=True,
        acceptor=True
    )
    gv.render('dot', 'pdf', fn)


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
