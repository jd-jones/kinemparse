import os
import logging
import json

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

    ann_seqs = {
        seq_name.replace('/', '_'): [ann for ann in ann_seq['annotation']]
        for seq_name, ann_seq in gt_segments['database'].items()
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

    return seq_names, action_labels


def plot_event_labels(
        fn, event_index_seq, action_index_seq, part_activity_seq, seg_bounds,
        event_vocab, action_vocab, part_vocab):
    def make_labels(seg_bounds, seg_labels, default_val):
        label_shape = (seg_bounds.max() + 1,) + seg_labels.shape[1:]
        labels = np.full(label_shape, default_val, dtype=seg_labels.dtype)
        for (start, end), l in zip(seg_bounds, seg_labels):
            labels[start:end + 1] = l
        return labels

    f, axes = plt.subplots(3, sharex=True, figsize=(12, 12))

    axes[0].plot(make_labels(seg_bounds, event_index_seq, event_vocab.index('NA')))
    axes[0].set_yticks(range(len(event_vocab)))
    axes[0].set_yticklabels(event_vocab)
    axes[1].plot(make_labels(seg_bounds, action_index_seq, action_vocab.index('NA')))
    axes[1].set_yticks(range(len(action_vocab)))
    axes[1].set_yticklabels(action_vocab)
    axes[2].imshow(
        make_labels(seg_bounds, part_activity_seq, False).T,
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

    annotation_dir = os.path.join(annotation_dir, 'action_annotations')

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_labels_dir = os.path.join(out_dir, 'labels')
    if not os.path.exists(out_labels_dir):
        os.makedirs(out_labels_dir)

    event_vocab_df, part_vocab, action_vocab = load_vocabs(
        os.path.join(data_dir, 'ANU_ikea_dataset', 'indexing_files', 'atomic_action_list.txt')
    )
    event_vocab_df.to_csv(os.path.join(out_labels_dir, 'event-vocab.csv'))

    label_fn = os.path.join(annotation_dir, 'gt_segments.json')
    seq_ids, event_labels = load_action_labels(label_fn, event_vocab_df)

    logger.info(f"Loaded {len(seq_ids)} sequences from {label_fn}")

    event_vocab = event_vocab_df.index.tolist()
    event_to_index = {name: i for i, name in enumerate(event_vocab)}
    action_to_index = {name: i for i, name in enumerate(action_vocab)}
    part_to_index = {name: i for i, name in enumerate(part_vocab)}

    counts = np.zeros((len(action_vocab), len(part_vocab)), dtype=int)
    for i, seq_id in enumerate(seq_ids):
        event_seq = event_labels[i]
        event_seq = event_seq.loc[event_seq['event'] != 'NA']
        if not event_seq.any(axis=None):
            logger.warning(f"No event labels for sequence {seq_id}")
            continue

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

        for part_activity_row, action_index in zip(part_is_active, action_indices):
            for i, is_active in enumerate(part_activity_row):
                part_index = part_to_index[part_names[i]]
                counts[action_index, part_index] += int(is_active)

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
