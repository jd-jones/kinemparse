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
            return 'NA', 'NA', ''

        for name in action_vocab:
            if event_name.startswith(name):
                action_name = name
                break
        else:
            raise AssertionError(f"No action in vocab matching {event_name}")

        for name in part_vocab:
            if event_name == 'align leg screw with table thread':
                part_name = 'leg'
                break
            if event_name == 'align side panel holes with front panel dowels':
                part_name = 'side panel'
                break
            if event_name == 'attach shelf to table':
                part_name = 'shelf'
                break
            if event_name == 'position the drawer right side up':
                part_name = 'drawer'
                break
            if event_name == 'slide bottom of drawer':
                part_name = 'bottom panel'
                break
            if event_name in ('NA', 'other'):
                part_name = ''
                break
            if name != '' and event_name.endswith(name):
                part_name = name
                break
        else:
            raise AssertionError(f"No part in vocab matching {event_name}")

        return event_name, action_name, part_name

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
        columns=['name', 'action', 'part']
    )
    event_df = event_df.set_index('name')

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

    event_vocab, part_vocab, action_vocab = load_vocabs(
        os.path.join(data_dir, 'ANU_ikea_dataset', 'indexing_files', 'atomic_action_list.txt')
    )
    event_vocab.to_csv(os.path.join(out_labels_dir, 'event-vocab.csv'))

    label_fn = os.path.join(annotation_dir, 'gt_segments.json')
    seq_ids, action_labels = load_action_labels(label_fn, event_vocab)

    logger.info(f"Loaded {len(seq_ids)} sequences from {label_fn}")

    action_to_index = {name: i for i, name in enumerate(action_vocab)}
    part_to_index = {name: i for i, name in enumerate(part_vocab)}

    counts = np.zeros((len(action_vocab), len(part_vocab)), dtype=int)
    for i, seq_id in enumerate(seq_ids):
        label_seq = action_labels[i]
        label_seq.to_csv(os.path.join(out_labels_dir, f"{seq_id}.csv"), index=False)

        part_seq = np.array([part_to_index[name] for name in label_seq['part']])
        action_seq = np.array([action_to_index[name] for name in label_seq['action']])

        for part_index, action_index in zip(part_seq, action_seq):
            counts[action_index, part_index] += 1

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
