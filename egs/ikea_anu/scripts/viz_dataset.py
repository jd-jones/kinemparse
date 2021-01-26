import os
import logging
import json
import collections

import yaml
import pandas as pd
import graphviz as gv
# import numpy as np
# from matplotlib import pyplot as plt

from kinemparse import assembly as lib_asm
from mathtools import utils
from seqtools import fstutils_openfst as fstutils
import pywrapfst as libfst


logger = logging.getLogger(__name__)


def convert_labels(labels):
    new_labels = tuple(gen_labels(labels))

    def gen_filled_labels(labels):
        for start, end, action, arg1 in labels:
            if arg1.startswith('leg'):
                yield start, end, action, arg1, 'table top 1'
            elif arg1.startswith('shelf'):
                yield start, end, action, arg1, 'leg 1'
                yield start, end, action, arg1, 'leg 2'
                yield start, end, action, arg1, 'leg 3'
                yield start, end, action, arg1, 'leg 4'
            elif arg1.startswith('side panel'):
                all_args = tuple(label[3] for label in labels)
                if 'back panel 1' in all_args and 'front panel 1' not in all_args:
                    yield start, end, action, arg1, 'front panel 1'
                elif 'front panel 1' in all_args and 'back panel 1' not in all_args:
                    yield start, end, action, arg1, 'back panel 1'
                else:
                    warn_str = f"Can't guess arg2 for side panel: {all_args}"
                    raise AssertionError(warn_str)
            elif arg1.startswith('bottom panel'):
                yield start, end, action, arg1, 'side panel 1'
                yield start, end, action, arg1, 'side panel 2'
                yield start, end, action, arg1, 'front panel 1'
            elif arg1.startswith('pin'):
                yield start, end, action, arg1, '??? FIXME'
            elif arg1.startswith('front panel') or arg1.startswith('back panel'):
                yield start, end, action, arg1, 'side panel 1'
                yield start, end, action, arg1, 'side panel 2'
                yield start, end, action, arg1, 'bottom panel 1'

    new_new_labels = pd.DataFrame(
        tuple(gen_filled_labels(new_labels)),
        columns=('start', 'end', 'action', 'arg1', 'arg2')
    )

    return new_new_labels


def gen_labels(labels):
    event_starts = {}

    def get_event_bounds(part_name, start_index, end_index):
        if part_name in event_starts:
            start_index = event_starts[part_name]
            del event_starts[part_name]
        else:
            warn_str = f"  No start action for {part_name}"
            logger.warning(warn_str)
            start_index = start_index
        return (start_index, end_index)

    part_names = collections.defaultdict(list)

    def get_part_name(base_name):
        part_num = len(part_names[base_name]) + 1
        part_name = f"{base_name} {part_num}"
        part_names[base_name].append(part_name)
        return part_name

    for row in labels.itertuples(index=False):
        label = row.label
        i_start = row.start
        i_end = row.end
        if label.startswith('pick up'):
            part_name = label.split('pick up ')[1]
            if part_name in event_starts:
                warn_str = f"  Repeated pick up action: {label}"
                logger.warning(warn_str)
            event_starts[part_name] = i_start
        elif label.startswith('lay down'):
            part_name = label.split('lay down ')[1]
            if part_name not in event_starts:
                warn_str = f"  No pick up action before {label}"
                logger.warning(warn_str)
                continue
            del event_starts[part_name]
        elif label == 'spin leg':
            base_name = 'leg'
            start_index, end_index = get_event_bounds(base_name, i_start, i_end)
            part_name = get_part_name(base_name)
            yield (start_index, end_index, 'attach', part_name)
        elif label == 'attach shelf to table':
            base_name = 'shelf'
            start_index, end_index = get_event_bounds(base_name, i_start, i_end)
            part_name = get_part_name(base_name)
            yield (start_index, end_index, 'attach', part_name)
        elif label.startswith('attach drawer'):
            base_name = label.split('attach drawer ')[1]
            start_index, end_index = get_event_bounds(base_name, i_start, i_end)
            part_name = get_part_name(base_name)
            yield (start_index, end_index, 'attach', part_name)
        elif label == 'attach drawer back panel':
            base_name = 'back panel'
            start_index, end_index = get_event_bounds(base_name, i_start, i_end)
            part_name = get_part_name(base_name)
            yield (start_index, end_index, 'attach', part_name)
        elif label == 'slide bottom of drawer':
            base_name = 'bottom panel'
            start_index, end_index = get_event_bounds(base_name, i_start, i_end)
            part_name = get_part_name(base_name)
            yield (start_index, end_index, 'attach', part_name)
        elif label == 'insert drawer pin':
            base_name = 'pin'
            start_index, end_index = get_event_bounds(base_name, i_start, i_end)
            part_name = get_part_name(base_name)
            logger.warning('  SKIPPING PIN-INSERTION ACTIONS')
            continue
            # yield (start_index, end_index, 'attach', part_name)


def parse_assembly_actions(actions, kinem_vocab):
    def gen_segments(actions):
        prev_start = actions['start'][0]
        prev_end = actions['end'][0]
        prev_start_index = 0
        for row in actions.itertuples(index=True):
            i = row.Index
            # label = row.label
            i_start = row.start
            i_end = row.end
            # arg1 = row.arg1
            # arg2 = row.arg2

            if i_start != prev_start or i_end != prev_end:
                yield prev_start_index, i - 1
                prev_start = i_start
                prev_end = i_end
                prev_start_index = i
        else:
            yield prev_start_index, i

    def gen_kinem_labels(actions):
        state = lib_asm.Assembly()
        action_segs = tuple(gen_segments(actions))
        for start, end in action_segs:
            segment = actions.loc[start:end]
            for row in segment.itertuples(index=False):
                # label = row.label
                # i_start = row.start
                # i_end = row.end
                arg1 = row.arg1
                arg2 = row.arg2

                parent = lib_asm.Link(arg1)
                child = lib_asm.Link(arg2)
                joint = lib_asm.Joint((arg1, arg2), 'rigid', arg1, arg2)
                state = state.add_joint(
                    joint, parent, child,
                    # directed=False,
                    in_place=False
                )
                joint = lib_asm.Joint((arg2, arg1), 'rigid', arg2, arg1)
                state = state.add_joint(
                    joint, child, parent,
                    # directed=False,
                    in_place=False
                )
            start_idx = actions.loc[start]['start']
            end_idx = actions.loc[end]['end']
            state_idx = utils.getIndex(state, kinem_vocab)
            yield start_idx, end_idx, state_idx

    kinem_labels = tuple(gen_kinem_labels(actions))
    return pd.DataFrame(kinem_labels, columns=['start', 'end', 'state'])


def make_goal_state(furn_name):
    if furn_name == 'Kallax_Shelf_Drawer':
        connections = (
            ('side panel 1', 'front panel 1'),
            ('side panel 2', 'front panel 1'),
            ('bottom panel 1', 'front panel 1'),
            ('bottom panel 1', 'side panel 1'),
            ('bottom panel 1', 'side panel 2'),
            ('back panel 1', 'side panel 1'),
            ('back panel 1', 'side panel 2'),
            ('back panel 1', 'bottom panel 1'),
        )
    elif furn_name == 'Lack_Coffee_Table':
        connections = (
            ('leg 1', 'table top 1'),
            ('leg 2', 'table top 1'),
            ('leg 3', 'table top 1'),
            ('leg 4', 'table top 1'),
            ('shelf 1', 'leg 1'),
            ('shelf 1', 'leg 2'),
            ('shelf 1', 'leg 3'),
            ('shelf 1', 'leg 4')
        )
    elif furn_name == 'Lack_TV_Bench':
        connections = (
            ('leg 1', 'table top 1'),
            ('leg 2', 'table top 1'),
            ('leg 3', 'table top 1'),
            ('leg 4', 'table top 1'),
            ('shelf 1', 'leg 1'),
            ('shelf 1', 'leg 2'),
            ('shelf 1', 'leg 3'),
            ('shelf 1', 'leg 4')
        )
    elif furn_name == 'Lack_Side_Table':
        connections = (
            ('leg 1', 'table top 1'),
            ('leg 2', 'table top 1'),
            ('leg 3', 'table top 1'),
            ('leg 4', 'table top 1'),
        )
    else:
        err_str = f"Unrecognized furniture name: {furn_name}"
        raise ValueError(err_str)

    goal_state = lib_asm.Assembly()
    for arg1, arg2 in connections:
        link1 = lib_asm.Link(arg1)
        link2 = lib_asm.Link(arg2)
        joint_12 = lib_asm.Joint((arg1, arg2), 'rigid', arg1, arg2)
        joint_21 = lib_asm.Joint((arg2, arg1), 'rigid', arg2, arg1)
        goal_state = goal_state.add_joint(joint_12, link1, link2, in_place=False)
        goal_state = goal_state.add_joint(joint_21, link2, link1, in_place=False)

    return goal_state


def _convert_labels(labels):
    def ignore(label):
        ignore_prefixes = (
            'push', 'align', 'tighten', 'rotate', 'flip', 'position',
            'pick up', 'lay down'
        )
        for prefix in ignore_prefixes:
            if label.startswith(prefix):
                return True
        return False

    filtered_labels = tuple(label for label in labels if not ignore(label))
    label_pairs = []
    for cur_label, next_label in zip(filtered_labels[:-1], filtered_labels[1:]):
        if cur_label.startswith('pick up') and next_label.startswith('lay down'):
            pick_name = cur_label.split('pick up')[1]
            place_name = next_label.split('lay down')[1]
            if pick_name == place_name:
                continue
            else:
                logger.info(f"{pick_name} != {place_name}")
        label_pairs.append((cur_label, next_label))
    if not any(label_pairs):
        return []
    new_labels = (label_pairs[0][0],) + tuple(second for first, second in label_pairs)
    return new_labels


def main(out_dir=None, data_dir=None, annotation_dir=None):
    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    annotation_dir = os.path.expanduser(annotation_dir)

    annotation_dir = os.path.join(annotation_dir, 'action_annotations')

    vocab_fn = os.path.join(
        data_dir, 'ANU_ikea_dataset', 'indexing_files', 'atomic_action_list.txt'
    )
    with open(vocab_fn, 'rt') as file_:
        action_vocab = file_.read().split('\n')
    part_names = (
        label.split('pick up ')[1] for label in action_vocab
        if label.startswith('pick up')
    )
    new_action_vocab = tuple(f"{part}" for part in part_names)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_labels_dir = os.path.join(out_dir, 'labels')
    if not os.path.exists(out_labels_dir):
        os.makedirs(out_labels_dir)

    # gt_action = np.load(os.path.join(annotation_dir, 'gt_action.npy'), allow_pickle=True)
    with open(os.path.join(annotation_dir, 'gt_segments.json'), 'r') as _file:
        gt_segments = json.load(_file)

    ann_seqs = {
        seq_name: [ann for ann in ann_seq['annotation']]
        for seq_name, ann_seq in gt_segments['database'].items()
    }

    kinem_vocab = [lib_asm.Assembly()]

    all_label_index_seqs = collections.defaultdict(list)
    for seq_name, ann_seq in ann_seqs.items():
        logger.info(f"Processing sequence {seq_name}...")
        furn_name, other_name = seq_name.split('/')
        goal_state = make_goal_state(furn_name)

        label_seq = tuple(ann['label'] for ann in ann_seq)
        segment_seq = tuple(ann['segment'] for ann in ann_seq)
        start_seq, end_seq = tuple(zip(*segment_seq))
        df = pd.DataFrame({'start': start_seq, 'end': end_seq, 'label': label_seq})
        df = df.loc[df['label'] != 'NA']

        if not df.any().any():
            warn_str = f"No labels: {furn_name}, {other_name}"
            logger.warning(warn_str)
            continue

        out_path = os.path.join(out_labels_dir, furn_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        try:
            new_df = convert_labels(df)
        except AssertionError as e:
            logger.warning(f"  Skipping video: {e}")
            continue

        new_label_seq = tuple(' '.join(part_name.split()[:-1]) for part_name in new_df['arg1'])
        label_index_seq = tuple(new_action_vocab.index(label) for label in new_label_seq)
        all_label_index_seqs[furn_name].append(label_index_seq)

        kinem_df = parse_assembly_actions(new_df, kinem_vocab)
        kinem_states = tuple(kinem_vocab[i] for i in kinem_df['state'])
        if not kinem_states[-1] == goal_state:
            warn_str = f"  Final structure != goal structure:\n{kinem_states[-1]}"
            logger.warning(warn_str)

        lib_asm.writeAssemblies(
            os.path.join(out_path, f"{other_name}_kinem-state.txt"),
            kinem_states
        )

        df.to_csv(os.path.join(out_path, f"{other_name}_human.csv"), index=False)
        new_df.to_csv(os.path.join(out_path, f"{other_name}_kinem-action.csv"), index=False)
        kinem_df.to_csv(os.path.join(out_path, f"{other_name}_kinem-state.csv"), index=False)

        if not any(label_seq):
            logger.warning(f"No labels: {seq_name}")

    lib_asm.writeAssemblies(
        os.path.join(out_labels_dir, "kinem-vocab.txt"),
        kinem_vocab
    )
    symbol_table = fstutils.makeSymbolTable(new_action_vocab)
    for furn_name, label_index_seqs in all_label_index_seqs.items():
        label_fsts = tuple(
            fstutils.fromSequence(label_index_seq, symbol_table=symbol_table)
            for label_index_seq in label_index_seqs
        )
        union_fst = libfst.determinize(fstutils.easyUnion(*label_fsts))
        union_fst.minimize()

        # for i, label_fst in enumerate(label_fsts):
        #     fn = os.path.join(fig_dir, f"{furn_name}-{i}")
        #     label_fst.draw(
        #         fn, isymbols=symbol_table, osymbols=symbol_table,
        #         # vertical=True,
        #         portrait=True,
        #         acceptor=True
        #     )
        #     gv.render('dot', 'pdf', fn)
        fn = os.path.join(fig_dir, f"{furn_name}-union")
        union_fst.draw(
            fn, isymbols=symbol_table, osymbols=symbol_table,
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
