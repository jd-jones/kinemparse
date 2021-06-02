import os
import logging
import glob

import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mathtools import utils


logger = logging.getLogger(__name__)


def plotLabels(fn, label_seq):
    fig, axis = plt.subplots(1, figsize=(12, 9))

    axis.plot(label_seq)

    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def writeLabels(fn, label_seq, vocab):
    seg_label_idxs, seg_durs = utils.computeSegments(label_seq)

    seg_durs = np.array(seg_durs)
    seg_ends = np.cumsum(seg_durs) - 1
    seg_starts = np.array([0] + (seg_ends + 1)[:-1].tolist())
    seg_labels = tuple(str(vocab[i]) for i in seg_label_idxs)
    d = {
        'start': seg_starts,
        'end': seg_ends,
        'label': seg_labels
    }
    pd.DataFrame(d).to_csv(fn, index=False)


def parseActions(assembly_actions, num_frames, vocab):
    def makeJoint(part1, part2):
        return tuple(sorted([part1, part2]))

    def updateAssembly(assembly, joint):
        return tuple(sorted(cur_assembly + (joint,)))

    assembly_index_seq = np.zeros(num_frames, dtype=int)

    cur_assembly = tuple()
    prev_start = -1
    prev_end = -1
    for i, row in assembly_actions.iterrows():
        if row.start != prev_start or row.end != prev_end:
            cur_assembly_index = utils.getIndex(cur_assembly, vocab)
            assembly_index_seq[prev_end:row.end + 1] = cur_assembly_index
            prev_start = row.start
            prev_end = row.end

        if row.action == 'connect':
            joint = makeJoint(row.part1, row.part2)
            cur_assembly = updateAssembly(cur_assembly, joint)
        elif row.action == 'pin':
            continue
        else:
            raise ValueError()
    cur_assembly_index = utils.getIndex(cur_assembly, vocab)
    assembly_index_seq[prev_end:] = cur_assembly_index
    return assembly_index_seq


def main(out_dir=None, data_dir=None, labels_dir=None):
    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    labels_dir = os.path.expanduser(labels_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    filenames = [
        utils.stripExtension(fn)
        for fn in glob.glob(os.path.join(labels_dir, '*.csv'))
    ]

    metadata = utils.loadMetadata(data_dir)
    metadata['seq_id'] = metadata.index
    metadata = metadata.set_index('dir_name', drop=False).loc[filenames].set_index('seq_id')

    seq_ids = np.sort(metadata.index.to_numpy())
    logger.info(f"Loaded {len(seq_ids)} sequences from {labels_dir}")

    vocab = []
    for i, seq_id in enumerate(seq_ids):
        seq_id_str = f"seq={seq_id}"
        seq_dir_name = metadata['dir_name'].loc[seq_id]
        labels_fn = os.path.join(labels_dir, f'{seq_dir_name}.csv')
        event_labels = utils.loadVariable(f'{seq_id_str}_labels', data_dir)

        assembly_actions = pd.read_csv(labels_fn)
        label_seq = parseActions(assembly_actions, event_labels.shape[0], vocab)
        utils.saveVariable(label_seq, f'{seq_id_str}_label-seq', out_data_dir)

        plotLabels(os.path.join(fig_dir, f'{seq_id_str}_labels.png'), label_seq)
        writeLabels(os.path.join(fig_dir, f'{seq_id_str}_labels.csv'), label_seq, vocab)

    utils.saveMetadata(metadata, out_data_dir)
    utils.saveVariable(vocab, 'vocab', out_data_dir)


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
