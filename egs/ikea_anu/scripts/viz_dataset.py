import os
import logging
import json

import yaml
import numpy as np
# from matplotlib import pyplot as plt

from mathtools import utils


logger = logging.getLogger(__name__)


def main(out_dir=None, data_dir=None):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)

    # logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_rgb_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_rgb_data_dir):
        os.makedirs(out_rgb_data_dir)

    gt_action = np.load(os.path.join(data_dir, 'gt_action.npy'), allow_pickle=True)
    with open(os.path.join(data_dir, 'gt_segments.json'), 'r') as _file:
        gt_segments = json.load(_file)

    all_labels = tuple(
        ann['label']
        for ann_seq in gt_segments['database'].values()
        for ann in ann_seq['annotation']
    )

    unique_labels = set(all_labels)

    import pdb; pdb.set_trace()


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
