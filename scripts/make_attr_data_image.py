import os

import yaml
import joblib
import numpy as np

from mathtools import utils


def makeBasketLabels(action_label_seq, num_rgb_frames):
    placement_actions = action_label_seq[action_label_seq['action'] < 2]

    num_blocks = 8  # len(definitions.blocks)
    block_is_in_basket = np.ones((num_rgb_frames, num_blocks), dtype=int)
    for block_id in range(num_blocks):
        is_object = placement_actions['object'] == block_id
        is_target = placement_actions['target'] == block_id
        block_actions = placement_actions[is_object | is_target]
        if len(block_actions) == 0:
            continue
        first_placement_idx = block_actions['start'][0]

        block_is_in_basket[first_placement_idx:, block_id] = 0

    return block_is_in_basket


def main(out_dir=None, data_dir=None):
    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    if not os.path.exists(data_dir):
        raise AssertionError(f"{data_dir} does not exist")

    logger.info(f"Reading from: {data_dir}")
    logger.info(f"Writing to: {out_dir}")

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromDataDir(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    trial_ids = utils.getUniqueIds(data_dir)

    if not trial_ids:
        logger.warning(f"No recognizable data in {data_dir}")

    for i, trial_id in enumerate(trial_ids):

        trial_str = f"trial-{trial_id}"

        logger.info(f"Processing video {i + 1} / {len(trial_ids)}  (trial {trial_id})")

        action_seq = loadFromDataDir(f"{trial_str}_action-seq")
        rgb_frame_seq = loadFromDataDir(f"{trial_str}_rgb-frame-seq")
        num_frames = len(rgb_frame_seq)

        # TODO: MAKE LABELS HERE
        label_seq = makeBasketLabels(action_seq, num_frames)

        saveVariable(label_seq, f'{trial_str}_label-seq')


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
