import argparse
import os
import glob

import yaml
import joblib
import numpy as np

from mathtools import utils


def getUniqueTrialIds(dir_path):
    trial_ids = set(
        int(os.path.basename(fn).split('-')[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"trial-*.pkl"))
    )
    return sorted(tuple(trial_ids))


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

    trial_ids = getUniqueTrialIds(data_dir)
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
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--preprocess_dir')
    parser.add_argument('--start_from')
    parser.add_argument('--stop_after')

    args = vars(parser.parse_args())
    args = {k: yaml.safe_load(v) for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
    else:
        config_fn = os.path.basename(config_file_path)
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
