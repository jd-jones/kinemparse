import os
import argparse
import logging

import yaml
import pandas as pd

from mathtools import utils


logger = logging.getLogger(__name__)


def main(out_dir=None, results_file=None, sweep_param_name=None):
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    results_file = os.path.expanduser(results_file)

    # Set NUMEXPR_MAX_THREADS to avoid warnings
    os.environ["NUMEXPR_MAX_THREADS"] = '10'

    data = pd.read_csv(results_file)

    if sweep_param_name is None:
        means = data.mean() * 100
        stds = data.std() * 100
        for name in data.columns:
            if name == 'loss':
                continue
            logger.info(f"{name}: {means[name]:.1f}% +/- {stds[name]:.1f}%")

    else:
        sweep_vals = data[sweep_param_name].unique()

        def gen_rows(sweep_vals):
            for val in sweep_vals:
                matches_val = data[sweep_param_name] == sweep_vals
                matching_data = data.iloc[matches_val]
                means = matching_data.mean().rename(columns=lambda x: f'{x}_mean')
                stds = matching_data.std().rename(columns=lambda x: f'{x}_std')
                yield pd.concat([means, stds], axis=1)

        results = pd.concat([gen_rows(sweep_vals)], axis=0)
        results.to_csv(os.path.join(out_dir, "results_aggregate.csv"))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--results_file')
    parser.add_argument('--sweep_param_name')

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
    if os.path.exists(config_file_path):
        with open(config_file_path, 'rt') as config_file:
            config = yaml.safe_load(config_file)
    else:
        config = {}

    for k, v in args.items():
        if isinstance(v, dict) and k in config:
            config[k].update(v)
        else:
            config[k] = v

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    utils.autoreload_ipython()

    main(**config)
