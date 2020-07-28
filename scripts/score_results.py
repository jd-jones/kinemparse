import os
import glob
import csv
import argparse
import warnings

import numpy as np
from matplotlib import pyplot as plt
import yaml
import joblib

from mathtools import utils, metrics
# from blocks.estimation import metrics


def getUniqueTrialIds(dir_path):
    trial_ids = set(
        int(os.path.basename(fn).split('-')[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"trial-*.pkl"))
    )
    return sorted(tuple(trial_ids))


def main(
        out_dir=None, decode_dir=None, corpus_name=None,
        metric_names=None, resolution=None, denom_mode=None,
        num_bootstrap_samples=None, confidence=None,
        param_sweep_csv_fn=None,
        param_vals={}):

    out_dir = os.path.expanduser(out_dir)
    decode_dir = os.path.expanduser(decode_dir)

    def loadFromDecodeDir(var_name):
        return joblib.load(os.path.join(decode_dir, f"{var_name}.pkl"))

    logger.info(f'Scoring results in decode directory: {decode_dir}')
    logger.info(f'Writing to: {out_dir}')

    trial_ids = getUniqueTrialIds(decode_dir)

    col_names = ['trial', 'edit'] + metric_names
    num_cols = len(col_names)
    num_trials = len(trial_ids)
    numerators = np.zeros((num_trials, num_cols))
    denominators = np.zeros((num_trials, num_cols))

    numerators[:, 0] = trial_ids
    denominators[:, 0] = trial_ids

    for i, trial_id in enumerate(trial_ids):
        trial_str = f"trial-{trial_id}"
        true_state_seq_orig = loadFromDecodeDir(f'{trial_str}_true-state-seq-orig')
        true_state_seq = loadFromDecodeDir(f'{trial_str}_true-state-seq')
        pred_state_seq = loadFromDecodeDir(f'{trial_str}_pred-state-seq')

        if len(pred_state_seq) != len(true_state_seq):
            logger.info(
                f"  Skipping video: "
                f"{len(pred_state_seq)} pred states != "
                f"{len(true_state_seq)} gt states"
            )
            continue

        dist, (num_true, num_pred) = metrics.levenshtein(
            true_state_seq_orig, pred_state_seq,
            segment_level=True,
            return_num_elems=True,
            corpus=corpus_name,
            resolution=resolution,
            normalized=False,
        )
        num_elems = max(num_true, num_pred)
        numerators[i, 1] = dist
        denominators[i, 1] = num_elems

        for j, metric_name in enumerate(metric_names):
            cur_num_correct, cur_num_total = metrics.countSeq(
                true_state_seq, pred_state_seq,
                precision=metric_name, denom_mode=denom_mode
            )
            numerators[i, 2 + j] = cur_num_correct
            denominators[i, 2 + j] = cur_num_total

    trial_scores = numerators.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered")
        trial_scores[:, 1:] = trial_scores[:, 1:] / denominators[:, 1:]

    np.savetxt(
        os.path.join(out_dir, f"trial-scores.csv"), trial_scores,
        fmt=('%d', '%.4f') + ('%.4f',) * len(metric_names),
        delimiter=',',
        header=','.join(col_names)
    )

    if num_bootstrap_samples is None:
        sample_idxs = slice(None, None, None)
    else:
        sample_idxs = np.random.randint(0, high=trial_scores.shape[0], size=num_bootstrap_samples)
    samples = trial_scores[sample_idxs, 1:]

    q_l = (1 - confidence) / 2
    mean = np.nanmean(samples, 0)
    confidence_intervals = np.nanquantile(samples, (q_l, 1 - q_l), axis=0)

    max_width = max(map(len, col_names[1:]))
    for i, name in enumerate(col_names[1:]):
        mean_score = mean[i]
        ci = confidence_intervals[:, i]
        logger.info(
            f"{name:>{max_width}}: {mean_score:6.3f}  "
            f"({confidence * 100:.0f}% CI is [{ci[0]:.3f}, {ci[1]:.3f}])"
        )

    f, axes = plt.subplots(samples.shape[1], sharex=False, sharey=True)
    for i, name in enumerate(col_names[1:]):
        col = samples[:, i]
        non_nan_samples = col[~np.isnan(col)]
        axes[i].hist(non_nan_samples, bins=50, label="counts")
        axes[i].fill_between(
            confidence_intervals[:, i],
            *(axes[i].get_ylim()),
            alpha=0.25,
            label=f"{confidence*100:.0f}% CI"
        )
        axes[i].set_ylabel(name)
        axes[i].legend()
    if num_bootstrap_samples is None:
        title = f'Performance statistics (jackknife)'
    else:
        title = f'Performance statistics (bootstrap, N={num_bootstrap_samples})'
    axes[0].set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'performance-stats.png'))

    row = mean
    header = [f"{name}" for name in col_names[1:]]

    if param_sweep_csv_fn is not None:
        file_exists = os.path.exists(param_sweep_csv_fn)
        with open(param_sweep_csv_fn, 'a') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                header += list(param_vals.keys())
                writer.writerow(header)
            row += list(param_vals.values())
            writer.writerow(row)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--decode_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.expanduser(
            os.path.join(
                '~', 'repo', 'blocks', 'blocks', 'estimation', 'scripts', 'config',
                config_fn
            )
        )
    else:
        config_fn = os.path.basename(config_file_path)
    with open(config_file_path, 'rt') as config_file:
        config = yaml.safe_load(config_file)
    config.update(args)

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
