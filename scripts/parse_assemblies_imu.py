import argparse
import os

import yaml
import joblib
import numpy as np
from matplotlib import pyplot as plt

from mathtools import utils
from kinemparse import imu


def makeErrorSignal(imu_sample_seq, assembly_seq):
    error_signal = np.ones_like(imu_sample_seq) * np.nan
    for assembly in assembly_seq:
        segment_slice = slice(*assembly.getStartEndFrames())
        components = assembly.getComponents(include_singleton_blocks=True)
        error_signal[segment_slice] = imu.error(imu_sample_seq[segment_slice], components)

    return error_signal


def makeLabelSignal(imu_sample_seq, assembly_seq, action=False):
    label_seq = np.zeros_like(imu_sample_seq, dtype=bool)
    for assembly in assembly_seq:
        if action:
            segment_slice = slice(*assembly.getActionStartEndFrames())
        else:
            segment_slice = slice(*assembly.getStartEndFrames())
        connected_indices = assembly.symmetrized_connections.any(axis=0).nonzero()[0]
        label_seq[segment_slice, connected_indices] = True
    return label_seq


def plotError(
        trial_id, signal, error_signal,
        component_label_seq, block_label_seq, block_action_label_seq,
        fn=None):
    subplot_width = 12
    subplot_height = 2

    num_plots = error_signal.shape[1] + 1
    figsize = (subplot_width, num_plots * subplot_height)
    f, axes = plt.subplots(num_plots, figsize=figsize, sharex=True)

    axes[0].set_title(f"Error signal, video {trial_id}")
    axes[0].plot(component_label_seq)
    for i, axis in enumerate(axes[1:]):
        axis.set_ylabel(f"IMU {i}")
        axis.plot(signal[:, i])
        axis.plot(error_signal[:, i])
        axis = axis.twinx()
        axis.plot(block_action_label_seq[:, i], c='tab:red')
        axis.plot(block_label_seq[:, i], c='tab:green')

    plt.tight_layout()

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def main(
        out_dir=None, data_dir=None, model_name=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None, independent_signals=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadVariable(var_name):
        return joblib.load(os.path.join(data_dir, f'{var_name}.pkl'))

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    # Load data
    trial_ids = loadVariable('trial_ids')
    imu_sample_seqs = loadVariable('imu_sample_seqs')
    imu_label_seqs = loadVariable('imu_label_seqs')
    assembly_seqs = loadVariable('assembly_seqs')

    imu_sample_seqs = tuple(map(np.squeeze, imu_sample_seqs))
    errors = utils.batchProcess(makeErrorSignal, imu_sample_seqs, assembly_seqs)
    state_labels = utils.batchProcess(
        makeLabelSignal, imu_sample_seqs, assembly_seqs,
        static_kwargs={'action': False}
    )
    action_labels = utils.batchProcess(
        makeLabelSignal, imu_sample_seqs, assembly_seqs,
        static_kwargs={'action': True}
    )

    plot_args = zip(
        trial_ids, imu_sample_seqs, errors,
        imu_label_seqs, state_labels, action_labels
    )
    for args in plot_args:
        plotError(*args, fn=os.path.join(fig_dir, f"{args[0]}.png"))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.join(
            os.path.expanduser('~'), 'repo', 'kinemparse', 'scripts', config_fn
        )
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
