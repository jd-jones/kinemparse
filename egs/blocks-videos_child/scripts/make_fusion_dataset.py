import os
import logging

import yaml
import numpy as np
import torch
from torch.nn import functional as F

from mathtools import utils, torchutils


logger = logging.getLogger(__name__)


def make_attribute_features(score_seq):
    prob_seq = F.softmax(score_seq, dim=1)
    feat_seq = torch.reshape(
        prob_seq.transpose(-1, -2),
        (prob_seq.shape[0], -1)
    )

    return 2 * feat_seq.float() - 1


def resample(rgb_attribute_seq, rgb_timestamp_seq, imu_attribute_seq, imu_timestamp_seq):
    imu_attribute_seq = utils.resampleSeq(imu_attribute_seq, imu_timestamp_seq, rgb_timestamp_seq)
    return rgb_attribute_seq, imu_attribute_seq


class FusionDataset(object):
    def __init__(
            self, trial_ids, rgb_attributes_dir, rgb_data_dir, imu_attributes_dir, imu_data_dir,
            device=None, modalities=None):
        self.trial_ids = trial_ids

        self.metadata = utils.loadMetadata(rgb_data_dir, rows=trial_ids)
        self.vocab = utils.loadVariable('vocab', rgb_attributes_dir)

        self.rgb_attributes_dir = rgb_attributes_dir
        self.rgb_data_dir = rgb_data_dir
        self.imu_attributes_dir = imu_attributes_dir
        self.imu_data_dir = imu_data_dir
        self.device = device
        self.modalities = modalities

    def loadInputs(self, seq_id):
        if self.modalities == ['rgb']:
            return self.loadInputsRgb(seq_id)

        trial_prefix = f"trial={seq_id}"
        rgb_attribute_seq = torch.tensor(
            utils.loadVariable(f"{trial_prefix}_score-seq", self.rgb_attributes_dir),
            dtype=torch.float, device=self.device
        )
        rgb_timestamp_seq = utils.loadVariable(
            f"{trial_prefix}_rgb-frame-timestamp-seq",
            from_dir=self.rgb_data_dir
        )
        imu_attribute_seq = torch.tensor(
            utils.loadVariable(f"{trial_prefix}_score-seq", self.imu_attributes_dir),
            dtype=torch.float, device=self.device
        )
        imu_timestamp_seq = utils.loadVariable(f"{trial_prefix}_timestamp-seq", self.imu_data_dir)
        rgb_attribute_seq, imu_attribute_seq = resample(
            rgb_attribute_seq, rgb_timestamp_seq,
            imu_attribute_seq, imu_timestamp_seq
        )

        attribute_feats = {
            'rgb': make_attribute_features(rgb_attribute_seq),
            'imu': make_attribute_features(imu_attribute_seq)
        }

        attribute_feats = torch.cat(
            tuple(attribute_feats[name] for name in self.modalities),
            dim=1
        )

        return attribute_feats

    def loadInputsRgb(self, seq_id):
        trial_prefix = f"trial={seq_id}"
        rgb_attribute_seq = torch.tensor(
            utils.loadVariable(f"{trial_prefix}_score-seq", self.rgb_attributes_dir),
            dtype=torch.float, device=self.device
        )

        attribute_feats = {'rgb': make_attribute_features(rgb_attribute_seq)}
        attribute_feats = torch.cat(
            tuple(attribute_feats[name] for name in self.modalities),
            dim=1
        )

        return attribute_feats

    def loadTargets(self, seq_id):
        trial_prefix = f"trial={seq_id}"
        true_label_seq = torch.tensor(
            utils.loadVariable(f'{trial_prefix}_true-label-seq', self.rgb_attributes_dir),
            dtype=torch.long, device=self.device
        )
        return true_label_seq


def main(
        out_dir=None, modalities=['rgb', 'imu'], gpu_dev_id=None, plot_io=None,
        rgb_data_dir=None, rgb_attributes_dir=None, imu_data_dir=None, imu_attributes_dir=None):

    out_dir = os.path.expanduser(out_dir)
    rgb_data_dir = os.path.expanduser(rgb_data_dir)
    rgb_attributes_dir = os.path.expanduser(rgb_attributes_dir)
    imu_data_dir = os.path.expanduser(imu_data_dir)
    imu_attributes_dir = os.path.expanduser(imu_attributes_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    # Load data
    if modalities == ['rgb']:
        trial_ids = utils.getUniqueIds(rgb_data_dir, prefix='trial=', to_array=True)
        logger.info(f"Processing {len(trial_ids)} videos")
    else:
        rgb_trial_ids = utils.getUniqueIds(rgb_data_dir, prefix='trial=', to_array=True)
        imu_trial_ids = utils.getUniqueIds(imu_data_dir, prefix='trial=', to_array=True)
        trial_ids = np.array(sorted(set(rgb_trial_ids.tolist()) & set(imu_trial_ids.tolist())))
        logger.info(
            f"Processing {len(trial_ids)} videos common to "
            f"RGB ({len(rgb_trial_ids)} total) and IMU ({len(imu_trial_ids)} total)"
        )

    device = torchutils.selectDevice(gpu_dev_id)
    dataset = FusionDataset(
        trial_ids, rgb_attributes_dir, rgb_data_dir, imu_attributes_dir, imu_data_dir,
        device=device, modalities=modalities,
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)
    utils.saveVariable(dataset.vocab, 'vocab', out_data_dir)

    for i, trial_id in enumerate(trial_ids):
        logger.info(f"Processing sequence {trial_id}...")

        true_label_seq = dataset.loadTargets(trial_id)
        attribute_feats = dataset.loadInputs(trial_id)

        # (Process the samples here if we need to)

        attribute_feats = attribute_feats.cpu().numpy()
        true_label_seq = true_label_seq.cpu().numpy()

        trial_prefix = f"trial={trial_id}"
        utils.saveVariable(attribute_feats, f'{trial_prefix}_feature-seq', out_data_dir)
        utils.saveVariable(true_label_seq, f'{trial_prefix}_label-seq', out_data_dir)

        if plot_io:
            fn = os.path.join(fig_dir, f'{trial_prefix}.png')
            utils.plot_array(
                attribute_feats.T,
                (true_label_seq,),
                ('gt',),
                fn=fn
            )


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
