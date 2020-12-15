import os
import logging

import yaml
import numpy as np
import torch
from torch.nn import functional as F
import joblib
from matplotlib import pyplot as plt

from mathtools import utils, metrics, torchutils
from blocks.core import labels


logger = logging.getLogger(__name__)


def make_labels(assembly_seq, vocab):
    labels = np.zeros(assembly_seq[-1].end_idx, dtype=int)
    for assembly in assembly_seq:
        i = utils.getIndex(assembly, vocab)
        labels[assembly.start_idx:assembly.end_idx] = i
    return labels


def make_attribute_features(score_seq):
    prob_seq = F.softmax(score_seq, dim=1)
    feat_seq = torch.reshape(
        prob_seq.transpose(-1, -2),
        (prob_seq.shape[0], -1)
    )

    return 2 * feat_seq.float() - 1


class AttributeModel(torch.nn.Module):
    def __init__(self, rgb_attribute_labels, imu_attribute_labels, device=None):
        super().__init__()

        self.device = device

        self.rgb_signatures = self._make_signatures(rgb_attribute_labels)
        self.imu_signatures = self._make_signatures(imu_attribute_labels)

    def _make_signatures(self, attribute_labels):
        attribute_labels = torch.tensor(attribute_labels, dtype=torch.long, device=self.device)
        attribute_labels = torch.reshape(
            F.one_hot(attribute_labels),
            (attribute_labels.shape[0], -1)
        )
        return 2 * attribute_labels.float() - 1

    def forward(self, rgb_inputs, imu_inputs):
        rgb_outputs = F.log_softmax(self.rgb_signatures @ rgb_inputs.transpose(0, 1), dim=0)
        imu_outputs = F.log_softmax(self.imu_signatures @ imu_inputs.transpose(0, 1), dim=0)
        outputs = rgb_outputs + imu_outputs
        return outputs

    def predict(self, outputs):
        idxs = outputs.argmax(dim=0)
        return idxs


def components_equivalent(x, y):
    return (labels.inSameComponent(x) == labels.inSameComponent(y)).all()


def eval_metrics(pred_seq, true_seq, name_suffix=''):
    tp = metrics.truePositives(pred_seq, true_seq)
    tn = metrics.trueNegatives(pred_seq, true_seq)
    fp = metrics.falsePositives(pred_seq, true_seq)
    fn = metrics.falseNegatives(pred_seq, true_seq)

    acc = (tp + tn) / (tp + tn + fp + fn)

    metric_dict = {'State Accuracy' + name_suffix: acc}

    return metric_dict


def resample(rgb_attribute_seq, rgb_timestamp_seq, imu_attribute_seq, imu_timestamp_seq):
    imu_attribute_seq = utils.resampleSeq(imu_attribute_seq, imu_timestamp_seq, rgb_timestamp_seq)
    return rgb_attribute_seq, imu_attribute_seq


def main(
        out_dir=None, rgb_data_dir=None, rgb_attributes_dir=None, rgb_vocab_dir=None,
        imu_data_dir=None, imu_attributes_dir=None,
        gpu_dev_id=None, plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    out_dir = os.path.expanduser(out_dir)
    rgb_data_dir = os.path.expanduser(rgb_data_dir)
    rgb_attributes_dir = os.path.expanduser(rgb_attributes_dir)
    rgb_vocab_dir = os.path.expanduser(rgb_vocab_dir)
    imu_data_dir = os.path.expanduser(imu_data_dir)
    imu_attributes_dir = os.path.expanduser(imu_attributes_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    if results_file is None:
        results_file = os.path.join(out_dir, 'results.csv')
    else:
        results_file = os.path.expanduser(results_file)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_rgb_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_rgb_data_dir):
        os.makedirs(out_rgb_data_dir)

    def loadVariable(var_name, from_dir=rgb_data_dir):
        var = joblib.load(os.path.join(from_dir, f"{var_name}.pkl"))
        return var

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_rgb_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, rgb_data_dir, prefix='trial='):
        def loadOne(seq_id):
            fn = os.path.join(rgb_data_dir, f'{prefix}{seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    rgb_trial_ids = utils.getUniqueIds(rgb_data_dir, prefix='trial=', to_array=True)
    imu_trial_ids = utils.getUniqueIds(imu_data_dir, prefix='trial=', to_array=True)
    trial_ids = np.array(sorted(set(rgb_trial_ids.tolist()) & set(imu_trial_ids.tolist())))
    logger.info(
        f"Processing {len(trial_ids)} videos common to "
        f"RGB ({len(rgb_trial_ids)} total) and IMU ({len(imu_trial_ids)} total)"
    )

    vocab = loadVariable('vocab', from_dir=rgb_vocab_dir)
    # parts_vocab = loadVariable('parts-vocab')
    rgb_edge_labels = loadVariable('part-labels', from_dir=rgb_vocab_dir)
    imu_edge_labels = np.stack([
        labels.inSameComponent(a, lower_tri_only=True)
        for a in vocab
    ])

    device = torchutils.selectDevice(gpu_dev_id)
    model = AttributeModel(rgb_edge_labels, imu_edge_labels, device=device)

    if plot_predictions:
        figsize = (12, 3)
        fig, axis = plt.subplots(1, figsize=figsize)
        axis.imshow(rgb_edge_labels.T, interpolation='none', aspect='auto')
        plt.savefig(os.path.join(fig_dir, "edge-labels.png"))
        plt.close()

    for trial_id in trial_ids:
        logger.info(f"Processing sequence {trial_id}...")

        trial_prefix = f"trial={trial_id}"

        assembly_seq = loadVariable(f'{trial_prefix}_assembly-seq')
        true_label_seq = torch.tensor(
            make_labels(assembly_seq, vocab),
            dtype=torch.long, device=device
        )

        rgb_attribute_seq = torch.tensor(
            loadVariable(f"{trial_prefix}_score-seq", from_dir=rgb_attributes_dir),
            dtype=torch.float, device=device
        )
        rgb_timestamp_seq = loadVariable(
            f"{trial_prefix}_rgb-frame-timestamp-seq",
            from_dir=rgb_data_dir
        )
        imu_attribute_seq = torch.tensor(
            loadVariable(f"{trial_prefix}_score-seq", from_dir=imu_attributes_dir),
            dtype=torch.float, device=device
        ).permute(1, 2, 0)[:, :2]  # FIXME: save in the correct shape to avoid this reshape
        imu_timestamp_seq = loadVariable(f"{trial_prefix}_timestamp-seq", from_dir=imu_data_dir)
        rgb_attribute_seq, imu_attribute_seq = resample(
            rgb_attribute_seq, rgb_timestamp_seq,
            imu_attribute_seq, imu_timestamp_seq
        )

        rgb_attribute_seq = make_attribute_features(rgb_attribute_seq)
        imu_attribute_seq = make_attribute_features(imu_attribute_seq)

        score_seq = model(rgb_attribute_seq, imu_attribute_seq)
        pred_label_seq = model.predict(score_seq)

        rgb_attribute_seq = rgb_attribute_seq.cpu().numpy()
        score_seq = score_seq.cpu().numpy()
        true_label_seq = true_label_seq.cpu().numpy()
        pred_label_seq = pred_label_seq.cpu().numpy()

        saveVariable(score_seq.T, f'{trial_prefix}_score-seq')
        saveVariable(true_label_seq.T, f'{trial_prefix}_label-seq')

        if plot_predictions:
            fn = os.path.join(fig_dir, f'{trial_prefix}.png')
            utils.plot_array(
                rgb_attribute_seq.T,
                (true_label_seq, pred_label_seq, score_seq),
                ('gt', 'pred', 'scores'),
                fn=fn
            )

        metric_dict = eval_metrics(pred_label_seq, true_label_seq)
        for name, value in metric_dict.items():
            logger.info(f"  {name}: {value * 100:.2f}%")

        utils.writeResults(results_file, metric_dict, sweep_param_name, model_params)


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
