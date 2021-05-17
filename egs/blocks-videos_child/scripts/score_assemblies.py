import os
import logging

import yaml
import numpy as np
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt

from make_fusion_dataset import FusionDataset as FusionDataset_
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
    def __init__(self, *attribute_labels, device=None):
        super().__init__()

        self.device = device

        self.signatures = torch.cat(
            tuple(self._make_signatures(labels) for labels in attribute_labels),
            dim=1
        )

    def _make_signatures(self, attribute_labels):
        attribute_labels = torch.tensor(attribute_labels, dtype=torch.long, device=self.device)
        attribute_labels = torch.reshape(
            F.one_hot(attribute_labels),
            (attribute_labels.shape[0], -1)
        )
        return 2 * attribute_labels.float() - 1

    def forward(self, inputs):
        outputs = F.log_softmax(self.signatures @ inputs.transpose(0, 1), dim=0)
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


def revise_edge_labels(edge_labels, input_seqs):
    for rgb_scores, imu_scores in input_seqs:
        # Make new edge labels from RGB preds
        # FIXME: edge_labels is a dict
        best_edge_labels = rgb_scores.argmax(axis=-1)
        labels_match = np.all(best_edge_labels[None, :, :] == edge_labels[:, None, :], axis=-1)
        is_oov = ~labels_match.any(axis=-1)
        new_labels = best_edge_labels[is_oov]
        labels_match = np.all(new_labels[None, :, :] == new_labels[:, None, :], axis=-1)
        is_dup = np.any(labels_match * np.eye(labels_match.shape[0], dtype=bool), axis=-1)
        new_labels = new_labels[~is_dup]
        edge_labels = np.vstack((edge_labels, new_labels))

        # TODO: convert edge labels to in-component labels

    return edge_labels


def make_transition_scores_deprecated(rgb_edge_labels):
    num_diffs = np.sum(rgb_edge_labels[None, :, :] != rgb_edge_labels[:, None, :], axis=-1)
    bigram_counts = (num_diffs == 1).astype(float)

    denominator = bigram_counts.sum(1)
    transition_probs = np.divide(
        bigram_counts, denominator[:, None],
        out=np.zeros_like(bigram_counts),
        where=denominator[:, None] != 0
    )
    return transition_probs


def make_transition_scores(vocab):
    def is_one_block_difference(diff):
        for i in range(diff.connections.shape[0]):
            c = diff.connections.copy()
            c[i, :] = 0
            c[:, i] = 0
            if not c.any():
                return True
        return False

    bigram_counts = np.zeros((len(vocab), len(vocab)), dtype=float)
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            try:
                diff = a - b
            except ValueError:
                continue
            if is_one_block_difference(diff):
                bigram_counts[i, j] = 1

    denominator = bigram_counts.sum(1)
    transition_probs = np.divide(
        bigram_counts, denominator[:, None],
        out=np.zeros_like(bigram_counts),
        where=denominator[:, None] != 0
    )
    return transition_probs


class FusionDataset(FusionDataset_):
    def loadTargets(self, seq_id):
        trial_prefix = f"trial={seq_id}"
        assembly_seq = utils.loadVariable(f'{trial_prefix}_assembly-seq', self.rgb_data_dir)
        true_label_seq = torch.tensor(
            make_labels(assembly_seq, self.vocab),
            dtype=torch.long, device=self.device
        )
        return true_label_seq


def main(
        out_dir=None, rgb_data_dir=None, rgb_attributes_dir=None, rgb_vocab_dir=None,
        imu_data_dir=None, imu_attributes_dir=None, modalities=['rgb', 'imu'],
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

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveVariable(var, var_name, to_dir=out_data_dir):
        utils.saveVariable(var, var_name, to_dir)

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
        device=device, modalities=modalities
    )
    utils.saveMetadata(dataset.metadata, out_data_dir)
    saveVariable(dataset.vocab, 'vocab')

    # parts_vocab = loadVariable('parts-vocab')
    edge_labels = {
        'rgb': utils.loadVariable('part-labels', rgb_vocab_dir),
        'imu': np.stack([
            labels.inSameComponent(a, lower_tri_only=True)
            for a in dataset.vocab
        ])
    }
    # edge_labels = revise_edge_labels(edge_labels, input_seqs)

    attribute_labels = tuple(edge_labels[name] for name in modalities)

    logger.info('Making transition probs...')
    transition_probs = make_transition_scores(dataset.vocab)
    saveVariable(transition_probs, 'transition-probs')

    model = AttributeModel(*attribute_labels, device=device)

    if plot_predictions:
        figsize = (12, 3)
        fig, axis = plt.subplots(1, figsize=figsize)
        axis.imshow(edge_labels['rgb'].T, interpolation='none', aspect='auto')
        plt.savefig(os.path.join(fig_dir, "edge-labels.png"))
        plt.close()

    for i, trial_id in enumerate(trial_ids):
        logger.info(f"Processing sequence {trial_id}...")

        trial_prefix = f"trial={trial_id}"

        true_label_seq = dataset.loadTargets(trial_id)
        attribute_feats = dataset.loadInputs(trial_id)

        score_seq = model(attribute_feats)
        pred_label_seq = model.predict(score_seq)

        attribute_feats = attribute_feats.cpu().numpy()
        score_seq = score_seq.cpu().numpy()
        true_label_seq = true_label_seq.cpu().numpy()
        pred_label_seq = pred_label_seq.cpu().numpy()

        saveVariable(score_seq.T, f'{trial_prefix}_score-seq')
        saveVariable(true_label_seq.T, f'{trial_prefix}_label-seq')

        if plot_predictions:
            fn = os.path.join(fig_dir, f'{trial_prefix}.png')
            utils.plot_array(
                attribute_feats.T,
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
