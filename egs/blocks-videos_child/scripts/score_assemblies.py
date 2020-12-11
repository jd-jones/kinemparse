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


def make_features(feature_seq, raw_scores=False):
    if raw_scores:
        raise NotImplementedError()

    feature_seq = 2 * torch.nn.functional.softmax(feature_seq[..., 0:2], dim=-1) - 1

    return feature_seq


def make_signatures(unique_assemblies):
    signatures = torch.stack([
        torch.tensor(labels.inSameComponent(a, lower_tri_only=True))
        for a in unique_assemblies
    ])
    signatures = 2 * signatures.float() - 1

    return signatures


class AttributeModel(torch.nn.Module):
    def __init__(self, attribute_labels, device=None):
        super().__init__()

        self.device = device
        attribute_labels = torch.tensor(attribute_labels, dtype=torch.long, device=self.device)
        attribute_labels = torch.reshape(
            F.one_hot(attribute_labels),
            (attribute_labels.shape[0], -1)
        )
        self.signatures = 2 * attribute_labels.float() - 1

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


def main(
        out_dir=None, data_dir=None, attributes_dir=None,
        use_gt_segments=None, segments_dir=None, cv_data_dir=None, ignore_trial_ids=None,
        gpu_dev_id=None,
        plot_predictions=None, results_file=None, sweep_param_name=None,
        model_params={}, cv_params={}, train_params={}, viz_params={}):

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    attributes_dir = os.path.expanduser(attributes_dir)
    if segments_dir is not None:
        segments_dir = os.path.expanduser(segments_dir)
    if cv_data_dir is not None:
        cv_data_dir = os.path.expanduser(cv_data_dir)

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

    def loadVariable(var_name, from_dir=data_dir):
        var = joblib.load(os.path.join(from_dir, f"{var_name}.pkl"))
        return var

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir, prefix='trial='):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'{prefix}{seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    vocab = loadVariable('vocab', from_dir=attributes_dir)
    # parts_vocab = loadVariable('parts-vocab')
    edge_labels = loadVariable('part-labels', from_dir=attributes_dir)

    device = torchutils.selectDevice(gpu_dev_id)
    model = AttributeModel(edge_labels, device=device)

    if plot_predictions:
        figsize = (12, 3)
        fig, axis = plt.subplots(1, figsize=figsize)
        axis.imshow(edge_labels.T, interpolation='none', aspect='auto')
        plt.savefig(os.path.join(fig_dir, "edge-labels.png"))
        plt.close()

    for trial_id in trial_ids:
        logger.info(f"Processing sequence {trial_id}...")

        trial_prefix = f"trial={trial_id}"
        assembly_seq = loadVariable(f'{trial_prefix}_assembly-seq')
        attribute_seq = torch.tensor(
            loadVariable(f"{trial_prefix}_score-seq", from_dir=attributes_dir),
            dtype=torch.float, device=device
        )
        attribute_seq = F.softmax(attribute_seq, dim=1)
        attribute_seq = torch.reshape(
            attribute_seq.transpose(-1, -2),
            (attribute_seq.shape[0], -1)
        )

        true_label_seq = torch.tensor(
            make_labels(assembly_seq, vocab),
            dtype=torch.long, device=device
        )

        score_seq = model(2 * attribute_seq - 1)
        pred_label_seq = model.predict(score_seq)

        attribute_seq = attribute_seq.cpu().numpy()
        score_seq = score_seq.cpu().numpy()
        true_label_seq = true_label_seq.cpu().numpy()
        pred_label_seq = pred_label_seq.cpu().numpy()

        saveVariable(score_seq.T, f'{trial_prefix}_score-seq')
        saveVariable(true_label_seq.T, f'{trial_prefix}_label-seq')

        if plot_predictions:
            fn = os.path.join(fig_dir, f'{trial_prefix}.png')
            utils.plot_array(
                attribute_seq.T,
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
