import os
import logging

import yaml
import numpy as np
import torch
from torch.nn import functional as F
import joblib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

from mathtools import utils, metrics, torchutils
from blocks.core import labels
from kinemparse.assembly import Assembly
from kinemparse import assembly as lib_assembly
from kinemparse import sim2real
from visiontools import render, imageprocessing


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
    edge_labels = {
        'rgb': loadVariable('part-labels', from_dir=rgb_vocab_dir),
        'imu': np.stack([
            labels.inSameComponent(a, lower_tri_only=True)
            for a in vocab
        ])
    }
    attribute_labels = tuple(edge_labels[name] for name in modalities)

    new_vocab = tuple(Assembly.from_blockassembly(a) for a in vocab)
    device = torchutils.selectDevice(gpu_dev_id)

    intrinsic_matrix = torch.tensor(render.intrinsic_matrix, dtype=torch.float).cuda()
    camera_pose = torch.tensor(render.camera_pose, dtype=torch.float).cuda()
    colors = torch.tensor(render.object_colors, dtype=torch.float).cuda()
    image_size = min(render.IMAGE_WIDTH, render.IMAGE_HEIGHT)
    renderer = render.TorchSceneRenderer(
        intrinsic_matrix=intrinsic_matrix,
        camera_pose=camera_pose,
        colors=colors,
        light_intensity_ambient=1,
        image_size=image_size,
        orig_size=image_size
    )

    def canonicalPose(num_samples=1):
        angles = torch.zeros(num_samples, dtype=torch.float)
        rotations = Rotation.from_euler('Z', angles)
        R = torch.tensor(rotations.as_matrix()).float().cuda()
        t = torch.stack((torch.zeros_like(angles),) * 3, dim=1).float().cuda()
        return R, t

    R, t = canonicalPose()

    for i, assembly in enumerate(vocab[1:25], start=1):
        rgb_images, depth_images, label_images = sim2real.renderTemplates(
            renderer, assembly, t, R
        )
        rgb_images[rgb_images > 1] = 1
        imageprocessing.displayImages(
            *(rgb_images.cpu().numpy()),
            file_path=os.path.join(fig_dir, f"{i:03d}_old.png")
        )

    for i, assembly in enumerate(new_vocab[1:25], start=1):
        G = lib_assembly.draw_graph(assembly, name=f'{i:03d}_graph')
        G.render(directory=fig_dir, format='png', cleanup=True)

        rgb_images, depth_images, label_images = lib_assembly.render(renderer, assembly, t, R)
        rgb_images[rgb_images > 1] = 1
        imageprocessing.displayImages(
            *(rgb_images.cpu().numpy()),
            file_path=os.path.join(fig_dir, f"{i:03d}_new.png")
        )
    import pdb; pdb.set_trace()

    model = AttributeModel(*attribute_labels, device=device)

    if plot_predictions:
        figsize = (12, 3)
        fig, axis = plt.subplots(1, figsize=figsize)
        axis.imshow(edge_labels['rgb'].T, interpolation='none', aspect='auto')
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

        attribute_feats = {
            'rgb': make_attribute_features(rgb_attribute_seq),
            'imu': make_attribute_features(imu_attribute_seq)
        }
        attribute_feats = torch.cat(tuple(attribute_feats[name] for name in modalities), dim=1)

        score_seq = model(attribute_feats)
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
