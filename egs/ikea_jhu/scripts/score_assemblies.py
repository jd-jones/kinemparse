import os
import collections
import logging

import yaml
import torch
import joblib
import numpy as np

from mathtools import utils, torchutils, metrics
from kinemparse import assembly as lib_assembly
import make_data


logger = logging.getLogger(__name__)


class DummyClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_set_size, binary_multiclass=False):
        super().__init__()

        self.input_dim = input_dim
        self.out_set_size = out_set_size

        logger.info(
            f'Initialized dummy classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        return input_seq.transpose(1, 2)

    def predict(self, outputs):
        __, preds = torch.max(outputs, -1)
        return preds


def adjacencies(assembly, link_names, lower_tri_only=False):
    name_to_index = {name: i for i, name in enumerate(link_names)}
    num_links = len(link_names)
    adjacencies = torch.zeros(num_links, num_links, dtype=torch.bool)
    for joint in assembly.joints.values():
        parent_index = name_to_index[joint.parent_name]
        child_index = name_to_index[joint.child_name]
        adjacencies[parent_index, child_index] = True
        adjacencies[child_index, parent_index] = True

    if lower_tri_only:
        rows, cols = np.tril_indices(adjacencies.shape[1], k=-1)
        adjacencies = adjacencies[rows, cols]

    return adjacencies


class AssemblyClassifier(torch.nn.Module):
    def __init__(
            self, assemblies, link_names,
            scale=1, alpha=1, update_params=False, eq_classes=None):
        super().__init__()

        self.assembly_vocab = assemblies
        self.link_vocab = link_names
        self.eq_classes = torch.tensor(eq_classes, dtype=torch.float)

        self._scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=torch.float),
            requires_grad=update_params
        )
        self._alpha = torch.nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=update_params
        )

        self._part_pair_names = make_data.makePairs(link_names, lower_tri_only=True)
        self._pair_is_possible = make_data.possibleConnections(self._part_pair_names)

    def forward(self, input_seq):
        output_seq = tuple(self._score_assembly(input_seq, a) for a in self.assembly_vocab)
        output_seq = torch.stack(output_seq, axis=-1)

        if self.eq_classes is not None:
            output_seq = output_seq @ self.eq_classes

        return output_seq

    def _score_assembly(self, input_seq, assembly):
        """
        Parameters
        ----------
        input_seq : torch.Tensor, shape (batch_size, num_edges, num_samples, num_features)
        assembly : assembly.Assembly

        Returns
        -------
        scores : torch.Tensor, shape (batch_size, num_samples)
        """
        input_seq = input_seq.sum(axis=-1)
        edge_is_observed = ~torch.isnan(input_seq)
        input_seq[~edge_is_observed] = 0

        edge_is_present = adjacencies(assembly, self.link_vocab, lower_tri_only=True)
        edge_is_present = edge_is_present[self._pair_is_possible]

        if not edge_is_present.any():
            shape = (input_seq.shape[0], input_seq.shape[-1])
            edge_scores = torch.zeros(shape, dtype=torch.float)
        else:
            edge_features = input_seq[:, edge_is_present, :]
            edge_scores = torch.sum(
                -edge_features * edge_is_observed[:, edge_is_present, :],
                axis=1
            )

        if not (~edge_is_present).any():
            shape = (input_seq.shape[0], input_seq.shape[-1])
            no_edge_scores = torch.zeros(shape, dtype=torch.float)
        else:
            no_edge_features = input_seq[:, ~edge_is_present, :]
            no_edge_scores = torch.sum(
                torch.ones_like(no_edge_features) * edge_is_observed[:, ~edge_is_present, :],
                axis=1
            )

        scores = self._scale * edge_scores + self._alpha * no_edge_scores
        return scores

    def predict(self, outputs):
        __, preds = torch.max(outputs, -1)
        return preds


def makeEqClasses(assembly_vocab, part_symmetries):
    def renameLink(assembly, old_name, new_name):
        link_dict = {}
        for link in assembly.links.values():
            new_link = lib_assembly.Link(
                link.name.replace(old_name, new_name),
                pose=link.pose
            )
            if new_link.name in link_dict:
                if link_dict[new_link.name] != new_link:
                    raise AssertionError()
                continue
            else:
                link_dict[new_link.name] = new_link
        new_links = list(link_dict.values())

        new_joints = [
            lib_assembly.Joint(
                joint.name.replace(old_name, new_name),
                joint.joint_type,
                joint.parent_name.replace(old_name, new_name),
                joint.child_name.replace(old_name, new_name),
                transform=joint._transform
            )
            for joint in assembly.joints.values()
        ]

        new_assembly = lib_assembly.Assembly(links=new_links, joints=new_joints)
        return new_assembly

    eq_class_vocab = []
    assembly_eq_classes = []
    for vocab_index, assembly in enumerate(assembly_vocab):
        for new_name, old_names in part_symmetries.items():
            for old_name in old_names:
                assembly = renameLink(assembly, old_name, new_name)
        eq_class_index = utils.getIndex(assembly, eq_class_vocab)
        assembly_eq_classes.append(eq_class_index)

    assembly_eq_classes = np.array(assembly_eq_classes)

    eq_classes = np.zeros((len(assembly_vocab), len(eq_class_vocab)), dtype=float)
    for vocab_index, eq_class_index in enumerate(assembly_eq_classes):
        eq_classes[vocab_index, eq_class_index] = 1
    eq_classes /= eq_classes.sum(axis=0)

    # import pdb; pdb.set_trace()

    return eq_classes, assembly_eq_classes, eq_class_vocab


def main(
        out_dir=None, data_dir=None, model_name=None, part_symmetries=None,
        gpu_dev_id=None, batch_size=None, learning_rate=None,
        model_params={}, cv_params={}, train_params={}, viz_params={},
        plot_predictions=None, results_file=None, sweep_param_name=None):

    if part_symmetries is None:
        part_symmetries = {
            'beam_side': (
                'backbeam_hole_1', 'backbeam_hole_2',
                'frontbeam_hole_1', 'frontbeam_hole_2'
            ),
            'beam_top': ('backbeam_hole_3', 'frontbeam_hole_3'),
            'backrest': ('backrest_hole_1', 'backrest_hole_2')
        }

    data_dir = os.path.expanduser(data_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

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

    def saveVariable(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f'{var_name}.pkl'))

    def loadAll(seq_ids, var_name, data_dir):
        def loadOne(seq_id):
            fn = os.path.join(data_dir, f'trial={seq_id}_{var_name}')
            return joblib.load(fn)
        return tuple(map(loadOne, seq_ids))

    # Load vocab
    with open(os.path.join(data_dir, "part-vocab.yaml"), 'rt') as f:
        link_vocab = yaml.safe_load(f)
    assembly_vocab = joblib.load(os.path.join(data_dir, 'assembly-vocab.pkl'))

    # Load data
    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=')
    feature_seqs = loadAll(trial_ids, 'feature-seq.pkl', data_dir)
    label_seqs = loadAll(trial_ids, 'label-seq.pkl', data_dir)

    if part_symmetries:
        # Construct equivalence classes from vocab
        eq_classes, assembly_eq_classes, eq_class_vocab = makeEqClasses(
            assembly_vocab, part_symmetries
        )
        lib_assembly.writeAssemblies(
            os.path.join(fig_dir, 'eq-class-vocab.txt'),
            eq_class_vocab
        )
        label_seqs = tuple(assembly_eq_classes[label_seq] for label_seq in label_seqs)
        saveVariable(eq_class_vocab, 'assembly-vocab')
    else:
        eq_classes = None

    def impute_nan(input_seq):
        input_is_nan = np.isnan(input_seq)
        logger.info(f"{input_is_nan.sum()} NaN elements")
        input_seq[input_is_nan] = 0  # np.nanmean(input_seq)
        return input_seq

    # feature_seqs = tuple(map(impute_nan, feature_seqs))

    for trial_id, label_seq, feat_seq in zip(trial_ids, label_seqs, feature_seqs):
        saveVariable(feat_seq, f"trial={trial_id}_feature-seq")
        saveVariable(label_seq, f"trial={trial_id}_label-seq")

    device = torchutils.selectDevice(gpu_dev_id)

    # Define cross-validation folds
    dataset_size = len(trial_ids)
    cv_folds = utils.makeDataSplits(dataset_size, **cv_params)

    def getSplit(split_idxs):
        split_data = tuple(
            tuple(s[i] for i in split_idxs)
            for s in (feature_seqs, label_seqs, trial_ids)
        )
        return split_data

    for cv_index, cv_splits in enumerate(cv_folds):
        train_data, val_data, test_data = tuple(map(getSplit, cv_splits))

        train_feats, train_labels, train_ids = train_data
        train_set = torchutils.SequenceDataset(
            train_feats, train_labels,
            device=device, labels_dtype=torch.long, seq_ids=train_ids,
            transpose_data=True
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )

        test_feats, test_labels, test_ids = test_data
        test_set = torchutils.SequenceDataset(
            test_feats, test_labels,
            device=device, labels_dtype=torch.long, seq_ids=test_ids,
            transpose_data=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        val_feats, val_labels, val_ids = val_data
        val_set = torchutils.SequenceDataset(
            val_feats, val_labels,
            device=device, labels_dtype=torch.long, seq_ids=val_ids,
            transpose_data=True
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

        logger.info(
            f'CV fold {cv_index + 1} / {len(cv_folds)}: {len(trial_ids)} total '
            f'({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)'
        )

        input_dim = train_set.num_obsv_dims
        output_dim = train_set.num_label_types
        if model_name == 'linear':
            model = torchutils.LinearClassifier(
                input_dim, output_dim, **model_params
            ).to(device=device)
        elif model_name == 'dummy':
            model = DummyClassifier(input_dim, output_dim, **model_params)
        elif model_name == 'AssemblyClassifier':
            model = AssemblyClassifier(
                assembly_vocab, link_vocab, eq_classes=eq_classes,
                **model_params
            )
        else:
            raise AssertionError()

        criterion = torch.nn.CrossEntropyLoss()
        if model_name != 'dummy':
            train_epoch_log = collections.defaultdict(list)
            val_epoch_log = collections.defaultdict(list)
            metric_dict = {
                'Avg Loss': metrics.AverageLoss(),
                'Accuracy': metrics.Accuracy(),
                'Precision': metrics.Precision(),
                'Recall': metrics.Recall(),
                'F1': metrics.Fmeasure()
            }

            optimizer_ft = torch.optim.Adam(
                model.parameters(), lr=learning_rate,
                betas=(0.9, 0.999), eps=1e-08,
                weight_decay=0, amsgrad=False
            )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=1.00)

            model, last_model_wts = torchutils.trainModel(
                model, criterion, optimizer_ft, lr_scheduler,
                train_loader, val_loader,
                device=device,
                metrics=metric_dict,
                train_epoch_log=train_epoch_log,
                val_epoch_log=val_epoch_log,
                **train_params
            )

        logger.info(f'scale={float(model._scale)}')
        logger.info(f'alpha={float(model._alpha)}')

        # Test model
        metric_dict = {
            'Avg Loss': metrics.AverageLoss(),
            'Accuracy': metrics.Accuracy(),
            'Precision': metrics.Precision(),
            'Recall': metrics.Recall(),
            'F1': metrics.Fmeasure()
        }
        test_io_history = torchutils.predictSamples(
            model.to(device=device), test_loader,
            criterion=criterion, device=device,
            metrics=metric_dict, data_labeled=True, update_model=False,
            seq_as_batch=train_params['seq_as_batch'],
            return_io_history=True
        )

        metric_str = '  '.join(str(m) for m in metric_dict.values())
        logger.info('[TST]  ' + metric_str)

        d = {k: v.value for k, v in metric_dict.items()}
        utils.writeResults(results_file, d, sweep_param_name, model_params)

        if plot_predictions:
            io_fig_dir = os.path.join(fig_dir, 'model-io')
            if not os.path.exists(io_fig_dir):
                os.makedirs(io_fig_dir)

            label_names = ('gt', 'pred')
            preds, scores, inputs, gt_labels, ids = zip(*test_io_history)
            for batch in test_io_history:
                batch = tuple(
                    x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    for x in batch
                )
                for preds, _, inputs, gt_labels, seq_id in zip(*batch):
                    fn = os.path.join(io_fig_dir, f"trial={seq_id}_model-io.png")
                    utils.plot_array(
                        inputs.sum(axis=-1), (gt_labels, preds), label_names,
                        fn=fn, **viz_params
                    )

        def saveTrialData(pred_seq, score_seq, feat_seq, label_seq, trial_id):
            saveVariable(pred_seq, f'trial={trial_id}_pred-label-seq')
            saveVariable(score_seq, f'trial={trial_id}_score-seq')
            saveVariable(label_seq, f'trial={trial_id}_true-label-seq')
        for batch in test_io_history:
            batch = tuple(
                x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                for x in batch
            )
            for io in zip(*batch):
                saveTrialData(*io)

        saveVariable(train_ids, f'cvfold={cv_index}_train-ids')
        saveVariable(test_ids, f'cvfold={cv_index}_test-ids')
        saveVariable(val_ids, f'cvfold={cv_index}_val-ids')
        saveVariable(train_epoch_log, f'cvfold={cv_index}_{model_name}-train-epoch-log')
        saveVariable(val_epoch_log, f'cvfold={cv_index}_{model_name}-val-epoch-log')
        saveVariable(metric_dict, f'cvfold={cv_index}_{model_name}-metric-dict')
        saveVariable(model, f'cvfold={cv_index}_{model_name}-best')

        train_fig_dir = os.path.join(fig_dir, 'train-plots')
        if not os.path.exists(train_fig_dir):
            os.makedirs(train_fig_dir)

        if train_epoch_log:
            torchutils.plotEpochLog(
                train_epoch_log,
                subfig_size=(10, 2.5),
                title='Training performance',
                fn=os.path.join(train_fig_dir, f'cvfold={cv_index}_train-plot.png')
            )

        if val_epoch_log:
            torchutils.plotEpochLog(
                val_epoch_log,
                subfig_size=(10, 2.5),
                title='Heldout performance',
                fn=os.path.join(train_fig_dir, f'cvfold={cv_index}_val-plot.png')
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
