import os
import functools
import time
import argparse
import pdb

import yaml
import joblib
import scipy
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch_struct

import numpy as np
from mathtools import utils, metrics, torchutils
from kinemparse import models, airplanecorpus


def assign_hands(hand_detections):
    new_detections = np.zeros_like(hand_detections)
    hand_a = hand_detections[0, :2]
    # hand_b = hand_detections[0, 2:]
    for t, detections in enumerate(hand_detections):
        if np.isnan(detections).any():
            new_detections[t, :] = np.nan

        loc_a = detections[:2]
        loc_b = detections[2:]
        aa = np.linalg.norm(hand_a - loc_a)
        ba = np.linalg.norm(hand_a - loc_b)
        # ab = np.linalg.norm(hand_b - loc_a)
        # bb = np.linalg.norm(hand_b - loc_b)
        if aa < ba:
            hand_a = loc_a
            # hand_b = loc_b
            new_detections[t, :2] = loc_a
            new_detections[t, 2:] = loc_b
        else:
            # hand_b = loc_a
            hand_a = loc_b
            new_detections[t, :2] = loc_b
            new_detections[t, 2:] = loc_a
    return new_detections


def main(
        config_path=None, out_dir=None, scores_dir=None, airplane_corpus_dir=None,
        subsample_period=None, window_size=None, corpus_name=None, debug=None,
        default_annotator=None, cv_scheme=None, model_config=None, overwrite=None,
        ignore_objects_in_comparisons=None, gpu_dev_id=None,
        start_from_fold=None, max_folds=None, presegment=False, min_dur=None):

    out_dir = os.path.expanduser(out_dir)
    airplane_corpus_dir = os.path.expanduser(airplane_corpus_dir)
    if scores_dir is not None:
        scores_dir = os.path.expanduser(scores_dir)

    model_name = model_config.pop('model_name')

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    logger.info(f"Starting run")

    device = torchutils.selectDevice(gpu_dev_id)

    hand_detection_seqs, action_seqs, trial_ids, part_info = airplanecorpus.loadCorpus(
        airplane_corpus_dir, subsample_period=None, reaching_only=True, parse_actions=False,
        # airplane_corpus_dir, subsample_period=None, reaching_only=False, parse_actions=False,
        ignore_objects_in_comparisons=ignore_objects_in_comparisons,
    )

    part_idxs_to_models, model_names, model_names_to_idxs = airplanecorpus.loadModelAssignments()
    part_idxs_to_models = torch.tensor(part_idxs_to_models).float()

    # Split into train and test sets
    if cv_scheme == 'leave one out':
        num_seqs = len(trial_ids)
        cv_folds = []
        for i in range(num_seqs):
            test_fold = (i,)
            train_fold = tuple(range(0, i)) + tuple(range(i + 1, num_seqs))
            cv_folds.append((train_fold, test_fold))
    elif cv_scheme == 'test on train set':
        if not debug:
            err_str = "You can't test on the train set if you're not debugging!"
            raise ValueError(err_str)
        num_seqs = len(trial_ids)
        cv_folds = []
        for i in range(num_seqs):
            test_fold = (i,)
            train_fold = tuple(range(0, num_seqs))
            cv_folds.append((train_fold, test_fold))
        logger.warning(f"TESTING ON THE TRAINING SET---DON'T BELIEVE THESE NUMBERS")
    saveToWorkingDir(cv_folds, f'cv-folds')

    if start_from_fold is not None:
        cv_folds = cv_folds[start_from_fold:]

    if max_folds is not None:
        cv_folds = cv_folds[:max_folds]

    num_cv_folds = len(cv_folds)
    num_equivalent = 0
    for fold_index, (train_idxs, test_idxs) in enumerate(cv_folds):
        logger.info(f"CV FOLD {fold_index + 1} / {num_cv_folds}")
        if cv_scheme != 'test on train set':
            utils.validateCvFold(train_idxs, test_idxs)

        model = getattr(models, model_name)(
            *part_info, device=device, **model_config['init_kwargs']
        )

        selectTrain = functools.partial(utils.select, train_idxs)
        train_hand_detection_seqs = selectTrain(hand_detection_seqs)
        train_action_seqs = selectTrain(action_seqs)

        logger.info(f"  Training model on {len(train_idxs)} sequences...")
        model.fit(
            train_action_seqs, train_hand_detection_seqs,
            **model_config['fit_kwargs']
        )
        logger.info(f'    Model trained on {model.num_states} unique assembly states')

        logger.info(f"  Testing model on {len(test_idxs)} sequences...")
        for i, test_idx in enumerate(test_idxs):

            trial_id = trial_ids[test_idx]
            hand_detection_seq = hand_detection_seqs[test_idx]
            true_action_seq = action_seqs[test_idx]
            true_action_names = tuple(
                model._obsv_model._part_names[i] for i in true_action_seq[:, 0]
            )

            """
            hand_detection_seq = assign_hands(hand_detection_seq)
            f, axes = plt.subplots(4)
            axes[0].plot(hand_detection_seq[:, 0])
            axes[1].plot(hand_detection_seq[:, 1])
            axes[2].plot(hand_detection_seq[:, 2])
            axes[3].plot(hand_detection_seq[:, 3])
            # axis.plot(hand_detection_seq[:, 2], hand_detection_seq[:, 3])
            plt.savefig(os.path.join(fig_dir, f"hands-{trial_id}.png"))
            plt.close()
            continue
            """

            fst = model.showFST(action_dict=model._obsv_model._part_names)
            fst.render(os.path.join(fig_dir, f"fst-{trial_id}"), cleanup=True)

            detection_scores = torch.tensor(
                scipy.io.loadmat(
                    os.path.join(scores_dir, f'trial-{trial_id}-detection-scores.mat')
                )['detection_scores']
            ).float()

            num_samples, num_bins = detection_scores.shape
            _, axis = plt.subplots(1, figsize=(12, 4))
            for i in range(num_bins):
                axis.plot(detection_scores[:, i].numpy(), label=f"bin {i}")
            axis.set_title(f"Detections, {trial_id}")
            axis.legend()
            axis.grid()
            plt.savefig(os.path.join(fig_dir, f"detections-{trial_id}.png"))
            plt.close()

            def makeSegmentScores(sample_scores, min_dur=1):
                x = torch.any(sample_scores > 0, dim=1).int()
                changepoints = x[1:] - x[:-1]
                start_idxs = (torch.nonzero(changepoints == 1)[:, 0] + 1).tolist()
                end_idxs = (torch.nonzero(changepoints == -1)[:, 0] + 1).tolist()
                if x[0]:
                    start_idxs = [0] + start_idxs

                assert(len(start_idxs) == len(end_idxs))

                segment_scores = tuple(
                    sample_scores[seg_start:seg_end].mean(dim=0)
                    for seg_start, seg_end in zip(start_idxs, end_idxs)
                    if seg_end - seg_start > min_dur
                )

                return torch.stack(segment_scores)

            if presegment:
                try:
                    detection_scores = makeSegmentScores(detection_scores, min_dur=min_dur)
                except AssertionError as e:
                    logger.warning(e)
                    continue

            row_is_inf = torch.isinf(detection_scores[:, :-1]).all(-1)
            detection_scores = detection_scores[~row_is_inf, :]
            # logger.info(f"{int(row_is_inf.sum())} inf-valued rows in detection array")

            # duration_scores = None
            duration_scores = torch.tensor(
                scipy.io.loadmat(
                    os.path.join(scores_dir, f'trial-{trial_id}-duration-scores.mat')
                )['duration_scores']
            ).float()

            _, axes = plt.subplots(1, 2, figsize=(12, 12))
            axes[0].matshow(detection_scores)
            axes[0].set_ylabel('detection')
            axes[1].matshow(duration_scores)
            axes[0].set_ylabel('duration')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"scores-{trial_id}.png"))
            plt.close()

            logger.info(f'    Decoding video {trial_id}...')

            # start_time = time.process_time()
            pred_action_idxs, pred_scores = model.predictSeq(
                hand_detection_seq,
                obsv_scores=detection_scores, dur_scores=duration_scores,
                **model_config['decode_kwargs']
            )
            # end_time = time.process_time()
            # logger.info(utils.makeProcessTimeStr(end_time - start_time))

            semiring = torch_struct.LogSemiring()
            best_score = semiring.sum(pred_scores.max(dim=-1).values, dim=0)
            logger.info(f"    Path score: {best_score}")

            _, axes = plt.subplots(2, figsize=(12, 12))
            axes[0].matshow(pred_scores.numpy().T)
            axes[1].matshow(pred_scores.exp().numpy().T)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"marginals-{trial_id}.png"))
            plt.close()

            pdb.set_trace()

            pred_action_idx_segs, _ = utils.computeSegments(pred_action_idxs.tolist())
            pred_action_names = tuple(
                model._obsv_model._part_names[i]
                for i in pred_action_idx_segs
            )
            pred_action_names = tuple(a for a in pred_action_names if a != 'null')

            num_samples, num_bins = detection_scores.shape
            _, axes = plt.subplots(2, sharex=True, figsize=(10, 10))
            for i in range(num_bins):
                axes[0].plot(detection_scores[:, i].numpy(), label=f"bin {i}")
                axes[0].scatter(range(num_samples), detection_scores[:, i].numpy())
            axes[0].set_title(f"System I/O, {trial_id}")
            axes[0].legend()
            axes[0].grid()
            axes[-1].plot(pred_action_idxs.numpy())
            axes[-1].scatter(range(num_samples), pred_action_idxs.numpy())
            axes[-1].set_yticks(range(len(model._obsv_model._part_names)))
            axes[-1].set_yticklabels(model._obsv_model._part_names)
            axes[-1].grid()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"io-{trial_id}.png"))
            plt.close()

            raw_bin_idxs = detection_scores.argmax(-1).numpy()
            pred_bin_idxs = model._obsv_model._part_to_bin[pred_action_idxs].numpy()
            num_samples, num_bins = detection_scores.shape
            _, axis = plt.subplots(1, figsize=(10, 5))
            axis.plot(raw_bin_idxs, label='raw')
            axis.scatter(range(num_samples), raw_bin_idxs)
            axis.set_title(f"Bin predictions, {trial_id}")
            axis.plot(pred_bin_idxs, label='decode')
            axis.scatter(range(num_samples), pred_bin_idxs)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid()
            axis.legend()
            plt.savefig(os.path.join(fig_dir, f"bins-{trial_id}.png"))
            plt.close()

            last_state_pred = frozenset(pred_action_names)
            last_state_true = frozenset(true_action_names)
            residual = last_state_pred ^ last_state_true
            logger.info(f'    Actions, pred: {models.stateToString(pred_action_names)}')
            logger.info(f'    Actions, true: {models.stateToString(true_action_names)}')
            logger.info(f'    Errors: {models.stateToString(sorted(tuple(residual)))}')

            file_path = os.path.join(out_dir, f'{trial_id}-pred.txt')
            with open(file_path, "w") as text_file:
                for action in pred_action_names:
                    print(action, file=text_file)

            file_path = os.path.join(out_dir, f'{trial_id}-true.txt')
            with open(file_path, "w") as text_file:
                for action in true_action_names:
                    print(action, file=text_file)

            def getModel(action_names):
                model_names = ('a', 'b', 'c')
                model_names_to_idxs = {n: i for i, n in enumerate(model_names)}
                model_names_to_idxs['ab'] = [0, 1]
                model_counts = torch.zeros(3)
                for n in action_names:
                    if n.startswith('nose') or n.startswith('wing') or n.startswith('tail'):
                        model_name = n[:-1].split('_')[-1]
                        model_idx = model_names_to_idxs[model_name]
                        model_counts[model_idx] += 1

                best_model_idx = model_counts.argmax()
                best_model_name = model_names[best_model_idx]
                return best_model_name

            def predictModel():
                # model_scores = torch.einsum('tp,pm->tm', pred_scores, part_idxs_to_models)
                model_scores = semiring.matmul(pred_scores, part_idxs_to_models.log())
                model_scores = semiring.sum(model_scores, dim=0)
                best_model_idx = model_scores.argmax(-1)
                best_model_name = model_names[best_model_idx]
                return best_model_name

            true_model = getModel(true_action_names)
            pred_model = predictModel()
            models_match = true_model == pred_model
            # equiv_upto_optional = residual <= frozenset(['wheel1', 'wheel2'])
            logger.info(f"    MODEL CORRECT: {models_match}  (p {pred_model} | t {true_model})")
            num_equivalent += int(models_match)

            edit_dist = metrics.levenshtein(
                true_action_names, pred_action_names, segment_level=False
            )
            logger.info(
                f"    EDIT DISTANCE: {edit_dist}  "
                f"({len(true_action_names)} true, {len(pred_action_names)} pred)"
            )

            # Save intermediate results
            logger.info(f"Saving output...")
            saveToWorkingDir(true_action_seq, f'true_action_seq-{trial_id}')
            saveToWorkingDir(true_action_names, f'true_action_names-{trial_id}')
            saveToWorkingDir(pred_action_names, f'pred_action_names-{trial_id}')
            saveToWorkingDir(pred_action_idxs, f'pred_action_idxs-{trial_id}')

    num_seqs = len(trial_ids)
    final_score = num_equivalent / num_seqs
    logger.info(
        f"TOTAL: {final_score * 100:.2f}% ({num_equivalent} / {num_seqs})"
        " of final states correct"
    )


if __name__ == '__main__':
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
        config_file_path = os.path.expanduser(
            os.path.join('~', 'repo', 'kinemparse', 'scripts', 'config', config_fn)
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
