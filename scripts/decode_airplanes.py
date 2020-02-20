import os
import functools
import time
import argparse

# import numpy as np
import yaml
import joblib
import scipy
# import sklearn
# from matplotlib import pyplot as plt

from mathtools import utils, metrics
from kinemparse import models, airplanecorpus
# from blocks.core import labels


def main(
        config_path=None, out_dir=None, airplane_corpus_dir=None,
        subsample_period=None, window_size=None, corpus_name=None, debug=None,
        default_annotator=None, cv_scheme=None, model_config=None, overwrite=None,
        ignore_objects_in_comparisons=None, max_folds=-1):

    out_dir = os.path.expanduser(out_dir)
    airplane_corpus_dir = os.path.expanduser(airplane_corpus_dir)

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

    hand_detection_seqs, action_seqs, trial_ids, part_info = airplanecorpus.loadCorpus(
        airplane_corpus_dir, subsample_period=None, reaching_only=True, parse_actions=False,
        ignore_objects_in_comparisons=ignore_objects_in_comparisons,
    )

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

    if max_folds > 0:
        cv_folds = cv_folds[:max_folds]

    num_cv_folds = len(cv_folds)
    num_equivalent = 0
    for fold_index, (train_idxs, test_idxs) in enumerate(cv_folds):
        logger.info(f"CV FOLD {fold_index + 1} / {num_cv_folds}")
        if cv_scheme != 'test on train set':
            utils.validateCvFold(train_idxs, test_idxs)

        selectTrain = functools.partial(utils.select, train_idxs)
        train_hand_detection_seqs = selectTrain(hand_detection_seqs)
        train_action_seqs = selectTrain(action_seqs)

        model = getattr(models, model_name)(*part_info, **model_config['init_kwargs'])

        train_hand_detection_seqs = selectTrain(hand_detection_seqs)

        logger.info(f"  Training process model on {len(train_idxs)} sequences...")
        model.fit(
            train_action_seqs, train_hand_detection_seqs,
            # action_means=action_means, action_covs=action_covs, bin_contents=bin_contents,
            **model_config['fit_kwargs']
        )
        logger.info(f'    Model trained on {model.num_states} unique assembly states')
        logger.info(f"  Testing model on {len(test_idxs)} sequences...")
        for i, test_idx in enumerate(test_idxs):

            trial_id = trial_ids[test_idx]
            hand_detection_seq = hand_detection_seqs[test_idx]
            true_action_seq = action_seqs[test_idx]

            true_assembly_seq = tuple(model._part_names[i] for i in true_action_seq[:, 0])

            detection_scores = scipy.io.loadmat(
                os.path.join(
                    airplane_corpus_dir, 'detection-scores',
                    f"detection-scores_{trial_id}.mat"
                )
            )['detection_scores']

            logger.info(f'    Decoding video {trial_id}...')

            start_time = time.process_time()
            pred_assembly_seq = model.predictSeq(
                hand_detection_seq, scores=detection_scores,
                **model_config['decode_kwargs']
            )[0]
            end_time = time.process_time()
            logger.info(utils.makeProcessTimeStr(end_time - start_time))

            last_state_pred = set(pred_assembly_seq)  # pred_assembly_seq[-1]
            last_state_true = set(true_assembly_seq)  # last_state_true = true_assembly_seq[-1]
            # residual = last_state_pred.assembly_state ^ last_state_true.assembly_state
            residual = last_state_pred ^ last_state_true
            # logger.info(f'    Last state, pred: {last_state_pred.assembly_state}')
            # logger.info(f'    Last state, true: {last_state_true.assembly_state}')
            logger.info(f'    Last state, pred: {last_state_pred}')
            logger.info(f'    Last state, true: {last_state_true}')
            logger.info(f'    Errors: {residual}')

            file_path = os.path.join(out_dir, f'{trial_id}-pred.txt')
            with open(file_path, "w") as text_file:
                for action in pred_assembly_seq:
                    print(action, file=text_file)

            file_path = os.path.join(out_dir, f'{trial_id}-true.txt')
            with open(file_path, "w") as text_file:
                for action in true_assembly_seq:
                    print(action, file=text_file)

            non_true_parts = residual  # - last_state_true.assembly_state
            equiv_upto_optional = non_true_parts <= set(['wheel1', 'wheel2', 'sticker'])
            logger.info(f"Last state correct: {equiv_upto_optional}")
            num_equivalent += int(equiv_upto_optional)

            true_assembly_segs, _ = utils.computeSegments(true_assembly_seq)
            pred_assembly_segs, _ = utils.computeSegments(pred_assembly_seq)
            edit_dist = metrics.levenshtein(
                true_assembly_segs, pred_assembly_segs, segment_level=False
            )
            logger.info(
                f"    EDIT DISTANCE: {edit_dist}  "
                f"({len(true_assembly_segs)} true, {len(pred_assembly_segs)} pred)"
            )

            # Save intermediate results
            logger.info(f"Saving output...")
            # saveToWorkingDir(true_assembly_seq_orig, f'true_state_seq_orig-{trial_id}')
            saveToWorkingDir(true_action_seq, f'true_action_seq-{trial_id}')
            saveToWorkingDir(true_assembly_seq, f'true_state_seq-{trial_id}')
            saveToWorkingDir(pred_assembly_seq, f'pred_state_seq-{trial_id}')

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
