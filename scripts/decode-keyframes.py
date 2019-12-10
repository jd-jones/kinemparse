import os
import functools
import time
import argparse

import yaml
import joblib

from blocks.core import labels, duplocorpus
from mathtools import metrics, utils
from visiontools import imageprocessing, render
from kinemparse import videoprocessing, models


def main(
        out_dir=None, scores_dir=None, preprocessed_data_dir=None,
        keyframe_model_name=None,
        subsample_period=None, window_size=None, corpus_name=None,
        default_annotator=None, cv_scheme=None, model_name=None, model_config={},
        camera_params_config={}):

    out_dir = os.path.expanduser(out_dir)
    scores_dir = os.path.expanduser(scores_dir)
    preprocessed_data_dir = os.path.expanduser(preprocessed_data_dir)

    def loadFromWorkingDir(var_name):
        return joblib.load(os.path.join(scores_dir, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_dir, f"{var_name}.pkl"))

    # Load camera parameters from external file and add them to model config kwargs
    model_config['init_kwargs'].update(
        render.loadCameraParams(**camera_params_config, as_dict=True)
    )

    trial_ids = joblib.load(os.path.join(preprocessed_data_dir, 'trial_ids.pkl'))

    corpus = duplocorpus.DuploCorpus(corpus_name)
    assembly_seqs = tuple(
        labels.parseLabelSeq(corpus.readLabels(trial_id, default_annotator)[0])
        for trial_id in trial_ids
    )

    logger.info(f"Selecting keyframes...")
    keyframe_idx_seqs = []
    rgb_keyframe_seqs = []
    depth_keyframe_seqs = []
    seg_keyframe_seqs = []
    background_keyframe_seqs = []
    assembly_keyframe_seqs = []
    for seq_idx, trial_id in enumerate(trial_ids):
        trial_str = f"trial-{trial_id}"
        rgb_frame_seq = loadFromWorkingDir(f'{trial_str}_rgb-frame-seq')
        depth_frame_seq = loadFromWorkingDir(f'{trial_str}_depth-frame-seq')
        segment_seq = loadFromWorkingDir(f'{trial_str}_segment-seq')
        frame_scores = loadFromWorkingDir(f'{trial_str}_frame-scores')
        background_plane_seq = loadFromWorkingDir(f'{trial_str}_background-plane-seq')

        assembly_seq = assembly_seqs[seq_idx]
        # FIXME: Get the real frame index numbers instead of approximating
        assembly_seq[-1].end_idx = len(rgb_frame_seq) * subsample_period

        keyframe_idxs = videoprocessing.selectSegmentKeyframes(
            frame_scores, score_thresh=0, prepend_first=True
        )

        selectKeyframes = functools.partial(utils.select, keyframe_idxs)
        rgb_keyframe_seq = selectKeyframes(rgb_frame_seq)
        depth_keyframe_seq = selectKeyframes(depth_frame_seq)
        seg_keyframe_seq = selectKeyframes(segment_seq)
        background_keyframe_seq = selectKeyframes(background_plane_seq)

        # FIXME: Get the real frame index numbers instead of approximating
        keyframe_idxs_orig = keyframe_idxs * subsample_period
        assembly_keyframe_seq = labels.resampleStateSeq(keyframe_idxs_orig, assembly_seq)

        # Store all keyframe sequences in memory
        keyframe_idx_seqs.append(keyframe_idxs)
        rgb_keyframe_seqs.append(rgb_keyframe_seq)
        depth_keyframe_seqs.append(depth_keyframe_seq)
        seg_keyframe_seqs.append(seg_keyframe_seq)
        background_keyframe_seqs.append(background_keyframe_seq)
        assembly_keyframe_seqs.append(assembly_keyframe_seq)

    # Split into train and test sets
    if cv_scheme == 'leave one out':
        num_seqs = len(trial_ids)
        cv_folds = []
        for i in range(num_seqs):
            test_fold = (i,)
            train_fold = tuple(range(0, i)) + tuple(range(i + 1, num_seqs))
            cv_folds.append((train_fold, test_fold))
    elif cv_scheme == 'train on child':
        child_corpus = duplocorpus.DuploCorpus('child')
        child_trial_ids = utils.loadVariable('trial_ids', 'preprocess-all-data', 'child')
        train_assembly_seqs = tuple(
            labels.parseLabelSeq(child_corpus.readLabels(trial_id, 'Cathryn')[0])
            for trial_id in child_trial_ids
        )
        model = getattr(models, model_name)(**model_config['init_kwargs'])
        logger.info(f"  Training {model_name} on {len(train_assembly_seqs)} sequences...")
        model.fit(train_assembly_seqs, **model_config['fit_kwargs'])
        logger.info(f'    Model trained on {model.num_states} unique assembly states')
        saveToWorkingDir(model, f'model-fold0')
        cv_folds = [(tuple(range(len(child_trial_ids))), tuple(range(len(trial_ids))))]

    num_cv_folds = len(cv_folds)
    saveToWorkingDir(cv_folds, f'cv-folds')
    for fold_index, (train_idxs, test_idxs) in enumerate(cv_folds):
        logger.info(f"CV FOLD {fold_index + 1} / {num_cv_folds}")

        # Initialize and train model
        if cv_scheme == 'train on child':
            pass
        else:
            utils.validateCvFold(train_idxs, test_idxs)
            selectTrain = functools.partial(utils.select, train_idxs)
            # train_trial_ids = selectTrain(trial_ids)
            train_assembly_seqs = selectTrain(assembly_keyframe_seqs)
            model = getattr(models, model_name)(**model_config['init_kwargs'])
            logger.info(f"  Training {model_name} on {len(train_idxs)} sequences...")
            model.fit(train_assembly_seqs, **model_config['fit_kwargs'])
            logger.info(f'    Model trained on {model.num_states} unique assembly states')
            saveToWorkingDir(model, f'model-fold{fold_index}')

        # Decode on the test set
        selectTest = functools.partial(utils.select, test_idxs)
        test_trial_ids = selectTest(trial_ids)
        test_rgb_keyframe_seqs = selectTest(rgb_keyframe_seqs)
        test_depth_keyframe_seqs = selectTest(depth_keyframe_seqs)
        test_seg_keyframe_seqs = selectTest(seg_keyframe_seqs)
        test_background_keyframe_seqs = selectTest(background_keyframe_seqs)
        test_assembly_keyframe_seqs = selectTest(assembly_keyframe_seqs)
        test_assembly_seqs = selectTest(assembly_seqs)

        logger.info(f"  Testing model on {len(test_idxs)} sequences...")
        for i, trial_id in enumerate(test_trial_ids):
            rgb_frame_seq = test_rgb_keyframe_seqs[i]
            depth_frame_seq = test_depth_keyframe_seqs[i]
            seg_frame_seq = test_seg_keyframe_seqs[i]
            background_plane_seq = test_background_keyframe_seqs[i]
            true_assembly_seq = test_assembly_keyframe_seqs[i]
            true_assembly_seq_orig = test_assembly_seqs[i]

            rgb_frame_seq = tuple(
                imageprocessing.saturateImage(rgb_image, background_mask=segment_image == 0)
                for rgb_image, segment_image in zip(rgb_frame_seq, seg_frame_seq)
            )

            # NOTE: Casting uint16 -> float here --- maybe we want to make this
            #   happen somewhere more intuitive, like when the frame is loaded.
            depth_frame_seq = tuple(depth_image.astype(float) for depth_image in depth_frame_seq)

            rgb_background_seq, depth_background_seq = utils.batchProcess(
                model.renderPlane, background_plane_seq, unzip=True
            )

            logger.info(f'    Decoding video {trial_id}...')
            start_time = time.process_time()
            out = model.predictSeq(
                rgb_frame_seq, depth_frame_seq, seg_frame_seq,
                rgb_background_seq, depth_background_seq,
                **model_config['decode_kwargs']
            )
            pred_assembly_seq, pred_idx_seq, max_log_probs, log_likelihoods, poses_seq = out
            end_time = time.process_time()
            logger.info(utils.makeProcessTimeStr(end_time - start_time))

            num_correct, num_total = metrics.numberCorrect(true_assembly_seq, pred_assembly_seq)
            logger.info(f'    ACCURACY: {num_correct} / {num_total}')
            num_correct, num_total = metrics.numberCorrect(
                true_assembly_seq, pred_assembly_seq,
                ignore_empty_true=True
            )
            logger.info(f'    RECALL: {num_correct} / {num_total}')
            num_correct, num_total = metrics.numberCorrect(
                true_assembly_seq, pred_assembly_seq,
                ignore_empty_pred=True
            )
            logger.info(f'    PRECISION: {num_correct} / {num_total}')

            # Save intermediate results
            logger.info(f"Saving output...")
            saveToWorkingDir(segment_seq, f'segment_seq-{trial_id}')
            saveToWorkingDir(true_assembly_seq_orig, f'true_state_seq_orig-{trial_id}')
            saveToWorkingDir(true_assembly_seq, f'true_state_seq-{trial_id}')
            saveToWorkingDir(pred_assembly_seq, f'pred_state_seq-{trial_id}')
            saveToWorkingDir(poses_seq, f'poses_seq-{trial_id}')
            saveToWorkingDir(background_plane_seq, f'background_plane_seq-{trial_id}')
            saveToWorkingDir(max_log_probs, f'max_log_probs-{trial_id}')
            saveToWorkingDir(log_likelihoods, f'log_likelihoods-{trial_id}')

            # Save figures
            rgb_rendered_seq, depth_rendered_seq, label_rendered_seq = utils.batchProcess(
                model.renderScene,
                background_plane_seq, pred_assembly_seq, poses_seq,
                unzip=True
            )
            if utils.in_ipython_console():
                file_path = None
            else:
                trial_str = f"trial-{trial_id}"
                file_path = os.path.join(out_dir, f'{trial_str}_best-frames.png')
            imageprocessing.displayImages(
                *rgb_frame_seq, *rgb_rendered_seq, num_rows=2, file_path=file_path
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
            os.path.join('~', 'repo', 'kinemparse', 'scripts', config_fn)
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
