import os
import functools
import time
import glob
import argparse

import numpy as np
import joblib
import yaml

from blocks.core import utils, labels, duplocorpus
from blocks.estimation import imageprocessing, models, render, metrics


def removeBackground(image, foreground_mask, replace_with=None):
    if replace_with is None:
        replace_with = np.zeros_like(image)

    new_image = image.copy().astype(float)
    new_image[~foreground_mask] = replace_with[~foreground_mask]

    return new_image


def getUniqueTrialIds(dir_path):
    trial_ids = set(
        int(os.path.basename(fn).split('-')[1].split('_')[0])
        for fn in glob.glob(os.path.join(dir_path, f"trial-*.pkl"))
    )
    return sorted(tuple(trial_ids))


def main(
        out_dir=None, data_dir=None, preprocess_dir=None, detections_dir=None,
        data_scores_dir=None, keyframes_dir=None,
        num_seqs=None, only_task_ids=None, resume=None, num_folds=None,
        scores_run_name=None, keyframe_model_name=None, reselect_keyframes=None,
        subsample_period=None, window_size=None, corpus_name=None, debug=None,
        remove_skin=None, remove_background=None,
        default_annotator=None, cv_scheme=None, model_config=None, overwrite=None,
        legacy_mode=None):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    preprocess_dir = os.path.expanduser(preprocess_dir)
    detections_dir = os.path.expanduser(detections_dir)
    if data_scores_dir is not None:
        data_scores_dir = os.path.expanduser(data_scores_dir)
    if keyframes_dir is not None:
        keyframes_dir = os.path.expanduser(keyframes_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    if overwrite is None:
        overwrite = debug

    if legacy_mode:
        model_config['decode_kwargs']['legacy_mode'] = legacy_mode

    def loadFromDataDir(var_name):
        return joblib.load(os.path.join(data_dir, f"{var_name}.pkl"))

    def loadFromPreprocessDir(var_name):
        return joblib.load(os.path.join(preprocess_dir, f"{var_name}.pkl"))

    def loadFromDetectionsDir(var_name):
        return joblib.load(os.path.join(detections_dir, f"{var_name}.pkl"))

    def loadFromKeyframesDir(var_name):
        return joblib.load(os.path.join(keyframes_dir, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = getUniqueTrialIds(detections_dir)
    corpus = duplocorpus.DuploCorpus(corpus_name)

    if num_seqs is not None and num_seqs > 0:
        logger.info(f"Ignoring all but the first {num_seqs} videos")
        trial_ids = trial_ids[:num_seqs]

    logger.info(f"Loading data...")
    kept_trial_ids = []
    rgb_keyframe_seqs = []
    depth_keyframe_seqs = []
    seg_keyframe_seqs = []
    background_keyframe_seqs = []
    assembly_keyframe_seqs = []
    assembly_seqs = []
    label_keyframe_seqs = []
    foreground_mask_seqs = []
    for seq_idx, trial_id in enumerate(trial_ids):
        try:
            trial_str = f"trial-{trial_id}"

            rgb_frame_seq = loadFromDataDir(f'{trial_str}_rgb-frame-seq')
            depth_frame_seq = loadFromDataDir(f'{trial_str}_depth-frame-seq')
            rgb_timestamp_seq = loadFromDataDir(f'{trial_str}_rgb-frame-timestamp-seq')
            action_seq = loadFromDataDir(f'{trial_str}_action-seq')

            foreground_mask_seq = loadFromPreprocessDir(
                f'{trial_str}_foreground-mask-seq_no-ref-model'
            )
            background_plane_seq = loadFromPreprocessDir(f'{trial_str}_background-plane-seq')

            # FIXME: I need a better way of handling this. For child data it's
            #   better to get segments from block detections, but for the easy
            #   dataset it's better to use the original foreground segments.
            if legacy_mode:
                segment_seq = loadFromDetectionsDir(f'{trial_str}_block-segment-frame-seq')
            else:
                segment_seq = loadFromPreprocessDir(f'{trial_str}_segment-frame-seq')
            label_frame_seq = loadFromDetectionsDir(f'{trial_str}_class-label-frame-seq')

            assembly_seq = labels.parseLabelSeq(action_seq, timestamps=rgb_timestamp_seq)
            assembly_seq[-1].end_idx = len(rgb_frame_seq) - 1
        except FileNotFoundError as e:
            logger.warning(e)
            logger.info(f"Skipping trial {trial_id} --- no data in {scores_run_name}")
            continue

        task_id = corpus.getTaskIndex(trial_id)
        assembly_seqs.append(assembly_seq)

        if keyframes_dir is not None:
            keyframe_idxs = loadFromKeyframesDir(f'{trial_str}_keyframe-idxs')
            assembly_seq = labels.resampleStateSeq(keyframe_idxs, assembly_seq)
            rgb_frame_seq = rgb_frame_seq[keyframe_idxs]
            depth_frame_seq = depth_frame_seq[keyframe_idxs]
            segment_seq = segment_seq[keyframe_idxs]
            background_plane_seq = tuple(
                background_plane_seq[i] for i in keyframe_idxs
            )
            label_frame_seq = label_frame_seq[keyframe_idxs]
            foreground_mask_seq = foreground_mask_seq[keyframe_idxs]

        if not only_task_ids or task_id in only_task_ids:
            rgb_keyframe_seqs.append(rgb_frame_seq)
            depth_keyframe_seqs.append(depth_frame_seq)
            seg_keyframe_seqs.append(segment_seq)
            background_keyframe_seqs.append(background_plane_seq)
            assembly_keyframe_seqs.append(assembly_seq)
            label_keyframe_seqs.append(label_frame_seq)
            foreground_mask_seqs.append(foreground_mask_seq)
            kept_trial_ids.append(trial_id)
    trial_ids = kept_trial_ids

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
        hmm = models.EmpiricalImageHmm(**model_config['init_kwargs'])
        logger.info(f"  Training model on {len(train_assembly_seqs)} sequences...")
        hmm.fit(train_assembly_seqs, **model_config['fit_kwargs'])
        logger.info(f'    Model trained on {hmm.num_states} unique assembly states')
        saveToWorkingDir(hmm, f'hmm-fold0')
        cv_folds = [(tuple(range(len(child_trial_ids))), tuple(range(len(trial_ids))))]

    num_cv_folds = len(cv_folds)
    saveToWorkingDir(cv_folds, f'cv-folds')
    total_correct = 0
    total_items = 0
    for fold_index, (train_idxs, test_idxs) in enumerate(cv_folds):
        if num_folds is not None and fold_index >= num_folds:
            break

        logger.info(f"CV FOLD {fold_index + 1} / {num_cv_folds}")

        # Initialize and train model
        if cv_scheme == 'train on child':
            pass
        else:
            utils.validateCvFold(train_idxs, test_idxs)
            selectTrain = functools.partial(utils.select, train_idxs)
            # train_trial_ids = selectTrain(trial_ids)
            train_assembly_seqs = selectTrain(assembly_keyframe_seqs)
            hmm = models.EmpiricalImageHmm(**model_config['init_kwargs'])
            logger.info(f"  Training model on {len(train_idxs)} sequences...")
            hmm.fit(train_assembly_seqs, **model_config['fit_kwargs'])
            logger.info(f'    Model trained on {hmm.num_states} unique assembly states')
            saveToWorkingDir(hmm, f'hmm-fold{fold_index}')

        # Decode on the test set
        logger.info(f"  Testing model on {len(test_idxs)} sequences...")
        for i, test_index in enumerate(test_idxs):

            trial_id = trial_ids[test_index]
            rgb_frame_seq = rgb_keyframe_seqs[test_index]
            depth_frame_seq = depth_keyframe_seqs[test_index]
            seg_frame_seq = seg_keyframe_seqs[test_index]
            background_plane_seq = background_keyframe_seqs[test_index]
            true_assembly_seq = assembly_keyframe_seqs[test_index]
            true_assembly_seq_orig = assembly_seqs[test_index]
            label_frame_seq = label_keyframe_seqs[test_index]
            foreground_mask_seq = foreground_mask_seqs[test_index]

            if data_scores_dir is not None:
                data_scores = joblib.load(
                    os.path.join(data_scores_dir, f"trial-{trial_id}_data-scores.pkl")
                )
            else:
                data_scores = None

            rgb_frame_seq = tuple(
                imageprocessing.saturateImage(
                    rgb_image, background_mask=~foreground_mask,
                    remove_background=remove_background
                )
                for rgb_image, foreground_mask in zip(rgb_frame_seq, foreground_mask_seq)
            )

            depth_bkgrd_frame_seq = tuple(
                render.renderPlane(
                    background_plane,
                    camera_params=render.intrinsic_matrix,
                    camera_pose=render.camera_pose,
                    plane_appearance=render.object_colors[0]
                )[1]
                for background_plane in background_plane_seq
            )

            depth_frame_seq = tuple(
                removeBackground(depth_image, foreground_mask, replace_with=depth_bkgrd)
                for depth_image, foreground_mask, depth_bkgrd
                in zip(depth_frame_seq, foreground_mask_seq, depth_bkgrd_frame_seq)
            )

            # FIXME: This is a really hacky way of dealing with the fact that
            #   fitScene takes a background plane but stateLogLikelihood takes
            #   a background plane IMAGE
            if legacy_mode:
                background_seq = depth_bkgrd_frame_seq
            else:
                background_seq = background_plane_seq

            logger.info(f'    Decoding video {trial_id}...')

            num_oov = sum(int(s not in hmm.states) for s in true_assembly_seq)
            logger.info(f"    {num_oov} out-of-vocabulary states in ground-truth")

            start_time = time.process_time()
            ret = hmm.predictSeq(
                rgb_frame_seq, depth_frame_seq, seg_frame_seq, label_frame_seq,
                background_seq, log_likelihoods=data_scores,
                **model_config['decode_kwargs']
            )
            pred_assembly_seq, pred_idx_seq, max_log_probs, log_likelihoods, poses_seq = ret
            end_time = time.process_time()
            logger.info(utils.makeProcessTimeStr(end_time - start_time))

            if data_scores_dir is not None:
                # FIXME: I only save the pose of the best sequence, but I should
                # save all of them
                # poses_seq = joblib.load(
                #     os.path.join(data_scores_dir, f"trial-{trial_id}_poses-seq.pkl")
                # )
                if legacy_mode:
                    poses_seq = tuple(
                        ((0, np.zeros(2)),) * len(s.connected_components)
                        for s in pred_assembly_seq
                    )
                else:
                    poses_seq = tuple(
                        ((np.eye(3), np.zeros(3)),) * len(s.connected_components)
                        for s in pred_assembly_seq
                    )

            if len(pred_assembly_seq) == len(true_assembly_seq):
                num_correct, num_total = metrics.numberCorrect(true_assembly_seq, pred_assembly_seq)
                logger.info(f'    ACCURACY: {num_correct} / {num_total}')
                total_correct += num_correct
                total_items += num_total
            else:
                logger.info(
                    f"    Skipping accuracy computation: "
                    f"{len(pred_assembly_seq)} pred states != "
                    f"{len(true_assembly_seq)} gt states"
                )

            # Save intermediate results
            logger.info(f"Saving output...")
            trial_str = f"trial-{trial_id}"
            saveToWorkingDir(true_assembly_seq_orig, f'{trial_str}_true-state-seq-orig')
            saveToWorkingDir(true_assembly_seq, f'{trial_str}_true-state-seq')
            saveToWorkingDir(pred_assembly_seq, f'{trial_str}_pred-state-seq')
            saveToWorkingDir(poses_seq, f'{trial_str}_poses-seq')
            saveToWorkingDir(max_log_probs, f'{trial_str}_viterbi-scores')
            saveToWorkingDir(log_likelihoods, f'{trial_str}_data-scores')

            # Save figures
            if legacy_mode:
                renders = tuple(
                    render.makeFinalRender(
                        p, assembly=a,
                        rgb_background=np.zeros_like(rgb),
                        depth_background=depth_bkgrd,
                        camera_pose=render.camera_pose,
                        camera_params=render.intrinsic_matrix,
                        block_colors=render.object_colors
                    )
                    for p, a, rgb, depth, depth_bkgrd in zip(
                        poses_seq, pred_assembly_seq, rgb_frame_seq, depth_frame_seq,
                        depth_bkgrd_frame_seq
                    )
                )
                rgb_rendered_seq, depth_rendered_seq, label_rendered_seq = tuple(
                    zip(*renders)
                )
                gt_poses_seq = tuple(
                    ((0, np.zeros(2)),) * len(s.connected_components)
                    for s in true_assembly_seq
                )
                renders = tuple(
                    render.makeFinalRender(
                        p, assembly=a,
                        rgb_background=np.zeros_like(rgb),
                        depth_background=depth_bkgrd,
                        camera_pose=render.camera_pose,
                        camera_params=render.intrinsic_matrix,
                        block_colors=render.object_colors
                    )
                    for p, a, rgb, depth, depth_bkgrd in zip(
                        gt_poses_seq, true_assembly_seq, rgb_frame_seq, depth_frame_seq,
                        depth_bkgrd_frame_seq
                    )
                )
                rgb_rendered_seq_gt, depth_rendered_seq_gt, label_rendered_seq_gt = tuple(
                    zip(*renders)
                )
            else:
                rgb_rendered_seq, depth_rendered_seq, label_rendered_seq = utils.batchProcess(
                    render.renderScene,
                    background_plane_seq, pred_assembly_seq, poses_seq,
                    static_kwargs={
                        'camera_pose': render.camera_pose,
                        'camera_params': render.intrinsic_matrix,
                        'object_appearances': render.object_colors
                    },
                    unzip=True
                )
                gt_poses_seq = tuple(
                    ((np.eye(3), np.zeros(3)),) * len(s.connected_components)
                    for s in true_assembly_seq
                )
                renders = utils.batchProcess(
                    render.renderScene,
                    background_plane_seq, true_assembly_seq, gt_poses_seq,
                    static_kwargs={
                        'camera_pose': render.camera_pose,
                        'camera_params': render.intrinsic_matrix,
                        'object_appearances': render.object_colors
                    },
                    unzip=True
                )
                rgb_rendered_seq_gt, depth_rendered_seq_gt, label_rendered_seq_gt = renders
            if utils.in_ipython_console():
                file_path = None
            else:
                trial_str = f"trial-{trial_id}"
                file_path = os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            diff_images = tuple(np.abs(f - r) for f, r in zip(rgb_frame_seq, rgb_rendered_seq))
            imageprocessing.displayImages(
                *rgb_frame_seq, *diff_images, *rgb_rendered_seq, *rgb_rendered_seq_gt,
                *seg_frame_seq, *label_frame_seq,
                num_rows=6, file_path=file_path
            )
    logger.info(f'AVG ACCURACY: {total_correct / total_items * 100: .1f}%')


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file')
    parser.add_argument('--out_dir')
    parser.add_argument('--data_dir')
    parser.add_argument('--preprocess_dir')
    parser.add_argument('--detections_dir')
    parser.add_argument('--keyframes_dir')
    parser.add_argument('--data_scores_dir')
    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}

    # Load config file and override with any provided command line args
    config_file_path = args.pop('config_file', None)
    if config_file_path is None:
        file_basename = utils.stripExtension(__file__)
        config_fn = f"{file_basename}.yaml"
        config_file_path = os.path.expanduser(
            os.path.join(
                '~', 'repo', 'blocks', 'blocks', 'estimation', 'scripts', 'config',
                config_fn
            )
        )
    else:
        config_fn = os.path.basename(config_file_path)
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
