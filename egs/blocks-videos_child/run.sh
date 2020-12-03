#!/bin/bash
set -ue

eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/data/output/blocks/child-videos"

start_at="6"
stop_after="6"

data_dir="${output_dir}/raw-data"
background_dir="${output_dir}/background-detections"
detections_dir="${output_dir}/object-detections"
seg_labels_dir="${output_dir}/image-segment-labels"
sim_pretrain_dir="${output_dir}/pretrained-models-sim"
assembly_scores_dir="${output_dir}/assembly-scores_epochs=50"
postprocessed_assembly_scores_dir="${output_dir}/assembly-scores-postprocessed_epochs=50"
seq_preds_dir="${output_dir}/seq-preds"
decode_dir="${output_dir}/decode"

cd ${scripts_dir}

STAGE=0

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Download video data"
    python download_blocks_data.py \
        --out_dir "${data_dir}" \
        --metadata_file "~/data/blocks/data/blocks_file_index.xlsx" \
        --metadata_criteria "{'GroupID': 'Child'}" \
        --corpus_name "child" \
        --default_annotator "Cathryn" \
        --use_annotated_keyframes "False" \
        --subsample_period '2' \
        --start_video_from_first_touch "True" \
        --save_video_before_first_touch "True" \
        --rgb_as_float "False" \
        --modalities "['video']"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Detect background"
    python detect_background.py \
        --out_dir "${background_dir}" \
        --data_dir "${data_dir}/data" \
        --background_data_dir "${data_dir}/data" \
        --depth_bg_detection_kwargs "{'plane_distance_thresh': 10, 'max_trials': 50}" \
        --rgb_bg_detection_kwargs "{'px_distance_thresh': 0.2}" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Detect objects"
    python detect_objects.py \
        --out_dir "${detections_dir}" \
        --data_dir "${data_dir}/data" \
        --gpu_dev_id "'0'" \
        --batch_size "2" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Label foreground segments"
    python make_seg_labels.py \
        --out_dir "${seg_labels_dir}" \
        --data_dir "${data_dir}/data" \
        --bg_masks_dir "${background_dir}/data" \
        --person_masks_dir "${detections_dir}/data" \
        --sat_thresh "0.15" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Pre-train assembly detector"
    python train_assembly_detector.py \
        --out_dir "${sim_pretrain_dir}" \
        --data_dir "${data_dir}/data" \
        --pretrain_dir "${sim_pretrain_dir}/data" \
        --gpu_dev_id "'2'" \
        --batch_size "10" \
        --learning_rate "0.0002" \
        --model_name "Connections" \
        --kornia_tfs "{ \
            'ColorJitter': { \
                'brightness': 0.1, \
                'contrast': 0.1, \
                'saturation': 0.0, \
                'hue': 0.1 \
            } \
        }" \
        --load_masks_params "{'masks_dir': '${detections_dir}/data', 'num_per_video': 10}" \
        --train_params "{'num_epochs': 1500, 'test_metric': 'F1', 'seq_as_batch': False}" \
        --model_params "{}" \
        --num_disp_imgs "10" \
        --viz_params "{}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Score assemblies"
    python score_assemblies.py \
        --out_dir "${assembly_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --segs_dir "${seg_labels_dir}/data" \
        --pretrained_model_dir "${sim_pretrain_dir}/data" \
        --gpu_dev_id "'2'" \
        --model_name "pretrained" \
        --batch_size "20" \
        --learning_rate "0.0002" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}" \
        --train_params "{'num_epochs': 50, 'test_metric': 'F1', 'seq_as_batch': True}" \
        --viz_params "{}"
    python analysis.py \
        --out_dir "${assembly_scores_dir}/system-performance" \
        --results_file "${assembly_scores_dir}/results.csv"
    python postprocess_assembly_scores.py \
        --out_dir "${postprocessed_assembly_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --segs_dir "${seg_labels_dir}/data" \
        --scores_dir "${assembly_scores_dir}/data" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict actions"
    python predict_seq_pytorch.py \
        --out_dir "${seq_preds_dir}" \
        --data_dir "${postprocessed_assembly_scores_dir}/data" \
        --feature_fn_format "score-seq.pkl" \
        --label_fn_format "true-label-seq.pkl" \
        --gpu_dev_id "'2'" \
        --model_name "'TCN'" \
        --batch_size "1" \
        --learning_rate "0.0002" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}" \
        --train_params "{'num_epochs': 250, 'test_metric': 'F1', 'seq_as_batch': 'seq mode'}" \
        --model_params "{ \
            'binary_multiclass': True, \
            'tcn_channels': [8,  8, 16, 16, 32, 32], \
            'kernel_size': 5, \
            'dropout': 0.2 \
        }" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${seq_preds_dir}/system-performance" \
        --results_file "${seq_preds_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
