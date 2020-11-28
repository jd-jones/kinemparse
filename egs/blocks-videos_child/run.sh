#!/bin/bash
set -ue

eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/data/output/blocks/child-videos"

start_at="5"
stop_after="5"

data_dir="${output_dir}/raw-data"
background_dir="${output_dir}/background-detections"
detections_dir="${output_dir}/object-detections"
seg_labels_dir="${output_dir}/image-segment-labels"
sim_pretrain_dir="${output_dir}/pretrained-models-sim"
assembly_scores_dir="${output_dir}/assembly-scores_epochs=5"
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
        --learning_rate "0.001" \
        --model_name "Connections" \
        --train_params "{'num_epochs': 100, 'test_metric': 'F1', 'seq_as_batch': False}" \
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
        --batch_size "10" \
        --learning_rate "0.001" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}" \
        --train_params "{'num_epochs': 0, 'test_metric': 'F1', 'seq_as_batch': True}" \
        --num_disp_imgs "10" \
        --viz_params "{}"
    python analysis.py \
        --out_dir "${assembly_scores_dir}/system-performance" \
        --results_file "${assembly_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Decode best paths"
    python decode_keyframes.py \
        --config_file "$config_dir/decode_keyframes.yaml" \
        --out_dir "$decode_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data" \
        --detections_dir "$detections_dir/data" \
        --data_scores_dir "$register_dir/data"
    python score_results.py \
        --config_file "$config_dir/score_results.yaml" \
        --out_dir "$decode_dir/system-performance" \
        --decode_dir "$decode_dir/data"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
