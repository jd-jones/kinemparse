#! /bin/bash
set -ue

# --=(SET CONFIG OPTIONS)==----------------------------------------------------
# SET WHICH PROCESSING STAGES ARE RUN
start_at="7"
stop_after="7"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="$HOME/repo/kinemparse/data/output/segment-seqs"
activity_dir="${output_dir}/activity"
activity_data_dir="${activity_dir}/dataset"
activity_preds_dir="${activity_dir}/preds"
activity_smoothed_dir="${activity_dir}/preds-smoothed"
action_dir="${output_dir}/action_activity-labels=true_model=tcn"
action_data_dir="${action_dir}/dataset"
action_preds_dir="${action_dir}/preds"
action_smoothed_dir="${action_dir}/preds-smoothed"
keyframes_dir="${output_dir}/keyframes_activity-labels=true_action-labels=pred_smoothed"

# IMU DIRS --- READONLY
imu_attr_dir="$HOME/repo/kinemparse/data/output/block-connections-imu"
imu_data_dir="${imu_attr_dir}/connections-dataset_untrimmed"
attr_scores_dir="${imu_attr_dir}/predict-attributes_tcn_untrimmed"

# VIDEO DIRS --- READONLY
video_preprocess_dir="$HOME/repo/kinemparse/data/output/blocks_child_2020-05-04"
video_data_dir="${video_preprocess_dir}/raw-data"
keyframe_scores_dir="${video_preprocess_dir}/keyframes"


# --=(SCRIPT SETUP)==----------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=1


# --=(PHASE 1: PREDICT ACTIVITY)==---------------------------------------------
if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make activity dataset"
    python make_activity_dataset.py \
        --out_dir "${activity_data_dir}" \
        --video_data_dir "${video_data_dir}/data" \
        --imu_data_dir "${imu_data_dir}/data" \
        --video_seg_scores_dir "${keyframe_scores_dir}/data" \
        --imu_seg_scores_dir "${attr_scores_dir}/data" \
        --gt_keyframes_dir "${video_data_dir}/data" \
        --label_kwargs "{'use_end_label': False}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict activity"
    python predict_seq_pytorch.py \
        --config_file "${config_dir}/activity/predict_seq_pytorch.yaml" \
        --out_dir "${activity_preds_dir}" \
        --data_dir "${activity_data_dir}/data" \
        --gpu_dev_id "'0'"
    python analysis.py \
        --out_dir "${activity_preds_dir}/system-performance" \
        --results_file "${activity_preds_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth activity predictions"
    python predict_seq_lctm.py \
        --config_file "${config_dir}/activity/predict_seq_lctm.yaml" \
        --out_dir "${activity_smoothed_dir}" \
        --data_dir "${activity_data_dir}/data" \
        --scores_dir "${activity_preds_dir}/data"
    python analysis.py \
        --out_dir "${activity_smoothed_dir}/system-performance" \
        --results_file "${activity_smoothed_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


# --=(PHASE 2: PREDICT ACTIONS)==----------------------------------------------
if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make action dataset"
    python make_action_dataset.py \
        --out_dir "${action_data_dir}" \
        --video_data_dir "${video_data_dir}/data" \
        --features_dir "${activity_data_dir}/data" \
        --activity_labels_dir "${activity_smoothed_dir}/data" \
        --use_gt_activity_labels "True"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict actions"
    python predict_seq_pytorch.py \
        --config_file "${config_dir}/actions/predict_seq_pytorch.yaml" \
        --out_dir "${action_preds_dir}" \
        --data_dir "${action_data_dir}/data" \
        --gpu_dev_id "'0'"
    python analysis.py \
        --out_dir "${action_preds_dir}/system-performance" \
        --results_file "${action_preds_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth action predictions"
    python predict_seq_lctm.py \
        --config_file "${config_dir}/actions/predict_seq_lctm.yaml" \
        --out_dir "${action_smoothed_dir}" \
        --data_dir "${action_data_dir}/data" \
        --scores_dir "${action_preds_dir}/data"
    python analysis.py \
        --out_dir "${action_smoothed_dir}/system-performance" \
        --results_file "${action_smoothed_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


# --=(PHASE 3: SELECT KEYFRAMES)==---------------------------------------------
if [ "$start_at" -le "${STAGE}" ]; then
    python select_keyframes.py \
        --out_dir "${keyframes_dir}" \
        --video_data_dir "${video_data_dir}/data" \
        --keyframe_scores_dir "${keyframe_scores_dir}/data" \
        --activity_labels_dir "${activity_smoothed_dir}/data" \
        --action_labels_dir "${action_smoothed_dir}/data" \
        --use_gt_activity "True" \
        --use_gt_actions "False"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
