#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at="7"
stop_after="7"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="${HOME}/data/output/parse_airplanes"
detections_dir="${output_dir}/track_hands"
viz_dir="${output_dir}/viz_detections"
action_dir="${output_dir}/action"
action_data_dir="${action_dir}/dataset"
action_preds_dir="${action_dir}/preds"
assembly_dir="${output_dir}/assembly"
assembly_data_dir="${assembly_dir}/dataset"
assembly_preds_dir="${assembly_dir}/preds"

# DATA DIRS
airplane_data_dir="${HOME}/data/toy_airplane"
airplane_videos_dir="${airplane_data_dir}/videos"
airplane_labels_dir="${airplane_data_dir}/labels"
airplane_detections_dir="${airplane_data_dir}/hand_detections"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir


STAGE=1


# -=( PHASE 1: HAND DETECTION AND TRACKING )==---------------------------------
if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Track hand detections"
    python track_hands.py \
        --out_dir "${detections_dir}" \
        --videos_dir "${airplane_videos_dir}" \
        --hand_detections_dir "${airplane_detections_dir}"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Visualize hand detections"
    python viz_hand_detections.py \
        --out_dir "${viz_dir}" \
        --videos_dir "${airplane_videos_dir}" \
        --hand_detections_dir "${detections_dir}/data"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Copy videos to onedrive for viewing"
    # Copy videos to onedrive so I can view them
    rclone="$HOME/BACKUP/anaconda3/envs/kinemparse/bin/rclone"
    $rclone copy "${viz_dir}" "onedrive_jhu:workspace" --stats-one-line -P --stats 2s
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))


# -=( PHASE 2: ACTION PREDICTION )==-------------------------------------------
if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make action dataset"
    python make_action_data.py \
        --out_dir "${action_data_dir}" \
        --hand_detections_dir "${detections_dir}/data" \
        --labels_dir "${airplane_labels_dir}"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict actions"
    python predict_seq_pytorch.py \
        --config_file "${config_dir}/actions/predict_seq_pytorch.yaml" \
        --out_dir "${action_preds_dir}" \
        --data_dir "${action_data_dir}/data" \
        --data_dir "${action_data_dir}/data" \
        --gpu_dev_id "'2'"
    python analysis.py \
        --out_dir "${action_preds_dir}/system-performance" \
        --results_file "${action_preds_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


# -=( PHASE 3: ASSEMBLY PREDICTION )==-----------------------------------------
if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make assembly dataset"
    python make_assembly_data.py \
        --out_dir "${assembly_data_dir}" \
        --bin_scores_dir "${action_preds_dir}/data" \
        --action_labels_dir "${airplane_labels_dir}"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth assembly predictions"
    python predict_seq_lctm.py \
        --config_file "${config_dir}/assembly/predict_seq_lctm.yaml" \
        --out_dir "${assembly_preds_dir}" \
        --data_dir "${assembly_data_dir}/data" \
        --scores_dir "${assembly_data_dir}/data"
    python analysis.py \
        --out_dir "${assembly_preds_dir}/system-performance" \
        --results_file "${assembly_preds_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
