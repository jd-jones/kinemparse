#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at=3
stop_after=3

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
base_dir="${HOME}/data/output/blocks/child-videos_keyframes-only"
output_dir="${base_dir}/fuse-modalities"
video_scores_dir="${output_dir}/assembly-scores_rgb_normalization=per-pixel"
imu_scores_dir="${output_dir}/assembly-scores_imu"
fused_scores_dir="${output_dir}/predict-assemblies_imu"

# IMU DATA DIRS --- READ-ONLY
imu_dir="${base_dir}/block-connections-imu"
imu_data_dir="${imu_dir}/connections-dataset"
imu_features_dir="${imu_dir}/predict-attributes"

# VIDEO DATA DIRS --- READ-ONLY
corpus_dir="${HOME}/data/blocks/data/child"
detections_dir="${base_dir}/detections"
decode_dir="${base_dir}/decode_rgb"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd "${scripts_dir}"

STAGE=0

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Convert decode_keyframes output"
    python restructure_output_decode_keyframes.py \
        --data_dir "${decode_dir}/data" \
        --out_dir "${video_scores_dir}" \
        --detections_dir "${detections_dir}/data" \
        --modality "RGB"
        # --normalization "per-pixel" \
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Predict assemblies"
    python _score_attributes.py \
        --config_file "${config_dir}/score_attributes.yaml" \
        --out_dir "${imu_scores_dir}" \
        --data_dir "${imu_data_dir}/data" \
        --cv_data_dir "${video_scores_dir}/data" \
        --attributes_dir "${imu_features_dir}/data" \
        --results_file "${imu_scores_dir}/results.csv" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${imu_scores_dir}/system-performance" \
        --results_file "${imu_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Fuse assembly predictions"
    python _combine_scores.py \
        --config_file "${config_dir}/combine_scores.yaml" \
        --out_dir "${fused_scores_dir}" \
        --data_dir "${imu_data_dir}/data" \
        --cv_data_dir "${video_scores_dir}/data" \
        --score_dirs "[${imu_scores_dir}/data, ${video_scores_dir}/data]" \
        --fusion_method "rgb_only" \
        --decode "True"
    python analysis.py \
        --out_dir "${fused_scores_dir}/system-performance" \
        --results_file "${fused_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Evaluate model predictions"
    eval_dir="${fused_scores_dir}/system-eval_ignore-initial"
    python eval_preds.py \
        --out_dir "${eval_dir}" \
        --data_dir "${fused_scores_dir}/data" \
        --plot_predictions "False" \
        --draw_paths "False" \
        --ignore_initial_state "True"
    python analysis.py \
        --out_dir "${eval_dir}/system-performance" \
        --results_file "${eval_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
