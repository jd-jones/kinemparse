#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at=1
stop_after=1

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="$HOME/repo/kinemparse/data/output/fuse-modalities"
keyframe_decode_scores_dir="$output_dir/register-keyframes_normalization=per-pixel"
fused_scores_dir="$output_dir/predict-assemblies_per-pixel_learned"
# decode_dir="$output_dir/decode"

# IMU DATA DIRS --- READ-ONLY
imu_dir="$HOME/repo/kinemparse/data/output/predict-joined"
imu_data_dir="$imu_dir/imu-data"
state_scores_dir="$imu_dir/predict-assemblies_attr"

# VIDEO DATA DIRS --- READ-ONLY
corpus_dir="$HOME/repo/blocks/data/input/child"
new_blocks_dir="$HOME/repo/kinemparse/data/output/blocks_child_keyframes-only_2020-05-04"
blocks_dir="$HOME/repo/kinemparse/data/output/blocks_child_keyframes-only_2020-01-26"
rgb_data_dir="$new_blocks_dir/raw-data"
preprocess_dir="$blocks_dir/preprocess"
detections_dir="$blocks_dir/detections"
keyframes_dir="$blocks_dir/keyframes"
register_dir="$blocks_dir/register_rgb"
decode_dir="$blocks_dir/decode_rgb"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=0

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Convert decode_keyframes output"
    python restructure_output_decode_keyframes.py \
        --data_dir "${decode_dir}/data" \
        --out_dir $keyframe_decode_scores_dir \
        --detections_dir "${detections_dir}/data" \
        --normalization "per-pixel" \
        --modality "RGB"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Fuse assembly predictions"
    python combine_scores.py \
        --config_file "${config_dir}/combine_scores.yaml" \
        --out_dir "${fused_scores_dir}" \
        --data_dir "${imu_data_dir}/data" \
        --cv_data_dir "${keyframe_decode_scores_dir}/data" \
        --results_file "${fused_scores_dir}/results.csv" \
        --score_dirs "[${state_scores_dir}/data, ${keyframe_decode_scores_dir}/data]" \
        --prune_imu "False" \
        --standardize "False" \
        --plot_predictions "False" \
        --fusion_method "sum" \
        --decode "True" \
        --metadata_file "${corpus_dir}/meta-data.csv" \
        --gpu_dev_id "'0'" \
        --train_params "{num_epochs: 15, test_metric: 'Accuracy', seq_as_batch: True}"
    python analysis.py \
        --out_dir "${fused_scores_dir}/system-performance" \
        --results_file "${fused_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))
