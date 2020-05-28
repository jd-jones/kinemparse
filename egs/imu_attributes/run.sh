#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at="6"
stop_after="6"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="$HOME/repo/kinemparse/data/output/predict-joined"
data_dir="$output_dir/imu-data"
attr_scores_dir="$output_dir/predict-attributes_tcn"
attr_smoothed_dir="$output_dir/predict-attributes_sm-crf"
seg_dir="$output_dir/segments"
state_scores_dir="$output_dir/predict-assemblies_attr"
# state_fused_dir="$output_dir/predict-assemblies_rgb-only"
state_fused_dir="$output_dir/predict-assemblies_fused_standardized"
keyframe_decode_scores_dir="$output_dir/register-keyframes"

# EXTERNAL DATA DIRS
blocks_dir="$HOME/repo/blocks/data/output/blocks_child_keyframes-only_2020-01-26"
rgb_state_scores_dir="$blocks_dir/register_rgb"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir


if [ "$start_at" -le "0" ]; then
    echo "STAGE 0: Convert decode_keyframes output"
    python restructure_output_decode_keyframes.py \
        --data_dir "${rgb_state_scores_dir}/data/" \
        --out_dir $keyframe_decode_scores_dir
fi
if [ "$stop_after" -eq "0" ]; then
    exit 1
fi

if [ "$start_at" -le "1" ]; then
    echo "STAGE 1: Make data"
    python make_attr_data_imu.py \
        --config_file "${config_dir}/make_attr_data_imu.yaml" \
        --out_dir $data_dir
fi
if [ "$stop_after" -eq "1" ]; then
    exit 1
fi


if [ "$start_at" -le "2" ]; then
    echo "STAGE 2: Predict attributes"
    python predict_from_imu.py \
        --config_file "${config_dir}/predict_from_imu.yaml" \
        --out_dir "${attr_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --results_file "${scores_dir}/results.csv"
    python analysis.py \
        --out_dir "${attr_scores_dir}/system-performance" \
        --results_file "${attr_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: Smooth attribute predictions"
    python predict_from_imu_lctm.py \
        --config_file "${config_dir}/predict_from_imu_lctm.yaml" \
        --out_dir "${attr_smoothed_dir}" \
        --data_dir "${data_dir}/data" \
        --scores_dir "${attr_scores_dir}/data" \
        --results_file "${attr_smoothed_dir}/results.csv"
    python analysis.py \
        --out_dir "${attr_smoothed_dir}/system-performance" \
        --results_file "${attr_smoothed_dir}/results.csv"
fi
if [ "$stop_after" -eq "3" ]; then
    exit 1
fi

if [ "$start_at" -le "4" ]; then
    echo "STAGE 4: Segment signal"
    python segment_from_imu.py \
        --config_file "${config_dir}/segment_from_imu.yaml" \
        --out_dir "${seg_dir}" \
        --imu_data_dir "${data_dir}/data" \
        --video_data_dir "~/repo/kinemparse/data/output/blocks_child_2020-05-04/raw-data/data" \
        --predictions_dir "${attr_smoothed_dir}/data" \
        --results_file "${seg_dir}/results.csv"
    python analysis.py \
        --out_dir "${seg_dir}/system-performance" \
        --results_file "${seg_dir}/results.csv"
fi
if [ "$stop_after" -eq "4" ]; then
    exit 1
fi

if [ "$start_at" -le "5" ]; then
    echo "STAGE 5: Predict assemblies"
    python score_attributes.py \
        --config_file "${config_dir}/score_attributes.yaml" \
        --out_dir "${state_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --cv_data_dir "${keyframe_decode_scores_dir}/data" \
        --attributes_dir "${attr_scores_dir}/data" \
        --results_file "${state_scores_dir}/results.csv"
    python analysis.py \
        --out_dir "${state_scores_dir}/system-performance" \
        --results_file "${state_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "5" ]; then
    exit 1
fi

if [ "$start_at" -le "6" ]; then
    echo "STAGE 6: Fuse assembly predictions"
    python combine_scores.py \
        --config_file "${config_dir}/combine_scores.yaml" \
        --out_dir "${state_fused_dir}" \
        --data_dir "${data_dir}/data" \
        --cv_data_dir "${keyframe_decode_scores_dir}/data" \
        --results_file "${state_fused_dir}/results.csv" \
        --score_dirs "[${state_scores_dir}/data, ${keyframe_decode_scores_dir}/data]" \
        --plot_predictions "False"
        # --score_dirs "[${keyframe_decode_scores_dir}/data]"
    python analysis.py \
        --out_dir "${state_fused_dir}/system-performance" \
        --results_file "${state_fused_dir}/results.csv"
fi
if [ "$stop_after" -eq "6" ]; then
    exit 1
fi
