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
keyframe_decode_scores_dir="$output_dir/register-keyframes"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=0

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE 1: Make data"
    python make_attr_data_imu.py \
        --config_file "${config_dir}/make_attr_data_imu.yaml" \
        --out_dir $data_dir
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Predict attributes"
    python predict_from_imu.py \
        --config_file "${config_dir}/predict_from_imu.yaml" \
        --out_dir "${attr_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --results_file "${scores_dir}/results.csv"
    python analysis.py \
        --out_dir "${attr_scores_dir}/system-performance" \
        --results_file "${attr_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Smooth attribute predictions"
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
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Segment signal"
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
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Predict assemblies"
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
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))
