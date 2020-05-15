#! /bin/bash
set -ue

eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="$HOME/repo/kinemparse/data/output/predict-joined"

start_at="4"
stop_after="4"

data_dir="$output_dir/imu-data"
scores_dir="$output_dir/predictions_tcn"
smoothed_dir="$output_dir/predictions_sm-crf"
seg_dir="$output_dir/segments"

cd $scripts_dir

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
    echo "STAGE 2: Score connections"
    python predict_from_imu.py \
        --config_file "${config_dir}/predict_from_imu.yaml" \
        --out_dir "${scores_dir}" \
        --data_dir "${data_dir}/data" \
        --results_file "${scores_dir}/results.csv"
    python analysis.py \
        --out_dir "${scores_dir}/system-performance" \
        --results_file "${scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: Post-process connection scores"
    python predict_from_imu_lctm.py \
        --config_file "${config_dir}/predict_from_imu_lctm.yaml" \
        --out_dir "${smoothed_dir}" \
        --data_dir "${data_dir}/data" \
        --scores_dir "${scores_dir}/data" \
        --results_file "${smoothed_dir}/results.csv"
    python analysis.py \
        --out_dir "${smoothed_dir}/system-performance" \
        --results_file "${smoothed_dir}/results.csv"
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
        --predictions_dir "${smoothed_dir}/data" \
        --results_file "${seg_dir}/results.csv"
    python analysis.py \
        --out_dir "${seg_dir}/system-performance" \
        --results_file "${seg_dir}/results.csv"
fi
if [ "$stop_after" -eq "4" ]; then
    exit 1
fi
