#!/bin/bash

scripts_dir="$HOME/repo/kinemparse/scripts/"
output_dir="~/repo/kinemparse/data/output/predict-joined"

start_at="3"
stop_after="10"

data_dir="$output_dir/imu-data"

param_name="kernel_size"
param_values=(11 25)

cd $scripts_dir

if [ "$start_at" -le "1" ]; then
    echo "STAGE 1: Make data"
    python make_imu_data.py \
        --out_dir $data_dir
fi
if [ "$stop_after" -eq "1" ]; then
    exit 1
fi


if [ "$start_at" -le "2" ]; then
    echo "STAGE 2: Score connections"
    for param_val in ${param_values[@]}; do
        python predict_from_imu.py \
            --out_dir "$output_dir/predictions_${param_name}=${param_val}_trial=$i" \
            --data_dir "$data_dir/data" \
            --model_params "$param_name: $param_val" \
            --results_file "$output_dir/results.csv" \
            --sweep_param_name "${param_name}"
    done
    python analysis.py \
        --out_dir "${output_dir}/predictions_torch-tcn_labels-01a/analysis" \
        --results_file "${output_dir}/predictions_torch-tcn_labels-01a/results.csv"
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: Segment connections"
    for param_val in ${param_values[@]}; do
        python predict_from_imu_lctm.py \
            --out_dir "$output_dir/predictions_${param_name}=${param_val}_trial=$i" \
            --data_dir "$data_dir/data" \
            --scores_dir ""
            --results_file "$output_dir/results.csv" \
            --sweep_param_name "${param_name}"
    done
    python analysis.py \
        --out_dir "${output_dir}/predictions_torch-tcn_labels-01a/analysis" \
        --results_file "${output_dir}/predictions_torch-tcn_labels-01a/results.csv"
fi
if [ "$stop_after" -eq "3" ]; then
    exit 1
fi
