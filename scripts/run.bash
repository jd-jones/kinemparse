#!/bin/bash

scripts_dir="$HOME/repo/kinemparse/scripts/"
# output_dir="~/repo/kinemparse/data/output/test-run"
output_dir="~/repo/kinemparse/data/output/predict-joined"

start_at="3"
stop_after="10"

data_dir="$output_dir/imu-data"

num_trials=1
param_name="kernel_size"
param_values=(11 25)

cd $scripts_dir

if [ "$start_at" -le "1" ]; then
    echo "STAGE 1: make data"
    python make_imu_data.py \
        --out_dir $data_dir
fi
if [ "$stop_after" -eq "1" ]; then
    exit 1
fi


if [ "$start_at" -le "2" ]; then
    echo "STAGE 2: predict_from_imu.py"
    for param_val in ${param_values[@]}; do
        for i in $(seq 1 $num_trials); do
            python predict_from_imu.py \
                --out_dir "$output_dir/predictions_${param_name}=${param_val}_trial=$i" \
                --data_dir "$data_dir/data" \
                --model_params "$param_name: $param_val" \
                --results_file "$output_dir/results.csv" \
                --sweep_param_name "${param_name}"
        done
    done
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: analyze data"
    python analysis.py \
        --out_dir "${output_dir}/predictions_torch-tcn_labels-01/analysis" \
        --results_file "${output_dir}/predictions_torch-tcn_labels-01/results.csv"
        # --out_dir "$output_dir/analysis" \
        # --results_file "$output_dir/results.csv" \
        # --sweep_param_name "${param_name}"
fi
if [ "$stop_after" -eq "3" ]; then
    exit 1
fi
