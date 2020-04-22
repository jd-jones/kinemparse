#!/bin/bash

scripts_dir="$HOME/repo/kinemparse/scripts/"
output_dir="~/repo/kinemparse/data/output/predict-activity"

kernel_sizes=$@

cd $scripts_dir

echo "STAGE1: make data"
#python make_imu_data.py  #comment out for testing

echo "STAGE2: predict_from_imu.py"
for kernel_size in $kernel_sizes; do
	for i in {1..5}; do
	    python predict_from_imu.py\
		--out_dir "$output_dir/predictions_k=$kernel_size" \
	 	--model_params "kernel_size: $kernel_size"\
    		--results_file "results_k=$kernel_size.csv"
	done
done

echo "STAGE3: analyze data"
#python analysis.py
