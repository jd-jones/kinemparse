#!/bin/bash

scripts_dir="$HOME/repo/kinemparse/scripts/"
recipe_config_dir=$(pwd)
output_dir="~/repo/kinemparse/data/output/blocks_child_2020-05-04"
start_at="3"
stop_after="3"

data_dir="$output_dir/raw-data"
preprocess_dir="$output_dir/preprocess"
features_dir="$output_dir/features_alt"

cd $scripts_dir

if [ "$start_at" -le "1" ]; then
    echo "STAGE 1: Downloading videos"
    python download_blocks_videos.py \
        --config_file "$recipe_config_dir/download_blocks_videos.yaml" \
        --out_dir "$data_dir"
fi
if [ "$stop_after" -eq "1" ]; then
    exit 1
fi

if [ "$start_at" -le "2" ]; then
    echo "STAGE 2: Preprocessing videos"
    python preprocess_blocks_videos.py \
        --config_file "$recipe_config_dir/preprocess_blocks_videos.yaml" \
        --out_dir "$preprocess_dir" \
        --data_dir "$data_dir/data" \
        --start_from 68
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: Extracting segment features"
    python make_image_data.py \
        --config_file "$recipe_config_dir/make_image_data.yaml" \
        --out_dir "$features_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data"
fi
if [ "$stop_after" -eq "3" ]; then
    exit 1
fi
