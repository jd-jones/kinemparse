#!/bin/bash
set -ue

eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/repo/kinemparse/data/output/blocks_child_keyframes-only_2020-05-04"

start_at="1"
stop_after="1"

data_dir="$output_dir/raw-data"
preprocess_dir="$output_dir/preprocess"
features_dir="$output_dir/features_alt"

cd $scripts_dir

if [ "$start_at" -le "1" ]; then
    echo "STAGE 1: Downloading videos"
    python download_blocks_videos.py \
        --config_file "$config_dir/download_blocks_videos.yaml" \
        --out_dir "$data_dir"
fi
if [ "$stop_after" -eq "1" ]; then
    exit 1
fi

if [ "$start_at" -le "2" ]; then
    echo "STAGE 2: Preprocessing videos"
    python preprocess_blocks_videos.py \
        --config_file "$config_dir/preprocess_blocks_videos.yaml" \
        --out_dir "$preprocess_dir" \
        --data_dir "$data_dir/data" # \
        # --start_from 68
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: Making attribute data"
    python make_attr_data_image.py \
        --config_file "$config_dir/make_attr_data_image.yaml" \
        --out_dir "$features_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data"
fi
if [ "$stop_after" -eq "3" ]; then
    exit 1
fi
