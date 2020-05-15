#!/bin/bash
set -ue

eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/repo/kinemparse/data/output/blocks_easy_2020-04-10"

start_at="4"
stop_after="4"

data_dir="$output_dir/raw-data"
preprocess_dir="$output_dir/preprocess"
detections_dir="$output_dir/detections"
keyframes_dir="$output_dir/keyframes"
register_dir="$output_dir/register"
decode_dir="$output_dir/decode"

cd $scripts_dir

if [ "$start_at" -le "1" ]; then
    echo "STAGE 1: Downloading video data"
    python download_blocks_videos.py \
        --config_file "$config_dir/download_blocks_videos.yaml" \
        --out_dir "$data_dir"
fi
if [ "$stop_after" -eq "1" ]; then
    exit 1
fi

if [ "$start_at" -le "2" ]; then
    echo "STAGE 2: Pre-processing videos"
    python preprocess_blocks_videos.py \
        --config_file "$config_dir/preprocess_blocks_videos.yaml" \
        --out_dir "$preprocess_dir" \
        --data_dir "$data_dir/data"
fi
if [ "$stop_after" -eq "2" ]; then
    exit 1
fi

if [ "$start_at" -le "3" ]; then
    echo "STAGE 3: Detecting objects"
    python detect_objects.py \
        --config_file "$config_dir/detect_objects.yaml" \
        --out_dir "$detections_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data"
fi
if [ "$stop_after" -eq "3" ]; then
    exit 1
fi

if [ "$start_at" -le "4" ]; then
    echo "STAGE 4: Selecting keyframes"
    python select_keyframes.py \
        --config_file "$config_dir/select_keyframes.yaml" \
        --out_dir "$keyframes_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data"
fi
if [ "$stop_after" -eq "4" ]; then
    exit 1
fi

if [ "$start_at" -le "5" ]; then
    echo "STAGE 5: Registering templates"
    python decode_keyframes.py \
        --config_file "$config_dir/decode_keyframes_register.yaml" \
        --out_dir "$register_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data" \
        --detections_dir "$detections_dir/data" \
        --keyframes_dir "$keyframes_dir/data"
    python score_results.py \
        --config_file "$config_dir/score_results.yaml" \
        --out_dir "$register_dir/system-performance" \
        --decode_dir "$register_dir/data"
fi
if [ "$stop_after" -eq "5" ]; then
    exit 1
fi

if [ "$start_at" -le "6" ]; then
    echo "STAGE 6: Decoding best paths"
    python decode_keyframes.py \
        --config_file "$config_dir/decode_keyframes.yaml" \
        --out_dir "$decode_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data" \
        --detections_dir "$detections_dir/data" \
        --keyframes_dir "$keyframes_dir/data" \
        --data_scores_dir "$register_dir/data"
    python score_results.py \
        --config_file "$config_dir/score_results.yaml" \
        --out_dir "$decode_dir/system-performance" \
        --decode_dir "$decode_dir/data"
fi
if [ "$stop_after" -eq "6" ]; then
    exit 1
fi
