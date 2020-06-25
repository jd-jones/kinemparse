#!/bin/bash
set -ue

eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/repo/kinemparse/data/output/blocks_child_2020-05-04"

start_at="6"
stop_after="6"

data_dir="$output_dir/raw-data"
preprocess_dir="$output_dir/preprocess"
detections_dir="$output_dir/detections"
keyframes_dir="$output_dir/keyframes"
register_dir="$output_dir/register"
decode_dir="$output_dir/decode"

cd $scripts_dir

STAGE=0

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Download video data"
    python download_blocks_videos.py \
        --config_file "$config_dir/download_blocks_videos.yaml" \
        --out_dir "$data_dir"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Pre-process videos"
    python preprocess_blocks_videos.py \
        --config_file "$config_dir/preprocess_blocks_videos.yaml" \
        --out_dir "$preprocess_dir" \
        --data_dir "$data_dir/data" \
        --start_from "50"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "3" ]; then
    echo "STAGE ${STAGE}: Detect objects"
    python detect_objects.py \
        --config_file "$config_dir/detect_objects.yaml" \
        --out_dir "$detections_dir" \
        --data_dir "$data_dir/data" \
        --preprocess_dir "$preprocess_dir/data"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Select keyframes"
    python select_keyframes.py \
        --config_file "${config_dir}/select_keyframes.yaml" \
        --out_dir "${keyframes_dir}" \
        --data_dir "${data_dir}/data" \
        --preprocess_dir "${preprocess_dir}/data" \
        --segments_dir "~/repo/kinemparse/data/output/predict-joined/segments/data"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make segment data"
    python segment_videos.py \
        --config_file "${config_dir}/segment_videos.yaml" \
        --out_dir "${output_dir}/segment-data" \
        --data_dir "$data_dir/data" \
        --video_seg_scores_dir "${keyframes_dir}/data"
        # --imu_seg_scores_dir ""
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict segments"
    python predict_seq_pytorch.py \
        --config_file "${config_dir}/predict_seq_pytorch.yaml" \
        --out_dir "${output_dir}/predict-segments" \
        --data_dir "${output_dir}/segment-data/data"
    python analysis.py \
        --out_dir "${output_dir}/predict-segments/system-performance" \
        --results_file "${output_dir}/predict-segments/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth segments"
    python predict_seq_lctm.py \
        --config_file "${config_dir}/predict_seq_lctm.yaml" \
        --out_dir "${output_dir}/smooth-segments" \
        --data_dir "${output_dir}/segment-data/data" \
        --scores_dir "${output_dir}/predict-segments/data"
    python analysis.py \
        --out_dir "${output_dir}/smooth-segments/system-performance" \
        --results_file "${output_dir}/smooth-segments/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Register templates"
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
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Decode best paths"
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
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
