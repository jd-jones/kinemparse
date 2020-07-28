#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at="3"
stop_after="3"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="$HOME/repo/kinemparse/data/output/segment-seqs"
seg_dir="${output_dir}/imu-segments"

# IMU DIRS --- READONLY
imu_attr_dir="$HOME/repo/kinemparse/data/output/block-connections-imu"
imu_data_dir="${imu_attr_dir}/connections-dataset_untrimmed"
attr_scores_dir="${imu_attr_dir}/predict-attributes_tcn_untrimmed"

# VIDEO DIRS --- READONLY
video_preprocess_dir="$HOME/repo/kinemparse/data/output/blocks_child_2020-05-04"
video_data_dir="${video_preprocess_dir}/raw-data"
keyframes_dir="${video_preprocess_dir}/keyframes"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=0

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Segment signal"
    python make_activity_dataset_imu.py \
        --config_file "${config_dir}/make_activity_dataset.yaml" \
        --out_dir "${seg_dir}" \
        --imu_data_dir "${imu_data_dir}/data" \
        --video_data_dir "~/repo/kinemparse/data/output/blocks_child_2020-05-04/raw-data/data" \
        --predictions_dir "${attr_scores_dir}/data" \
        --results_file "${seg_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make segment data"
    python make_activity_dataset_video.py \
        --out_dir "${output_dir}/segment-data" \
        --video_data_dir "${video_data_dir}/data" \
        --imu_data_dir "${imu_data_dir}/data" \
        --video_seg_scores_dir "${keyframes_dir}/data" \
        --imu_seg_scores_dir "${attr_scores_dir}/data" \
        --gt_keyframes_dir "${video_data_dir}/data"
        # --imu_seg_scores_dir ""
        # --config_file "${config_dir}/make_activity_dataset_video.yaml"
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

