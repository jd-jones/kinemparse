#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at="1"
stop_after="1"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="${HOME}/data/output/parse_ikea"
part_pose_dir="${output_dir}/part_poses"
data_dir="${output_dir}/dataset"

# IMU DIRS --- READONLY
ikea_data_dir="${HOME}/data/output/ikea_part_tracking"
marker_pose_dir="${ikea_data_dir}/marker_poses"
marker_bundles_dir="${ikea_data_dir}/marker_bundles"
labels_dir="${HOME}/data/ikea/labels"
urdf_dir="${ikea_data_dir}/urdf"


# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=0

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Compute part poses"
    python markers_to_parts.py \
        --out_dir "${part_pose_dir}" \
        --marker_pose_dir "${marker_pose_dir}" \
        --marker_bundles_dir "${marker_bundles_dir}" \
        --urdf_file "${urdf_dir}/ikea_chair_backrest.xacro" \
        --labels_dir "${labels_dir}" \
        --rename_parts "{'back_beam': 'backbeam', 'front_beam': 'frontbeam', 'seat': 'cushion'}"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make assembly dataset"
    python make_data.py \
        --out_dir "${data_dir}" \
        --data_dir "${part_pose_dir}/data"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
