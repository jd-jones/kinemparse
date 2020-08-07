#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at="0"
stop_after="0"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="${HOME}/data/output/parse_ikea"
part_pose_dir="${output_dir}/part_poses/"

# IMU DIRS --- READONLY
ikea_data_dir="${HOME}/data/output/ikea_part_tracking"
marker_pose_dir="${ikea_data_dir}/marker_poses"
marker_bundles_dir="${ikea_data_dir}/marker_bundles"
urdf_dir="${ikea_data_dir}/urdf"


# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=0

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Compute part poses"
    python -m pdb markers_to_parts.py \
        --out_dir "${part_pose_dir}" \
        --marker_pose_dir "${marker_pose_dir}" \
        --marker_bundles_dir "${marker_bundles_dir}" \
        --urdf_file "${urdf_dir}/ikea_chair_backrest.xacro"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
