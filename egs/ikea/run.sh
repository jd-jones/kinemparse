#! /bin/bash
set -ue

# --=(SET CONFIG OPTIONS)==----------------------------------------------------
# SET WHICH PROCESSING STAGES ARE RUN
start_at="4"
stop_after="4"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="${HOME}/data/output/parse_ikea"
part_pose_dir="${output_dir}/hole_poses"
data_dir="${output_dir}/hole_dataset"
preds_dir="${output_dir}/preds"
smoothed_dir="${output_dir}/preds-smoothed"

# DATA DIRS --- READONLY
ikea_data_dir="${HOME}/data/output/ikea_part_tracking"
marker_pose_dir="${ikea_data_dir}/marker_poses"
marker_bundles_dir="${ikea_data_dir}/marker_bundles"
labels_dir="${HOME}/data/ikea/labels"
urdf_dir="${ikea_data_dir}/urdf"


# --=(SCRIPT SETUP)==----------------------------------------------------------
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
        --urdf_file "${urdf_dir}/ikea_chair_all_parts.urdf.xacro" \
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

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Score assemblies"
    python score_assemblies.py \
        --config_file "${config_dir}/score_assemblies.yaml" \
        --out_dir "${preds_dir}" \
        --data_dir "${data_dir}/data" \
        --gpu_dev_id "None"
    python analysis.py \
        --out_dir "${preds_dir}/system-performance" \
        --results_file "${preds_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth assembly predictions"
    python predict_seq_lctm.py \
        --config_file "${config_dir}/predict_seq_lctm.yaml" \
        --out_dir "${smoothed_dir}" \
        --data_dir "${data_dir}/data" \
        --scores_dir "${preds_dir}/data"
    python analysis.py \
        --out_dir "${smoothed_dir}/system-performance" \
        --results_file "${smoothed_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth assembly predictions"
    action_dir="${smoothed_dir}_action"
    python eval_output.py \
        --out_dir "${action_dir}" \
        --preds_dir "${smoothed_dir}/data" \
        --data_dir "${data_dir}/data"
    python analysis.py \
        --out_dir "${action_dir}/system-performance" \
        --results_file "${action_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
