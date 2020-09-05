#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at="1"
stop_after="3"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="${HOME}/data/output/parse_airplanes"
detections_dir="${output_dir}/track_hands/"
viz_dir="${output_dir}/viz_detections/"

# DATA DIRS
airplane_data_dir="${HOME}/data/toy_airplane/"
airplane_videos_dir="${airplane_data_dir}/videos/"
airplane_detections_dir="${airplane_data_dir}/hand_detections/"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=1

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Compute part poses"
    python track_hands.py \
        --out_dir "${detections_dir}" \
        --videos_dir "${airplane_videos_dir}" \
        --hand_detections_dir "${airplane_detections_dir}"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "${start_at}" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Compute part poses"
    python viz_hand_detections.py \
        --out_dir "${viz_dir}" \
        --videos_dir "${airplane_videos_dir}" \
        --hand_detections_dir "${detections_dir}/data"
fi
if [ "${stop_after}" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    # Copy videos to onedrive so I can view them
    rclone="$HOME/BACKUP/anaconda3/envs/kinemparse/bin/rclone"
    $rclone copy "${viz_dir}" "onedrive_jhu:workspace" --stats-one-line -P --stats 2s
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))
