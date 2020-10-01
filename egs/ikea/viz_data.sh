#!/bin/bash
set -ue

# SET CONFIG OPTIONS HERE
ikea_data_dir="${HOME}/data/ikea"
fpv_video_dir="${ikea_data_dir}/assembly_videos_fpv"
pose_dir="${HOME}/data/output/parse_ikea/hole_poses/data"
viz_dir="${HOME}/data/output/parse_ikea/viz_hole_poses"
video_dir="${viz_dir}/combined_videos"
onedrive_dir="onedrive_jhu:workspace/hole_poses"

start_at=3
stop_after=3

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir


STAGE=1


if [ "$start_at" -le $STAGE ]; then
    # Visualize all marker poses by drawing them onto the video frames
    python viz_dataset.py \
        --out_dir "${viz_dir}" \
        --data_dir "${pose_dir}" \
        --frames_dir "${fpv_video_dir}" \
        --metadata_parent_dir "${HOME}/data/output/ikea_part_tracking/marker_poses"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le $STAGE ]; then
    # Make all the output frames into video
    mkdir -p "${video_dir}"
    for video_path in "${viz_dir}/images"/*; do
        video_id=$(basename ${video_path})
        if [[ "${video_id}" == "combined_videos" ]]; then
            continue
        fi
        ffmpeg \
            -framerate 10 \
            -pattern_type glob -i "${video_path}/*.png" \
            -c:v libx264 -r 30 -pix_fmt yuv420p \
            "${video_dir}/${video_id}.mp4"
    done
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le $STAGE ]; then
    # Copy videos to onedrive so I can view them
    rclone="$HOME/BACKUP/anaconda3/envs/kinemparse/bin/rclone"
    $rclone mkdir "${onedrive_dir}"
    $rclone copy "${video_dir}" "${onedrive_dir}" --stats-one-line -P --stats 2s
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))
