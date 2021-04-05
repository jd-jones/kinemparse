#!/bin/bash
set -ue


# -=( SET DEFAULTS )==---------------------------------------------------------
data_dir="${HOME}/data/ikea_anu/data/ANU_ikea_dataset_video"
out_dir="${HOME}/data/ikea_anu/video_frames"

# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
	case $arg in
		--data_dir=*)
			data_dir="${arg#*=}"
			shift
			;;
		--out_dir=*)
			out_dir="${arg#*=}"
			shift
			;;
		*) # Unknown option: print help and exit
            # TODO: print help
            exit 0
			;;
	esac
done


for furn_dir in "${data_dir}"/*/; do
    furn_name="$(basename "${furn_dir}")"
    for seq_dir in "${furn_dir}"*/; do
        seq_name="$(basename "${seq_dir}")"
        frames_dir="${out_dir}/${furn_name}_${seq_name}"
        video_fn="${seq_dir}dev3/images/scan_video.avi"
        mkdir -p "${frames_dir}"
        ffmpeg -i "${video_fn}" -vf "scale=iw/4:ih/4" "${frames_dir}/%05d.jpg"
    done
done
