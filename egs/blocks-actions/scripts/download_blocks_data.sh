#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
dest_dir="${HOME}/data/blocks"
mount=true
copy=true
source_videos_dir=''
dest_videos_dir=''

# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
	case $arg in
		--dest_dir=*)
			dest_dir="${arg#*=}"
			shift
			;;
		--skip_mount)
            mount=false
			shift
			;;
		--skip_copy)
            copy=false
			shift
			;;
        --source_videos_dir=*)
			source_videos_dir="${arg#*=}"
			shift
			;;
        --dest_videos_dir=*)
			dest_videos_dir="${arg#*=}"
			shift
			;;
		*) # Unknown option: print help and exit
            # TODO: print help
            exit 0
			;;
	esac
done


# -=( MAIN SCRIPT )==----------------------------------------------------------


if [ "${mount}" == "true" ]; then
    echo "STEP 1: Mount dataset"
    remote_drive='//cloud.nas.jh.edu/lcsr-cogsci-blocks$'
    mkdir -p "${dest_dir}"
    opts=( \
        'username=jjone229,' \
        'domain=WIN,' \
        'uid=jjone229,' \
        'gid=lcsr,' \
        'dir_mode=0700,' \
        'file_mode=0700,' \
        'vers=3.0' \
    )
    opt_str=$(IFS=; echo "${opts[*]}")
    sudo mount -t cifs "${remote_drive}" "${dest_dir}" -o "${opt_str}"
fi


if [ "${copy}" == "true" ]; then
    echo "STEP 2: Copy frames"
    for video_dir in "${source_videos_dir}"/*-rgb/; do
        video_id=$(basename "${video_dir%-*}")
        dest_video_dir="${dest_videos_dir}/${video_id}"

        if [ -d "${dest_video_dir}" ]; then
            echo "  Skipping video ${video_id}: dest directory exists"
            continue
        fi

        if [ -z "$(ls -A ${video_dir})" ]; then
            echo "  Skipping video ${video_id}: source directory empty"
            continue
        fi

        mkdir -p "${dest_video_dir}"
        echo "  Converting frames for video ${video_id}"
        for source_file in "${video_dir}"*.png; do
            orig_fn=$(basename "${source_file}")
            name="${orig_fn%.*}"
            dest_file="${dest_video_dir}/${name}.jpg"
            convert "${source_file}" "${dest_file}"
        done
    done
fi
