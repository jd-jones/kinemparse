#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
dest_dir="${HOME}/data/meccano"
download=true
make_symlinks=true
frames_dir="${dest_dir}/active_object_frames"
virtual_data_dir="${dest_dir}/video_frames"

# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
	case $arg in
		--dest_dir=*)
			dest_dir="${arg#*=}"
			shift
			;;
		--skip_download)
            download=false
			shift
			;;
		--skip_symlinks)
            make_symlinks=false
			shift
			;;
		--frames_dir=*)
			frames_dir="${arg#*=}"
			shift
			;;
		--virtual_data_dir=*)
			virtual_data_dir="${arg#*=}"
			shift
			;;
		*) # Unknown option: print help and exit
            # TODO: print help
            exit 0
			;;
	esac
done


# -=( MAIN SCRIPT )==----------------------------------------------------------
urls=( \
    'https://iplab.dmi.unict.it/MECCANO/downloads/MECCANO_verb_temporal_annotations.zip', \
    'https://iplab.dmi.unict.it/MECCANO/downloads/MECCANO_action_temporal_annotations.zip', \
    'https://iplab.dmi.unict.it/MECCANO/downloads/MECCANO_active_objects_annotations_frames.zip'
)

if [ "${download}" = "true" ]; then
    echo "STEP 1: Download dataset"
    for url in ${urls[@]}; do
        fn=$(basename "${url}")
        dest_file="${dest_dir}/${fn}"
        curl "${url}" -o "${dest_file}"
        tar -xv "${dest_file}"
    done
fi

if [ "${make_symlinks}" = "true" ]; then
    echo "STEP 2: Make virtual frame dataset"
    for file_path in "${frames_dir}"/*.jpg; do
        orig_fn=$(basename "${file_path}")
        frame_fn="${orig_fn#*_}"
        vid_id="${orig_fn%_*}"
        vid_id=${vid_id##+(0)}  # remove leading zeros
        dest_dir="${virtual_data_dir}/${vid_id}"
        mkdir -p "${dest_dir}"
        ln -s "${file_path}" "${dest_dir}/${frame_fn}"
    done
fi
