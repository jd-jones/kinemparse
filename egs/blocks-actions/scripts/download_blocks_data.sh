#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
dest_dir="${HOME}/data/blocks"
mount=true

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
