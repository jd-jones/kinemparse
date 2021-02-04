#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
output_dir="~/data/output/ikea_anu"

# READONLY DIRS
input_dir="~/data/ikea_anu"
data_dir="${input_dir}/data"
annotation_dir="${input_dir}/annotations"
frames_dir="${input_dir}/video_frames"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
viz_dir="${output_dir}/viz_dataset"
action_data_dir="${output_dir}/action_dataset"

start_at="0"
stop_after="100"

debug_str=""

# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
	case $arg in
		-s=*|--start_at=*)
			start_at="${arg#*=}"
			shift
			;;
		-e=*|--stop_after=*)
			stop_after="${arg#*=}"
			shift
			;;
		--stage=*)
			start_at="${arg#*=}"
			stop_after="${arg#*=}"
			shift
			;;
        --debug)
            debug_str="-m pdb"
            ;;
		*) # Unknown option: print help and exit
            # TODO: print help
            exit 0
			;;
	esac
done


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Visualize dataset"
    python ${debug_str} viz_dataset.py \
        --out_dir "${viz_dir}" \
        --data_dir "${data_dir}" \
        --annotation_dir "${annotation_dir}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make action dataset"
    python ${debug_str} make_action_data.py \
        --out_dir "${action_data_dir}" \
        --data_dir "${data_dir}" \
        --annotation_dir "${annotation_dir}" \
        --frames_dir "${frames_dir}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
