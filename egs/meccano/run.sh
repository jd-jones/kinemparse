#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'  # 'action', 'event', or 'part'
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
        --label_type=*)
            label_type="${arg#*=}"
            ;;
		*) # Unknown option: print help and exit
            # TODO: print help
            exit 0
			;;
	esac
done


# -=( SET I/O PATHS )==--------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
output_dir="~/data/output/meccano"

# READONLY DIRS
input_dir="~/data/meccano"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
dataset_dir="${output_dir}/dataset"
phase_dir="${output_dir}/${label_type}s-from-video"
cv_folds_dir="${phase_dir}/cv-folds"


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Download meccano data"
    ./download_meccano_data.sh \
        --dest_dir=${input_dir} \
        --skip_download
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make action dataset"
    python ${debug_str} make_action_data.py \
        --out_dir "${dataset_dir}" \
        --annotation_dir "${input_dir}" \
        --frames_dir "${input_dir}/video_frames" \
        --slowfast_csv_params "{'sep': ','}" \
        --win_params "{'win_size': 100, 'stride': 10}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make cross-validation folds"
    python ${debug_str} make_cv_folds.py \
        --out_dir "${cv_folds_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --prefix "seq=" \
        --feature_fn_format "frame-fns.json" \
        --label_fn_format "labels.npy" \
        --cv_params "{ \
            'by_group': 'split_name', \
            'group_folds': [['train', 'val', 'test']] \
        }" \
        --slowfast_csv_params "{'sep': ',',}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Train action recognition models"
    qsub run_slowfast_sge.sh --label_type=event
    qsub run_slowfast_sge.sh --label_type=action
    qsub run_slowfast_sge.sh --label_type=part
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))
