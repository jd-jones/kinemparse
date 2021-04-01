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
            echo "Error: Unrecognized argument ${arg}" >&2
            exit 1
            ;;
	esac
done


# -=( SET I/O PATHS )==--------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="${HOME}/data/output/blocks-actions"

# READONLY DIRS
input_dir="${HOME}/data/blocks"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
dataset_dir="${output_dir}/dataset"
phase_dir="${output_dir}/${label_type}s-from-video"
cv_folds_dir="${phase_dir}/cv-folds"

# Figure out how many classes there are by counting commas in the vocab file.
# (This won't work if the vocab contains non-alphanumeric objects or if
# something in the vocab contains a comma)
# vocab_file="${dataset_dir}/${label_type}-dataset/vocab.json"
# num_classes=$((`cat ${vocab_file} | tr -cd ',' | wc -c`+1))

case $label_type in
    'event' | 'action')
        eval_crit='topk_accuracy'
        eval_crit_params='["k", 1]'
        eval_crit_name='top1_acc'
        loss_func='cross_entropy'
        ;;
    'part')
        eval_crit='F1'
        eval_crit_params='["background_index", 0]'
        eval_crit_name='F1'
        loss_func='bce_logit'
        # Decrease num_classes by one to ignore background class
        num_classes=$((num_classes-1))
        ;;
    *)
        echo "Error: Unrecognized label_type ${label_type}" >&2
        exit 1
        ;;
esac


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Download meccano data"
    ./download_blocks_data.sh \
        --dest_dir=${input_dir}
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make action dataset"
    python ${debug_str} make_action_data.py \
        --out_dir "${dataset_dir}" \
        --metadata_file "~/data/blocks/data/blocks_file_index.xlsx" \
        --metadata_criteria "{'GroupID': 'Child'}" \
        --corpus_name "child" \
        --default_annotator "Cathryn" \
        --slowfast_csv_params "{'sep': ','}" \
        --win_params "{'win_size': 100, 'stride': 10}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
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
    echo "STAGE ${STAGE}: Train action recognition model"
    qsub run_slowfast_sge.sh \
        --config_dir="${config_dir}" \
        --data_dir="${frames_dir}" \
        --base_dir="${output_dir}" \
        --label_type="${label_type}" \
        --num_classes="${num_classes}" \
        --loss_func="${loss_func}" \
        --eval_crit="${eval_crit}" \
        --eval_crit_params="${eval_crit_params}" \
        --eval_crit_name="${eval_crit_name}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))