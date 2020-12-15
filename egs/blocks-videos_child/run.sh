#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/data/output/blocks/child-videos"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
data_dir="${output_dir}/raw-data"
background_dir="${output_dir}/background-detections"
detections_dir="${output_dir}/object-detections"
seg_labels_dir="${output_dir}/image-segment-labels"
sim_pretrain_dir="${output_dir}/pretrained-models-sim"
edge_label_dir="${output_dir}/edge-label-preds"
edge_label_smoothed_dir="${output_dir}/edge-label-preds-smoothed"
assembly_scores_dir="${output_dir}/assembly-scores"
decode_dir="${output_dir}/assembly-decode"

edge_label_batches_dir="${edge_label_dir}/batches"
edge_label_eval_dir="${edge_label_dir}/eval"
smoothed_eval_dir="${edge_label_smoothed_dir}/eval"
decode_eval_dir="${decode_dir}/eval"

start_at="0"
stop_after="100"


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
		--phase=*)
			start_at="${arg#*=}"
			stop_after="${arg#*=}"
			shift
			;;
		*) # Unknown option: print help and exit
            # TODO: print help
            exit 0
			;;
	esac
done


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
PHASE=0


if [ "${start_at}" -le "${PHASE}" ]; then
    echo "PHASE ${PHASE}: Download data"
    python download_blocks_data.py \
        --out_dir "${data_dir}" \
        --metadata_file "~/data/blocks/data/blocks_file_index.xlsx" \
        --metadata_criteria "{'GroupID': 'Child'}" \
        --corpus_name "child" \
        --default_annotator "Cathryn" \
        --use_annotated_keyframes "False" \
        --subsample_period '2' \
        --start_video_from_first_touch "True" \
        --save_video_before_first_touch "True" \
        --rgb_as_float "False" \
        --modalities "['video', 'imu']"
fi
if [ "$stop_after" -eq "${PHASE}" ]; then
    exit 1
fi
((++PHASE))


if [ "${start_at}" -le "${PHASE}" ]; then
    echo "PHASE ${PHASE}: Detect edge labels from video"
    source ../edge_labels_from_video.sh
fi
if [ "$stop_after" -eq "${PHASE}" ]; then
    exit 1
fi
((++PHASE))


if [ "${start_at}" -le "${PHASE}" ]; then
    echo "PHASE ${PHASE}: Detect edge labels from IMU"
    source ../edge_labels_from_imu.sh
fi
if [ "$stop_after" -eq "${PHASE}" ]; then
    exit 1
fi
((++PHASE))


if [ "${start_at}" -le "${PHASE}" ]; then
    echo "PHASE ${PHASE}: Predict assemblies from edge labels"
    source ../assemblies_from_edge_labels.sh
fi
if [ "$stop_after" -eq "${PHASE}" ]; then
    exit 1
fi
((++PHASE))
