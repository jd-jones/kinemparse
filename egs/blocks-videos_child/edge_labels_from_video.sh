#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/data/output/blocks/child-videos"

# INPUT TO SCRIPT
data_dir="${output_dir}/raw-data"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/edge-labels-from-video"
background_dir="${phase_dir}/background-detections"
detections_dir="${phase_dir}/object-detections"
seg_labels_dir="${phase_dir}/image-segment-labels"
sim_pretrain_dir="${phase_dir}/pretrained-models-sim"
cv_folds_dir="${phase_dir}/cv-folds_LOMO"
edge_label_dir="${phase_dir}/edge-label-preds_LOMO"
edge_label_smoothed_dir="${phase_dir}/edge-label-preds-smoothed"

edge_label_batches_dir="${edge_label_dir}/batches"
edge_label_eval_dir="${edge_label_dir}/eval"
smoothed_eval_dir="${edge_label_smoothed_dir}/eval"

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
    echo "STAGE ${STAGE}: Detect background"
    python ${debug_str} detect_background.py \
        --out_dir "${background_dir}" \
        --data_dir "${data_dir}/data" \
        --background_data_dir "${data_dir}/data" \
        --depth_bg_detection_kwargs "{'plane_distance_thresh': 10, 'max_trials': 50}" \
        --rgb_bg_detection_kwargs "{'px_distance_thresh': 0.2}" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Detect objects"
    python ${debug_str} detect_objects.py \
        --out_dir "${detections_dir}" \
        --data_dir "${data_dir}/data" \
        --gpu_dev_id "'0'" \
        --batch_size "2" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Label foreground segments"
    python ${debug_str} make_seg_labels.py \
        --out_dir "${seg_labels_dir}" \
        --data_dir "${data_dir}/data" \
        --bg_masks_dir "${background_dir}/data" \
        --person_masks_dir "${detections_dir}/data" \
        --sat_thresh "0.15" \
        --num_disp_imgs "10"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Pre-train assembly detector"
    python ${debug_str} train_assembly_detector.py \
        --out_dir "${sim_pretrain_dir}" \
        --data_dir "${data_dir}/data" \
        --gpu_dev_id "'0'" \
        --batch_size "10" \
        --learning_rate "0.0002" \
        --model_name "Labeled Connections" \
        --load_masks_params "{'masks_dir': '${detections_dir}/data', 'num_per_video': 10}" \
        --train_params "{'num_epochs': 500, 'test_metric': 'F1', 'seq_as_batch': False}" \
        --model_params "{}" \
        --num_disp_imgs "10" \
        --viz_params "{}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make cross-validation folds"
    python ${debug_str} make_cv_folds.py \
        --out_dir "${cv_folds_dir}" \
        --data_dir "${data_dir}/data" \
        --feature_fn_format "rgb-frame-fn-seq.pkl" \
        --label_fn_format "action-seq.pkl" \
        --cv_params "{'val_ratio': 0.25, 'by_group': 'TaskID'}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict edge labels"
    python ${debug_str} predict_edge_labels.py \
        --out_dir "${edge_label_batches_dir}" \
        --data_dir "${data_dir}/data" \
        --segs_dir "${seg_labels_dir}/data" \
        --pretrained_model_dir "${sim_pretrain_dir}/data" \
        --gpu_dev_id "'0'" \
        --model_name "pretrained" \
        --batch_size "20" \
        --learning_rate "0.0002" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --train_params "{'num_epochs': 50, 'test_metric': 'F1', 'seq_as_batch': 'sample mode'}" \
        --viz_params "{}"
    python analysis.py \
        --out_dir "${edge_label_batches_dir}/system-performance" \
        --results_file "${edge_label_batches_dir}/results.csv"
    python postprocess_assembly_scores.py \
        --out_dir "${edge_label_dir}" \
        --data_dir "${data_dir}/data" \
        --scores_dir "${edge_label_batches_dir}/data"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate edge label predictions"
    python ${debug_str} eval_system_output.py \
        --out_dir "${edge_label_eval_dir}" \
        --data_dir "${data_dir}/data" \
        --segs_dir "${seg_labels_dir}/data" \
        --scores_dir "${edge_label_dir}/data" \
        --vocab_dir "${sim_pretrain_dir}/data" \
        --gpu_dev_id "'2'" \
        --num_disp_imgs "10"
    python analysis.py \
        --out_dir "${edge_label_eval_dir}/system-performance" \
        --results_file "${edge_label_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth predictions"
    python ${debug_str} predict_seq_pytorch.py \
        --out_dir "${edge_label_smoothed_dir}" \
        --data_dir "${edge_label_dir}/data" \
        --feature_fn_format "score-seq.pkl" \
        --label_fn_format "true-label-seq.pkl" \
        --gpu_dev_id "'2'" \
        --predict_mode "'multiclass'" \
        --model_name "'TCN'" \
        --batch_size "1" \
        --learning_rate "0.0002" \
        --dataset_params "{'transpose_data': True, 'flatten_feats': True}" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}" \
        --train_params "{'num_epochs': 500, 'test_metric': 'F1', 'seq_as_batch': 'seq mode'}" \
        --model_params "{ \
            'tcn_channels': [8,  8, 16, 16, 32, 32], \
            'kernel_size': 5, \
            'dropout': 0.2 \
        }" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${edge_label_smoothed_dir}/system-performance" \
        --results_file "${edge_label_smoothed_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate smoothed predictions"
    python ${debug_str} eval_system_output.py \
        --out_dir "${smoothed_eval_dir}" \
        --data_dir "${data_dir}/data" \
        --segs_dir "${seg_labels_dir}/data" \
        --scores_dir "${edge_label_smoothed_dir}/data" \
        --vocab_dir "${sim_pretrain_dir}/data" \
        --gpu_dev_id "'2'" \
        --num_disp_imgs "10"
    python analysis.py \
        --out_dir "${seq_eval_dir}/system-performance" \
        --results_file "${seq_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
