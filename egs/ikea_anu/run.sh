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
        --label-type=*)
            label_type="${arg#*=}"
            ;;
        *) # Unknown option: print error and exit
            echo "Error: Unrecognized argument ${arg}" >&2
            exit 1
            ;;
    esac
done


# -=( SET I/O PATHS )==--------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
output_dir="~/data/output/ikea_anu"

# INPUT TO SCRIPT
input_dir="~/data/ikea_anu"
raw_data_dir="${input_dir}/data"
annotation_dir="${input_dir}/annotations"
frames_dir="${input_dir}/video_frames"

# OUTPUT OF SCRIPT
dataset_dir="${output_dir}/dataset"
phase_dir="${output_dir}/${label_type}s-from-video"
viz_dir="${phase_dir}/visualize"
cv_folds_dir="${phase_dir}/cv-folds"
preds_dir="${phase_dir}/action-preds"
batch_preds_dir="${preds_dir}/batches"
eval_dir="${preds_dir}/eval"


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Visualize dataset"
    python ${debug_str} viz_dataset.py \
        --out_dir "${viz_dir}" \
        --data_dir "${raw_data_dir}" \
        --annotation_dir "${annotation_dir}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make action dataset"
    python ${debug_str} make_action_data.py \
        --out_dir "${dataset_dir}" \
        --data_dir "${raw_data_dir}" \
        --annotation_dir "${annotation_dir}" \
        --frames_dir "${frames_dir}" \
        --col_format "ikea_tk" \
        --slowfast_csv_params "{'sep': ' '}"
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
        --cv_params "{'by_group': 'split_name', 'n_splits': 2, 'val_ratio': 0}" \
        --slowfast_csv_params "{'sep': ' '}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict edge labels"
    python ${debug_str} predict_video_pytorch.py \
        --out_dir "${batch_preds_dir}" \
        --data_dir "${dataset_dir}/action-dataset" \
        --prefix "seq=" \
        --file_fn_format "frame-fns.json" \
        --label_fn_format "labels.npy" \
        --gpu_dev_id "'0'" \
        --batch_size "20" \
        --learning_rate "0.001" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --train_params "{'num_epochs': 5, 'test_metric': 'Accuracy', 'seq_as_batch': 'sample mode'}" \
        --model_params "{'finetune_extractor': True, 'feature_extractor_name': 'resnet18'}" \
        --viz_params "{}"
    # python analysis.py \
    #     --out_dir "${edge_label_batches_dir}/system-performance" \
    #     --results_file "${edge_label_batches_dir}/results.csv"
    # python postprocess_assembly_scores.py \
    #     --out_dir "${edge_label_dir}" \
    #     --data_dir "${data_dir}/data" \
    #     --scores_dir "${edge_label_batches_dir}/data" \
    #     --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))
