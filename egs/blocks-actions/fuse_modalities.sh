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
output_dir="~/data/output/blocks-actions"

# INPUT TO SCRIPT
input_dir="${HOME}/data/blocks"
dataset_dir="${output_dir}/dataset"
action_scores_dir="${output_dir}/actions-from-video/scores"
part_scores_dir="${output_dir}/parts-from-video/scores"
event_scores_dir="${output_dir}/events-from-video/scores"
assembly_scores_dir="$HOME/data/output/blocks-assemblies/assemblies-from-edge-labels/assembly-scores"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/fuse-modalities"
fusion_dataset_dir="${phase_dir}/dataset_event-to-event"
cv_folds_dir="${phase_dir}/cv-folds"
scores_dir="${phase_dir}/scores_event-to-event"

scores_eval_dir="${scores_dir}/eval"


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make fusion dataset"
    python ${debug_str} make_fusion_data.py \
        --out_dir "${fusion_dataset_dir}" \
        --data_dir "${dataset_dir}" \
        --actions_dir "${action_scores_dir}/data" \
        --parts_dir "${part_scores_dir}/data" \
        --events_dir "${event_scores_dir}/data" \
        --edges_dir "${assembly_scores_dir}/data" \
        --dataset_params "{ \
            'modalities': ['events'], \
            'labels': 'edge diffs' \
        }" \
        --plot_io "True" \
        --prefix "seq="
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make cross-validation folds"
    python ${debug_str} make_cv_folds.py \
        --out_dir "${cv_folds_dir}" \
        --data_dir "${fusion_dataset_dir}/data" \
        --prefix "seq=" \
        --feature_fn_format "feature-seq.npy" \
        --label_fn_format "label-seq.npy" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}" \
        --slowfast_csv_params "{'sep': ',',}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


# [WITHOUT] loss: 0.19150  acc: 92.84%  prc: 75.87%  rec: 78.25%  F_1: 77.04%
# [WITH]    loss: 0.20334  acc: 92.24%  prc: 72.75%  rec: 78.50%  F_1: 75.51%
# d=512  |  acc: 58.16%  prc: 37.76%  rec: 21.62%  F_1: 27.50%
if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Smooth predictions"
    python ${debug_str} predict_seq_pytorch.py \
        --out_dir "${scores_dir}" \
        --data_dir "${fusion_dataset_dir}/data" \
        --prefix="seq=" \
        --feature_fn_format "feature-seq.npy" \
        --label_fn_format "label-seq.npy" \
        --gpu_dev_id "'2'" \
        --predict_mode "'classify'" \
        --model_name "'LSTM'" \
        --batch_size "1" \
        --learning_rate "0.0002" \
        --output_dim_from_vocab "True" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --dataset_params "{'transpose_data': False, 'flatten_feats': False}" \
        --train_params "{'num_epochs': 100, 'test_metric': 'Accuracy', 'seq_as_batch': 'seq mode'}" \
        --model_params "{ \
            'hidden_dim': 512, \
            'num_layers': 1, \
            'bias': True, \
            'batch_first': True, \
            'dropout': 0, \
            'bidirectional': True, \
            'binary_multiclass': False \
        }" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${scores_dir}/system-performance" \
        --results_file "${scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    python ${debug_str} eval_system_output.py \
        --out_dir "${scores_eval_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --scores_dir "${scores_dir}/data" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "False" \
        --prefix "seq="
    python ${debug_str} analysis.py \
        --out_dir "${scores_eval_dir}/aggregate-results" \
        --results_file "${scores_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    event_labels_dir="${phase_dir}/dataset_event-to-event"
    event_scores_dir="${phase_dir}/scores_edge-diff-to-event"
    event_scores_eval_dir="${event_scores_dir}/eval"
    python ${debug_str} edges_to_event.py \
        --out_dir "${event_scores_dir}" \
        --event_scores_dir "${event_labels_dir}/data" \
        --edge_scores_dir "${scores_dir}/data" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "False" \
        --prefix "seq="
    python ${debug_str} analysis.py \
        --out_dir "${event_scores_dir}/aggregate-results" \
        --results_file "${event_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    event_labels_dir="${phase_dir}/dataset_event-to-event"
    event_scores_dir="${phase_dir}/scores_edge-diff-to-event"
    event_scores_eval_dir="${event_scores_dir}/eval"
    python ${debug_str} eval_system_output.py \
        --out_dir "${event_scores_eval_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --scores_dir "${event_scores_dir}/data" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "False" \
        --prefix "seq="
    python ${debug_str} analysis.py \
        --out_dir "${event_scores_eval_dir}/aggregate-results" \
        --results_file "${event_scores_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))
