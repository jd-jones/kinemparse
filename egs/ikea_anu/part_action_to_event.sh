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
output_dir="~/data/output/ikea_anu"

# INPUT TO SCRIPT
input_dir="~/data/ikea_anu"
raw_data_dir="${input_dir}/data"
annotation_dir="${input_dir}/annotations"
frames_dir="${input_dir}/video_frames"
dataset_dir="${output_dir}/dataset"
cv_folds_dir="${output_dir}/events-from-video/cv-folds"
action_scores_dir="${output_dir}/actions-from-video/scores_BACKUP"
part_scores_dir="${output_dir}/parts-from-video/scores_BACKUP"
event_scores_dir="${output_dir}/events-from-video/scores_BACKUP"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/events-from-parts+actions"
scores_dir="${phase_dir}/scores"

scores_eval_dir="${scores_dir}/eval"


# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Predict events from action & part scores"
    python ${debug_str} part_action_to_event.py \
        --out_dir "${scores_dir}" \
        --data_dir "${dataset_dir}" \
        --event_scores_dir "${event_scores_dir}/data" \
        --action_scores_dir "${action_scores_dir}/data" \
        --part_scores_dir "${part_scores_dir}/data" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --only_fold 1 \
        --plot_io "False" \
        --prefix "seq=" \
        --as_atomic_events "true"
    python ${debug_str} analysis.py \
        --out_dir "${scores_dir}/aggregate-results" \
        --results_file "${scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    python ${debug_str} eval_system_output.py \
        --out_dir "${scores_eval_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --scores_dir "${scores_dir}/data" \
        --frames_dir "${frames_dir}" \
        --vocab_from_scores_dir "True" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --only_fold 1 \
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
    python ${debug_str} compare_results.py \
        --out_dir "${scores_eval_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --scores_dirs "{ \
            'composed': ${scores_dir}/eval/data, \
            'atomic': ${event_scores_dir}/eval/data, \
        }" \
        --vocab_from_scores_dir "False" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --only_fold 1 \
        --plot_io "False" \
        --prefix "seq="
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))
