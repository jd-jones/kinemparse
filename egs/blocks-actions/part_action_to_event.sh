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
dataset_dir="${output_dir}/dataset"
cv_folds_dir="${output_dir}/fuse-modalities/cv-folds"
action_scores_dir="${output_dir}/fuse-modalities/scores_action-to-action"
part_scores_dir="${output_dir}/fuse-modalities/scores_part-to-part"
event_scores_dir="${output_dir}/events-from-video/scores"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/events-from-parts+actions"
scores_dir="${phase_dir}/scores-smoothed"

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
        --plot_io "True" \
        --prefix "seq="
    python ${debug_str} analysis.py \
        --out_dir "${scores_dir}/aggregate-results" \
        --results_file "${scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))
