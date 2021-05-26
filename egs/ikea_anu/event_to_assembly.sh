#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
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
output_dir="${HOME}/data/output/ikea_anu"

# INPUT TO SCRIPT
dataset_dir="${output_dir}/dataset"
phase_dir="${output_dir}/events-from-video"
cv_folds_dir="${phase_dir}/cv-folds"
event_scores_dir="${phase_dir}/scores-smoothed"
event_attr_fn="${HOME}/data/event_metadata/ikea-anu.json"
connection_attr_fn="${HOME}/data/action_to_connection/ikea-anu.csv"
assembly_attr_fn="${HOME}/data/assembly_structures/ikea-anu.json"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/assemblies-from-events"
assembly_scores_dir="${phase_dir}/event-scores_decode_full-model"
assembly_scores_eval_dir="${assembly_scores_dir}/eval"



# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Compute assembly scores from event scores"
    export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/kinemparse/lib"
    python ${debug_str} event_to_assembly.py \
        --out_dir "${assembly_scores_dir}" \
        --data_dir "${dataset_dir}/event-dataset" \
        --scores_dir "${event_scores_dir}/data" \
        --event_attr_fn "${event_attr_fn}" \
        --connection_attr_fn "${connection_attr_fn}" \
        --assembly_attr_fn "${assembly_attr_fn}" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "True" \
        --prefix "seq=" \
        --background_action "NA" \
        --model_params "{ \
            'decode_type': 'joint', \
            'output_stage': 3, \
            'return_label': 'input', \
            'reduce_order': 'post', \
            'allow_self_transitions': False \
        }"
    python ${debug_str} analysis.py \
        --out_dir "${assembly_scores_dir}/aggregate-results" \
        --results_file "${assembly_scores_dir}/results.csv"
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


