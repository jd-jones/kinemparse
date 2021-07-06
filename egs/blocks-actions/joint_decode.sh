#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
start_at="0"
stop_after="100"
debug_str=""

label_type='event'


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
        --label_type=*)
            label_type="${arg#*=}"
            shift
            ;;
        --debug)
            debug_str="-m pdb"
            shift
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
output_dir="${HOME}/data/output/blocks-actions"

# INPUT TO SCRIPT
# event_data_dir="${output_dir}/events-from-parts+actions/scores-smoothed"
event_data_dir="$HOME/data/output/blocks-actions/events-from-video/scores"
assembly_data_dir="$HOME/data/output/blocks-assemblies/assemblies-from-edge-labels/assembly-scores"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/event-assembly-decode"
decode_dataset_dir="${phase_dir}/dataset"
cv_folds_dir="${phase_dir}/cv-folds"
scores_dir="${phase_dir}/joint-scores_parts"
decode_dir="${phase_dir}/joint-decode_calibrated-scores"

decode_eval_dir="${decode_dir}/eval"

# -=( MAIN SCRIPT )==----------------------------------------------------------
cd ${scripts_dir}
STAGE=0



if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Create decode dataset"
    python ${debug_str} make_decode_dataset.py \
        --out_dir "${decode_dataset_dir}" \
        --data_dirs "{ \
            'assembly': ${assembly_data_dir}/data, \
            'event': ${event_data_dir}/data \
        }" \
        --prefix "{'assembly': 'trial=', 'event': 'seq='}" \
        --feature_fn_format "{'assembly': 'score-seq', 'event': 'score-seq'}" \
        --label_fn_format "{'assembly': 'label-seq', 'event': 'true-label-seq'}" \
        --stride "{'assembly': 5, 'event': null}" \
        --take_log "{'assembly': False, 'event': True}" \
        --draw_vocab "True"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make cross-validation folds"
    python ${debug_str} make_cv_folds.py \
        --out_dir "${cv_folds_dir}" \
        --data_dir "${decode_dataset_dir}/assembly-data" \
        --prefix "seq=" \
        --feature_fn_format "score-seq.npy" \
        --label_fn_format "true-label-seq.npy" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    # names=('event' 'assembly')
    names=('event')
    for name in ${names[@]}; do
        source_dir="${decode_dataset_dir}/${name}-data"
        eval_dir="${source_dir}/eval"
        python ${debug_str} eval_system_output.py \
            --out_dir "${eval_dir}" \
            --data_dir "${source_dir}" \
            --scores_dir "${source_dir}" \
            --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
            --plot_io "True" \
            --draw_labels "False" \
            --vocab_fig_dir "${decode_dataset_dir}/figures/${name}/vocab" \
            --prefix "seq=" \
            --draw_labels "True" \
            --vocab_fig_dir "${decode_dataset_dir}/figures/${name}/vocab"
        python ${debug_str} analysis.py \
            --out_dir "${eval_dir}/aggregate-results" \
            --results_file "${eval_dir}/results.csv"
    done
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Fuse assembly and event scores"
    python ${debug_str} predict_joint.py \
        --out_dir "${scores_dir}" \
        --assembly_scores_dir "${decode_dataset_dir}/assembly-data" \
        --event_scores_dir "${decode_dataset_dir}/event-data" \
        --labels_from "${label_type}s" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "True" \
        --prefix "seq=" \
        --feature_fn_format "score-seq" \
        --label_fn_format "true-label-seq" \
        --batch_size "1" \
        --learning_rate "0.005" \
        --train_params "{'num_epochs': 250, 'test_metric': 'Accuracy', 'seq_as_batch': 'seq mode'}" \
        --model_params "{}" \
        --out_type "parts" \
        --gpu_dev_id "'cpu'" \
        --plot_predictions "True"
    python ${debug_str} analysis.py \
        --out_dir "${scores_dir}/aggregate-results" \
        --results_file "${scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Compute assembly scores from event scores"
    export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/kinemparse/lib"
    python ${debug_str} joint_decode.py \
        --out_dir "${decode_dir}" \
        --assembly_scores_dir "${decode_dataset_dir}/assembly-data" \
        --event_scores_dir "${decode_dataset_dir}/event-data" \
        --joint_scores_dir "${scores_dir}/data" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "True" \
        --prefix "seq=" \
        --feature_fn_format "score-seq" \
        --label_fn_format "true-label-seq" \
        --labels_from "${label_type}s" \
        --standardize_inputs "False" \
        --model_params "{ \
            'decode_type': 'joint', \
            'output_stage': 3, \
            'return_label': 'input', \
            'reduce_order': 'pre', \
            'allow_self_transitions': False \
        }"
    python ${debug_str} analysis.py \
        --out_dir "${decode_dir}/aggregate-results" \
        --results_file "${decode_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    python ${debug_str} eval_system_output.py \
        --out_dir "${decode_eval_dir}" \
        --data_dir "${decode_dataset_dir}/${label_type}-data" \
        --scores_dir "${decode_dir}/data" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "True" \
        --prefix "seq=" \
        --draw_labels "True" \
        --vocab_fig_dir "${decode_dataset_dir}/figures/${label_type}/vocab"
    python ${debug_str} analysis.py \
        --out_dir "${decode_eval_dir}/aggregate-results" \
        --results_file "${decode_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


