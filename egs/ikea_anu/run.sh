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
output_dir="${HOME}/data/output/ikea_anu"

# INPUT TO SCRIPT
input_dir="${HOME}/data/ikea_anu"
raw_data_dir="${input_dir}/data"
annotation_dir="${input_dir}/annotations"
frames_dir="${input_dir}/video_frames"

# OUTPUT OF SCRIPT
dataset_dir="${output_dir}/dataset"
phase_dir="${output_dir}/${label_type}s-from-video"
viz_dir="${phase_dir}/visualize"
cv_folds_dir="${phase_dir}/cv-folds"
scores_dir="${phase_dir}/scores"
connections_dir="${phase_dir}/event-to-connection"

slowfast_scores_dir="${phase_dir}/run-slowfast"
scores_eval_dir="${scores_dir}/eval"

# Figure out how many classes there are by counting commas in the vocab file.
# (This won't work if the vocab contains non-alphanumeric objects or if
# something in the vocab contains a comma)
vocab_file="${dataset_dir}/${label_type}-dataset/vocab.json"
num_classes=$((`cat ${vocab_file} | tr -cd ',' | wc -c`+1))


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
        --slowfast_csv_params "{'sep': ','}" \
        --win_params "{'win_size': 150, 'stride': 15}"
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
        --slowfast_csv_params "{'sep': ','}"
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
        --num_classes="${num_classes}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Post-process slowfast output"
    python ${debug_str} postprocess_slowfast_output.py \
        --out_dir "${scores_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --results_file "${slowfast_scores_dir}/results_test.pkl" \
        --cv_file "${cv_folds_dir}/data/cvfold=1_test_slowfast-labels_win.csv" \
        --col_format "ikea_tk" \
        --win_params "{'win_size': 150, 'stride': 15}" \
        --slowfast_csv_params "{'sep': ' '}"
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
    echo "STAGE ${STAGE}: Compute connection scores from action scores"
    export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/kinemparse/lib"
    event_attr_fn="${HOME}/data/event_metadata/ikea-anu.csv"
    connection_attr_fn="${HOME}/data/action_to_connection/ikea-anu.csv"
    part_info_fn="${HOME}/data/assembly_structures/ikea-anu.json"
    python ${debug_str} event_to_connection.py \
        --out_dir "${connections_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --scores_dir "${scores_dir}/data" \
        --event_attr_fn "${event_attr_fn}" \
        --connection_attr_fn "${connection_attr_fn}" \
        --assembly_attr_fn "${part_info_fn}" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "True" \
        --only_fold 1 \
        --prefix "seq=" \
        --background_action "NA" \
        --model_params "{ \
            'decode_type': 'joint', \
            'output_stage': 3, \
            'return_label': 'output', \
            'reduce_order': 'post' \
        }" \
        --stop_after 5
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


