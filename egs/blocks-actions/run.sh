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
frames_dir="${HOME}/data/blocks-videos-as-jpg/child"

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
dataset_dir="${output_dir}/dataset"
cv_folds_dir="${output_dir}/cv-folds"
phase_dir="${output_dir}/${label_type}s-from-video"
slowfast_cv_folds_dir="${phase_dir}/cv-folds"
slowfast_scores_dir="${phase_dir}/slowfast-scores"
scores_dir="${phase_dir}/probs"
smoothed_scores_dir="${phase_dir}/probs-smoothed"

scores_eval_dir="${scores_dir}/eval"
smoothed_eval_dir="${smoothed_scores_dir}/eval"

# Figure out how many classes there are by counting commas in the vocab file.
# (This won't work if the vocab contains non-alphanumeric objects or if
# something in the vocab contains a comma)
vocab_file="${dataset_dir}/${label_type}-dataset/vocab.json"
num_classes=$((`cat ${vocab_file} | tr -cd ',' | wc -c`+1))

case $label_type in
    'event' | 'action')
        eval_crit='topk_accuracy'
        eval_crit_params='["k", 1]'
        eval_crit_name='top1_acc'
        loss_func='cross_entropy'
        head_act='softmax'
        ;;
    'part')
        eval_crit='F1'
        eval_crit_params='["background_index", 0]'
        eval_crit_name='F1'
        loss_func='bce_logit'
        # Decrease num_classes by one to ignore background class
        num_classes=$((num_classes-1))
        head_act='sigmoid'
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
    echo "STAGE ${STAGE}: Download blocks data"
    ./download_blocks_data.sh \
        --dest_dir=${input_dir} \
        --source_videos_dir="${input_dir}/data/child/video-frames" \
        --dest_videos_dir="${HOME}/data/blocks-videos-as-jpg/child" \
        --skip_mount
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
        --data_dir "${dataset_dir}/event-dataset" \
        --prefix "seq=" \
        --feature_fn_format "frame-fns.json" \
        --label_fn_format "labels.npy" \
        --cv_params "{'val_ratio': 0.25, 'n_splits': 5, 'shuffle': True}" \
        --slowfast_csv_params "{'sep': ',',}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Split slowfast label files"
    python ${debug_str} make_cv_folds.py \
        --out_dir "${slowfast_cv_folds_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --prefix "seq=" \
        --feature_fn_format "frame-fns.json" \
        --label_fn_format "labels.npy" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
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
        --head_act="${head_act}" \
        --eval_crit="${eval_crit}" \
        --eval_crit_params="${eval_crit_params}" \
        --eval_crit_name="${eval_crit_name}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Post-process slowfast output"
    for cvfold_dir in ${slowfast_scores_dir}/*; do
        cvfold_str="`basename ${cvfold_dir}`"
        python ${debug_str} postprocess_slowfast_output.py \
            --out_dir "${scores_dir}" \
            --data_dir "${dataset_dir}/${label_type}-dataset" \
            --results_file "${cvfold_dir}/results_test.pkl" \
            --cv_file "${slowfast_cv_folds_dir}/data/${cvfold_str}_test_slowfast-labels_win.csv" \
            --slowfast_csv_params "{'sep': ','}" \
            --win_params "{'win_size': 100, 'stride': 10}" \
            --take_log "False"
    done
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
    echo "STAGE ${STAGE}: Smooth predictions"
    python ${debug_str} predict_seq_pytorch.py \
        --out_dir "${smoothed_scores_dir}" \
        --data_dir "${scores_dir}/data" \
        --prefix="seq=" \
        --feature_fn_format "score-seq.npy" \
        --label_fn_format "true-label-seq.npy" \
        --gpu_dev_id "'2'" \
        --predict_mode "'binary multiclass'" \
        --model_name "'LSTM'" \
        --batch_size "1" \
        --learning_rate "0.0002" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --dataset_params "{'transpose_data': False, 'flatten_feats': False}" \
        --train_params "{'num_epochs': 100, 'test_metric': 'F1', 'seq_as_batch': 'seq mode'}" \
        --model_params "{ \
            'hidden_dim': 256, \
            'num_layers': 1, \
            'bias': True, \
            'batch_first': True, \
            'dropout': 0, \
            'bidirectional': True, \
            'binary_multiclass': True \
        }" \
        --plot_predictions "True"
        # --predict_mode "'classify'"
    python analysis.py \
        --out_dir "${smoothed_scores_dir}/system-performance" \
        --results_file "${smoothed_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate system output"
    python ${debug_str} eval_system_output.py \
        --out_dir "${smoothed_eval_dir}" \
        --data_dir "${dataset_dir}/${label_type}-dataset" \
        --scores_dir "${smoothed_scores_dir}/data" \
        --frames_dir "${frames_dir}" \
        --cv_params "{'precomputed_fn': ${cv_folds_dir}/data/cv-folds.json}" \
        --plot_io "False" \
        --prefix "seq="
    python ${debug_str} analysis.py \
        --out_dir "${smoothed_eval_dir}/aggregate-results" \
        --results_file "${smoothed_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 0
fi
((++STAGE))
