#! /bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="${HOME}/data/output/blocks/child-videos"

# INPUT TO SCRIPT
raw_data_dir="${output_dir}/raw-data_imu"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/edge-labels-from-imu"
data_dir="${output_dir}/connections-dataset"
cv_folds_dir="${output_dir}/cv-folds_LOMO"
attr_scores_dir="${output_dir}/edge-label-preds_LOMO"
state_scores_dir="${output_dir}/predict-assemblies_attr"

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
		-e=*|--stop_at=*)
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

# SKIP THIS: SHOULD BE DEPRECATED BUT HAVEN'T TESTED REPLACEMENT
if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Download data"
    python ${debug_str} download_blocks_data.py \
        --out_dir "${raw_data_dir}" \
        --metadata_file "~/data/blocks/data/blocks_file_index.xlsx" \
        --metadata_criteria "{'GroupID': 'Child'}" \
        --corpus_name "child" \
        --default_annotator "Cathryn" \
        --use_annotated_keyframes "False" \
        --subsample_period '2' \
        --start_video_from_first_touch "True" \
        --save_video_before_first_touch "True" \
        --rgb_as_float "False" \
        --modalities "['imu']"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Make attributes"
    python ${debug_str} make_attr_data_imu.py \
        --data_dir "${raw_data_dir}/data" \
        --out_dir "${data_dir}" \
        --remove_before_first_touch "True" \
        --output_data "pairwise components" \
        --resting_from_gt "True" \
        --fig_type "array" \
        --include_signals "['gyro']"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Make cross-validation folds"
    python ${debug_str} make_cv_folds.py \
        --out_dir "${cv_folds_dir}" \
        --data_dir "${fused_data_dir}/data" \
        --feature_fn_format "feature-seq.npy" \
        --label_fn_format "label-seq.npy" \
        --cv_params "{'val_ratio': 'group', 'by_group': 'TaskID'}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Predict attributes"
    python ${debug_str} predict_seq_pytorch.py \
        --out_dir "${attr_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --independent_signals "True" \
        --active_only "True" \
        --gpu_dev_id "'2'" \
        --model_name "TCN" \
        --batch_size "1" \
        --learning_rate "0.001" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --train_params "{'num_epochs': 15, 'test_metric': 'F1', 'seq_as_batch': 'seq mode'}" \
        --model_params "{ \
            'binary_multiclass': False, \
            'tcn_channels': [8, 8, 16, 16, 32, 32], \
            'kernel_size': 25, \
            'dropout': 0.2 \
        }" \
        --plot_predictions "True" \
        # --label_mapping "{3: 2, 4: 1}" \
        # --eval_label_mapping "{2: 0}"
    python analysis.py \
        --out_dir "${attr_scores_dir}/system-performance" \
        --results_file "${attr_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))

# SKIP THIS: DEPRECATED
if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Predict assemblies"
    python ${debug_str} score_attributes.py \
        --out_dir "${state_scores_dir}" \
        --data_dir "${data_dir}/data" \
        --cv_data_dir "${video_scores_dir}/data" \
        --attributes_dir "${attr_scores_dir}/data" \
        --results_file "${state_scores_dir}/results.csv"
    python analysis.py \
        --out_dir "${state_scores_dir}/system-performance" \
        --results_file "${state_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))
