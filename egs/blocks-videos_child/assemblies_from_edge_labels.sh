#!/bin/bash
set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
output_dir="~/data/output/blocks/child-videos"

# INPUT TO SCRIPT
rgb_phase_dir="${output_dir}/edge-labels-from-video"
seg_labels_dir="${rgb_phase_dir}/image-segment-labels"
rgb_data_dir="${rgb_phase_dir}/raw-data"
rgb_vocab_dir="${rgb_phase_dir}/pretrained-models-sim"
rgb_edge_label_dir="${rgb_phase_dir}/edge-label-preds"

imu_phase_dir="${output_dir}/edge-labels-from-imu"
imu_data_dir="${imu_phase_dir}/connections-dataset"
imu_edge_label_dir="${imu_phase_dir}/edge-label-preds"

# OUTPUT OF SCRIPT
phase_dir="${output_dir}/assemblies-from-edge-labels"
fused_data_dir="${phase_dir}/fusion-dataset_TEST"
cv_folds_dir="${phase_dir}/cv-folds_TEST"
fused_scores_dir="${phase_dir}/edge-label-preds_fused_TEST"
assembly_scores_dir="${phase_dir}/assembly-scores_oov_TEST"
decode_dir="${phase_dir}/assembly-decode_oov_TEST"

decode_eval_dir="${decode_dir}/eval"

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
    echo "STAGE ${STAGE}: Make fusion dataset"
    python ${debug_str} make_fusion_dataset.py \
        --out_dir "${fused_data_dir}" \
        --rgb_data_dir "${rgb_data_dir}/data" \
        --rgb_attributes_dir "${rgb_edge_label_dir}/data" \
        --imu_data_dir "${imu_data_dir}/data" \
        --imu_attributes_dir "${imu_edge_label_dir}/data" \
        --modalities "['imu', 'rgb']" \
        --gpu_dev_id "'2'" \
        --plot_io "True"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
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
        --cv_params "{'val_ratio': 0, 'by_group': 'TaskID'}"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Fuse modalities"
    python ${debug_str} predict_seq_pytorch.py \
        --out_dir "${fused_scores_dir}" \
        --data_dir "${fused_data_dir}/data" \
        --feature_fn_format "feature-seq.npy" \
        --label_fn_format "label-seq.npy" \
        --gpu_dev_id "'2'" \
        --predict_mode "'multiclass'" \
        --model_name "'TCN'" \
        --batch_size "1" \
        --learning_rate "0.0002" \
        --dataset_params "{'transpose_data': True, 'flatten_feats': True}" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --train_params "{'num_epochs': 1, 'test_metric': 'F1', 'seq_as_batch': 'seq mode'}" \
        --model_params "{ \
            'tcn_channels': [8,  8, 16, 16, 32, 32], \
            'kernel_size': 5, \
            'dropout': 0.2 \
        }" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${fused_scores_dir}/system-performance" \
        --results_file "${fused_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Score assemblies"
    python ${debug_str} score_assemblies.py \
        --out_dir "${assembly_scores_dir}" \
        --rgb_data_dir "${rgb_data_dir}/data" \
        --rgb_attributes_dir "${rgb_edge_label_dir}/data" \
        --rgb_vocab_dir "${rgb_vocab_dir}/data" \
        --imu_data_dir "${imu_data_dir}/data" \
        --imu_attributes_dir "${imu_edge_label_dir}/data" \
        --modalities "['imu', 'rgb']" \
        --gpu_dev_id "'2'" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${assembly_scores_dir}/system-performance" \
        --results_file "${assembly_scores_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Decode assembly predictions"
    python ${debug_str} predict_seq_lctm.py \
        --out_dir "${decode_dir}" \
        --data_dir "${assembly_scores_dir}/data" \
        --feature_fn_format "score-seq.npy" \
        --label_fn_format "label-seq.npy" \
        --cv_params "{'precomputed_fn': '${cv_folds_dir}/data/cv-folds.json'}" \
        --model_name "PretrainedModel" \
        --pre_init_pw "True" \
        --model_params "{ \
            'inference': 'segmental', 'segmental': True, \
            'start_prior': True, 'end_prior': True \
        }" \
        --viz_params "{'labels_together': True}" \
        --plot_predictions "True"
    python analysis.py \
        --out_dir "${decode_dir}/system-performance" \
        --results_file "${decode_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))


if [ "$start_at" -le "${STAGE}" ]; then
    echo "STAGE ${STAGE}: Evaluate decoder predictions"
    python ${debug_str} eval_system_output.py \
        --out_dir "${decode_eval_dir}" \
        --data_dir "${rgb_data_dir}/data" \
        --segs_dir "${seg_labels_dir}/data" \
        --scores_dir "${decode_dir}/data" \
        --vocab_dir "${rgb_vocab_dir}/data" \
        --gpu_dev_id "'2'" \
        --label_type "assembly" \
        --num_disp_imgs "10"
    python analysis.py \
        --out_dir "${decode_eval_dir}/system-performance" \
        --results_file "${decode_eval_dir}/results.csv"
fi
if [ "$stop_after" -eq "${STAGE}" ]; then
    exit 1
fi
((++STAGE))
