#!/usr/bin/env bash
#$ -wd /home/jdjones/data/output/grid_logs
#$ -V
#$ -N train_slowfast
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jdjones@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=2,hostname=b1[123456789]|c0*|c1[123456789]
#$ -q g.q

set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'


# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --label_type=*)
            label_type="${arg#*=}"
            shift
            ;;
        *) # Unknown option: print error and exit
            echo "Error: Unrecognized argument ${arg}" >&2
            exit 1
            ;;
    esac
done

case $label_type in
    'event')
        num_classes=33
        ;;
    'action')
        num_classes=14
        ;;
    'part')
        num_classes=11
        ;;
    *) # Unknown option: print error and exit
        echo "Error: Unrecognized label_type ${arg}" >&2
        exit 1
        ;;
esac


# -=( SET I/O PATHS )==--------------------------------------------------------
data_dir='/home/jdjones/data/ikea_anu/video_frames'
base_dir="/home/jdjones/data/output/ikea_anu"
phase_dir="${base_dir}/${label_type}s-from-video"
folds_dir="${phase_dir}/cv-folds/data"
out_dir="${phase_dir}/train_slowfast"
config_dir="/home/jdjones/repo/kinemparse/egs/ikea_anu/config"


# -=( PREPARE ENVIRONMENT )==--------------------------------------------------
conda activate kinemparse
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 2)
cd "/home/jdjones/repo/CompositionalActions/slowfast"


# -=( MAIN SCRIPT )==----------------------------------------------------------
python tools/run_net.py \
  --cfg "${config_dir}/I3D_8x8_R50.yaml" \
  NUM_GPUS 2 \
  DATA_LOADER.NUM_WORKERS 4 \
  MODEL.NUM_CLASSES "${num_classes}" \
  DATA.TRAIN_CSV 'cvfold=0_slowfast_train.csv' \
  DATA.VAL_CSV 'cvfold=0_slowfast_test.csv' \
  DATA.PATH_TO_DATA_DIR "${folds_dir}" \
  DATA.PATH_PREFIX "${data_dir}" \
  TRAIN.CHECKPOINT_FILE_PATH "${base_dir}/I3D_8x8_R50.pkl" \
  TRAIN.ENABLE 'True' \
  TRAIN.CHECKPOINT_TYPE 'caffe2' \
  SOLVER.MAX_EPOCH 100 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.BATCH_SIZE 32 \
  TRAIN.BALANCED_SAMPLING "False" \
  TEST.BATCH_SIZE 32 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  TEST.SAVE_RESULTS_PATH "results_test.pkl" \
  TENSORBOARD.ENABLE "True" \
  LOG_MODEL_INFO "True" \
  SOLVER.BASE_LR 0.05 \
  BN.USE_PRECISE_STATS "False" \
  OUTPUT_DIR "${out_dir}"
