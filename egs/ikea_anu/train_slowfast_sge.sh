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

conda activate kinemparse

cd "/home/jdjones/repo/CompositionalActions/slowfast"

data_dir='/home/jdjones/data/ikea_anu/video_frames'
base_dir='/home/jdjones/data/output/ikea_anu/actions-from-video'
folds_dir="${base_dir}/cv-folds/data"
out_dir="${base_dir}/train_slowfast"

export CUDA_VISIBLE_DEVICES=$(free-gpu -n 2)

python tools/run_net.py \
  --cfg "../costar/I3D_8x8_R50.yaml" \
  NUM_GPUS 2 \
  DATA_LOADER.NUM_WORKERS 4 \
  MODEL.NUM_CLASSES 33 \
  DATA.TRAIN_CSV 'original_train.csv' \
  DATA.VAL_CSV 'original_val.csv' \
  DATA.PATH_TO_DATA_DIR "${folds_dir}" \
  DATA.PATH_PREFIX "${data_dir}" \
  TRAIN.CHECKPOINT_FILE_PATH "${base_dir}/I3D_8x8_R50.pkl" \
  TRAIN.ENABLE 'True' \
  TRAIN.CHECKPOINT_TYPE 'caffe2' \
  SOLVER.MAX_EPOCH 100 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.BATCH_SIZE 32 \
  TEST.BATCH_SIZE 32 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  TEST.SAVE_RESULTS_PATH "${out_dir}" \
  TENSORBOARD.ENABLE "True" \
  LOG_MODEL_INFO "True" \
  SOLVER.BASE_LR 0.05 \
  BN.USE_PRECISE_STATS "False" \
  OUTPUT_DIR "${out_dir}/logs/"

cd -
