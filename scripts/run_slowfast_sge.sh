#!/usr/bin/env bash
#$ -wd /home/jdjones/data/output/grid_logs
#$ -V
#$ -N run_slowfast
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jdjones@jhu.edu
#$ -m e
#$ -l ram_free=15G,mem_free=15G,gpu=2,hostname=b1[123456789]|c0*|c1[123456789]
#$ -q g.q

set -ue


# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'
config_dir="/home/jdjones/repo/kinemparse/egs/ikea_anu/config"
data_dir='/home/jdjones/data/ikea_anu/video_frames'
base_dir="/home/jdjones/data/output/ikea_anu"
loss_func='cross_entropy'
head_act='softmax'
eval_crit='topk_accuracy'
eval_crit_params='["k", 1]'
eval_crit_name='top1_acc'
num_classes=''

# -=( PREPARE ENVIRONMENT )==--------------------------------------------------
conda activate kinemparse
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 2)
cd "/home/jdjones/repo/CompositionalActions/slowfast"


# -=( MAIN SCRIPT )==----------------------------------------------------------
python tools/run_net.py \
    --cfg "${config_dir}/I3D_8x8_R50.yaml" \
    OUTPUT_DIR "${out_dir}" \
    MODEL.NUM_CLASSES "${num_classes}" \
    MODEL.LOSS_FUNC "${loss_func}" \
    DATA.PATH_TO_DATA_DIR "${folds_dir}" \
    DATA.PATH_PREFIX "${data_dir}" \
    TRAIN.CHECKPOINT_FILE_PATH "${pretrained_checkpoint_file}" \
    TRAIN.EVAL_CRIT "${eval_crit}" \
    TRAIN.EVAL_CRIT_PARAMS "${eval_crit_params}" \
    TRAIN.EVAL_CRIT_NAME "${eval_crit_name}" \
    TEST.CHECKPOINT_FILE_PATH "${trained_checkpoint_file}"
