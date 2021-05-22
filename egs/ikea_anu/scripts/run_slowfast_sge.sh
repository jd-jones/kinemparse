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
cd '/home/jdjones/repo/kinemparse/egs/ikea_anu/scripts'


# -=( MAIN SCRIPT )==----------------------------------------------------------
./run_slowfast_all-cvfolds.sh \
    --config_dir="${config_dir}" \
    --data_dir="${data_dir}" \
    --base_dir="${base_dir}" \
    --label_type="${label_type}" \
    --num_classes="${num_classes}" \
    --loss_func="${loss_func}" \
    --head_act="${head_act}" \
    --eval_crit="${eval_crit}" \
    --eval_crit_params="${eval_crit_params}" \
    --eval_crit_name="${eval_crit_name}"
