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
config_dir="/home/jdjones/repo/kinemparse/egs/blocks-actions/config"
data_dir='/home/jdjones/data/blocks-videos-as-jpg/child'
base_dir="/home/jdjones/data/output/blocks-actions"
loss_func='cross_entropy'
head_act='softmax'
eval_crit='topk_accuracy'
eval_crit_params='["k", 1]'
eval_crit_name='top1_acc'
num_classes=''


# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --out_dir_name=*)
            out_dir_name="${arg#*=}"
            shift
            ;;
        --config_dir=*)
            config_dir="${arg#*=}"
            shift
            ;;
        --data_dir=*)
            data_dir="${arg#*=}"
            shift
            ;;
        --base_dir=*)
            base_dir="${arg#*=}"
            shift
            ;;
        --label_type=*)
            label_type="${arg#*=}"
            shift
            ;;
        --num_classes=*)
            num_classes="${arg#*=}"
            shift
            ;;
        --loss_func=*)
            loss_func="${arg#*=}"
            shift
            ;;
        --head_act=*)
            head_act="${arg#*=}"
            shift
            ;;
        --eval_crit=*)
            eval_crit="${arg#*=}"
            shift
            ;;
        --eval_crit_params=*)
            eval_crit_params="${arg#*=}"
            shift
            ;;
        --eval_crit_name=*)
            eval_crit_name="${arg#*=}"
            shift
            ;;
        --train_fold_fn=*)
            train_fold_fn="${arg#*=}"
            shift
            ;;
        --val_fold_fn=*)
            val_fold_fn="${arg#*=}"
            shift
            ;;
        --test_fold_fn=*)
            test_fold_fn="${arg#*=}"
            shift
            ;;
        --copy_to=*)
            copy_to="${arg#*=}"
            shift
            ;;
        *) # Unknown option: print error and exit
            echo "Error: Unrecognized argument ${arg}" >&2
            exit 1
            ;;
    esac
done


# -=( PREPARE ENVIRONMENT )==--------------------------------------------------
conda activate kinemparse
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 2)
# source /home/gqin2/scripts/acquire-gpu
cd '/home/jdjones/repo/kinemparse/egs/blocks-actions/scripts'

echo `hostname`


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
