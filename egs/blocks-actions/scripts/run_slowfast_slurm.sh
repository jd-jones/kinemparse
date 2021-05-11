#!/usr/bin/env bash
#SBATCH --chdir=/home/jdjones/data/output/grid_logs
#SBATCH --job-name="run_slowfast"
#SBATCH --output=$SBATCH_JOB_NAME-$SLURM_JOB_ID.out
#SBATCH --mail-user=jdjones@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=15000
#SBATCH --gres=gpu:2

set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'
config_dir="/home/jdjones/repo/kinemparse/egs/ikea_anu/config"
data_dir='/home/jdjones/data/ikea_anu/video_frames'
base_dir="/home/jdjones/data/output/ikea_anu"
num_classes=33
loss_func='cross_entropy'
eval_crit='topk_accuracy'
eval_crit_params='["k", 1]'
eval_crit_name='top1_acc'

# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
    case $arg in
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
        *) # Unknown option: print error and exit
            echo "Error: Unrecognized argument ${arg}" >&2
            exit 1
            ;;
    esac
done

# -=( SET I/O PATHS )==--------------------------------------------------------
phase_dir="${base_dir}/${label_type}s-from-video"
folds_dir="${phase_dir}/cv-folds/data"
out_dir="${phase_dir}/run-slowfast"

pretrained_checkpoint_file="${base_dir}/I3D_8x8_R50.pkl"
trained_checkpoint_file=''


# -=( PREPARE ENVIRONMENT )==--------------------------------------------------
conda activate kinemparse
export CUDA_VISIBLE_DEVICES=$(free-gpu -n 2)
cd "/home/jdjones/repo/CompositionalActions/slowfast"


# -=( MAIN SCRIPT )==----------------------------------------------------------
srun python tools/run_net.py \
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
