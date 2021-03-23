#!/usr/bin/env bash
#$ -wd /home/jdjones/data/output/grid_logs
#$ -V
#$ -N run_slowfast
#$ -j y -o $JOB_NAME-$JOB_ID.out
#$ -M jdjones@jhu.edu
#$ -m e
#$ -l ram_free=10G,mem_free=10G,gpu=2,hostname=b1[123456789]|c0*|c1[123456789]
#$ -q g.q

set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'
config_dir="/home/jdjones/repo/kinemparse/egs/ikea_anu/config"
data_dir='/home/jdjones/data/ikea_anu/video_frames'
base_dir="/home/jdjones/data/output/ikea_anu"


# -=( PARSE CLI ARGS )==-------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --label_type=*)
            label_type="${arg#*=}"
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
python tools/run_net.py \
    --cfg "${config_dir}/I3D_8x8_R50.yaml" \
    OUTPUT_DIR "${out_dir}" \
    MODEL.NUM_CLASSES "${num_classes}" \
    DATA.PATH_TO_DATA_DIR "${folds_dir}" \
    DATA.PATH_PREFIX "${data_dir}" \
    TRAIN.CHECKPOINT_FILE_PATH "${pretrained_checkpoint_file}" \
    TEST.CHECKPOINT_FILE_PATH "${trained_checkpoint_file}"
