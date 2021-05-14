#!/usr/bin/zsh
#SBATCH --job-name="run_slowfast"
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=jdjones@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=15000
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00

set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'
out_dir_name="run-slowfast"
config_dir="/home/jdjones/repo/kinemparse/egs/ikea_anu/config"
data_dir='/home/jdjones/data/ikea_anu/video_frames'
base_dir="/home/jdjones/data/output/ikea_anu"
num_classes=''
loss_func='cross_entropy'
eval_crit='topk_accuracy'
eval_crit_params='["k", 1]'
eval_crit_name='top1_acc'
train_fold_fn='cvfold=0_train_slowfast-labels_seg.csv'
val_fold_fn='cvfold=0_val_slowfast-labels_win.csv'
test_fold_fn='cvfold=0_test_slowfast-labels_win.csv'
copy_to=''

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

# -=( SET I/O PATHS )==--------------------------------------------------------
phase_dir="${base_dir}/${label_type}s-from-video"
dataset_dir="${base_dir}/dataset/${label_type}-dataset"
folds_dir="${phase_dir}/cv-folds/data"
out_dir="${phase_dir}/${out_dir_name}"

pretrained_checkpoint_file="${base_dir}/I3D_8x8_R50.pkl"
trained_checkpoint_file=''


# -=( PREPARE ENVIRONMENT )==--------------------------------------------------
if [[ ${num_classes} == '' ]]; then
    vocab_file="${dataset_dir}/vocab.json"
    num_classes=$((`cat ${vocab_file} | tr -cd ',' | wc -c`+1))
fi

cd "/home/map6/jon/CompositionalActions/slowfast"


# -=( MAIN SCRIPT )==----------------------------------------------------------
python tools/run_net.py \
    --cfg "${config_dir}/I3D_8x8_R50.yaml" \
    OUTPUT_DIR "${out_dir}" \
    MODEL.NUM_CLASSES "${num_classes}" \
    MODEL.LOSS_FUNC "${loss_func}" \
    DATA.PATH_TO_DATA_DIR "${folds_dir}" \
    DATA.PATH_PREFIX "${data_dir}" \
    DATA.TRAIN_CSV "${train_fold_fn}" \
    DATA.VAL_CSV "${val_fold_fn}" \
    DATA.TEST_CSV "${test_fold_fn}" \
    TRAIN.CHECKPOINT_FILE_PATH "${pretrained_checkpoint_file}" \
    TRAIN.EVAL_CRIT "${eval_crit}" \
    TRAIN.EVAL_CRIT_PARAMS "${eval_crit_params}" \
    TRAIN.EVAL_CRIT_NAME "${eval_crit_name}" \
    TEST.CHECKPOINT_FILE_PATH "${trained_checkpoint_file}"

if [[ ${copy_to} != '' ]]; then
    dest_dir="${copy_to}/${label_type}s-from_video"
    rsync -a "${out_dir}" "${dest_dir}"
fi
