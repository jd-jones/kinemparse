#!/usr/bin/zsh

set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'
config_dir="/home/map6/jon/kinemparse/egs/blocks-actions/config"
data_dir='/wrk/map6/blocks_data/blocks-videos-as-jpg/child'
base_dir="/wrk/map6/blocks_output/blocks-actions"
loss_func='cross_entropy'
head_act='softmax'
eval_crit='topk_accuracy'
eval_crit_params='["k", 1]'
eval_crit_name='top1_acc'
num_classes=''

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
        *) # Unknown option: print error and exit
            echo "Error: Unrecognized argument ${arg}" >&2
            exit 1
            ;;
    esac
done


# -=( MAIN SCRIPT )==----------------------------------------------------------
cv_fold_indices=(0 1 2 3 4)

for i in ${cv_fold_indices[@]}; do
	sbatch --begin="now+$(( i * 8 ))hour" run_slowfast_slurm.sh \
        --copy_to="thin6:/home/jdjones/data/files-from-mike" \
        --out_dir_name="run-slowfast_labels=${label_type}_cvfold=${i}" \
        --train_fold_fn="cvfold=${i}_train_slowfast-labels_seg.csv" \
        --val_fold_fn="cvfold=${i}_val_slowfast-labels_win.csv" \
        --test_fold_fn="cvfold=${i}_test_slowfast-labels_win.csv" \
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
done
