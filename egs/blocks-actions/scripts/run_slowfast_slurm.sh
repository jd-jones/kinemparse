#!/usr/bin/zsh
#SBATCH --job-name="run_slowfast"
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=jdjones@jhu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem-per-gpu=15000
#SBATCH -c 20
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00

set -ue


# -=( SET CONFIG )==-----------------------------------------------------------
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


# -=( MAIN SCRIPT )==----------------------------------------------------------
srun ./run_slowfast_all-cvfolds.sh \
    --copy_to="thin6:/home/jjone229/data/files-from-mike" \
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
