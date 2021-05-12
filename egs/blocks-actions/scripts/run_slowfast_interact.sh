# #!/usr/bin/zsh

set -ue

# -=( SET DEFAULTS )==---------------------------------------------------------
label_type='event'
config_dir="/home/map6/jon/kinemparse/egs/ikea_anu/config"
data_dir='/wrk/map6/blocks_data/blocks-videos-as-jpg/child'
base_dir="/wrk/map6/blocks_output/blocks-actions"
num_classes=33
loss_func='cross_entropy'
eval_crit='topk_accuracy'
eval_crit_params='["k", 1]'
eval_crit_name='top1_acc'

vocab_file="${base_dir}/dataset/${label_type}-dataset/vocab.json"
num_classes=$((`cat ${vocab_file} | tr -cd ',' | wc -c`+1))

# -=( SET I/O PATHS )==--------------------------------------------------------
phase_dir="${base_dir}/${label_type}s-from-video"
folds_dir="${phase_dir}/cv-folds/data"
out_dir="${phase_dir}/run-slowfast"

pretrained_checkpoint_file="${base_dir}/I3D_8x8_R50.pkl"
trained_checkpoint_file=''


# -=( PREPARE ENVIRONMENT )==--------------------------------------------------
cd "/home/map6/jon/CompositionalActions/slowfast"


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
