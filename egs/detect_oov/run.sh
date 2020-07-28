#! /bin/bash
set -ue

# SET WHICH PROCESSING STAGES ARE RUN
start_at=1
stop_after=1

# DATA DIRS CREATED OR MODIFIED BY THIS SCRIPT
output_dir="$HOME/repo/kinemparse/data/output/detect_oov"
oov_dir="${output_dir}/oov-collapsed"

# READ-ONLY DATA DIRS
imu_dir="$HOME/repo/kinemparse/data/output/predict-joined"
imu_data_dir="${imu_dir}/imu-data"
fusion_dir="$HOME/repo/kinemparse/data/output/fuse-modalities"
fused_scores_dir="${fusion_dir}/predict-assemblies_per-pixel"
keyframe_decode_scores_dir="${fusion_dir}/register-keyframes_normalization=per-pixel"

# DEFINE THE FILE STRUCTURE USED BY THIS SCRIPT
eg_root=$(pwd)
scripts_dir="${eg_root}/scripts"
config_dir="${eg_root}/config"
cd $scripts_dir

STAGE=1

if [ "$start_at" -le $STAGE ]; then
    echo "STAGE ${STAGE}: Detect OOV assemblies"
    python detect_oov.py \
        --config_file "${config_dir}/detect_oov.yaml" \
        --out_dir "${oov_dir}" \
        --data_dir "${imu_data_dir}/data" \
        --scores_dir "${fused_scores_dir}/data" \
        --cv_data_dir "${keyframe_decode_scores_dir}/data" \
        --eq_class "is oov"
fi
if [ "$stop_after" -eq $STAGE ]; then
    exit 1
fi
((++STAGE))
