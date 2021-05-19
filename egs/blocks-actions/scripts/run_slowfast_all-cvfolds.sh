#!/usr/bin/zsh

set -ue


# -=( MAIN SCRIPT )==----------------------------------------------------------
# cv_fold_indices=(0 1 2 3 4)
cv_fold_indices=(4)

for i in ${cv_fold_indices[@]}; do
    datetime=$(date +"%Y-%m-%d-%H:%M")
	./run_slowfast.sh \
        --out_dir_name="run-slowfast_cvfold=${i}_${datetime}" \
        --train_fold_fn="cvfold=${i}_train_slowfast-labels_seg.csv" \
        --val_fold_fn="cvfold=${i}_val_slowfast-labels_win.csv" \
        --test_fold_fn="cvfold=${i}_test_slowfast-labels_win.csv" \
        "$@"
done
