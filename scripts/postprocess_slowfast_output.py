import os
import logging
import pickle

import pandas as pd
import yaml

from mathtools import utils


logger = logging.getLogger(__name__)


def main(
        out_dir=None, data_dir=None, results_file=None, cv_file=None,
        col_format=None, win_params={}, slowfast_csv_params={}):
    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)
    results_file = os.path.expanduser(results_file)
    cv_file = os.path.expanduser(cv_file)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    vocab = utils.loadVariable('vocab', data_dir)
    metadata = utils.loadMetadata(data_dir)
    slowfast_labels = pd.read_csv(
        cv_file, keep_default_na=False, index_col=0,
        **slowfast_csv_params
    )
    seg_ids = slowfast_labels.index.to_numpy()
    vid_names = slowfast_labels['video_name'].unique().tolist()
    metadata['seq_id'] = metadata.index
    vid_ids = metadata.set_index('dir_name').loc[vid_names].set_index('seq_id').index
    metadata = metadata.drop('seq_id', axis=1)

    with open(results_file, 'rb') as file_:
        model_scores, gt_labels = pickle.load(file_)
        model_scores = model_scores.numpy()
        gt_labels = gt_labels.numpy()

    if len(model_scores) != len(seg_ids):
        err_str = f"{len(model_scores)} segment scores != {slowfast_labels.shape[0]} CSV rows"
        raise AssertionError(err_str)

    logger.info(f"Loaded {len(seg_ids)} segments, {len(vid_ids)} videos")

    for vid_id, vid_name in zip(vid_ids, vid_names):
        matches_video = (slowfast_labels['video_name'] == vid_name).to_numpy()
        win_labels = gt_labels[matches_video]
        win_scores = model_scores[matches_video, :]

        if win_labels.shape == win_scores.shape:
            win_preds = (win_scores > 0.5).astype(int)
        else:
            win_preds = win_scores.argmax(axis=1)

        seq_id_str = f"seq={vid_id}"
        utils.saveVariable(win_scores, f'{seq_id_str}_score-seq', out_data_dir)
        utils.saveVariable(win_labels, f'{seq_id_str}_true-label-seq', out_data_dir)
        utils.saveVariable(win_preds, f'{seq_id_str}_pred-label-seq', out_data_dir)
        utils.plot_array(
            win_scores.T, (win_labels.T, win_preds.T), ('true', 'pred'),
            tick_names=vocab,
            fn=os.path.join(fig_dir, f"{seq_id_str}.png"),
            subplot_width=12, subplot_height=5
        )
    utils.saveVariable(vocab, 'vocab', out_data_dir)
    utils.saveMetadata(metadata, out_data_dir)


if __name__ == "__main__":
    # Parse command-line args and config file
    cl_args = utils.parse_args(main)
    config, config_fn = utils.parse_config(cl_args, script_name=__file__)

    # Create output directory, instantiate log file and write config options
    out_dir = os.path.expanduser(config['out_dir'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, config_fn), 'w') as outfile:
        yaml.dump(config, outfile)
    utils.copyFile(__file__, out_dir)

    main(**config)
