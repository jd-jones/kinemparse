import os
import math

import joblib
import yaml
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
import skimage

from mathtools import utils, torchutils
from visiontools import imageprocessing


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def drawBBs(d_pred, image):
    image = draw_bounding_boxes(
        image, d_pred['boxes'], d_pred['labels'],
        label_names=COCO_INSTANCE_CATEGORY_NAMES, colors=None, width=1
    )
    return image


def draw_bounding_boxes(image, boxes, labels, label_names=None, colors=None, width=1):
    """
    Draws bounding boxes on given image.
    Args:
        image (Tensor): Tensor of shape (C x H x W) or (1 x C x H x W)
        bboxes (Tensor): Tensor of size (N, 4)
            containing bounding boxes in (xmin, ymin, xmax, ymax) format.
        labels (Tensor): Tensor of size (N) Labels for each bounding boxes.
        label_names (List): List containing labels excluding background.
        colors (dict): Dict with key as label id and value as color name.
        width (int): Width of bounding box.
    """

    if not isinstance(image, torch.Tensor):
        raise TypeError(f'tensor expected, got {type(image)}')

    if image.dim() == 4:
        if image.shape[0] == 1:
            image = image.squeeze(0)
        else:
            raise ValueError("Batch size > 1 is not supported. Pass images with batch size 1 only")

    # if label_names is not None:
    #     # Since for our detection models class 0 is background
    #     label_names.insert(0, "__background__")

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Neceassary check to remove grad if present
    if boxes.requires_grad:
        boxes = boxes.detach()

    boxes = boxes.to(torch.int64).tolist()
    labels = labels.to(torch.int64).tolist()

    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)

    for bbox, label in zip(boxes, labels):
        if colors is None:
            draw.rectangle(bbox, width=width)
        else:
            draw.rectangle(bbox, width=width, outline=colors[label])

        if label_names is None:
            draw.text((bbox[0], bbox[1]), str(label))
        else:
            draw.text((bbox[0], bbox[1]), label_names[int(label)])

    return torch.from_numpy(np.array(img_to_draw))


def detectCategories(model, inputs):
    def makeCategoryMask(d_pred, i, category_label):
        masks = d_pred['masks']
        label_matches_category = d_pred['labels'] == category_label

        if not label_matches_category.any():
            category_mask = torch.zeros(
                (1,) + inputs.shape[-2:],
                dtype=torch.uint8, device=inputs.device
            )
            return category_mask

        category_masks = masks[label_matches_category]
        category_mask, _ = (category_masks > 0.5).max(dim=0)
        return category_mask

    maskrcnn_preds = model([img.to(device=model.device) for img in inputs])
    person_masks = torch.stack([
        makeCategoryMask(d_pred, i, 1).to(device=inputs.device)
        for i, d_pred in enumerate(maskrcnn_preds)
    ])
    background_masks = torch.stack([
        makeCategoryMask(d_pred, i, 0).to(device=inputs.device)
        for i, d_pred in enumerate(maskrcnn_preds)
    ])
    return person_masks, background_masks


def main(
        out_dir=None, data_dir=None, gpu_dev_id=None, batch_size=None,
        start_from=None, stop_at=None, num_disp_imgs=None):

    out_dir = os.path.expanduser(out_dir)
    data_dir = os.path.expanduser(data_dir)

    logger = utils.setupRootLogger(filename=os.path.join(out_dir, 'log.txt'))

    fig_dir = os.path.join(out_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    out_data_dir = os.path.join(out_dir, 'data')
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)

    def loadFromDir(var_name, dir_name):
        return joblib.load(os.path.join(dir_name, f"{var_name}.pkl"))

    def saveToWorkingDir(var, var_name):
        joblib.dump(var, os.path.join(out_data_dir, f"{var_name}.pkl"))

    trial_ids = utils.getUniqueIds(data_dir, prefix='trial=', to_array=True)

    device = torchutils.selectDevice(gpu_dev_id)

    for seq_idx, trial_id in enumerate(trial_ids):

        if start_from is not None and seq_idx < start_from:
            continue

        if stop_at is not None and seq_idx > stop_at:
            break

        trial_str = f"trial={trial_id}"

        logger.info(f"Processing video {seq_idx + 1} / {len(trial_ids)}  (trial {trial_id})")

        logger.info("  Loading data...")
        try:
            rgb_frame_seq = loadFromDir(f"{trial_str}_rgb-frame-seq", data_dir)
            rgb_frame_seq = np.stack(
                tuple(skimage.img_as_float(f) for f in rgb_frame_seq),
                axis=0
            )
        except FileNotFoundError as e:
            logger.info(e)
            continue

        logger.info("  Detecting objects...")
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = model.to(device=device)
        model.device = device
        model.eval()

        inputs = np.moveaxis(rgb_frame_seq, 3, 1)

        if batch_size is None:
            batch_size = inputs.shape[0]

        def detectBatch(batch_index):
            start = batch_size * batch_index
            end = start + batch_size
            in_batch = torch.tensor(inputs[start:end], dtype=torch.float)
            out_batches = detectCategories(model, in_batch)
            return tuple(batch.numpy().squeeze(axis=1) for batch in out_batches)

        num_batches = math.ceil(inputs.shape[0] / batch_size)
        person_mask_seq, bg_mask_seq = map(
            np.vstack,
            zip(*(detectBatch(i) for i in range(num_batches)))
        )
        person_mask_seq = person_mask_seq.astype(bool)

        logger.info("  Saving output...")
        saveToWorkingDir(person_mask_seq, f'{trial_str}_person-mask-seq')

        if num_disp_imgs is not None:
            if rgb_frame_seq.shape[0] > num_disp_imgs:
                idxs = np.arange(rgb_frame_seq.shape[0])
                np.random.shuffle(idxs)
                idxs = idxs[:num_disp_imgs]
            else:
                idxs = slice(None, None, None)
            imageprocessing.displayImages(
                *(rgb_frame_seq[idxs]), *(person_mask_seq[idxs]), *(bg_mask_seq[idxs]),
                num_rows=3, file_path=os.path.join(fig_dir, f'{trial_str}_best-frames.png')
            )


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
