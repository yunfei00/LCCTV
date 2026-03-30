import math

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """Extract a square crop centered at target_bb.

    This is intentionally kept local to the standalone inference path so that
    `tracking/run_inference.py` does not depend on the training data package.
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
    if crop_sz < 1:
        crop_sz = 1

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)
    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)

    height, width, _ = im_crop_padded.shape
    att_mask = np.ones((height, width))
    end_x = -x2_pad
    end_y = -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode="constant", value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = F.interpolate(
            mask_crop_padded[None, None],
            (output_sz, output_sz),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    if mask is None:
        return im_crop_padded, att_mask.astype(np.bool_), 1.0
    return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded


def transform_image_to_crop(
    box_in: torch.Tensor,
    box_extract: torch.Tensor,
    resize_factor: float,
    crop_sz: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Map a box from image coordinates to crop coordinates."""
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]
    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    return box_out
