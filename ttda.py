"""Test time augmentation for image classification and segmentation tasks"""
import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
import cv2
from albumentations.augmentations.geometric.functional import smallest_max_size, resize, longest_max_size
from albumentations.augmentations.crops.functional import crop, center_crop
from albumentations.augmentations.functional import hflip, pad
from typing import List, Union


def get_bbox(img, thres=1200, dilated=10):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[0], img.shape[1]
    x_min = 0
    for i in range(width // 3):
        if img[:, i].sum() > thres:
            x_min = i
            break
    x_min = max(0, x_min - dilated)

    x_max = width - 1
    for i in range(width - 1, width * 2 // 3, -1):
        if img[:, i].sum() > thres:
            x_max = i
            break
    x_max = min(width - 1, x_max + dilated)

    y_min = 0
    for i in range(height // 3):
        if img[i, :].sum() > thres:
            y_min = i
            break
    y_min = max(0, y_min - dilated)

    y_max = height - 1
    for i in range(height - 1, height * 2 // 3, -1):
        if img[i, :].sum() > thres:
            y_max = i
            break
    y_max = min(height - 1, y_max + dilated)

    return (x_min, x_max, y_min, y_max)


def preprocess(img, dsize=896, thres=1200, dilated=10):
    # resize_img = longest_max_size(img, dsize + 128, interpolation=cv2.INTER_LINEAR)
    resize_img = smallest_max_size(img, dsize, interpolation=cv2.INTER_LINEAR)
    bbox = get_bbox(resize_img, thres, dilated)
    crop_img = resize_img[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
    # crop_img = longest_max_size(crop_img, max_size=dsize, interpolation=cv2.INTER_LINEAR)
    # pad_img = pad(crop_img, 896, 896, border_mode=cv2.BORDER_CONSTANT, value=0)

    return crop_img


def fivecrop(
    image: np.ndarray,
    resize_max_size: int = 256,
    crop_size: List[int] = [224, 224],
    flip: bool = True
) -> np.ndarray:
    """five crop image augment

    Args:
        image (np.ndarray): H x W x C
        resize_max_size (int, optional): Defaults to 256.
        crop_size (List[int], optional): Defaults to [224, 224].
        flip (bool): Defaults to True

    Returns:
        np.ndarray: 5/10 x C x crop_size
    """
    resized_image = smallest_max_size(image, max_size=resize_max_size, interpolation=cv2.INTER_LINEAR)
    height, width = resized_image.shape[0], resized_image.shape[1]
    crop_height, crop_width = crop_size
    assert (
        crop_height <= height and crop_width <= width
    ), "crop size {} does not match smalest_max_size {}".format(crop_size, resize_max_size)

    tl = crop(resized_image, 0, 0, crop_width, crop_height)
    tr = crop(resized_image, width - crop_width, 0,  width, crop_height)
    bl = crop(resized_image, 0, height - crop_height, crop_width, height)
    br = crop(resized_image, width - crop_width, height - crop_height, width, height)

    cc = center_crop(resized_image, crop_height, crop_width)

    if flip:
        ret = np.stack(
            (cc, tl, tr, bl, br, hflip(cc), hflip(tl), hflip(tr), hflip(bl), hflip(br))
        )
    else:
        ret = np.stack((cc, tl, tr, bl, br))

    ret = np.einsum("nijc->ncij", ret)
    return ret


def centercrop(
    image: np.ndarray,
    resize_max_size: int = 256,
    crop_size: List[int] = [224, 224],
    flip: bool = True
) -> np.ndarray:
    resized_image = smallest_max_size(image, max_size=resize_max_size, interpolation=cv2.INTER_LINEAR)
    height, width = resized_image.shape[0], resized_image.shape[1]
    crop_height, crop_width = crop_size
    assert (
        crop_height <= height and crop_width <= width
    ), "crop size {} does not match smalest_max_size {}".format(crop_size, resize_max_size)

    cc = center_crop(resized_image, crop_height, crop_width)

    if flip:
        ret = np.stack((cc, hflip(cc)))
    else:
        ret = np.expand_dims(cc, axis=0)
    ret = np.einsum("nijc->ncij", ret)

    return ret


def resize_and_centercrop(
    image: np.ndarray,
    resize_max_size: int = 256,
    crop_size: List[int] = [224, 224],
    flip: bool = True
) -> np.ndarray:
    resized_image = smallest_max_size(image, max_size=resize_max_size, interpolation=cv2.INTER_CUBIC)
    height, width = resized_image.shape[0], resized_image.shape[1]
    crop_height, crop_width = crop_size
    assert (
        crop_height <= height and crop_width <= width
    ), "crop size {} does not match smalest_max_size {}".format(crop_size, resize_max_size)

    rs = resize(resized_image, crop_height, crop_width)
    cc = center_crop(resized_image, crop_height, crop_width)

    if flip:
        ret = np.stack((rs, cc, hflip(rs), hflip(cc)))
    else:
        ret = np.stack((rs, cc))
    ret = np.einsum("nijc->ncij", ret)

    return ret


def resize_and_flip(
    image: np.ndarray,
    dst_size: List[int] = [224, 224],
    flip: bool = False
):
    height, width = dst_size
    rs = resize(image, height, width)

    if flip:
        ret = np.stack((rs, hflip(rs)))
    else:
        ret = np.expand_dims(rs, axis=0)
    ret = np.einsum("nijc->ncij", ret)

    return ret


def resize_smallest_and_flip(
    image: np.ndarray,
    dist_size: Union[List[int], int] = [224, 256],
    flip: bool = False
):
    if isinstance(dist_size, int):
        dist_size = [dist_size]
    ret = []
    for ss in dist_size:
        rs = smallest_max_size(
            image, max_size=ss, interpolation=cv2.INTER_LINEAR
        )
        if flip:
            rs = np.stack((rs, hflip(rs)))
        else:
            rs = np.expand_dims(rs, axis=0)
        rs = np.einsum("nijc->ncij", rs)
        ret.append(rs)
    if len(ret) == 1:
        ret = ret[0]
    return ret


def normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 3:
        mean = image.mean()
        std = image.std()
        denominator = np.reciprocal(std, dtype=np.float32)
        out = (image - mean) * denominator
    else:
        mean = image.mean(axis=[1, 2, 3])

    return out


def geometric_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute geometric mean along given dimension.
    This implementation assume values are in range (0...1) (Probabilities)
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension to reduce
    Returns:
        Tensor
    """
    return x.log().mean(dim=dim).exp()


def fuse_predicts(x: torch.Tensor, reduce: str = "mean"):
    if reduce == "mean":
        return x.mean(dim=0)
    elif reduce == "max":
        return x.max(dim=0)[0]
    elif reduce == "gmean":
        return geometric_mean(x, dim=0)
    elif reduce == "tsharpen":
        return (x ** 0.5).mean(dim=0)
    else:
        raise NotImplementedError("Invalid reduce method : {}".format(reduce))


def augment(input: np.ndarray, aug_cfg: DictConfig):
    if aug_cfg.method == "fivecrop":
        return fivecrop(
            input,
            resize_max_size=aug_cfg.resize_small_size,
            crop_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "centercrop":
        return centercrop(
            input,
            resize_max_size=aug_cfg.resize_small_size,
            crop_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "resize_and_centercrop":
        return resize_and_centercrop(
            input,
            resize_max_size=aug_cfg.resize_small_size,
            crop_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "resize": 
    # target size for model : 512 x 512
    # original size : 512 x 800 ratio : width/height = 1.5625
    # outpu size of the func: 512 x 512
        return resize_and_flip(
            input,
            dst_size=aug_cfg.crop_size,
            flip=aug_cfg.flip
        )
    elif aug_cfg.method == "resize_smallest":
    # target size for model : 512 x any / any x 512
    # original size : 512 x 800 ratio : width/height = 1.5625
    # output size is : 512 x 800 (keep ratio of the image)
        return resize_smallest_and_flip(
            input,
            dist_size=aug_cfg.resize_small_size,
            flip=aug_cfg.flip
        )
    else:
        raise NotImplementedError(
            "Invalid agument method : {}".format(aug_cfg.method)
        )
