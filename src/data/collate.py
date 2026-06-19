from typing import List

import torch
from torch import Tensor


def _max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for i, item in enumerate(sublist):
            maxes[i] = max(maxes[i], item)
    return maxes


def _max_by_axis_pad(the_list: List[List[int]]) -> List[int]:
    maxes = _max_by_axis(the_list)
    block = 128
    for i in range(2):
        maxes[i + 1] = ((maxes[i + 1] - 1) // block + 1) * block
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> Tensor:
    if tensor_list[0].ndim != 3:
        raise ValueError("Only 3D CHW tensors are supported")
    max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros((b, c, h, w), dtype=dtype, device=device)
    for img, pad_img in zip(tensor_list, tensor):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return tensor


def collate_fn_crowd(batch):
    batch_new = []
    for imgs, points in batch:
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i, :, :, :], points[i]))
    batch = list(zip(*batch_new))
    return nested_tensor_from_tensor_list(batch[0]), batch[1]
