"""
Utilities for Batch Samplers

This module provides utility functions for working with batch samplers, including
functions for computing aspect ratios of datasets, repeating elements to meet
a minimum count, and creating aspect ratio groups for efficient batching.

Functions:
    - repeat_to_at_least: Repeats elements of an iterable to meet a minimum count.
    - compute_aspect_ratios: Computes aspect ratios for various dataset types.
    - _create_aspect_ratio_groups: Creates groups based on aspect ratios for batching.
    - _quantize: Quantizes values into bins.

Dependencies:
    - torch: For dataset utilities and data loading.
    - torchvision: For dataset-specific utilities.
    - numpy: For numerical operations.
    - tqdm: For progress visualization.
    - PIL: For image processing in VOC datasets.
    - atria.logger: For logging.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from atria_logger import get_logger

logger = get_logger(__name__)


def _repeat_to_at_least(iterable, n):
    """
    Repeats elements of an iterable to meet or exceed a minimum count.

    Args:
        iterable (iterable): The iterable to repeat.
        n (int): The minimum number of elements required.

    Returns:
        list: A list containing repeated elements from the iterable.
    """
    import math
    from itertools import chain, repeat

    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


def _compute_aspect_ratios_slow(dataset, indices=None):
    """
    Computes aspect ratios for a dataset using a slow method.

    This method loads each sample individually and computes its aspect ratio.

    Args:
        dataset: The dataset to compute aspect ratios for.
        indices (list[int], optional): A list of indices to compute aspect ratios for.
            If None, computes for the entire dataset.

    Returns:
        list[float]: A list of aspect ratios for the dataset.
    """

    import torch
    from torch.utils.data.sampler import Sampler
    from tqdm import tqdm

    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=14,
        collate_fn=lambda x: x[0],
    )
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    """
    Computes aspect ratios for a custom dataset with a `get_height_and_width` method.

    Args:
        dataset: The dataset to compute aspect ratios for.
        indices (list[int], optional): A list of indices to compute aspect ratios for.
            If None, computes for the entire dataset.

    Returns:
        list[float]: A list of aspect ratios for the dataset.
    """

    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    """
    Computes aspect ratios for a COCO dataset.

    Args:
        dataset: The COCO dataset to compute aspect ratios for.
        indices (list[int], optional): A list of indices to compute aspect ratios for.
            If None, computes for the entire dataset.

    Returns:
        list[float]: A list of aspect ratios for the dataset.
    """

    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    """
    Computes aspect ratios for a VOC dataset.

    Args:
        dataset: The VOC dataset to compute aspect ratios for.
        indices (list[int], optional): A list of indices to compute aspect ratios for.
            If None, computes for the entire dataset.

    Returns:
        list[float]: A list of aspect ratios for the dataset.
    """

    from PIL import Image

    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    """
    Computes aspect ratios for a subset of a dataset.

    Args:
        dataset: The subset dataset to compute aspect ratios for.
        indices (list[int], optional): A list of indices to compute aspect ratios for.
            If None, computes for the entire dataset.

    Returns:
        list[float]: A list of aspect ratios for the dataset.
    """
    if indices is None:
        indices = range(len(dataset))
    ds_indices = [dataset.indices[i] for i in indices]
    return _compute_aspect_ratios(dataset.dataset, ds_indices)


def _compute_aspect_ratios(dataset, indices=None):
    """
    Computes aspect ratios for a dataset, selecting the appropriate method based on the dataset type.

    Args:
        dataset: The dataset to compute aspect ratios for.
        indices (list[int], optional): A list of indices to compute aspect ratios for.
            If None, computes for the entire dataset.

    Returns:
        list[float]: A list of aspect ratios for the dataset.
    """

    from torch.utils.data import Subset
    from torchvision import datasets

    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)
    if isinstance(dataset, datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)
    if isinstance(dataset, datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)
    if isinstance(dataset, Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    """
    Quantizes values into bins.

    Args:
        x (list[float]): The values to quantize.
        bins (list[float]): The bin edges.

    Returns:
        list[int]: The quantized values as bin indices.
    """
    import bisect
    import copy

    bins = sorted(copy.deepcopy(bins))
    return list(map(lambda y: bisect.bisect_right(bins, y), x))  # noqa: C417


def _create_aspect_ratio_groups(dataset, k=0):
    """
    Creates groups based on aspect ratios for batching.

    Args:
        dataset: The dataset to group.
        k (int): The number of bins for aspect ratio quantization.

    Returns:
        list[int]: A list of group IDs for each sample in the dataset.
    """

    import numpy as np

    aspect_ratios = _compute_aspect_ratios(dataset)
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]
    groups = _quantize(aspect_ratios, bins)
    counts = np.unique(groups, return_counts=True)[1]
    fbins = [0] + bins + [np.inf]
    logger.info(f"Using {fbins} as bins for aspect ratio quantization")
    logger.info(f"Count of instances per bin: {counts}")
    return groups
