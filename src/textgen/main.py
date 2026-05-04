from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath("external"))
sys.path.insert(0, os.path.abspath("src"))

from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
from atria_datasets import Dataset
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.registry.ser.funsd import *  # noqa
from atria_logger import get_logger
from atria_types import DocumentInstance
from pydantic import BaseModel
from textgen.transforms import TransformV1


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


def load_dataset(
    name: str,
    max_samples: int | None = None,
    data_dir: str | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 0,
    train_transform: BaseModel | None = None,
    eval_transform: BaseModel | None = None,
) -> Dataset:
    from atria_datasets import load_dataset_config

    dataset_config = load_dataset_config(
        name,
        max_train_samples=max_samples,
        max_test_samples=max_samples,
        max_validation_samples=max_samples,
    )
    logger.info(f"Loaded dataset config:\n{dataset_config}")

    if data_dir is not None:
        data_dir += "/" + name.split("/")[0]

    dataset = dataset_config.build(
        data_dir=data_dir,
        access_token=access_token,
        overwrite_existing_cached=overwrite_existing_cached,
        num_processes=num_processes,
        enable_cached_splits=True,
        max_cache_image_size=1024,
    )

    try:
        dataset.train.output_transform = train_transform
    except SplitNotFoundError:
        logger.warning(
            "Train split not found in dataset, skipping train transform assignment."
        )

    try:
        dataset.validation.output_transform = eval_transform
    except SplitNotFoundError:
        logger.warning(
            "Validation split not found in dataset, skipping validation transform assignment."
        )

    try:
        dataset.test.output_transform = TransformV1(train=False)
    except SplitNotFoundError:
        logger.warning(
            "Test split not found in dataset, skipping test transform assignment."
        )

    return dataset


def load_transform(train: bool = True) -> TransformV1:
    transform = TransformV1(train=train)
    logger.info(f"Loaded transform: {transform}")
    return transform


def build_dataloader(
    dataset_split: Any,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x,
    )


def iterate_dataset(
    name: str,
    batch_size: int = 4,
    data_dir: str | None = None,
    max_samples: int | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 0,
    output_dir: str = "./output",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        name=name,
        max_samples=max_samples,
        data_dir=data_dir,
        access_token=access_token,
        overwrite_existing_cached=overwrite_existing_cached,
        num_processes=num_processes,
        train_transform=load_transform(train=True),
        eval_transform=load_transform(train=False),
    )

    # visualize first sample in the train dataset
    for sample_index, sample in enumerate(dataset.train):
        logger.info(
        "TRAIN_SAMPLE | idx=%d | text=%s | keys=%s",
        sample_index,
        sample["text"],
        list(sample.keys())
        )

        image = sample["image"]
        mask = sample["mask"]

        assert image.size == (256, 256), f"Bad image size: {image.size}"
        assert mask.size == (256, 256), f"Bad mask size: {mask.size}"

        image.save(output_path / "train" / f"image_{sample_index}.png")
        mask.save(output_path / "train" / f"mask_{sample_index}.png")

        break

    for sample_index, sample in enumerate(dataset.test):
        logger.info(
        "TEST_SAMPLE | idx=%d | keys=%s",
        sample_index,
        list(sample.keys()) if isinstance(sample, dict) else type(sample)
        )

        if isinstance(sample, dict):
           image = sample["image"]
           mask = sample["mask"]

           image.save(output_path / "test" / f"image_{sample_index}.png")
           mask.save(output_path / "test" / f"mask_{sample_index}.png")

        else:

            if hasattr(sample, "viz"):
               sample.viz.visualize(output_path / "test" / f"sample_{sample_index}.png")

        break

    train_dataloader = build_dataloader(
        dataset.train, batch_size=batch_size, shuffle=True
    )
    test_dataloader = build_dataloader(
        dataset.test, batch_size=batch_size, shuffle=False
    )

    for batch_index, batch in enumerate(train_dataloader):
        logger.info(
            "Processing batch %s with %s samples", batch_index, len(batch)
        )
        logger.info("Sample 1 in batch: %s", batch[0])
        break

    for batch_index, batch in enumerate(test_dataloader):
        logger.info(
            "Processing batch %s with %s samples", batch_index, len(batch)
        )
        logger.info("Sample 1 in batch: %s", batch[0])
        break


def main():
    fire.Fire(iterate_dataset)


if __name__ == "__main__":
    main()