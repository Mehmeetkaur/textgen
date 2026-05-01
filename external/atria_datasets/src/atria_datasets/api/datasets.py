"""API functions for loading and preprocessing datasets."""

from __future__ import annotations

from atria_logger import get_logger
from atria_types._common import DatasetSplitType

from atria_datasets.core import Dataset, DatasetConfig
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.registry import DATASET

logger = get_logger(__name__)


def load_dataset_config(dataset_name: str, **kwargs) -> DatasetConfig:
    logger.debug(
        f"Loading dataset config for dataset: {dataset_name} with params: {kwargs}"
    )
    return DATASET.load_module_config(dataset_name, **kwargs)


def load_dataset(
    dataset_name: str,
    data_dir: str | None = None,
    split: DatasetSplitType | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    allowed_keys: set[str] | None = None,
    num_processes: int = 8,
    cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
    enable_cached_splits: bool = False,
    store_artifact_content: bool = True,
    max_cache_image_size: int | None = None,
    **kwargs,
) -> Dataset:
    config = load_dataset_config(dataset_name, **kwargs)
    return config.build(
        data_dir=data_dir,
        split=split,
        access_token=access_token,
        overwrite_existing_cached=overwrite_existing_cached,
        allowed_keys=allowed_keys,
        num_processes=num_processes,
        cached_storage_type=cached_storage_type,
        enable_cached_splits=enable_cached_splits,
        store_artifact_content=store_artifact_content,
        max_cache_image_size=max_cache_image_size,
    )
