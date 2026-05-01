"""Dataset Builder Module"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from atria_logger import get_logger
from atria_types import DatasetSplitType

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
)
from atria_datasets.core.dataset._common import (
    DatasetConfig,
    _get_storage_manager,
    _save_dataset_info,
)
from atria_datasets.core.dataset._dataset_builders import DefaultOutputTransformer
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)


class ComposedTransform:
    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, sample):
        for transform in self._transforms:
            sample = transform(sample)
        return sample


class DatasetProcessor:
    def __init__(
        self,
        dataset: Dataset,
        transform: Callable,
        split: DatasetSplitType | None = None,
        allowed_keys: set[str] | None = None,
        processed_data_dir: str | None = None,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
    ):
        self._dataset = dataset
        self._split = split
        self._allowed_keys = allowed_keys
        self._transform = transform
        self._storage_dir = self._setup_storage_dir(
            processed_data_dir=processed_data_dir
        )
        self._cached_storage_type = cached_storage_type
        self._overwrite_existing_cached = overwrite_existing_cached
        self._store_artifact_content = store_artifact_content
        self._max_cache_image_size = max_cache_image_size
        self._num_processes = num_processes

    @property
    def dataset(self) -> Dataset:
        """Get the dataset being built."""
        return self._dataset

    @property
    def transform(self) -> Callable:
        """Get the transform applied to the dataset being built."""
        return self._transform

    @property
    def dataset_name(self) -> str:
        """Get the name of the dataset being built."""
        return (
            self.dataset_config.name
            if self.dataset_config.name
            else self._dataset.__class__.__name__
        )

    @property
    def dataset_config(self) -> DatasetConfig:
        """Get the configuration of the dataset being built."""
        return self._dataset.config

    @property
    def dataset_metadata(self):
        """Get the metadata of the dataset being built."""
        return self._dataset.metadata

    @property
    def data_model(self):
        """Get the data model of the dataset being built."""
        return self._dataset.data_model

    def _setup_storage_dir(self, processed_data_dir: str | None) -> Path:
        return (
            Path(processed_data_dir or _DEFAULT_ATRIA_DATASETS_CACHE_DIR)
            / self.dataset_name
            / ("processed_" + _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR)
        )

    def _prepare_split(
        self,
        split: DatasetSplitType,
        store_artifact_content: bool = True,
        resize_images: bool = False,
        image_max_size: int = 1024,
    ) -> SplitIterator:
        if split not in self.dataset._available_splits():
            raise ValueError(f"Split {split} is not available in the dataset.")
        split_iterator = self.dataset._split_iterators[split]

        # get detault output transform for the split
        output_transform = DefaultOutputTransformer(
            data_dir=self._storage_dir,
            store_artifact_content=store_artifact_content,
            resize_images=resize_images,
            image_max_size=image_max_size,
        )

        # compose the transforms now
        split_iterator.output_transform = ComposedTransform(
            transforms=[output_transform, self._transform]
        )
        return split_iterator

    def process_splits(self) -> dict[DatasetSplitType, SplitIterator]:
        """Prepare cached splits using DeltaLake / Msgpack storage."""
        storage_manager = _get_storage_manager(
            self._cached_storage_type,
            storage_dir=str(self._storage_dir),
            config_name=self.dataset_config.config_name,
            num_processes=self._num_processes,
        )

        info_saved = False
        split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        for split in self.dataset._available_splits():
            if self._split is not None and split != self._split:
                continue
            split_exists = storage_manager.split_exists(split=split)
            if split_exists and self._overwrite_existing_cached:
                logger.warning(f"Overwriting existing cached split {split.value}")
                storage_manager.purge_split(split)
                split_exists = False

            if not split_exists:
                split_iterator = self._prepare_split(
                    split=split,
                    store_artifact_content=self._store_artifact_content,
                    resize_images=self._max_cache_image_size is not None,
                    image_max_size=self._max_cache_image_size,
                )
                logger.info(
                    f"Caching split [{split.value}] to {self._storage_dir} with max_len={split_iterator._max_len}"
                )
                storage_manager.write_split(split_iterator=split_iterator)
                if not info_saved:
                    _save_dataset_info(
                        self._storage_dir,
                        self.dataset_config.config_name,
                        self.dataset_config.model_dump(),
                        self.dataset_metadata.model_dump(),
                    )
                    info_saved = True
            else:
                logger.info(
                    f"Loading cached split {split.value} from {storage_manager.split_dir(split)}"
                )

        if not info_saved:
            _save_dataset_info(
                self._storage_dir,
                self.dataset_config.config_name,
                self.dataset_config.model_dump(),
                self.dataset_metadata.model_dump(),
            )
            info_saved = True

        for split in self.dataset._available_splits():
            if self._split is not None and split != self._split:
                continue
            split_iterators[split] = storage_manager.read_split(
                split=split, data_model=self.data_model, allowed_keys=self._allowed_keys
            )

        return split_iterators
