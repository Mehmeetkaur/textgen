"""Hugging Face Dataset Module

This module defines the `HuggingfaceDataset` class, which extends the `Dataset`
class to support datasets hosted on Hugging Face. It provides functionality for managing
dataset splits, configurations, metadata, and runtime transformations specific to Hugging
Face datasets.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import aiohttp
from atria_logger import get_logger
from atria_registry._module_base import ConfigurableModule
from atria_types import (
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
)

from atria_datasets.core.constants import _DEFAULT_DOWNLOAD_PATH
from atria_datasets.core.dataset._common import (
    HuggingfaceDatasetConfig,
    T_BaseDataInstance,
    T_HuggingfaceDatasetConfig,
)
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    import datasets


logger = get_logger(__name__)


class HuggingfaceDataset(
    Dataset[T_HuggingfaceDatasetConfig, T_BaseDataInstance],
    Generic[T_HuggingfaceDatasetConfig, T_BaseDataInstance],
):
    """
    A dataset class for Hugging Face datasets.

    This class extends the `AtriaDataset` class to provide functionality specific
    to datasets hosted on Hugging Face, including metadata extraction, split management,
    and runtime transformations.

    Attributes:
        _data_dir (Path): The directory where dataset files are stored.
        _config (AtriaDatasetConfig): The configuration for the datasets.
        _runtime_transforms (DataTransformsDict): Runtime transformations for training and evaluation.
        _active_split (DatasetSplit): The currently active dataset split.
        _active_split_config (SplitConfig): The configuration for the active split.
        _downloaded_files (Dict[str, Path]): A dictionary of downloaded files.
        _prepared_split_iterator (Iterator): The prepared iterator for the active split.
        _subset_indices (Optional[torch.Tensor]): Indices for a random subset of the datasets.
        _prepared_metadata (DatasetMetadata): Metadata for the datasets.
        _download_dir (Path): The directory for downloaded files.
        _download_manager (DownloadManager): The download manager for the datasets.
    """

    __abstract__ = True
    __config__: type[T_HuggingfaceDatasetConfig] = HuggingfaceDatasetConfig

    def __init__(
        self,
        config: T_HuggingfaceDatasetConfig,
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
    ):
        ConfigurableModule.__init__(self, config=config)

        self._hf_dataset_builder = self._load_hf_dataset_builder(data_dir=data_dir)
        self._hf_split_generators = None
        self._dataset_builder = self._prepare_dataset_builder(
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
        self._split_iterators = self._dataset_builder.prepare_splits()

    def _load_hf_dataset_builder(self, data_dir: str) -> datasets.DatasetBuilder:
        from datasets import load_dataset_builder

        return load_dataset_builder(
            self.config.hf_repo,
            name=self.config.hf_config_name,
            cache_dir=data_dir,
            storage_options={
                "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
            },
        )

    def _custom_download(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Prepares the dataset by downloading and extracting files if data URLs are provided.

        Args:
            data_dir (str): The directory where the dataset files are stored.

        Returns:
            None
        """

        pass

    def _prepare_download_manager(
        self, data_dir: str, download_dir: str
    ) -> datasets.DownloadManager:
        """
        Prepares the download manager for the datasets.

        Returns:
            DownloadManager: The prepared download manager.
        """

        if "packaged_modules" in str(self._hf_dataset_builder.__module__):
            import datasets

            # If it is a packaged module, use the Hugging Face download manager.
            return datasets.DownloadManager(
                dataset_name=self.config.hf_config_name,
                data_dir=data_dir,
                download_config=datasets.DownloadConfig(
                    cache_dir=download_dir,
                    force_download=False,
                    force_extract=False,
                    use_etag=False,
                    delete_extracted=False,
                    storage_options={
                        "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
                    },
                ),
                record_checksums=False,
            )
        else:
            import datasets

            return datasets.DownloadManager(
                data_dir=data_dir,
                download_config=datasets.DownloadConfig(
                    cache_dir=download_dir,
                    force_download=False,
                    force_extract=False,
                    use_etag=False,
                    delete_extracted=False,
                    storage_options={
                        "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
                    },
                ),
                record_checksums=False,
            )

    def _available_splits(self) -> list[DatasetSplitType]:
        """
        Returns a list of available dataset splits.

        Returns:
            List[DatasetSplitType]: A list of available dataset splits.
        """
        if self._hf_split_generators is None:
            download_dir = (
                Path(self._dataset_builder._data_dir) / _DEFAULT_DOWNLOAD_PATH
            )
            download_dir.mkdir(parents=True, exist_ok=True)
            download_manager = self._prepare_download_manager(
                self._dataset_builder._data_dir, download_dir=str(download_dir)
            )
            self._hf_split_generators = self._hf_dataset_builder._split_generators(
                download_manager
            )
            HF_SPLIT_TO_ATRIA_SPLIT = {
                "train": DatasetSplitType.train,
                "validation": DatasetSplitType.validation,
                "test": DatasetSplitType.test,
            }
            self._hf_split_generators = {
                HF_SPLIT_TO_ATRIA_SPLIT[split_generator.name]: split_generator
                for split_generator in self._hf_split_generators
            }
        return list(self._hf_split_generators.keys())

    def _metadata(self) -> DatasetMetadata:
        """
        Extracts metadata from the Hugging Face datasets.

        Returns:
            DatasetMetadata: The metadata for the datasets.
        """
        from atria_types import DatasetMetadata

        return DatasetMetadata.from_huggingface_info(self._hf_dataset_builder.info)

    def _split_iterator(  # type: ignore
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[Any, None, None]:
        """
        Returns an iterator for a specific dataset split.

        Args:
            split (DatasetSplit): The dataset split.
            hf_split_generator (datasets.SplitGenerator): The Hugging Face split generator.

        Yields:
            BaseDataInstanceType: The dataset instances for the specified split.
        """
        return self._hf_dataset_builder._as_streaming_dataset_single(
            self._hf_split_generators[split.value]
        )


class HuggingfaceImageDataset(
    HuggingfaceDataset[T_HuggingfaceDatasetConfig, ImageInstance],
    Generic[T_HuggingfaceDatasetConfig],
):
    """
    AtriaImageDataset is a specialized dataset class for handling image datasets.
    It inherits from AtriaDataset and provides additional functionality specific to image data.
    """

    __abstract__ = True
    __data_model__ = ImageInstance


class HuggingfaceDocumentDataset(
    HuggingfaceDataset[T_HuggingfaceDatasetConfig, DocumentInstance],
    Generic[T_HuggingfaceDatasetConfig],
):
    """
    AtriaDocumentDataset is a specialized dataset class for handling document datasets.
    It inherits from AtriaDataset and provides additional functionality specific to document data.
    """

    __abstract__ = True
    __data_model__ = DocumentInstance
