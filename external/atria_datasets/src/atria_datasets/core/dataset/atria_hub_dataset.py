"""
Atria Hub Dataset Module

This module provides dataset classes for loading and managing datasets from the Atria Hub.
It includes support for both streaming and local dataset access, with specialized classes
for different data types (images and documents).

Classes:
    - AtriaHubDataset: Base class for datasets loaded from the Atria Hub
    - AtriaHubImageDataset: Specialized dataset for image data
    - AtriaHubDocumentDataset: Specialized dataset for document data

Dependencies:
    - atria_hub: Required for hub connectivity and dataset operations
    - omegaconf: For configuration management
    - pathlib: For file path operations
    - atria_core: For core utilities and types

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

from atria_logger import get_logger
from atria_types import (
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
    SplitConfig,
)

from atria_datasets.core.dataset._datasets import (
    Dataset,
    DatasetConfig,
    DatasetLoadingMode,
)
from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
from atria_datasets.core.storage.utilities import FileStorageType
from atria_datasets.core.typing.common import T_BaseDataInstance

if TYPE_CHECKING:
    from atria_hub.api.datasets import (
        Dataset as DatasetInfo,  # type: ignore[import-not-found]
    )
    from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

logger = get_logger(__name__)


class AtriaHubDatasetConfig(DatasetConfig):
    """
    Configuration class for Atria Hub datasets.

    This class extends AtriaDatasetConfig to include specific parameters for datasets
    hosted on the Atria Hub, such as the repository name and branch.
    """

    username: str
    branch: str = "main"


class AtriaHubDataset(Dataset[T_BaseDataInstance]):
    """
    A dataset class for loading and managing datasets from the Atria Hub.

    This class provides functionality to download, cache, and iterate through datasets
    stored in the Atria Hub. It supports both streaming and local access modes,
    and handles various data instance types through generic typing.

    Attributes:
        username (str): The username of the dataset owner
        dataset_name (str): The name of the dataset
        branch (str | None): The branch/version of the dataset
        streaming (bool): Whether to stream data or download locally

    Example:
        ```python
        # Load a dataset for local access
        dataset = AtriaHubDataset.load_from_hub(
            name="username/dataset_name", split=DatasetSplitType.TRAIN, streaming=False
        )

        # Load a dataset for streaming
        streaming_dataset = AtriaHubDataset.load_from_hub(
            name="username/dataset_name/branch", streaming=True
        )
        ```
    """

    __abstract__ = True
    __config__ = AtriaHubDatasetConfig

    def __init__(self, hub: AtriaHub | None = None, **kwargs) -> None:
        """
        Initialize the AtriaHubDataset.

        Args:
            username: The username of the dataset owner on Atria Hub
            dataset_name: The name of the dataset
            branch: The branch/version to use. If None, uses the default branch
            streaming: Whether to stream data directly from hub (True) or download locally (False)

        Raises:
            ImportError: If atria_hub package is not installed
            ValueError: If dataset configuration is invalid
        """
        super().__init__(**kwargs)

        # Initialize hub connection and dataset info
        self._hub = hub or self._initialize_hub()
        self._dataset_info = self._initialize_dataset_info()

    @classmethod
    def load_from_hub(
        cls,
        name: str,
        branch: str = "main",
        config_name: str | None = None,
        data_dir: str | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        dataset_load_mode: DatasetLoadingMode = DatasetLoadingMode.in_memory,
        overwrite_existing_cached: bool = False,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        shard_storage_type: FileStorageType | None = None,
        hub: AtriaHub | None = None,
        **sharded_storage_kwargs,
    ) -> Self:
        """
        Load a dataset from the Atria Hub.

        Args:
            name: Dataset name in format 'username/dataset_name' or 'username/dataset_name/branch'
            config_name: Configuration variant name (e.g., "main")
            preprocess_transform: Optional preprocessing function to apply to data
            access_token: Access token for private datasets
            overwrite_existing_shards: Whether to overwrite existing local shards
            allowed_keys: Set of keys to include in the data. If None, includes all keys
            shard_storage_type: Type of storage for sharded data
            overwrite_existing_cached: Whether to overwrite existing cached data
            sharded_storage_kwargs: Additional arguments for sharded storage
            streaming: Whether to stream data from hub or download locally

        Returns:
            AtriaHubDataset: Configured dataset instance ready for use

        Raises:
            ValueError: If dataset name format is invalid
            ImportError: If required packages are not available

        Example:
            ```python
            # Load training split of a public dataset
            dataset = AtriaHubDataset.load_from_hub(
                name="username/my_dataset", split=DatasetSplitType.TRAIN
            )

            # Load with streaming enabled
            streaming_dataset = AtriaHubDataset.load_from_hub(
                name="username/my_dataset/v1.0", streaming=True
            )
            ```
        """
        username, dataset_name = cls._validate_dataset_name(name)
        config_name = config_name or cls.__default_config_name__
        dataset = cls(
            username=username,
            dataset_name=dataset_name,
            branch=branch,
            config_name=config_name,
            hub=hub,
        )
        dataset.build(
            data_dir=data_dir,
            preprocess_transform=preprocess_transform,
            access_token=access_token,
            dataset_load_mode=dataset_load_mode,
            overwrite_existing_shards=overwrite_existing_shards,
            allowed_keys=allowed_keys,
            overwrite_existing_cached=overwrite_existing_cached,
            shard_storage_type=shard_storage_type,
            **sharded_storage_kwargs,
        )
        return cast(AtriaHubDataset[T_BaseDataInstance], dataset), dataset._dataset_info

    @classmethod
    def _validate_dataset_name(cls, name: str) -> tuple[str, str, str | None]:
        """
        Validate and parse dataset name format.

        Args:
            name: Dataset name in format 'username/dataset_name'

        Returns:
            tuple: (username, dataset_name, branch) where branch can be None

        Raises:
            ValueError: If dataset name format is invalid
        """
        if "/" not in name:
            raise ValueError(
                f"Invalid dataset name format: {name}. "
                "Expected format is 'username/dataset_name'."
            )

        parts = name.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            raise ValueError(
                f"Invalid dataset name format: {name}. "
                "Expected format is 'username/dataset_name'."
            )

    # ==================== Properties ====================

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        """
        Get the data model class for this dataset.

        Returns:
            type[T_BaseDataInstance]: The data model class (DocumentInstance or ImageInstance)

        Raises:
            ValueError: If the data instance type is not supported
        """
        from atriax_client.models.data_instance_type import DataInstanceType

        if self._dataset_info.data_instance_type == DataInstanceType.DOCUMENT_INSTANCE:
            return DocumentInstance
        elif self._dataset_info.data_instance_type == DataInstanceType.IMAGE_INSTANCE:
            return ImageInstance
        else:
            raise ValueError(
                f"Unsupported data instance type: {self._dataset_info.data_instance_type}. "
                "Supported types are 'document' and 'image'."
            )

    def _initialize_hub(self) -> AtriaHub:
        """
        Initialize the Atria Hub connection.

        Returns:
            AtriaHub: Initialized hub connection

        Raises:
            ImportError: If atria_hub package is not available
            Exception: If hub initialization fails
        """
        try:
            from atria_hub.hub import AtriaHub  # type: ignore[import-not-found]

            return AtriaHub().initialize()
        except ImportError:
            raise ImportError(
                "The 'atria_hub' package is required to load datasets from the hub. "
                "Please install it using 'uv add https://github.com/saifullah3396/atria_hub'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Atria Hub: {e}")
            raise e

    def _initialize_dataset_info(self) -> DatasetInfo:
        """
        Retrieve dataset information from the hub.

        Returns:
            DatasetInfo: Dataset metadata and information
        """

        if self.config.username is None:
            self.config.username = self._hub.auth.username
        self._dataset_info: DatasetInfo = self._hub.datasets.get_by_name(
            username=self.config.username, name=self.config.dataset_name
        )
        self.config.branch = self.config.branch or self._dataset_info.default_branch

        available_configs = self._hub.datasets.get_available_configs(
            self._dataset_info.repo_id, branch=self.config.branch
        )

        # available configs that start with given config name
        matching_configs = [
            cfg for cfg in available_configs if cfg.startswith(self.config.config_name)
        ]

        if not matching_configs:
            raise ValueError(
                f"No configurations found for dataset '{self.config.username}/{self._dataset_info.name}' "
                f"with config name '{self.config.config_name}' on branch '{self.config.branch}'. "
                f"Available configurations: {available_configs}"
            )

        if len(matching_configs) > 1:
            raise RuntimeError(
                f"Multiple configurations found for dataset '{self.config.username}/{self._dataset_info.name}' "
                f"with config name '{self.config.config_name}' on branch '{self.config.branch}': {matching_configs}. "
                f"Please specify the config name explicitly."
            )

        # Use the matched config if it differs from the original
        matched_config = matching_configs[0]
        if matched_config != self.config.config_name:
            logger.info(
                f"Using available configuration '{matched_config}' for dataset "
                f"'{self.config.username}/{self._dataset_info.name}' on branch '{self.config.branch}'."
            )
            self.config.config_name = matched_config
        return self._dataset_info

    # ==================== Path Management ====================

    def split_dir(self, storage_dir: str, split: DatasetSplitType) -> Path:
        """
        Get the directory path for a specific dataset split.

        Args:
            split: The dataset split type

        Returns:
            Path: Path to the split directory
        """
        return Path(storage_dir) / f"{self.config.config_name}/delta/{split.value}"

    def is_dataset_downloaded(self, storage_dir: str) -> bool:
        """
        Check if the dataset exists locally.

        Returns:
            bool: True if dataset exists locally, False otherwise
        """
        return (Path(storage_dir) / f"{self.config.config_name}/delta/").exists()

    def split_exists(self, storage_dir: str, split: DatasetSplitType) -> bool:
        """
        Check if a specific split exists locally.

        Args:
            split: The dataset split to check

        Returns:
            bool: True if split exists locally, False otherwise
        """
        return self.split_dir(storage_dir=storage_dir, split=split).exists()

    # ==================== Dataset Preparation ====================

    def prepare_downloads(self, data_dir: str, access_token: str | None = None) -> None:
        """
        Download dataset files from the hub to local storage.

        Args:
            data_dir: Directory where dataset files should be stored
            access_token: Optional access token for private datasets

        Note:
            This method is only effective when streaming=False. For streaming datasets,
            no downloads are performed.
        """
        if self._dataset_load_mode == DatasetLoadingMode.online_streaming:
            return

        if not self._downloads_prepared:
            if self.is_dataset_downloaded(storage_dir=self._storage_dir):
                if self._overwrite_existing_cached:
                    logger.warning(
                        f"Overwriting existing dataset '{self.config.dataset_name}' in '{self._storage_dir}'."
                    )
                else:
                    logger.info(
                        f"Dataset '{self.config.dataset_name}' already exists in '{self._storage_dir}'. Skipping download."
                    )
                    self._downloads_prepared = True
                    return
            logger.info(
                f"Downloading dataset '{self.config.dataset_name}' to '{self._storage_dir}' from repository '{self._dataset_info.name}'."
            )
            self._hub.datasets.download_files(
                dataset_repo_id=self._dataset_info.repo_id,
                branch=self.config.branch,
                config_dir=self.config.config_name,
                destination_path=str(self._storage_dir),
            )
            self._downloads_prepared = True

    def build(
        self,
        data_dir: str,
        runtime_transforms: Callable | None = None,
        preprocess_transform: Callable | None = None,
        access_token: str | None = None,
        dataset_load_mode: DatasetLoadingMode = DatasetLoadingMode.in_memory,
        overwrite_existing_cached: bool = False,
        overwrite_existing_shards: bool = False,
        allowed_keys: set[str] | None = None,
        **sharded_storage_kwargs,
    ) -> None:
        return super().build(
            data_dir=data_dir,
            runtime_transforms=runtime_transforms,
            preprocess_transform=preprocess_transform,
            access_token=access_token,
            dataset_load_mode=dataset_load_mode,
            overwrite_existing_cached=overwrite_existing_cached,
            overwrite_existing_shards=overwrite_existing_shards,
            allowed_keys=allowed_keys,
            enable_cached_splits=False,
            **sharded_storage_kwargs,
        )

    def _metadata(self) -> DatasetMetadata:
        """
        Get metadata for the dataset.

        This method should be implemented by subclasses to provide specific metadata
        for the dataset, such as description, version, and other relevant information.

        Returns:
            DatasetMetadata: Metadata object containing dataset information
        """

        metadata = self._hub.datasets.get_metadata(
            dataset_repo_id=self._dataset_info.repo_id, branch=self.config.branch
        )
        return DatasetMetadata(**metadata)

    def _available_splits(self) -> list[SplitConfig]:
        """
        Get configuration for all available splits.

        Args:
            data_dir: Data directory path

        Returns:
            list[SplitConfig]: List of split configurations
        """
        return self._hub.datasets.get_splits(
            self._dataset_info.repo_id, self.config.branch, self.config.config_name
        )

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Sequence | Generator[Any, None, None]:
        """
        Create an iterator for a specific dataset split.

        Args:
            split: The dataset split to iterate over
            allowed_keys: Set of keys to include in the data

        Returns:
            Sequence | Generator: Iterator over the split data

        Raises:
            SplitNotFoundError: If the split doesn't exist locally in non-streaming mode
        """
        if self._dataset_load_mode != DatasetLoadingMode.online_streaming:
            assert self.split_exists(storage_dir=self._storage_dir, split=split), (
                f"Dataset split {split.value} not found in {self._storage_dir}. "
                "Please ensure the dataset is downloaded or the split exists."
            )

        # Always include essential keys
        if self._allowed_keys is not None:
            self._allowed_keys = self._allowed_keys.copy()
            self._allowed_keys.update({"index", "sample_id"})

        if self._dataset_load_mode == DatasetLoadingMode.online_streaming:
            storage_options = self._hub.get_storage_options()
            path = self._hub.datasets.dataset_table_path(
                dataset_repo_id=self._dataset_info.repo_id,
                branch=self.config.branch,
                config_name=self.config.config_name,
                split=split.value,
            )
            return DeltalakeReader.from_mode(
                mode=self._dataset_load_mode,
                table_path=path,
                data_model=self.data_model,
                allowed_keys=self._allowed_keys,
                storage_options=storage_options,
            )
        else:
            return DeltalakeReader.from_mode(
                mode=self._dataset_load_mode,
                table_path=str(
                    self.split_dir(storage_dir=self._storage_dir, split=split)
                ),
                storage_dir=self._storage_dir,
                config_name=self.config.config_name,
                data_model=self.data_model,
                allowed_keys=self._allowed_keys,
            )


class AtriaHubImageDataset(AtriaHubDataset[ImageInstance]):
    """
    Specialized dataset class for handling image datasets from the Atria Hub.

    This class inherits from AtriaHubDataset and is specifically typed for ImageInstance
    data models, providing type safety and specialized functionality for image data.

    Example:
        ```python
        image_dataset = AtriaHubImageDataset.load_from_hub(
            name="username/image_dataset", split=DatasetSplitType.TRAIN
        )

        for image_instance in image_dataset:
            # image_instance is guaranteed to be of type ImageInstance
            process_image(image_instance.image)
        ```
    """

    __data_model__ = ImageInstance


class AtriaHubDocumentDataset(AtriaHubDataset[DocumentInstance]):
    """
    Specialized dataset class for handling document datasets from the Atria Hub.

    This class inherits from AtriaHubDataset and is specifically typed for DocumentInstance
    data models, providing type safety and specialized functionality for document data.

    Example:
        ```python
        doc_dataset = AtriaHubDocumentDataset.load_from_hub(
            name="username/document_dataset", split=DatasetSplitType.TRAIN
        )

        for doc_instance in doc_dataset:
            # doc_instance is guaranteed to be of type DocumentInstance
            process_document(doc_instance.text)
        ```
    """

    __data_model__ = DocumentInstance
