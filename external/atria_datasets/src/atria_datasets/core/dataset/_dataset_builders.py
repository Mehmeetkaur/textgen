"""Dataset Builder Module"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from atria_logger import get_logger
from atria_types import DatasetSplitType
from atria_types._data_instance._base import BaseDataInstance
from atria_types._data_instance._document_instance import DocumentInstance
from atria_types._data_instance._image_instance import ImageInstance

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CACHE_DIR,
    _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR,
    _DEFAULT_DOWNLOAD_PATH,
)
from atria_datasets.core.dataset._common import (
    DatasetConfig,
    _get_storage_manager,
    _save_dataset_info,
)
from atria_datasets.core.dataset._datasets import Dataset
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class DefaultOutputTransformer:
    def __init__(
        self,
        data_dir: str,
        store_artifact_content: bool = True,
        resize_images: bool = False,
        image_max_size: int = 1024,
    ):
        self._data_dir = data_dir
        self._store_artifact_content = store_artifact_content
        self._resize_images = resize_images
        self._image_max_size = image_max_size

    def __call__(self, sample: BaseDataInstance) -> BaseDataInstance:
        if self._store_artifact_content:
            sample = sample.load()
        if (
            self._resize_images
            and isinstance(sample, (ImageInstance, DocumentInstance))
            and sample.image is not None
        ):
            image = sample.image.ops.resize_with_aspect_ratio(
                max_size=self._image_max_size
            )
            sample = sample.update(image=image)

        return sample.ops.convert_file_paths_to_relative(parent_dir=self._data_dir)


class DatasetBuilder:
    def __init__(
        self,
        dataset: Dataset,
        data_dir: str,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        allowed_keys: set[str] | None = None,
        split_iterator_type: type[SplitIterator] = SplitIterator,
    ):
        self._dataset = dataset
        self._downloads_prepared = False
        if data_dir is None:
            logger.warning(
                f"No 'data_dir' provided. Using default data dir directory: {self.default_data_dir}"
            )
            data_dir = self.default_data_dir
        self._data_dir = self._validate_data_dir(data_dir)
        self._split = split
        self._access_token = access_token
        self._allowed_keys = allowed_keys
        self._storage_dir = (
            Path(self.default_data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        )
        self._split_iterator_type = split_iterator_type

    @property
    def dataset(self) -> Dataset:
        """Get the dataset being built."""
        return self._dataset

    @property
    def dataset_name(self) -> str:
        """Get the name of the dataset being built."""
        return (
            self.dataset_config.dataset_name
            if self.dataset_config.dataset_name
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
    def default_data_dir(self) -> Path:
        """Get the data directory of the dataset being built."""
        return _DEFAULT_ATRIA_DATASETS_CACHE_DIR / self.dataset_name

    @property
    def data_model(self):
        """Get the data model of the dataset being built."""
        return self._dataset.data_model

    @property
    def max_split_samples(self) -> dict[DatasetSplitType, int | None]:
        """
        Get the maximum number of samples allowed for a specific split.

        Args:
            split: The dataset split to check

        Returns:
            Maximum number of samples, or None if no limit is set
        """
        limits = {
            DatasetSplitType.train: self.dataset_config.max_train_samples,
            DatasetSplitType.validation: self.dataset_config.max_validation_samples,
            DatasetSplitType.test: self.dataset_config.max_test_samples,
        }
        return limits

    def prepare_splits(self) -> dict[DatasetSplitType, SplitIterator]:
        """Prepare splits without caching (direct iteration)."""
        split_iterators: dict[DatasetSplitType, SplitIterator] = {}
        for split in self.dataset._available_splits():
            if self._split is not None and split != self._split:
                continue
            self._prepare_downloads(
                data_dir=self._data_dir, access_token=self._access_token
            )
            split_iterators[split] = self._prepare_split(split)
        return split_iterators

    def _validate_data_dir(self, data_dir: str | Path) -> Path:
        """
        Validate and create data directory if needed.

        Args:
            data_dir: Directory path to validate

        Returns:
            Validated Path object

        Raises:
            AssertionError: If path exists but is not a directory
        """
        data_dir = Path(data_dir)

        if data_dir.exists():
            assert data_dir.is_dir(), (
                f"Data directory `{data_dir.absolute()}` exists but is not a directory."
            )
        else:
            logger.warning(
                f"Data directory `{data_dir.absolute()}` does not exist. Creating it."
            )
            data_dir.mkdir(parents=True, exist_ok=True)

        return str(data_dir)

    def _prepare_downloads(
        self, data_dir: str, access_token: str | None = None
    ) -> None:
        """
        Download and prepare remote dataset files.

        Args:
            data_dir: Directory to download files to
            access_token: Authentication token for private resources

        Note:
            This method is idempotent - subsequent calls will not re-download files.
        """
        if not self._downloads_prepared:
            if self.dataset.__requires_access_token__:
                if access_token is not None:
                    self._access_token = access_token
                else:
                    logger.warning(
                        "access_token must be passed to download this dataset. "
                        f"See `{self.dataset.metadata.homepage}` for instructions to get the access token"
                    )

            if self.dataset._custom_download.__func__ is not Dataset._custom_download:
                self._downloaded_files = self.dataset._custom_download(
                    data_dir, access_token
                )
            else:
                from atria_datasets.core.download_manager.download_manager import (
                    DownloadManager,
                )

                download_dir = Path(data_dir) / _DEFAULT_DOWNLOAD_PATH
                download_dir.mkdir(parents=True, exist_ok=True)

                download_manager = DownloadManager(
                    data_dir=Path(data_dir), download_dir=download_dir
                )

                download_urls = self.dataset._download_urls()
                if len(download_urls) > 0:
                    self._downloaded_files = download_manager.download_and_extract(
                        download_urls,
                        extract=self.dataset.__extract_downloads__,
                        access_token=access_token,
                    )
                    logger.info(f"Downloaded files {self._downloaded_files}")

            self._downloads_prepared = True

    def _prepare_split(
        self,
        split: DatasetSplitType,
        store_artifact_content: bool = True,
        resize_images: bool = False,
        image_max_size: int = 1024,
    ) -> SplitIterator:
        return self._split_iterator_type(
            split=split,
            data_model=self.data_model,
            input_transform=self.dataset._input_transform,
            base_iterator=self.dataset._split_iterator(split, self._data_dir),
            max_len=self.max_split_samples[split],
            output_transform=DefaultOutputTransformer(
                data_dir=self._data_dir,
                store_artifact_content=store_artifact_content,
                resize_images=resize_images,
                image_max_size=image_max_size,
            ),
        )


class CachedDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        dataset: Dataset,
        data_dir: str,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        allowed_keys: set[str] | None = None,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        enable_cached_splits: bool = True,
        overwrite_existing_cached: bool = False,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
        num_processes: int = 8,
        split_iterator_type: type[SplitIterator] = SplitIterator,
    ):
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
            split=split,
            access_token=access_token,
            allowed_keys=allowed_keys,
            split_iterator_type=split_iterator_type,
        )
        self._cached_storage_type = cached_storage_type
        self._enable_cached_splits = enable_cached_splits
        self._overwrite_existing_cached = overwrite_existing_cached
        self._store_artifact_content = store_artifact_content
        self._max_cache_image_size = max_cache_image_size
        self._num_processes = num_processes
        self._storage_dir = (
            Path(self._data_dir) / _DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR
        )

    def prepare_splits(self) -> dict[DatasetSplitType, SplitIterator]:
        """Prepare cached splits using DeltaLake / Msgpack storage."""
        unique_config_name = (
            self.dataset_config.config_name + "-" + self.dataset_config.hash
        )
        storage_manager = _get_storage_manager(
            self._cached_storage_type,
            storage_dir=str(self._storage_dir),
            config_name=unique_config_name,
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
                self._prepare_downloads(
                    data_dir=str(self._data_dir), access_token=self._access_token
                )
                logger.info(
                    f"Caching split [{split.value}] to {self._storage_dir} with max_len={self.max_split_samples[split]}"
                )
                split_iterator = self._prepare_split(
                    split=split,
                    store_artifact_content=self._store_artifact_content,
                    resize_images=self._max_cache_image_size is not None,
                    image_max_size=self._max_cache_image_size,
                )
                storage_manager.write_split(split_iterator=split_iterator)
                if not info_saved:
                    _save_dataset_info(
                        self._storage_dir,
                        unique_config_name,
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
                unique_config_name,
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
