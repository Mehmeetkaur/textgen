"""Defines the base Dataset class for Atria datasets."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

from atria_logger import get_logger
from atria_registry import ConfigurableModule
from atria_types import (
    BaseDataInstance,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    ImageInstance,
)

from atria_datasets.core.dataset._common import (
    DatasetConfig,
    T_BaseDataInstance,
    T_DatasetConfig,
)
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.core.dataset._split_iterators import SplitIterator
from atria_datasets.core.storage.utilities import FileStorageType

logger = get_logger(__name__)

if TYPE_CHECKING:
    from atria_datasets.core.dataset._dataset_builders import DatasetBuilder


class Dataset(
    ConfigurableModule[T_DatasetConfig], Generic[T_DatasetConfig, T_BaseDataInstance]
):
    """
    Generic base class for datasets in the Atria application.

    This class provides a comprehensive framework for managing datasets with support for:
    - Multiple data splits (train/validation/test)
    - Flexible storage backends (DeltaLake, sharded files)
    - Download management for remote datasets
    - Runtime and preprocessing transformations
    - Dataset versioning and configuration management
    - Hub integration for dataset sharing

    Type Parameters:
        T_BaseDataInstance: The type of data instances this dataset contains
            (must inherit from BaseDataInstance)

    Attributes:
        __data_model__: The data model class used for type validation
        __default_config_path__: Default path for dataset configuration files
        __default_metadata_path__: Default path for dataset metadata files
        __repr_fields__: Fields included in string representation

    Example:
        ```python
        # Create a custom dataset
        class MyDataset(AtriaDataset[DocumentInstance]):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [SplitConfig(split=DatasetSplitType.train, gen_kwargs={})]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Return iterator for the split
                pass


        # Load and use dataset
        dataset = MyDataset(dataset_name="my_dataset")
        dataset.build_split(DatasetSplitType.train)
        for sample in dataset.train:
            print(sample)
        ```
    """

    __abstract__ = True
    __requires_access_token__ = False
    __extract_downloads__ = True
    __data_model__: type[T_BaseDataInstance]
    __repr_fields__ = {"data_model", "data_dir", "split_iterators"}
    __config__: type[T_DatasetConfig] = DatasetConfig

    def __init__(
        self,
        config: T_DatasetConfig,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        overwrite_existing_cached: bool = False,
        allowed_keys: set[str] | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        enable_cached_splits: bool = True,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
    ) -> None:
        super().__init__(config=config)

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

    def _prepare_dataset_builder(
        self,
        data_dir: str | None = None,
        split: DatasetSplitType | None = None,
        access_token: str | None = None,
        overwrite_existing_cached: bool = False,
        allowed_keys: set[str] | None = None,
        num_processes: int = 8,
        cached_storage_type: FileStorageType = FileStorageType.MSGPACK,
        enable_cached_splits: bool = True,
        store_artifact_content: bool = True,
        max_cache_image_size: int | None = None,
    ) -> DatasetBuilder:
        from atria_datasets.core.dataset._dataset_builders import (
            CachedDatasetBuilder,
            DatasetBuilder,
        )

        kwargs = {
            "data_dir": data_dir,
            "split": split,
            "access_token": access_token,
            "allowed_keys": allowed_keys,
        }
        if enable_cached_splits:
            kwargs.update(
                {
                    "cached_storage_type": cached_storage_type,
                    "store_artifact_content": store_artifact_content,
                    "max_cache_image_size": max_cache_image_size,
                    "overwrite_existing_cached": overwrite_existing_cached,
                    "num_processes": num_processes,
                }
            )

        dataset_builder = (
            DatasetBuilder(dataset=self, **kwargs)
            if not enable_cached_splits
            else CachedDatasetBuilder(dataset=self, **kwargs)
        )
        return dataset_builder

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "__abstract__" in cls.__dict__ and cls.__dict__["__abstract__"]:
            return

        data_model = cls.__data_model__
        if data_model is None:
            raise TypeError(
                f"Class '{cls.__name__}' must define a __data_model__ attribute "
                "to specify the type of data instances."
            )
        if not issubclass(data_model, BaseDataInstance):
            raise TypeError(
                f"Class '{cls.__name__}.__data_model__' must be a type, "
                f"got {type(data_model).__name__}: {data_model}"
            )
        assert isinstance(cls.__requires_access_token__, bool), (
            f"Class '{cls.__name__}' must define __requires_access_token__ as a boolean."
        )
        assert isinstance(cls.__extract_downloads__, bool), (
            f"Class '{cls.__name__}' must define __extract_downloads__ as a boolean."
        )

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata containing description, version, and other information."""
        return self._metadata()

    @property
    def data_model(self) -> type[T_BaseDataInstance]:
        """The data model class used for type validation and instantiation."""
        return self.__data_model__

    @property
    def downloaded_files(self) -> dict[str, Path]:
        """Dictionary of downloaded file paths."""
        return self._dataset_builder._downloaded_files

    @property
    def access_token(self) -> str | None:
        """Access token used for downloading private datasets."""
        return self._dataset_builder._access_token

    @property
    def train(self) -> SplitIterator[T_BaseDataInstance]:
        """Training split iterator. Returns None if training split is not available."""
        if DatasetSplitType.train not in self._split_iterators:
            raise SplitNotFoundError("Training split iterator is not available. ")
        return self._split_iterators[DatasetSplitType.train]

    @train.setter
    def train(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the training split iterator."""
        self._split_iterators[DatasetSplitType.train] = value

    @property
    def validation(self) -> SplitIterator[T_BaseDataInstance]:
        """Validation split iterator. Returns None if validation split is not available."""
        if DatasetSplitType.validation not in self._split_iterators:
            raise SplitNotFoundError("Validation split iterator is not available. ")
        return self._split_iterators[DatasetSplitType.validation]

    @validation.setter
    def validation(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the validation split iterator."""
        self._split_iterators[DatasetSplitType.validation] = value

    @property
    def test(self) -> SplitIterator[T_BaseDataInstance]:
        """Test split iterator. Returns None if test split is not available."""
        if DatasetSplitType.test not in self._split_iterators:
            raise SplitNotFoundError("Test split iterator is not available. ")
        return self._split_iterators[DatasetSplitType.test]

    @test.setter
    def test(self, value: SplitIterator[T_BaseDataInstance]) -> None:
        """Set the test split iterator."""
        self._split_iterators[DatasetSplitType.test] = value

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, SplitIterator[T_BaseDataInstance]]:
        """Get all split iterators as a dictionary."""
        return self._split_iterators

    def _input_transform(self, sample: Any) -> T_BaseDataInstance:
        """
        Transform raw sample data into the dataset's data model.

        Args:
            sample: Raw sample data (dict, data model instance, or other format)

        Returns:
            Transformed sample as data model instance

        Raises:
            TypeError: If sample cannot be converted to the data model
        """
        if isinstance(sample, self.data_model):
            return sample
        elif isinstance(sample, dict):
            return self.data_model(**sample)
        else:
            raise TypeError(
                f"Cannot convert sample of type {type(sample)} to data model {self.data_model}"
            )

    def _download_urls(self) -> dict[str, tuple[str, str]] | list[str]:
        """
        Get the list of URLs for downloading dataset files.

        This method should be overridden by subclasses to provide specific URLs
        for the dataset being implemented.

        Returns:
            List of URLs as strings
        """
        return []

    def _custom_download(
        self, data_dir: str, access_token: str | None = None
    ) -> dict[str, Path]:
        """This method can be overridden by subclasses to implement custom download logic.

        Args:
            data_dir: Directory to save downloaded files
            access_token: Authentication token for private resources

        Returns:
            Dictionary mapping download keys to downloaded file paths
        """
        raise NotImplementedError(
            "Subclasses must implement the `_custom_download` method to handle "
            "specific download logic."
        )

    @abstractmethod
    def _metadata(self) -> DatasetMetadata:
        """
        Create and return dataset metadata.

        Subclasses should override this method to provide specific metadata
        including description, version, license, and other relevant information.

        Returns:
            DatasetMetadata object with dataset information
        """
        raise NotImplementedError("Subclasses must implement the `_metadata` method.")

    @abstractmethod
    def _available_splits(self) -> list[DatasetSplitType]:
        """
        List available dataset splits.

        Subclasses should override this method to return the splits that are
        available for the dataset (e.g., train, validation, test).

        Returns:
            List of DatasetSplitType values representing available splits
        """
        raise NotImplementedError(
            "Subclasses must implement the `_available_splits` method."
        )

    @abstractmethod
    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable:
        """
        Create an iterator for a specific dataset split.

        Args:
            split: The dataset split to create iterator for
            **kwargs: Additional arguments from split configuration

        Returns:
            Iterator or generator yielding data samples for the split

        Note:
            Subclasses must implement this method to define how to iterate
            over the data for each split. The iterator should yield raw data
            that will be transformed by _input_transform.
        """
        raise NotImplementedError(
            "Subclasses must implement the `_split_iterator` method to provide "
            "an iterator for the specified dataset split."
        )


class ImageDataset(Dataset[T_DatasetConfig, ImageInstance], Generic[T_DatasetConfig]):
    """
    Specialized dataset class for handling image datasets.

    This class inherits from AtriaDataset and is specifically typed for ImageInstance
    data models, providing type safety and specialized functionality for image data.

    The class automatically handles:
    - Image-specific data validation
    - Proper type hints for image data
    - Integration with image processing pipelines

    Example:
        ```python
        class CustomImageDataset(AtriaImageDataset):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [
                    SplitConfig(
                        split=DatasetSplitType.train,
                        gen_kwargs={"image_dir": f"{data_dir}/train"},
                    )
                ]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Yield image data samples
                pass
        ```
    """

    __abstract__: bool = True
    __data_model__ = ImageInstance


class DocumentDataset(
    Dataset[T_DatasetConfig, DocumentInstance], Generic[T_DatasetConfig]
):
    """
    Specialized dataset class for handling document datasets.

    This class inherits from AtriaDataset and is specifically typed for DocumentInstance
    data models, providing type safety and specialized functionality for document data.

    The class automatically handles:
    - Document-specific data validation
    - Proper type hints for document data
    - Integration with text processing pipelines

    Example:
        ```python
        class CustomDocumentDataset(AtriaDocumentDataset):
            def _split_configs(self, data_dir: str) -> list[SplitConfig]:
                return [
                    SplitConfig(
                        split=DatasetSplitType.train,
                        gen_kwargs={"text_dir": f"{data_dir}/train"},
                    )
                ]

            def _split_iterator(self, split: DatasetSplitType, **kwargs):
                # Yield document data samples
                pass
        ```
    """

    __abstract__: bool = True
    __data_model__ = DocumentInstance
