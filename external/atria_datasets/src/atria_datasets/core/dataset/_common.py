"""Common Dataset Types and Enums"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import yaml
from atria_logger import get_logger
from atria_registry._module_base import ModuleConfig
from atria_types import BaseDataInstance
from atria_types._common import DatasetSplitType
from pydantic import ConfigDict

from atria_datasets.core.constants import (
    _DEFAULT_ATRIA_DATASETS_CONFIG_PATH,
    _DEFAULT_ATRIA_DATASETS_METADATA_PATH,
)
from atria_datasets.core.storage.utilities import FileStorageType

if TYPE_CHECKING:
    from atria_datasets.core.dataset._datasets import Dataset

logger = get_logger(__name__)


class DatasetConfig(ModuleConfig):
    model_config = ConfigDict(extra="forbid")
    dataset_name: str
    config_name: str = "default"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None

    def build(
        self,
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
        return super().build(
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
            **kwargs,
        )


class HuggingfaceDatasetConfig(DatasetConfig):
    hf_repo: str
    hf_config_name: str


class DatasetLoadingMode(str, enum.Enum):
    """
    Enum to represent the streaming mode of the dataset.

    Attributes:
        LOCAL: Dataset is downloaded and stored locally.
        STREAMING: Dataset is streamed directly from the Atria Hub.
    """

    in_memory = "in_memory"
    local_streaming = "local_streaming"
    online_streaming = "online_streaming"


def _save_dataset_info(
    storage_dir: str, config_name: str, config: dict, metadata: dict
) -> None:
    """
    Save dataset configuration and metadata to files.

    Creates YAML files containing:
    - Dataset configuration (config.yaml)
    - Dataset metadata (metadata.yaml)
    """

    def write_yaml_file(file_path: Path, data: dict) -> None:
        """Write data to YAML file, creating directories as needed."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    # Save configuration
    config_file_path = (
        Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_CONFIG_PATH
    )
    logger.info("Saving dataset configuration to %s", config_file_path)
    write_yaml_file(config_file_path, config)

    metadata_file_path = (
        Path(storage_dir) / config_name / _DEFAULT_ATRIA_DATASETS_METADATA_PATH
    )
    logger.info("Saving dataset metadata to %s", metadata_file_path)
    write_yaml_file(metadata_file_path, metadata)


def _get_storage_manager(
    cached_storage_type: FileStorageType,
    storage_dir: str,
    config_name: str,
    num_processes: int,
):
    if cached_storage_type == FileStorageType.DELTALAKE:
        from atria_datasets.core.storage.deltalake_storage_manager import (
            DeltalakeStorageManager,
        )

        return DeltalakeStorageManager(
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
        )
    elif cached_storage_type == FileStorageType.MSGPACK:
        from atria_datasets.core.storage.msgpack_storage_manager import (
            MsgpackStorageManager,
        )

        return MsgpackStorageManager(
            storage_dir=storage_dir,
            config_name=config_name,
            num_processes=num_processes,
        )
    else:
        raise ValueError(f"Unsupported storage type: {cached_storage_type}")


T_DatasetConfig = TypeVar("T_DatasetConfig", bound=DatasetConfig)
T_HuggingfaceDatasetConfig = TypeVar(
    "T_HuggingfaceDatasetConfig", bound=HuggingfaceDatasetConfig
)
T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)
