"""
Utilities Module for Dataset Storage

This module provides utility classes and functions for managing dataset storage. It includes
classes for handling file storage types, generating file URLs, and managing filenames for
Ray datasets. Additionally, it provides helper functions for data type conversions.

Classes:
    - FileStorageType: Enum for supported file storage types (e.g., Msgpack, WebDataset).
    - FileUrlProvider: Utility class for generating file paths and patterns for dataset storage.
    - RayDatasetFilenameProvider: Filename provider for Ray datasets.

Functions:
    - _convert_to_python_type: Converts NumPy types in a dictionary to native Python types.

Dependencies:
    - glob: For file pattern matching.
    - hashlib: For generating hashes.
    - os: For file system operations.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - numpy: For numerical operations.
    - pydantic.BaseModel: For data validation and serialization.
    - ray.data.block.Block: For Ray dataset blocks.
    - ray.data.datasource.FilenameProvider: For managing filenames in Ray datasets.
    - atria_logger: For logging utilities.
    - atria_core.utilities.imports: For importing base paths.
    - atria_datasets.core.datasets.metadata: For dataset metadata.
    - atria_datasets.core.datasets.splits: For dataset split information.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from atria_logger import get_logger

from atria_datasets.core.utilities import _get_package_base_path

if TYPE_CHECKING:
    from ray.data.block import Block

logger = get_logger(__name__)

_BASE_SRC_PATH = str(Path(_get_package_base_path("atria_datasets")).parent)
# _BASE_CORE_SRC_PATH = str(Path(_get_atria_core_base_path()).parent)
# _RAY_RUNTIME_ENV = {
#     "env_vars": {"PYTHONPATH": f"{_BASE_SRC_PATH}:{_BASE_CORE_SRC_PATH}/"}
# }
T = TypeVar("T")


class FileStorageType(str, Enum):
    """
    Enum for supported file storage types.

    Attributes:
        MSGPACK (str): Msgpack-based storage format.
        WEBDATASET (str): WebDataset-based storage format.
    """

    DELTALAKE = "DELTALAKE"
    MSGPACK = "MSGPACK"
    WEBDATASET = "WEBDATASET"

    def get_file_format(self) -> str:
        """
        Returns the file format extension for the storage type.

        Returns:
            str: The file format extension (e.g., "msgpack", "tar").

        Raises:
            ValueError: If the storage type is unsupported.
        """
        if self == FileStorageType.MSGPACK:
            return "msgpack"
        elif self == FileStorageType.WEBDATASET:
            return "tar"
        elif self == FileStorageType.DELTALAKE:
            return "parquet"
        else:
            raise ValueError(f"Unsupported file storage type: {self}")


class FileUrlProvider:
    """
    Utility class for generating file paths and patterns for dataset storage.

    This class provides methods for generating output directories, file patterns,
    and shard file paths based on dataset metadata and storage configurations.

    Attributes:
        _base_dir (Path): The base directory for storage.
        _file_key (Optional[str]): The key used for naming storage files.
        _storage_type (FileStorageType): The type of storage format.
        _use_hash (bool): Whether to use a hash for directory naming.
        _max_hash_length (int): The maximum length of the hash.
    """

    def __init__(
        self,
        base_dir: str,
        file_key: str | None = None,
        storage_type: FileStorageType = FileStorageType.MSGPACK,
        use_hash: bool = True,
        max_hash_length: int = 16,
    ) -> None:
        """
        Initializes the `FileUrlProvider`.

        Args:
            dir (str): The base directory for storage.
            file_key (Optional[str]): The key used for naming storage files. Defaults to None.
            storage_type (FileStorageType): The type of storage format. Defaults to Msgpack.
            use_hash (bool): Whether to use a hash for directory naming. Defaults to True.
            max_hash_length (int): The maximum length of the hash. Defaults to 16.
        """
        self._base_dir = Path(base_dir) / file_key if file_key else Path(base_dir)
        self._storage_type = storage_type
        self._use_hash = use_hash
        self._max_hash_length = max_hash_length

        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def split_info_path(self) -> Path:
        return self._base_dir / "split_info.json"

    def get_output_file_pattern(self, process: int) -> str:
        """
        Generates the file naming pattern for output files.

        Args:
            split (DatasetSplit): The dataset split (e.g., train, validation, test).
            process (int): The process index.

        Returns:
            str: The file naming pattern.
        """
        return str(
            self._base_dir
            / f"{process:06d}-%06d.{self._storage_type.get_file_format()}"
        )

    def get_shard_files(self) -> list[str]:
        """
        Retrieves the list of shard files for the dataset.

        Args:
            dataset_config (dict): The dataset configuration.
            split (DatasetSplit): The dataset split.

        Returns:
            List[str]: The list of shard file paths.
        """
        import glob

        return glob.glob(str(self._base_dir / self._storage_type.get_file_format()))


class RayDatasetFilenameProvider:
    """
    Filename provider for Ray datasets.

    This class generates filenames for Ray dataset blocks based on the dataset split,
    storage file key, and file format.

    Attributes:
        _split (str): The dataset split.
        _storage_file_key (str): The storage file key.
        _file_format (str): The file format.
    """

    def __init__(self, split: str, storage_file_key: str, file_format: str):
        """
        Initializes the `RayDatasetFilenameProvider`.

        Args:
            split (str): The dataset split.
            storage_file_key (str): The storage file key.
            file_format (str): The file format.
        """
        self._split = split
        self._file_format = file_format
        self._storage_file_key = storage_file_key

    def get_filename_for_block(
        self, block: "Block", task_index: int, block_index: int
    ) -> str:
        """
        Generates a filename for a Ray dataset block.

        Args:
            block (Block): The dataset block.
            task_index (int): The task index.
            block_index (int): The block index.

        Returns:
            str: The generated filename.
        """
        file_name = f"{self._storage_file_key}-" if self._storage_file_key else ""
        file_name += (
            f"{self._split}-{task_index:06d}-{block_index:06}.{self._file_format}"
        )
        return file_name


def _convert_to_python_type(sample: dict) -> dict:
    """
    Converts NumPy types in a dictionary to native Python types.

    Args:
        sample (dict): The dictionary containing NumPy types.

    Returns:
        dict: The dictionary with NumPy types converted to Python types.
    """
    import numpy as np

    for k, v in sample.items():
        if isinstance(v, np.ndarray | np.generic):
            sample[k] = v.item()
        elif isinstance(v, dict):
            sample[k] = _convert_to_python_type(v)
    return sample


def default_decoder(
    sample: dict[str, Any],
    format: bool | str | None = True,
    allowed_keys: set[str] | None = None,
) -> dict[str, Any]:
    """A default decoder for webdataset.

    This handles common file extensions: .txt, .cls, .cls2,
        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own decoder.

    Args:
        sample: sample, modified in place
    """
    import gzip
    import io

    import numpy as np

    decoded_sample = {}
    for key, stream in sample.items():
        if key.startswith("__"):
            continue
        extensions = key.split(".")
        if len(extensions) == 2:
            key = extensions[0]
        elif len(extensions) == 3:
            key = extensions[1]
        if len(extensions) < 1:
            continue
        if allowed_keys is not None and key not in allowed_keys:
            continue
        extension = extensions[-1]
        if extension in ["gz"]:
            decompressed = gzip.decompress(stream.read())
            stream = io.BytesIO(decompressed)
            if len(extensions) < 2:
                decoded_sample[key] = stream
                continue
            extension = extensions[-2]
        elif extension in ["txt", "text"]:
            if isinstance(stream, bytes):
                stream = io.BytesIO(stream)
            value = stream.read()
            decoded_sample[key] = value.decode("utf-8")
        elif extension in ["cls", "cls2"]:
            if isinstance(stream, bytes):
                stream = io.BytesIO(stream)
            value = stream.read()
            decoded_sample[key] = int(value.decode("utf-8"))  # type: ignore[assignment]
        elif extension in ["jpg", "png", "ppm", "pgm", "pbm", "pnm"]:
            if format == "PIL":
                import PIL.Image

                decoded_sample[key] = PIL.Image.open(stream)
            elif format == "numpy":
                import numpy as np

                decoded_sample[key] = np.asarray(PIL.Image.open(stream))  # type: ignore[assignment]
            else:
                raise ValueError(f"Unknown format: {format}")
        elif extension == "json":
            import json

            value = stream.read()
            decoded_sample[key] = json.loads(value)
        elif extension == "npy":
            import numpy as np

            decoded_sample[key] = np.load(stream)
        elif extension == "mp":
            import msgpack

            if isinstance(stream, bytes):
                stream = io.BytesIO(stream)

            value = stream.read()
            decoded_sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ["pt", "pth"]:
            import torch  # type: ignore[import-not-found]

            decoded_sample[key] = torch.load(stream)
        elif extension in ["pickle", "pkl"]:
            import pickle

            decoded_sample[key] = pickle.load(stream)
    return decoded_sample


def batch_iterator(iterator: Iterator[T], batch_size: int) -> Iterator[list[T]]:
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
