# ruff: noqa

from typing import TYPE_CHECKING

import lazy_loader as lazy

# Ensure registry is initialized immediately
import atria_datasets.registry  # noqa: F401

if TYPE_CHECKING:
    from atria_datasets.api.datasets import load_dataset, load_dataset_config  # noqa
    import atria_datasets.registry  # noqa: F401 # Import the registry to ensure it is initialized
    from atria_datasets.core.dataset._datasets import (
        Dataset,
        DocumentDataset,
        ImageDataset,
    )

    # from atria_datasets.core.dataset.atria_hub_dataset import AtriaHubDataset
    from atria_datasets.core.dataset._hf_datasets import (
        HuggingfaceDataset,
        HuggingfaceDatasetConfig,
        HuggingfaceDocumentDataset,
        HuggingfaceImageDataset,
    )
    from atria_datasets.core.dataset._split_iterators import SplitIterator
    from atria_datasets.core.dataset_splitters.standard_splitter import StandardSplitter
    from atria_datasets.core.download_manager.download_file_info import DownloadFileInfo
    from atria_datasets.core.download_manager.download_manager import DownloadManager
    from atria_datasets.core.download_manager.file_downloader import (
        FileDownloader,
        FTPFileDownloader,
        GoogleDriveDownloader,
        HTTPDownloader,
    )
    from atria_datasets.core.storage.deltalake_reader import DeltalakeReader
    from atria_datasets.core.storage.deltalake_storage_manager import (
        DeltalakeStorageManager,
    )
    from atria_datasets.core.storage.msgpack_shard_writer import (
        MsgpackFileWriter,
        MsgpackShardWriter,
    )
    from atria_datasets.core.storage.utilities import FileStorageType
    from atria_datasets.registry import DATASET, DATA_PIPELINE, BATCH_SAMPLER


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "api.datasets": ["load_dataset", "load_dataset_config"],
        "core.dataset._datasets": ["Dataset", "DocumentDataset", "ImageDataset"],
        "core.dataset._hf_datasets": [
            "HuggingfaceDataset",
            "HuggingfaceDatasetConfig",
            "HuggingfaceDocumentDataset",
            "HuggingfaceImageDataset",
        ],
        "core.dataset._split_iterators": ["SplitIterator"],
        "core.dataset_splitters.standard_splitter": ["StandardSplitter"],
        "core.download_manager.download_file_info": ["DownloadFileInfo"],
        "core.download_manager.download_manager": ["DownloadManager"],
        "core.download_manager.file_downloader": [
            "FileDownloader",
            "FTPFileDownloader",
            "GoogleDriveDownloader",
            "HTTPDownloader",
        ],
        "core.storage.deltalake_reader": ["DeltalakeReader"],
        "core.storage.deltalake_storage_manager": ["DeltalakeStorageManager"],
        "core.storage.msgpack_shard_writer": [
            "MsgpackFileWriter",
            "MsgpackShardWriter",
        ],
        "core.storage.utilities": ["FileStorageType"],
        "registry": ["DATASET", "DATA_PIPELINE", "BATCH_SAMPLER"],
    },
)
