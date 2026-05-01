"""Constants Module

This module defines default constants used throughout the Atria Datasets package,
including default cache directories and download paths.
"""

import os
from pathlib import Path

DEFAULT_ATRIA_CACHE_DIR = os.environ.get(
    "DEFAULT_ATRIA_CACHE_DIR", Path.home() / ".cache/atria/"
)
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_DATASETS_CACHE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "datasets/"
"""
str: The default directory for caching Atria resources. This value can be overridden
by setting the `ATRIA_CACHE_DIR` environment variable. If not set, it defaults to
`~/.cache/atria/`.
"""

_DEFAULT_ATRIA_DATASETS_STORAGE_SUBDIR = "storage"
"""
str: The default subdirectory name for storing dataset-related resources within
the Atria datasets cache directory.
"""

_DEFAULT_ATRIA_DATASETS_CONFIG_PATH = "config.yaml"
"""
Path: The default path template for dataset configuration files. The
`{config_name}` placeholder should be replaced with the actual configuration name
when constructing the full path to a specific dataset configuration file.
"""

_DEFAULT_ATRIA_DATASETS_METADATA_PATH = "metadata.yaml"
"""
Path: The default path for dataset metadata files within the dataset storage
directory.
"""

_DEFAULT_ATRIA_FILE_STORAGE_DIR = Path(DEFAULT_ATRIA_CACHE_DIR) / "fs/"
"""
Path: The default directory for Atria file storage. This is a subdirectory of
`_DEFAULT_ATRIA_CACHE_DIR` and is used to store file system-related resources.
"""

_DEFAULT_DOWNLOAD_PATH = ".download_cache"
"""
Path: The default download path for temporary storage of downloaded files. This
directory is used to store files temporarily during download operations.
"""
