"""
Download File Info Module

This module defines the `DownloadFileInfo` class, which provides utilities for managing
and processing information about files to be downloaded. It includes functionality for
parsing URLs, generating hashed file names, determining file paths, and checking file
statuses (e.g., whether a file is compressed or download is complete).

Classes:
    - DownloadFileInfo: Represents information about a file to be downloaded.

Dependencies:
    - hashlib: For generating SHA-256 hashes of URLs.
    - re: For regular expression operations.
    - pathlib.Path: For handling file paths.
    - urllib.parse: For parsing URLs.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import hashlib
import re
from pathlib import Path
from urllib.parse import ParseResult, urlparse, urlunparse

from atria_types._utilities._repr import RepresentationMixin

_SUPPORTED_URLS = ["http", "https", "ftp"]
_COMPRESSED_FILES_REGEX = r"\.(zip|tar|tar\.gz|tgz)(\..+)?$"


class DownloadFileInfo(RepresentationMixin):
    """
    Represents information about a file to be downloaded.

    This class provides utilities for managing and processing information about files
    to be downloaded, including URL parsing, hashed file names, file paths, and file
    status checks.

    Attributes:
        url (str): The URL of the file to be downloaded.
        output_file_name (str): The name of the output file.
        data_dir (Path): The directory where the downloaded file will be stored.
        download_dir (Path): The directory where temporary download files will be stored.
        url_ext (str): Indicates if the file is an archive.
    """

    def __init__(
        self,
        url: str,
        rel_output_file_path: str,
        data_dir: Path,
        download_dir: Path,
        url_ext: str | None = None,
    ):
        """
        Initializes the `DownloadFileInfo` object.

        Args:
            url (str): The URL of the file to be downloaded.
            output_file_name (str): The name of the output file.
            data_dir (Path): The directory where the downloaded file will be stored.
            download_dir (Path): The directory where temporary download files will be stored.
            url_ext (str): Indicates if the file is an archive.

        Raises:
            ValueError: If the URL is invalid or the URL scheme is not supported.
        """
        self.url = url
        self.rel_output_file_path = rel_output_file_path
        self.data_dir = Path(data_dir)
        self.download_dir = Path(download_dir)
        self.url_ext = url_ext
        parsed = urlparse(self.url)
        if parsed.scheme == "":
            raise ValueError(
                f"URL {url} is invalid. URL must have a scheme (http, https, ftp)."
            )
        if parsed.scheme not in _SUPPORTED_URLS:
            raise ValueError(
                f"URL {url} is not supported. Supported URL schemes are: {', '.join(_SUPPORTED_URLS)}"
            )
        download_dir.mkdir(parents=True, exist_ok=True)

    def update_extract_path(self):
        if self.is_compressed:
            match = re.search(_COMPRESSED_FILES_REGEX, self.rel_output_file_path)
            if match:
                self.rel_output_file_path = self.rel_output_file_path.replace(
                    match.group(), ""
                )

    @property
    def parsed_url(self) -> ParseResult:
        """
        Parses the URL into its components.

        Returns:
            ParseResult: The parsed URL components.
        """
        return urlparse(self.url)

    @property
    def hashed_url(self) -> str:
        """
        Generates a SHA-256 hash of the URL.

        Returns:
            str: The hashed URL as a hexadecimal string.
        """
        return hashlib.sha256(self.url.encode()).hexdigest()

    @property
    def hashed_url_without_part(self) -> str:
        """
        Generates a SHA-256 hash of the URL without the part file extension.

        Returns:
            str: The hashed URL as a hexadecimal string.
        """
        url_without_part = urlunparse(
            (
                self.parsed_url.scheme,
                self.parsed_url.netloc,
                str(Path(self.parsed_url.path).with_suffix("")),
                self.parsed_url.params,
                self.parsed_url.query,
                self.parsed_url.fragment,
            )
        )
        return hashlib.sha256(url_without_part.encode()).hexdigest()

    @property
    def url_path_ext(self) -> str:
        """
        Extracts the file extension(s) from the URL path.

        Returns:
            str: The file extension(s) from the URL path.
        """
        return (
            "".join(Path(self.parsed_url.path).suffixes)
            if self.url_ext is None
            else self.url_ext
        )

    @property
    def download_path(self) -> Path:
        """
        Determines the path where the file will be downloaded.

        Returns:
            Path: The download path for the file.
        """
        return self.download_dir / (self.hashed_url + self.url_path_ext)

    @property
    def extractable_path(self) -> Path:
        """
        Determines the path for the extracted file.

        If the file is a part file (e.g., `.zip.001`), the path is adjusted accordingly.

        Returns:
            Path: The path for the extracted file.
        """
        if self.is_part_file:
            return self.download_dir / (
                self.hashed_url_without_part + Path(self.parsed_url.path).suffixes[-2]
            )
        else:
            return self.download_path

    @property
    def extracted_path(self) -> Path:
        """
        Determines the directory where the file will be extracted.

        Returns:
            Path: The path to the extracted directory.
        """
        if self.is_part_file:
            return self.data_dir / (self.hashed_url_without_part)
        else:
            return self.data_dir / self.hashed_url

    @property
    def is_part_file(self) -> bool:
        """
        Checks if the file is a part of a multi-part archive (e.g., `.zip.001`).

        Returns:
            bool: True if the file is a part file, False otherwise.
        """
        return bool(re.search(r"\.(zip|tar|tar\.gz|tgz)\.\d+$", self.parsed_url.path))

    @property
    def is_compressed(self) -> bool:
        """
        Checks if the file is a compressed archive (e.g., `.zip`, `.tar.gz`).

        Returns:
            bool: True if the file is compressed, False otherwise.
        """
        if self.url_ext is not None:
            return bool(
                re.search(_COMPRESSED_FILES_REGEX, self.parsed_url.path)
            ) or bool(re.search(_COMPRESSED_FILES_REGEX, self.url_ext))
        else:
            return bool(re.search(_COMPRESSED_FILES_REGEX, self.parsed_url.path))

    @property
    def output_path(self) -> Path:
        """
        Determines the final output path for the downloaded file.

        Returns:
            Path: The output path for the file.
        """
        return self.data_dir / self.rel_output_file_path

    @property
    def is_download_completed(self) -> bool:
        """
        Checks if the download is completed by verifying the existence of the output path.

        Returns:
            bool: True if the download is completed, False otherwise.
        """
        return self.data_dir.exists() and self.output_path.exists()
