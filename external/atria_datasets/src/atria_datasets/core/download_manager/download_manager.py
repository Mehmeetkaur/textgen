"""
Download Manager Module

This module defines the `DownloadManager` class, which provides utilities for downloading,
merging, extracting, and finalizing files from various sources. It supports handling
compressed files, multi-part archives, and directory management for downloaded files.

Classes:
    - DownloadManager: Manages the downloading and extraction of dataset files.

Dependencies:
    - shutil: For file operations such as moving and extracting archives.
    - pathlib.Path: For handling file paths.
    - typing: For type annotations.
    - atria_corelogger.logger: For logging utilities.
    - atria.data.datasets.downloads.download_file_info: For managing file download information.
    - atria.data.datasets.downloads.file_downloader: For downloading files from URLs.

Author: Your Name (your.email@example.com)
Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

import shutil
from pathlib import Path

import tqdm
from atria_logger import get_logger
from atria_types._utilities._repr import RepresentationMixin

from atria_datasets.core.download_manager.download_file_info import DownloadFileInfo
from atria_datasets.core.download_manager.file_downloader import FileDownloader

logger = get_logger(__name__)


class DownloadManager(RepresentationMixin):
    """
    Manages the downloading and extraction of dataset files.

    This class provides methods for preparing URLs, downloading files, merging multi-part
    archives, extracting compressed files, and finalizing file paths.

    Attributes:
        data_dir (Path): The directory where the final dataset files will be stored.
        download_dir (Path): The directory where temporary download files will be stored.
    """

    def __init__(self, data_dir: Path, download_dir: Path):
        """
        Initializes the `DownloadManager`.

        Args:
            data_dir (Path): The directory where the final dataset files will be stored.
            download_dir (Path): The directory where temporary download files will be stored.
        """
        self.data_dir = data_dir
        self.download_dir = download_dir

    def _prepare_urls_and_dirs(
        self, data_urls: str | list[str] | dict[str, str]
    ) -> list[DownloadFileInfo]:
        """
        Prepares the URLs and directories for downloading files.

        Args:
            data_urls (Union[str, List[str], Dict[str, str]]): The URLs of the files to download.

        Returns:
            List[DownloadFileInfo]: A list of `DownloadFileInfo` objects for each file.

        Raises:
            ValueError: If Google Drive URLs are provided in list format.
            AssertionError: If `data_urls` is not a list or dictionary.
        """
        if isinstance(data_urls, str):
            data_urls = [data_urls]
        if isinstance(data_urls, list):
            if any(x.startswith("https://drive.google.com/") for x in data_urls):
                raise ValueError(
                    "Google Drive URLs are not supported in list format. Use dictionary format instead."
                )
            data_urls = {Path(url).name: url for url in data_urls}
        assert isinstance(data_urls, dict), (
            f"data_urls must be a list or a dictionary, got {data_urls}"
        )
        download_file_infos = []
        for download_path, url in data_urls.items():
            if isinstance(url, tuple):
                url, url_ext = url
            else:
                url_ext = None
            download_file_infos.append(
                DownloadFileInfo(
                    url=url,
                    rel_output_file_path=download_path,
                    data_dir=self.data_dir,
                    download_dir=self.download_dir,
                    url_ext=url_ext,
                )
            )
        return download_file_infos

    def _download_files(
        self,
        download_file_infos: list[DownloadFileInfo],
        access_token: str | None = None,
    ):
        """
        Downloads files from the provided URLs.

        Args:
            download_file_infos (List[DownloadFileInfo]): A list of `DownloadFileInfo` objects.
        """
        for download_file_info in download_file_infos:
            if (
                download_file_info.download_path.exists()
                or download_file_info.extracted_path.exists()
                or download_file_info.output_path.exists()
            ):
                continue

            url = download_file_info.parsed_url

            if "{access_token}" in url:
                url = url.format(access_token=access_token)

            file_downloader = FileDownloader.from_url(url, timeout=10)
            file_downloader.download(download_file_info)

    def _merge_part_files(self, download_file_infos: list[DownloadFileInfo]):
        """
        Merges multi-part archive files into a single file.

        Args:
            download_file_infos (List[DownloadFileInfo]): A list of `DownloadFileInfo` objects.
        """
        merged_files: dict[Path, list[Path]] = {}
        for download_file_info in download_file_infos:
            if download_file_info.is_part_file:
                merged_files.setdefault(download_file_info.extractable_path, []).append(
                    str(download_file_info.download_path)
                )

        for merged_file, parts in merged_files.items():
            parts = sorted(parts, key=lambda path: int(path.split(".")[-1]))
            expected_size = sum(Path(part).stat().st_size for part in parts)

            if merged_file.exists():
                actual_size = merged_file.stat().st_size
                if actual_size == expected_size:
                    logger.info(
                        f"Skipping merge, already exists and size matches: {merged_file}"
                    )
                    continue
                else:
                    logger.warning(
                        f"Merged file {merged_file} exists but size mismatch "
                        f"(expected: {expected_size}, actual: {actual_size}), re-merging."
                    )
                    merged_file.unlink(missing_ok=True)

            logger.info(f"Merging {len(parts)} parts into {merged_file}")
            try:
                with open(merged_file, "wb") as f:
                    for part in tqdm.tqdm(parts, "Merging parts", unit="part"):
                        assert Path(part).exists(), f"Part file {part} not found"
                        with open(part, "rb") as part_file:
                            f.write(part_file.read())
            except KeyboardInterrupt:
                logger.warning("Merge interrupted by user, cleaning up.")
                merged_file.unlink(missing_ok=True)
                raise
            except Exception as e:
                merged_file.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Failed to merge part files into {merged_file}"
                ) from e

    def _extract_archives(self, download_file_infos: list[DownloadFileInfo]):
        """
        Extracts compressed archive files.

        Args:
            download_file_infos (List[DownloadFileInfo]): A list of `DownloadFileInfo` objects.

        Raises:
            RuntimeError: If extraction fails for any file.
        """
        for download_file_info in download_file_infos:
            if (
                not download_file_info.is_compressed
                or download_file_info.extracted_path.exists()
                or download_file_info.output_path.exists()
            ):
                continue
            logger.info(
                f"Extracting {download_file_info.extractable_path} to {download_file_info.extracted_path}"
            )
            try:
                incomplete_extracted_path = (
                    download_file_info.extracted_path.with_suffix(
                        download_file_info.download_path.suffix + ".incomplete"
                    )
                )
                if incomplete_extracted_path.exists():
                    incomplete_extracted_path.unlink()
                shutil.unpack_archive(
                    download_file_info.extractable_path, incomplete_extracted_path
                )
                incomplete_extracted_path.rename(download_file_info.extracted_path)
            except KeyboardInterrupt:
                logger.warning("Extraction interrupted by user, cleaning up.")
                if incomplete_extracted_path.exists():
                    shutil.rmtree(incomplete_extracted_path)
                raise
            except Exception as e:
                if incomplete_extracted_path.exists():
                    shutil.rmtree(incomplete_extracted_path)
                raise RuntimeError(
                    f"Failed to extract {download_file_info.extractable_path}"
                ) from e

    def _finalize_files(
        self, download_file_infos: list[DownloadFileInfo], extract: bool = True
    ):
        """
        Finalizes the downloaded files by moving them to the output directory.

        Args:
            download_file_infos (List[DownloadFileInfo]): A list of `DownloadFileInfo` objects.
        """
        for download_file_info in download_file_infos:
            logger.info("Finalizing file: %s", download_file_info.download_path)
            if download_file_info.output_path.exists():
                continue
            if download_file_info.is_part_file:
                continue
            if download_file_info.is_compressed and extract:
                if download_file_info.extracted_path.exists():
                    logger.debug(
                        f"Removing compressed file {download_file_info.download_path}"
                    )
                    download_file_info.extractable_path.unlink(missing_ok=True)
                shutil.move(
                    download_file_info.extracted_path, download_file_info.output_path
                )
            else:
                logger.debug(
                    f"Moving {download_file_info.download_path} to {download_file_info.output_path}"
                )
                shutil.move(
                    download_file_info.download_path, download_file_info.output_path
                )

    def download_and_extract(
        self,
        data_urls: str | list[str] | dict[str, str],
        extract: bool = True,
        access_token: str | None = None,
    ) -> dict[str, Path]:
        """
        Downloads and extracts files from the provided URLs.

        Args:
            data_urls (Union[str, List[str], Dict[str, str]]): The URLs of the files to download.

        Returns:
            Dict[str, Path]: A dictionary mapping file names to their final output paths.
        """
        download_file_infos = self._prepare_urls_and_dirs(data_urls)
        if extract:
            for download_file_info in download_file_infos:
                download_file_info.update_extract_path()

        if any(
            not download_file_info.is_download_completed
            for download_file_info in download_file_infos
        ):
            logger.info(
                f"Downloading {len(download_file_infos)} files to {self.download_dir}"
            )
            self._download_files(download_file_infos)
            self._merge_part_files(download_file_infos)
            if extract:
                self._extract_archives(download_file_infos)
            self._finalize_files(download_file_infos, extract=extract)
        return {
            download_file_info.output_path.name: download_file_info.output_path
            for download_file_info in download_file_infos
        }
