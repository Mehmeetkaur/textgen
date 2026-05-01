from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from atria_datasets.core.typing.common import T_BaseDataInstance

    from atria_datasets.core.dataset._datasets import DatasetLoadingMode

logger = get_logger(__name__)


def transform_file_path(value: str, storage_dir: str, config_name: str) -> str:
    from urllib.parse import urlparse, urlunparse

    base_dir = f"{storage_dir}/{config_name}"
    parsed = urlparse(value)
    # Remove leading slash from path if exists, then join
    path = parsed.path.lstrip("/")
    new_path = f"{base_dir}/{path}"

    # Rebuild the URL with the original scheme and the new path
    transformed_url = urlunparse(
        (parsed.scheme, "", new_path, "", parsed.query, parsed.fragment)
    )
    return transformed_url


def get_presigned_url_with_original_query(
    original_url: str,
    s3_client,
    bucket_name: str,
    key_prefix: str,
    expires_in: int = 3600,
) -> str:
    """
    Generate a presigned S3 URL for the given original URL, preserving
    its original query parameters by merging them with the presigned URL's query.

    Args:
        original_url: Original file URL (may include query params).
        s3_client: boto3 S3 client.
        bucket_name: S3 bucket name.
        key_prefix: prefix to add before the key (e.g. branch/config path).
        expires_in: Expiry seconds for presigned URL.

    Returns:
        Presigned URL string with merged query params.
    """
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    parsed = urlparse(original_url)
    original_query = parse_qs(parsed.query)
    key_path = parsed.path.lstrip("/")  # strip leading slash for S3 key

    # Generate presigned URL without original query params
    presigned_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": f"{key_prefix}/{key_path}"},
        ExpiresIn=expires_in,
    )
    parsed_presigned = urlparse(presigned_url)
    presigned_query = parse_qs(parsed_presigned.query)

    # Merge queries, original query params overwrite presigned if duplicated
    merged_query = {**presigned_query, **original_query}

    # Flatten merged query for urlencode
    flat_query = []
    for k, vlist in merged_query.items():
        for v in vlist:
            flat_query.append((k, v))
    new_query = urlencode(flat_query)

    # Rebuild full URL
    return urlunparse(
        (
            parsed_presigned.scheme,
            parsed_presigned.netloc,
            parsed_presigned.path,
            parsed_presigned.params,
            new_query,
            parsed_presigned.fragment,
        )
    )


class DeltalakeReader(Sequence["T_BaseDataInstance"]):
    def __init__(
        self,
        table_path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        **kwargs,
    ):
        self.path = table_path
        self.data_model = data_model
        self.allowed_keys = allowed_keys
        self._length = 0

    @classmethod
    def from_mode(
        cls,
        mode: DatasetLoadingMode,
        table_path: str,
        data_model: type[T_BaseDataInstance],
        allowed_keys: set[str] | None = None,
        storage_dir: str | None = None,
        config_path: str | None = None,
        storage_options: dict | None = None,
        presign_expiry: int = 3600,
    ) -> DeltalakeReader:
        from atria_datasets.core.dataset._datasets import DatasetLoadingMode

        kwargs = {
            "table_path": table_path,
            "data_model": data_model,
            "allowed_keys": allowed_keys,
            "storage_dir": storage_dir,
            "config_path": config_path,
            "storage_options": storage_options,
            "presign_expiry": presign_expiry,
        }

        if mode == DatasetLoadingMode.in_memory:
            return InMemoryDeltalakeReader(**kwargs)
        elif mode == DatasetLoadingMode.local_streaming:
            return LocalDeltalakeReader(**kwargs)
        elif mode == DatasetLoadingMode.online_streaming:
            return OnlineDeltalakeReader(**kwargs)
        else:
            raise ValueError(f"Unsupported loading mode: {mode}")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> T_BaseDataInstance:
        if isinstance(index, list):
            return self.__getitems__(index)
        return self._load_and_process_rows([index])[0]

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:
        return self._load_and_process_rows(indices)

    def _process_row(self, row: dict) -> T_BaseDataInstance:
        if self.allowed_keys:
            row = {k: v for k, v in row.items() if k in self.allowed_keys}
        return self.data_model.from_row(row)

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        raise NotImplementedError

    def dataframe(self) -> pd.DataFrame:
        raise NotImplementedError


class InMemoryDeltalakeReader(DeltalakeReader):
    def __init__(self, *args, **kwargs):
        from deltalake import DeltaTable

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        self.storage_options = kwargs.pop("storage_options", None)
        super().__init__(*args, **kwargs)
        self._df = DeltaTable(
            self.path, storage_options=self.storage_options
        ).to_pandas()
        if self.allowed_keys:
            self.allowed_keys = [
                col
                for col in self._df.columns
                if col.startswith(tuple(self.allowed_keys))
            ]
        self._length = len(self._df)

    def _process_row(self, row: dict) -> Any:
        for key, value in row.items():
            if (
                isinstance(value, str)
                and "file_path" in key
                and self.storage_dir
                and self.config_name
            ):
                row[key] = transform_file_path(
                    value, self.storage_dir, self.config_name
                )
        return super()._process_row(row)

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        rows = self._df.iloc[indices]
        if self.allowed_keys:
            rows = rows[self.allowed_keys]
        row_dicts = rows.to_dict(orient="records")
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._df


class LocalDeltalakeReader(DeltalakeReader):
    def __init__(self, *args, **kwargs):
        import pyarrow.dataset as ds
        from deltalake import DeltaTable
        from pyarrow.fs import S3FileSystem

        self.storage_dir = kwargs.pop("storage_dir", None)
        self.config_name = kwargs.pop("config_name", None)
        self.storage_options = kwargs.pop("storage_options", None)
        super().__init__(*args, **kwargs)
        delta = DeltaTable(self.path, storage_options=self.storage_options)
        file_uris = delta.file_uris()
        if file_uris[0].startswith("lakefs://"):
            # set AWS_EC2_METADATA_DISABLED to improve load time of s3fs see https://github.com/apache/arrow/issues/37136
            filesystem = S3FileSystem(
                endpoint_override=self.storage_options["AWS_ENDPOINT"],
                access_key=self.storage_options["AWS_ACCESS_KEY_ID"],
                secret_key=self.storage_options["AWS_SECRET_ACCESS_KEY"],
            )
            file_uris = [file_uri.replace("lakefs://", "") for file_uri in file_uris]
            self._dataset = ds.dataset(file_uris, filesystem=filesystem)
        else:
            self._dataset = ds.dataset(self.path)
        self._length = self._dataset.count_rows()

    def _process_row(self, row: dict) -> Any:
        for key, value in row.items():
            if (
                isinstance(value, str)
                and "file_path" in key
                and self.storage_dir
                and self.config_name
            ):
                row[key] = transform_file_path(
                    value, self.storage_dir, self.config_name
                )
        return super()._process_row(row)

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        row_dicts = self._dataset.take(indices).to_pylist()
        return [self._process_row(row) for row in row_dicts]

    def dataframe(self) -> pd.DataFrame:
        return self._dataset.to_table().to_pandas()


class OnlineDeltalakeReader(LocalDeltalakeReader):
    def __init__(self, *args, **kwargs):
        self.presign_expiry = kwargs.pop("presign_expiry", 3600)
        super().__init__(*args, **kwargs)
        self._initialize_s3_client()
        self._extract_s3_path_parts()

    def _initialize_s3_client(self):
        import boto3
        from botocore.config import Config

        self._s3_client = boto3.client(
            "s3",
            endpoint_url=self.storage_options.get("AWS_ENDPOINT"),
            aws_access_key_id=self.storage_options.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self.storage_options.get("AWS_SECRET_ACCESS_KEY"),
            region_name="stub",
            config=Config(signature_version="s3v4"),
        )

    def _extract_s3_path_parts(self):
        parts = self.path.split("/")
        self.repo_name = parts[2]
        self.branch_name = parts[3]
        self.config_name = parts[4]

    def _load_and_process_rows(self, indices: list[int]) -> list[T_BaseDataInstance]:
        row_dicts = self._dataset.take(indices).to_pylist()
        for row in row_dicts:
            for key, value in row.items():
                if isinstance(value, str) and "file_path" in key:
                    row[key] = get_presigned_url_with_original_query(
                        value,
                        self._s3_client,
                        self.repo_name,
                        f"{self.branch_name}/{self.config_name}",
                        expires_in=self.presign_expiry,
                    )
        return [self._process_row(row) for row in row_dicts]
