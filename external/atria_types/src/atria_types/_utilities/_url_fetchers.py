from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests


def _load_bytes_from_uri(uri: str) -> bytes:
    parsed = urlparse(uri)
    query = parse_qs(parsed.query)
    path = parsed.path

    if parsed.scheme in ["http", "https", "ftp"]:
        if path.endswith(".tar"):
            try:
                offset = int(query.get("offset", [None])[0])  # type: ignore
                length = int(query.get("length", [None])[0])  # type: ignore
            except (TypeError, ValueError):
                raise ValueError(
                    f"Missing or invalid 'offset' and 'length' in tar URI: {uri}"
                ) from None
            # Remove offset and length from query dict
            query.pop("offset", None)
            query.pop("length", None)

            # Rebuild the query string without offset and length
            new_query = urlencode(query, doseq=True)

            # Rebuild the URL without offset and length params
            new_url = urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment,
                )
            )

            response = requests.get(
                new_url,
                headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                stream=True,
            )
            response.raise_for_status()
            return response.content
        else:
            response = requests.get(uri)
            response.raise_for_status()
            return response.content

    elif parsed.scheme in ["", "file", "tar"]:
        # local file or tar file with offset/length
        if path.endswith(".tar"):
            try:
                offset = int(query.get("offset", [None])[0])  # type: ignore
                length = int(query.get("length", [None])[0])  # type: ignore
            except (TypeError, ValueError):
                raise ValueError(
                    f"Missing or invalid 'offset' and 'length' in tar URI: {uri}"
                ) from None

            local_path = Path(path)
            if not local_path.exists():
                raise FileNotFoundError(f"TAR archive not found: {local_path}")

            with open(local_path, "rb") as f:
                f.seek(offset)
                data = f.read(length)
                return data
        else:
            local_path = Path(path if parsed.scheme != "file" else parsed.path)
            if not local_path.exists():
                raise FileNotFoundError(f"File not found: {local_path}")
            if not local_path.is_file():
                raise ValueError(f"Provided path is not a file: {local_path}")

            return local_path.read_bytes()
    else:
        raise ValueError(f"Unsupported URI scheme: {parsed.scheme}")
