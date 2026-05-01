import base64
import gzip
import io


def _compress_string(input: str) -> bytes:
    with io.BytesIO() as buffer:
        with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
            f.write(input.encode("utf-8"))
        return buffer.getvalue()


def _encode_string(input: str) -> str:
    return base64.b64encode(_compress_string(input)).decode("utf-8")


def _decompress_string(input: bytes) -> str:
    try:
        with io.BytesIO(input) as buffer:
            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                return f.read().decode("utf-8")
    except gzip.BadGzipFile:
        return input.decode("utf-8") if isinstance(input, bytes) else input


def _decode_string(input: str | bytes) -> str:
    if isinstance(input, bytes):
        return _decompress_string(base64.b64decode(input))
    return input
