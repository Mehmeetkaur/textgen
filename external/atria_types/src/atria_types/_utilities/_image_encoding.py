import base64
import io

from PIL import Image
from PIL.Image import Image as PILImage


def _pil_image_to_bytes(image: "PILImage", format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def _image_to_bytes(image: PILImage, format: str = "PNG") -> bytes:
    from PIL.Image import Image as PILImage

    if not isinstance(image, PILImage):
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected PIL.Image.Image."
        )
    return _pil_image_to_bytes(image, format=format)


def _image_to_base64(image: PILImage) -> str:
    return base64.b64encode(_image_to_bytes(image)).decode("utf-8")


def _bytes_to_image(encoded_image: bytes) -> "PILImage":
    return Image.open(io.BytesIO(encoded_image))


def _base64_to_image(encoded_image: str) -> "PILImage":
    import io

    return Image.open(io.BytesIO(base64.b64decode(encoded_image)))
