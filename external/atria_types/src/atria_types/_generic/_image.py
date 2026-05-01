from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from pydantic import field_validator
from rich.repr import RichReprResult

from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    OptIntField,
    OptStrField,
    ValidatedPILImage,
)

if TYPE_CHECKING:
    from ._ops._image_ops import ImageOps

logger = get_logger(__name__)


class Image(BaseDataModel):
    """
    Clean image data class.
    Operations accessed via: image.ops
    """

    file_path: OptStrField = None
    content: ValidatedPILImage = None

    source_width: OptIntField = None
    source_height: OptIntField = None

    # -------------------------------------
    # Bound service object
    # -------------------------------------
    @property
    def ops(self) -> ImageOps:
        from ._ops._image_ops import ImageOps

        return ImageOps(self)

    # -------------------------------------
    # Basic attributes
    # -------------------------------------
    @property
    def size(self) -> tuple[int, int]:
        assert self.content is not None, "Image content is not loaded."
        return self.content.size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def channels(self) -> int:
        assert self.content is not None
        return len(self.content.getbands())

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, *self.size)

    # -------------------------------------
    # Validators
    # -------------------------------------
    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, value: Any) -> str | None:
        if isinstance(value, Path):
            return str(value)
        return value

    # -------------------------------------
    # Load operation
    # -------------------------------------
    def load(self) -> Image:
        """
        Load image bytes from local path or URI.
        """
        if self.content is None:
            from atria_types._utilities._image_encoding import _bytes_to_image
            from atria_types._utilities._url_fetchers import _load_bytes_from_uri

            if self.file_path is None:
                raise ValueError("file_path must be set before loading")

            img = _bytes_to_image(_load_bytes_from_uri(self.file_path))

            return self.model_copy(
                update={
                    "content": img,
                    "source_width": img.width,
                    "source_height": img.height,
                }
            )
        elif self.source_width is None or self.source_height is None:
            return self.model_copy(
                update={
                    "source_width": self.content.width,
                    "source_height": self.content.height,
                }
            )
        return self

    # -------------------------------------
    # Rich repr
    # -------------------------------------
    def __rich_repr__(self) -> RichReprResult:
        yield from super().__rich_repr__()
        if self.content is not None:
            yield "width", self.width
            yield "height", self.height
            yield "channels", self.channels
