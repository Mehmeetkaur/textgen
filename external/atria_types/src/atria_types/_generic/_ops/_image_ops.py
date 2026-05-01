from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from atria_logger import get_logger
from PIL.Image import Resampling

from atria_types._base._ops._base_ops import StandardOps
from atria_types._generic._image import Image

logger = get_logger(__name__)

if TYPE_CHECKING:
    import torch


class ImageOps(StandardOps[Image]):
    """
    All image operations live here.
    Bound service object used through: image.ops
    """

    @property
    def image(self) -> Image:
        return self.model

    @property
    def content(self):
        assert self.image.content is not None, "Image content is missing."
        return self.image.content

    # -----------------------------
    # Conversion methods
    # -----------------------------
    def to_tensor(self) -> torch.Tensor:
        from torchvision.transforms.functional import to_tensor

        return to_tensor(self.content)

    def to_numpy(self) -> np.ndarray:
        return np.array(self.content)

    # -----------------------------
    # Color space conversions
    # -----------------------------
    def to_rgb(self) -> Image:
        return self.image.model_copy(update={"content": self.content.convert("RGB")})

    def to_grayscale(self) -> Image:
        return self.image.model_copy(update={"content": self.content.convert("L")})

    # -----------------------------
    # Resizing
    # -----------------------------
    def resize(
        self, width: int, height: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        return self.image.model_copy(
            update={"content": self.content.resize((width, height), resample)}
        )

    def resize_with_aspect_ratio(
        self, max_size: int, resample: Resampling = Resampling.BICUBIC
    ) -> Image:
        assert max_size > 0, "max_size must be > 0"

        width, height = self.image.size
        if max(width, height) <= max_size:
            return self.image

        if width >= height:
            new_w = max_size
            new_h = int(height * (max_size / width))
        else:
            new_h = max_size
            new_w = int(width * (max_size / height))

        return self.resize(new_w, new_h, resample)
