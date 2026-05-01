from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Annotated, Any

from atria_logger import get_logger
from pydantic import field_validator, model_validator

from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    ListFloatField,
    TableSchemaMetadata,
)

if TYPE_CHECKING:
    from atria_types._generic._ops._bbox_ops import BoundingBoxOps

logger = get_logger(__name__)


class BoundingBoxMode(str, enum.Enum):
    XYXY = "xyxy"  # (x1, y1, x2, y2)
    XYWH = "xywh"  # (x1, y1, width, height)


class BoundingBox(BaseDataModel):
    value: ListFloatField
    mode: Annotated[BoundingBoxMode, TableSchemaMetadata(pa_type="string")] = (
        BoundingBoxMode.XYXY
    )
    normalized: bool = False

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: Any) -> BoundingBoxMode:
        if isinstance(value, str):
            return BoundingBoxMode(value)
        return value

    @field_validator("value", mode="after")
    @classmethod
    def validate_value(cls, value: Any) -> list[float]:
        assert len(value) == 4, "Expected a 1D list of shape (4,) for bounding boxes."
        return value

    @model_validator(mode="after")
    def check_value_mode_consistency(self) -> BoundingBox:
        if self.mode == BoundingBoxMode.XYXY:
            x1, y1, x2, y2 = self.value
            assert x2 >= x1, "In XYXY mode, x2 must be greater than or equal to x1."
            assert y2 >= y1, "In XYXY mode, y2 must be greater than or equal to y1."
        elif self.mode == BoundingBoxMode.XYWH:
            x1, y1, width, height = self.value
            assert width >= 0, "In XYWH mode, width must be non-negative."
            assert height >= 0, "In XYWH mode, height must be non-negative."
        return self

    @model_validator(mode="after")
    def validate_normalized(self) -> BoundingBox:
        if self.normalized:
            for coord in self.value:
                assert 0.0 <= coord <= 1.0, (
                    "All bounding box coordinates must be in the range [0, 1] "
                    "when 'normalized' is True."
                )
        return self

    # -------------------------------------
    # Bound service object
    # -------------------------------------
    @property
    def ops(self) -> BoundingBoxOps:
        from atria_types._generic._ops._bbox_ops import BoundingBoxOps

        return BoundingBoxOps(self)

    # -------------------------------------
    # Basic attributes
    # -------------------------------------
    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def x1(self) -> float:
        return self.value[0]

    @property
    def y1(self) -> float:
        return self.value[1]

    @property
    def x2(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.x1 + self.width
        else:
            return self.value[2]

    @property
    def y2(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.y1 + self.height
        else:
            return self.value[3]

    @property
    def width(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.value[2]
        else:
            return self.x2 - self.x1

    @property
    def height(self) -> float:
        if self.mode == BoundingBoxMode.XYWH:
            return self.value[3]
        else:
            return self.y2 - self.y1
