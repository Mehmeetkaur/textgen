from typing import Annotated

from atria_types._base._data_model import BaseDataModel
from atria_types._generic._bounding_box import BoundingBox
from atria_types._generic._label import Label
from atria_types._pydantic import (
    BoolField,
    TableSchemaMetadata,
)


class AnnotatedObject(BaseDataModel):
    label: Label
    bbox: BoundingBox
    segmentation: Annotated[
        list[list[float]] | None,
        TableSchemaMetadata(pa_type="list<list<float64>>"),
    ] = None
    iscrowd: BoolField = False
