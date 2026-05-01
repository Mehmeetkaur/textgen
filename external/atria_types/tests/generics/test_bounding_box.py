import pyarrow as pa
import pytest
from pydantic import ValidationError

from atria_types._factory import BoundingBoxFactory
from atria_types._generic._bounding_box import BoundingBox, BoundingBoxMode
from tests.data_model_test_base import DataModelTestBase


class TestBoundingBox(DataModelTestBase):
    """
    Test class for BoundingBox.
    """

    factory = BoundingBoxFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"value": pa.list_(pa.float64()), "mode": pa.string()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"value": pa.list_(pa.float64()), "mode": pa.string()}


#########################################################
# Basic BoundingBox Tests
#########################################################
@pytest.fixture
def valid_bbox() -> BoundingBox:
    return BoundingBox(
        value=[10.0, 20.0, 30.0, 40.0], mode=BoundingBoxMode.XYXY
    )  # ignore[arg-type]


def test_initialization(valid_bbox: BoundingBox) -> None:
    assert valid_bbox.value == [10.0, 20.0, 30.0, 40.0]
    assert valid_bbox.mode == BoundingBoxMode.XYXY


def test_bbox_switch_mode(valid_bbox: BoundingBox) -> None:
    assert valid_bbox.x1 == 10.0
    assert valid_bbox.y1 == 20.0
    assert valid_bbox.x2 == 30.0
    assert valid_bbox.y2 == 40.0
    assert valid_bbox.width == 20.0
    assert valid_bbox.height == 20.0

    valid_bbox.ops.switch_mode()

    assert valid_bbox.x1 == 10.0
    assert valid_bbox.y1 == 20.0
    assert valid_bbox.x2 == 30.0
    assert valid_bbox.y2 == 40.0
    assert valid_bbox.width == 20.0
    assert valid_bbox.height == 20.0


def test_switch_mode(valid_bbox: BoundingBox) -> None:
    assert valid_bbox.mode == BoundingBoxMode.XYXY
    old_x1 = valid_bbox.x1
    old_y1 = valid_bbox.y1
    old_x2 = valid_bbox.x2
    old_y2 = valid_bbox.y2
    old_width = valid_bbox.width
    old_height = valid_bbox.height

    updated_bbox = valid_bbox.ops.switch_mode()

    assert updated_bbox.mode == BoundingBoxMode.XYWH
    assert updated_bbox.x1 == old_x1
    assert updated_bbox.y1 == old_y1
    assert updated_bbox.x2 == old_x2
    assert updated_bbox.y2 == old_y2
    assert updated_bbox.width == old_width
    assert updated_bbox.height == old_height


def test_fails_on_invalid_bboxes() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(
            value=[-10.0, -20.0, -30.0, -40.0], mode=BoundingBoxMode.XYXY
        )  # ignore[arg-type]

        BoundingBox(
            value=[10.0, 20.0, 5.0, 40.0], mode=BoundingBoxMode.XYXY
        )  # ignore[arg-type]


def test_normalize_bbox(valid_bbox: BoundingBox) -> None:
    normalized_bbox = valid_bbox.ops.normalize(width=100.0, height=200.0)
    assert normalized_bbox.value == [0.1, 0.1, 0.3, 0.2]
    assert normalized_bbox.mode == BoundingBoxMode.XYXY

    switched_valid_bbox = valid_bbox.ops.switch_mode()
    normalized_bbox = switched_valid_bbox.ops.normalize(width=100.0, height=200.0)
    assert normalized_bbox.value == [0.1, 0.1, 0.2, 0.1]
    assert normalized_bbox.mode == BoundingBoxMode.XYWH


def test_bbox_area(valid_bbox: BoundingBox) -> None:
    assert valid_bbox.area == 400.0
