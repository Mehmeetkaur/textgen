from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest
from PIL import Image as PILImageModule

from atria_types._factory import ImageFactory
from atria_types._generic._image import Image
from atria_types._utilities._image_encoding import _image_to_bytes
from tests.data_model_test_base import DataModelTestBase


class TestImage(DataModelTestBase):
    """
    Test class for Image.
    """

    factory = ImageFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "file_path": pa.string(),
            "content": pa.binary(),
            "source_width": pa.int64(),
            "source_height": pa.int64(),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "file_path": pa.string(),
            "content": pa.binary(),
            "source_width": pa.int64(),
            "source_height": pa.int64(),
        }


#########################################################
# Basic Image Tests
#########################################################
@pytest.fixture
def valid_image_path(tmp_path: Path) -> str:
    image_path = tmp_path / "test_image.jpg"
    PILImageModule.new("RGB", (100, 100)).save(image_path)
    return str(image_path)


@pytest.fixture
def valid_raw_image(valid_image_path: str) -> Image:
    return Image(file_path=valid_image_path)


def test_image_initialization(valid_image_path: str) -> None:
    raw_image = Image(file_path=valid_image_path)
    assert str(raw_image.file_path) == valid_image_path
    assert raw_image.content is None


def test_load_from_file(valid_raw_image: Image) -> None:
    valid_raw_image = valid_raw_image.load()
    assert valid_raw_image.content is not None
    assert valid_raw_image.size == (100, 100)


@patch("requests.get")
def test_load_from_url(mock_get: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = _image_to_bytes(PILImageModule.new("RGB", (100, 100)))
    mock_get.return_value = mock_response

    raw_image = Image(file_path="https://example.com/test_image.jpg")
    raw_image = raw_image.load()
    assert raw_image.content is not None


def test_shape(valid_raw_image: Image) -> None:
    valid_raw_image = valid_raw_image.load()
    assert valid_raw_image.shape == (3, 100, 100)


def test_size(valid_raw_image: Image) -> None:
    valid_raw_image = valid_raw_image.load()
    assert valid_raw_image.size == (100, 100)


def test_channels(valid_raw_image: Image) -> None:
    valid_raw_image = valid_raw_image.load()
    assert valid_raw_image.channels == 3


def test_to_rgb(valid_raw_image: Image) -> None:
    valid_raw_image = valid_raw_image.load()
    valid_raw_image = valid_raw_image.ops.to_grayscale()
    assert valid_raw_image.channels == 1
    valid_raw_image = valid_raw_image.ops.to_rgb()
    assert valid_raw_image.channels == 3


def test_resize(valid_raw_image: Image) -> None:
    valid_raw_image = valid_raw_image.load()
    valid_raw_image = valid_raw_image.ops.resize(50, 50)
    assert valid_raw_image.size == (50, 50)
