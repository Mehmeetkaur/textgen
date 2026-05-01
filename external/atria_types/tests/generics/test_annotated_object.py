import pyarrow as pa

from atria_types._factory import AnnotatedObjectFactory
from tests.data_model_test_base import DataModelTestBase


class TestAnnotatedObject(DataModelTestBase):
    """
    Test class for AnnotatedObject.
    """

    factory = AnnotatedObjectFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "label": {"value": pa.int64(), "name": pa.string()},
            "bbox": {"value": pa.list_(pa.float64()), "mode": pa.string()},
            "segmentation": pa.list_(pa.list_(pa.float64())),
            "iscrowd": pa.bool_(),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "label__value": pa.int64(),
            "label__name": pa.string(),
            "bbox__value": pa.list_(pa.float64()),
            "bbox__mode": pa.string(),
            "segmentation": pa.list_(pa.list_(pa.float64())),
            "iscrowd": pa.bool_(),
        }
