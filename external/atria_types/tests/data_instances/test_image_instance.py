import pyarrow as pa

from atria_types._factory import ImageInstanceFactory
from tests.data_model_test_base import DataModelTestBase


class TestImageInstance(DataModelTestBase):
    """
    Test class for Label.
    """

    factory = ImageInstanceFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "index": pa.int64(),
            "sample_id": pa.string(),
            "image": {
                "file_path": pa.string(),
                "content": pa.binary(),
                "source_width": pa.int64(),
                "source_height": pa.int64(),
            },
            "annotations": pa.string(),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "index": pa.int64(),
            "sample_id": pa.string(),
            "image__file_path": pa.string(),
            "image__content": pa.binary(),
            "image__source_width": pa.int64(),
            "image__source_height": pa.int64(),
            "annotations": pa.string(),
        }
