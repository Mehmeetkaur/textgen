import pyarrow as pa

from atria_types._factory import LabelFactory
from tests.data_model_test_base import DataModelTestBase


class TestLabel(DataModelTestBase):
    """
    Test class for Label.
    """

    factory = LabelFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"name": pa.string(), "value": pa.int64()}

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {"name": pa.string(), "value": pa.int64()}
