import pyarrow as pa

from tests.data_model_test_base import DataModelTestBase
from tests.mock_data_models import MockDataModelParentFactory


class TestMockDataModelParent(DataModelTestBase):
    """
    Test class for MockDataModelParent.
    """

    factory = MockDataModelParentFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "required_integer_attribute": pa.int64(),
            "required_integer_list_attribute": pa.list_(pa.int64()),
            "integer_attribute": pa.int64(),
            "float_attribute": pa.float64(),
            "string_attribute": pa.string(),
            "list_attribute": pa.list_(pa.int64()),
            "integer_list_attribute": pa.list_(pa.int64()),
            "float_list_attribute": pa.list_(pa.float64()),
            "string_list_attribute": pa.list_(pa.string()),
            "example_data_model_child": {
                "required_integer_attribute": pa.int64(),
                "required_integer_list_attribute": pa.list_(pa.int64()),
                "integer_attribute": pa.int64(),
                "float_attribute": pa.float64(),
                "string_attribute": pa.string(),
                "list_attribute": pa.list_(pa.int64()),
                "integer_list_attribute": pa.list_(pa.int64()),
                "float_list_attribute": pa.list_(pa.float64()),
                "string_list_attribute": pa.list_(pa.string()),
            },
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "required_integer_attribute": pa.int64(),
            "required_integer_list_attribute": pa.list_(pa.int64()),
            "integer_attribute": pa.int64(),
            "float_attribute": pa.float64(),
            "string_attribute": pa.string(),
            "list_attribute": pa.list_(pa.int64()),
            "integer_list_attribute": pa.list_(pa.int64()),
            "float_list_attribute": pa.list_(pa.float64()),
            "string_list_attribute": pa.list_(pa.string()),
            "example_data_model_child__required_integer_attribute": pa.int64(),
            "example_data_model_child__required_integer_list_attribute": pa.list_(
                pa.int64()
            ),
            "example_data_model_child__integer_attribute": pa.int64(),
            "example_data_model_child__float_attribute": pa.float64(),
            "example_data_model_child__string_attribute": pa.string(),
            "example_data_model_child__list_attribute": pa.list_(pa.int64()),
            "example_data_model_child__integer_list_attribute": pa.list_(pa.int64()),
            "example_data_model_child__float_list_attribute": pa.list_(pa.float64()),
            "example_data_model_child__string_list_attribute": pa.list_(pa.string()),
        }
