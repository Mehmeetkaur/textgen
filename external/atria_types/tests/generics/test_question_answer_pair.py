import pyarrow as pa

from atria_types._factory import QAPairFactory
from tests.data_model_test_base import DataModelTestBase


class TestExtractiveQAPair(DataModelTestBase):
    """
    Test class for ExtractiveQAPair.
    """

    factory = QAPairFactory

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        """
        Expected table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "id": pa.int64(),
            "question_text": pa.string(),
            "answer_spans": pa.string(),
        }

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        """
        Expected flattened table schema for the BaseDataModel.
        This should be overridden by child classes to provide specific schemas.
        """
        return {
            "id": pa.int64(),
            "question_text": pa.string(),
            "answer_spans": pa.string(),
        }
