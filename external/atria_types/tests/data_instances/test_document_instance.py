import pyarrow as pa

from atria_types._factory import DocumentInstanceFactory
from tests.data_model_test_base import DataModelTestBase


class TestDocumentInstance(DataModelTestBase):
    """
    Test class
    """

    factory = DocumentInstanceFactory

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
            "pdf": {
                "file_path": pa.string(),
                "num_pages": pa.int64(),
            },
            "content": {
                "text_elements": pa.string(),
                "text": pa.string(),
            },
            "annotations": pa.string(),
            "ocr": {
                "file_path": pa.string(),
                "content": pa.binary(),
                "type": pa.string(),
            },
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
            "pdf__num_pages": pa.int64(),
            "pdf__file_path": pa.string(),
            "content__text_elements": pa.string(),
            "content__text": pa.string(),
            "annotations": pa.string(),
            "ocr__file_path": pa.string(),
            "ocr__content": pa.binary(),
            "ocr__type": pa.string(),
        }
