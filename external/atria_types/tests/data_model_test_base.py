import factory
import pyarrow as pa
import pytest
from atria_logger import get_logger
from pydantic import ValidationError

from atria_types._base._data_model import BaseDataModel
from tests.utilities import _compare_dicts_recursively

logger = get_logger(__name__)


class DataModelTestBase:
    throws_error_on_operations = False
    factory: type[factory.Factory]

    def expected_table_schema(self) -> dict[str, pa.DataType]:
        raise NotImplementedError(
            "Child classes must implement the `expected_table_schema` method."
        )

    def expected_table_schema_flattened(self) -> dict[str, pa.DataType]:
        raise NotImplementedError(
            "Child classes must implement the `expected_table_schema_flattened` method."
        )

    @pytest.fixture(params=list(range(1)))
    def model_instance(self) -> BaseDataModel:
        """
        Fixture to provide an instance of the BaseDataModel for testing.
        """
        return self.factory.build().load()

    def test_initialize(self, model_instance: BaseDataModel) -> None:
        """
        Test the initialization of the model instance.
        """
        assert isinstance(model_instance, BaseDataModel), (
            "Model instance is not a BaseDataModel"
        )

    def test_assert_instance_frozen(self, model_instance: BaseDataModel) -> None:
        """
        Test that the model instance is frozen (immutable).
        """
        for field in model_instance.__class__.model_fields:
            with pytest.raises(ValidationError):
                setattr(model_instance, field, None)

    def test_load_unload(self, model_instance: BaseDataModel) -> None:
        """
        Test the load method of the model instance.
        """
        loaded_model_instance = model_instance.load()
        assert type(loaded_model_instance) is type(model_instance), (
            "Loaded model instance type mismatch"
        )

    def test_to_from_row(self, model_instance: BaseDataModel) -> None:
        """
        Test the conversion to and from a row representation.
        """
        row = model_instance.to_row()
        assert isinstance(row, dict), "Row representation should be a dictionary"
        new_instance = model_instance.from_row(row)
        assert isinstance(new_instance, BaseDataModel), (
            "New instance should be a BaseDataModel"
        )
        assert new_instance.model_dump() == model_instance.model_dump(), (
            f"New instance does not match original, "
            f"Expected: {model_instance.model_dump()}, Got: {new_instance.model_dump()}"
        )

    def test_schema(self, model_instance: BaseDataModel) -> None:
        """
        Test the schema generation of the model instance.
        """
        schema = model_instance.table_schema()
        expected = self.expected_table_schema()

        differences = _compare_dicts_recursively(schema, expected)
        assert schema == expected, (
            "Schema does not match expected schema.\n"
            "Differences found:\n" + "\n".join(differences)
        )

    def test_schema_flattened(self, model_instance: BaseDataModel) -> None:
        """
        Test the schema generation of the model instance.
        """
        schema = model_instance.table_schema_flattened()
        expected = self.expected_table_schema_flattened()
        differences = _compare_dicts_recursively(schema, expected)
        assert schema == expected, (
            "Schema does not match expected schema.\n"
            "Differences found:\n" + "\n".join(differences)
        )
