import types
from types import NoneType
from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from atria_logger import get_logger
from pydantic import BaseModel

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "__"
) -> dict[str, Any]:
    """
    Recursively flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested fields
        sep: Separator to use between nested keys

    Returns:
        dict[str, Any]: Flattened dictionary

    Example:
        >>> _flatten_dict({"a": {"b": 1, "c": 2}})
        {"a_b": 1, "a_c": 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(
    flat: dict[str, Any], schema: dict[str, Any], sep: str = "__"
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, sub_schema in schema.items():
        if isinstance(sub_schema, dict):
            nested = {
                k.replace(f"{key}{sep}", ""): v
                for k, v in flat.items()
                if k.startswith(f"{key}{sep}")
            }
            if all(value is None for value in nested.values()):
                result[key] = None
            else:
                result[key] = _unflatten_dict(
                    nested,
                    sub_schema,
                )
        else:
            # Primitive value
            if key in flat:
                result[key] = flat[key]

    return result


def _extract_pyarrow_schema(model_cls: type[BaseModel]) -> dict[str, type | dict]:
    """
    Extract PyArrow schema from a BaseDataModel model class.

    Args:
        model_cls: Model class to extract schema from

    Returns:
        dict[str, Any]: Schema mapping field names to PyArrow types

    Raises:
        TypeError: If model_cls is not a valid BaseDataModel class
    """
    from atria_types._base._data_model import BaseDataModel
    from atria_types._pydantic import TableSchemaMetadata

    if not issubclass(model_cls, BaseModel) or not issubclass(model_cls, BaseDataModel):
        raise TypeError(
            f"Expected a subclass of {BaseDataModel} subclass, got {model_cls}"
        )

    schema: dict[str, type | dict] = {}

    type_hints = get_type_hints(model_cls, include_extras=True)
    for field_name, annotated_type in type_hints.items():
        try:
            origin = get_origin(annotated_type)
            args = get_args(annotated_type)

            # Handle Optional[Annotated[...]] or Union types
            if origin in {Union, types.UnionType} and len(args) > 1:
                # Strip NoneType from Union to handle Optional
                non_none_args = [arg for arg in args if arg is not NoneType]
                if len(non_none_args) == 1:
                    annotated_type = non_none_args[0]
                    origin = get_origin(annotated_type)
                    args = get_args(annotated_type)

            # Handle Annotated types
            if hasattr(annotated_type, "__metadata__"):  # Annotated type
                base_type = args[0] if args else annotated_type
                metadata = (
                    args[1:]
                    if len(args) > 1
                    else getattr(annotated_type, "__metadata__", [])
                )

                # Check if base_type is a BaseDataModel subclass
                if isinstance(base_type, type) and issubclass(base_type, BaseDataModel):
                    nested_schema = _extract_pyarrow_schema(base_type)
                    # Flatten nested schema with prefixed field names
                    for nested_field_name, nested_field_type in nested_schema.items():
                        schema[f"{field_name}_{nested_field_name}"] = nested_field_type
                else:
                    # Search for TableSchemaMetadata in metadata
                    for meta in metadata:
                        if isinstance(meta, TableSchemaMetadata):
                            schema[field_name] = meta.get_type()
                            break
                    else:
                        logger.debug(
                            f"No TableSchemaMetadata found for field {field_name}"
                        )

            # Handle direct BaseDataModel subclass (not annotated)
            elif isinstance(annotated_type, type) and issubclass(
                annotated_type, BaseDataModel
            ):
                nested_schema = _extract_pyarrow_schema(annotated_type)
                schema[field_name] = nested_schema

        except Exception as e:
            raise RuntimeError(
                f"Failed to process field {field_name} in {model_cls.__name__}"
            ) from e

    return schema
