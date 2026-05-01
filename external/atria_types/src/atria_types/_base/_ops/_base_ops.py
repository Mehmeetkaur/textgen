from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from atria_logger import get_logger

from atria_types._generic._bounding_box import BaseDataModel

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseDataModel)


class OpsBase(Generic[T]):
    """Base Operations for Base Data Models

    A mixin class that provides basic operations for base data models.
    """

    def __init__(self, model: T):
        self.model = model


class FilePathOps(OpsBase[T]):
    def convert_file_paths_to_relative(self, parent_dir: str) -> T:
        """Convert all file path fields to relative paths recursively.

        Args:
            parent_dir (str): The base directory to which paths should be made relative.

        Returns:
            BaseDataModel: A new instance of the data model with updated file paths.
        """

        updates = {}
        for field_name in self.model.__class__.model_fields:
            try:
                field_value = getattr(self.model, field_name)
                if isinstance(field_value, BaseDataModel):
                    updates[field_name] = (
                        field_value.ops.convert_file_paths_to_relative(
                            parent_dir=parent_dir
                        )
                    )
                elif "file_path" in field_name and field_value is not None:
                    path_obj = Path(field_value)
                    if path_obj.is_absolute():
                        relative_path = path_obj.relative_to(parent_dir)
                        updates[field_name] = str(relative_path)
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to relative path: {e}"
                ) from e
        return self.model.model_copy(update=updates)

    def convert_file_paths_to_absolute(self, parent_dir: str) -> T:
        """Convert all file path fields to absolute paths recursively.

        Args:
            parent_dir (str): The base directory from which relative paths should be made absolute.

        Returns:
            BaseDataModel: A new instance of the data model with updated file paths.
        """
        updates = {}
        for field_name in self.model.__class__.model_fields:
            try:
                field_value = getattr(self.model, field_name)
                if isinstance(field_value, BaseDataModel):
                    updates[field_name] = (
                        field_value.ops.convert_file_paths_to_absolute(
                            parent_dir=parent_dir
                        )
                    )
                elif "file_path" in field_name and field_value is not None:
                    path_obj = Path(field_value)
                    if not path_obj.is_absolute():
                        absolute_path = Path(parent_dir) / path_obj
                        updates[field_name] = str(absolute_path.resolve())
            except Exception as e:
                raise RuntimeError(
                    f"Error converting field '{field_name}' to absolute path: {e}"
                ) from e
        return self.model.model_copy(update=updates)


class StandardOps(FilePathOps[T], Generic[T]):
    """Standard Operations for Base Data Models

    A mixin class that combines standard operations for base data models.
    """

    pass
