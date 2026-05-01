from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.pretty import RichReprResult


class RepresentationMixin:
    """
    Mixin class for rich representation of objects.

    This class provides methods for generating string representations of objects
    using the `rich` library. It includes support for both developer-friendly
    (`__repr__`) and human-readable (`__str__`) representations.

    Set `__repr_fields__` as a class attribute to specify which fields to include.
    """

    __repr_fields__: set[str] = set()

    def __repr_name__(self) -> str:
        """
        Returns the name of the class for use in the `__repr__` method.

        Returns:
            str: The name of the class.
        """
        return self.__class__.__name__

    def __rich_repr__(self) -> RichReprResult:  # type: ignore
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs for the specified fields only.
        """
        import types

        repr_fields = getattr(self.__class__, "__repr_fields__", set())  # type: ignore
        if len(repr_fields) == 0:
            repr_fields = self.__dict__.keys()

        for field_name in repr_fields:
            if not hasattr(self, field_name):
                continue

            # do not add private fields
            if field_name.startswith("_"):
                continue

            value = getattr(self, field_name)
            if isinstance(value, types.MethodType):
                safe_value = value.__func__
            else:
                safe_value = value

            yield field_name, safe_value

    def __repr__(self) -> str:
        """
        Generates a developer-friendly string representation of the object.

        Returns:
            str: A developer-friendly string representation of the object.
        """

        from rich.pretty import pretty_repr

        from atria_types._constants import _MAX_REPR_PRINT_ELEMENTS

        return pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=128, max_depth=8
        )

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the object.

        Returns:
            str: A human-readable string representation of the object.
        """

        from rich.pretty import pretty_repr

        from atria_types._constants import _MAX_REPR_PRINT_ELEMENTS

        return pretty_repr(
            self, max_length=_MAX_REPR_PRINT_ELEMENTS, max_string=128, max_depth=8
        )
