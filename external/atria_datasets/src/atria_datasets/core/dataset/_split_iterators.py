"""Module defining the SplitIterator class for dataset splits."""

import sys
import traceback
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import rich
import rich.pretty
from atria_logger import get_logger
from atria_types import BaseDataInstance, DatasetSplitType
from atria_types._utilities._repr import RepresentationMixin

from atria_datasets.core.dataset._common import T_BaseDataInstance

if TYPE_CHECKING:
    import pandas as pd
logger = get_logger(__name__)


class InstanceTransform:
    def __init__(
        self,
        data_model: T_BaseDataInstance,
        input_transform: Callable | None = None,
        output_transform: Callable | None = None,
    ):
        self._data_model = data_model
        self._input_transform = input_transform
        self._output_transform = output_transform

    def __call__(self, index: int, data_instance: Any) -> BaseDataInstance:
        # apply input transformation
        if self._input_transform is not None:
            data_instance: BaseDataInstance = self._input_transform(data_instance)

        # assert that the transformed instance is of the expected data model type
        assert isinstance(data_instance, self._data_model), (
            f"self._input_transform(sample) should return {self._data_model}, but got {type(data_instance)}"
        )

        # apply index
        if data_instance.index is None:
            data_instance = data_instance.update(index=index)

        # yield the transformed data instance if output transform is enabled
        if self._output_transform is not None:
            data_instance = self._output_transform(data_instance)
        return data_instance


class SplitIterator(Sequence[T_BaseDataInstance], RepresentationMixin):
    __repr_fields__ = [
        "base_iterator",
        "input_transform",
        "output_transform",
        "subset_indices",
    ]

    def __init__(
        self,
        split: DatasetSplitType,
        base_iterator: Sequence | Generator,
        data_model: type[T_BaseDataInstance],
        input_transform: Callable | None = None,
        output_transform: Callable | None = None,
        max_len: int | None = None,
        subset_indices: list[int] | None = None,
    ):
        self._split = split
        self._base_iterator = base_iterator
        self._max_len = max_len
        self._subset_indices = subset_indices
        self._tf = InstanceTransform(
            input_transform=input_transform,
            data_model=data_model,
            output_transform=output_transform,
        )
        self._tf_enabled = True
        self._is_iterable = isinstance(self._base_iterator, Iterable)
        self._supports_indexing = hasattr(self._base_iterator, "__getitem__")
        self._supports_multi_indexing = hasattr(self._base_iterator, "__getitems__")
        if not self._is_iterable:
            assert hasattr(self._base_iterator, "__len__"), (
                f"T he base iterator {self._base_iterator} must implement __len__ to support indexing. "
            )
            assert self._supports_indexing or self._supports_multi_indexing, (
                f"The base iterator {self._base_iterator} must implement either __iter__, __getitem__ or __getitems__ "
            )

    def enable_tf(self) -> None:
        self._tf_enabled = True

    def disable_tf(self) -> None:
        self._tf_enabled = False

    @property
    def split(self) -> DatasetSplitType:
        return self._split

    @property
    def base_iterator(self) -> Iterable:
        return self._base_iterator

    @property
    def input_transform(self) -> Callable:
        return self._tf._input_transform

    @property
    def output_transform(self) -> Callable | None:
        return self._tf._output_transform

    @output_transform.setter
    def output_transform(self, value: Callable) -> None:
        self._tf._output_transform = value

    @property
    def subset_indices(self) -> list[int] | None:
        """
        Returns the subset indices if available, otherwise None.
        """
        return self._subset_indices

    @subset_indices.setter
    def subset_indices(self, indices: list[int]) -> None:
        """
        Sets the subset indices for the iterator.

        Args:
            indices (list[int]): A list of indices to set as the subset.
        """
        self._subset_indices = indices

    @property
    def data_model(self) -> T_BaseDataInstance:
        return self._tf._data_model

    def dataframe(self) -> "pd.DataFrame":
        """
        Displays the dataset split information in a rich format.
        """
        if hasattr(self._base_iterator, "dataframe"):
            return self._base_iterator.dataframe()
        else:
            raise RuntimeError(
                "This dataset is not backed by a DataFrame or does not support dataframe representation."
            )

    def __iter__(self) -> Iterator[T_BaseDataInstance]:
        try:
            if not self._supports_indexing:
                if self._subset_indices is not None:
                    raise RuntimeError(
                        "You are trying to iterate over a subset of the dataset, "
                        "but the base iterator does not support indexing. "
                    )

                for index, sample in enumerate(self._base_iterator):
                    if self._tf_enabled:
                        yield self._tf(index, sample)
                    else:
                        yield index, sample

                    if self._max_len is not None and index + 1 >= self._max_len:
                        break
            else:
                for index in range(len(self)):
                    yield self[index]
        except Exception:
            raise RuntimeError("".join(traceback.format_exception(*sys.exc_info())))

    def __getitem__(self, index: int) -> T_BaseDataInstance:  # type: ignore[override]
        try:
            if isinstance(index, list):
                return self.__getitems__(index)
            assert self._supports_indexing, (
                "The base iterator does not support multi-indexing. "
                "Please use __getitem__ for single index access."
            )
            if self._subset_indices is not None:
                index = self._subset_indices[index]
            if self._tf_enabled:
                return self._tf(index, self._base_iterator[index])  # type: ignore
            return index, self._base_iterator[index]  # type: ignore
        except Exception:
            raise RuntimeError("".join(traceback.format_exception(*sys.exc_info())))

    def __getitems__(self, indices: list[int]) -> list[T_BaseDataInstance]:  # type: ignore
        try:
            if self._subset_indices is not None:
                indices = [self._subset_indices[idx] for idx in indices]
            if self._supports_multi_indexing:
                data_instances = self._base_iterator[indices]  # type: ignore
            else:
                assert self._supports_indexing, (
                    "The base iterator does not support multi-indexing. "
                    "Please use __getitem__ for single index access."
                )
                data_instances = [self._base_iterator[index] for index in indices]  # type: ignore
            if self._tf_enabled:
                return [
                    self._tf(index, data_instance)
                    for index, data_instance in zip(
                        indices, data_instances, strict=True
                    )
                ]
            return [
                (index, data_instance)
                for index, data_instance in zip(indices, data_instances, strict=True)
            ]
        except Exception:
            raise RuntimeError("".join(traceback.format_exception(*sys.exc_info())))

    def __len__(self) -> int:
        if hasattr(self._base_iterator, "__len__"):
            iterator = (
                self._subset_indices
                if self._subset_indices is not None
                else self._base_iterator
            )
            if self._max_len is not None:
                return min(self._max_len, len(iterator))  # type: ignore
            return len(iterator)  # type: ignore
        elif self._max_len is not None:
            return self._max_len
        raise RuntimeError(
            "The dataset does not support length calculation. "
            "Please implement the `__len__` method in your dataset class."
        )

    def __rich_repr__(self) -> rich.pretty.RichReprResult:
        """
        Generates a rich representation of the object.

        Yields:
            RichReprResult: A generator of key-value pairs or values for the object's attributes.
        """
        yield from super().__rich_repr__()
        try:
            yield "num_rows", len(self)
        except Exception:
            yield "num_rows", "unknown"


class HFSplitIterator(SplitIterator):
    def __init__(
        self,
        split: DatasetSplitType,
        base_iterator: Sequence | Generator,
        data_model: type[T_BaseDataInstance],
        input_transform: Callable | None = None,
        output_transform: Callable | None = None,
        max_len: int | None = None,
        subset_indices: list[int] | None = None,
    ):
        self._split = split
        self._base_iterator = base_iterator
        self._max_len = max_len
        self._subset_indices = subset_indices
        self._tf = InstanceTransform(
            input_transform=input_transform,
            data_model=data_model,
            output_transform=output_transform,
        )
        self._tf_enabled = True
        self._is_iterable = isinstance(self._base_iterator, Iterable)
        self._supports_indexing = False
        self._supports_multi_indexing = False
