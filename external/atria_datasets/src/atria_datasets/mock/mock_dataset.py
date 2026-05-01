from collections.abc import Generator
from typing import Any

from atria_types import (
    AtriaDatasetConfig,
    BaseDataInstance,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    SplitConfig,
)

from atria_datasets.core.dataset._datasets import Dataset

DATASET_SIZE = 100
NUM_LABELS = 10
LABELS = [f"class_{idx}" for idx in range(NUM_LABELS)]


class MockAtriaIndexableDataset(Dataset[BaseDataInstance]):
    _REGISTRY_CONFIGS = [
        AtriaDatasetConfig(dataset_name="default", data_urls=["http://example.com"]),
        AtriaDatasetConfig(
            dataset_name="multiple_files",
            data_urls={
                "sample-1": "https://getsamplefiles.com/download/zip/sample-1.zip",
                "sample-2": "https://getsamplefiles.com/download/zip/sample-2.zip",
            },
        ),
    ]

    def _data_model(self) -> type[BaseDataInstance]:
        return BaseDataInstance

    def _metadata(self):
        return DatasetMetadata(
            dataset_name=self.__class__.__name__,
            config=self._config,
            citation="Test citation",
            homepage="http://example.com",
            license="MIT",
            dataset_labels=DatasetLabels(classification=LABELS),
        )

    def _split_configs(self):
        return [SplitConfig(split=DatasetSplitType.train, gen_kwargs={})]

    def _split_iterator(self, split: DatasetSplitType, **kwargs):
        return [{"index": index} for index in range(DATASET_SIZE)]

    def _input_transform(self, sample: dict[str, Any]) -> BaseDataInstance:
        return BaseDataInstance(index=sample["index"])


class MockAtriaIterableDataset(Dataset[BaseDataInstance]):
    _REGISTRY_CONFIGS = [
        AtriaDatasetConfig(dataset_name="default", data_urls=["http://example.com"]),
        AtriaDatasetConfig(
            dataset_name="multiple_files",
            data_urls={
                "sample-1": "https://getsamplefiles.com/download/zip/sample-1.zip",
                "sample-2": "https://getsamplefiles.com/download/zip/sample-2.zip",
            },
        ),
    ]

    def _data_model(self) -> type[BaseDataInstance]:
        return BaseDataInstance

    def _metadata(self):
        return DatasetMetadata(
            homepage="http://example.com",
            description="Test dataset",
            license="MIT",
            citation="Test citation",
            dataset_labels=DatasetLabels(classification=LABELS),
        )

    def _split_configs(self):
        return [SplitConfig(split=DatasetSplitType.train, gen_kwargs={})]

    def _split_iterator(
        self, split: DatasetSplitType, **kwargs
    ) -> Generator[Any, None, None]:
        for index in range(DATASET_SIZE):
            yield {"index": index}

    def _input_transform(self, sample: dict[str, Any]) -> BaseDataInstance:
        return BaseDataInstance(index=sample["index"])
