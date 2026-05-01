from collections.abc import Generator, Sequence

from atria_types import (
    AtriaDatasetConfig,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    SplitConfig,
)
from atria_types.factory import DocumentInstanceFactory

from atria_datasets.core.dataset._datasets import Dataset

TRAIN_SIZE = 100
TEST_SIZE = 20
VALIDATION_SIZE = 30
DATASET_SIZE = 100
NUM_LABELS = 10
LABELS = [f"class_{idx}" for idx in range(NUM_LABELS)]


class MockDocumentDatasetMixin:
    _REGISTRY_CONFIGS = [
        AtriaDatasetConfig(config_name="default", data_urls=["http://example.com"]),
        AtriaDatasetConfig(
            config_name="multiple_files",
            data_urls=["http://example.com", "http://example.com"],
        ),
    ]

    def _data_model(self) -> type[DocumentInstance]:
        return DocumentInstance

    def _metadata(self):
        return DatasetMetadata(
            homepage="http://example.com",
            description="Test dataset",
            license="MIT",
            citation="Test citation",
            dataset_labels=DatasetLabels(classification=LABELS),
        )

    def _split_configs(self, data_dir: str) -> list[SplitConfig]:
        return [
            SplitConfig(split=DatasetSplitType.train),
            SplitConfig(split=DatasetSplitType.validation),
        ]

    def _input_transform(self, sample: DocumentInstance) -> DocumentInstance:
        return sample


class MockDocumentIndexableDataset(MockDocumentDatasetMixin, Dataset[DocumentInstance]):
    def _split_iterator(
        self, split: DatasetSplitType, **kwargs
    ) -> Sequence[DocumentInstance]:
        size = TRAIN_SIZE if split == DatasetSplitType.train else VALIDATION_SIZE
        return DocumentInstanceFactory.build_batch(size)


class MockDocumentIterableDataset(MockDocumentDatasetMixin, Dataset[DocumentInstance]):
    def _split_iterator(
        self, split: DatasetSplitType, **kwargs
    ) -> Generator[DocumentInstance, None, None]:
        size = TRAIN_SIZE if split == DatasetSplitType.train else VALIDATION_SIZE
        yield from DocumentInstanceFactory.build_batch(size)
