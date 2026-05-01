from collections.abc import Generator, Sequence

from atria_types import (
    AtriaDatasetConfig,
    BaseDataInstance,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    Image,
    ImageInstance,
    Label,
    SplitConfig,
)

from atria_datasets.core.dataset._datasets import Dataset

TRAIN_SIZE = 100
TEST_SIZE = 20
VALIDATION_SIZE = 30
DATASET_SIZE = 100
NUM_LABELS = 10
LABELS = [f"class_{idx}" for idx in range(NUM_LABELS)]


class MockImageDatasetMixin(Dataset[BaseDataInstance]):
    _REGISTRY_CONFIGS = [
        AtriaDatasetConfig(config_name="default", data_urls=["http://example.com"]),
        AtriaDatasetConfig(
            config_name="multiple_files",
            data_urls=["http://example.com", "http://example.com"],
        ),
    ]

    def _data_model(self) -> type[BaseDataInstance]:
        return ImageInstance

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

    def _input_transform(self, sample: Sequence[tuple[Image, int]]) -> ImageInstance:
        image_instance = ImageInstance(
            image=Image(content=sample[0]),
            label=Label(
                index=sample[1],
                name=self.metadata.dataset_labels.instance_classification[sample[1]],
            ),
        ).load()
        return image_instance


class MockImageIndexableDataset(MockImageDatasetMixin, Dataset[ImageInstance]):
    def _split_iterator(
        self, split: DatasetSplitType, **kwargs
    ) -> Sequence[tuple[Image, int]]:
        from PIL import Image

        size = TRAIN_SIZE if split == DatasetSplitType.train else VALIDATION_SIZE
        mock_image = Image.new("RGB", (32, 32))
        return [(mock_image, i // 10) for i in range(size)]


class MockImageIterableDataset(MockImageDatasetMixin, Dataset[ImageInstance]):
    def _split_iterator(  # type: ignore
        self, split: DatasetSplitType, **kwargs
    ) -> Generator[tuple[Image, int], None, None]:
        from PIL import Image

        mock_image = Image.new("RGB", (32, 32))
        size = TRAIN_SIZE if split == DatasetSplitType.train else VALIDATION_SIZE
        for i in range(size):
            yield (mock_image, i // 10)
