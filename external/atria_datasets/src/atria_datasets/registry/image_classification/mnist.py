from typing import Any

from atria_types import ClassificationAnnotation, Image, ImageInstance, Label

from atria_datasets import DATASET, HuggingfaceImageDataset
from atria_datasets.core.dataset._hf_datasets import HuggingfaceDatasetConfig


@DATASET.register(
    "mnist",
    configs={
        "default": HuggingfaceDatasetConfig(
            dataset_name="mnist",
            config_name="default",
            hf_repo="ylecun/mnist",
            hf_config_name="mnist",
        ),
        "1k": HuggingfaceDatasetConfig(
            dataset_name="mnist",
            config_name="1k",
            hf_repo="ylecun/mnist",
            hf_config_name="mnist",
            max_train_samples=1000,
            max_test_samples=1000,
            max_validation_samples=1000,
        ),
    },
)
class MNIST(HuggingfaceImageDataset):
    def _input_transform(self, sample: dict[str, Any]) -> ImageInstance:
        return ImageInstance(
            image=Image(content=sample["image"]),
            annotations=[
                ClassificationAnnotation(
                    label=Label(
                        value=sample["label"],
                        name=self.metadata.dataset_labels.classification[
                            sample["label"]
                        ],
                    )
                )
            ],
        )
