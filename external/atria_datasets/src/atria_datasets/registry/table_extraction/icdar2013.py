import os
from collections.abc import Generator
from pathlib import Path

from atria_logger import get_logger
from atria_types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentContent,
    DocumentInstance,
    Image,
    LayoutAnalysisAnnotation,
)

from atria_datasets import DATASET, DocumentDataset
from atria_datasets.core.dataset._common import DatasetConfig

from .utilities import read_pascal_voc, read_words_json

logger = get_logger(__name__)

_CITATION = """\
@software{smock2021tabletransformer,
    author = {Smock, Brandon and Pesala, Rohith},
    month = {06},
    title = {{Table Transformer}},
    url = {https://github.com/microsoft/table-transformer},
    version = {1.0.0},
    year = {2021}
}
"""

_DESCRIPTION = """\
ICDAR-2013: Towards comprehensive table extraction from unstructured documents.
"""

_HOMEPAGE = "https://github.com/microsoft/table-transformer/"

_LICENSE = "https://github.com/microsoft/table-transformer/blob/main/LICENSE"

_URLS = [
    "https://huggingface.co/datasets/bsmock/ICDAR-2013.c/resolve/main/ICDAR-2013.c-Structure.tar.gz",
    "https://huggingface.co/datasets/bsmock/ICDAR-2013.c/resolve/main/ICDAR-2013.c-PDF_Annotations.tar.gz",
]

_CLASSES = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
]


class SplitIterator:
    def __init__(self, split: DatasetSplitType, data_dir: str):
        self.split = split
        self.data_dir = Path(data_dir)

        base_path = self.data_dir / "ICDAR-2013.c-Structure" / "ICDAR-2013.c-Structure"

        if split == DatasetSplitType.test:
            split_path = "test"
        elif split == DatasetSplitType.validation:
            split_path = "val"

        self.xmls_dir = base_path / split_path
        self.images_dir = base_path / "images"
        self.words_dir = base_path / "words"

    def __iter__(self) -> Generator[DocumentInstance, None, None]:
        xml_filenames = [
            elem for elem in os.listdir(self.xmls_dir) if elem.endswith(".xml")
        ]
        for filename in xml_filenames:
            xml_filepath = self.xmls_dir / filename
            image_file_path = self.images_dir / filename.replace(".xml", ".jpg")
            word_file_path = self.words_dir / filename.replace(".xml", "_words.json")
            yield image_file_path, xml_filepath, word_file_path

    def __len__(self) -> int:
        xml_filenames = [
            elem for elem in os.listdir(self.xmls_dir) if elem.endswith(".xml")
        ]
        return len(xml_filenames)


class ICDAR2013Config(DatasetConfig):
    dataset_name: str = "icdar2013"


@DATASET.register("icdar2013")
class ICDAR2013(DocumentDataset):
    __config__ = ICDAR2013Config

    def _download_urls(self) -> list[str]:
        return _URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(layout=_CLASSES),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [DatasetSplitType.validation, DatasetSplitType.test]

    def _split_iterator(
        self, split: DatasetSplitType, data_dir: str
    ) -> Generator[DocumentInstance, None, None]:
        return SplitIterator(split=split, data_dir=data_dir)

    def _input_transform(self, inputs: tuple[Path, Path, Path]) -> DocumentInstance:
        image_file_path, xml_filepath, word_file_path = inputs
        image = Image(file_path=image_file_path).load()
        annotated_objects = read_pascal_voc(
            xml_filepath,
            labels=_CLASSES,
            image_width=image.width,
            image_height=image.height,
        )
        text_elements = read_words_json(
            word_file_path, image_width=image.width, image_height=image.height
        )

        return DocumentInstance(
            sample_id=Path(image_file_path).name,
            image=image,
            content=DocumentContent(text_elements=text_elements),
            annotations=[LayoutAnalysisAnnotation(annotated_objects=annotated_objects)],
        )
