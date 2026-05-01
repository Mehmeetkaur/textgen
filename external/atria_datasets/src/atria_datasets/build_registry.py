# from atria_datasets.pipelines.atria_data_pipeline import *  # noqa
from atria_datasets.registry import DATASET
from atria_datasets.registry.document_classification.rvlcdip import *  # noqa  # noqa
from atria_datasets.registry.document_classification.tobacco3482 import *  # noqa  # noqa
from atria_datasets.registry.image_classification.cifar10 import *  # noqa
from atria_datasets.registry.image_classification.cifar10_huggingface import *  # noqa
from atria_datasets.registry.layout_analysis.doclaynet import *  # noqa
from atria_datasets.registry.layout_analysis.icdar2019 import *  # noqa
from atria_datasets.registry.layout_analysis.publaynet import *  # noqa
from atria_datasets.registry.ser.cord import *  # noqa
from atria_datasets.registry.ser.docbank import *  # noqa
from atria_datasets.registry.ser.docile import *  # noqa
from atria_datasets.registry.ser.funsd import *  # noqa
from atria_datasets.registry.ser.sroie import *  # noqa
from atria_datasets.registry.ser.wild_receipts import *  # noqa
from atria_datasets.registry.table_extraction.fintabnet import *  # noqa
from atria_datasets.registry.table_extraction.icdar2013 import *  # noqa
from atria_datasets.registry.table_extraction.pubtables1m import *  # noqa
from atria_datasets.registry.vqa.due import *  # noqa


def main():
    DATASET.dump()


if __name__ == "__main__":
    main()
