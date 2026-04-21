from typing import Any

from atria_logger import get_logger
from atria_transforms import DATA_TRANSFORM
from atria_transforms.core import DataTransform
from atria_types import DocumentInstance

logger = get_logger(__name__)


@DATA_TRANSFORM.register("transform_v1")
class TransformV1(DataTransform[Any]):
    train: bool = True

    def __call__(self, document: DocumentInstance) -> Any:
        assert isinstance(document, DocumentInstance), (
            "Input must be a DocumentInstance"
        )
        if self.train:
            # For training, we can apply some data augmentation or preprocessing
            # Here we just log the document and return it unchanged
            logger.info("TransformV1 (train mode) - no changes applied to document")
        else:
            # For evaluation, we might want to ensure the document is in a consistent format
            logger.info("TransformV1 (eval mode) - no changes applied to document")

        return document
