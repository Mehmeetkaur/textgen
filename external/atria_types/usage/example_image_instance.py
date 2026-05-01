"""Example: Creating and using ImageInstance (image classification/detection)."""

from atria_logger import get_logger
from PIL import Image as PILImage

from atria_types import (
    AnnotatedObject,
    BoundingBox,
    BoundingBoxMode,
    Image,
    ImageInstance,
    Label,
)
from atria_types._generic._annotations import (
    ClassificationAnnotation,
    ObjectDetectionAnnotation,
)

logger = get_logger(__name__)


def main() -> None:
    # Create a simple image instance for classification
    pil_img = PILImage.new("RGB", (800, 600), color="blue")

    img_instance = ImageInstance(
        sample_id="img_001",
        image=Image(
            file_path="/data/images/sample.jpg",
            content=pil_img,
        ),
        annotations=[ClassificationAnnotation(label=Label(value=1, name="cat"))],
    )

    logger.info("Example Image Instance:\n%s", img_instance)

    # Create an image instance with object detection annotations
    img_instance_with_det_annotations = ImageInstance(
        sample_id="img_002",
        image=Image(
            file_path="/data/images/street.jpg",
            content=PILImage.new("RGB", (1920, 1080)),
        ),
        annotations=[
            ObjectDetectionAnnotation(
                annotated_objects=[
                    AnnotatedObject(
                        label=Label(value=1, name="person"),
                        bbox=BoundingBox(
                            value=[100.0, 200.0, 250.0, 500.0],
                            mode=BoundingBoxMode.XYXY,
                        ),
                    ),
                    AnnotatedObject(
                        label=Label(value=2, name="car"),
                        bbox=BoundingBox(
                            value=[400.0, 300.0, 800.0, 600.0],
                            mode=BoundingBoxMode.XYXY,
                        ),
                    ),
                    AnnotatedObject(
                        label=Label(value=1, name="person"),
                        bbox=BoundingBox(
                            value=[1200.0, 250.0, 1350.0, 550.0],
                            mode=BoundingBoxMode.XYXY,
                        ),
                    ),
                ]
            ),
        ],
    )

    logger.info(
        "Example Image Instance with Object Annotations:\n%s",
        img_instance_with_det_annotations,
    )


if __name__ == "__main__":
    main()
