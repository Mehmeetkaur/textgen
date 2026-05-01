"""Example: Creating and using AnnotatedObject (object detection)."""

from atria_logger import get_logger

from atria_types import AnnotatedObject, BoundingBox, BoundingBoxMode, Label

logger = get_logger(__name__)


def main() -> None:
    # Create an annotated object (e.g., detected person)
    person = AnnotatedObject(
        label=Label(value=1, name="person"),
        bbox=BoundingBox(
            value=[100.0, 150.0, 300.0, 400.0],
            mode=BoundingBoxMode.XYXY,
            normalized=False,
        ),
        iscrowd=False,
    )

    # Log details
    logger.info("Example Annotation:\n%s", person)

    # Create multiple annotated objects
    annotated_objects = [
        AnnotatedObject(
            label=Label(value=1, name="person"),
            bbox=BoundingBox(
                value=[50.0, 60.0, 150.0, 300.0], mode=BoundingBoxMode.XYXY
            ),
        ),
        AnnotatedObject(
            label=Label(value=2, name="car"),
            bbox=BoundingBox(
                value=[200.0, 250.0, 450.0, 400.0], mode=BoundingBoxMode.XYXY
            ),
        ),
        AnnotatedObject(
            label=Label(value=1, name="person"),
            bbox=BoundingBox(
                value=[500.0, 100.0, 600.0, 350.0], mode=BoundingBoxMode.XYXY
            ),
            segmentation=[[500.0, 100.0, 600.0, 100.0, 600.0, 350.0, 500.0, 350.0]],
        ),
    ]

    logger.info("All annotations:\n%s", annotated_objects)

    # Convert to dict
    bbox_dict = person.model_dump()
    logger.info("Serialized: %s", bbox_dict)

    # Serialize to JSON
    bbox_json = person.model_dump_json()
    logger.info("As JSON: %s", bbox_json)


if __name__ == "__main__":
    main()
