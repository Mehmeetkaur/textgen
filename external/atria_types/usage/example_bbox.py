"""Example: Creating and using BoundingBox objects."""

from atria_logger import get_logger

from atria_types import BoundingBox, BoundingBoxMode

logger = get_logger(__name__)


def main() -> None:
    # Create a bounding box in XYXY format (x1, y1, x2, y2)
    bbox_xyxy = BoundingBox(
        value=[100.0, 150.0, 300.0, 400.0],
        mode=BoundingBoxMode.XYXY,
        normalized=False,
    )
    logger.info("XYXY BBox: %s", bbox_xyxy)

    # Create a bounding box in XYWH format (x, y, width, height)
    bbox_xywh = BoundingBox(
        value=[100.0, 150.0, 200.0, 250.0],
        mode=BoundingBoxMode.XYWH,
        normalized=False,
    )
    logger.info("XYWH BBox: %s", bbox_xywh)

    # Create a normalized bounding box (coordinates in [0, 1])
    bbox_normalized = bbox_xyxy.ops.normalize(width=400, height=600)
    logger.info("Normalized BBox: %s", bbox_normalized)

    # Use operations
    bbox_converted = bbox_normalized.ops.switch_mode()
    logger.info("Converted to XYWH: %s", bbox_converted)

    # Convert to dict
    bbox_dict = bbox_normalized.model_dump()
    logger.info("Serialized: %s", bbox_dict)

    # Serialize to JSON
    bbox_json = bbox_normalized.model_dump_json()
    logger.info("As JSON: %s", bbox_json)


if __name__ == "__main__":
    main()
