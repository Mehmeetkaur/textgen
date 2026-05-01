"""Example: Creating and using Image objects."""

from atria_logger import get_logger
from PIL import Image as PILImage

from atria_types import Image

logger = get_logger(__name__)


def main() -> None:
    # Image with both path and content
    image_full = Image(
        file_path="/path/to/original.jpg",
        content=PILImage.new("RGB", (800, 600), color="red"),
        source_width=1920,
        source_height=1080,
    )
    logger.info("Full Image:\n%s", image_full)

    # Convert to dict
    image_dict = image_full.model_dump()
    logger.info("Serialized:")
    logger.info(image_dict)

    # Use image operations
    resized = image_full.ops.resize(width=400, height=300)
    logger.info("Resized Image:\n%s", resized)

    grayscale = image_full.ops.to_grayscale()
    logger.info("Grayscale Image:\n%s", grayscale)


if __name__ == "__main__":
    main()
