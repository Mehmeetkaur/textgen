from __future__ import annotations

import random
from typing import TYPE_CHECKING

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

if TYPE_CHECKING:
    from atria_types._generic._bounding_box import BoundingBox
    from atria_types._generic._image import Image

# Color palette for bounding boxes
_COLOR_PALETTE = [
    (255, 99, 71),  # tomato
    (255, 140, 0),  # dark orange
    (255, 165, 0),  # orange
    (255, 69, 0),  # red-orange
    (255, 215, 0),  # gold
    (255, 182, 80),  # light orange
]

# Semi-transparent black background for text labels
_TEXT_BACKGROUND_COLOR = (0, 0, 0, 180)

# Text styling constants
_TEXT_COLOR = (255, 255, 255, 255)  # White
_TEXT_PADDING = 3
_LABEL_OFFSET = 4
_MIN_FONT_SIZE = 12
_FONT_SIZE_RATIO = 0.015
_BBOX_WIDTH = 2


def _create_label_color_mapping(
    labels: list[str] | None,
) -> dict[str, tuple[int, int, int]]:
    """Create a consistent color mapping for unique labels.

    Args:
        labels: List of label strings.

    Returns:
        Dictionary mapping each unique label to an RGB color tuple.
    """
    if not labels:
        return {}

    unique_labels = set(labels)
    return {
        label: _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]
        for idx, label in enumerate(unique_labels)
    }


def _calculate_adaptive_font_size(image_height: int) -> int:
    """Calculate font size based on image dimensions.

    Args:
        image_height: Height of the image in pixels.

    Returns:
        Font size as an integer.
    """
    return max(_MIN_FONT_SIZE, int(image_height * _FONT_SIZE_RATIO))


def _get_text_dimensions(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    """Get text dimensions with fallback for older Pillow versions.

    Args:
        draw: ImageDraw object.
        text: Text to measure.
        font: Font to use for measurement.

    Returns:
        Tuple of (width, height) in pixels.
    """
    try:
        # Modern Pillow API
        text_bbox = draw.textbbox((0, 0), text, font=font)
        width = text_bbox[2] - text_bbox[0]
        height = text_bbox[3] - text_bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        width, height = font.getsize(text)  # type: ignore

    return width, height


def _get_bbox_color(
    label: str | None,
    label_to_color: dict[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Determine color for a bounding box based on its label.

    Args:
        label: Label string or None.
        label_to_color: Mapping of labels to colors.

    Returns:
        RGB color tuple.
    """
    if label and label in label_to_color:
        return label_to_color[label]

    return random.choice(_COLOR_PALETTE)


def _draw_bbox_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: BoundingBox,
    color: tuple[int, int, int],
) -> bool:
    """Draw a bounding box rectangle.

    Args:
        draw: ImageDraw object.
        bbox: BoundingBox to draw.
        color: RGB color tuple.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Add alpha channel for the outline color
        outline_color = color + (255,)
        draw.rectangle(bbox.value, outline=outline_color, width=_BBOX_WIDTH)
        return True
    except Exception as e:
        print(f"Error drawing bounding box {bbox}: {e}")
        return False


def _draw_text_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: BoundingBox,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw a text label above a bounding box with background.

    Args:
        draw: ImageDraw object.
        text: Text to draw.
        bbox: BoundingBox to position the label relative to.
        font: Font to use for the text.
    """
    # Calculate text dimensions
    text_width, text_height = _get_text_dimensions(draw, text, font)

    # Position text above the bounding box
    text_x = bbox.value[0]
    text_y = max(0, bbox.value[1] - text_height - _LABEL_OFFSET)

    # Draw semi-transparent background behind text
    background_bbox = [
        text_x,
        text_y,
        text_x + text_width + (2 * _TEXT_PADDING),
        text_y + text_height + _LABEL_OFFSET,
    ]
    draw.rectangle(background_bbox, fill=_TEXT_BACKGROUND_COLOR)

    # Draw white text on top
    text_position = (text_x + _TEXT_PADDING, text_y + 2)
    draw.text(text_position, text, fill=_TEXT_COLOR, font=font)


def _draw_bboxes_on_image(
    image: Image,
    bboxes: list[BoundingBox],
    bboxes_text: list[str] | None = None,
    bbox_labels: list[str] | None = None,
) -> PILImage.Image:
    """Draw bounding boxes with warm colors and readable transparent labels.

    Args:
        image: Input Image object to draw on.
        bboxes: List of BoundingBox objects to draw.
        bbox_labels: Optional list of label strings for each bounding box.

    Returns:
        PIL Image with bounding boxes drawn.
    """
    # Create color mapping for labels
    label_to_color = _create_label_color_mapping(bbox_labels)

    # Prepare image for drawing
    draw = ImageDraw.Draw(image, "RGBA")

    # Setup font
    font = ImageFont.load_default()

    # Draw each bounding box
    for index, bbox in enumerate(bboxes):
        # Get label and color
        text = bboxes_text[index] if bboxes_text else None
        label = bbox_labels[index] if bbox_labels else None
        color = _get_bbox_color(label, label_to_color)

        # Draw the bounding box rectangle
        if not _draw_bbox_rectangle(draw, bbox, color):
            continue

        # Draw label if available
        if label is not None:
            if text is not None:
                label = f"{label}: {text}"
            _draw_text_label(draw, label, bbox, font)
        elif text is not None:
            _draw_text_label(draw, text, bbox, font)

    return image
