from __future__ import annotations

from typing import TYPE_CHECKING

from PIL.Image import Image as PILImage

from atria_types._data_instance._visualizers._base import Visualizer
from atria_types._utilities._viz import _draw_bboxes_on_image

if TYPE_CHECKING:
    from atria_types._data_instance._document_instance import DocumentInstance


class DocumentVisualizer(Visualizer):
    def __init__(self, instance: DocumentInstance) -> None:
        self.instance = instance

    def _draw_on_image(self, image: PILImage) -> PILImage:
        # Placeholder for drawing logic specific to the data instance type
        if len(self.instance.content.bbox_list) > 0:
            bboxes = [
                bbox.ops.unnormalize(width=image.width, height=image.height)
                for bbox in self.instance.content.bbox_list
                if bbox is not None
            ]

            # Draw bounding boxes on the image
            image = _draw_bboxes_on_image(
                image=image,
                bboxes=bboxes,
                bboxes_text=self.instance.content.text_list,
            )
        return image
