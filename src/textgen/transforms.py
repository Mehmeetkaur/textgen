from typing import Any
import random
import numpy as np
from PIL import Image
from atria_types import DocumentInstance
from pydantic import BaseModel
from atria_logger import get_logger

logger = get_logger(__name__)

class TransformV1(BaseModel):
    train: bool = True
    seed: int = 42

    def __call__(self, document: DocumentInstance):


        random.seed(self.seed)

        content = getattr(document, "content", None)
        if not content or not content.text_elements:
            logger.warning("No text elements found")
            return None

        elements = content.text_elements

        elem = random.choice(elements)
        word = elem.text
        bbox = [float(x) for x in elem.bbox.value]

        image = document.image.content
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        w, h = image.size

        # normalize → pixel
        x0 = int(bbox[0] * w)
        y0 = int(bbox[1] * h)
        x1 = int(bbox[2] * w)
        y1 = int(bbox[3] * h)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        size = 256
        half = size // 2

        left = max(cx - half, 0)
        top = max(cy - half, 0)

        right = min(left + size, w)
        bottom = min(top + size, h)

        patch = image.crop((left, top, right, bottom)).resize((256, 256))

        mask = np.zeros((256, 256), dtype=np.uint8)

        bx0 = max(0, x0 - left)
        by0 = max(0, y0 - top)
        bx1 = min(256, x1 - left)
        by1 = min(256, y1 - top)

        mask[by0:by1, bx0:bx1] = 1
        mask = Image.fromarray(mask * 255)

        return {
            "image": patch,
            "mask": mask,
            "text": word
        }