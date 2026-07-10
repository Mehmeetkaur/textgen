# src/textgen/transforms.py

from typing import Any
import random
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch
from atria_types import DocumentInstance
from pydantic import BaseModel
from atria_logger import get_logger

logger = get_logger(__name__)


def pad_image_to_square(image: Image.Image, target_size: int, fill: int = 127) -> Image.Image | None:
    """
    Paste the image at the top-left of a (target_size x target_size) canvas.
    Returns None if the image is already larger than target_size in any dimension.

    Why fill=127?  A mid-gray value sits at 0.0 after our [-1,1] normalisation
    (127/255 * 2 - 1 ≈ 0), so the padded region contributes no signal to the
    diffusion model — it is effectively neutral / invisible.
    """
    w, h = image.size
    if w > target_size or h > target_size:
        return None                                      # caller will skip this word
    canvas = Image.new(image.mode, (target_size, target_size), fill)
    canvas.paste(image, (0, 0))                          # top-left placement
    return canvas


def pad_mask_to_square(mask_np: np.ndarray, target_size: int) -> np.ndarray | None:
    """
    Embed a 2-D mask (H x W, values 0 or 255) in a (target_size x target_size)
    zero array.  The padded region is 0 (background), so the diffusion model is
    never asked to inpaint there.
    Returns None if the mask is already larger than target_size.
    """
    h, w = mask_np.shape
    if w > target_size or h > target_size:
        return None
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    canvas[:h, :w] = mask_np                             # top-left placement
    return canvas


class TransformV1(BaseModel):
    """
    Converts a DocumentInstance into a dict with three keys:
        "image"     : torch.Tensor [3, 256, 256] in [-1, 1]
        "mask"      : torch.Tensor [1, 256, 256] binary {0, 1}
        "text"      : str  — the word whose region is masked
        "image_pil" : PIL Image (for visualisation only)
        "mask_pil"  : PIL Image (for visualisation only)

    Steps
    -----
    1. Pick a random word element from the document.
    2. Convert its normalised bounding box to pixel coordinates.
    3. Add context padding around the word box and crop that region.
    4. Build a binary mask — 1 where the word is, 0 everywhere else.
    5. Pad both crop and mask to patch_size × patch_size (no resizing).
    6. Convert to tensors.
    """

    model_config = {"arbitrary_types_allowed": True}

    patch_size: int = 256   # output spatial size (both H and W)
    padding: int = 16       # context pixels added around the word box
    train: bool = True

    def __call__(self, document: DocumentInstance) -> Any:
        content = getattr(document, "content", None)
        if not content or not content.text_elements:
            logger.warning("No text elements found in document.")
            return None

        # Shuffle so we get a truly random word each call
        elements = list(content.text_elements)
        random.shuffle(elements)

        for elem in elements:
            result = self._process_element(elem, document)
            if result is not None:
                return result

        logger.warning("No word patch fit within patch_size=%d. Skipping document.", self.patch_size)
        return None

    # ------------------------------------------------------------------
    def _process_element(self, elem, document) -> dict | None:
        word = elem.text
        if not word or not word.strip():
            return None

        bbox = [float(x) for x in elem.bbox.value]

        # ── Load image ────────────────────────────────────────────────
        image = document.image.content
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        W, H = image.size

        # ── Normalised bbox → pixel coords ────────────────────────────
        # bbox format: [x0, y0, x1, y1] all in [0, 1]
        x0 = int(bbox[0] * W)
        y0 = int(bbox[1] * H)
        x1 = int(bbox[2] * W)
        y1 = int(bbox[3] * H)

        # ── Add context padding (clamped to image borders) ────────────
        cx0 = max(0, x0 - self.padding)
        cy0 = max(0, y0 - self.padding)
        cx1 = min(W,  x1 + self.padding)
        cy1 = min(H,  y1 + self.padding)

        patch = image.crop((cx0, cy0, cx1, cy1))
        pw, ph = patch.size

        # ── Build binary mask (same size as raw patch) ────────────────
        # The word box, expressed relative to the top-left of the patch
        bx0 = x0 - cx0   # how far the word starts inside the patch
        by0 = y0 - cy0
        bx1 = x1 - cx0
        by1 = y1 - cy0

        # Clamp — should never be needed but protects against rounding
        bx0, bx1 = max(0, bx0), min(pw, bx1)
        by0, by1 = max(0, by0), min(ph, by1)

        mask_np = np.zeros((ph, pw), dtype=np.uint8)
        mask_np[by0:by1, bx0:bx1] = 255   # white = masked region

        # ── Pad to patch_size × patch_size (NO resizing) ─────────────
        # patch_padded = pad_image_to_square(patch, self.patch_size, fill=127)
        # mask_padded  = pad_mask_to_square(mask_np, self.patch_size)

        # if patch_padded is None or mask_padded is None:
        #     logger.debug(
        #         "Word '%s' patch (%dx%d) exceeds patch_size=%d — skipping.",
        #         word, pw, ph, self.patch_size,
        #     )
        #     return None

        # mask_img = Image.fromarray(mask_padded)
        # ── Resize crop and mask to patch_size × patch_size ─────────────
        mask_img = Image.fromarray(mask_np)
        patch_padded = patch.resize(
            (self.patch_size, self.patch_size),
            Image.BICUBIC,
            )
        mask_img = mask_img.resize(
            (self.patch_size, self.patch_size),
            Image.NEAREST,
            )





        # ── Convert to tensors ────────────────────────────────────────
        # Image: [3, H, W] in [0,1]  →  scale to [-1, 1]
        image_tensor = TF.to_tensor(patch_padded.convert("RGB"))
        image_tensor = image_tensor * 2.0 - 1.0           # [-1, 1]

        # Mask: [1, H, W] in [0,1]  →  threshold to {0, 1}
        mask_tensor = TF.to_tensor(mask_img)
        mask_tensor = (mask_tensor > 0.5).float()          # binary

        logger.info(
            "TransformV1: word='%s' | raw patch=%dx%d | padded to %dx%d",
            word, pw, ph, self.patch_size, self.patch_size,
        )

        return {
            "image":     image_tensor,   # torch.Tensor [3, 256, 256]
            "mask":      mask_tensor,    # torch.Tensor [1, 256, 256]
            "text":      word,           # str
            "image_pil": patch_padded,   # PIL Image (for visualisation)
            "mask_pil":  mask_img,       # PIL Image (for visualisation)
        }