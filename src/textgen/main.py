from __future__ import annotations

import sys
import os

from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
from atria_datasets import Dataset
from atria_datasets.core.dataset._exceptions import SplitNotFoundError
from atria_datasets.registry.ser.funsd import *  # noqa
from atria_logger import get_logger
from atria_types import DocumentInstance
from pydantic import BaseModel
from textgen.transforms import TransformV1


if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = get_logger(__name__)

# main.py or viz.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visualize_sample(sample: dict, output_path: Path, prefix: str, index: int):
    """
    Creates a side-by-side visualization:
      [patch image] | [mask overlaid on patch]
    with the text label annotated, and saves to output_path.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    patch: Image.Image = sample["image_pil"]   # ← was sample["image"]
    mask: Image.Image  = sample["mask_pil"]    # ← was sample["mask"]
    text: str          = sample["text"]

    # --- 1. Mask overlay: blend mask (red tint) over patch ---
    patch_rgb = patch.convert("RGB")
    mask_np   = np.array(mask.convert("L"))     # 0 or 255

    overlay = np.array(patch_rgb).copy()
    overlay[mask_np > 127] = (
        overlay[mask_np > 127] * 0.5
        + np.array([255, 0, 0]) * 0.5           # red highlight
    ).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # --- 2. Annotate text label on the patch ---
    # draw = ImageDraw.Draw(patch_rgb)


    # --- 3. Stitch side by side ---
    w, h = patch_rgb.size
    canvas = Image.new("RGB", (w * 2 + 10, h), color=(40, 40, 40))
    canvas.paste(patch_rgb,   (0, 0))
    canvas.paste(overlay_img, (w + 10, 0))

    save_path = output_path / f"{prefix}_viz_{index}.png"
    canvas.save(save_path)
    logger.info(f"Saved visualization → {save_path}")

def load_dataset(
    name: str,
    max_samples: int | None = None,
    data_dir: str | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 0,
    train_transform: BaseModel | None = None,
    eval_transform: BaseModel | None = None,
) -> Dataset:
    from atria_datasets import load_dataset_config

    dataset_config = load_dataset_config(
        name,
        max_train_samples=max_samples,
        max_test_samples=max_samples,
        max_validation_samples=max_samples,
    )
    logger.info(f"Loaded dataset config:\n{dataset_config}")

    if data_dir is not None:
        data_dir += "/" + name.split("/")[0]

    dataset = dataset_config.build(
        data_dir=data_dir,
        access_token=access_token,
        overwrite_existing_cached=overwrite_existing_cached,
        num_processes=num_processes,
        enable_cached_splits=True,
        max_cache_image_size=1024,
    )

    try:
        dataset.train.output_transform = train_transform
    except SplitNotFoundError:
        logger.warning(
            "Train split not found in dataset, skipping train transform assignment."
        )

    try:
        dataset.validation.output_transform = eval_transform
    except SplitNotFoundError:
        logger.warning(
            "Validation split not found in dataset, skipping validation transform assignment."
        )

    try:
        dataset.test.output_transform = TransformV1(train=False)
    except SplitNotFoundError:
        logger.warning(
            "Test split not found in dataset, skipping test transform assignment."
        )

    return dataset


def load_transform(train: bool = True, patch_size: int = 256) -> TransformV1:
    transform = TransformV1(train=train, patch_size=patch_size)
    logger.info(f"Loaded transform: {transform}")
    return transform


def build_dataloader(
    dataset_split: Any,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset_split,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x,
    )


def iterate_dataset(
    name: str,
    batch_size: int = 4,
    data_dir: str | None = None,
    max_samples: int | None = None,
    access_token: str | None = None,
    overwrite_existing_cached: bool = False,
    num_processes: int = 0,
    output_dir: str = "./output",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        name=name,
        max_samples=max_samples,
        data_dir=data_dir,
        access_token=access_token,
        overwrite_existing_cached=overwrite_existing_cached,
        num_processes=num_processes,
        train_transform=load_transform(train=True),
        eval_transform=load_transform(train=False),
    )

    # visualize first sample in the train dataset
    for sample_index, sample in enumerate(dataset.train):
        logger.info(
        "TRAIN_SAMPLE | idx=%d | text=%s | keys=%s",
        sample_index,
        sample["text"],
        list(sample.keys())
        )

        image = sample["image_pil"]
        mask = sample["mask_pil"]
        logger.info(f"Image size: {image.size}")
        logger.info(f"Mask size: {mask.size}")
        (output_path / "train").mkdir(parents=True, exist_ok=True)
        image.save(output_path / "train" / f"image_{sample_index}.png")
        mask.save(output_path / "train" / f"mask_{sample_index}.png")

        visualize_sample(sample, output_path / "train", prefix="train", index=sample_index)
        break


    for sample_index, sample in enumerate(dataset.test):
        logger.info(
        "TEST_SAMPLE | idx=%d | keys=%s",
        sample_index,
        list(sample.keys()) if isinstance(sample, dict) else type(sample)
        )

        if isinstance(sample, dict):
            image = sample["image_pil"]
            mask = sample["mask_pil"]
            (output_path / "test").mkdir(parents=True, exist_ok=True)
            image.save(output_path / "test" / f"image_{sample_index}.png")
            mask.save(output_path / "test" / f"mask_{sample_index}.png")

        else:

            if hasattr(sample, "viz"):
               sample.viz.visualize(output_path / "test" / f"sample_{sample_index}.png")

        break

    train_dataloader = build_dataloader(
        dataset.train, batch_size=batch_size, shuffle=True
    )
    test_dataloader = build_dataloader(
        dataset.test, batch_size=batch_size, shuffle=False
    )

    for batch_index, batch in enumerate(train_dataloader):
        logger.info(
            "Processing batch %s with %s samples", batch_index, len(batch)
        )
        logger.info("Sample 1 in batch: %s", batch[0])
        break

    for batch_index, batch in enumerate(test_dataloader):
        logger.info(
            "Processing batch %s with %s samples", batch_index, len(batch)
        )
        logger.info("Sample 1 in batch: %s", batch[0])
        break

# ══════════════════════════════════════════════════════════════════════
#  `train` command  — Step 6: text-conditioned diffusion
# ══════════════════════════════════════════════════════════════════════

def train_diffusion(
    name: str,
    image_size:     int   = 256,
    batch_size:     int   = 8,
    num_epochs:     int   = 100,
    lr:             float = 1e-4,
    noise_steps:    int   = 1000,
    model_channels: int   = 128,
    context_dim:    int   = 128,
    num_heads:      int   = 4,
    data_dir:       str | None = None,
    max_samples:    int | None = None,
    access_token:   str | None = None,
    output_dir:     str   = "./output/diffusion",
    device:         str | None = None,
    save_every:     int   = 10,
    sample_every:   int   = 10,
):
    """
    Train the text-conditioned diffusion inpainting model (Step 6).

    What happens when you run this
    --------------------------------
    1. The FUNSD dataset is loaded and TransformV1 is attached to each split.
       Every sample becomes a dict:
           { image: Tensor[3,256,256], mask: Tensor[1,256,256], text: "word" }

    2. DiffusionTrainer builds:
           - UNetModel   : 7-channel input UNet with CharacterEncoder + cross-attention
           - EMA copy    : smoothed model weights used for sampling
           - GaussianDiffusion : DDPM schedule (T=noise_steps steps)
           - AdamW + CosineAnnealingLR

    3. Each training step:
           a. Convert word string → character index tensor  [B, 20]
           b. CharacterEncoder → context sequence           [B, 20, context_dim]
           c. Add noise to image at random timestep t
           d. Build 7-ch input: [noisy | masked_original | mask]
           e. UNet predicts noise
           f. MSE loss (masked region only) → backprop → step → EMA update

    4. Every `save_every` epochs  : checkpoint saved to output_dir/checkpoints/
       Every `sample_every` epochs: sample grid saved to output_dir/samples/
           Grid columns: [masked input | generated | ground truth]

    Args:
        name            : atria dataset name, e.g. "funsd/ser"
        batch_size      : samples per gradient step (reduce if OOM)
        num_epochs      : total training epochs
        lr              : AdamW learning rate (1e-4 works well)
        noise_steps     : DDPM diffusion steps T (1000 is standard)
        model_channels  : UNet base channel width (128 for faster training,
                          256 for better quality)
        context_dim     : character embedding dimension — must be divisible
                          by num_heads (128 / 4 = 32 per head)
        num_heads       : attention heads in SpatialTransformer blocks
        data_dir        : local directory for downloaded dataset files
        max_samples     : cap samples per split (useful for quick smoke-tests)
        access_token    : HuggingFace token for gated datasets
        output_dir      : directory for checkpoints and sample images
        device          : "cuda" or "cpu" (auto-detected if None)
        save_every      : save checkpoint every N epochs
        sample_every    : generate visual samples every N epochs

    Example:
        # Quick smoke-test (fast, low quality)
        python -m textgen.main train --name funsd/ser --max_samples 50 \\
            --num_epochs 5 --batch_size 4 --sample_every 1

        # Full training run
        python -m textgen.main train --name funsd/ser \\
            --num_epochs 200 --batch_size 8 --model_channels 128
    """
    from textgen.trainer import DiffusionTrainer

    # ── Load dataset with transforms ──────────────────────────────────
    dataset = load_dataset(
        name=name,
        max_samples=max_samples,
        data_dir=data_dir,
        access_token=access_token,
        train_transform=load_transform(train=True,  patch_size=image_size),
        eval_transform= load_transform(train=False, patch_size=image_size),
    )

    # Validation split is optional — fall back gracefully
    val_split = None
    try:
        val_split = dataset.validation
        logger.info("Using validation split for sample generation.")
    except (SplitNotFoundError, AttributeError):
        logger.warning(
            "No validation split found — will use train split for sample visualisation."
        )
        print("\n===== train_diffusion =====")
        print("noise_steps:", noise_steps)
        print("type:", type(noise_steps))
        print("device:", device)
        print("===========================\n")
    # ── Build trainer and start training ─────────────────────────────
    trainer = DiffusionTrainer(
        image_size      = image_size,
        dataset_train   = dataset.train,
        dataset_val     = val_split,
        batch_size      = batch_size,
        lr              = lr,
        num_epochs      = num_epochs,
        noise_steps     = noise_steps,
        model_channels  = model_channels,
        context_dim     = context_dim,
        num_heads       = num_heads,
        output_dir      = output_dir,
        device          = device,
        save_every      = save_every,
        sample_every    = sample_every,
    )
    trainer.train()


# ══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    """
    Exposes two subcommands via python-fire:

        iterate  — verify dataset + TransformV1 (Steps 2 & 4)
        train    — train the diffusion model     (Step 6)

    Usage:
        python -m textgen.main iterate --name funsd/ser
        python -m textgen.main train   --name funsd/ser --num_epochs 100
    """
    fire.Fire({
        "iterate": iterate_dataset,
        "train":   train_diffusion,
    })


if __name__ == "__main__":
    main()
