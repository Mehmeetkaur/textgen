# src/textgen/trainer.py
"""
Training loop for text-conditioned diffusion inpainting.

Follows WordStylist's train.py structure:
  - Same character vocabulary (a-z, A-Z + PAD token)
  - Same label_padding approach to fixed-length character sequences
  - EMA (Exponential Moving Average) of model weights for stable sampling
  - AdamW optimiser with cosine LR schedule
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from atria_logger import get_logger

from textgen.unet import UNetModel, CharacterEncoder
from textgen.diffusion import GaussianDiffusion

logger = get_logger(__name__)


#  Character vocabulary  (identical to WordStylist)

# All printable ASCII letters — same set WordStylist uses for IAM
C_CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTER2INDEX = {c: i for i, c in enumerate(C_CLASSES)}
PAD_TOKEN = len(C_CLASSES)  # index 52
VOCAB_SIZE = len(C_CLASSES) + 1  # 53  (letters + PAD)
MAX_SEQ_LEN = 20  # maximum word length in characters


def text_to_tensor(word: str, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    """
    Convert a word string to a fixed-length integer tensor.

    Characters not in the vocabulary (digits, punctuation, etc.) are
    skipped.  The sequence is right-padded with PAD_TOKEN to max_len.

    Example:
        "Hello" → [33, 30, 37, 37, 40, 52, 52, ..., 52]  (length 20)

    Args:
        word    : the word string (from the document)
        max_len : output sequence length (pads or truncates)

    Returns:
        torch.LongTensor of shape [max_len]
    """
    indices = [LETTER2INDEX[c] for c in word if c in LETTER2INDEX]
    indices = indices[:max_len]  # truncate if too long
    pad_len = max_len - len(indices)
    indices = indices + [PAD_TOKEN] * pad_len  # pad to max_len
    return torch.tensor(indices, dtype=torch.long)


#  EMA  (from WordStylist's train.py — verbatim)


class EMA:
    """
    Exponential Moving Average of model parameters.

    During training, the model weights jump around due to mini-batch noise.
    EMA maintains a smoothed copy of the weights:
        ema_params = beta * ema_params  +  (1 - beta) * current_params

    The EMA model is used *only for sampling*, not for gradient updates.
    This typically gives cleaner, more coherent generated images.

    beta=0.995 means the EMA decays slowly — new weights contribute only
    0.5% per step, so the EMA is very stable.
    """

    def __init__(self, beta: float = 0.995):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, model):
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data = self._update(ema_p.data, p.data)

    def _update(self, old, new):
        return old * self.beta + (1 - self.beta) * new if old is not None else new

    def step_ema(self, ema_model, model, warmup_steps: int = 2000):
        """
        For the first `warmup_steps` steps just copy weights directly.
        After that, apply the exponential averaging.
        This prevents the EMA from starting with bad initial weights.
        """
        if self.step < warmup_steps:
            ema_model.load_state_dict(model.state_dict())
        else:
            self.update_model_average(ema_model, model)
        self.step += 1


#  Collate function


def collate_fn(batch):
    """
    Custom collate: filter None samples (from TransformV1 skipping oversized
    patches), then stack tensors and convert text to character index tensors.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    images = torch.stack([b["image"] for b in batch])  # [B, 3, 256, 256]
    masks = torch.stack([b["mask"] for b in batch])  # [B, 1, 256, 256]
    words = [b["text"] for b in batch]  # list[str]

    # Convert each word to a fixed-length character index tensor
    word_tensors = torch.stack([text_to_tensor(w) for w in words])  # [B, MAX_SEQ_LEN]

    return {
        "image": images,
        "mask": masks,
        "text": words,  # keep strings for logging
        "word_tensor": word_tensors,  # [B, 20] for CharacterEncoder
    }


#  Trainer


class DiffusionTrainer:
    """
    Manages the complete training loop.

    Args
    ----
    dataset_train   : atria dataset split (train)
    dataset_val     : atria dataset split (validation, optional)
    batch_size      : samples per gradient step
    lr              : AdamW learning rate
    num_epochs      : total training epochs
    noise_steps     : DDPM diffusion steps T
    model_channels  : UNet base channel width
    context_dim     : character embedding dimension (must equal model's context_dim)
    num_heads       : attention heads in SpatialTransformer
    output_dir      : where to save checkpoints and sample images
    device          : 'cuda' or 'cpu'
    save_every      : save checkpoint every N epochs
    sample_every    : generate sample images every N epochs
    """

    def __init__(
        self,
        dataset_train,
        dataset_val=None,
        image_size: int = 256,
        batch_size: int = 8,
        lr: float = 1e-4,
        num_epochs: int = 100,
        noise_steps: int = 1000,
        model_channels: int = 128,
        context_dim: int = 128,
        num_heads: int = 4,
        output_dir: str = "./output/diffusion",
        device: str | None = None,
        save_every: int = 10,
        sample_every: int = 10,
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.out = Path(output_dir)
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.sample_every = sample_every

        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.out / "samples").mkdir(parents=True, exist_ok=True)

        # ── Dataloaders ───────────────────────────────────────────────
        self.train_loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # ← 0 disables multiprocessing
            drop_last=True,
        )
        self.val_loader = (
            DataLoader(
                dataset_val, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
            )  # ← 0 here too
            if dataset_val
            else None
        )

        # ── UNet ──────────────────────────────────────────────────────
        #
        # in_channels=7 because we concatenate three tensors:
        #   channels 0-2 : x_t (noisy version of the image)
        #   channels 3-5 : x0 * (1 - mask) = original with mask region zeroed
        #   channel  6   : binary mask (1 = region to generate)
        #
        # The model sees WHERE the mask is and what the context looks like,
        # so it only needs to hallucinate the masked word region.
        self.model = UNetModel(
            image_size=self.image_size,
            in_channels=7,
            model_channels=model_channels,
            out_channels=3,
            num_res_blocks=1,
            attention_resolutions=(1, 2),
            channel_mult=(1, 2, 4),
            num_heads=num_heads,
            use_scale_shift_norm=True,
            context_dim=context_dim,
            vocab_size=VOCAB_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            device=self.device,
        ).to(self.device)

        # EMA copy — updated after every step, used only for sampling
        self.ema = EMA(beta=0.995)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

        print("\n===== Before GaussianDiffusion =====")
        print("noise_steps:", noise_steps)
        print("type:", type(noise_steps))
        print("====================================\n")

        # ── Diffusion ─────────────────────────────────────────────────
        self.diffusion = GaussianDiffusion(
            noise_steps=noise_steps,
            device=self.device,
        )

        # ── Optimiser ─────────────────────────────────────────────────
        self.opt = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=num_epochs
        )
        self.mse = nn.MSELoss()

    # ── Single training step ──────────────────────────────────────────
    def _train_step(self, batch: dict) -> float:
        print("Entered _train_step")
        images = batch["image"].to(self.device)  # [B, 3, 256, 256]
        masks = batch["mask"].to(self.device)  # [B, 1, 256, 256]
        word_tensor = batch["word_tensor"].to(self.device)  # [B, 20]

        B = images.shape[0]
        print("Loaded images")
        print("Loaded masks")
        print("Loaded word tensors")

        # Random timestep for each sample in the batch
        t = self.diffusion.sample_timesteps(B)

        # Encode text: character indices → context sequence [B, seq_len, D]
        context = self.model.word_emb(word_tensor)
        print("Context created")
        # Compute diffusion loss
        loss = self.diffusion.p_losses(self.model, images, masks, context, t)
        print("Loss computed")
        self.opt.zero_grad()
        loss.backward()
        print("Backward completed")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        print("Optimizer step completed")
        self.ema.step_ema(self.ema_model, self.model)

        return loss.item()

    # ── Generate and save sample images ───────────────────────────────
    @torch.no_grad()
    def _generate_samples(self, batch: dict, tag: str):
        """
        Run the full reverse diffusion chain on a fixed batch and save a grid.
        Grid layout: [masked input | EMA-generated | ground truth]
        """
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)
        word_tensor = batch["word_tensor"].to(self.device)

        masked_img = images * (1.0 - masks)
        context = self.ema_model.word_emb(word_tensor)

        generated, frames = self.diffusion.sample(
            self.ema_model, masked_img, masks, context
        )
        # Save the 10-frame denoising sequence (first sample in batch only)
        if len(frames) > 0:
            sequence = torch.stack([f[0] for f in frames], dim=0)  # [10, 3, H, W]
            sequence = sequence * 0.5 + 0.5  # [-1,1] → [0,1]
            save_image(
                sequence,
                self.out / "samples" / f"{tag}_sequence.png",
                nrow=len(frames),
            )
            logger.info("Saved sequence → %s/samples/%s_sequence.png", self.out, tag)
        # Stack: [masked | generated | original] side by side
        vis = torch.cat([masked_img, generated, images], dim=0)

        # Convert from [-1, 1] → [0, 1] for saving
        save_image(
            vis * 0.5 + 0.5, self.out / "samples" / f"{tag}.png", nrow=images.shape[0]
        )
        logger.info("Saved samples → %s/samples/%s.png", self.out, tag)

    # ── Full training loop ────────────────────────────────────────────
    def train(self):
        logger.info(
            "Starting training on device=%s for %d epochs", self.device, self.num_epochs
        )

        # Fix one batch for consistent visual progress tracking
        fixed_batch = next(iter(self.val_loader or self.train_loader))
        if fixed_batch is None:
            raise RuntimeError(
                "First batch was None — check your dataset and transforms."
            )

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            total_loss, n_steps = 0.0, 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:03d}")
            for batch in pbar:
                if batch is None:
                    continue
                try:
                    loss = self._train_step(batch)
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    raise
                total_loss += loss
                n_steps += 1
                pbar.set_postfix(loss=f"{loss:.4f}")

            avg_loss = total_loss / max(n_steps, 1)
            self.scheduler.step()
            logger.info(
                "Epoch %03d | avg_loss=%.4f | lr=%.2e",
                epoch,
                avg_loss,
                self.scheduler.get_last_lr()[0],
            )

            # ── Save checkpoint ───────────────────────────────────────
            if epoch % self.save_every == 0:
                ckpt_path = self.out / "checkpoints" / f"epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "ema_model": self.ema_model.state_dict(),
                        "opt": self.opt.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Checkpoint saved → %s", ckpt_path)

            # ── Generate samples ──────────────────────────────────────
            if epoch % self.sample_every == 0:
                self.ema_model.eval()
                self._generate_samples(fixed_batch, tag=f"epoch_{epoch:03d}")
