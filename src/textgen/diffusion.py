"""
diffusion.py — WordStylist Diffusion class adapted for inpainting on FUNSD.

Source: https://github.com/koninik/WordStylist/blob/main/train.py  (Diffusion class)
Changes from original WordStylist:
  - Added inpainting masking (paste known pixels back each reverse step)
  - Added p_losses() for masked MSE training loss
  - Added save_n_frames to sample() for 10-frame denoising sequence
  - Removed VAE / latent-space / argparse dependencies
  - Removed cfg_scale / style-conditioning (not needed for FUNSD)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion:
    """
    DDPM noise schedule + forward/reverse process.

    Follows WordStylist's Diffusion class structure exactly:
      - Linear beta schedule
      - noise_images()      forward noising
      - sample_timesteps()  random t for training
      - sample()            full reverse chain (with intermediate frames)

    Added for inpainting:
      - p_losses()          masked MSE training loss
      - _paste_known()      paste unmasked pixels back each step
    """

    def __init__(
        self,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.noise_steps = noise_steps
        self.device = device

        # ── Linear beta schedule (same as WordStylist) ────────────────
        self.beta = self._prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # verbatim from WordStylist
    def _prepare_noise_schedule(self):
        return torch.linspace(
            self.beta_start if hasattr(self, "beta_start") else 1e-4,
            self.beta_end if hasattr(self, "beta_end") else 0.02,
            self.noise_steps,
        )

    # ── Forward process (verbatim from WordStylist) ───────────────────
    def noise_images(self, x, t):
        """
        Add noise to x at timestep t.
        Returns (x_t, noise).
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    # verbatim from WordStylist
    def sample_timesteps(self, n):
        return torch.randint(
            low=1, high=self.noise_steps, size=(n,), device=self.device
        )

    # ── Training loss (inpainting extension) ─────────────────────────
    def p_losses(self, model, x0, mask, context, t):
        """
        DDPM training loss — MSE inside the masked region only.

        Steps:
          1. Add noise to x0 at timestep t  →  x_t
          2. Build 7-channel input: [x_t | x0*(1-mask) | mask]
          3. Model predicts noise
          4. MSE only inside mask

        Args:
            model   : UNetModel
            x0      : clean image [B, 3, H, W]
            mask    : binary inpainting mask [B, 1, H, W]
            context : character embeddings [B, seq_len, D]
            t       : timestep indices [B]
        """
        x_t, noise = self.noise_images(x0, t)
        masked_img = x0 * (1.0 - mask)  # known pixels
        model_input = torch.cat([x_t, masked_img, mask], dim=1)  # 7 channels

        noise_pred = model(model_input, t, context)

        # MSE only inside masked region (same motivation as Palette)
        loss = F.mse_loss(noise_pred * mask, noise * mask)
        return loss

    # ── One reverse step ──────────────────────────────────────────────
    @torch.no_grad()
    def _p_sample_step(self, model, x_t, masked_img, mask, t_idx, context):
        """
        One reverse step: x_t → x_{t-1}   (following WordStylist sampling logic)

        After predicting x0, known pixels are pasted back so the model
        only ever generates inside the masked region.
        """
        B = x_t.shape[0]
        t_tens = (torch.ones(B) * t_idx).long().to(self.device)

        model_input = torch.cat([x_t, masked_img, mask], dim=1)
        noise_pred = model(model_input, t_tens, context)

        alpha = self.alpha[t_idx]
        alpha_hat = self.alpha_hat[t_idx]
        beta = self.beta[t_idx]

        # WordStylist reverse formula (same DDPM equation):
        #   x_{t-1} = 1/√α * (x_t - (1-α)/√(1-ᾱ) * ε_θ)  + √β * z
        if t_idx > 1:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)

        x_prev = (
            1.0
            / torch.sqrt(alpha)
            * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred)
            + torch.sqrt(beta) * noise
        )

        # Inpainting trick: paste known pixels back
        x_prev = x_prev * mask + masked_img * (1.0 - mask)
        return x_prev

    # ── Full reverse chain (with 10-frame sequence) ───────────────────
    @torch.no_grad()
    def sample(self, model, masked_img, mask, context, save_n_frames=10):
        """
        Full DDPM reverse chain from pure noise to generated image.

        Args:
            model        : EMA UNetModel (eval mode)
            masked_img   : [B, 3, H, W] original with mask region zeroed
            mask         : [B, 1, H, W] binary mask
            context      : [B, seq_len, D] character embeddings
            save_n_frames: how many intermediate frames to collect (default 10)

        Returns:
            x      : final generated image [B, 3, H, W]  in [-1, 1]
            frames : list of save_n_frames tensors (CPU), X_T first → X_0 last
        """
        model.eval()

        # Start from pure Gaussian noise (same as WordStylist)
        x = torch.randn_like(masked_img).to(self.device)

        # Which timesteps to snapshot evenly across the chain
        snapshot_steps = (
            set(
                int(i)
                for i in torch.linspace(self.noise_steps - 1, 0, save_n_frames).tolist()
            )
            if save_n_frames > 0
            else set()
        )

        frames = []

        # WordStylist iterates reversed(range(1, noise_steps))
        for t in tqdm(
            reversed(range(1, self.noise_steps)),
            desc="Sampling",
            total=self.noise_steps - 1,
            leave=False,
        ):
            x = self._p_sample_step(model, x, masked_img, mask, t, context)
            if t in snapshot_steps:
                frames.append(x.clone().cpu())

        # frames are collected newest-first; reverse so index 0 = most noisy
        frames.reverse()

        return x, frames
