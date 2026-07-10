"""
unet.py — WordStylist UNet adapted for inpainting on FUNSD patches.

Source: https://github.com/koninik/WordStylist/blob/main/unet.py
Changes from original WordStylist:
  - in_channels = 7  (noisy image 3ch + masked original 3ch + mask 1ch)
  - Removed VAE / latent-space code (we work in pixel space)
  - Removed wandb / argparse dependency; device passed as plain string
  - CharacterEncoder and Word_Attention kept exactly as WordStylist
  - UNetModel kept exactly as WordStylist except constructor uses device str
"""

from abc import abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction

# ── Checkpoint utility (verbatim from WordStylist) ────────────────────────────


def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.float().detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors, ctx.input_params, output_tensors
        return (None, None) + input_grads


# ── Utility functions (verbatim from WordStylist) ─────────────────────────────


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


# ── Timestep embedding (verbatim from WordStylist) ────────────────────────────


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args_t = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args_t), torch.sin(args_t)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


# ── Feed-forward (verbatim from WordStylist) ──────────────────────────────────


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


# ── Cross-attention (verbatim from WordStylist) ───────────────────────────────


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        ctx = default(context, x)
        k, v = self.to_k(ctx), self.to_v(ctx)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim.masked_fill_(~mask, max_neg_value(sim))
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# ── BasicTransformerBlock (verbatim from WordStylist) ────────────────────────


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = use_checkpoint

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# ── SpatialTransformer (verbatim from WordStylist) ────────────────────────────


class SpatialTransformer(nn.Module):
    def __init__(
        self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, 1))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


# ── TimestepBlock / TimestepEmbedSequential (verbatim from WordStylist) ───────


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


# ── Upsample / Downsample (verbatim from WordStylist) ────────────────────────


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ── ResBlock (verbatim from WordStylist) ──────────────────────────────────────


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = self.h_upd(in_rest(x))
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ── Character Encoder (verbatim from WordStylist) ─────────────────────────────


class Word_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)
        scores = self.softmax(query @ key.transpose(-2, -1))
        return scores @ value


class CharacterEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Word_Attention(hidden_size, hidden_size)
        self.embedding_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = self._make_pe()

    def forward(self, x):
        x = self.embedding(x)
        x += self.positional_encoding[: x.size(1), :].to(x.device)
        return self.attention(x)

    def _make_pe(self):
        pe = torch.zeros(self.max_seq_len, self.embedding_dim)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / self.embedding_dim)))
                if i + 1 < self.embedding_dim:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((i + 1) / self.embedding_dim))
                    )
        return pe


# ── UNetModel (WordStylist architecture, 7-channel inpainting input) ──────────


class UNetModel(nn.Module):
    """
    WordStylist UNet adapted for inpainting.

    in_channels = 7:
        channels 0-2  noisy image  x_t
        channels 3-5  masked original  x0 * (1 - mask)
        channel  6    binary mask
    """

    def __init__(
        self,
        image_size,
        in_channels=7,
        model_channels=320,
        out_channels=3,
        num_res_blocks=1,
        attention_resolutions=(1, 1),
        dropout=0.0,
        channel_mult=(1, 1),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        num_heads=4,
        num_head_channels=-1,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=320,
        vocab_size=53,
        max_seq_len=20,
        device="cpu",
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dtype = torch.float32
        self.device_str = device

        time_embed_dim = model_channels * 4

        # ── Timestep MLP ──────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ── Character encoder (from WordStylist) ──────────────────────
        self.word_emb = CharacterEncoder(vocab_size, context_dim, max_seq_len)

        # ── Encoder ───────────────────────────────────────────────────
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    nn.Conv2d(in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        ch = model_channels
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        # ── Bottleneck ────────────────────────────────────────────────
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ── Decoder ───────────────────────────────────────────────────
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )

                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # ── Output projection ─────────────────────────────────────────
        self.out_proj = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, t, context):
        """
        Args:
            x       : [B, 7, H, W]  7-channel inpainting input
            t       : [B] integer timesteps
            context : [B, seq_len, context_dim] from CharacterEncoder
        Returns:
            predicted noise [B, 3, H, W]
        """
        t_emb = timestep_embedding(t, self.model_channels).to(x.device)
        emb = self.time_embed(t_emb)

        skips = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            skips.append(h)

        h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, emb, context)

        return self.out_proj(h.type(torch.float32))
