# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DeepSeek-OCR model (Python model executor target).

Architecture (deepseek-ai/DeepSeek-OCR, model_type ``deepseek_vl_v2``):
  * SAM-ViT-B vision tower with window attention (window 14, global attention
    at layers [2, 5, 8, 11]) + decomposed relative position embeddings. A
    3-layer conv neck + two stride-2 convs downsample the feature map 4x, so
    each input view of size ``S`` yields a ``(S/64)^2`` grid of 1024-dim
    tokens.
  * A CLIP-L "refine" tower (24 layers, 1024 dim) that consumes the SAM
    feature map as patch embeddings (NOT pixels —
    ``CLIPVisionEmbeddings.forward`` takes the SAM grid through the
    ``patch_embeds`` argument) and adds a class token.
  * Per-token features are ``[clip_out[:, 1:] || sam_grid]`` (2048-dim) fed
    through a linear projector to the text hidden size (1280). Each grid row
    is terminated by a learned ``image_newline`` embedding.
  * Decoder: 12-layer DeepSeek-V2-style MHA (10 heads x 128, standard 1-D
    RoPE, theta 10000, no qk-norm, no bias). Layer 0 is dense; layers 1..11
    are MoE (64 routed experts, top-6 softmax routing without renormalize, 2
    shared experts).

Ordering quirk (must be preserved for parity): the HF prompt token grid is
``[global-view grid, crops grid]`` while the assembled feature rows are
``[local(crops) grid, global grid, view_seperator]``. ``masked_scatter_``
fills the token positions in order, so the global-grid tokens receive local
features and vice versa. The checkpoint was trained with this mapping — both
the C++ prompt processor and ``DeepseekVisionModel.encode_with_inputs``
reproduce it exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    apply_rotary_half,
    gather_half_rope_cos_sin,
)
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    gqa_head_split,
    load_causal_lm_weights,
    load_gqa_fused_attention,
)
from xllm.python.models.base import PyModelBase


def _queries(view_size: int, patch_size: int, downsample_ratio: int) -> int:
    """Tokens per grid edge for an ``view_size`` input (HF: 16/4 constants)."""
    return math.ceil((view_size // patch_size) / downsample_ratio)


@dataclass
class DeepseekOCRConfig:
    """Flat ModelArgs view of the DeepSeek-OCR config (see PyCausalLM)."""

    hidden_size: int = 1280
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 10
    head_dim: int = 128
    intermediate_size: int = 6848
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    vocab_size: int = 129280
    tie_word_embeddings: bool = False
    sliding_window: int = 0
    attention_bias: bool = False
    first_k_dense_replace: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 896
    shared_expert_intermediate_size: int = 1792
    norm_topk_prob: bool = False
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    n_group: int = 1
    topk_group: int = 1
    image_token_id: int = 128815
    # Vision constants: the SAM/CLIP geometry is fixed by the checkpoint (HF
    # hardcodes it in deepencoder.py's build_sam_vit_b / build_clip_l too, the
    # config.json vision_config.width block is unused). Only the view sizes
    # are configurable (Gundam recipe default: 1024 global / 640 crops).
    mm_image_base_size: int = 1024
    mm_image_size: int = 640
    mm_patch_size: int = 16
    mm_downsample_ratio: int = 4
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekOCRConfig:
        kwargs = {f.name: d[f.name] for f in fields(cls) if d.get(f.name) is not None}
        for name, key in (
            ("n_layers", "num_hidden_layers"),
            ("n_heads", "num_attention_heads"),
            ("n_kv_heads", "num_key_value_heads"),
        ):
            if name not in kwargs and d.get(key) is not None:
                kwargs[name] = d[key]
        cfg = cls(**kwargs)
        if "n_kv_heads" not in kwargs:
            cfg.n_kv_heads = cfg.n_heads
        if d.get("head_dim") is None:
            cfg.head_dim = cfg.hidden_size // cfg.n_heads
        if cfg.shared_expert_intermediate_size == 0:
            cfg.shared_expert_intermediate_size = cfg.moe_intermediate_size * cfg.n_shared_experts
        return cfg

    def head_split(self) -> tuple[int, int]:
        """Per-rank ``(num_heads, num_kv_heads)``."""
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


# ---------------------------------------------------------------------------
# SAM vision tower (window attention + decomposed relative position bias)
# ---------------------------------------------------------------------------


def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Relative position embedding lookup with linear interpolation (HF SAM)."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        dtype = rel_pos.dtype
        rel_pos = rel_pos.to(torch.float32)
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        ).to(dtype)
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def _add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: tuple[int, int],
    k_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    q_h, q_w = q_size
    k_h, k_w = k_size
    rh = _get_rel_pos(q_h, k_h, rel_pos_h)
    rw = _get_rel_pos(q_w, k_w, rel_pos_w)
    bsz, _, dim = q.shape
    r_q = q.reshape(bsz, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw)
    return rel_h.reshape(bsz, q_h * q_w, k_h, 1), rel_w.reshape(bsz, q_h * q_w, 1, k_w)


def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def _window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


class SamAttention(nn.Module):
    """SAM multi-head attention with decomposed relative position bias."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        input_size: tuple[int, int] | None = None,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.use_rel_pos = input_size is not None
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, b * self.num_heads, h * w, -1).unbind(0)

        if self.use_rel_pos:
            # HF add_decomposed_rel_pos operates on the flattened
            # (b * num_heads, h * w, head_dim) view and returns biases of
            # shape (b * num_heads, h * w, k_h, k_w) built from them.
            rel_h, rel_w = _add_decomposed_rel_pos(
                q,
                self.rel_pos_h,
                self.rel_pos_w,
                (h, w),
                (h, w),
            )
            attn_bias = rel_h + rel_w  # (b*heads, h*w, k_h, k_w)
            out = F.scaled_dot_product_attention(
                q.view(b, self.num_heads, h * w, -1),
                k.view(b, self.num_heads, h * w, -1),
                v.view(b, self.num_heads, h * w, -1),
                attn_mask=attn_bias.view(b, self.num_heads, h * w, h * w),
            )
        else:
            out = F.scaled_dot_product_attention(
                q.view(b, self.num_heads, h * w, -1),
                k.view(b, self.num_heads, h * w, -1),
                v.view(b, self.num_heads, h * w, -1),
            )
        out = out.view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        return self.proj(out)


class SamMLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act_layer: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.act = act_layer()
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class SamBlock(nn.Module):
    """SAM transformer block with optional window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        window_size: int,
        input_size: tuple[int, int],
        norm_layer: type[nn.Module],
        act_layer: type[nn.Module],
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SamAttention(
            dim,
            num_heads=num_heads,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = SamMLPBlock(dim, int(dim * mlp_ratio), act_layer)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, pad_hw, (h, w))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm (SAM neck); parameters match HF LayerNorm2d."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    """Conv patch embedding; nested ``.proj`` matches the checkpoint name
    (``model.sam_model.patch_embed.proj.weight``)."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)  # (B, h, w, C)


class SamVisionTower(nn.Module):
    """SAM-ViT-B image encoder with the DeepSeek-OCR neck/downsample.

    ``img_size`` only sizes the stored ``pos_embed``; actual inputs may be
    any multiple of 16 (the position embedding is bicubic-interpolated at
    runtime, exactly like ``get_abs_pos_sam``).
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        global_attn_indexes: tuple[int, ...] = (2, 5, 8, 11),
        window_size: int = 14,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList(
            SamBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
            )
            for i in range(depth)
        )
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(out_chans, out_chans * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(out_chans * 2, out_chans * 4, kernel_size=3, stride=2, padding=1, bias=False)

    def _abs_pos(self, grid: int) -> torch.Tensor:
        """Bicubic-interpolated absolute position embedding (SAM variant).

        Computed in fp32 and cast back to the parameter dtype, mirroring
        ``get_abs_pos_sam`` (which interpolates on the fp32 copy).
        """
        if grid == self.pos_embed.shape[1]:
            return self.pos_embed
        pos = self.pos_embed.to(torch.float32).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(grid, grid), mode="bicubic", antialias=True)
        return pos.permute(0, 2, 3, 1).to(self.pos_embed.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, 3, H, W)`` -> ``(B, 1024, H/64, W/64)`` feature map."""
        x = self.patch_embed(x)  # (B, h, w, C)
        x = x + self._abs_pos(x.size(1))
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.net_2(x)
        return self.net_3(x)


# ---------------------------------------------------------------------------
# CLIP refine tower (consumes the SAM grid as patch embeddings)
# ---------------------------------------------------------------------------


class ClipVisionEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(hidden_size))
        self.patch_embedding = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        num_patches = (image_size // patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches + 1, hidden_size)

    def _abs_pos(self, tgt_tokens: int) -> torch.Tensor:
        """CLIP absolute position embedding with bicubic interpolation.

        Mirrors ``get_abs_pos``: fp32 interpolation cast back to the
        parameter dtype before re-joining the class token.
        """
        abs_pos = self.position_embedding.weight
        dim = abs_pos.size(-1)
        cls_token, old_pos = abs_pos[:1], abs_pos[1:]
        src_size = int(math.sqrt(old_pos.shape[0]))
        tgt_size = int(math.sqrt(tgt_tokens))
        if src_size != tgt_size:
            old_pos = old_pos.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).to(torch.float32)
            new_pos = F.interpolate(
                old_pos,
                size=(tgt_size, tgt_size),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            ).to(abs_pos.dtype)
            new_pos = new_pos.permute(0, 2, 3, 1).reshape(tgt_size * tgt_size, dim)
            return torch.cat([cls_token, new_pos], dim=0)
        return abs_pos

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        """``patch_embeds: (B, C, h, w)`` SAM grid -> ``(B, h*w + 1, C)``."""
        batch_size = patch_embeds.shape[0]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (B, h*w, C)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        return embeddings + self._abs_pos(embeddings.size(1) - 1).unsqueeze(0)


class ClipAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv_proj(x).view(bsz, seqlen, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.permute(0, 2, 1, 3).reshape(bsz, seqlen, -1))


class ClipFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    @staticmethod
    def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self._quick_gelu(self.fc1(x)))


class ClipBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.self_attn = ClipAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = ClipFeedForward(hidden_size, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(self.layer_norm1(x))
        return h + self.mlp(self.layer_norm2(h))


class ClipTransformer(nn.Module):
    """Block list holder; the ``layers`` attribute matches the checkpoint
    (``vision_model.transformer.layers.<i>.*``)."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.layers = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return x


class ClipRefineTower(nn.Module):
    """CLIP-L tower. The HF module attribute is spelled ``pre_layrnorm``
    (sic); keep it so checkpoint names resolve without aliases."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        ffn_hidden_size: int = 4096,
    ) -> None:
        super().__init__()
        self.embeddings = ClipVisionEmbeddings(hidden_size)
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.transformer = ClipTransformer(
            nn.ModuleList(ClipBlock(hidden_size, num_heads, ffn_hidden_size) for _ in range(num_layers))
        )

    def forward(self, sam_grid: torch.Tensor) -> torch.Tensor:
        """``sam_grid: (B, 1024, h, w)`` -> ``(B, h*w + 1, 1024)``."""
        x = self.embeddings(sam_grid)
        x = self.pre_layrnorm(x)
        return self.transformer(x)


class MlpProjector(nn.Module):
    """Linear projector; module attribute ``layers`` matches the checkpoint
    (``model.projector.layers.{weight,bias}``)."""

    def __init__(self, input_dim: int, n_embed: int) -> None:
        super().__init__()
        self.layers = nn.Linear(input_dim, n_embed, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Vision model: towers + projector + per-image token assembly
# ---------------------------------------------------------------------------


class DeepseekVisionModel(nn.Module):
    """Combines the towers and assembles per-image token embeddings.

    Feature-row order (HF ``DeepseekOCRModel.forward``): per image
    ``[local(crops) rows, global rows, view_seperator]``. This is the order
    the C++ side scatters into the prompt token grid (see module docstring
    for the ordering quirk).
    """

    def __init__(self, text_hidden_size: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.sam_model = SamVisionTower()
        self.vision_model = ClipRefineTower()
        # Projector input is [clip_refined (1024) || sam_features (1024)] —
        # see _refine's torch.cat — NOT the SAM width alone.
        self.projector = MlpProjector(input_dim=self.sam_model.net_3.out_channels * 2, n_embed=text_hidden_size)
        self.image_newline = nn.Parameter(torch.randn(text_hidden_size))
        self.view_seperator = nn.Parameter(torch.randn(text_hidden_size))
        self.dtype = dtype
        self.device = device
        self.to(dtype=dtype, device=device)

    def _refine(self, views: torch.Tensor) -> torch.Tensor:
        """``(B, 3, S, S)`` views -> projected ``(B, q^2, hidden)`` tokens."""
        sam_out = self.sam_model(views)  # (B, 1024, q, q)
        clip_out = self.vision_model(sam_out)  # (B, q^2 + 1, 1024)
        features = torch.cat((clip_out[:, 1:], sam_out.flatten(2).permute(0, 2, 1)), dim=-1)
        return self.projector(features)

    def _grid_tokens(self, tokens: torch.Tensor, rows: int, cols: int, dim: int) -> torch.Tensor:
        """Reshape a ``(R, C, dim)`` grid to ``R * (C + 1)`` rows, appending
        an ``image_newline`` embedding at the end of every row (HF layout)."""
        grid = tokens.reshape(rows, cols, dim)
        grid = torch.cat([grid, self.image_newline[None, None, :].expand(rows, 1, dim)], dim=1)
        return grid.reshape(-1, dim)

    def _crop_grid_tokens(
        self,
        tokens: torch.Tensor,
        wc: int,
        hc: int,
        q: int,
        dim: int,
    ) -> torch.Tensor:
        """Crop grid with HF's row-interleaved reshape.

        ``tokens`` are the per-crop token grids in crop emission order
        (row-major over the ``(hc, wc)`` crop grid). HF reorders them as
        ``view(hc, wc, q, q).permute(0, 2, 1, 3, 4).reshape(hc*q, wc*q)``
        so that output row ``r`` concatenates crop-row ``r // q`` of every
        crop column — a simple ``view`` would keep whole crops contiguous
        and scramble the layout.
        """
        grid = tokens.reshape(hc, wc, q, q, dim)
        grid = grid.permute(0, 2, 1, 3, 4).reshape(hc * q, wc * q, dim)
        return self._grid_tokens(grid, hc * q, wc * q, dim)

    def encode(
        self,
        pixel_values: torch.Tensor,
        pixel_values_crops: torch.Tensor | None,
        image_spatial_crop: torch.Tensor,
        base_size: int,
        crop_size: int,
    ) -> torch.Tensor:
        """All images of the batch in one pass.

        Args:
            pixel_values: ``(N, 3, base_size, base_size)`` global views.
            pixel_values_crops: ``(M, 3, crop_size, crop_size)`` crops for
                all images concatenated in image order (None when no image
                was cropped).
            image_spatial_crop: ``(N, 2)`` ``(width_crops, height_crops)``
                per image.
        Returns:
            ``(total_tokens, hidden)`` embeddings concatenated in image order.
        """
        pixel_values = pixel_values.to(dtype=self.dtype, device=self.device)
        if pixel_values_crops is not None:
            pixel_values_crops = pixel_values_crops.to(dtype=self.dtype, device=self.device)

        num_images = pixel_values.shape[0]
        global_tokens = self._refine(pixel_values)  # (N, q_base^2, hidden)

        crops_per_image = [0] * num_images
        if pixel_values_crops is not None:
            for i in range(num_images):
                wc = int(image_spatial_crop[i][0])
                hc = int(image_spatial_crop[i][1])
                crops_per_image[i] = wc * hc if (wc > 1 or hc > 1) else 0
        num_crops = sum(crops_per_image)
        crop_tokens: torch.Tensor | None = None
        if num_crops > 0:
            crop_tokens = self._refine(pixel_values_crops)  # (M, q^2, hidden)

        q_base = _queries(base_size, 16, 4)
        q_crop = _queries(crop_size, 16, 4)
        dim = global_tokens.shape[-1]
        # Batched global grids: every image contributes q_base * (q_base + 1)
        # rows (grid + trailing newline per row).
        global_rows = self._grid_tokens(global_tokens, num_images * q_base, q_base, dim)
        sep_rows = self.view_seperator[None, :].expand(num_images, dim)

        parts: list[torch.Tensor] = []
        crop_offset = 0
        for i in range(num_images):
            # Local (crops) rows first, then global rows, then the separator —
            # the HF masked_scatter order (see module docstring).
            n_crops = crops_per_image[i]
            if n_crops > 0:
                wc = int(image_spatial_crop[i][0])
                hc = int(image_spatial_crop[i][1])
                rows = crop_tokens[crop_offset : crop_offset + n_crops]
                crop_offset += n_crops
                parts.append(self._crop_grid_tokens(rows, wc, hc, q_crop, dim))
            parts.append(global_rows[i * q_base * (q_base + 1) : (i + 1) * q_base * (q_base + 1)])
            parts.append(sep_rows[i : i + 1])
        return torch.cat(parts, dim=0)

    def load_weights(self, state_dicts: list, tp_rank: int) -> None:
        """All vision weights are replicated (no tensor parallelism).

        Parameter names under this module (``sam_model.*``, ``vision_model.*``,
        ``projector.*``, ``image_newline``, ``view_seperator``) already match
        the checkpoint's ``model.<same>`` layout, because the top-level model
        class stores this module as its own ``vision_model`` attribute.
        """
        loader = ScopedWeightLoader(state_dicts, src_prefixes=("model.", ""))
        for name, param in self.named_parameters():
            loader.load_tensor(param, name)
        # position ids buffers are recomputed, never loaded.


# ---------------------------------------------------------------------------
# Decoder (MHA + small MoE, DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class DeepseekOCRAttention(nn.Module):
    """Plain MHA with standard 1-D RoPE (no q/k norm, no bias)."""

    def __init__(self, cfg: DeepseekOCRConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self._use_fused_rope = kernels.has_fused_qk_rotary(cfg.head_dim, dtype, device)
        num_heads, num_kv_heads = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        # Unsharded kv head count (the fused loader derives the split from it).
        self.n_kv_heads_full = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            sliding_window=cfg.sliding_window,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        num_tokens = qkv.size(0)
        q = qkv[:, : self.q_size].view(num_tokens, self.num_heads, self.head_dim)
        k = qkv[:, self.q_size : self.q_size + self.kv_size].view(num_tokens, self.num_kv_heads, self.head_dim)
        v = qkv[:, self.q_size + self.kv_size :]

        if self._use_fused_rope:
            q, k = kernels.apply_qk_rotary(q, k, rope_cos, rope_sin)
        else:
            q = apply_rotary_half(q, rope_cos, rope_sin)
            k = apply_rotary_half(k, rope_cos, rope_sin)

        attn_out = self.attn(
            q.reshape(num_tokens, self.q_size),
            k.reshape(num_tokens, self.kv_size),
            v,
        )
        return self.o_proj(attn_out)

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        load_gqa_fused_attention(self, weights, context, self.n_kv_heads_full, qk_norm=False)
        self.o_proj.process_weights_after_loading()


class DeepseekMoE(nn.Module):
    """64-expert top-6 softmax-routed MoE + 2 shared experts (NPU bf16).

    Weight layout matches ``kernels.grouped_moe_bf16``: ``w13`` is
    ``(E, hidden, 2 * intermediate)`` and ``w2`` is ``(E, intermediate,
    hidden)``. Only single-rank configurations are supported (moe_tp=ep=dp=1);
    attention and lm_head tensor parallelism are unaffected.
    """

    def __init__(self, cfg: DeepseekOCRConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        if cfg.dp_size > 1 or cfg.moe_tp_size > 1 or cfg.ep_size > 1:
            raise NotImplementedError("deepseek_ocr MoE currently supports dp=moe_tp=ep=1 only")
        if cfg.scoring_func != "softmax" or cfg.norm_topk_prob:
            raise NotImplementedError("deepseek_ocr MoE supports softmax scoring without renormalize only")
        if cfg.routed_scaling_factor != 1.0:
            raise NotImplementedError("deepseek_ocr MoE ignores routed_scaling_factor; only 1.0 is supported")
        self.num_experts = cfg.n_routed_experts
        self.top_k = cfg.num_experts_per_tok
        self.gate = nn.Linear(cfg.hidden_size, self.num_experts, bias=False, dtype=dtype, device=device)
        self.w13 = nn.Parameter(
            torch.empty(
                self.num_experts,
                cfg.hidden_size,
                2 * cfg.moe_intermediate_size,
                dtype=dtype,
                device=device,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                self.num_experts,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                dtype=dtype,
                device=device,
            )
        )
        self.shared_experts = GatedMLP(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            cfg.tp_size,
            dtype,
            device,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = kernels.moe_fused_topk(self.gate(hidden), self.top_k, False, "softmax")
        routed = kernels.grouped_moe_bf16(
            hidden,
            topk_weights,
            topk_ids,
            self.w13,
            self.w2,
            self.num_experts,
            0,
            self.num_experts,
        )
        return routed + self.shared_experts(hidden)

    def load_weights(self, state: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        state.load_tensor(self.gate.weight, "gate.weight")
        gate = torch.stack([state.get_tensor(f"experts.{i}.gate_proj.weight") for i in range(self.num_experts)])
        up = torch.stack([state.get_tensor(f"experts.{i}.up_proj.weight") for i in range(self.num_experts)])
        # NPU grouped-matmul weight layout: [E, hidden, 2 * inter].
        state.copy(
            self.w13,
            torch.cat((gate, up), dim=1).transpose(1, 2).contiguous(),
            "experts.{gate,up}_proj.weight",
        )
        down = torch.stack([state.get_tensor(f"experts.{i}.down_proj.weight") for i in range(self.num_experts)])
        state.copy(
            self.w2,
            down.transpose(1, 2).contiguous(),
            "experts.down_proj.weight",
        )
        self.shared_experts.load_weights(state.with_prefix("shared_experts."), context)


class DeepseekOCRDecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekOCRConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = DeepseekOCRAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        if layer_id < cfg.first_k_dense_replace:
            self.mlp: Any = GatedMLP(cfg.hidden_size, cfg.intermediate_size, cfg.tp_size, dtype, device)
        else:
            self.mlp = DeepseekMoE(cfg, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(hidden, rope_cos, rope_sin)

        hidden, residual_out = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual_out

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        weights.load_tensor(self.input_layernorm.weight, "input_layernorm.weight")
        weights.load_tensor(self.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        if hasattr(self.mlp, "process_weights_after_loading"):
            self.mlp.process_weights_after_loading()


class DeepseekOCRModel(nn.Module):
    def __init__(self, cfg: DeepseekOCRConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self._use_fused_rope = kernels.has_fused_qk_rotary(cfg.head_dim, dtype, device)
        self._inputs_embeds: torch.Tensor | None = None
        tp = cfg.tp_size
        if tp <= 0 or cfg.hidden_size % tp != 0:
            raise ValueError("hidden_size must be divisible by a positive tp_size")
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = cast(
            "Sequence[DeepseekOCRDecoderLayer]",
            nn.ModuleList(DeepseekOCRDecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)),
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if self._inputs_embeds is not None:
            hidden = self._inputs_embeds
            self._inputs_embeds = None
        else:
            hidden = self.embed_tokens(input_ids)

        positions = positions.to(torch.int64).contiguous()
        rope_cos, rope_sin = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions)
        if self._use_fused_rope:
            head_dim = self.rotary.head_dim
            num_tokens = rope_cos.size(0)
            rope_cos = torch.cat((rope_cos, rope_cos), dim=-1).view(1, num_tokens, 1, head_dim)
            rope_sin = torch.cat((rope_sin, rope_sin), dim=-1).view(1, num_tokens, 1, head_dim)
        residual: torch.Tensor | None = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, rope_cos, rope_sin)
            record_layer_event(i)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class DeepseekOCRForCausalLM(PyModelBase):
    """DeepSeek-OCR for the Python model executor (``--backend=vlm
    --model_impl=python``)."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekOCRConfig.from_dict(config)
        cfg = self.cfg
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.image_token_id = cfg.image_token_id
        self.base_size = cfg.mm_image_base_size
        self.crop_size = cfg.mm_image_size

        self.vision_model = DeepseekVisionModel(cfg.hidden_size, dtype, device)
        self.model = DeepseekOCRModel(cfg, dtype=dtype, device=device)
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.vocab_size // cfg.tp_size,
            cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def encode_with_inputs(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Called by PyExecutorImpl with the gathered multimodal batch dict.

        Keys: "pixel_values" (N, 3, base, base), "image_spatial_crop"
        (N, 2), and "pixel_values_crops" (M, 3, crop, crop) when any image
        was tiled (all images' crops concatenated in image order).
        """
        return self.vision_model.encode(
            inputs["pixel_values"],
            inputs.get("pixel_values_crops"),
            inputs["image_spatial_crop"],
            self.base_size,
            self.crop_size,
        )

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        video_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if image_embeds is not None:
            mask = input_ids == self.image_token_id
            # Row-order overwrite: image token positions in the flattened
            # batch receive the encoder rows in order. A batch (e.g. a chunked
            # prefill chunk after the image tokens) may carry zero image
            # positions — skip the scatter then, like the other python VLMs.
            if mask.any():
                inputs_embeds[mask] = image_embeds.to(inputs_embeds.dtype)
        self.model._inputs_embeds = inputs_embeds
        return inputs_embeds

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        self.vision_model.load_weights(state_dicts, tp_rank)
        context = ParallelLoadContext.from_config(self.cfg, tp_rank, tp_size)
        all_weights = ScopedWeightLoader(state_dicts, src_prefixes=("model.", ""))
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            all_weights,
            context,
            tie_word_embeddings=self.cfg.tie_word_embeddings,
        )


# ---------------------------------------------------------------------------
# Image preprocessing (called from the C++ multimodal processor via pybind)
# ---------------------------------------------------------------------------


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Any,
    min_num: int = 2,
    max_num: int = 9,
    image_size: int = 640,
) -> tuple[list[Any], tuple[int, int]]:
    """HF ``dynamic_preprocess``: aspect-ratio-driven tiling into
    ``image_size`` squares (crop order is row-major over the crop grid)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    best = _find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * best[0]
    target_height = image_size * best[1]
    resized_img = image.resize((target_width, target_height))
    processed: list[Any] = []
    for i in range(best[0] * best[1]):
        box = (
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))
    return processed, best


def preprocess_images(
    images: list[torch.Tensor],
    base_size: int,
    crop_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Gundam-recipe preprocessing for the C++ multimodal processor.

    Args:
        images: decoded uint8 ``(C, H, W)`` RGB tensors (one per image). The
            upstream OpenCV decode drops EXIF, so orientation handling is the
            caller's concern (same as the HF reference).
        base_size: global-view canvas (pad to square).
        crop_size: local-view tile size. HF tiles an image into
            ``crop_size`` squares when either side exceeds ``crop_size``
            (its infer() hardcodes 640 there); passing any other value keeps
            the threshold and tile size consistent.

    Returns one dict per image with keys ``pixel_values`` (the base view),
    ``pixel_values_crops`` (tiles; absent when no crop) and
    ``image_spatial_crop`` (``[width_crops, height_crops]``).
    """
    from PIL import Image, ImageOps

    mean_fill = tuple(int(x * 255) for x in (0.5, 0.5, 0.5))

    def _transform(pil_image: Any) -> torch.Tensor:
        """ToTensor + Normalize(0.5, 0.5), as one fp32->bf16 conversion.

        Matches torchvision transforms.ToTensor (uint8 HWC -> float CHW,
        scaled by 1/255) followed by Normalize((0.5,) * 3, (0.5,) * 3) —
        the HF BasicImageTransform — without a torchvision dependency.
        """
        import numpy as np

        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.from_numpy(arr).permute(2, 0, 1).to(torch.bfloat16)

    results: list[dict[str, torch.Tensor]] = []
    for raw in images:
        pil_image = Image.fromarray(raw.permute(1, 2, 0).contiguous().numpy().astype("uint8"))

        crops_raw: list[Any] = []
        crop_ratio: tuple[int, int] = (1, 1)
        if pil_image.size[0] > crop_size or pil_image.size[1] > crop_size:
            crops_raw, crop_ratio = _dynamic_preprocess(pil_image, image_size=crop_size)

        global_view = ImageOps.pad(pil_image, (base_size, base_size), color=mean_fill)
        pixel_values = _transform(global_view).unsqueeze(0)

        item: dict[str, torch.Tensor] = {
            "pixel_values": pixel_values,
            "image_spatial_crop": torch.tensor([[crop_ratio[0], crop_ratio[1]]], dtype=torch.long),
        }
        if crops_raw:
            item["pixel_values_crops"] = torch.stack([_transform(crop) for crop in crops_raw], dim=0)
        results.append(item)
    return results
