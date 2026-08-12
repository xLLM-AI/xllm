# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL multimodal model (Python model executor target).

Architecture: ViT with deepstack feature extraction + Qwen3 LLM with
deepstack residual injection at the first N decoder layers.

Deepstack flow:
  pixel_values → ViT blocks → [intermediate layers 8,16,24] → deepstack_merger_list
                → main merger → cat([main, ds_0, ds_1, ds_2])
  → split into main + multiscale
  → main: masked_scatter into text embeds at image_token positions
  → multiscale: scattered into zero tensors → injected as residual at LLM layers 0,1,2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import kernels, ops
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3 import Qwen3Config, Qwen3DecoderLayer


# ---------------------------------------------------------------------------
# Vision config
# ---------------------------------------------------------------------------


@dataclass
class Qwen3VLVisionConfig:
    """Configuration for the Qwen3-VL vision tower."""

    deepstack_visual_indexes: List[int] = field(
        default_factory=lambda: [8, 16, 24]
    )
    depth: int = 27
    hidden_size: int = 1152
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    intermediate_size: int = 4304
    out_hidden_size: int = 5120
    in_channels: int = 3
    hidden_act: str = "gelu_pytorch_tanh"
    num_position_embeddings: int = 2304

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3VLVisionConfig":
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        # PyCausalLM passes a flat ModelArgs dict (per REGISTER_MODEL_ARGS in
        # models/vlm/qwen3_vl.h): vision fields carry the ``mm_`` prefix. Fall
        # back to the nested HF ``vision_config.*`` layout for standalone tests.
        return cls(
            deepstack_visual_indexes=list(
                pick(
                    "mm_deepstack_visual_indexes",
                    "deepstack_visual_indexes",
                    default=[8, 16, 24],
                )
            ),
            depth=int(pick("mm_num_hidden_layers", "depth", default=27)),
            hidden_size=int(pick("mm_hidden_size", "hidden_size", default=1152)),
            num_heads=int(
                pick("mm_num_attention_heads", "num_heads", default=16)
            ),
            patch_size=int(pick("mm_patch_size", "patch_size", default=16)),
            temporal_patch_size=int(
                pick("mm_temporal_patch_size", "temporal_patch_size", default=2)
            ),
            spatial_merge_size=int(
                pick("mm_spatial_merge_size", "spatial_merge_size", default=2)
            ),
            intermediate_size=int(
                pick("mm_intermediate_size", "intermediate_size", default=4304)
            ),
            out_hidden_size=int(
                pick("mm_projection_dim", "out_hidden_size", default=5120)
            ),
            in_channels=int(pick("mm_num_channels", "in_channels", default=3)),
            hidden_act=str(
                pick("mm_hidden_act", "hidden_act", default="gelu_pytorch_tanh")
            ),
            num_position_embeddings=int(
                pick(
                    "mm_num_position_embeddings",
                    "num_position_embeddings",
                    default=2304,
                )
            ),
        )


# ---------------------------------------------------------------------------
# Vision rotary position embedding helpers
# ---------------------------------------------------------------------------


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Pre-computed cos/sin cache for the 2-D vision rotary.

    Uses ``partial_rotary_factor = 0.5`` (only the first ``head_dim // 2``
    dimensions of each head are rotated).  The rotary dimension is further
    split into two halves: the first half encodes the *height* position and
    the second half encodes the *width* position (2-D rotary).
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int = 8192,
        theta: float = 10000.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim // 2  # partial_rotary_factor = 0.5
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float32, device=device
                )
                / self.rotary_dim
            )
        )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # (max_position, rotary_dim // 2)
        cos = freqs.cos()
        sin = freqs.sin()
        if dtype is not None:
            cos = cos.to(dtype)
            sin = sin.to(dtype)
        self.register_buffer("cos_cache", cos.contiguous(), persistent=False)
        self.register_buffer("sin_cache", sin.contiguous(), persistent=False)

    def forward(
        self, pos_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Index the cache by 2-D position ids.

        Args:
            pos_ids: ``(total_tokens, 2)`` — (h_pos, w_pos) per token.

        Returns:
            ``(cos, sin)`` each of shape ``(total_tokens, rotary_dim)``.
        """
        # cos_cache[pos_ids]: (total_tokens, 2, rotary_dim // 2)
        # flatten(1):        (total_tokens, rotary_dim)
        cos = self.cos_cache[pos_ids].flatten(1)
        sin = self.sin_cache[pos_ids].flatten(1)
        return cos, sin


def _apply_vision_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the 2-D vision rotary to Q and K over the FULL head_dim (fused).

    ``cos_full``/``sin_full`` are the pre-expanded ``(1, T, 1, head_dim)`` neox
    tables — the half-width cos/sin (``[h || w]`` frequencies) duplicated via
    ``cat((cos, cos))`` exactly as HF ``Qwen3VLVisionTransformer`` builds
    ``emb = cat((rotary_pos_emb, rotary_pos_emb))`` (modeling_qwen3_vl.py:790),
    then a full ``head_dim`` rotation. Expansion is done ONCE for all 27 layers
    in :meth:`Qwen3VLVisionTransformer.forward` (the table is identical across
    layers), avoiding a per-layer ``cat``+``reshape`` on the same inputs.

    One fused ``torch_npu.npu_rotary_mul`` call per tensor (``rotary_mode``
    defaults to ``'half'`` = neox), mirroring vllm-ascend's
    ``AscendApplyRotaryEmb``. Bit-identical to HF
    ``apply_rotary_pos_emb_vision`` (CPU-verified, maxdiff 0.0).

    Args:
        q, k: ``(total_tokens, num_heads, head_dim)``.
        cos_full, sin_full: ``(1, total_tokens, 1, head_dim)``.
    """
    q_embed = kernels.vision_rotary_mul(q, cos_full, sin_full)
    k_embed = kernels.vision_rotary_mul(k, cos_full, sin_full)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Vision building blocks
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchMerger(nn.Module):
    """Patch merger: norm → reshape → linear_fc1 → GELU → linear_fc2.

    When ``use_postshuffle_norm`` is ``False`` (main merger), the LayerNorm
    is applied *before* the spatial-merge reshape (per-patch norm).
    When ``True`` (deepstack mergers), the norm is applied *after* the
    reshape (per-merge-group norm).
    """

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True
        )
        # Exact GELU (erf), matching HF/vllm Qwen3VLVisionPatchMerger — NOT the
        # tanh approximation the ViT MLP uses (gelu_pytorch_tanh).
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(
            self.hidden_size, d_model, bias=True
        )
        if dtype is not None and device is not None:
            self.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        return self.linear_fc2(x)


# npu_fusion_attention natively supports head_dim in {64, 128, ...}. Qwen3-VL
# ViT head_dim is 72, so q/k/v are padded to 128 then sliced back. Mirrors
# vllm-ascend's AscendMMEncoderAttention (MIN_PAD_SIZE/MAX_PAD_SIZE).
_MIN_FUSED_HEAD_DIM = 64
_MAX_FUSED_HEAD_DIM = 128


class Qwen3VLVisionAttention(nn.Module):
    """ViT attention: 2-D rotary + fused varlen flash attention.

    All frames run in a single ``torch_npu.npu_fusion_attention`` varlen call
    (TND layout, ``actual_seq_qlen`` = cumulative per-frame lengths from
    ``cu_seqlens``), mirroring vllm-ascend's ``AscendMMEncoderAttention``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        # head_dim 72 is not a native npu_fusion_attention size: pad to 128.
        self.pad_to_max = (
            self.head_dim > _MIN_FUSED_HEAD_DIM
            and self.head_dim < _MAX_FUSED_HEAD_DIM
        )
        if dtype is not None and device is not None:
            self.to(dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        actual_seq: List[int],
        rotary_cos_full: torch.Tensor,
        rotary_sin_full: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = x.shape[0]
        qkv = self.qkv(x).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=1)  # each (seq_length, num_heads, head_dim)

        q, k = _apply_vision_rotary(q, k, rotary_cos_full, rotary_sin_full)

        # actual_seq (CPU cumulative per-frame lengths) is pre-computed once in
        # the ViT forward and reused across all layers — see comment there.
        if self.pad_to_max:
            pad = _MAX_FUSED_HEAD_DIM - self.head_dim
            q = F.pad(q, (0, pad), value=0)
            k = F.pad(k, (0, pad), value=0)
            v = F.pad(v, (0, pad), value=0)
        attn_output = kernels.vision_fusion_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            actual_seq_qlen=actual_seq,
            actual_seq_kvlen=actual_seq,
            num_heads=self.num_heads,
            scale=self.scaling,
            input_layout="TND",
        )
        if self.pad_to_max:
            attn_output = attn_output[..., : self.head_dim]

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3VLVisionMLP(nn.Module):
    """ViT MLP: linear_fc1 → GELU → linear_fc2."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu_pytorch_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        if hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        if dtype is not None and device is not None:
            self.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VLVisionBlock(nn.Module):
    """Pre-norm ViT transformer block with LayerNorm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        hidden_act: str = "gelu_pytorch_tanh",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(
            hidden_size, num_heads, dtype=dtype, device=device
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Qwen3VLVisionMLP(
            hidden_size, intermediate_size, hidden_act, dtype=dtype, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        actual_seq: List[int],
        rotary_cos_full: torch.Tensor,
        rotary_sin_full: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), actual_seq, rotary_cos_full, rotary_sin_full
        )
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full vision transformer
# ---------------------------------------------------------------------------


class Qwen3VLVisionTransformer(nn.Module):
    """Qwen3-VL vision tower with deepstack feature extraction.

    Forward returns a concatenation of the main merger output and the
    deepstack merger outputs:
        ``cat([main, ds_0, ds_1, ...], dim=1)``
    of shape ``(total_image_tokens, out_hidden_size * (1 + num_deepstacks))``.
    """

    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = vision_config
        self.dtype = dtype
        self.device = device

        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)

        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        # Patch embedding (Conv3d)
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.patch_embed = nn.Conv3d(
            vision_config.in_channels,
            self.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

        # Learned position embedding
        self.pos_embed = nn.Embedding(
            self.num_position_embeddings, self.hidden_size
        )

        # Rotary position embedding for ViT attention
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            head_dim, max_position=8192, dtype=dtype, device=device
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    self.hidden_size,
                    self.num_heads,
                    vision_config.intermediate_size,
                    hidden_act=vision_config.hidden_act,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(vision_config.depth)
            ]
        )

        # Main patch merger
        self.merger = Qwen3VLVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
            dtype=dtype,
            device=device,
        )

        # Deepstack mergers
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

        self.to(dtype=dtype, device=device)

    # -- position helpers --------------------------------------------------

    @staticmethod
    def _rot_pos_ids(
        h: int, w: int, spatial_merge_size: int, device: torch.device
    ) -> torch.Tensor:
        """Compute (h*w, 2) position ids for the 2-D rotary."""
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size

        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h_div, spatial_merge_size, w_div, spatial_merge_size
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h_div, spatial_merge_size, w_div, spatial_merge_size
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

        return torch.stack([hpos_ids, wpos_ids], dim=-1)

    def _compute_rot_pos_emb(
        self, grid_thw_list: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_grid_size = max(max(h, w) for _, h, w in grid_thw_list)
        pos_ids = []
        for t, h, w in grid_thw_list:
            ids = self._rot_pos_ids(h, w, self.spatial_merge_size, self.device)
            if t > 1:
                ids = ids.repeat(t, 1)
            pos_ids.append(ids)
        pos_ids = torch.cat(pos_ids, dim=0)
        cos, sin = self.rotary_pos_emb(pos_ids)
        return cos, sin

    def _fast_pos_embed_interpolate(
        self, grid_thw_list: List[List[int]]
    ) -> torch.Tensor:
        """Bilinearly interpolate the learned pos_embed to each image's grid."""
        num_grid = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(
                0, num_grid - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(
                h_floor, w_floor, indexing="ij"
            )
            h_ceil_grid, w_ceil_grid = torch.meshgrid(
                h_ceil, w_ceil, indexing="ij"
            )

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack(
                [h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid]
            )
            w_grid = torch.stack(
                [w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid]
            )
            indices = (h_grid * num_grid + w_grid).reshape(4, -1)
            weights = torch.stack(
                [w00, w01, w10, w11], dim=0
            ).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)  # (4, h*w, hidden_dim)
            embeds = embeds * weights
            combined = embeds.sum(dim=0)  # (h*w, hidden_dim)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    @staticmethod
    def _compute_cu_seqlens(
        grid_thw: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Cumulative sequence lengths for per-frame varlen attention."""
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        return torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), cu_seqlens]
        )

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = pixel_values.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )

        # Patch embedding
        L = hidden_states.shape[0]
        hidden_states = hidden_states.view(
            L,
            -1,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.patch_embed(hidden_states).view(
            L, self.hidden_size
        )

        # Convert grid_thw to list for position computations
        grid_thw_list = grid_thw.tolist()

        # Add position embeddings
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds

        # Rotary position embeddings. The half-width cos/sin table
        # (rotary_dim = head_dim // 2) is identical across all 27 blocks, so
        # expand it to the full head_dim neox table ONCE here (cat((cos, cos))
        # + reshape to the 1S1D r1 layout npu_rotary_mul expects) instead of
        # recomputing it in every layer's attention. Mirrors vllm-ascend's
        # AscendApplyRotaryEmb cat trick.
        rotary_cos, rotary_sin = self._compute_rot_pos_emb(grid_thw_list)
        head_dim = self.hidden_size // self.num_heads
        rotary_cos_full = torch.cat((rotary_cos, rotary_cos), dim=-1).reshape(
            1, -1, 1, head_dim
        )
        rotary_sin_full = torch.cat((rotary_sin, rotary_sin), dim=-1).reshape(
            1, -1, 1, head_dim
        )

        # Cumulative per-frame sequence lengths for varlen attention. The CPU
        # python list is materialized ONCE here (a single d2h sync) and reused
        # by all 27 layers — mirrors vllm-ascend's sequence_lengths pre-compute
        # hook (its seq_lens_cpu_cache), which exists precisely to avoid a
        # per-layer .cpu().tolist() sync point.
        cu_seqlens = self._compute_cu_seqlens(
            grid_thw.to(device=self.device), self.device
        )
        actual_seq: List[int] = cu_seqlens[1:].cpu().tolist()

        # Transformer blocks with deepstack feature collection
        deepstack_features: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states, actual_seq, rotary_cos_full, rotary_sin_full
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                ds_feat = self.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(ds_feat)

        # Main merger
        main_feature = self.merger(hidden_states)

        # Concatenate: [main, ds_0, ds_1, ...]
        hidden_states = torch.cat(
            [main_feature] + deepstack_features, dim=1
        )  # (total_image_tokens, out_hidden_size * (1 + num_deepstacks))
        return hidden_states


# ---------------------------------------------------------------------------
# mRoPE cos/sin cache (LLM)
# ---------------------------------------------------------------------------


def build_mrope_table(
    head_dim: int,
    max_position_embeddings: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the ``[max_pos, head_dim]`` = ``[cos_half | sin_half]`` cache.

    Consumed by ``torch_npu.npu_mrope`` (``cache_mode='default'``): the
    time/height/width section combination and the rotation are done by
    ``npu_mrope`` from ``positions [3, N]`` + ``mrope_section``, so — unlike a
    hand-rolled apply — no Python-side section slicing is needed. Mirrors
    Mirrors vllm-ascend's ``AscendMRotaryEmbedding`` usage. The caller passes
    ``rotary_mode='half'`` (NeoX ``rotate_half``) and ``cache_mode='interleave'``
    (interleaved cos/sin layout): ``mrope_interleaved`` governs how the T/H/W
    frequencies are arranged, which ``npu_mrope`` applies via ``mrope_section``
    + the interleaved cache; the Q/K rotation itself stays NeoX. Verified
    bit-identical to HF ``apply_rotary_pos_emb`` by
    ``tests/python/verify_qwen3_vl_mrope_vs_hf.py`` (maxdiff 0).
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim/2]
    cos_half = freqs.cos().to(dtype)
    sin_half = freqs.sin().to(dtype)
    return torch.cat([cos_half, sin_half], dim=-1).contiguous()


# ---------------------------------------------------------------------------
# LLM with deepstack injection
# ---------------------------------------------------------------------------


class Qwen3VLModel(nn.Module):
    """Qwen3 LLM extended with deepstack residual injection.

    After each of the first ``num_deepstacks`` decoder layers, the
    corresponding deepstack embedding is added to the hidden states.
    """

    def __init__(
        self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        self.cfg = cfg
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.head_dim = cfg.head_dim
        # mRoPE: build the concat [cos|sin] table when configured. Layout is
        # [max_pos, 2*head_dim] = [cos(freqs),cos(freqs) | sin(freqs),sin(freqs)],
        # matching C++ get_concat_rotary_embedding(head_dim, ...)
        # (rotary_embedding_util.cpp) so apply_sliced_mrope indexing is in-bounds.
        self.mrope_section = list(getattr(cfg, "mrope_section", []) or [])
        if self.mrope_section:
            self.register_buffer(
                "_mrope_cos_sin",
                build_mrope_table(
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    cfg.rope_theta,
                    dtype,
                    device,
                ),
                persistent=False,
            )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(cfg, i, dtype, device)
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        # Set externally by get_input_embeddings before the runner kicks in.
        self._inputs_embeds: Optional[torch.Tensor] = None
        self.deepstack_input_embeds: Optional[List[torch.Tensor]] = None

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
        cos, sin = None, None
        mrope_sec: Optional[List[int]] = None
        cos_sin_cache = self.rotary.cos_sin_cache
        # Prefill carries mRoPE positions [3, N] (time/height/width): the
        # attention applies torch_npu.npu_mrope with the [cos_half|sin_half]
        # table + mrope_section. Decode positions are collapsed to 1-D by
        # PyExecutorImpl (mRoPE == standard RoPE then) -> fused qk_norm+rope.
        if positions.dim() == 2 and self.mrope_section:
            cos_sin_cache = self._mrope_cos_sin
            mrope_sec = self.mrope_section
        else:
            cos, sin = self.rotary(positions)

        residual: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(
                hidden, residual, positions, cos_sin_cache, cos, sin, mrope_sec
            )
            # Deepstack injection: add deepstack embeds after the first
            # N layers.
            if (
                self.deepstack_input_embeds is not None
                and i < len(self.deepstack_input_embeds)
            ):
                hidden = hidden + self.deepstack_input_embeds[i]

        hidden, _ = self.norm(hidden, residual)
        # Clear deepstack embeds after use.
        self.deepstack_input_embeds = None
        return hidden


# ---------------------------------------------------------------------------
# Top-level conditional generation model
# ---------------------------------------------------------------------------


def _scatter_multimodal(
    inputs_embeds: torch.Tensor,
    deepstack_acc: Optional[torch.Tensor],
    embeds: torch.Tensor,
    mask: torch.Tensor,
    visual_dim: int,
    num_level: int,
) -> None:
    """Scatter one modality's embeddings into ``inputs_embeds`` + deepstack buffer.

    Splits ``embeds`` (last dim = ``visual_dim * (1 + num_level)``) into the main
    projection (``visual_dim``) and the multiscale / deepstack part
    (``num_level * visual_dim``). The main part overwrites the ``inputs_embeds``
    rows selected by ``mask``; the multiscale part is written into ``deepstack_acc``
    at the same rows. ``mask`` is a boolean over the flattened batch selecting one
    modality's placeholder-token positions; image and video masks are disjoint, so
    accumulation across modalities never collides.

    Assumes every multimodal token of the modality is in the current batch (i.e.
    enable_chunked_prefill=False). In-place: both tensors are mutated.
    """
    if num_level > 0:
        main, multiscale = torch.split(
            embeds, [visual_dim, visual_dim * num_level], dim=-1
        )
    else:
        main = embeds
        multiscale = None
    inputs_embeds[mask] = main.to(inputs_embeds.dtype)
    if multiscale is not None:
        deepstack_acc[mask] = multiscale.to(deepstack_acc.dtype)


class Qwen3VLForConditionalGeneration(PyModelBase):
    """Qwen3-VL top-level model: ViT + LLM + lm_head.

    The ``self.model`` attribute (required by ``PyModelBase``) is the
    :class:`Qwen3VLModel` LLM.  Vision encoding and embedding merge happen
    in :meth:`encode` and :meth:`get_input_embeddings`, which are called
    before the runner drives ``self.model``.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        # Parse configs. PyCausalLM hands us a FLAT ModelArgs dict (built by
        # build_config_dict via visit_properties): text fields are top-level
        # (hidden_size, n_layers, n_heads, tie_word_embeddings, tp_size, ...),
        # vision fields are "mm_"-prefixed, and mrope_section is
        # "rope_scaling_mrope_section". Both Qwen3Config.from_dict and
        # Qwen3VLVisionConfig.from_dict read this flat layout directly.
        vision_cfg = Qwen3VLVisionConfig.from_dict(config)
        text_cfg = Qwen3Config.from_dict(config)
        # Qwen3-VL uses multimodal RoPE (mRoPE): positions arrive as [3, N]
        # (time/height/width). Propagate mrope_section so the LLM can build the
        # mRoPE cos/sin table for prefill.
        rope_scaling = config.get("text_config", {}).get("rope_scaling", {}) or {}
        mrope_section = config.get("rope_scaling_mrope_section") or rope_scaling.get(
            "mrope_section"
        ) or []
        text_cfg.mrope_section = list(mrope_section)

        dtype = self.resolve_dtype(
            config.get("dtype") or config.get("torch_dtype")
        )
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        self.vision_cfg = vision_cfg
        self.text_cfg = text_cfg
        self.image_token_id = int(config.get("image_token_id", 151655))
        self.video_token_id = int(config.get("video_token_id", 151656))
        self.vision_start_token_id = int(
            config.get("vision_start_token_id", 151652)
        )
        self.vision_end_token_id = int(
            config.get("vision_end_token_id", 151653)
        )

        # Deepstack parameters
        self.deepstack_num_level = len(vision_cfg.deepstack_visual_indexes)
        self.visual_dim = vision_cfg.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        # Vision tower
        self.vision_model = Qwen3VLVisionTransformer(
            vision_cfg, dtype=dtype, device=device
        )

        # LLM (self.model is required by PyModelBase / the executor runner)
        self.model = Qwen3VLModel(text_cfg, dtype=dtype, device=device)

        # LM head
        tp = text_cfg.tp_size
        assert text_cfg.vocab_size % tp == 0
        self.lm_head = ColumnParallelLinear(
            text_cfg.hidden_size,
            text_cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    # ------------------------------------------------------------------
    # Connection logic: ViT → LLM
    # ------------------------------------------------------------------

    def encode(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Stage 1: vision encode → image_embeds.

        Args:
            pixel_values: ``(total_patches, C*t*p*p)`` flattened patches.
            grid_thw: ``(num_images, 3)`` T/H/W per image.

        Returns:
            ``image_embeds`` of shape
            ``(total_image_tokens, out_hidden_size * (1 + num_deepstacks))``.
        """
        pixel_values = pixel_values.to(
            dtype=self.vision_model.dtype, device=self.vision_model.device
        )
        grid_thw = grid_thw.to(dtype=torch.int32, device=self.vision_model.device)
        image_embeds = self.vision_model(pixel_values, grid_thw)
        return image_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        video_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage 2: merge text + multimodal embeddings.

        Each present modality's embedding is scattered at its placeholder-token
        positions (``image_token_id`` for still images, ``video_token_id`` for
        video) into ``inputs_embeds``; the multiscale part is accumulated into a
        shared deepstack buffer. The two masks are disjoint, so image and video
        coexist in one batch.

        This assumes every multimodal token is in the current batch (i.e.
        enable_chunked_prefill=False): the mask spans the full flattened batch and
        embedding rows are matched left-to-right. Sets ``_inputs_embeds`` and
        ``deepstack_input_embeds`` for the runner-driven ``Qwen3VLModel.forward``.
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        modalities: List[Tuple[torch.Tensor, int]] = []
        if image_embeds is not None:
            modalities.append((image_embeds, self.image_token_id))
        if video_embeds is not None:
            modalities.append((video_embeds, self.video_token_id))

        if not modalities:
            # No multimodal tokens in this step (text-only prefill / decode):
            # keep the attributes clear so the runner falls back to embed_tokens.
            self.model._inputs_embeds = None
            self.model.deepstack_input_embeds = None
            return inputs_embeds

        visual_dim = self.visual_dim
        num_level = self.deepstack_num_level
        seq_len = input_ids.shape[0]

        deepstack_acc: Optional[torch.Tensor] = None
        if num_level > 0:
            deepstack_acc = torch.zeros(
                seq_len,
                num_level * visual_dim,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )

        for embeds, token_id in modalities:
            mask = input_ids == token_id
            if not mask.any():
                continue
            _scatter_multimodal(
                inputs_embeds, deepstack_acc, embeds, mask, visual_dim, num_level
            )

        if deepstack_acc is not None:
            # Reshape: (seq_len, num_level, visual_dim) -> list of tensors.
            deepstack_acc = deepstack_acc.view(seq_len, num_level, visual_dim)
            self.model.deepstack_input_embeds = [
                deepstack_acc[:, i, :].contiguous() for i in range(num_level)
            ]
        else:
            self.model.deepstack_input_embeds = None

        # Store inputs_embeds for the runner to pick up.
        self.model._inputs_embeds = inputs_embeds
        return inputs_embeds

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        """Load ViT + LLM + lm_head weights from a HuggingFace Qwen3-VL checkpoint.

        Key layout: ``model.visual.*`` -> ``vision_model.*``,
        ``model.language_model.*`` -> ``model.*``, ``lm_head.*`` -> ``lm_head.*``.
        """
        self._load_vision_weights(state_dicts)
        self._load_llm_weights(state_dicts, tp_rank, tp_size)
        self._load_lm_head(state_dicts, tp_rank, tp_size)

    def _copy(self, param_name: str, tensor: torch.Tensor) -> None:
        """Copy a (possibly sharded) checkpoint tensor into a model parameter."""
        param = self.get_parameter(param_name)
        param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

    def _load_vision_weights(self, state_dicts: list) -> None:
        """Load vision tower weights (replicated, no TP sharding)."""
        prefix = "model.visual."

        def load(name: str) -> torch.Tensor:
            full = prefix + name
            for sd in state_dicts:
                if sd.has(full):
                    return sd.get_tensor(full)
            raise KeyError(f"checkpoint tensor not found: {full}")

        def copy(param_name: str, ckpt_name: Optional[str] = None) -> None:
            self._copy("vision_model." + param_name, load(ckpt_name or param_name))

        # patch_embed (Conv3d): checkpoint names carry an extra ".proj".
        copy("patch_embed.weight", "patch_embed.proj.weight")
        copy("patch_embed.bias", "patch_embed.proj.bias")
        copy("pos_embed.weight")

        # Every other vision tensor maps 1:1 (same trailing key); load by list.
        block_ts = (
            "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
            "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
            "attn.proj.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
        )
        merger_ts = (
            "norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
            "linear_fc2.weight", "linear_fc2.bias",
        )
        for i in range(self.vision_cfg.depth):
            for t in block_ts:
                copy(f"blocks.{i}.{t}")
        for t in merger_ts:
            copy(f"merger.{t}")
        for i in range(len(self.vision_cfg.deepstack_visual_indexes)):
            for t in merger_ts:
                copy(f"deepstack_merger_list.{i}.{t}")

    def _load_llm_weights(
        self, state_dicts: list, tp_rank: int, tp_size: int
    ) -> None:
        """Load LLM weights with TP sharding (same logic as qwen3.py)."""
        cfg = self.text_cfg
        kv_replicas = tp_size // cfg.n_kv_heads if cfg.n_kv_heads < tp_size else 1
        kv_rank = tp_rank // kv_replicas if kv_replicas > 1 else tp_rank
        kv_world = tp_size // kv_replicas if kv_replicas > 1 else tp_size

        def find(name: str):
            for pre in ("model.language_model.", "model."):
                full = pre + name
                for sd in state_dicts:
                    if sd.has(full):
                        return sd, full
            raise KeyError(f"checkpoint tensor not found: {name}")

        def load(name: str) -> torch.Tensor:
            sd, full = find(name)
            return sd.get_tensor(full)

        def shard(name: str, dim: int, kv: bool = False) -> torch.Tensor:
            t = load(name)
            w = kv_world if kv else tp_size
            if w <= 1:
                return t
            chunk = t.size(dim) // w
            r = kv_rank if kv else tp_rank
            return t.narrow(dim, r * chunk, chunk).contiguous()

        def copy(param_name: str, ckpt_name: Optional[str] = None) -> None:
            self._copy("model." + param_name, load(ckpt_name or param_name))

        # embed_tokens (sharded on hidden dim).
        self._copy("model.embed_tokens.weight", shard("embed_tokens.weight", dim=1))

        norm_ts = (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        )
        for i in range(cfg.n_layers):
            p = f"layers.{i}."
            for t in norm_ts:
                copy(p + t)
            # qkv_proj = concat of the TP-sharded q / k / v projections.
            self._copy(
                "model." + p + "self_attn.qkv_proj.weight",
                torch.cat(
                    [
                        shard(p + "self_attn.q_proj.weight", 0),
                        shard(p + "self_attn.k_proj.weight", 0, kv=True),
                        shard(p + "self_attn.v_proj.weight", 0, kv=True),
                    ],
                    dim=0,
                ),
            )
            self._copy(
                "model." + p + "self_attn.o_proj.weight",
                shard(p + "self_attn.o_proj.weight", dim=1),
            )
            # gate_up_proj = concat of the TP-sharded gate / up projections.
            self._copy(
                "model." + p + "mlp.gate_up_proj.weight",
                torch.cat(
                    [shard(p + "mlp.gate_proj.weight", 0), shard(p + "mlp.up_proj.weight", 0)],
                    dim=0,
                ),
            )
            self._copy(
                "model." + p + "mlp.down_proj.weight",
                shard(p + "mlp.down_proj.weight", dim=1),
            )

        copy("norm.weight")

    def _load_lm_head(
        self, state_dicts: list, tp_rank: int, tp_size: int
    ) -> None:
        """Load lm_head weights (sharded on vocab dim); tied embed skips."""
        if self.text_cfg.tie_word_embeddings:
            return  # tied to embed_tokens — already loaded.
        t = next(
            (sd.get_tensor("lm_head.weight") for sd in state_dicts if sd.has("lm_head.weight")),
            None,
        )
        assert t is not None, "checkpoint tensor not found: lm_head.weight"
        if tp_size > 1:
            chunk = t.size(0) // tp_size
            t = t.narrow(0, tp_rank * chunk, chunk).contiguous()
        self._copy("lm_head.weight", t)
