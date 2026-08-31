# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""glm5_3_flash (model_type=glm5_3_flash) causal LM — Python model executor target.

Hybrid decoder: 3 of every 4 layers are KDA (Kimi Delta Attention) linear
attention; the 4th is MLA full-attention driven by a DSA ``kPool`` sparse
indexer. MLP is dense SwiGLU for the first ``first_k_dense_replace`` layers and
DeepSeek-V2-style MoE (sigmoid + noaux_tc) thereafter. bf16 throughout, matching
the patched HuggingFace ``Glm53FlashForCausalLM`` reference for tensor alignment.

This is a faithful pure-torch port of the transformers implementation
(``transformers/src/transformers/models/glm5_3_flash/modeling_glm5_3_flash.py``) so
that per-tensor alignment against the reference is exact (same ops, same fp32
cast points, same eps, same interleaved RoPE, same scatter-based mask).

KDA goes through the stable ``fused_recurrent_kda`` /
``chunk_kda`` interfaces (same signatures as the transformers
``@use_kernel_func_from_hub``-decorated functions). Today those run the faithful
pure-torch delta-rule bodies (matching transformers' recurrent/chunk paths for
alignment); an NPU small-kernel implementation can later be swapped in behind
the same interface without touching the layer. No fla_npu dependency.

Per-layer linear state (conv_state + recurrent_state) is managed by the
framework: the executor binds per-sequence ``(conv_cache, ssm_cache)`` slots
onto each KDA layer and the layer reads/advances/writes them via the
``linear_state_indices`` / ``has_initial_state`` metadata view, giving correct
multi-sequence batch and cross-step decode. When no metadata/slots are
available (standalone align path) every call is treated as a fresh full
prefill.

v1 runs purely on torch/torch_npu with no xllm_ops / paged backend, so
the model is alignable standalone. Framework-layer / ops / TP / paged-cache
integration is a follow-up. The mHC (multi-residual-stream) hyper-connection
residual is implemented and always on, matching the patched transformers
reference (the ``mhc`` config field is not consulted by the reference forward).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None  # type: ignore[assignment]

from xllm.python import distributed
from xllm.python import kernels
from xllm.python.attention.backend import MlaIndexContext
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
)

_has_mhc_fused = hasattr(kernels, "hc_pre") and kernels.hc_pre is not None

# Per-chunk cap (bytes) for the indexer scores slab [B, sub, n_heads, n_pools]
# in Glm53FlashIndexer.select_topk. The unchunked tensor grows with n_pools
# (= kv_len / index_kpool): 8 GiB at 32K context, 16 GiB at 64K — beyond the
# activation headroom and a hard device OOM mid prefill.
_SCORES_SLAB_CAP_BYTES = int(1.5 * 1024 ** 3)
from xllm.python.models.base import PyModelBase
from xllm.python.models.glm5_3_flash_kpool import (
    alloc_pool_cache,
    compress_completed_pools,
    pooled_states as _kpool_pooled_states,
    read_pools,
)
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.layers.qlinear import QLinear
from xllm.python.layers.embedding import HiddenParallelEmbedding

# xllm Attention base — present in the real engine (full xllm.python package).
# Under the standalone stub-loader align path the package is not wired, so fall
# back to a stub nn.Module that accepts (and ignores) the same __init__ kwargs;
# the layers still work as plain modules there (the executor's
# isinstance(module, Attention) check is never reached on that path).
try:
    from xllm.python.layers.attention import Attention
except Exception:  # pragma: no cover - stub-loader path
    class Attention(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            super().__init__()


def _use_acl_graph(config: dict) -> bool:
    """Whether this run captures decode ACL graphs (mirrors deepseek_v32)."""
    graph_backend = str(config.get("python_graph_backend", "off")).lower()
    if graph_backend == "aclgraph":
        return True
    return graph_backend in ("", "off", "none", "0") and bool(
        config.get("enable_graph", False)
    )


def _in_acl_graph() -> bool:
    """Whether the current forward runs under ACL graph warmup/capture."""
    ctx = get_forward_context_or_none()
    return ctx is not None and (
        ctx.acl_graph is not None or ctx.execution_state is not None
    )


# Paged pool cache: write-time incremental compression + direct pool read
# (see glm5_3_flash_kpool.py).


def _capturing_acl_graph() -> bool:
    """Whether the current forward is being recorded by NPUGraph capture."""
    ctx = get_forward_context_or_none()
    return ctx is not None and ctx.acl_graph is not None


# ---------------------------------------------------------------------------
# Small faithful helpers (mirror transformers modeling_glm5_3_flash exactly).
# ---------------------------------------------------------------------------
def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style l2norm: sqrt(sum(x^2)+eps) then divide (NOT F.normalize)."""
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# Upper bound on the flattened seq dim (T = total tokens across every sequence
# in the batch) for which RMSNorm is split into per-row S=1 calls. The NPU
# reduction over the hidden dim is tiled as a function of the *full 2D tensor
# shape*, so a decode/verify batch flattened to [1, T, D] (engine varlen layout
# — see ``_current_q_seq_lens``) picks a different accumulation order for
# different T. Single-concurrency verify is T=2; with N concurrent verify
# sequences (num_spec=1) T = 2N, so e.g. 3-way concurrency gives T=6 — which the
# old ``<= 4`` cap left on the batched path while single-concurrency ran the
# row-wise S=1 path, flipping 1 ULP that cascades to an argmax divergence (the
# same root class as the MTP-vs-non-MTP fix). The cap must cover the largest
# decode/verify batch (max_seqs_per_batch * (num_speculative_tokens+1)) while
# staying well under prefill lengths (thousands) so prefill keeps the fast
# batched path. Default 64 covers max_seqs_per_batch=16, num_spec=3 with margin.
_RMSNORM_ROWWISE_MAX_S = 64


def _rowwise_rms(fn, x, *extra):
    """Split a small-S RMSNorm call into per-row S=1 calls.

    Returns ``None`` when the row-wise path does not apply (env off, S<=1, or
    S above the cap), so the caller falls through to its batched implementation.
    Slicing (``x[..., i:i+1, :]``) is a pure view — no device tensor creation,
    no H2D sync — so it is legal under aclgraph capture (see the _RMSNorm
    comment on why index_select was removed).
    """
    if (os.environ.get("GLM5_RMSNORM_ROWWISE") == "1"
            and x.dim() >= 2 and 1 < x.shape[-2] <= _RMSNORM_ROWWISE_MAX_S):
        return torch.cat(
            [fn(x[..., i:i + 1, :], *[e[..., i:i + 1, :] for e in extra])
             for i in range(x.shape[-2])],
            dim=-2)
    return None


class _RMSNorm(nn.Module):
    """Pure-torch RMSNorm matching transformers (fp32 compute, cast back)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # MTP spec-verify packs [anchor, draft] rows into one [1, S=2, D] call
        # and, under concurrency, multiple verify sequences flatten to S = 2N.
        # NPU reduce tiling is shape/process-state dependent: a batch whose
        # flattened S differs picks a different reduction path, flipping the
        # last bf16 bit near a rounding boundary (bisect10-14: bit-identical
        # input + weight, bit-identical offline replay, yet 1-ULP output
        # divergence, amplified through 45 layers to a 6e-2 final_norm drift
        # and eventual argmax flips). Row-wise dispatch makes each call use the
        # exact S=1 op shapes of single-token decode, pinning every batch size
        # onto the same kernel path. See ``_rowwise_rms`` for the S cap; large
        # prefill (S in the hundreds/thousands) stays on the fast batched path.
        rowwise = _rowwise_rms(self.forward, x)
        if rowwise is not None:
            return rowwise
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x).to(input_dtype)


class _UnweightedRMSNorm(nn.Module):
    """Unweighted RMSNorm (transformers Glm53FlashTextUnweightedRMSNorm).

    Used inside the mHC input projection: no weight parameter, just rescale by
    the fp32 RMS then cast back (input_norm in the reference HyperConnection).
    """

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # Same concurrency tiling hazard as _RMSNorm (this runs in the mHC
        # input projection on the same flattened [1, T, D] decode batch);
        # pin S=1 per row so the reduce path is batch-size-independent.
        rowwise = _rowwise_rms(self.forward, x)
        if rowwise is not None:
            return rowwise
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(input_dtype)


class _RMSNormGated(nn.Module):
    """RMSNorm + sigmoid gate (transformers Glm53FlashRMSNormGated)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        # o_norm runs on the 4D KDA head output [B, S, num_heads, head_dim]; the
        # RMS reduction over head_dim has row count = B*S*num_heads, so the NPU
        # tiling depends on the flattened batch size S (T under concurrency).
        # Single-concurrency (T=2) happened to match non-MTP decode (T=1) for the
        # 4172-char baseline, but T=6 under 3-way concurrency picks a different
        # reduction path and flips 1 ULP at @391/@322. Split along the seq dim
        # (dim=1, NOT dim=-2=heads) so every per-row call is [B,1,nh,hd] with a
        # fixed nh row count — identical to non-MTP single-token decode, which is
        # S=1 (no split) and therefore the same [B,1,nh,hd] reduction. Slicing is
        # a pure view (graph-safe, no H2D/sync); large prefill S stays batched.
        if (os.environ.get("GLM5_RMSNORM_ROWWISE") == "1"
                and x.dim() == 4 and 1 < x.shape[1] <= _RMSNORM_ROWWISE_MAX_S):
            return torch.cat(
                [self.forward(x[:, i:i + 1], gate[:, i:i + 1])
                 for i in range(x.shape[1])],
                dim=1)
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = self.weight.to(torch.float32) * x
        x = x * torch.sigmoid(gate.to(torch.float32))
        return x.to(input_dtype)


def _causal_conv1d_fn(mixed_qkv: torch.Tensor, weight: torch.Tensor,
                      activation: str = "silu") -> torch.Tensor:
    """Depthwise causal conv1d (left-pad K-1) + activation, fp32 weight."""
    # mixed_qkv: [B, conv_dim, S]; weight: [conv_dim, K] (squeezed)
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        mixed_qkv.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=None,
        padding=padding,
        groups=mixed_qkv.shape[1],
    )[:, :, : mixed_qkv.shape[-1]]
    if activation == "silu":
        out = F.silu(out)
    return out.to(mixed_qkv.dtype)


def _causal_conv1d_update(mixed_qkv: torch.Tensor, conv_state: torch.Tensor,
                          weight: torch.Tensor, activation: str = "silu"
                          ) -> torch.Tensor:
    """Incremental depthwise causal conv1d + activation (single-token decode).

    Faithful port of transformers ``causal_conv1d_update``: prepend the
    ``conv_state`` (last K-1 conv inputs), run a width-0-pad conv over the
    concatenation, slice the tail, and update ``conv_state`` in place.

    Args:
        mixed_qkv: [B, conv_dim, S] (S == 1 for decode).
        conv_state: [B, conv_dim, K-1], updated in place to the last K-1 inputs.
        weight: [conv_dim, K] (squeezed).
    """
    _, hidden_size, seq_len = mixed_qkv.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight=weight.unsqueeze(1), bias=None, padding=0,
        groups=hidden_size,
    )[:, :, -seq_len:]
    if activation == "silu":
        out = F.silu(out)
    return out.to(mixed_qkv.dtype)


def _round_mantissa_rne(x: torch.Tensor, keep_bits: int = 11) -> torch.Tensor:
    """Round fp32 to ``keep_bits`` explicit mantissa bits (round-to-nearest-even).

    Pure integer-op transform (ACL-graph capturable; ``view(dtype)`` is a
    metadata-only reinterpret). Standard RNE: add ``0x7FF + lsb`` then mask.
    """
    keep = 23 - keep_bits
    xi = x.contiguous().view(torch.int32)
    xi = xi + (((1 << (keep - 1)) - 1) + ((xi >> keep) & 1))
    return (xi & ~((1 << keep) - 1)).view(torch.float32)


def _causal_conv1d_update_graph(mixed_qkv: torch.Tensor, conv_state: torch.Tensor,
                                weight: torch.Tensor, activation: str = "silu"
                                ) -> torch.Tensor:
    """Graph-capturable twin of ``_causal_conv1d_update`` (ACL-graph decode).

    F.conv1d lowers to the aclop Conv2D, which NPUGraph cannot capture (and
    ``allow_internal_format=False`` would break the MoE W8A8 NZ weights), so
    the depthwise width-K conv is unrolled into elementwise multiply-adds
    reproducing the aclop kernel's numeric contract bit for bit (see the
    accumulate block below). Only called on the graph decode path; eager
    keeps the F.conv1d original byte-for-byte. NOTE the engine's conv weight
    is bf16 (checkpoint overrides the fp32 init), so the eager conv runs in
    bf16 — emulating the fp32 conv contract here returns a wrong-SHAPE
    tensor (bf16 view(int32) halves the last dim) and crashes capture.
    """
    _, hidden_size, seq_len = mixed_qkv.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    k_size = weight.shape[-1]
    # Unified aclop conv contract (reverse-engineered bitwise on NPU): operands
    # are rounded RNE to 11 explicit mantissa bits (a no-op for bf16/fp16
    # sources). Since the engine's conv weight is always bf16 and hidden_states
    # are cast to bf16 above, the RNE rounding is a no-op — float() alone
    # preserves the exact bf16 value in fp32. We skip _round_mantissa_rne to
    # avoid RightShift/BitwiseAnd on AI_CPU (~7.4% of total decode time).
    h_r = hidden_states_new.float()
    w_r = weight.float()
    out = w_r[:, 0:1].unsqueeze(0) * h_r[:, :, 0:seq_len]
    for k in range(1, k_size):
        out = out + w_r[:, k:k + 1].unsqueeze(0) * h_r[:, :, k:k + seq_len]
    out = out.to(weight.dtype)
    if activation == "silu":
        out = F.silu(out)
    return out.to(mixed_qkv.dtype)


def fused_recurrent_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA fused recurrent delta-rule (single-token decode path).

    Stable interface matching transformers ``fused_recurrent_kda``
    (``@use_kernel_func_from_hub_with_fallback``-decorated); the pure-torch body is the
    faithful port of the reference fallback. An NPU small kernel can be swapped
    in behind this interface without changing the layer.
    """
    initial_dtype = query.dtype
    # transformers recurrent path: NO transpose; shapes stay [B, S, nh, hd].
    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale
    core_attn_out = torch.zeros(
        batch_size, sequence_length, num_heads, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    for i in range(sequence_length):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = g[:, i][..., None].exp()
        b_i = beta[:, i][..., None]
        last_recurrent_state = last_recurrent_state * g_i
        kv_mem = (last_recurrent_state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        last_recurrent_state = (
            last_recurrent_state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        )
        core_attn_out[:, i] = (
            (last_recurrent_state * q_i.unsqueeze(-1)).sum(dim=-2)
        )
    final_state = last_recurrent_state if output_final_state else None
    return core_attn_out.to(initial_dtype), final_state


def chunk_kda(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA chunked delta-rule (multi-token prefill path).

    Stable interface matching transformers ``chunk_kda``
    (``@use_kernel_func_from_hub_with_fallback``-decorated); the pure-torch body is the
    faithful port of the reference fallback. An NPU small kernel can be swapped
    in behind this interface without changing the layer.
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Intra chunk
    g = g.cumsum(dim=-2)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (
            (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i])
            .sum(dim=-1).masked_fill(mask, 0)
        )
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    final_state = last_recurrent_state if output_final_state else None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, final_state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Glm53FlashConfig:
    """glm5_3_flash architecture parameters (transformers schema)."""

    model_type: str = "glm5_3_flash"
    hidden_size: int = 4096
    n_layers: int = 45
    n_heads: int = 64
    n_kv_heads: int = 64
    intermediate_size: int = 12288
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1104096
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # MLA
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    # KDA
    kda_num_heads: int = 64
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    linear_lower_bound: Optional[float] = -5.0
    swiglu_limit: float = 10.0
    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 3
    # DSA / kPool indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 1
    index_kpool_compress: bool = False
    index_kpool_always_select_tail: bool = False
    # mHC (multi-stream hyper-connection residual) — always on, per reference
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    # derived
    layer_types: list = field(default_factory=list)   # "linear_attention" / "deepseek_sparse_attention"
    mlp_layer_types: list = field(default_factory=list)  # "dense" / "sparse"
    indexer_types: list = field(default_factory=list)  # "full" / "shared"
    tp_size: int = 1
    tp_rank: int = 0
    # Load-time static flag: decode ACL graph capture is enabled, so graph
    # branches (fixed-shape indexer pooling etc.) may be taken at runtime when
    # the forward context confirms capture/warmup (_in_acl_graph()).
    use_acl_graph: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_rope_head_dim + self.qk_nope_head_dim

    @classmethod
    def from_dict(cls, d: dict) -> "Glm53FlashConfig":
        # Multimodal full-weight configs nest text-model fields under
        # "text_config"; single-model configs are flat. Merge text_config into
        # the top level (without clobbering top-level overrides) so the flat
        # picks below work for both layouts. Mirrors JsonReader::resolve.
        tc = d.get("text_config")
        if isinstance(tc, dict):
            d = {**tc, **d}

        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        hidden = int(pick("hidden_size", default=4096))
        n_heads = int(pick("n_heads", "num_attention_heads", default=64))
        n_layers = int(pick("n_layers", "num_hidden_layers", default=45))
        first_k_dense = int(pick("first_k_dense_replace", default=3))

        # KDA linear_attn_config (transformers stores it as a dict)
        lac = pick("linear_attn_config", default=None) or {}
        kda_heads = int(lac.get("num_heads", 64))
        kda_dim = int(lac.get("head_dim", 128))
        conv_k = int(lac.get("short_conv_kernel_size", 4))
        full_attn_layers = lac.get("full_attn_layers")
        if full_attn_layers is None:
            full_attn_layers = [i for i in range(n_layers) if i % 4 == 3]

        # Forget-gate lower-bound resolution — mirrors reference config __post_init__:
        # the dict key is ``gate_lower_bound`` (NOT ``lower_bound``); field default
        # is -5.0; if safe_gate (default True) and the bound is None, force -5.0.
        lower_bound = lac.get("gate_lower_bound", -5.0)
        if lac.get("safe_gate", True) and lower_bound is None:
            lower_bound = -5.0

        cfg = cls(
            model_type=str(pick("model_type", default="glm5_3_flash")),
            hidden_size=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            intermediate_size=int(pick("intermediate_size", default=12288)),
            vocab_size=int(pick("vocab_size", default=154880)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            rope_theta=float(pick("rope_theta", default=10000.0)),
            max_position_embeddings=int(pick("max_position_embeddings", default=1104096)),
            hidden_act=str(pick("hidden_act", default="silu")),
            attention_bias=bool(pick("attention_bias", default=False)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            q_lora_rank=int(pick("q_lora_rank", default=1536)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=256)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=0)),
            v_head_dim=int(pick("v_head_dim", default=256)),
            kda_num_heads=kda_heads,
            kda_head_dim=kda_dim,
            short_conv_kernel_size=conv_k,
            linear_lower_bound=lower_bound,
            swiglu_limit=float(pick("swiglu_limit", default=10.0)) or 10.0,
            moe_intermediate_size=int(pick("moe_intermediate_size", default=2048)),
            n_routed_experts=int(
                pick("n_routed_experts", "num_local_experts", "num_experts", default=288)
            ),
            n_shared_experts=int(pick("n_shared_experts", default=1)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=8)),
            n_group=int(pick("n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=2.5)),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            first_k_dense_replace=first_k_dense,
            index_n_heads=int(pick("index_n_heads", default=32)),
            index_head_dim=int(pick("index_head_dim", default=128)),
            index_topk=int(pick("index_topk", default=2048)),
            index_kpool=int(pick("index_kpool", default=1)),
            index_kpool_compress=bool(pick("index_kpool_compress", default=False)),
            index_kpool_always_select_tail=bool(
                pick("index_kpool_always_select_tail", default=False)
            ),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            use_acl_graph=_use_acl_graph(d),
            # mHC fields: ModelArgs may emit a 0 default (un-plumbed); treat 0
            # /None as unset and fall back to the real 300B defaults.
            hc_mult=(int(pick("hc_mult", default=4)) or 4),
            hc_eps=(float(pick("hc_eps", default=1e-6)) or 1e-6),
            hc_sinkhorn_iters=(int(pick("hc_sinkhorn_iters", default=20)) or 20),
        )
        cfg._resolve_schedules(full_attn_layers, d)
        return cfg

    def _resolve_schedules(self, full_attn_layers: list, d: dict) -> None:
        n = self.n_layers
        lt = d.get("layer_types")
        if isinstance(lt, list) and lt:
            self.layer_types = list(lt)
        else:
            self.layer_types = [
                "deepseek_sparse_attention" if i in full_attn_layers
                else "linear_attention"
                for i in range(n)
            ]
        mlt = d.get("mlp_layer_types")
        if isinstance(mlt, list) and mlt:
            self.mlp_layer_types = list(mlt)
        else:
            n_dense = min(self.first_k_dense_replace, n)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (n - n_dense)
        it = d.get("indexer_types")
        if isinstance(it, list) and it:
            self.indexer_types = list(it)
        else:
            offset = int(d.get("index_skip_topk_offset", 1))
            freq = int(d.get("index_topk_freq", 1))
            self.indexer_types = [
                "full" if (max(i - offset + 1, 0) % max(freq, 1)) == 0 else "shared"
                for i in range(n)
            ]

    def is_dsa(self, layer_id: int) -> bool:
        return (
            layer_id < len(self.layer_types)
            and self.layer_types[layer_id] == "deepseek_sparse_attention"
        )

    def is_moe(self, layer_id: int) -> bool:
        if layer_id < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer_id] == "sparse"
        return layer_id >= self.first_k_dense_replace

    def indexer_shared(self, layer_id: int) -> bool:
        return (
            layer_id < len(self.indexer_types)
            and self.indexer_types[layer_id] == "shared"
        )


# ---------------------------------------------------------------------------
# KDA (Kimi Delta Attention) linear-attention layer
# ---------------------------------------------------------------------------
class Glm53FlashForgetGate(nn.Module):
    """forget gate: g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)."""

    def __init__(self, cfg: Glm53FlashConfig, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.head_dim = cfg.kda_head_dim
        # Head-sharded TP (mirrors DSA): each rank owns kda_num_heads//tp heads.
        # f_b_proj / A_log / dt_bias are per-head (column-parallel, dim 0);
        # f_a_proj feeds the shared head_dim latent so it stays replicated.
        self.tp = cfg.tp_size
        self.num_heads = cfg.kda_num_heads // cfg.tp_size  # local per-rank
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.safe_gate_lower_bound = cfg.linear_lower_bound

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_shape = (*hidden_states.shape[:2], -1, self.head_dim)
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (forget_gate.float() + self.dt_bias.float().view(1, 1, -1)).view(
            hidden_shape
        )
        decay_rate = torch.exp(self.A_log.float().view(1, 1, self.num_heads, 1))

        # Safe lower bound decay (reference Glm53FlashTextForgetGate): when a bound
        # is set, the gate is `-bound * sigmoid(decay_rate * g)` instead of the
        # softplus form. For the default config linear_lower_bound=-5.0 -> this
        # branch is taken.
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)

        g_softplus = torch.where(
            g > 20.0, g, torch.log(1.0 + torch.exp(g))
        )
        return -decay_rate * g_softplus


class Glm53FlashKdaAttention(Attention):
    """KDA linear-attention layer (conv1d + delta-rule + gated norm + o_proj).

    Subclasses xllm ``Attention`` (reports num_heads/head_dim/scale for the
    executor's bookkeeping) but overrides forward to dispatch the conv1d +
    delta-rule through the attention backend
    (``NpuPagedAttentionBackend.execute_linear``), which owns the per-layer
    conv/ssm state slots (conv_cache: ``[num_slots, conv_state_len, conv_dim]``,
    ssm_cache: ``[num_slots, nh, k_hd, v_hd]`` fp32) and the per-sequence
    slot/cold-start view (``linear_state_indices`` / ``has_initial_state``).

    The KDA math goes through the stable ``fused_recurrent_kda`` /
    ``chunk_kda`` interfaces (or fla_npu fused ops when
    ``GLM5NEXT_KDA_BACKEND=fla_npu``); see their docstrings.
    """

    def __init__(self, cfg: Glm53FlashConfig, layer_id: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__(
            num_heads=cfg.kda_num_heads, num_kv_heads=cfg.kda_num_heads,
            head_dim=cfg.kda_head_dim, scale=cfg.kda_head_dim ** -0.5,
            sliding_window=0, layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.hidden_size = cfg.hidden_size
        # Head-sharded TP (mirrors DSA). super().__init__ reports the FULL
        # kda_num_heads (the executor's C++ cache shape divides it by tp via
        # linear_num_key_heads/world_size — see kv_cache_shape.cpp), so the
        # framework conv/ssm slots are already this rank's head-subset
        # ([slots, conv_dim/tp, len] / [slots, nh/tp, k_hd, v_hd]). The model
        # therefore computes ONLY its local heads so mixed_qkv ([B, conv_dim/tp,
        # S]) matches the framework conv_state ([B, conv_dim/tp, len]) — without
        # this the cat at the conv window mismatches ([conv_dim/tp] vs
        # [conv_dim]) at tp>1. At tp==1 num_heads_local==num_heads (no-op), so
        # the standalone align path is unchanged.
        self.tp = cfg.tp_size
        self.num_heads = cfg.kda_num_heads  # full (reported to base / bookkeeping)
        self.num_heads_local = cfg.kda_num_heads // cfg.tp_size
        self.head_dim = cfg.kda_head_dim
        self.qkv_dim = self.head_dim * self.num_heads_local  # local per-rank
        self.conv_kernel_size = cfg.short_conv_kernel_size
        self.conv_dim = self.qkv_dim * 3  # local conv_dim = 3 * qkv_dim_local
        self.activation = cfg.hidden_act
        self.eps = cfg.rms_norm_eps

        # q/k/v: column-parallel (per-head, dim 0) — each rank produces its
        # head-subset's qkv_dim_local channels.
        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        # conv1d: depthwise over the LOCAL conv_dim (groups=conv_dim_local); the
        # loader shards each of q/k/v_conv1d by head then cats so the channel
        # order [q_loc|k_loc|v_loc] matches mixed_qkv. fp32 in transformers.
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )
        self.conv1d.weight = nn.Parameter(
            self.conv1d.weight.detach().to(torch.float32)
        )
        self.forget_gate = Glm53FlashForgetGate(cfg, dtype, device)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads_local, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = _RMSNormGated(self.head_dim, self.eps, dtype, device)
        # o_proj: row-parallel + all_reduce. With KDA now head-sharded, each
        # rank's attention output is its head-subset's partial sum over
        # qkv_dim_local; o_proj ([hidden, qkv_dim_local]) produces a partial
        # hidden that must be all-reduced across ranks (mirrors DSA o_proj).
        self.o_proj = QLinear(
            self.qkv_dim,
            self.hidden_size,
            device=device,
            dtype=dtype,
            kind="static",
            row_parallel=True,
        )


    def forward(self, hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                prev_topk_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        mixed_qkv = torch.cat(
            [self.q_proj(hidden_states),
             self.k_proj(hidden_states),
             self.v_proj(hidden_states)],
            dim=-1,
        ).transpose(1, 2)  # [B, 3*qkv_dim, S]

        g = self.forget_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # KDA conv1d + delta-rule + conv/ssm state is owned by the backend
        # (NpuPagedAttentionBackend.execute_linear). No self-contained fallback
        # — KDA must run inside the engine with a bound backend.
        ctx = get_forward_context_or_none()
        backend = getattr(ctx, "attention_backend", None) if ctx is not None else None
        if backend is None or getattr(backend, "execute_linear", None) is None:
            raise RuntimeError(
                "Glm53FlashKdaAttention requires an attention backend with "
                "execute_linear; run inside the engine."
            )
        core_attn_out = backend.execute_linear(mixed_qkv, g, beta, self)

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        # KDA is head-sharded: each rank's o_proj (row-parallel, input
        # qkv_dim_local) yields a partial hidden summed over its head-subset;
        # all-reduce across ranks to assemble the full hidden (mirrors DSA
        # o_proj). At tp==1 this is a no-op.
        o = self.o_proj(output)
        if self.tp > 1:
            distributed.all_reduce_(o)
        return o


def _current_q_seq_lens(num_seqs: int, num_tokens: int) -> list[int]:
    """Per-sequence query lengths of the current forward's varlen batch.

    The engine flattens every batch to ``[1, T, ...]``; the real sequence
    boundaries live in the attention metadata's ``q_cu_seq_lens`` (cumulative
    offsets, same source the KDA layers use).
    """
    q_cu = get_forward_context().metadata.q_cu_seq_lens
    if q_cu is None:
        raise RuntimeError(
            f"varlen batch without q_cu_seq_lens: cannot split "
            f"num_tokens={num_tokens} across num_seqs={num_seqs}")
    ends = q_cu[1:].cpu().tolist()
    if len(ends) != num_seqs or ends[-1] != num_tokens:
        raise RuntimeError(
            f"q_cu_seq_lens does not describe this batch: ends={ends}, "
            f"num_seqs={num_seqs}, num_tokens={num_tokens}")
    return [ends[0]] + [ends[i] - ends[i - 1] for i in range(1, len(ends))]


# ---------------------------------------------------------------------------
# kPool DSA indexer (assembled from small ops, faithful to transformers)
# ---------------------------------------------------------------------------
class Glm53FlashIndexer(nn.Module):
    """GlmMoeDsaRecomputeKPoolIndexer port: packed [k, gate, valid] cache."""

    def __init__(self, cfg: Glm53FlashConfig, layer_id: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim  # 0 for the 300B default
        self.topk = cfg.index_topk
        self.index_kpool = cfg.index_kpool
        self.index_kpool_compress = cfg.index_kpool_compress
        self.index_kpool_always_select_tail = cfg.index_kpool_always_select_tail
        self.use_acl_graph = cfg.use_acl_graph
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=dtype, device=device)
        )
        # raw Parameter (F.linear), matching transformers' state-dict key
        # ``indexer.index_kpool_compress_gate`` (no ``.weight`` suffix).
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, cfg.hidden_size, dtype=dtype, device=device)
        )
        # Paged pool cache per DSA layer (lazily allocated on first eager
        # forward, before graph capture): layer_id -> [blocks, bs//4, 1, D].
        self._pool_caches: dict[int, torch.Tensor] = {}

    def get_token_visible(self, key_valid: torch.Tensor,
                         local_valid: torch.Tensor,
                         total_len: int, seq_len: int,
                         current_length: int) -> torch.Tensor:
        device = key_valid.device
        key_pos = torch.arange(total_len, device=device)
        query_pos = current_length - seq_len + torch.arange(seq_len, device=device)
        causal = key_pos[None, None, :] <= query_pos[None, :, None]
        return causal & key_valid[:, None, :] & local_valid[:, :, None]

    def get_pooled_states(self, packed_states: torch.Tensor,
                          key_valid: torch.Tensor):
        pool_keys, pool_indices, pool_valid = _kpool_pooled_states(
            packed_states, key_valid, self.index_kpool_compress_ape,
            self.head_dim, self.index_kpool,
        )
        if self.use_acl_graph and _in_acl_graph():
            # Graph branch (fixed shapes): keep ALL pools. The boolean
            # ``pool_keys[:, keep]`` filter below produces a data-dependent
            # output shape (aclnnNonzeroV2) that ACL graph capture cannot
            # record. It was only a compaction: select_topk masks invalid
            # pools to -inf via candidate_valid before the topk and fills
            # their slots with -1 after it, so retaining them changes neither
            # the selected valid indices nor the output width.
            return pool_keys, pool_indices, pool_valid
        keep = pool_valid.any(0)
        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def get_packed_states(self, hidden_states: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-token indexer cache row ``[B, S, head_dim*2+1]`` =
        [k(128), gate(128), valid(1)]. NoPE: no RoPE applied to k."""
        k = self.k_norm(self.wk(hidden_states)).view(
            hidden_states.shape[0], hidden_states.shape[1], -1, self.head_dim
        )
        k = k.squeeze(2)
        gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
        valid_channel = attention_mask.to(k.dtype).unsqueeze(-1)
        return torch.cat([k, gate_scores, valid_channel], dim=-1)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, q_resid: torch.Tensor,
                attention_mask: torch.Tensor,
                kv_len: int) -> torch.Tensor:
        # Single-call path (fresh prefill / standalone align): the current
        # tokens' packed states ARE the full index history. Build them here and
        # delegate to select_topk so both paths share one pooling/selection body.
        packed_states = self.get_packed_states(hidden_states, attention_mask)
        return self.select_topk(
            q_resid, hidden_states, attention_mask,
            kv_len=kv_len, current_length=kv_len,
            packed_states=packed_states,
        )

    def _select_topk_fused_pa(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        key_valid: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_cache: torch.Tensor,
        pool_block_table: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run PoolKeyIndexer through the paged PA_BBND pool cache."""
        if pool_cache.dtype != query.dtype:
            return None
        if not _in_acl_graph() and not bool(attention_mask.all().item()):
            return None

        token_count = key_valid.to(torch.int32).sum(-1)
        # PA_BBND addresses complete pooled keys and uses pool_tail_k for the
        # remaining tokens in the current tail pool.
        pool_tail_k = torch.remainder(
            token_count, self.index_kpool
        ).to(torch.int32).contiguous()
        actual_seq_k = torch.div(
            token_count, self.index_kpool, rounding_mode="floor"
        ).to(torch.int32).contiguous()
        indices, _ = kernels.pool_key_indexer(
            query,
            pool_cache,
            weights,
            pool_tail_k,
            self.topk,
            self.index_kpool,
            return_value=False,
            actual_seq_k=actual_seq_k,
            block_table=pool_block_table.to(
                dtype=torch.int32
            ).contiguous(),
            layout_k="PA_BBND",
        )

        output_width = self.topk + (
            self.index_kpool - 1
            if self.index_kpool_always_select_tail
            else 0
        )
        return indices[..., :output_width].long()

    def select_topk(
        self,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_len: int,
        current_length: int,
        packed_states: torch.Tensor | None = None,
        pool_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        key_valid: torch.Tensor | None = None,
        pool_block_table: torch.Tensor | None = None,
        pool_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Top-k pool selection over the FULL packed index history.

        ``packed_states`` is ``[B, kv_len, head_dim*2+1]`` (k, gate, valid) for
        every kv token accumulated so far (the reference keeps this in the
        indexer cache and pools over the whole of it each step); ``q_resid`` /
        ``hidden_states`` are the CURRENT query tokens ``[B, S_q, ...]``.
        Alternatively, pass ``pool_data``/``key_valid`` from the paged pool
        cache (read_pools) to skip both the dense gather and the per-step
        re-pooling. Returns absolute kv-position top-k indices
        ``[B, S_q, topk]`` (int64, -1 = invalid), matching the reference
        indexer output.
        """
        batch_size, seq_len = q_resid.shape[:2]
        device = q_resid.device

        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)

        raw_key_cache = (
            packed_states is not None
            and packed_states.shape[-1] == self.head_dim
        )
        if raw_key_cache and self.index_kpool != 1:
            raise ValueError(
                "index_kpool_compress=false requires index_kpool=1"
            )

        if pool_data is not None:
            # Pool-cache path: pools were compressed at write time; only the
            # visibility mask is derived here (rows past each sequence's live
            # length are invalid, mirroring gather_index_history's row_valid).
            pool_keys, pool_indices, pool_valid = pool_data
            token_visible = self.get_token_visible(
                key_valid, attention_mask, kv_len, seq_len, current_length
            )
        else:
            if key_valid is None:
                if raw_key_cache:
                    key_valid = torch.ones(
                        batch_size, packed_states.shape[1],
                        dtype=torch.bool, device=device,
                    )
                else:
                    key_valid = packed_states[..., -1].gt(0)
            token_visible = self.get_token_visible(
                key_valid, attention_mask, kv_len, seq_len, current_length
            )

        weights = self.weights_proj(
            hidden_states.to(self.weights_proj.weight.dtype)
        )
        if (
            self.index_kpool_compress
            and q.device.type == "npu"
            and hasattr(kernels, "pool_key_indexer")
            and pool_cache is not None
            and pool_block_table is not None
        ):
            try:
                scaled_weights = (
                    weights * (self.n_heads ** -0.5)
                ).to(q.dtype)
                fused_indices = self._select_topk_fused_pa(
                    q,
                    scaled_weights,
                    key_valid,
                    attention_mask,
                    pool_cache,
                    pool_block_table,
                )
                if fused_indices is not None:
                    return fused_indices
            except NotImplementedError:
                pass
        if pool_data is None:
            if pool_cache is not None and pool_block_table is not None:
                kv_lens = key_valid.to(torch.int64).sum(-1)
                n_pools = (kv_len + self.index_kpool - 1) // self.index_kpool
                pool_keys, pool_indices, pool_valid = read_pools(
                    pool_cache,
                    pool_block_table,
                    kv_lens,
                    n_pools,
                    self.index_kpool,
                )
            else:
                if raw_key_cache:
                    pool_keys = packed_states
                    if pool_keys.dim() == 4:
                        pool_keys = pool_keys.squeeze(-2)
                    pool_indices = torch.arange(
                        pool_keys.shape[1], device=device,
                    ).expand(batch_size, -1).unsqueeze(-1)
                    pool_valid = key_valid
                else:
                    pool_keys, pool_indices, pool_valid = self.get_pooled_states(
                        packed_states, key_valid
                    )
        weights = weights.float()
        # Query-dim sub-chunking: each query row's logits are
        # q[row] @ pool_keys.T and its pool score a head-weighted sum of those
        # logits — independent of every other row — so slicing the query dim
        # is bit-exact while keeping each transient scores slab under
        # _SCORES_SLAB_CAP_BYTES. Without it the full [B, S, n_heads, n_pools]
        # fp32 scores tensor grows linearly with n_pools (= kv_len /
        # index_kpool) and OOMs mid prefill. When the full tensor already fits
        # (decode: S=1; short prefills) sub == seq_len and the loop degenerates
        # to the original single matmul.
        n_pools = pool_keys.shape[1]
        pool_keys_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        q_f = q.float()
        slab_elems = self.n_heads * n_pools
        sub = (
            seq_len
            if slab_elems == 0
            else max(1, min(seq_len, _SCORES_SLAB_CAP_BYTES // (slab_elems * 4)))
        )
        pool_scores = torch.empty(
            batch_size, seq_len, n_pools, dtype=torch.float32, device=device
        )
        for start in range(0, seq_len, sub):
            end = min(start + sub, seq_len)
            slab = torch.matmul(q_f[:, start:end], pool_keys_t)
            # In-place relu/scale: bit-identical elementwise math, but avoids a
            # second [B, sub, n_heads, n_pools] fp32 allocation.
            slab = torch.relu_(slab)
            slab *= self.softmax_scale
            pool_scores[:, start:end] = torch.einsum(
                "bshp,bsh->bsp", slab, weights[:, start:end] * (self.n_heads ** -0.5)
            )
            del slab

        if pool_keys.shape[1] != 0:
            # Pool visibility uses the pool's FIRST slot (start), not its last
            # (end). With pool-END visibility, a pool spanning [0..kpool-1] is
            # invisible to the first (kpool-1) queries — which can only see
            # their own position — yielding an empty attention mask (-> 0 on
            # CPU, non-zero garbage on NPU sdpa) where the reference attends to
            # the causally-visible positions (tok_i -> {0..i}). Pool-START makes
            # pool0 a candidate for tok0..kpool-2; the per-position causal filter
            # below then keeps only the slots the query can actually see.
            pool_start = pool_indices[..., 0].clamp(0, kv_len - 1)
            # AICore-native gather replaces fancy indexing
            # token_visible[batch_idx, query_idx, pool_start[:, None, :]] which
            # falls back to aclnnIndex on AI_CPU. token_visible is
            # [B, S_q, kv_len]; index dim=2 by pool_start [B, n_pools],
            # broadcast over S_q -> [B, S_q, n_pools]. bool is cast uint8 for
            # gather (no bool kernel) then restored.
            pool_idx_expand = pool_start[:, None, :].expand(batch_size, seq_len, -1)
            pool_visible = token_visible.to(torch.uint8).gather(
                2, pool_idx_expand
            ).to(torch.bool)
            candidate_valid = (pool_visible & pool_valid[:, None]).to(torch.bool)
            pool_scores = pool_scores.masked_fill(
                ~candidate_valid, torch.finfo(pool_scores.dtype).min
            )
        else:
            candidate_valid = pool_valid[:, None].expand(batch_size, seq_len, -1).to(torch.bool)

        group_budget = self.topk // self.index_kpool
        select_k = min(group_budget, pool_scores.shape[-1])
        if select_k == 0:
            topk_indices = torch.empty(
                batch_size, seq_len, 0, dtype=torch.long, device=device
            )
        else:
            selected = pool_scores.topk(select_k, dim=-1).indices
            selected_valid = candidate_valid.gather(-1, selected)
            # AICore-native gather replaces fancy indexing
            # pool_indices[batch_pool_idx, selected] which falls back to
            # aclnnIndex on AI_CPU. pool_indices is [B, n_pools, rate]; index
            # dim=1 by selected [B, S_q, select_k] -> [B, S_q, select_k, rate].
            # Flatten (n_pools*rate) rows and gather, matching the
            # get_pooled_states pattern.
            rate = self.index_kpool
            rate_off = torch.arange(rate, device=device)
            sel_flat = (selected[..., None] * rate + rate_off).reshape(
                batch_size, -1
            )
            selected_indices = pool_indices.reshape(batch_size, -1).gather(
                1, sel_flat
            ).reshape(batch_size, seq_len, select_k, rate)
            topk_indices = selected_indices.flatten(-2)
            topk_indices = topk_indices.masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
                -1,
            )
            # Per-position causal filter. Pool-START visibility admits a whole
            # pool once the query sees its first slot, but the pool's later slots
            # may lie beyond the query's causal reach (notably the early tokens
            # of a prefill). Drop any selected position the query cannot
            # causally see, so tok0 -> {0}, tok1 -> {0,1}, ... (matches ref).
            safe_pos = topk_indices.clamp(0, kv_len - 1)
            # AICore-native gather replaces fancy indexing
            # token_visible[b_idx, q_idx, safe_pos] which falls back to
            # aclnnIndex on AI_CPU. token_visible is [B, S_q, kv_len]; index
            # dim=2 by safe_pos [B, S_q, select_k*rate]. bool cast uint8 for
            # gather then restored.
            pos_visible = token_visible.to(torch.uint8).gather(2, safe_pos).to(torch.bool)
            topk_indices = topk_indices.masked_fill(~pos_visible, -1)

        output_width = self.topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
        if topk_indices.shape[-1] < output_width:
            topk_indices = F.pad(
                topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1
            )
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.long()

    def select_qli(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        ctx: MlaIndexContext,
        layer: "Attention",
        backend,
    ) -> torch.Tensor:
        """kPool top-k selection over the paged index cache, SFA-ready.

        Mirrors ``DeepseekV3Indexer.select_qli``'s contract: write the current
        tokens' packed states into the paged index cache, gather the dense
        history, run ``select_topk``, then adapt the kPool output
        (``[B, S_q, topk]`` int64, absolute KV positions, -1 = invalid) to the
        SFA op's ``sparse_indices`` (``[T, 1, topk]`` int32; sparse_block_size
        == 1 so a block index equals the absolute token position).
        """
        batch_size, seq_len = hidden_states.shape[:2]
        num_tokens = batch_size * seq_len
        if self.index_kpool_compress:
            packed = self.get_packed_states(hidden_states, attention_mask)
        else:
            # The non-compressed cache stores one raw K per token. This is the
            # original small-operator path (index_kpool=1), so no gate/valid
            # channels are written to the narrower cache.
            packed = self.k_norm(self.wk(hidden_states)).view(
                hidden_states.shape[0], hidden_states.shape[1], -1, self.head_dim
            ).squeeze(2)
        if ctx.index_cache is not None and ctx.slot_mapping is not None:
            # kPool index cache is unquantized (no scale side-channel).
            ctx.update_index_cache(packed.reshape(num_tokens, -1), None)

        pool_cache = None
        if (
            self.index_kpool_compress
            and ctx.index_cache is not None
            and ctx.block_table is not None
            and ctx.slot_mapping is not None
            and ctx.actual_seq_kv is not None
        ):
            pool_cache = self._pool_caches.get(layer.layer_id)
            if pool_cache is None:
                # The ACL graph runner executes a non-captured static warmup
                # before capture. Allocate there so the captured decode graph
                # sees the stable PA_BBND pool-cache address.
                pool_cache = alloc_pool_cache(ctx.index_cache)
                self._pool_caches[layer.layer_id] = pool_cache
                print(
                    f"[kpool] layer {layer.layer_id}: pool cache "
                    f"{tuple(pool_cache.shape)} "
                    f"({pool_cache.numel() * 2 / 1024**3:.2f} GiB), "
                    f"index cache {tuple(ctx.index_cache.shape)}"
                )

        if pool_cache is not None:
            n_seqs = ctx.block_table.shape[0]
            kv_lens_t = ctx.actual_seq_kv.reshape(-1).to(torch.int64)
            pos_flat = positions.reshape(-1)
            # ---- write path: compress pools completed by this step ----
            if num_tokens == n_seqs:
                # Decode (graph or eager): one token per sequence; static
                # per-seq slices keep graph capture host-sync free.
                for s in range(n_seqs):
                    compress_completed_pools(
                        ctx.index_cache, pool_cache, ctx.block_table[s:s + 1],
                        pos_flat[s:s + 1], self.index_kpool_compress_ape,
                        self.head_dim, self.index_kpool,
                    )
            else:
                # Prefill chunk (eager): per-seq position slices from the
                # host-side query lengths.
                if n_seqs == 1:
                    slices = [(0, num_tokens)]
                else:
                    q_lens_w = _current_q_seq_lens(n_seqs, num_tokens)
                    starts = [0]
                    for l in q_lens_w[:-1]:
                        starts.append(starts[-1] + l)
                    slices = list(zip(starts, q_lens_w))
                for s, (lo, hi) in enumerate(slices):
                    compress_completed_pools(
                        ctx.index_cache, pool_cache, ctx.block_table[s:s + 1],
                        pos_flat[lo:hi], self.index_kpool_compress_ape,
                        self.head_dim, self.index_kpool,
                    )

            # ---- read path: direct pool read for decode ----
            if num_tokens == n_seqs:
                max_kv_cap = getattr(
                    backend, "graph_index_history_max_kv", None
                )
                if not _in_acl_graph():
                    max_kv = int(kv_lens_t.max().item())
                elif max_kv_cap is not None:
                    # Same static cap the dense gather used, so the graph
                    # runner's eager-fallback condition stays consistent.
                    page_size = ctx.index_cache.shape[1]
                    max_kv = min(
                        ctx.block_table.shape[1] * page_size, max_kv_cap
                    )
                else:
                    # No static cap: cannot size the read without a host
                    # sync; use the dense path for this step.
                    max_kv = None
                if max_kv is not None:
                    n_pools = (max_kv + self.index_kpool - 1) // self.index_kpool
                    key_valid = (
                        torch.arange(
                            n_pools * self.index_kpool, device=pool_cache.device
                        )[None, :]
                        < kv_lens_t.reshape(-1, 1)
                    )
                    kv_len = n_pools * self.index_kpool
                    qr_bsd = qr.view(n_seqs, 1, -1)
                    hidden_bsd = hidden_states.view(n_seqs, 1, -1)
                    mask_bsd = attention_mask.view(n_seqs, 1)
                    topk_indices = self.select_topk(
                        qr_bsd, hidden_bsd, mask_bsd,
                        kv_len=kv_len, current_length=kv_len,
                        key_valid=key_valid,
                        pool_cache=pool_cache,
                        pool_block_table=ctx.block_table,
                    )
                    return (
                        topk_indices.reshape(num_tokens, 1, -1).to(torch.int32)
                    )

        packed_history = backend.gather_index_history(layer, batch_size)
        num_seqs = packed_history.shape[0]
        if num_seqs == 1:
            q_lens = [num_tokens]
        elif num_tokens == num_seqs:
            q_lens = [1] * num_seqs
        else:
            q_lens = _current_q_seq_lens(num_seqs, num_tokens)
        max_q = max(q_lens)
        is_varlen = max_q * num_seqs != num_tokens
        if not is_varlen:
            # Uniform per-sequence lengths (single sequence, packed
            # one-token decode, equal-length batch): a plain view is exact
            # and allocation-free — the decode hot path stays untouched.
            qr_bsd = qr.view(num_seqs, max_q, -1)
            hidden_bsd = hidden_states.view(num_seqs, max_q, -1)
            mask_bsd = attention_mask.view(num_seqs, max_q)
        else:
            # Varlen batch (unequal prompts prefilled together): the engine
            # flattens the batch to [1, T, D]; scatter tokens to a padded
            # [num_seqs, max_q, ...] layout by the real sequence boundaries
            # and mask the pad rows; the top-k rows are gathered back to the
            # flat [T, ...] order below.
            starts = [0]
            for seq_len_i in q_lens[:-1]:
                starts.append(starts[-1] + seq_len_i)
            device = qr.device
            starts_t = torch.tensor(starts, device=device, dtype=torch.int64)
            lens_t = torch.tensor(q_lens, device=device, dtype=torch.int64)
            q_offsets = torch.arange(max_q, device=device, dtype=torch.int64)
            src = starts_t[:, None] + q_offsets[None, :]
            valid = q_offsets[None, :] < lens_t[:, None]
            src_flat = src.clamp(max=num_tokens - 1).reshape(-1)
            qr_bsd = qr.index_select(0, src_flat).view(num_seqs, max_q, -1)
            hidden_bsd = (
                hidden_states.reshape(num_tokens, -1)
                .index_select(0, src_flat)
                .view(num_seqs, max_q, -1)
            )
            mask_bsd = (
                attention_mask.reshape(-1)
                .index_select(0, src_flat)
                .view(num_seqs, max_q)
                & valid
            )
        kv_len = packed_history.shape[1]
        history_key_valid = None
        if not self.index_kpool_compress:
            history_key_valid = (
                torch.arange(kv_len, device=packed_history.device)[None, :]
                < ctx.actual_seq_kv[:num_seqs].to(torch.int64)[:, None]
            )
        topk_indices = self.select_topk(
            qr_bsd, hidden_bsd, mask_bsd,
            kv_len=kv_len, current_length=kv_len,
            packed_states=packed_history,
            key_valid=history_key_valid,
            pool_cache=pool_cache,
            pool_block_table=ctx.block_table,
        )
        if is_varlen:
            topk_indices = topk_indices[valid]
        return topk_indices.reshape(num_tokens, 1, -1).to(torch.int32)


# ---------------------------------------------------------------------------
# MLA (absorbed, NoPE) + DSA sparse attention via the SFA op
# ---------------------------------------------------------------------------
class Glm53FlashMlaAttention(Attention):
    """Absorbed MLA (NoPE, qk_rope=0) + DSA sparse attention via SFA op.

    Mirrors ``DeepseekV3MlaAttention``'s forward: q_latent = bmm(q_nope,
    W_UK); k_latent into the paged nope cache; SFA over the kPool topk;
    v_full = bmm(attn_out, W_UV). The only divergence is the indexer — kPool
    (``select_qli`` writes packed states into the 257-wide paged index cache,
    gathers the dense history, runs ``select_topk``, adapts to SFA's
    sparse_indices) vs DS V3.2's ``lightning_indexer`` — and NoPE (rope_dim=0,
    q_pe/k_pe = None).
    """

    def __init__(self, cfg: Glm53FlashConfig, layer_id: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__(
            num_heads=cfg.n_heads, num_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
            scale=(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5,
            sliding_window=0, layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.hidden_size = cfg.hidden_size
        tp = self.cfg.tp_size
        assert self.cfg.n_heads % tp == 0, (
            f"n_heads {self.cfg.n_heads} not divisible by tp_size {tp}"
        )
        num_heads = self.cfg.n_heads // tp  # per-rank head count
        self.num_heads_local = num_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim ** -0.5
        self.eps = cfg.rms_norm_eps
        shared = cfg.indexer_shared(layer_id)
        self.indexer = None if shared else Glm53FlashIndexer(cfg, layer_id, dtype, device)

        dev, dt = device, dtype
        # q_a_proj: replicated (hidden -> q_lora)
        self.q_a_proj = QLinear(self.hidden_size, self.q_lora_rank, device=dev,
                                dtype=dt, kind="static", bias=cfg.attention_bias)
        self.q_a_layernorm = _RMSNorm(self.q_lora_rank, self.eps, dtype, device)
        # q_b_proj: column-parallel (out = num_heads_local * (qk_nope + qk_rope))
        self.q_b_proj = QLinear(self.q_lora_rank, num_heads * self.qk_head_dim,
                                device=dev, dtype=dt, kind="static")
        # kv_a: replicated (latent + rope)
        self.kv_a_proj_with_mqa = QLinear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim,
            device=dev, dtype=dt, kind="static", bias=cfg.attention_bias,
        )
        self.kv_a_layernorm = _RMSNorm(self.kv_lora_rank, self.eps, dtype, device)
        # kv_b: column-parallel fp (absorbed split reads .weight; stays fp, see
        # design §3)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            tp, dtype=dt, device=dev,
        )
        # o_proj: row-parallel + all_reduce (out = hidden_size)
        self.o_proj = QLinear(num_heads * self.v_head_dim, self.hidden_size,
                              device=dev, dtype=dt, kind="static",
                              row_parallel=True, bias=cfg.attention_bias)
        # Absorbed-MLA: kv_b_proj weight = [num_heads*(qk_nope+v_hd), kv_lora].
        # Split into W_UK [H, qk_nope, kv_lora] and W_UV [H, kv_lora, v_hd]
        # (W_UV stored transposed, matching deepseek_v32 / glm5_2: bmm with the
        # [H, T, kv_lora] attn_out yields [H, T, v_hd]). Dead until Task 7
        # routes DSA through backend.execute_mla.
        self.register_buffer(
            "W_UK", torch.zeros(
                num_heads, self.qk_nope_head_dim, self.kv_lora_rank,
                dtype=dtype, device=device,
            ), persistent=False,
        )
        self.register_buffer(
            "W_UV", torch.zeros(
                num_heads, self.kv_lora_rank, self.v_head_dim,
                dtype=dtype, device=device,
            ), persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        # Split kv_b_proj.weight into absorbed W_UK / W_UV (mirrors
        # deepseek_v32.process_weights_after_loading split). glm5_3_flash is NoPE
        # (qk_rope_head_dim=0), so qk_head_dim == qk_nope_head_dim.
        w = self.kv_b_proj.weight.data
        w = w.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = w.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Absorbed MLA forward (mirrors DeepseekV3MlaAttention.forward).

        NoPE (qk_rope_head_dim=0): q_pe/k_pe are None and rope is skipped. The
        kPool indexer's ``select_qli`` writes packed states into the paged index
        cache, gathers the dense history, runs ``select_topk`` and adapts the
        output to the SFA op's ``sparse_indices`` — the same contract as DS
        V3.2's ``DeepseekV3Indexer.select_qli``.
        """
        num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        hidden = hidden_states.view(num_tokens, -1)
        q_a = self.q_a_proj(hidden)
        q_c = self.q_a_layernorm(q_a)
        backend = get_forward_context().attention_backend
        topk = None
        if self.indexer is not None:
            ctx = backend.mla_index_context(self)
            topk = self.indexer.select_qli(
                hidden_states, q_c, position_ids, attention_mask, ctx, self, backend
            )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer."
                )
            topk = prev_topk_indices.reshape(num_tokens, 1, -1).to(torch.int32)

        q = self.q_b_proj(q_c).view(
            num_tokens, self.num_heads_local, self.qk_head_dim
        )
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_latent = torch.bmm(q_nope.transpose(0, 1), self.W_UK).transpose(0, 1)

        # NoPE: qk_rope_head_dim == 0 -> q_rope/k_rope are empty -> q_pe/k_pe None.
        q_pe = None

        kv = self.kv_a_proj_with_mqa(hidden)
        k_latent_raw, k_rope_raw = kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)
        k_pe = None

        attn_out = backend.execute_mla(
            q_latent, q_pe, k_latent_3d, k_pe, self, topk=topk
        )
        v_full = torch.bmm(
            attn_out.transpose(0, 1), self.W_UV
        ).transpose(0, 1)
        v_full = v_full.reshape(num_tokens, self.num_heads_local * self.v_head_dim)
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(o)
        return o, topk

# ---------------------------------------------------------------------------
# MLP (dense + MoE)
# ---------------------------------------------------------------------------
class Glm53FlashMLP(nn.Module):
    def __init__(self, cfg: Glm53FlashConfig, intermediate_size: int,
                 dtype: torch.dtype, device: torch.device,
                 skip_tp_reduce: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.skip_tp_reduce = skip_tp_reduce
        self.swiglu_limit = cfg.swiglu_limit
        # TP: column-parallel SwiGLU (gate_up sharded on the intermediate dim)
        # + row-parallel down_proj (sharded on the input dim, summed via
        # all_reduce_ in forward). At tp==1 inter_local == intermediate_size so
        # the fp graph is byte-identical to the previous nn.Linear trio.
        tp = cfg.tp_size
        inter_local = intermediate_size // tp
        self.gate_up_proj = QLinear(
            cfg.hidden_size, 2 * inter_local, device=device,
            dtype=dtype, kind="dynamic",
        )
        self.down_proj = QLinear(
            inter_local, cfg.hidden_size, device=device,
            dtype=dtype, kind="dynamic", row_parallel=True,
        )

    def process_weights_after_loading(self) -> None:
        # W8A8: each QLinear's ``_w8a8`` submodule must run its
        # ``W8A8DynamicLinear.process_weights_after_loading`` transpose, which
        # flips the loaded ``[out, in]`` weight into ``[in, out]`` that
        # ``quant_matmul`` (transpose2=False) consumes. ``_call_process_weights``
        # is non-recursive, so without this forward Glm53FlashMLP leaves the
        # ``gate_up_proj`` / ``down_proj`` int8 weights untransposed and the
        # matmul aborts on a dim mismatch (hidden vs 2*inter_local). The bf16
        # path is a no-op (QLinear.process_weights_after_loading guards on
        # ``use_w8a8``), so this is safe for both.
        _call_process_weights_after_loading(self.gate_up_proj)
        _call_process_weights_after_loading(self.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        # GLM-5.3-Flash SwiGLU clamp (matches HF Glm53FlashTextMLP).
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out = self.down_proj(F.silu(gate) * up)
        if self.cfg.tp_size > 1 and not self.skip_tp_reduce:
            distributed.all_reduce_(out)
        return out


class Glm53FlashExperts(nn.Module):
    """3D-stacked experts (SwiGLU) dispatched via one-hot mask + index_add_.

    The 3D params are sized by the LOCAL (TP-sharded) intermediate so each rank
    holds a column-slice of the expert weights; the per-expert matmul emits a
    partial-sum along ``down_proj``'s output that the MoE's final all_reduce_
    combines across ranks.
    """

    def __init__(self, cfg: Glm53FlashConfig, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.num_experts = cfg.n_routed_experts
        self.tp = cfg.tp_size
        self.swiglu_limit = cfg.swiglu_limit
        # Local shard of the expert intermediate (inter // tp).
        self.intermediate_dim = cfg.moe_intermediate_size // cfg.tp_size
        self.hidden_dim = cfg.hidden_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim,
                        dtype=dtype, device=device)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim,
                        dtype=dtype, device=device)
        )

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        # Deterministic fp32 accumulation: index_add_ scatters in bf16 and (per
        # the transformers moe.py note) is non-deterministic on accelerator
        # atomicAdd. Accumulate each (token, topk-slot) expert contribution in a
        # fp32 [n_tokens, topk, hidden] buffer and sum the topk axis — stable
        # and matches the reference's eager path exactly.
        #
        # Two paths:
        #  1) Graph-friendly: batched gather + bmm — fixed-shape ops that stay
        #     compatible with ACL graph capture (no dynamic nonzero/torch.where).
        #  2) Eager: per-expert loop with nonzero/torch.where — memory-efficient
        #     for large n_tokens (prefill) where the batched gather would OOM.
        ctx = get_forward_context_or_none()
        if ctx is not None and ctx.acl_graph is not None:
            return self._forward_graph_friendly(
                hidden_states, top_k_index, top_k_weights)
        return self._forward_eager(hidden_states, top_k_index, top_k_weights)

    def _forward_eager(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, topk = top_k_index.shape
        hidden = hidden_states.shape[-1]
        final_f32 = torch.zeros(
            n_tokens, topk, hidden, dtype=torch.float32, device=hidden_states.device
        )
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            # GLM-5.3-Flash SwiGLU clamp (matches HF Glm53FlashTextExperts).
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_f32[token_idx, top_k_pos] += current.float()
        return final_f32.sum(1).to(hidden_states.dtype)

    def _forward_graph_friendly(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape batched expert dispatch for ACL graph capture.

        Replaces the eager per-expert loop (nonzero → torch.where → F.linear)
        with gather + bmm whose output shapes are static regardless of which
        experts are selected.  This avoids ``aclnnNonzero`` -triggered stream
        syncs that are illegal inside a captured stream.
        """
        n_tokens, topk = top_k_index.shape
        hidden = hidden_states.shape[-1]
        N = n_tokens * topk  # total number of token-expert assignments

        flat_indices = top_k_index.flatten()  # [N]
        flat_weights = top_k_weights.flatten()  # [N]

        # Repeat each token's hidden state for every expert slot it has.
        h = hidden_states.repeat_interleave(topk, dim=0)  # [N, hidden]

        # Gather the expert weight slices for all assignments at once.
        gate_up_w = self.gate_up_proj[flat_indices]  # [N, 2*inter_dim, hidden]
        down_w = self.down_proj[flat_indices]        # [N, hidden, inter_dim]

        # Batched gate + up projection.
        # [N, 1, hidden] × [N, hidden, 2*inter_dim] → [N, 1, 2*inter_dim]
        gate_up = torch.bmm(
            h.unsqueeze(1), gate_up_w.transpose(1, 2),
        ).squeeze(1)  # [N, 2*inter_dim]

        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # Batched down projection.
        # [N, 1, inter_dim] × [N, inter_dim, hidden] → [N, 1, hidden]
        current = torch.bmm(
            (F.silu(gate) * up).unsqueeze(1), down_w.transpose(1, 2),
        ).squeeze(1)  # [N, hidden]

        # Apply router weights and accumulate in fp32 for determinism.
        current = (current * flat_weights.unsqueeze(-1)).float()
        current = current.view(n_tokens, topk, hidden)
        final_f32 = torch.zeros(
            n_tokens, topk, hidden, dtype=torch.float32,
            device=hidden_states.device,
        )
        final_f32 = final_f32 + current
        return final_f32.sum(1).to(hidden_states.dtype)


class Glm53FlashMoE(nn.Module):
    def __init__(self, cfg: Glm53FlashConfig, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        tp = cfg.tp_size
        self.num_experts = cfg.n_routed_experts
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scaling = cfg.routed_scaling_factor
        self.moe_inter = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        assert self.moe_inter % tp == 0, (
            f"moe_intermediate_size {self.moe_inter} not divisible by tp {tp}")
        self.inter_local = self.moe_inter // tp
        self.use_w8a8: bool = False  # set in load_weights via probe_quant

        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False,
                              dtype=dtype, device=device)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            persistent=False,
        )

        # --- W8A8 int8 experts (照搬 DeepseekV3MoE, deepseek_v32.py:755-796) ---
        # LAZY: the int8 params are registered in _load_experts_w8a8 (the
        # per-layer loader probes the checkpoint first). Eager allocation here
        # costs ~17GB/card at TP16 of idle int8 on the bf16 path, which together
        # with the bf16 experts OOMs the card (KV-cache estimation aborts with
        # 0 available bytes).
        # --- bf16 experts branch: LAZY too (built only in _load_experts_bf16).
        # Eagerly building Glm53FlashExperts' 3D bf16 params (~38GB/card at TP8)
        # alongside the int8 experts OOMs the 60GB card.
        self.experts: Optional[Glm53FlashExperts] = None

        self.shared_experts = Glm53FlashMLP(
            cfg, cfg.moe_intermediate_size * cfg.n_shared_experts, dtype, device,
            skip_tp_reduce=True,
        )

    def process_weights_after_loading(self) -> None:
        """W8A8: transpose + NZ-format expert weights (照搬 DeepseekV3MoE).
        bf16: no-op on int8 expert weights, only shared experts processed."""
        if not self.use_w8a8:
            # bf16 分支: int8 experts 未加载, 不动; shared 走自己的 process
            _call_process_weights_after_loading(self.shared_experts)
            return
        assert torch.all(self.experts_w13_offset == 0), (
            "Glm53FlashMoE int8-grouped path needs symmetric int8 experts "
            "(experts_w13_offset == 0)")
        assert torch.all(self.experts_w2_offset == 0), (
            "Glm53FlashMoE int8-grouped path needs symmetric int8 experts "
            "(experts_w2_offset == 0)")
        # Transpose + NZ format-cast with the raw layout released before the
        # cast (mirrors DeepseekV3MoE._format_and_release_expert_weight): ACL
        # graph capture must see the NZ tensor as the sole allocation, not an
        # NCL allocation overlaid by a format view — otherwise the GMM kernel
        # aborts at capture with "weight Format expect FRACTAL_NZ, but got
        # [NCL]". Numerically identical to the previous transpose + cast.
        for param in (self.experts_w13, self.experts_w2):
            transposed = param.data.transpose(1, 2).contiguous()
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            param.data = kernels.format_cast_nz(transposed)
            del transposed
        self.experts_w13_scale.data = self.experts_w13_scale.data.view(
            self.num_experts, -1).contiguous()
        self.experts_w13_offset.data = self.experts_w13_offset.data.view(
            self.num_experts, -1).contiguous()
        self.experts_w2_scale.data = self.experts_w2_scale.data.view(
            self.num_experts, -1).contiguous()
        self.experts_w2_offset.data = self.experts_w2_offset.data.view(
            self.num_experts, -1).contiguous()
        _call_process_weights_after_loading(self.shared_experts)

    def _topk(self, hidden_states: torch.Tensor):
        cfg = self.cfg
        hidden_states = hidden_states.view(-1, cfg.hidden_size)
        router_logits = F.linear(
            hidden_states.float(), self.gate.weight.float()
        )
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, cfg.n_group, cfg.n_routed_experts // cfg.n_group)
            .topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=cfg.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, cfg.n_group, cfg.n_routed_experts // cfg.n_group)
            .reshape(-1, cfg.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * cfg.routed_scaling_factor
        return router_logits, topk_weights, topk_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden)
        if self.use_w8a8:
            # W8A8: fused grouped_moe kernel does routing+quantized matmul
            # (sigmoid + noaux_tc + norm_topk_prob, routed_scaling=1.0 inside).
            # Mirrors DeepseekV3MoE.forward (deepseek_v32.py:830-850).
            logits = self.gate(flat)
            routed = kernels.grouped_moe(
                flat, logits,
                self.experts_w13, self.experts_w2,
                self.experts_w13_scale, self.experts_w2_scale,
                self.e_score_correction_bias,
                self.topk, self.topk_group, self.n_group,
                self.cfg.norm_topk_prob,
                # Kernel applies routed_scaling inside its fused top-k; pass
                # 1.0 and keep the external multiply below (pre-graph
                # behavior, numerically identical since scaling is linear).
                routed_scaling_factor=1.0,
            )
            routed = routed * self.routed_scaling
            out = routed.view(*orig_shape)
        else:
            # bf16: existing _topk routing + Glm53FlashExperts per-expert loop.
            _, topk_weights, topk_indices = self._topk(hidden_states)
            # Debug hooks read these to compare the router against the reference.
            self._last_topk_weights = topk_weights.detach()
            self._last_topk_indices = topk_indices.detach()
            out = self.experts(flat, topk_indices, topk_weights).view(*orig_shape)
        final = out + self.shared_experts(hidden_states)
        # Single TP all-reduce on the combined routed+shared output. shared_experts
        # is built with skip_tp_reduce=True so its row-parallel reduce folds into
        # this one call (matches deepseek_v32 DeepseekV3MoE).
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(final)
        return final

    def gate_call(self, hidden_states: torch.Tensor):
        # routed via the local _topk (mirrors Glm53FlashTopkRouter).
        return self._topk(hidden_states)


# ---------------------------------------------------------------------------
# mHC (Manifold-constrained Hyper-Connection) residual — faithful port of
# transformers Glm53FlashTextHyperConnection / Glm53FlashTextHyperHead.
# ---------------------------------------------------------------------------
class Glm53FlashHyperConnection(nn.Module):
    """4-stream hyper-connection residual (reference 216-292).

    Owns the learned (fn, base, scale) parameters that turn the incoming
    ``hc_mult`` residual streams into collapse/expand weights. ``forward``
    returns ``(post, comb, collapsed)``: ``post`` scales the sublayer output
    per stream, ``comb`` is the Sinkhorn doubly-stochastic 4x4 stream mixer,
    ``collapsed`` is the single-sequence input to feed the sublayer.
    """

    def __init__(self, cfg: "Glm53FlashConfig", dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.input_norm = _UnweightedRMSNorm(cfg.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(
            torch.empty(mix, self.hc_mult * cfg.hidden_size, dtype=torch.float32, device=device)
        )
        self.base = nn.Parameter(torch.empty(mix, dtype=torch.float32, device=device))
        # 3 outputs: pre (collapse), post (placement), comb (mixer) scales.
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device))

    def forward(self, hidden_streams: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult

        if _has_mhc_fused:
            # --- Fused NPU kernel path (matches DeepSeek V4 C++ decoder layer) ---
            # hc_pre does rsqrt + linear + sinkhorn + weighted-sum-reduce in one
            # fused call.  x: [B, S, hc_mult, D] -> (output [B,S,D], post [B,S,hc_mult],
            # comb [B,S,hc_mult,hc_mult]).
            # .float() guards against weight-loading casting params back to bf16;
            # __init__ creates them in float32 (aligned with vLLM).
            collapsed, post, comb = kernels.hc_pre(
                hidden_streams,
                self.fn,
                self.scale.float(),
                self.base.float(),
                hc,
                self.hc_sinkhorn_iters,
                self.input_norm.variance_epsilon,
                self.hc_eps,
            )
            return post, comb, collapsed
        else:
            # --- Fallback: pure-Python reference implementation ---
            flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
            pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
            pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
            pre_scale, post_scale, comb_scale = self.scale.unbind(0)

            pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
            post = 2 * torch.sigmoid(post_w * post_scale + post_b)
            comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
            comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            for _ in range(self.hc_sinkhorn_iters - 1):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
            return post, comb, collapsed


class Glm53FlashHyperHead(nn.Module):
    """Final mHC stream collapse — unweighted mean over the hc_mult streams."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------
class Glm53FlashDecoderLayer(nn.Module):
    def __init__(self, cfg: Glm53FlashConfig, layer_id: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        if cfg.is_dsa(layer_id):
            self.self_attn = Glm53FlashMlaAttention(cfg, layer_id, dtype, device)
        else:
            self.self_attn = Glm53FlashKdaAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        if cfg.is_moe(layer_id):
            self.mlp = Glm53FlashMoE(cfg, dtype, device)
        else:
            self.mlp = Glm53FlashMLP(cfg, cfg.intermediate_size, dtype, device)
        # mHC residual sites (always on, per reference — `mhc` config is not consulted).
        self.attn_hc = Glm53FlashHyperConnection(cfg, dtype, device)
        self.ffn_hc = Glm53FlashHyperConnection(cfg, dtype, device)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                prev_topk_indices: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: [B, S, hc_mult, D] (4 residual streams).
        if _has_mhc_fused:
            return self._forward_fused(hidden_states, position_ids,
                                       attention_mask, prev_topk_indices)
        else:
            return self._forward_ref(hidden_states, position_ids,
                                     attention_mask, prev_topk_indices)

    def _forward_fused(self, hidden_states: torch.Tensor,
                       position_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       prev_topk_indices: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # --- Fused NPU kernel path (matches DeepSeek V4 C++ decoder layer) ---

        # Attention mHC pre-collapse
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, position_ids, attention_mask,
                                  prev_topk_indices)
        if isinstance(attn_out, tuple):
            hidden_states, topk = attn_out
        else:
            hidden_states, topk = attn_out, None
        # MLA attention returns [B*S, D] (2D); reshape to [B, S, D] for hc_post.
        # KDA attention already returns [B, S, D] so the dim check is a no-op.
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.view(
                residual.shape[0], residual.shape[1], -1)
        # Fused post-attention mHC recombination: hc_post returns [B,S,hc_mult,D]
        hidden_states = kernels.hc_post(
            hidden_states, residual, post, comb,
        )

        # FFN mHC pre-collapse
        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # MLP / MoE preserves shape, so hidden_states is already [B, S, D] (3D).
        hidden_states = kernels.hc_post(
            hidden_states, residual, post, comb,
        )
        return hidden_states, topk

    def _forward_ref(self, hidden_states: torch.Tensor,
                     position_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     prev_topk_indices: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # --- Fallback: pure-Python reference implementation ---
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)   # collapsed -> [B,S,D]
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, position_ids, attention_mask,
                                  prev_topk_indices)
        if isinstance(attn_out, tuple):
            hidden_states, topk = attn_out
        else:
            hidden_states, topk = attn_out, None
        # MLA attention returns [B*S, D] (2D); reshape to [B, S, D].
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.view(
                residual.shape[0], residual.shape[1], -1)
        dtype = hidden_states.dtype
        hidden_states = (
            post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2)
            + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        dtype = hidden_states.dtype
        hidden_states = (
            post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2)
            + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)
        )
        return hidden_states, topk


class Glm53FlashModel(nn.Module):
    def __init__(self, cfg: Glm53FlashConfig, dtype: torch.dtype,
                 device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // cfg.tp_size, cfg.tp_size,
            dtype=dtype, device=device,
        )
        self.layers = nn.ModuleList(
            [Glm53FlashDecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)]
        )
        self.norm = _RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hc_head = Glm53FlashHyperHead()
        # Set externally by the VL composer (get_input_embeddings) before the
        # runner drives forward(); when set, it replaces embed_tokens(input_ids)
        # so image/video embeddings merged into the sequence are used as-is. The
        # mHC 4-stream expand below applies to the merged hidden identically.
        self._inputs_embeds: Optional[torch.Tensor] = None

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # The C++ EagerRunner passes the flattened token tensor, which is 1-D
        # ``[num_tokens]`` for a single sequence; normalise to ``[B, S]``.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if self._inputs_embeds is not None:
            hidden = self._inputs_embeds
            self._inputs_embeds = None
        else:
            hidden = self.embed_tokens(input_ids)
        batch_size, seq_len = hidden.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=hidden.device
            )
        # Expand embedding to hc_mult residual streams (all streams start
        # identical — reference model forward, modeling line 1521). NoPE: no
        # position embeddings are computed or threaded (reference passes None).
        hidden = hidden.unsqueeze(2).expand(-1, -1, self.cfg.hc_mult, -1).contiguous()
        prev_topk: Optional[torch.Tensor] = None
        _trace = os.environ.get("GLM5_NEXT_TRACE")
        for i, layer in enumerate(self.layers):
            hidden, prev_topk = layer(hidden, position_ids, attention_mask, prev_topk)
        if _trace:
            print("[trace] final_norm", flush=True)
        # Final collapse: unweighted mean over the streams, then RMSNorm
        # (reference `self.norm(self.hc_head(hidden_states))`, line 1537).
        # Flatten [B, S, D] -> [B*S, D]: the engine's compute_logits does
        # ``hidden.index_select(0, selected_idxes)`` where selected_idxes are
        # token ids in the flattened sequence, so a 3-D output would select the
        # wrong (batch) axis and gather out of range for multi-token prefill.
        h = self.norm(self.hc_head(hidden)).view(-1, self.cfg.hidden_size)
        return h


def _install_dump_hooks(model: "Glm53FlashModel", lm_head: nn.Module,
                        outdir: str) -> list:
    """Capture per-layer/submodule + final-norm tensors during the engine's
    forward and save them after every ``model.forward``: the first (prefill)
    call writes ``<outdir>/tensors.safetensors``, subsequent single-token
    decode calls write ``<outdir>/decode_<k>.safetensors``. Keys mirror
    ``dump_model.py`` so ``compare_dumps.py`` works across xllm-engine vs
    transformers-ref; decode dumps let multi-step state be compared
    layer-by-layer.

    The model forward returns the final hidden (post hc_head + norm) for ALL
    positions; that hidden (``final_norm``) plus per-layer outputs are the
    comparable tensors (logits are computed separately by compute_logits, only
    for the sampled position, so they are not dumped here).
    """
    from safetensors.torch import save_file
    store: dict = {}
    handles: list = []
    # Per-forward counter: forward 0 is prefill (written as tensors.safetensors
    # for the existing compare tooling); decode steps write decode_<k>.safetensors
    # so multi-step state can be compared layer-by-layer against the reference.
    step = [0]

    def _cap(name: str, idx: int = 0):
        def fn(_m, _i, o):
            if _capturing_acl_graph():
                return
            t = o[idx] if isinstance(o, (tuple, list)) else o
            if t is None:
                return
            if t.dtype in (torch.int32, torch.int64, torch.bool):
                store[name] = t.detach().cpu()
            else:
                store[name] = t.detach().to(torch.float32).cpu()
        return fn

    def _pre_cap(name: str):
        def fn(_m, inp):
            if _capturing_acl_graph():
                return
            t = inp[0] if isinstance(inp, (tuple, list)) else inp
            store[name] = t.detach().to(torch.float32).cpu()
        return fn

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(_cap(f"layer{i}", 0)))
        p = f"L{i}."
        handles.append(layer.input_layernorm.register_forward_hook(_cap(p + "input_layernorm")))
        # mHC internals: layer input streams, the post-attn recombination (the
        # input to ffn_hc), and the learned post/comb weights — to localize the
        # first divergence between the attention site and the MLP site.
        handles.append(layer.attn_hc.register_forward_pre_hook(_pre_cap(p + "attn_hc.input")))
        handles.append(layer.ffn_hc.register_forward_pre_hook(_pre_cap(p + "ffn_hc.input")))
        handles.append(layer.attn_hc.register_forward_hook(_cap(p + "attn_hc.post", 0)))
        handles.append(layer.attn_hc.register_forward_hook(_cap(p + "attn_hc.comb", 1)))
        handles.append(layer.ffn_hc.register_forward_hook(_cap(p + "ffn_hc.post", 0)))
        handles.append(layer.ffn_hc.register_forward_hook(_cap(p + "ffn_hc.comb", 1)))
        handles.append(layer.attn_hc.register_forward_hook(_cap(p + "attn_hc.collapse", 2)))
        handles.append(layer.self_attn.register_forward_hook(_cap(p + "self_attn", 0)))
        sa = layer.self_attn
        if hasattr(sa, "o_norm"):
            # KDA internals: isolate where the first divergence enters the
            # attention (conv1d output, forget gate, gated-norm output).
            handles.append(sa.conv1d.register_forward_hook(_cap(p + "self_attn.conv1d")))
            handles.append(sa.forget_gate.register_forward_hook(_cap(p + "self_attn.forget_gate")))
            handles.append(sa.o_norm.register_forward_hook(_cap(p + "self_attn.o_norm")))
        if hasattr(layer.self_attn, "o_proj"):
            handles.append(layer.self_attn.o_proj.register_forward_hook(_cap(p + "self_attn.o_proj")))
        if getattr(layer.self_attn, "indexer", None) is not None:
            handles.append(layer.self_attn.indexer.register_forward_hook(_cap(p + "self_attn.indexer.topk")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_cap(p + "post_attention_layernorm")))
        handles.append(layer.ffn_hc.register_forward_hook(_cap(p + "ffn_hc.collapse", 2)))
        handles.append(layer.mlp.register_forward_hook(_cap(p + "mlp")))
        if hasattr(layer.mlp, "shared_experts") and getattr(layer.mlp, "shared_experts", None) is not None:
            handles.append(layer.mlp.shared_experts.register_forward_hook(_cap(p + "mlp.shared_experts")))
        if hasattr(layer.mlp, "experts") and getattr(layer.mlp, "experts", None) is not None:
            handles.append(layer.mlp.experts.register_forward_hook(_cap(p + "mlp.experts")))
            def _moe_cur(_m, _i, _o, _l=layer, _p=p):
                cs = getattr(_l.mlp.experts, "_debug_current_sum", None)
                if cs is not None:
                    store[_p + "mlp.experts_current_sum"] = cs.to(torch.float32).cpu()
            handles.append(layer.mlp.experts.register_forward_hook(_moe_cur))
            # router topk weights/indices (stored by Glm53FlashMoE.forward)
            def _moe_router(_m, _i, _o, _l=layer, _p=p):
                if hasattr(_l.mlp, "_last_topk_weights"):
                    store[_p + "mlp.topk_weights"] = (
                        _l.mlp._last_topk_weights.to(torch.float32).cpu())
                    store[_p + "mlp.topk_indices"] = _l.mlp._last_topk_indices.cpu()
            handles.append(layer.mlp.register_forward_hook(_moe_router))
    handles.append(model.norm.register_forward_hook(_cap("final_norm")))

    def _save(_m, _i, _o):
        if _capturing_acl_graph():
            # D2H dumps are illegal mid-capture; warmup forwards already
            # saved the same static-input values.
            return
        os.makedirs(outdir, exist_ok=True)
        _save_file = {k: v.contiguous().cpu() for k, v in store.items()}
        if step[0] == 0:
            fname = os.path.join(outdir, "tensors.safetensors")
        else:
            fname = os.path.join(outdir, f"decode_{step[0] - 1}.safetensors")
        save_file(_save_file, fname)
        print(f"[glm5_3_flash dump] saved {len(_save_file)} tensors to {fname}",
              flush=True)
        store.clear()
        step[0] += 1

    handles.append(model.register_forward_hook(_save))
    return handles


def _resolve_module(root: nn.Module, dotted: str) -> nn.Module:
    """Walk a dotted parameter name (``model.layers.0.self_attn.q_a_proj``)
    to the owning submodule, handling integer ``ModuleList`` indices."""
    obj: nn.Module = root
    for part in dotted.rstrip(".").split("."):
        if part.isdigit():
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj


def _w8a8_shard_dims(fp_dim: Optional[int]) -> Optional[dict]:
    """Static-W8A8 shard map for ``load_w8a8_a`` from the fp shard dim.

    - ``fp_dim is None`` (replicated, e.g. q_a_proj / kv_a_proj): no shard.
    - ``fp_dim == 0`` (column-parallel, e.g. q_b_proj): weight + deq_scale +
      quant_bias shard dim 0 (the W8A8 dequant/quant buffers are per-output).
    - ``fp_dim == 1`` (row-parallel, e.g. o_proj): only the weight shards dim 1;
      deq_scale/quant_bias are unsharded (the output is gathered, not split).
    Mirrors deepseek_v32's ``load_weights`` shard-dim choices.
    """
    if fp_dim is None:
        return None
    if fp_dim == 1:
        return {"weight": 1}
    return {"weight": 0, "deq_scale": 0, "quant_bias": 0}


def _call_process_weights_after_loading(module: nn.Module) -> None:
    """Invoke ``process_weights_after_loading`` if the module defines it.

    The fp path is a no-op for QLinear (only DSA's ``kv_b_proj`` W_UK/W_UV split
    does real work), but the call is harmless and keeps the w8a8 path ready."""
    fn = getattr(module, "process_weights_after_loading", None)
    if fn is not None:
        fn()


_HC_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.(attn|ffn)_hc\.(fn|base|scale)$")


def _real_ckpt_aliases(name: str) -> list:
    """Candidate real-checkpoint keys for a model state-dict key.

    The official hf checkpoint nests text weights under ``model.language_model.``
    (multimodal container) and flattens mHC params as ``hc_{site}_{param}``,
    while our model state-dict uses bare ``model.`` and ``{site}_hc.{param}``.
    Tries the alias list in order; the caller falls back through them.
    """
    out = []
    m = _HC_PATTERN.match(name)
    if m:  # mHC: {site}_hc.{param} -> hc_{site}_{param} (+ language_model)
        i, site, param = m.group(1), m.group(2), m.group(3)
        out.append(f"model.language_model.layers.{i}.hc_{site}_{param}")
    # forget_gate nesting: model self_attn.forget_gate.{x} -> real flat self_attn.{x}
    # (f_a_proj/f_b_proj/dt_bias/A_log). Strip the forget_gate. segment.
    if ".self_attn.forget_gate." in name:
        out.append(name.replace(".self_attn.forget_gate.", ".self_attn."))
    if name.startswith("model.") and not name.startswith("model.language_model."):
        out.append("model.language_model." + name[len("model."):])
    # Also try stripping forget_gate on the prefixed alias (applied after the
    # prefix transform above so order matters: rebuild from the base name).
    if ".self_attn.forget_gate." in name and name.startswith("model."):
        base = name.replace(".self_attn.forget_gate.", ".self_attn.")
        out.append("model.language_model." + base[len("model."):])
    return out


class Glm53FlashForCausalLM(PyModelBase):
    """glm5_3_flash causal LM. Registered under model_type='glm5_3_flash'."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Glm53FlashConfig.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get("tp_rank", 0))
        for name, val in (("n_heads", self.cfg.n_heads),
                          ("intermediate_size", self.cfg.intermediate_size),
                          ("moe_intermediate_size", self.cfg.moe_intermediate_size),
                          ("vocab_size", self.cfg.vocab_size),
                          ("hidden_size", self.cfg.hidden_size)):
            assert val % self.cfg.tp_size == 0, (
                f"{name} {val} not divisible by tp {self.cfg.tp_size}"
            )
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0" if torch_npu else "cpu"))
        self.dtype = dtype
        self.device = device
        self.model = Glm53FlashModel(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size, self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size, gather_output=True, dtype=dtype, device=device,
        )
        # The layer projections are created without an explicit dtype (they
        # default to float32 on the target device); move AND cast the whole
        # graph to the target dtype/device so the engine matches the reference
        # (which runs .to(bf16) over the model). The KDA conv1d is included in
        # this cast — the reference's conv1d is also bf16 after .to(dtype).
        self.to(device=device, dtype=dtype)
        _dump_dir = os.environ.get("GLM5_NEXT_DUMP_DIR")
        if _dump_dir:
            # TP4: each rank writes the SAME filename -> they clobber each other.
            # Split into a per-rank subdir so all 4 ranks' views survive.
            try:
                _rk = distributed.tp_rank(device)
            except Exception:
                _rk = 0
            _dump_dir = os.path.join(_dump_dir, f"rank{_rk}")
            _install_dump_hooks(self.model, self.lm_head, _dump_dir)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids, attention_mask)
        return self.lm_head(hidden)

    # -- weight loading ---------------------------------------------------
    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        from xllm.python.layers.qlinear import QLinearWeightLoader
        L = QLinearWeightLoader(self, state_dicts, tp_size, tp_rank)

        # Wrap the underlying loader's find/load_tensor with real-checkpoint key
        # aliasing (model. -> model.language_model. + mHC {site}_hc.{param} ->
        # hc_{site}_{param}). Exact key (fake weights) is tried first, then the
        # real-ckpt aliases, so both fake and real checkpoints load. load_tensor
        # must resolve to the aliased name too (it calls sd.get_tensor(name)).
        _orig_find = L._w8.find
        _orig_load_tensor = L._w8.load_tensor

        def _resolve(name):
            if _orig_find(name) is not None:
                return name
            for alias in _real_ckpt_aliases(name):
                if _orig_find(alias) is not None:
                    return alias
            return None

        def _aliased_find(name):
            resolved = _resolve(name)
            return _orig_find(resolved) if resolved is not None else None

        def _aliased_load_tensor(name):
            resolved = _resolve(name)
            assert resolved is not None, f"checkpoint tensor not found: {name}"
            return _orig_load_tensor(resolved)

        L._w8.find = _aliased_find
        L._w8.load_tensor = _aliased_load_tensor
        L.find = _aliased_find
        L.load_tensor = _aliased_load_tensor

        # embed_tokens: HiddenParallelEmbedding — shard the hidden dim (dim 1).
        L.load_fp("model.embed_tokens.weight", dim=1)
        for i in range(self.cfg.n_layers):
            p = f"model.layers.{i}."
            L.load_fp(p + "input_layernorm.weight")
            L.load_fp(p + "post_attention_layernorm.weight")
            attn = p + "self_attn."
            if self.cfg.is_dsa(i):
                self._load_dsa_attn(L, attn, i)
            else:
                self._load_kda_attn(L, attn, i)
            self._load_mlp(L, p + "mlp.", i)
            # mHC residual sites (always on, replicated).
            for hc in ("attn_hc", "ffn_hc"):
                for w in ("fn", "base", "scale"):
                    L.load_fp(p + hc + "." + w)
        L.load_fp("model.norm.weight")
        # lm_head: ColumnParallelLinear — shard the vocab dim (dim 0).
        L.load_fp("lm_head.weight", dim=0)

    def _load_qlinear(
        self, L, prefix: str, proj: str, fp_dim: Optional[int],
    ) -> None:
        """Probe fp-vs-w8a8 for a QLinear projection, resolve, and load.

        ``fp_dim`` is the TP shard dim for the fp ``.weight`` (None = replicated,
        0 = column-parallel, 1 = row-parallel). The w8a8 path routes static
        projections through ``load_w8a8_into_qlinear``, which writes into the
        QLinear's ``_w8a8`` submodule; dynamic ones never reach here (MLP uses
        ``_load_mlp_fp_or_w8a8`` -> ``load_w8a8_mlp_into_qlinear``). The w8a8
        branch is NOT exercised until real w8a8 checkpoints exist (Task 10); it
        is written to match deepseek_v32's documented shard-dim contract.
        """
        mod = _resolve_module(self, prefix + proj)
        assert isinstance(mod, QLinear), f"{prefix}{proj} is not a QLinear"
        if L.probe_quant(prefix, proj):
            mod.resolve_quant(True)
            shard_dims = _w8a8_shard_dims(fp_dim)
            mod.load_w8a8(L, prefix, proj, shard_dims)
        else:
            mod.resolve_quant(False)
            L.load_fp(prefix + proj + ".weight", dim=fp_dim)

    def _load_dsa_attn(self, L, attn: str, i: int) -> None:
        # q_a_proj / kv_a_proj_with_mqa: replicated QLinear (no shard).
        self._load_qlinear(L, attn, "q_a_proj", None)
        self._load_qlinear(L, attn, "kv_a_proj_with_mqa", None)
        # q_b_proj: column-parallel QLinear (shard dim 0).
        self._load_qlinear(L, attn, "q_b_proj", 0)
        # o_proj: row-parallel QLinear (shard dim 1).
        self._load_qlinear(L, attn, "o_proj", 1)
        # kv_b_proj: fp ColumnParallelLinear (NOT a QLinear) — column-parallel.
        L.load_fp(attn + "kv_b_proj.weight", dim=0)
        L.load_fp(attn + "q_a_layernorm.weight")
        L.load_fp(attn + "kv_a_layernorm.weight")
        if not self.cfg.indexer_shared(i):
            idx = attn + "indexer."
            # Indexer projections are plain nn.Linear and stay replicated: the
            # sparse kPool mask is shared across TP ranks (each rank attends its
            # head-subset to the SAME mask), so wq_b/wk/weights_proj hold the
            # full index-head tensors on every rank.
            for w in ("wq_b.weight", "wk.weight", "weights_proj.weight"):
                L.load_fp(idx + w)
            L.load_fp(idx + "index_kpool_compress_gate")
            L.load_fp(idx + "k_norm.weight")
            L.load_fp(idx + "k_norm.bias")
            L.load_fp(idx + "index_kpool_compress_ape")
        # Splits kv_b_proj.weight into absorbed W_UK / W_UV (NoPE path).
        _call_process_weights_after_loading(self.model.layers[i].self_attn)

    def _load_kda_attn(self, L, attn: str, i: int) -> None:
        # KDA is head-sharded (mirrors DSA). q/k/v/b/g_b are per-head projections
        # -> column-parallel (shard dim 0, the head/output dim); g_a_proj feeds
        # the shared head_dim latent -> replicated. The loader's shard() narrows
        # to this rank's contiguous head block, matching the framework's
        # head-sharded conv/ssm slots (kv_cache_shape.cpp divides
        # linear_num_key_heads by world_size).
        L.load_fp(attn + "q_proj.weight", dim=0)
        L.load_fp(attn + "k_proj.weight", dim=0)
        L.load_fp(attn + "v_proj.weight", dim=0)
        L.load_fp(attn + "b_proj.weight", dim=0)
        L.load_fp(attn + "g_a_proj.weight")          # replicated (head_dim)
        L.load_fp(attn + "g_b_proj.weight", dim=0)
        # conv1d: depthwise over [q|k|v] (conv_dim = 3*qkv_dim). The model holds
        # the LOCAL conv_dim (3*qkv_dim_local); to shard by head each of q/k/v
        # must be narrowed on dim 0 (its head block) BEFORE cat, so the channel
        # order [q_loc|k_loc|v_loc] matches mixed_qkv. (A single contiguous
        # narrow of the cat'd tensor would cross the q/k boundary.) Fake
        # single-key checkpoint is only run at tp==1, where the shard is a no-op.
        if L.find(attn + "conv1d.weight") is not None:
            conv = L.load_tensor(attn + "conv1d.weight")
            if L.tp_size > 1:
                conv = L.shard(conv, dim=0)
        else:
            parts = [L.load_tensor(attn + n + "_conv1d.weight")
                     for n in ("q", "k", "v")]
            if L.tp_size > 1:
                parts = [L.shard(p, dim=0) for p in parts]
            conv = torch.cat(parts, dim=0)
        L.copy_in(attn + "conv1d.weight", conv)
        # o_proj: row-parallel QLinear — shard the INPUT dim (dim 1, qkv_dim) so
        # each rank's [hidden, qkv_dim_local] weight consumes its head-subset's
        # partial output; the forward all-reduces the partials.
        self._load_qlinear(L, attn, "o_proj", 1)
        # forget_gate: model nests under forget_gate.*, real ckpt is flat
        # (self_attn.f_a_proj / f_b_proj / dt_bias / A_log). load_tensor goes
        # through the alias wrapper, so the bare name resolves to the flat key.
        # f_a_proj replicated (head_dim); f_b_proj/dt_bias/A_log per-head (dim 0).
        L.load_fp(attn + "forget_gate.f_a_proj.weight")
        L.load_fp(attn + "forget_gate.f_b_proj.weight", dim=0)
        L.load_fp(attn + "forget_gate.dt_bias", dim=0)
        L.load_fp(attn + "forget_gate.A_log", dim=0)
        L.load_fp(attn + "o_norm.weight")
        _call_process_weights_after_loading(self.model.layers[i].self_attn)

    def _load_mlp_fp_or_w8a8(self, L, mlp_pfx: str) -> None:
        """Load a ``Glm53FlashMLP`` (gate_up_proj + down_proj) from the OLD-style
        checkpoint keys ``gate_proj`` / ``up_proj`` / ``down_proj``.

        fp path: cat gate+up on dim 0, shard dim 0 -> ``gate_up_proj.weight``;
        ``down_proj.weight`` shards dim 1 (row-parallel). w8a8 dynamic path:
        ``load_w8a8_mlp_into_qlinear`` does the same cat+shard for the quant
        tensors, writing into each QLinear's ``_w8a8`` submodule (NOT exercised
        until real w8a8 checkpoints exist, Task 10).
        """
        gate_mod = _resolve_module(self, mlp_pfx + "gate_up_proj")
        down_mod = _resolve_module(self, mlp_pfx + "down_proj")
        assert isinstance(gate_mod, QLinear) and isinstance(down_mod, QLinear), (
            f"{mlp_pfx}gate_up_proj/down_proj must be QLinear"
        )
        is_w8a8 = (L.probe_quant(mlp_pfx, "gate_proj")
                   or L.probe_quant(mlp_pfx, "up_proj")
                   or L.probe_quant(mlp_pfx, "down_proj"))
        if is_w8a8:
            # dynamic w8a8: route through load_w8a8_mlp_into_qlinear (cat
            # gate+up, shard dim 0; down shard dim 1), NOT load_w8a8_a which
            # only handles static per-projection tensors. resolve_quant builds
            # the w8a8 submodules; the loader writes into ``_w8a8``.
            gate_mod.resolve_quant(True)
            down_mod.resolve_quant(True)
            L.load_w8a8_mlp_into_qlinear(mlp_pfx)
        else:
            gate_mod.resolve_quant(False)
            down_mod.resolve_quant(False)
            gw = L.load_tensor(mlp_pfx + "gate_proj.weight")
            uw = L.load_tensor(mlp_pfx + "up_proj.weight")
            L.copy_in(
                mlp_pfx + "gate_up_proj.weight",
                torch.cat([L.shard(gw, dim=0), L.shard(uw, dim=0)], dim=0).contiguous(),
            )
            L.load_fp(mlp_pfx + "down_proj.weight", dim=1)

    def _load_experts_w8a8(self, L, mlp: str, n: int) -> None:
        """W8A8 expert load (mirrors DeepseekV3MoE loop, deepseek_v32.py:1018-1044).

        Per expert: gate+up cat after shard dim0 -> experts_w13[j]; down shard
        dim1 -> experts_w2[j]; scale/offset copied likewise. Writes the int8
        expert params + buffers, leaving the bf16 ``Glm53FlashExperts`` params
        unset (the W8A8 forward branch never reads them).
        """
        se = mlp + "experts."
        # Lazily register the int8 expert params on first W8A8 layer (they are
        # NOT allocated in __init__ — see the lazy-allocation note there).
        layer_idx = int(mlp.split("layers.")[1].split(".")[0])
        moe_mod = self.model.layers[layer_idx].mlp
        ref = moe_mod.gate.weight
        num_experts, inter_local, hidden = (
            moe_mod.num_experts, moe_mod.inter_local, moe_mod.hidden)
        if not hasattr(moe_mod, "experts_w13"):
            moe_mod.experts_w13 = nn.Parameter(
                torch.empty(num_experts, 2 * inter_local, hidden,
                            dtype=torch.int8, device=ref.device),
                requires_grad=False)
            # Offsets must be zero: the int8-grouped path needs symmetric
            # experts (process_weights_after_loading asserts offset == 0).
            moe_mod.register_buffer("experts_w13_scale", torch.empty(
                num_experts, 2 * inter_local, 1,
                dtype=torch.float32, device=ref.device))
            moe_mod.register_buffer("experts_w13_offset", torch.zeros(
                num_experts, 2 * inter_local, 1,
                dtype=torch.float32, device=ref.device))
            moe_mod.experts_w2 = nn.Parameter(
                torch.empty(num_experts, hidden, inter_local,
                            dtype=torch.int8, device=ref.device),
                requires_grad=False)
            moe_mod.register_buffer("experts_w2_scale", torch.empty(
                num_experts, hidden, 1, dtype=torch.float32, device=ref.device))
            moe_mod.register_buffer("experts_w2_offset", torch.zeros(
                num_experts, hidden, 1, dtype=torch.float32, device=ref.device))
        w13 = self.get_parameter(mlp + "experts_w13")
        w2 = self.get_parameter(mlp + "experts_w2")
        w13s = self.get_buffer(mlp + "experts_w13_scale")
        w13o = self.get_buffer(mlp + "experts_w13_offset")
        w2s = self.get_buffer(mlp + "experts_w2_scale")
        w2o = self.get_buffer(mlp + "experts_w2_offset")
        for j in range(n):
            gw = L.load_tensor(se + f"{j}.gate_proj.weight")
            gs = L.load_tensor(se + f"{j}.gate_proj.weight_scale")
            go = L.load_tensor(se + f"{j}.gate_proj.weight_offset")
            uw = L.load_tensor(se + f"{j}.up_proj.weight")
            us = L.load_tensor(se + f"{j}.up_proj.weight_scale")
            uo = L.load_tensor(se + f"{j}.up_proj.weight_offset")
            dw = L.load_tensor(se + f"{j}.down_proj.weight")
            ds = L.load_tensor(se + f"{j}.down_proj.weight_scale")
            do_ = L.load_tensor(se + f"{j}.down_proj.weight_offset")
            w13.data[j].copy_(
                torch.cat([L.shard(gw, 0), L.shard(uw, 0)], dim=0).contiguous())
            w13s.data[j].copy_(
                torch.cat([L.shard(gs, 0), L.shard(us, 0)], dim=0).contiguous())
            w13o.data[j].copy_(
                torch.cat([L.shard(go, 0), L.shard(uo, 0)], dim=0).contiguous())
            w2.data[j].copy_(L.shard(dw, 1).contiguous())
            w2s.data[j].copy_(ds.contiguous())
            w2o.data[j].copy_(do_.contiguous())

    def _load_experts_bf16(self, L, mlp: str, n: int) -> None:
        """bf16 expert load (existing 3D cat-stack path, extracted from _load_mlp).

        Lazily constructs ``self.experts`` (Glm53FlashExperts) — it is left unbuilt
        in __init__ to avoid the int8+bf16 double allocation OOMing the card.
        Fills the bf16 3D params (gate_up_proj / down_proj); the W8A8 int8 expert
        params are left unset (bf16 forward never reads them).
        """
        # Lazily construct the bf16 Glm53FlashExperts (left unbuilt in __init__ to
        # avoid int8+bf16 double allocation OOMing the card). Derive dtype/device
        # from the already-built int8 expert param.
        layer_idx = int(mlp.split("layers.")[1].split(".")[0])
        moe_mod = self.model.layers[layer_idx].mlp
        if moe_mod.experts is None:
            # bf16 experts — dtype/device from the shared experts (the int8
            # params are lazily created only on the W8A8 path).
            ref = moe_mod.shared_experts.gate_up_proj.weight
            moe_mod.experts = Glm53FlashExperts(
                self.cfg, ref.dtype, ref.device)
        # Build per-expert [2*inter, hidden] gate_up as [n_exp, 2*inter, hidden]
        # with layout [gate | up] along dim 1. Shard gate/up SEPARATELY then cat
        # (see _load_mlp comment: a contiguous shard of the cat'd tensor crosses
        # the gate/up boundary). At tp==1 the shard is a no-op.
        if L.find(mlp + "experts.gate_up_proj") is not None:
            gu = L.load_tensor(mlp + "experts.gate_up_proj")
            gate_full, up_full = gu.split(
                [gu.size(1) // 2, gu.size(1) // 2], dim=1)
        else:
            gate_full = torch.stack(
                [L.load_tensor(mlp + f"experts.{e}.gate_proj.weight")
                 for e in range(n)], dim=0)
            up_full = torch.stack(
                [L.load_tensor(mlp + f"experts.{e}.up_proj.weight")
                 for e in range(n)], dim=0)
        gate = L.shard(gate_full, dim=1)
        up = L.shard(up_full, dim=1)
        gu = torch.cat([gate, up], dim=1)
        L.copy_in(mlp + "experts.gate_up_proj", gu)
        if L.find(mlp + "experts.down_proj") is not None:
            dn = L.load_tensor(mlp + "experts.down_proj")
        else:
            dn = torch.stack([
                L.load_tensor(mlp + f"experts.{e}.down_proj.weight")
                for e in range(n)], dim=0)
        L.copy_in(mlp + "experts.down_proj", L.shard(dn, dim=2))

    def _load_mlp(self, L, mlp: str, i: int) -> None:
        if self.cfg.is_moe(i):
            n = self.cfg.n_routed_experts
            moe = self.model.layers[i].mlp
            # Router (FLOAT, shared by both branches).
            L.load_fp(mlp + "gate.weight")
            # e_score_correction_bias quirk: checkpoint key lives under
            # ``gate.e_score_correction_bias`` but the buffer is ``e_score_...``.
            e = L.load_tensor(mlp + "gate.e_score_correction_bias")
            L.copy_in(mlp + "e_score_correction_bias", e)
            # Expert branch: probe exp0's gate_proj for a weight_scale tensor.
            # Real W8A8 checkpoints carry weight_scale; bf16 checkpoints do not.
            is_w8a8 = L.find(mlp + "experts.0.gate_proj.weight_scale") is not None
            moe.use_w8a8 = is_w8a8
            if is_w8a8:
                self._load_experts_w8a8(L, mlp, n)
            else:
                self._load_experts_bf16(L, mlp, n)
            self._load_mlp_fp_or_w8a8(L, mlp + "shared_experts.")
            _call_process_weights_after_loading(moe)
        else:
            self._load_mlp_fp_or_w8a8(L, mlp)
            _call_process_weights_after_loading(self.model.layers[i].mlp)


# Registration is centralised in xllm.python.registry; import there.
