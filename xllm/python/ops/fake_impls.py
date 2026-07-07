"""FakeTensor (meta) impls for xllm_ops — torch.compile prerequisite.

  - fused_add_rms_norm, fused_qk_norm_rope: register_fake (in-graph capable),
    return the mutated input.
  - reshape_paged_cache: disallow_in_graph (accesses kv_cache from
    forward_context outside the compiled graph).
  - batch_prefill / batch_decode / batch_chunked_prefill: disallow_in_graph
    (attention ops run in eager section)
  - update_*_plan: not traced (called conditionally in eager Python)
"""

from __future__ import annotations

from typing import Tuple

import torch


@torch.library.register_fake("xllm_ops::rms_norm")
def _rms_norm_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty_like(input)


@torch.library.register_fake("xllm_ops::fused_add_rms_norm")
def _fused_add_rms_norm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return input, residual


@torch.library.register_fake("xllm_ops::silu_and_mul")
def _silu_and_mul_fake(input: torch.Tensor) -> torch.Tensor:
    shape = list(input.shape)
    shape[-1] //= 2
    return input.new_empty(shape)


@torch.library.register_fake("xllm_ops::fused_qk_norm_rope")
def _fused_qk_norm_rope_fake(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    interleaved: bool,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    return qkv


@torch.library.register_fake("xllm_ops::reshape_paged_cache")
def _reshape_paged_cache_fake(
    slot_mapping: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    return k_cache


# Attention kernel ops + reshape_paged_cache: disallow_in_graph (split points).
# These run in the eager attention section, accessing kv_cache/metadata from
# forward_context (outside the compiled graph).
for _op_name in (
    "reshape_paged_cache",
    "batch_prefill",
    "batch_decode",
    "batch_chunked_prefill",
):
    _op = getattr(torch.ops.xllm_ops, _op_name)
    torch._dynamo.disallow_in_graph(getattr(_op, "default", _op))
