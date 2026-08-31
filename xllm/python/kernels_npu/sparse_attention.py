# Copyright 2026 The xLLM Authors.
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

"""NPU sparse-attention kernels."""

from __future__ import annotations

import torch

try:
    import cann_ops_transformer  # noqa: F401
except ImportError:
    cann_ops_transformer = None


def pool_key_indexer(
    query: torch.Tensor,
    pool_key: torch.Tensor,
    weights: torch.Tensor,
    pool_tail_k: torch.Tensor,
    topk: int,
    pool_size: int,
    *,
    return_value: bool = False,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    actual_seq_q: torch.Tensor | None = None,
    actual_seq_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    layout_q: str = "BSND",
    layout_k: str = "BSND",
    mask_mode: int = 3,
    quant_mode: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CANN's fused pool-key indexer for supported layouts."""
    if layout_q == "BSND":
        if query.dim() != 4 or query.size(-1) != 128 or query.numel() == 0:
            raise ValueError("query must have shape [B, S1, N1, 128]")
        batch_size = query.size(0)
        expected_weights_shape = query.shape[:3]
        if actual_seq_q is not None:
            raise ValueError("actual_seq_q is only valid for TND")
    elif layout_q == "TND":
        if query.dim() != 3 or query.size(-1) != 128 or query.numel() == 0:
            raise ValueError("query must have shape [T1, N1, 128]")
        if actual_seq_q is None:
            raise ValueError("layout_q=TND requires actual_seq_q")
        if actual_seq_q.dtype != torch.int32 or actual_seq_q.dim() != 1:
            raise TypeError("actual_seq_q must be a rank-1 int32 tensor")
        batch_size = pool_tail_k.numel()
        expected_weights_shape = query.shape[:2]
    else:
        raise ValueError("layout_q must be BSND or TND")

    if layout_k == "BSND":
        if (
            pool_key.dim() != 4
            or pool_key.size(0) != batch_size
            or pool_key.size(2) != 1
            or pool_key.size(-1) != 128
            or pool_key.numel() == 0
        ):
            raise ValueError("pool_key must have shape [B, S2, 1, 128]")
        if actual_seq_k is not None or block_table is not None:
            raise ValueError("actual_seq_k/block_table are not valid for BSND")
    elif layout_k == "TND":
        if (
            pool_key.dim() != 3
            or pool_key.size(1) != 1
            or pool_key.size(-1) != 128
            or pool_key.numel() == 0
        ):
            raise ValueError("pool_key must have shape [T2, 1, 128]")
        if actual_seq_k is None:
            raise ValueError("layout_k=TND requires actual_seq_k")
        if actual_seq_k.dtype != torch.int32 or actual_seq_k.dim() != 1:
            raise TypeError("actual_seq_k must be a rank-1 int32 tensor")
        if block_table is not None:
            raise ValueError("block_table is only valid for PA_BBND")
    elif layout_k == "PA_BBND":
        if (
            pool_key.dim() != 4
            or pool_key.size(2) != 1
            or pool_key.size(-1) != 128
            or pool_key.numel() == 0
        ):
            raise ValueError("pool_key must have shape [block_num, block_size, 1, 128]")
        if (
            actual_seq_k is None
            or block_table is None
            or actual_seq_k.dtype != torch.int32
            or actual_seq_k.dim() != 1
            or block_table.dtype != torch.int32
            or block_table.dim() != 2
            or actual_seq_k.numel() != batch_size
            or block_table.size(0) != batch_size
        ):
            raise ValueError("PA_BBND requires per-batch int32 actual_seq_k and block_table")
    else:
        raise ValueError("layout_k must be BSND, TND, or PA_BBND")
    if weights.shape != expected_weights_shape:
        raise ValueError("weights shape must match query layout")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("query must be float16 or bfloat16")
    if pool_key.dtype != query.dtype or weights.dtype != query.dtype:
        raise TypeError("query, pool_key, and weights must have the same dtype")
    if pool_tail_k.dtype != torch.int32 or pool_tail_k.dim() != 1:
        raise TypeError("pool_tail_k must be a rank-1 int32 tensor")
    if quant_mode not in (-1, 0, 1):
        raise ValueError("quant_mode must be -1, 0, or 1")
    if quant_mode == -1 and (q_descale is not None or k_descale is not None):
        raise ValueError("q_descale/k_descale require quant_mode >= 0")
    if q_descale is not None:
        if q_descale.dtype != torch.float32 or q_descale.shape != query.shape[:-1]:
            raise ValueError("q_descale must be float32 with query.shape[:-1]")
    if k_descale is not None:
        if k_descale.dtype != torch.float32 or k_descale.shape != pool_key.shape[:-1]:
            raise ValueError("k_descale must be float32 with pool_key.shape[:-1]")
    inputs = [pool_key, weights, pool_tail_k]
    for tensor in (
        q_descale,
        k_descale,
        actual_seq_q,
        actual_seq_k,
        block_table,
    ):
        if tensor is not None:
            inputs.append(tensor)
    if query.device.type != "npu" or any(
        tensor.device != query.device for tensor in inputs
    ):
        raise ValueError("all inputs must be on the same NPU device")
    if pool_tail_k.numel() != batch_size:
        raise ValueError("pool_tail_k must have one element per batch")
    if actual_seq_q is not None and actual_seq_q.numel() not in (
        batch_size,
        batch_size + 1,
    ):
        raise ValueError("actual_seq_q must contain one length per batch")
    if actual_seq_k is not None and layout_k == "TND" and actual_seq_k.numel() not in (
        batch_size,
        batch_size + 1,
    ):
        raise ValueError("actual_seq_k must contain one length per batch")
    # The CANN wheel accepts cumulative sequence ends without the optional
    # leading zero. Keep the public README-compatible B+1 form, but normalize
    # it before dispatch using shape-only slicing (graph-safe).
    actual_seq_q_op = actual_seq_q
    if actual_seq_q is not None and actual_seq_q.numel() == batch_size + 1:
        actual_seq_q_op = actual_seq_q[1:]
    actual_seq_k_op = actual_seq_k
    if actual_seq_k is not None and layout_k == "TND" and actual_seq_k.numel() == batch_size + 1:
        actual_seq_k_op = actual_seq_k[1:]
    if actual_seq_q_op is not None:
        actual_seq_q_op = actual_seq_q_op.to(torch.int32).contiguous()
    if actual_seq_k_op is not None:
        actual_seq_k_op = actual_seq_k_op.to(torch.int32).contiguous()
    if block_table is not None:
        block_table = block_table.to(torch.int32).contiguous()
    if pool_size < 1 or pool_size > 128:
        raise ValueError("pool_size must be in [1, 128]")
    if topk < 1 or topk > 8192 or topk % pool_size != 0:
        raise ValueError("topk must be in [1, 8192] and divisible by pool_size")
    try:
        op = torch.ops.cann_ops_transformer.pool_key_indexer
    except (AttributeError, RuntimeError) as exc:
        raise NotImplementedError(
            "CANN pool_key_indexer is unavailable in this runtime"
        ) from exc
    return op(
        query,
        pool_key,
        weights,
        pool_tail_k,
        q_descale=q_descale,
        k_descale=k_descale,
        actual_seq_q=actual_seq_q_op,
        actual_seq_k=actual_seq_k_op,
        block_table=block_table,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=quant_mode,
        return_value=return_value,
    )


def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    selected_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    return_value: bool,
) -> torch.Tensor:
    """Select the key blocks each query attends to.

    Args:
        query: Query tensor laid out as ``layout_query``.
        key: Key cache laid out as ``layout_key``.
        weights: Per-head indexer weights.
        query_seq_lengths: Query length of every sequence, or ``None``.
        key_seq_lengths: Key length of every sequence, or ``None``.
        block_table: Paged key-cache block table, or ``None``.
        layout_query: Query layout, ``"TND"`` or ``"BSND"``.
        layout_key: Key layout, for example ``"PA_BSND"``.
        selected_count: Key blocks kept per query.
        sparse_mode: Sparse masking mode.
        pre_tokens: Tokens visible before the query position.
        next_tokens: Tokens visible after the query position.
        return_value: Whether to also return the indexer scores.

    Returns:
        Selected key indices of dtype ``torch.int32``.
    """
    return torch.ops.xllm_ops.lightning_indexer(
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        layout_query,
        layout_key,
        selected_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
    )


def lightning_indexer_out(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    layout_query: str,
    layout_key: str,
    selected_count: int,
    sparse_mode: int,
    pre_tokens: int,
    next_tokens: int,
    return_value: bool,
    sparse_indices_out: torch.Tensor,
    sparse_values_out: torch.Tensor,
) -> torch.Tensor:
    """Select key blocks and write the results to caller-owned buffers."""
    return torch.ops.xllm_ops.lightning_indexer_out(
        query,
        key,
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        layout_query,
        layout_key,
        selected_count,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
        sparse_indices_out,
        sparse_values_out,
    )


def quant_lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    metadata: torch.Tensor,
    query_seq_lengths: torch.Tensor | None,
    key_seq_lengths: torch.Tensor | None,
    block_table: torch.Tensor | None,
    selected_count: int,
    cmp_ratio: int = 1,
) -> torch.Tensor:
    """Run INT8 LightningIndexer with per-token Q/K dequant scales."""
    indices, _ = torch.ops.xllm_ops.quant_lightning_indexer(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        0,
        0,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        metadata,
        "TND",
        "PA_BSND",
        selected_count,
        3,
        9223372036854775807,
        9223372036854775807,
        cmp_ratio,
        False,
    )
    return indices


def quant_lightning_indexer_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sparse_count: int,
    cmp_ratio: int,
) -> torch.Tensor:
    """Create reusable tiling metadata for QuantLightningIndexer."""
    return torch.ops.xllm_ops.quant_lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        0,
        0,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        actual_seq_lengths_key.numel(),
        max_seqlen_q,
        max_seqlen_k,
        "TND",
        "PA_BSND",
        sparse_count,
        3,
        9223372036854775807,
        9223372036854775807,
        cmp_ratio,
        str(actual_seq_lengths_query.device),
    )


def scatter_nd_update(
    value: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    """Write ``updates`` into ``value`` at ``indices``, in place.

    Args:
        value: Destination tensor, updated in place.
        indices: Index of every updated row, shape ``[num_updates, 1]``.
        updates: Rows written into ``value``.
    """
    torch.ops.xllm_ops.scatter_nd_update(value, indices, updates)


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor | None,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    scale_value: float,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
) -> torch.Tensor:
    """Attend to the key blocks selected by :func:`lightning_indexer`.

    Args:
        query: Query tensor laid out as ``layout_query``.
        key: Key cache laid out as ``layout_kv``.
        value: Value cache laid out as ``layout_kv``.
        sparse_indices: Key blocks selected per query.
        block_table: Paged cache block table, or ``None``.
        actual_seq_lengths_query: Query length of every sequence, or ``None``.
        actual_seq_lengths_kv: Key length of every sequence, or ``None``.
        query_rope: Rotary part of the query, or ``None``.
        key_rope: Rotary part of the key, or ``None``.
        scale_value: Softmax scale.
        sparse_block_size: Keys per selected block.
        layout_query: Query layout, ``"TND"`` or ``"BSND"``.
        layout_kv: Key and value layout.
        sparse_mode: Sparse masking mode.

    Returns:
        Attention output with the shape and dtype of ``query``.
    """
    return torch.ops.xllm_ops.sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
    )


def sparse_flash_attention_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_indices: torch.Tensor,
    block_table: torch.Tensor | None,
    actual_seq_lengths_query: torch.Tensor | None,
    actual_seq_lengths_kv: torch.Tensor | None,
    query_rope: torch.Tensor | None,
    key_rope: torch.Tensor | None,
    scale_value: float,
    sparse_block_size: int,
    layout_query: str,
    layout_kv: str,
    sparse_mode: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Attend to selected blocks and write the output into ``output``."""
    return torch.ops.xllm_ops.sparse_flash_attention_out(
        query,
        key,
        value,
        sparse_indices,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        scale_value,
        sparse_block_size,
        layout_query,
        layout_kv,
        sparse_mode,
        output,
    )


__all__ = [
    "lightning_indexer",
    "lightning_indexer_out",
    "pool_key_indexer",
    "quant_lightning_indexer",
    "quant_lightning_indexer_metadata",
    "scatter_nd_update",
    "sparse_flash_attention",
    "sparse_flash_attention_out",
]
