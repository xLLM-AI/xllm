# Copyright 2026 The xLLM Authors.
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

"""Public sparse-attention operators and FakeTensor contracts."""

from __future__ import annotations

from collections.abc import Callable

import torch


def lightning_indexer(*args, **kwargs) -> torch.Tensor:
    return torch.ops.xllm_ops.lightning_indexer(*args, **kwargs)


def scatter_nd_update(
    value: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> None:
    torch.ops.xllm_ops.scatter_nd_update(value, indices, updates)


def sparse_flash_attention(*args, **kwargs) -> torch.Tensor:
    return torch.ops.xllm_ops.sparse_flash_attention(*args, **kwargs)


def _lightning_indexer_fake(
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
):
    del (
        weights,
        query_seq_lengths,
        key_seq_lengths,
        block_table,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
    )
    key_head_num = key.size(1) if layout_key == "TND" else key.size(2)
    if layout_query == "BSND":
        out_shape = (query.size(0), query.size(1), key_head_num, selected_count)
    else:
        out_shape = (query.size(0), key_head_num, selected_count)
    return query.new_zeros(out_shape, dtype=torch.int32)


def _scatter_nd_update_fake(value, indices, updates):
    del value, indices, updates


def _sparse_flash_attention_fake(
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
):
    del (
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
    return query.new_empty(query.shape, dtype=query.dtype)


def _register_fake_if_available(name: str, fake: Callable) -> None:
    namespace, op_name = name.split("::", 1)
    if hasattr(getattr(torch.ops, namespace), op_name):
        torch.library.register_fake(name)(fake)


_register_fake_if_available("xllm_ops::lightning_indexer", _lightning_indexer_fake)
_register_fake_if_available("xllm_ops::scatter_nd_update", _scatter_nd_update_fake)
_register_fake_if_available(
    "xllm_ops::sparse_flash_attention", _sparse_flash_attention_fake
)

__all__ = ["lightning_indexer", "scatter_nd_update", "sparse_flash_attention"]
