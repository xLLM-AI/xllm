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

"""Dense DCP operator execution and entry-owned ACL graph resources."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch_npu

from xllm.python.attention.backend import AttentionMetadata, LayerCache
from xllm.python.attention.dense_dcp_metadata import DenseDcpMetadata
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_execution_context,
    get_execution_workspace,
    get_forward_context,
)


@dataclass(frozen=True, slots=True)
class _FiaBuffers:
    workspace: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    block_table: torch.Tensor


class DenseDcpFia:
    """Run paged FIA without exposing capture/task-update details to attention.

    Only prepare() selects the current entry's buffers. Each captured task binds
    its own entry metadata, buffers and layer Q/K/V; it never reads that mutable
    selection again. Host list changes take effect via FIA task update, not by
    merely mutating Python objects after capture.
    """

    def __init__(self, num_heads: int, num_kv_heads: int, head_dim: int, scale: float, dtype: torch.dtype) -> None:
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._scale = scale
        self._dtype = dtype
        self._cache: LayerCache | None = None
        self._buffers: _FiaBuffers | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        self._cache = next((cache for cache in layer_caches if cache.key is not None), None)
        if self._cache is None or self._cache.value is None:
            raise RuntimeError("Dense DCP FIA requires bound key and value caches")

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool) -> None:
        if not graph_mode:
            self._buffers = None
            return
        dcp_metadata = get_execution_context(DenseDcpMetadata)
        if dcp_metadata is None:
            raise RuntimeError("Dense DCP FIA requires builder metadata before graph preparation")
        table = metadata.block_table
        if table is None or table.dtype != torch.int32:
            raise RuntimeError("Dense DCP graph FIA requires the runner's persistent int32 block table")
        if dcp_metadata.block_table is not None and dcp_metadata.block_table.data_ptr() != table.data_ptr():
            raise RuntimeError("Dense DCP graph block-table address changed after preparation")
        if self._cache is None or self._cache.key is None or self._cache.value is None:
            raise RuntimeError("Dense DCP FIA requires bound layer caches")
        key_cache, value_cache = self._cache.key, self._cache.value
        num_tokens = len(dcp_metadata.query_seq_ends)
        if table.shape[0] != num_tokens:
            raise RuntimeError("Dense DCP graph block-table rows must match the decode capacity")
        block_size = int(key_cache.shape[1])
        key = (num_tokens, self._num_heads, self._head_dim, *key_cache.shape[:2], table.shape[1], str(self._dtype))
        output = get_execution_buffer(
            ("DENSE_DCP_FIA_OUTPUT", *key),
            lambda: torch.empty(
                (num_tokens, self._num_heads, self._head_dim), dtype=self._dtype, device=key_cache.device
            ),
        )
        lse = get_execution_buffer(
            ("DENSE_DCP_FIA_LSE", *key),
            lambda: torch.empty((num_tokens, self._num_heads, 1), dtype=torch.float32, device=key_cache.device),
        )
        # The workspace query only needs tensor descriptors. Reuse the output
        # and actual caches instead of allocating dummy Q/KV on every prepare.
        workspace = get_execution_workspace(
            ("DENSE_DCP_FIA_WORKSPACE", *key),
            lambda: torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                output,
                key_cache.view(key_cache.shape[0], block_size, -1),
                value_cache.view(value_cache.shape[0], block_size, -1),
                **self._kwargs(dcp_metadata, table, block_size),
            ),
        )
        dcp_metadata.block_table = table
        self._buffers = _FiaBuffers(workspace, output, lse, table)

    def _kwargs(self, metadata: DenseDcpMetadata, block_table: torch.Tensor, block_size: int) -> dict[str, object]:
        return {
            "pse_shift": None,
            "atten_mask": None,
            "actual_seq_lengths": metadata.query_seq_ends,
            "actual_seq_lengths_kv": metadata.local_kv_lengths,
            "block_table": block_table,
            "num_heads": self._num_heads,
            "scale": self._scale,
            "input_layout": "TND",
            "num_key_value_heads": self._num_kv_heads,
            "sparse_mode": 0,
            "block_size": block_size,
            "softmax_lse_flag": True,
        }

    def run(
        self, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, metadata: DenseDcpMetadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = int(key_cache.shape[1])
        key = key_cache.view(key_cache.shape[0], block_size, -1)
        value = value_cache.view(value_cache.shape[0], block_size, -1)
        graph_context = get_forward_context().acl_graph
        if graph_context is None:
            if metadata.block_table is None:
                raise RuntimeError("Dense DCP cached FIA requires a block table")
            return torch.ops.npu.npu_fused_infer_attention_score(
                query, key, value, **self._kwargs(metadata, metadata.block_table, block_size)
            )

        buffers = self._buffers
        if buffers is None or buffers.block_table is not metadata.block_table:
            raise RuntimeError("Dense DCP graph FIA resources are not prepared for this entry")

        graph_context.capture_task(
            torch.ops.npu.npu_fused_infer_attention_score.out,
            query,
            key,
            value,
            **self._kwargs(metadata, buffers.block_table, block_size),
            workspace=buffers.workspace,
            out=[buffers.output, buffers.lse],
        )
        return buffers.output, buffers.lse
