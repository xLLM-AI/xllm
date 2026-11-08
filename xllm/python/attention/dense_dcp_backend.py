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

"""Dense GQA decode-context-parallel attention for Ascend NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionMetadata,
    LayerCache,
    normalize_attention_lse,
)
from xllm.python.attention.dense_dcp_execution import DenseDcpFia
from xllm.python.attention.dense_dcp_metadata import DenseDcpMetadata
from xllm.python.attention.npu_paged_attention import SPARSE_MODE_RIGHT_DOWN_CAUSAL, NpuPagedAttentionBackend
from xllm.python.distributed import AllGatherContext, finish_all_gather, start_all_gather
from xllm.python.kernels_npu.triton.dcp_packed_a2a import (
    fused_dcp_lse_combine,
    pack_dcp_output_lse,
    packed_send_shape,
)
from xllm.python.model_executor.forward_context import get_execution_buffer, get_execution_context

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


class DenseDcpAttentionBackend(NpuPagedAttentionBackend):
    """FIA backend that shards replicated GQA KV cache across DCP ranks."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        dcp_group: ProcessGroup,
    ) -> None:
        if dcp_group.size() <= 1:
            raise ValueError("Dense DCP attention requires a group larger than one rank")
        if sliding_window:
            raise NotImplementedError("Dense DCP attention does not support sliding-window attention")
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            sliding_window=sliding_window,
            is_mla=False,
            device=device,
            dtype=dtype,
        )
        self._dcp_group = dcp_group
        self._fia = DenseDcpFia(num_heads * dcp_group.size(), num_kv_heads, head_dim, scale, dtype)

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        self._fia.bind_layer_caches(kv_caches)

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
        owned_metadata: bool = False,
    ) -> None:
        self._fia.prepare(metadata, graph_mode=graph_mode)

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        if not layer.causal:
            raise NotImplementedError("Dense DCP attention supports only causal attention")
        dcp_metadata = get_execution_context(DenseDcpMetadata)
        if dcp_metadata is None:
            raise RuntimeError("Dense DCP attention requires DenseDcpMetadata from its metadata builder")

        layer_cache = self._kv_caches[layer.layer_id]
        key_cache, value_cache = layer_cache.key, layer_cache.value
        if key_cache is None or value_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer.layer_id}")

        num_tokens = int(q.shape[0])
        query = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()

        query_gather: AllGatherContext | None = None
        if dcp_metadata.is_chunked_prefill or not dcp_metadata.is_prefill:
            query_gather = start_all_gather(query, dim=1, group=self._dcp_group)

        kernels.reshape_paged_cache(
            dcp_metadata.local_slot_mapping,
            key,
            value,
            key_cache,
            value_cache,
        )

        if query_gather is None:
            # All current K/V are available locally. A table, if supplied, only
            # addresses a DCP shard and must not select the parent's paged path.
            output, _ = self._run_current_chunk_attention(query, key, value, dcp_metadata)
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        gathered_query = finish_all_gather(query_gather)
        current_output = current_lse = None
        if dcp_metadata.is_chunked_prefill:
            current_output, current_lse = self._run_current_chunk_attention(gathered_query, key, value, dcp_metadata)
            local_head_start = self._dcp_group.rank() * self.num_heads
            local_head_end = local_head_start + self.num_heads
            current_output = current_output[:, local_head_start:local_head_end]
            current_lse = normalize_attention_lse(
                current_lse,
                num_tokens,
                self.num_heads * self._dcp_group.size(),
            )[:, local_head_start:local_head_end]
            if not dcp_metadata.has_context:
                return current_output.reshape(num_tokens, self.num_heads * self.head_dim)

        if dcp_metadata.has_local_context or not dcp_metadata.is_chunked_prefill:
            context_output, context_lse = self._fia.run(
                gathered_query,
                key_cache,
                value_cache,
                dcp_metadata,
            )
        else:
            context_output = get_execution_buffer(
                ("DCP_EMPTY_CONTEXT_OUTPUT", *gathered_query.shape, str(gathered_query.dtype)),
                lambda: torch.zeros_like(gathered_query),
            )
            context_lse = get_execution_buffer(
                ("DCP_EMPTY_CONTEXT_LSE", *gathered_query.shape[:2]),
                lambda: torch.full(
                    (*gathered_query.shape[:2], 1),
                    -torch.inf,
                    dtype=torch.float32,
                    device=gathered_query.device,
                ),
            )
        num_heads = int(context_output.shape[1])
        head_dim = self.head_dim
        dcp_size = self._dcp_group.size()
        lse = normalize_attention_lse(context_lse, num_tokens, num_heads)
        lse.masked_fill_(dcp_metadata.empty_kv_shards[:num_tokens], -torch.inf)

        send_shape = packed_send_shape(
            num_tokens, num_heads, head_dim, dcp_size, scatter_dim=1, dtype=context_output.dtype
        )
        send = get_execution_buffer(
            ("DCP_PACKED_A2A_SEND", *send_shape, str(context_output.dtype)),
            lambda: torch.empty(send_shape, dtype=context_output.dtype, device=context_output.device),
        )
        recv = get_execution_buffer(
            ("DCP_PACKED_A2A_RECV", *send_shape, str(context_output.dtype)),
            lambda: torch.empty(send_shape, dtype=context_output.dtype, device=context_output.device),
        )
        merged = get_execution_buffer(
            ("DCP_MERGE_OUT", num_tokens, num_heads // dcp_size, head_dim, str(context_output.dtype)),
            lambda: torch.empty(
                (num_tokens, num_heads // dcp_size, head_dim),
                dtype=context_output.dtype,
                device=context_output.device,
            ),
        )
        pack_dcp_output_lse(context_output, lse, dcp_size, scatter_dim=1, send=send)
        dist.all_to_all_single(recv, send, group=self._dcp_group)
        output = fused_dcp_lse_combine(
            recv,
            head_dim,
            scatter_dim=1,
            output=merged,
            local_output=current_output,
            local_lse=current_lse,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    def _run_current_chunk_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: DenseDcpMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            pse_shift=None,
            atten_mask=self._get_causal_mask(),
            actual_seq_lengths=metadata.query_seq_ends,
            actual_seq_lengths_kv=metadata.query_seq_ends,
            num_heads=query.shape[1],
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=SPARSE_MODE_RIGHT_DOWN_CAUSAL,
            softmax_lse_flag=metadata.is_chunked_prefill,
        )
