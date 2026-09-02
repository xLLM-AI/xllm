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
"""DCP sparse-flash-attention decode path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch
import torch.distributed as dist

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp_ref import merge_dcp_outputs, remap_sparse_indices


class GroupCoordinator(Protocol):
    world_size: int
    rank_in_group: int
    device_group: Any


def all_gather_async(
    input: torch.Tensor,
    group: GroupCoordinator,
    output: torch.Tensor | None = None,
    async_op: bool = True,
) -> tuple[torch.Tensor, torch.distributed.Work | None]:
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)


class DCPGatherContext(NamedTuple):
    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    split_sizes: tuple[int, ...]


@dataclass
class DCPContext:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_gather_block_ids: torch.Tensor | None = None
    kv_gather_block_table: torch.Tensor | None = None
    gather_context: DCPGatherContext | None = None


@dataclass
class AscendSFADCPMetadata:
    num_prefills: int
    dcp_context: DCPContext


class AscendSFADCPMetadataBuilder:
    def __init__(
        self,
        *,
        layout: KVShardLayout,
        device: torch.device,
        max_num_reqs: int,
    ) -> None:
        self.layout = layout
        self.dcp_size = layout.dcp_size
        self.device = device
        self.dcp_local_seq_lens_buf = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.dcp_rank_arange = torch.arange(
            self.dcp_size,
            dtype=torch.int32,
            device=device,
        )

    def _build_compact_kv_gather_metadata(
        self,
        dcp_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_block_ids, compact_block_table = dcp_block_table.flatten().unique(return_inverse=True)
        compact_block_table = compact_block_table.view_as(dcp_block_table)
        num_blocks = valid_block_ids.shape[0]
        remapped_block_table = (
            compact_block_table.unsqueeze(-1) + (self.dcp_rank_arange * num_blocks).view(1, 1, -1).to(dcp_block_table)
        ).reshape(dcp_block_table.shape[0], -1)
        return valid_block_ids, remapped_block_table.to(torch.int32)

    def build(
        self,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_reqs: int,
        num_input_tokens: int,
        *,
        dcp_local_seq_lens: torch.Tensor | None = None,
        num_prefills: int = 0,
    ) -> AscendSFADCPMetadata:
        dcp_block_table = block_table[:num_reqs]
        if dcp_local_seq_lens is None:
            dcp_local_seq_lens = self.layout.local_seq_lens(seq_lens[:num_reqs])
        local_seq_lens_src = dcp_local_seq_lens[:num_reqs].to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=True,
        )
        if num_reqs > self.dcp_local_seq_lens_buf.shape[0]:
            raise RuntimeError(
                f"dcp_local_seq_lens_buf is too small: "
                f"shape={tuple(self.dcp_local_seq_lens_buf.shape)}, num_reqs={num_reqs}"
            )
        self.dcp_local_seq_lens_buf[:num_reqs].copy_(local_seq_lens_src, non_blocking=True)
        local_seq_lens = self.dcp_local_seq_lens_buf[:num_reqs]

        kv_gather_block_ids = None
        kv_gather_block_table = None
        if num_prefills > 0:
            kv_gather_block_ids, kv_gather_block_table = self._build_compact_kv_gather_metadata(dcp_block_table)
        return AscendSFADCPMetadata(
            num_prefills=num_prefills,
            dcp_context=DCPContext(
                slot_mapping=slot_mapping[:num_input_tokens],
                block_table=dcp_block_table,
                seq_lens=local_seq_lens,
                kv_gather_block_ids=kv_gather_block_ids,
                kv_gather_block_table=kv_gather_block_table,
            ),
        )


class AscendSFADCPImpl:
    def __init__(
        self,
        dcp_group: GroupCoordinator,
        *,
        scale: float,
        index_topk: int,
        layout: KVShardLayout,
    ) -> None:
        self.dcp_group = dcp_group
        self.layout = layout
        self.dcp_size = layout.dcp_size
        self.scale = float(scale)
        self._dcp_index_topk = index_topk

    @staticmethod
    def _has_prefill(attn_metadata: AscendSFADCPMetadata) -> bool:
        return attn_metadata.num_prefills > 0

    def _record_dcp_kv_gather_context(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendSFADCPMetadata,
    ) -> None:
        if not self._has_prefill(attn_metadata):
            return

        valid_block_ids = attn_metadata.dcp_context.kv_gather_block_ids
        block_table = attn_metadata.dcp_context.kv_gather_block_table
        assert valid_block_ids is not None and block_table is not None
        kv = torch.index_select(kv_cache[0], 0, valid_block_ids)
        if len(kv_cache) < 2:
            raise RuntimeError("DCP SFA KV all-gather requires nope and rope KV caches.")
        key_rope = torch.index_select(kv_cache[1], 0, valid_block_ids)
        if kv.shape[:-1] != key_rope.shape[:-1] or kv.dtype != key_rope.dtype:
            raise RuntimeError(
                "Cannot fuse DCP KV gather for KV/nope and KV/rope caches with "
                f"shapes {tuple(kv.shape)} / {tuple(key_rope.shape)} and dtypes {kv.dtype} / {key_rope.dtype}."
            )
        attn_metadata.dcp_context.gather_context = self._start_dcp_gather(
            torch.cat([kv, key_rope], dim=-1).contiguous(),
            dim=0,
            split_sizes=(kv.shape[-1], key_rope.shape[-1]),
        )

    def _start_dcp_gather(
        self,
        x: torch.Tensor,
        dim: int,
        split_sizes: tuple[int, ...],
    ) -> DCPGatherContext:
        gathered, handle, restore_perm = self._all_gather_dim_async(x, dim)
        return DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=restore_perm,
            split_sizes=split_sizes,
        )

    @staticmethod
    def _finish_dcp_gather(
        context: DCPGatherContext,
    ) -> tuple[torch.Tensor, ...]:
        if context.handle is not None:
            context.handle.wait()
        gathered = context.gathered
        if context.restore_perm is not None:
            gathered = gathered.permute(context.restore_perm).contiguous()
        return torch.split(gathered, context.split_sizes, dim=-1)

    def _all_gather_dim_async(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[int, ...] | None]:
        if dim == 0:
            gathered, handle = all_gather_async(x.contiguous(), self.dcp_group)
            return gathered, handle, None

        perm = (dim, *[i for i in range(x.dim()) if i != dim])
        restore_perm = tuple(perm.index(i) for i in range(x.dim()))
        gathered, handle = all_gather_async(x.permute(perm).contiguous(), self.dcp_group)
        return gathered, handle, restore_perm

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        return remap_sparse_indices(topk_indices, self.layout, self._dcp_index_topk)

    def _all_to_all_dcp_tensor(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
    ) -> torch.Tensor:
        scatter_size = tensor.shape[scatter_dim]
        if scatter_size % self.dcp_size != 0:
            raise RuntimeError(
                "DCP output All2All requires the scatter dimension to be divisible "
                f"by dcp_size, got shape={tuple(tensor.shape)}, scatter_dim={scatter_dim}, "
                f"and dcp_size={self.dcp_size}."
            )

        local_scatter_size = scatter_size // self.dcp_size
        send = tensor.movedim(scatter_dim, 0).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.dcp_group.device_group)
        recv = recv.view(self.dcp_size, local_scatter_size, *send.shape[1:])
        return recv

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        output_recv = self._all_to_all_dcp_tensor(sfa_output, 1)
        lse_recv = self._all_to_all_dcp_tensor(softmax_lse, 1).squeeze(-1)
        return merge_dcp_outputs(output_recv, lse_recv)

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> DCPGatherContext:
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        fused_q = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
        return self._start_dcp_gather(
            fused_q,
            dim=1,
            split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
        )

    def _record_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: AscendSFADCPMetadata,
    ) -> None:
        if self._has_prefill(attn_metadata):
            return
        attn_metadata.dcp_context.gather_context = self._start_dcp_query_gather(ql_nope, q_pe)

    def _store_parallel_kv(
        self,
        k_pe: torch.Tensor | None,
        k_nope: torch.Tensor | None,
        k_li: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: AscendSFADCPMetadata,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if kv_cache is not None:
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
        return k_pe, k_nope, k_li

    def _npu_sparse_flash_attention(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        *,
        sparse_mode: int,
        return_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv = kv_cache[0]
        attention_out, softmax_max, softmax_sum = torch.ops.xllm_ops.sparse_flash_attention_lse(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=kv_cache[1],
            scale_value=self.scale,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            attention_mode=2,
            return_softmax_lse=return_lse,
        )
        if return_lse:
            return attention_out, softmax_max, softmax_sum
        return attention_out

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        attn_metadata: AscendSFADCPMetadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        dcp_context = attn_metadata.dcp_context
        if self._has_prefill(attn_metadata):
            gather_context = dcp_context.gather_context
            dcp_context.gather_context = None
            assert gather_context is not None
            gathered_kv_cache = self._finish_dcp_gather(gather_context)
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            return self._npu_sparse_flash_attention(
                ql_nope,
                q_pe,
                gathered_kv_cache,
                topk_indices,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table,
                sparse_mode=3,
                return_lse=False,
            )

        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        assert gather_context is not None
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_dcp_gather(gather_context)
        sfa_output, softmax_max, softmax_sum = self._npu_sparse_flash_attention(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            dcp_context.block_table,
            sparse_mode=0,
            return_lse=True,
        )
        softmax_lse = softmax_max + torch.log(softmax_sum)
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = sfa_output.dtype
        output = self._merge_dcp_outputs(sfa_output, softmax_lse)
        return output.to(output_dtype)
