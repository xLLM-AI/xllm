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
from xllm.python.layers.sfa_dcp_ref import remap_sparse_indices
from xllm.python.model_executor.forward_context import get_execution_buffer

# Must match xllm/python/kernels_npu/tilelang/sfa_dcp_remap.py AOT specializations.
_REMAP_TOPK = 2048
_REMAP_MAX_TOKENS = 256


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def _fused_remap_available() -> bool:
    ops = getattr(torch.ops, "xllm_ops", None)
    return ops is not None and hasattr(ops, "sfa_dcp_remap_out")


def _lse_as_token_head(
    softmax_lse: torch.Tensor,
    num_tokens: int,
    num_heads: int,
) -> torch.Tensor:
    """Normalize SFA LSE to ``[T, H]``.

    Graph capture can emit ``[1, T, H]``; eager decode may emit ``[T, H]`` or
    ``[T, H, 1]``.
    """
    lse = softmax_lse
    if lse.ndim >= 3 and lse.shape[-1] == 1:
        lse = lse.squeeze(-1)
    if lse.ndim >= 3 and lse.shape[0] == 1:
        lse = lse.squeeze(0)
    if tuple(lse.shape) != (num_tokens, num_heads):
        raise RuntimeError(f"softmax_lse must be [T, H]=[{num_tokens}, {num_heads}], got {tuple(softmax_lse.shape)}")
    return lse


def _can_use_fused_remap(
    topk_indices: torch.Tensor,
    *,
    index_topk: int,
    physical_block_size: int,
    dcp_size: int,
) -> bool:
    last_dim = int(topk_indices.shape[-1]) if topk_indices.ndim > 0 else 0
    num_tokens = int(topk_indices.numel() // last_dim) if last_dim > 0 else 0
    return (
        index_topk == _REMAP_TOPK
        and last_dim == _REMAP_TOPK
        and 0 < num_tokens <= _REMAP_MAX_TOKENS
        and _is_power_of_two(physical_block_size)
        and _is_power_of_two(dcp_size)
        and topk_indices.dtype == torch.int32
        and topk_indices.is_contiguous()
        and _is_npu_tensor(topk_indices)
        and _fused_remap_available()
    )


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
        out = get_execution_buffer(
            ("SFA_DCP_REMAP_OUT", tuple(topk_indices.shape)),
            lambda: torch.empty_like(topk_indices),
        )
        if not _can_use_fused_remap(
            topk_indices,
            index_topk=self._dcp_index_topk,
            physical_block_size=int(self.layout.physical_block_size),
            dcp_size=int(self.layout.dcp_size),
        ):
            out.copy_(
                remap_sparse_indices(
                    topk_indices,
                    self.layout,
                    index_topk=self._dcp_index_topk,
                )
            )
            return out

        num_tokens = int(topk_indices.numel() // self._dcp_index_topk)
        scratch_n = num_tokens * self._dcp_index_topk
        idx_scratch = get_execution_buffer(
            ("SFA_DCP_REMAP_SCRATCH", scratch_n),
            lambda: torch.empty(scratch_n, dtype=torch.int32, device=topk_indices.device),
        )
        return torch.ops.xllm_ops.sfa_dcp_remap_out(
            topk_indices,
            int(self.layout.physical_block_size),
            int(self.layout.dcp_size),
            int(self.layout.dcp_rank),
            out,
            idx_scratch,
        )

    def _merge_dcp_outputs(
        self,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        """Pack output+LSE, run one AllToAll, and fuse LSE combine.

        Payload layout is ``[dcp, H_local, T, D+4]`` for bf16/fp16.
        """
        from xllm.python.kernels_npu.triton.dcp_packed_a2a import (
            fused_dcp_lse_combine,
            pack_dcp_output_lse,
            packed_send_shape,
        )

        num_tokens, num_heads, head_dim = (int(x) for x in output.shape)
        lse_th = _lse_as_token_head(softmax_lse, num_tokens, num_heads)
        dcp_size = int(self.dcp_size)
        send_shape = packed_send_shape(
            num_tokens,
            num_heads,
            head_dim,
            dcp_size,
            scatter_dim=1,
            dtype=output.dtype,
        )
        send = get_execution_buffer(
            ("DCP_PACKED_A2A_SEND", *send_shape, str(output.dtype)),
            lambda: torch.empty(send_shape, dtype=output.dtype, device=output.device),
        )
        recv = get_execution_buffer(
            ("DCP_PACKED_A2A_RECV", *send_shape, str(output.dtype)),
            lambda: torch.empty(send_shape, dtype=output.dtype, device=output.device),
        )
        pack_dcp_output_lse(
            output,
            lse_th,
            dcp_size,
            scatter_dim=1,
            send=send,
        )
        dist.all_to_all_single(recv, send, group=self.dcp_group.device_group)
        h_local = num_heads // dcp_size
        merged = get_execution_buffer(
            ("SFA_DCP_MERGE_OUT", num_tokens, h_local, head_dim, str(output.dtype)),
            lambda: torch.empty(
                (num_tokens, h_local, head_dim),
                dtype=output.dtype,
                device=output.device,
            ),
        )
        return fused_dcp_lse_combine(recv, head_dim, scatter_dim=1, output=merged)

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
        return self._merge_dcp_outputs(sfa_output, softmax_lse)
