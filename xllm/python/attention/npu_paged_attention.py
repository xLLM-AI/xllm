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

"""NPU attention backend using Fused-Infer-Attention (FIA).

Registers as the PrivateUse1 (NPU) backend for the Python model executor.
Prefill uses FIA TND with causal mask; decode uses FIA TND with block_table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu

from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
)
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_forward_context,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


class NpuPagedAttentionBackend(AttentionBackend):
    """NPU attention backend dispatching to npu_fused_infer_attention_score."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device

        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._mla_actual_seq_q: torch.Tensor | None = None
        self._mla_actual_seq_kv: torch.Tensor | None = None
        self._causal_mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), 1)
            .to(torch.int8)
            .contiguous()
            .to(device)
        )

    @property
    def num_kv_blocks(self) -> int:
        if not self._kv_caches:
            return 0
        key_cache = self._kv_caches[0].key
        return key_cache.shape[0] if key_cache is not None else 0

    @property
    def page_size(self) -> int:
        if not self._kv_caches:
            return 1
        key_cache = self._kv_caches[0].key
        return key_cache.shape[1] if key_cache is not None else 1

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._metadata = metadata
        if metadata.q_cu_seq_lens is not None:
            self._actual_seq_lens: list[int] | None = (
                metadata.q_cu_seq_lens[1:].cpu().tolist()
            )
        else:
            self._actual_seq_lens = None

        if metadata.block_table is not None:
            self._block_table_i32 = metadata.block_table.to(torch.int32)

            real_batch = metadata.block_table.shape[0]

            kv_host = metadata.kv_seq_lens_host
            if kv_host is not None:
                kv_host = kv_host.cpu()
                if kv_host.numel() == real_batch + 1:
                    per_seq_kv = kv_host[1:] - kv_host[:-1]
                else:
                    per_seq_kv = kv_host
            else:
                per_seq_kv = torch.ones(real_batch, dtype=torch.int32)

            kv_list = per_seq_kv[:real_batch].tolist()

            self._actual_seq_q: list[int] = list(range(1, real_batch + 1))
            self._actual_seq_kv: list[int] = kv_list
        else:
            self._block_table_i32 = None

        if graph_mode and self._block_table_i32 is not None:
            graph_batch_size = self._block_table_i32.shape[0]
            if self._graph_workspace is None:
                block_size = self.page_size
                dummy_q = torch.empty(
                    graph_batch_size, self.num_heads, self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                dummy_kv = torch.empty(
                    self.num_kv_blocks, block_size,
                    self.num_kv_heads * self.head_dim,
                    dtype=self.dtype, device=self.device,
                )
                self._graph_workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=dummy_q,
                        key=dummy_kv,
                        value=dummy_kv,
                        block_table=self._block_table_i32,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=self._actual_seq_q,
                        actual_seq_lengths_kv=self._actual_seq_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        sparse_mode=0,
                        scale=self.scale,
                        softmax_lse_flag=False,
                    )
                )
            if graph_batch_size not in self._graph_outputs:
                self._graph_outputs[graph_batch_size] = torch.empty(
                    graph_batch_size,
                    self.num_heads,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._graph_lses[graph_batch_size] = torch.empty(
                    0, dtype=self.dtype, device=self.device
                )
            self._current_graph_output = self._graph_outputs[graph_batch_size]
            self._current_graph_lse = self._graph_lses[graph_batch_size]

        # Pre-cache MLA (sparse SFA) seq-lens once per step; shared by
        # execute_mla / mla_index_context instead of re-derived per layer.
        if metadata.kv_seq_lens is not None:
            kv_seq_lens = metadata.kv_seq_lens
            mla_device = kv_seq_lens.device
            self._mla_actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if metadata.q_cu_seq_lens is not None:
                self._mla_actual_seq_q = metadata.q_cu_seq_lens[1:].to(
                    torch.int32
                ).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                self._mla_actual_seq_q = torch.arange(
                    1, batch + 1, dtype=torch.int32, device=mla_device
                )
        else:
            self._mla_actual_seq_q = None
            self._mla_actual_seq_kv = None

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None

        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        k_cache, v_cache = layer_cache.key, layer_cache.value
        if k_cache is None or v_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer_id}")
        num_tokens = q.shape[0]

        # Write KV to paged cache (kernel expects [T, kv_heads, head_dim]).
        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        kernels.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        if metadata.is_prefill or metadata.is_chunked_prefill:
            return self._prefill(q_3d, k_3d, v_3d, metadata, num_tokens)
        return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        layer: "Attention",
        topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Absorbed-MLA attention. Returns [T, H, kv_lora]; caller bmm's W_UV."""
        metadata = self._metadata
        assert metadata is not None, "execute_mla called before prepare()"
        if topk is None:
            raise NotImplementedError(
                "dense MLA (topk=None) is not yet supported on "
                "NpuPagedAttentionBackend"
            )
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        # MLA reuses the K/V slots for the latent (nope) and rope caches.
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer_id}")

        torch.ops.xllm_ops.reshape_paged_cache(
            metadata.slot_mapping, k_latent_3d, k_pe_3d, nope_cache, rope_cache
        )
        return self._mla_sparse(
            q_latent, q_pe, nope_cache, rope_cache, topk, metadata.block_table
        )

    def mla_index_context(self, layer: "Attention") -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        index_cache = self._kv_caches[layer.layer_id].index
        if index_cache is None:
            raise RuntimeError(
                f"MLA index cache is missing for layer {layer.layer_id}"
            )
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=metadata.slot_mapping,
            block_table=metadata.block_table,
            actual_seq_q=self._mla_actual_seq_q,
            actual_seq_kv=self._mla_actual_seq_kv,
        )

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.ops.xllm_ops.sparse_flash_attention(
            q_latent, nope_cache, nope_cache, topk,
            block_table,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
            q_pe, rope_cache, self.scale, 1,
            "TND", "PA_BSND", 3,
        )
        return out  # [T, H, kv_lora]

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self, q_3d: torch.Tensor, k_3d: torch.Tensor, v_3d: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_3d, v_3d,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Decode: FIA with block_table (paged KV, no gather)
    # ------------------------------------------------------------------

    def _fia_out(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        block_size: int,
    ) -> None:
        torch.ops.npu.npu_fused_infer_attention_score.out(
            q, k, v,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q,
            actual_seq_lengths_kv=self._actual_seq_kv,
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=0,
            block_size=block_size,
            softmax_lse_flag=False,
            workspace=self._graph_workspace,
            out=[self._current_graph_output, self._current_graph_lse],
        )

    def _decode(
        self, q_3d: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        block_size = k_cache.size(1)
        k_flat = k_cache.view(k_cache.size(0), block_size, -1)
        v_flat = v_cache.view(v_cache.size(0), block_size, -1)

        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            if self._current_graph_output is None:
                raise RuntimeError("ACL graph output buffer is not prepared")
            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                self._fia_out(q_3d, k_flat, v_flat, block_size)
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)

            def _update_fia_args() -> None:
                self._fia_out(q_3d, k_flat, v_flat, block_size)

            graph_context.tasks.append(
                AclGraphTask(event, handle, _update_fia_args)
            )
            return self._current_graph_output.reshape(
                num_tokens, self.num_heads * self.head_dim
            )

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d, k_flat, v_flat,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=self._actual_seq_q[:num_tokens],
            actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
            block_table=self._block_table_i32,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=0,
            block_size=block_size,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cumulative_seq_lens(
        self, metadata: AttentionMetadata, num_tokens: int,
    ) -> list[int]:
        if self._actual_seq_lens is not None:
            return self._actual_seq_lens
        return [num_tokens]
