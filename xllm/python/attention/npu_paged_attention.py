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
from xllm.python.attention.expanded_decode_metadata import (
    resolve_expanded_decode_metadata,
)
from xllm.python.model_executor.cp_utils import cp_gather_kv
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_execution_buffer,
    get_forward_context,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.cp_utils import CpContext

# Ascend FIA sparse_mode values (see CANN aclnnFusedInferAttentionScore docs).
# 0: no compressed mask; used for single-query decode where no causal mask is
#    needed.
# 3: rightDownCausal; the causal mask is right-aligned to the KV tail, for the
#    prefix-cache / chunked-prefill case where q_len < kv_len so the new queries
#    attend the full cached prefix plus their own tokens (mode 2, leftUpCausal,
#    only aligns when q_len == kv_len and would misalign on a cache hit).
_SPARSE_MODE_NONE = 0
_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3


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
        # DeepSeek-V3.2 MLA uses a 512-wide latent cache and the sparse MLA
        # operators.  Ascend FIA graph tiling does not support that head
        # dimension, so it must not be initialized for this backend instance.
        self._is_mla = head_dim > 192 and num_kv_heads == 1

        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._use_expanded_decode = False
        self._block_table_i32: torch.Tensor | None = None
        self._actual_seq_lens: list[int] | None = None
        self._actual_seq_q: list[int] | torch.Tensor = []
        self._actual_seq_kv: list[int] | torch.Tensor = []
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

    @property
    def is_mla(self) -> bool:
        return self._is_mla

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        layer: "Attention",
    ) -> torch.Tensor:
        from xllm.python.kernels_npu.kda import (
            _causal_conv1d_fn,
            _causal_conv1d_update,
        )

        metadata = self._metadata
        assert metadata is not None, "execute_linear called before prepare()"
        layer_cache = self._kv_caches[layer.layer_id]
        conv_cache = layer_cache.conv
        ssm_cache = layer_cache.ssm
        assert conv_cache is not None and ssm_cache is not None, (
            "execute_linear requires a linear-attention layer cache (conv/ssm)")

        seq_len = mixed_qkv.shape[-1]
        conv_kernel_size = layer.conv_kernel_size
        conv_state_len = conv_kernel_size - 1
        head_dim = layer.head_dim
        qkv_dim = layer.qkv_dim
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill

        idx = metadata.linear_state_indices
        assert idx is not None, "execute_linear requires metadata.linear_state_indices"
        num_seqs = idx.shape[0]
        if idx.dtype != torch.int64:
            idx = idx.to(torch.int64)
        conv_state = conv_cache.index_select(0, idx)
        conv_state = conv_state.transpose(1, 2).contiguous()
        ssm_state = ssm_cache.index_select(0, idx)
        # Pooled slots may hold a prior request's stale state; zero cold slots
        # on prefill (decode continues its own warm slot).
        if is_prefill:
            his = metadata.has_initial_state
            assert his is not None, (
                "execute_linear requires metadata.has_initial_state for prefill")
            if not isinstance(his, torch.Tensor):
                his = torch.tensor(
                    his, dtype=torch.int64, device=conv_state.device
                )
            if his.shape[0] != num_seqs:
                raise ValueError(
                    f"has_initial_state length {his.shape[0]} != num_seqs "
                    f"{num_seqs}")
            # mask = [num_seqs, 1, ...] broadcasting to each state's dims; bool
            # mul zeroes cold slots without a zeros_like allocation.
            warm = his.to(torch.bool)
            conv_state = conv_state * warm.view(num_seqs, *([1] * (conv_state.ndim - 1)))
            ssm_state = ssm_state * warm.view(num_seqs, *([1] * (ssm_state.ndim - 1)))

        conv_weight = layer.conv1d.weight.squeeze(1)
        activation = layer.activation
        q_cu = metadata.q_cu_seq_lens
        if q_cu is None:
            q_cu = torch.tensor(
                [0, seq_len], dtype=torch.int64, device=mixed_qkv.device)
        else:
            q_cu = q_cu.to(torch.int64)
        q_cu_list = q_cu.tolist()
        outs = []
        for s in range(num_seqs):
            t0, t1 = q_cu_list[s], q_cu_list[s + 1]
            seg = mixed_qkv[:, :, t0:t1]   # [1, conv_dim, seg_len]
            cs = conv_state[s:s + 1]       # [1, conv_dim, state_len]
            seg_len = t1 - t0
            if seg_len == 1:
                outs.append(_causal_conv1d_update(
                    seg, cs, conv_weight, activation
                ))
            else:
                cin = torch.cat([cs, seg], dim=-1)
                outs.append(_causal_conv1d_fn(
                    cin, conv_weight, activation
                )[:, :, -seg_len:])
                conv_state[s] = cin[0, :, -conv_state_len:]
        mixed_qkv = torch.cat(outs, dim=-1)
        hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1
        )
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # Delta-rule on fla_npu fused ops (one varlen call for all sequences).
        core_attn_out, final_state = self._kda_fla_npu_varlen(
            query, key, value, gate, beta, ssm_state, head_dim, q_cu,
            is_prefill=is_prefill,
        )

        conv_cache.index_copy_(
            0, idx, conv_state.transpose(1, 2).contiguous()
        )
        ssm_cache.index_copy_(
            0, idx, final_state.float().contiguous()
        )
        return core_attn_out

    def _kda_fla_npu_varlen(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        gate: torch.Tensor, beta: torch.Tensor, ssm_state: torch.Tensor,
        head_dim: int, cu_seqlens: torch.Tensor, *, is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda
        from xllm.python.kernels_npu.kda import _l2norm

        scale = 1.0 / (head_dim ** 0.5)

        if not is_prefill:
            out, final_state = recurrent_kda(
                query, key, value, gate, beta,
                initial_state=ssm_state,
                cu_seqlens=cu_seqlens,
                layout="BSND", scale=scale,
                output_final_state=True, inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
            )
            return out.to(query.dtype), final_state

        q_in = _l2norm(query.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
        k_in = _l2norm(key.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
        v_in = value.to(torch.bfloat16).contiguous()
        result = chunk_kda_fwd(
            q_in, k_in, v_in, gate, beta, scale,
            chunk_size=64, layout="BSND",
            initial_state=ssm_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=False,
            return_intermediate_states=False,
        )
        core_attn_out = result[0].to(query.dtype)
        final_state = result[1]
        return core_attn_out, final_state

    @staticmethod
    def _query_sequence_ends(
        q_cu_seq_lens: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Accept both NPU q-cumulative layouts used by the runtime."""
        if q_cu_seq_lens is None:
            return None
        if q_cu_seq_lens.numel() == batch_size:
            return q_cu_seq_lens.to(torch.int32)
        if q_cu_seq_lens.numel() == batch_size + 1:
            return q_cu_seq_lens[1:].to(torch.int32)
        raise RuntimeError(
            "q cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        self._metadata = metadata
        expanded = resolve_expanded_decode_metadata(
            metadata, block_size=self.page_size
        )
        self._use_expanded_decode = expanded is not None
        block_table = (
            expanded.block_table if expanded is not None else metadata.block_table
        )
        kv_seq_lens = (
            expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        )
        kv_seq_lens_host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )

        if block_table is not None:
            self._block_table_i32 = block_table.to(torch.int32)
            real_batch = block_table.shape[0]
        else:
            self._block_table_i32 = None
            real_batch = 0

        if self._use_expanded_decode or graph_mode or self._is_mla:
            self._actual_seq_lens = None
        elif metadata.q_cu_seq_lens is not None:
            q_seq_lens = getattr(metadata, "q_seq_lens", None)
            if q_seq_lens is not None:
                batch_size = q_seq_lens.numel()
            elif metadata.block_table is not None:
                batch_size = metadata.block_table.shape[0]
            else:
                batch_size = max(metadata.q_cu_seq_lens.numel() - 1, 0)
            q_seq_ends = self._query_sequence_ends(
                metadata.q_cu_seq_lens,
                batch_size,
            )
            self._actual_seq_lens = q_seq_ends.cpu().tolist()
        else:
            self._actual_seq_lens = None

        if self._block_table_i32 is not None and not self._is_mla:
            if kv_seq_lens_host_values is None:
                raise RuntimeError(
                    "decode attention requires scheduler-provided host KV lengths"
                )
            if len(kv_seq_lens_host_values) != real_batch:
                if len(kv_seq_lens_host_values) > real_batch:
                    kv_seq_lens_host_values = kv_seq_lens_host_values[:real_batch]
                else:
                    raise RuntimeError(
                        "host KV lengths must have one entry per block-table row"
                    )
            self._actual_seq_q: list[int] = list(range(1, real_batch + 1))
            self._actual_seq_kv: list[int] = list(kv_seq_lens_host_values)
        else:
            self._actual_seq_q = []
            self._actual_seq_kv = []

        if (
            graph_mode
            and self._block_table_i32 is not None
            and not self._is_mla
        ):
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
                        sparse_mode=_SPARSE_MODE_NONE,
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
        if self._is_mla and kv_seq_lens is not None:
            mla_device = kv_seq_lens.device
            actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if self._use_expanded_decode:
                actual_seq_q = torch.arange(
                    1,
                    actual_seq_kv.numel() + 1,
                    dtype=torch.int32,
                    device=mla_device,
                )
            elif metadata.q_cu_seq_lens is not None:
                actual_seq_q = self._query_sequence_ends(
                    metadata.q_cu_seq_lens,
                    int(actual_seq_kv.numel()),
                ).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                actual_seq_q = torch.arange(
                    1, batch + 1, dtype=torch.int32, device=mla_device
                )
            if graph_mode:
                graph_batch = int(actual_seq_kv.numel())
                self._mla_actual_seq_q = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_Q", graph_batch),
                    lambda: torch.empty_like(actual_seq_q),
                )
                self._mla_actual_seq_kv = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_KV", graph_batch),
                    lambda: torch.empty_like(actual_seq_kv),
                )
                self._mla_actual_seq_q.copy_(actual_seq_q)
                self._mla_actual_seq_kv.copy_(actual_seq_kv)
            else:
                self._mla_actual_seq_q = actual_seq_q
                self._mla_actual_seq_kv = actual_seq_kv
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

        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        # Context-Parallel prefill: q/k/v are this rank's sequence shard while the
        # slot_mapping/metadata still describe the full global sequence (C++ does
        # not pre-shard the Python qwen3 path). All-gather K/V to the full
        # sequence, persist this rank's KV shard, and attend over its causal
        # prefix.
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            return self._prefill_cp(
                q_3d, k_3d, v_3d, metadata, cp_context, k_cache, v_cache
            )

        # Write KV to paged cache (kernel expects [T, kv_heads, head_dim]).
        kernels.reshape_paged_cache(
            metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache
        )

        if metadata.is_prefill or metadata.is_chunked_prefill:
            if self._use_expanded_decode:
                return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)
            return self._prefill(
                q_3d, k_3d, v_3d, k_cache, v_cache, metadata, num_tokens
            )
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
        if self._block_table_i32 is None:
            raise RuntimeError("MLA requires a block table")

        torch.ops.xllm_ops.reshape_paged_cache(
            metadata.slot_mapping, k_latent_3d, k_pe_3d, nope_cache, rope_cache
        )
        return self._mla_sparse(
            q_latent,
            q_pe,
            nope_cache,
            rope_cache,
            topk,
            self._block_table_i32,
            layer_id,
        )

    def mla_index_context(self, layer: "Attention") -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        assert self._block_table_i32 is not None
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        index_cache = self._kv_caches[layer.layer_id].index
        if index_cache is None:
            raise RuntimeError(
                f"MLA index cache is missing for layer {layer.layer_id}"
            )
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=metadata.slot_mapping,
            block_table=self._block_table_i32,
            actual_seq_q=self._mla_actual_seq_q,
            actual_seq_kv=self._mla_actual_seq_kv,
            update_index_cache=lambda values: self._update_mla_index_cache(
                index_cache, metadata.slot_mapping, values
            ),
        )

    @staticmethod
    def _update_mla_index_cache(
        index_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        cache_view = index_cache.view(-1, index_cache.size(-1))
        kernels.scatter_nd_update(
            cache_view,
            slot_mapping.reshape(-1, 1).clamp_min(0),
            values,
        )

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        out = get_execution_buffer(
            ("SFA_OUTPUT", layer_id) + tuple(q_latent.shape),
            lambda: torch.empty_like(q_latent),
        )
        return kernels.sparse_flash_attention_out(
            q_latent, nope_cache, nope_cache, topk,
            block_table,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
            q_pe, rope_cache, self.scale, 1,
            "TND", "PA_BSND", 3, out,
        )  # [T, H, kv_lora]

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self, q_3d: torch.Tensor, k_3d: torch.Tensor, v_3d: torch.Tensor,
        k_cache: torch.Tensor, v_cache: torch.Tensor,
        metadata: AttentionMetadata, num_tokens: int,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)

        # Prefix-cache hit (or chunked prefill with prior context): part of the
        # KV already lives in the paged cache, so this forward only carries the
        # new tokens (q_len < kv_len). Attend over the full paged KV via
        # block_table, mirroring _decode. Without this, the new query tokens
        # would only see their own KV (actual_seq_lengths_kv == q_len) and never
        # the cached prefix, diverging from a full recompute.
        if metadata.block_table is not None:
            block_size = k_cache.size(1)
            k_flat = k_cache.view(k_cache.size(0), block_size, -1)
            v_flat = v_cache.view(v_cache.size(0), block_size, -1)
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d, k_flat, v_flat,
                pse_shift=None,
                atten_mask=self._causal_mask,
                block_table=self._block_table_i32,
                actual_seq_lengths=actual_seq,
                actual_seq_lengths_kv=self._actual_seq_kv,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                block_size=block_size,
                sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
                softmax_lse_flag=False,
            )
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

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
            sparse_mode=_SPARSE_MODE_RIGHT_DOWN_CAUSAL,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Context-Parallel prefill: all-gather KV, attend over causal prefix
    # ------------------------------------------------------------------

    def _prefill_cp(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        metadata: AttentionMetadata,
        cp_context: "CpContext",
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill attention for this rank's zigzag sequence shard.

        q/k/v hold this rank's ``total_local`` rows (two owned chunks per
        sequence, padding rows zeroed). We all-gather K/V back to the full
        global-order sequence, write the complete KV into the paged cache (so a
        later non-CP decode sees every position), then run one FIA over this
        rank's real queries. Each owned (sequence, half) segment is a packed
        sub-sequence: its ``real_count`` queries attend the causal prefix
        ``[0, segment_start + real_count)`` selected by ``kv_gather_index``.
        With ``sparse_mode=3`` (right-aligned causal) query row ``i`` of a
        segment attends KV ``[0, segment_start + i]`` — its exact global causal
        range. Segments are independent sub-sequences delimited by
        ``q_cu_seqlens`` / ``kv_cu_seqlens``, so both owned chunks resolve in a
        single call.
        """
        local_tokens = q_3d.shape[0]

        kv_global_k = cp_gather_kv(k_3d, cp_context)
        kv_global_v = cp_gather_kv(v_3d, cp_context)

        # Persist the full global-order KV into this rank's paged cache.
        kernels.reshape_paged_cache(
            metadata.slot_mapping,
            kv_global_k.contiguous(),
            kv_global_v.contiguous(),
            k_cache,
            v_cache,
        )

        # A CP rank can own only padding chunks when every sequence in the batch
        # is shorter than the zigzag chunk grid (e.g. a 1-token prompt with
        # cp_size > 1). It then has no real queries. The KV all-gather above
        # already ran (collectives must stay in lockstep across ranks) and the
        # full global KV is now in this rank's paged cache, so skip the FIA:
        # calling it with a 0-row query and empty actual_seq_lengths is rejected
        # by npu_fused_infer_attention. Return the all-zero shard directly.
        if cp_context.query_index.numel() == 0:
            return q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)

        # Real queries this rank owns, packed per (sequence, half) segment.
        q_real = q_3d.index_select(0, cp_context.query_index).contiguous()
        # Each segment's causal KV prefix, packed in the same segment order.
        kv_prefix_k = kv_global_k.index_select(
            0, cp_context.kv_gather_index
        ).contiguous()
        kv_prefix_v = kv_global_v.index_select(
            0, cp_context.kv_gather_index
        ).contiguous()

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_real,
            kv_prefix_k,
            kv_prefix_v,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=cp_context.q_cu_seqlens,
            actual_seq_lengths_kv=cp_context.kv_cu_seqlens,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        output = output.reshape(-1, self.num_heads * self.head_dim)

        # Scatter real-query outputs back into the padded [total_local] layout;
        # padding rows stay zero (they are never selected by restore_index in
        # the subsequent all-gather merge).
        out_local = q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)
        out_local.index_copy_(0, cp_context.query_index, output)
        return out_local

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
            sparse_mode=_SPARSE_MODE_NONE,
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
            sparse_mode=_SPARSE_MODE_NONE,
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
