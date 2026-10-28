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
"""KDA (Kernelized Delta Attention) linear-attention forward + state I/O."""

from __future__ import annotations

from collections.abc import Callable
from itertools import accumulate
from typing import TYPE_CHECKING, Any

import torch

from xllm.python.attention.backend import resolve_linear_state_io_indices
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_forward_context_or_none,
    in_acl_graph,
)

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata, LayerCache
    from xllm.python.layers.kda import KdaForgetGate


class KdaLinearAttentionMixin:
    """KDA linear-attention + speculative verify state I/O, mixed into NpuPagedAttentionBackend."""

    _kda_validated_query_shape: tuple[int, int] | None = None

    _metadata: AttentionMetadata | None
    _kv_caches: list[LayerCache]
    _kda_checkpoint_stride: int | None
    _kda_query_start_locs: dict[tuple[int, int, torch.device], torch.Tensor]

    def _causal_conv1d(
        self,
        value: torch.Tensor,
        state: torch.Tensor,
        weight: torch.Tensor,
        activation: str,
        *,
        query_start_loc: list[int] | None = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Convolve staged channel-last states with fused activation."""
        inputs = value.transpose(1, 2).contiguous()
        if query_start_loc is not None:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        output = torch.ops.xllm_ops.causal_conv1d(
            inputs,
            weight,
            state,
            query_start_loc if query_start_loc is not None else [],
            1 if activation == "silu" else 0,
            0 if is_prefill else 1,
        )
        return output.reshape(value.shape[0], value.shape[2], value.shape[1]).transpose(1, 2)

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        beta: torch.Tensor,
        layer_id: int,
        conv_weight: torch.Tensor,
        forget_gate: KdaForgetGate,
        activation: str,
        raw_gate_proj: torch.Tensor,
    ) -> torch.Tensor:
        """Run KDA and update the framework's convolution/recurrent state slots.

        Returns ``[B, S, num_heads_local, head_dim]``. The conv1d + delta-rule
        math is identical to the model-layer self-contained path (validated
        against the transformers reference); only the state I/O moved here so
        both attention layer types dispatch through the backend.

        ``raw_gate_proj`` is the pre-gate projection ``f_b(f_a(x))``. Plain
        decode/prefill kernels fuse the safe-gate calculation in-kernel. Speculative
        verify uses framework checkpoints with a materialized safe-gate from
        ``KdaForgetGate.gate_from_raw``.
        """
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

        from xllm.python import kernels

        metadata = self._metadata
        assert metadata is not None, "execute_linear called before prepare()"
        layer_cache = self._kv_caches[layer_id]
        conv_cache = layer_cache.conv
        ssm_cache = layer_cache.ssm
        assert conv_cache is not None and ssm_cache is not None, (
            "execute_linear requires a linear-attention layer cache (conv/ssm)"
        )

        batch_size, _, seq_len = mixed_qkv.shape
        conv_state_len = conv_weight.shape[0] - 1
        head_dim = forget_gate.head_dim
        num_heads_local = forget_gate.num_heads
        qkv_dim = forget_gate.qkv_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        gate_lb = forget_gate.safe_gate_lower_bound
        read_idx, idx = resolve_linear_state_io_indices(metadata)
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        is_spec_verify = getattr(metadata, "is_spec_verify", False)
        if is_spec_verify and idx is None:
            raise RuntimeError("KDA spec-verify requires linear-state slots")
        num_seqs = idx.shape[0] if idx is not None else batch_size
        # ACL-graph decode with a flattened batch: the model forward unsqueezes
        # the 1-D ``[num_seqs]`` decode input into ``[1, num_seqs]``, so
        # mixed_qkv arrives as ``[1, conv_dim, num_seqs]`` with idx
        # ``[num_seqs]``. The eager multi-sequence branch (a Python loop over
        # q_cu_seq_lens with host syncs) is not graph-capturable; decode is
        # exactly one token per sequence, so reshape to ``[num_seqs, conv_dim,
        # 1]`` and take the simple per-sequence path with static shapes. gate
        # ``[1, T, nh, hd]`` / beta ``[1, T, nh]`` follow the same transpose.
        in_graph = in_acl_graph()
        is_decode = not is_prefill
        flatten_graph_decode = (
            in_graph
            and is_decode
            and not is_spec_verify
            and idx is not None
            and batch_size == 1
            and num_seqs > 1
            and seq_len == num_seqs
            # An expanded spec-verify batch keeps the [1, C, T] packing even
            # though its per-row slot count equals T: the KDA verify grouping
            # reads [B, C, rows_per_seq] from it (see the dispatch below).
            and getattr(metadata, "expanded_decode_metadata", None) is None
        )
        if flatten_graph_decode:
            mixed_qkv = mixed_qkv.transpose(0, 2)  # [1, C, T] -> [T, C, 1]
            batch_size, _, seq_len = mixed_qkv.shape
            hidden_shape = (batch_size, seq_len, -1, head_dim)
            beta = beta.transpose(0, 1)  # [1, T, nh] -> [T, 1, nh]
            raw_gate_proj = raw_gate_proj.transpose(0, 1)  # [1, T, nh, hd] -> [T, 1, nh, hd]
        if idx is None:
            device = mixed_qkv.device
            conv_state = torch.zeros(
                batch_size,
                conv_state_len,
                conv_weight.shape[1],
                dtype=mixed_qkv.dtype,
                device=device,
            )
            ssm_state = torch.zeros(
                batch_size,
                num_heads_local,
                head_dim,
                head_dim,
                dtype=torch.float32,
                device=device,
            )
        else:
            # Speculative verify expands one logical sequence into consecutive
            # batch rows (bonus + drafted tokens, e.g. q_cu=[0,1,2,3,4] for
            # two k=1 sequences, kv=[n,n+1,m,m+1] per row) while
            # linear_state_indices stays per SEQUENCE (one slot id per
            # sequence, e.g. [s1,s2] for the four rows above). Expand each
            # sequence's slot across its contiguous row group, then merge the
            # same-slot rows back into single sequences: the recurrent state
            # must chain bonus -> draft inside one sequence, and the cache
            # read/write needs one row per slot. Mapping rows to sequences by
            # position instead (rows==seqs) silently drops the tail rows of
            # every sequence past the first and corrupts the view/layout
            # downstream.
            q_cu_raw = metadata.q_cu_seq_lens
            if in_graph and getattr(metadata, "expanded_decode_metadata", None) is not None and q_cu_raw is not None:
                # Graph capture/replay of an expanded spec-verify batch. The
                # static metadata keeps the PER-ROW layout the attention
                # backends consume (q_cu/kv/block_table all per token row);
                # the per-sequence group count rides on q_seq_lens (N
                # entries of value spec_width). Pure shape math — no
                # device->host syncs. Row slots arrive per-row and pairwise
                # equal; take each group's row 0.
                num_rows = int(mixed_qkv.shape[2])
                q_seq_lens = getattr(metadata, "q_seq_lens", None)
                n_groups = int(q_seq_lens.numel()) if q_seq_lens is not None else int(q_cu_raw.numel()) - 1
                if num_seqs > 0 and n_groups > 0 and (num_rows % n_groups == 0):
                    if idx.numel() == num_rows:
                        group_idx = idx.view(n_groups, num_rows // n_groups)[:, 0].contiguous()
                    else:
                        group_idx = idx
                    return self._spec_verify(
                        mixed_qkv,
                        forget_gate.gate_from_raw(raw_gate_proj),
                        beta,
                        conv_weight,
                        activation,
                        group_idx,
                        metadata,
                        conv_cache,
                        ssm_cache,
                        recurrent_kda,
                    )
            q_rows = 0 if in_graph else (int(q_cu_raw.numel()) - 1 if q_cu_raw is not None else 0)
            per_row_idx = None
            per_row_cu = None
            if q_rows == idx.numel() and idx.numel() > 1 and bool((idx[1:] == idx[:-1]).any().item()):
                # Defensive: indices already duplicated per row.
                per_row_idx = idx
                per_row_cu = q_cu_raw
            elif q_rows > idx.numel() and q_rows % idx.numel() == 0:
                # Spec-verify expansion (uniform rows per sequence = k+1).
                per_row_idx = idx.repeat_interleave(q_rows // idx.numel())
                per_row_cu = q_cu_raw
            elif (
                0 < q_rows < idx.numel()
                and idx.numel() % q_rows == 0
                and idx.numel() > 1
                and bool((idx[1:] == idx[:-1]).any().item())
            ):
                # Chunked-typed spec verify: q_cu is sequence-scoped while
                # linear_state_indices is per-row (one row per token).
                # Normalize to the per-row view — every row is one token.
                per_row_idx = idx
                per_row_cu = torch.arange(idx.numel() + 1, dtype=torch.int64, device=idx.device)
            elif q_rows > idx.numel():
                raise RuntimeError(
                    f"unaligned linear-state batch: {q_rows} rows vs {idx.numel()} sequences with non-uniform expansion"
                )
            if per_row_idx is not None:
                if per_row_cu is None:
                    raise RuntimeError("KDA verify rows require cumulative query lengths")
                row_lengths = (per_row_cu[1:] - per_row_cu[:-1]).tolist()
                merged_lengths: list = []
                merged_slots: list = []
                for slot, length in zip(per_row_idx.tolist(), row_lengths, strict=True):
                    if merged_slots and slot == merged_slots[-1]:
                        merged_lengths[-1] += length
                    else:
                        merged_slots.append(slot)
                        merged_lengths.append(length)
                if len(set(merged_lengths)) != 1:
                    raise RuntimeError("KDA verify requires uniform rows per sequence")
                idx = torch.tensor(merged_slots, dtype=idx.dtype, device=idx.device)
            if per_row_idx is not None or (
                (not is_prefill or is_spec_verify)
                and getattr(metadata, "num_accepted_tokens", None) is not None
                and idx.numel() > 0
                and mixed_qkv.dim() == 3
                and mixed_qkv.shape[2] >= idx.numel()
                and mixed_qkv.shape[2] % idx.numel() == 0
                and (is_spec_verify or mixed_qkv.shape[2] > idx.numel())
            ):
                return self._spec_verify(
                    mixed_qkv,
                    forget_gate.gate_from_raw(raw_gate_proj),
                    beta,
                    conv_weight,
                    activation,
                    idx,
                    metadata,
                    conv_cache,
                    ssm_cache,
                    recurrent_kda,
                )
            if is_spec_verify:
                raise RuntimeError("KDA spec-verify requires a non-empty uniform token block within the decoding width")
            if idx.dtype != torch.int64:
                idx = idx.to(torch.int64)
            ssm_checkpoint_stride = self._kda_checkpoint_stride
            if ssm_checkpoint_stride is None:
                raise RuntimeError("KDA requires a bound linear-state checkpoint stride")
            conv_cache_base = conv_cache[:, :conv_state_len]
            state_read_idx = read_idx if is_prefill else idx
            conv_i = conv_cache_base.index_select(0, state_read_idx)
            ssm_i = ssm_cache.index_select(0, state_read_idx * ssm_checkpoint_stride)
            his = metadata.has_initial_state
            if his is not None and len(his) == num_seqs:
                if not isinstance(his, torch.Tensor):
                    his = torch.tensor(his, dtype=torch.int64, device=conv_i.device)
                cold = ~his.to(torch.bool)
                conv_i.masked_fill_(cold.view(num_seqs, 1, 1), 0)
                ssm_i.masked_fill_(cold.view(num_seqs, 1, 1, 1), 0)
            conv_state, ssm_state = conv_i, ssm_i.contiguous()
        scale = 1.0 / (head_dim**0.5)
        # Route on metadata, not seq_len: speculative decode can carry multiple
        # tokens per sequence (seq_len > 1) but is still a decode step; the
        # seq_len heuristic would wrongly send it to the chunked prefill path.
        device = mixed_qkv.device
        if num_seqs == batch_size:
            mixed_qkv = self._causal_conv1d(mixed_qkv, conv_state, conv_weight, activation, is_prefill=is_prefill)
        else:
            q_cu = metadata.q_cu_seq_lens
            assert q_cu is not None, "multi-sequence linear attention needs q_cu_seq_lens"
            host_values = getattr(metadata, "q_cu_seq_lens_host_values", None)
            if host_values is not None and len(host_values) == num_seqs + 1:
                q_cu_list = host_values
            elif host_values is not None and len(host_values) == num_seqs:
                q_cu_list = [0, *map(int, host_values)]
            else:
                q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
                if (
                    q_seq_lens_host is not None
                    and q_seq_lens_host.device.type == "cpu"
                    and q_seq_lens_host.numel() == num_seqs
                ):
                    q_cu_list = list(accumulate(map(int, q_seq_lens_host.tolist()), initial=0))
                else:
                    if q_cu is None:
                        raise RuntimeError("multi-sequence linear attention needs q_cu_seq_lens")
                    if q_cu.numel() != num_seqs + 1:
                        raise RuntimeError(
                            f"KDA query/state metadata length mismatch: q_cu={q_cu.numel()}, state={num_seqs}"
                        )
                    if q_cu.device.type != "cpu":
                        raise RuntimeError("NPU KDA multi-sequence prefill requires host q_cu_seq_lens metadata")
                    q_cu_list = [int(value) for value in q_cu.to(torch.int64).tolist()]
            del host_values
            mixed_qkv = self._causal_conv1d(
                mixed_qkv, conv_state, conv_weight, activation, query_start_loc=q_cu_list, is_prefill=is_prefill
            )
            seq_len = int(q_cu_list[-1])
            hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        # ``beta`` arrives as [B, S, nh] and is already correct for both
        # layouts: per-sequence [num_seqs, per_seq_len, nh] when the batch rows
        # map 1:1 to sequences, and flattened [1, T, nh] (T = sum of q_cu) for
        # the varlen multi-sequence path. Re-viewing it as
        # (num_seqs, seq_len, nh) assumes a uniform per-seq length == the
        # flattened total and crashes on multi-sequence decode batches
        # (e.g. 2 concurrent requests: view [2, 2, 4] on 8 elements).
        # fla_npu KDA ops require fp32 gate/beta (the pure-torch reference also
        # upcasts them); the model hands them in bf16.
        b = beta.to(torch.float32)
        fuse_gate = gate_lb is not None and -5.0 <= gate_lb < 0.0
        g = (raw_gate_proj if fuse_gate else forget_gate.gate_from_raw(raw_gate_proj)).view(hidden_shape).float()
        _gate_kwargs = (
            {
                "A_log": forget_gate.A_log.contiguous(),
                "dt_bias": forget_gate.dt_bias.contiguous(),
                "use_gate_in_kernel": True,
                "safe_gate": True,
                "lower_bound": gate_lb,
            }
            if fuse_gate
            else {"use_gate_in_kernel": False}
        )
        if not is_prefill:
            # decode (incl. multi-token speculative varlen): recurrent_kda on
            # packed TND [T, nh, hd] with cu_seqlens.
            q_tnd = query.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            k_tnd = key.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            v_tnd = value.reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
            g_tnd = g.reshape(-1, num_heads_local, head_dim).contiguous()
            b_tnd = b.reshape(-1, num_heads_local).contiguous()
            if num_seqs != batch_size:
                cu_seqlens = q_cu.to(torch.int32)
            elif seq_len == 1:
                # B independent single-token sequences.
                if in_graph:
                    # Constant content; allocate once into the persistent
                    # execution buffer so capture records no per-step H2D
                    # arange.
                    cu_seqlens = get_execution_buffer(
                        ("KDA_DECODE_CU_SEQLENS", num_seqs),
                        lambda: torch.arange(num_seqs + 1, dtype=torch.int32, device=device),
                    )
                else:
                    cu_seqlens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
            else:
                cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            # Plain decode hot path: fuse the safe-gate into the
            # kernel when the model handed a raw projection (``_gate_kwargs``
            # / ``g_raw_tnd``); else keep the python-materialized gate.
            core_attn_out, final_state = recurrent_kda(
                q_tnd,
                k_tnd,
                v_tnd,
                g_tnd,
                b_tnd,
                initial_state=ssm_state,
                cu_seqlens=cu_seqlens,
                layout="TND",
                scale=scale,
                output_final_state=True,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=True,
                **_gate_kwargs,
            )
            # recurrent_kda returns the packed TND [T, nh, hd] layout of its
            # inputs; restore the [B, S, nh, hd] grouping the model layer
            # expects (o_norm gates per head). For T == 1 the flat layout
            # happens to broadcast identically, which masked this for
            # single-stream decode; a multi-sequence decode batch (2
            # concurrent requests flattened to [1, 2, ...]) surfaced it.
            core_attn_out = core_attn_out.to(query.dtype).reshape(hidden_shape)
        else:
            q_in = kernels.l2_norm(query).to(torch.bfloat16).contiguous()
            k_in = kernels.l2_norm(key).to(torch.bfloat16).contiguous()
            v_in = value.to(torch.bfloat16).contiguous()
            cu_seqlens = (
                q_cu.to(torch.int32)
                if num_seqs != batch_size
                else torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            )
            if cu_seqlens.numel() > 2:
                # Seq-wise prefill: one single-sequence chunk_kda_fwd per
                # sequence. A merged multi-sequence prefill (engine batches the
                # concurrent requests' prefills) leaves per-seq conv/ssm state
                # that differs from a single-request prefill (state-fingerprint
                # verified: L0 state matches, L1+ diverges on seq1/seq2), and
                # that state drift propagates through every later verify step.
                # state_v_first=True pins the [HV,V,K] state layout so the
                # ssm state handed to decode matches the recurrent path
                # (default False is [HV,K,V] — K/V-transposed; K=V=128 hides
                # the shape mismatch while corrupting decode precision).
                _pout, _pstates = [], []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    sel = slice(int(t0), int(t1))
                    _r = chunk_kda_fwd(
                        q_in[:, sel].contiguous(),
                        k_in[:, sel].contiguous(),
                        v_in[:, sel].contiguous(),
                        g[:, sel].contiguous(),
                        b[:, sel].contiguous(),
                        scale,
                        chunk_size=64,
                        layout="BSND",
                        initial_state=ssm_state[s : s + 1],
                        output_final_state=True,
                        cu_seqlens=torch.tensor([0, int(t1 - t0)], dtype=torch.int32, device=device),
                        return_intermediate_states=False,
                        state_v_first=True,
                        **_gate_kwargs,
                    )
                    _pout.append(_r[0])
                    _pstates.append(_r[1])
                # Each _r[0] is [1, seq_len_s, nh, hd] (layout="BSND", one
                # sequence per call). The per-sequence token counts differ
                # across a multi-sequence prefill batch, so they must be
                # concatenated along the token axis (dim=1) to restore the
                # original [1, total_tokens, nh, hd] packing of q_in — cat on
                # dim=0 would require equal seq_len and crashes (aclnnCat 161002
                # "dim 1 of tensor 1 is [X], should be equal to tensor 0 [Y]")
                # at >=2 concurrent prefills of differing length.
                core_attn_out = torch.cat(_pout, dim=1).to(query.dtype)
                final_state = torch.cat(_pstates, dim=0)
            else:
                result = chunk_kda_fwd(
                    q_in,
                    k_in,
                    v_in,
                    g,
                    b,
                    scale,
                    chunk_size=64,
                    layout="BSND",
                    initial_state=ssm_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    return_intermediate_states=False,
                    state_v_first=True,
                    **_gate_kwargs,
                )
                core_attn_out = result[0].to(query.dtype)
                final_state = result[1]

        if idx is not None:
            conv_cache_base.index_copy_(0, idx, conv_state)
            ssm_cache.index_copy_(0, idx * ssm_checkpoint_stride, final_state.float().contiguous())
        # multi-seq path reshaped mixed_qkv to [num_seqs, ...]; flatten the
        # output back to [1, T, ...] so the KDA forward's hidden_shape [1, T]
        # aligns for o_norm / o_proj.
        if num_seqs != batch_size:
            core_attn_out = core_attn_out.reshape(1, -1, *core_attn_out.shape[2:])
        elif flatten_graph_decode:
            # The flatten-decode graph branch ran the simple path on
            # [num_seqs, 1, nh, hd]; restore the model's flattened [1, T, ...]
            # layout so o_norm's gate ([1, T, nh, hd]) aligns without
            # broadcasting.
            core_attn_out = core_attn_out.transpose(0, 1)
        return core_attn_out

    def _spec_verify(
        self,
        mixed_qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        conv_weight: torch.Tensor,
        activation: str,
        idx: torch.Tensor,
        metadata: AttentionMetadata,
        conv_cache: torch.Tensor,
        ssm_cache: torch.Tensor,
        recurrent_kda: Callable[..., Any],
    ) -> torch.Tensor:
        from xllm.python import kernels

        device = mixed_qkv.device
        num_seqs = idx.shape[0]
        if num_seqs == 0 or mixed_qkv.ndim != 3 or mixed_qkv.shape[0] != 1 or mixed_qkv.shape[2] % num_seqs:
            raise ValueError("KDA spec-verify requires flattened, equal-width sequence blocks")
        rows_per_seq = mixed_qkv.shape[2] // num_seqs
        in_graph = in_acl_graph()
        q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
        if (
            not in_graph
            and getattr(metadata, "is_spec_verify", False)
            and q_seq_lens_host is not None
            and q_seq_lens_host.numel() == num_seqs
            and self._kda_validated_query_shape != (num_seqs, rows_per_seq)
        ):
            if any(length != rows_per_seq for length in q_seq_lens_host.tolist()):
                raise ValueError("KDA spec-verify requires uniform query lengths")
            self._kda_validated_query_shape = (num_seqs, rows_per_seq)
        head_dim = gate.shape[-1]
        nh = gate.shape[-2]
        qkv_dim = nh * head_dim
        conv_dim = conv_weight.shape[1]
        conv_state_len = conv_weight.shape[0] - 1
        scale = 1.0 / (head_dim**0.5)
        required_conv_state_len = conv_state_len + rows_per_seq - 1
        if conv_cache.shape[1] < required_conv_state_len:
            raise RuntimeError("KDA conv checkpoint capacity is smaller than the verify width")

        num_accepted_tokens = getattr(metadata, "num_accepted_tokens", None)
        if num_accepted_tokens is None or num_accepted_tokens.ndim != 1 or num_accepted_tokens.numel() != num_seqs:
            raise RuntimeError("KDA requires one accepted-token count per logical sequence")
        num_accepted_tokens = num_accepted_tokens.to(torch.int32).contiguous()

        checkpoint_stride = self._kda_checkpoint_stride
        if checkpoint_stride is None:
            raise RuntimeError("KDA requires a bound linear-state checkpoint stride")
        if checkpoint_stride < rows_per_seq:
            raise RuntimeError("KDA SSM checkpoint capacity is smaller than the verify width")
        accepted_host = getattr(metadata, "num_accepted_tokens_host_values", ())
        if accepted_host and (
            len(accepted_host) != num_seqs or any(count < 1 or count > rows_per_seq for count in accepted_host)
        ):
            raise ValueError("KDA accepted-token counts must be in [1, verify width]")

        idx32 = idx.to(torch.int32).contiguous()
        context = get_forward_context_or_none()
        ssm_state_indices = None if context is None else context.layer_shared_cache.get(self)
        if ssm_state_indices is None:
            # The kernel reads column accepted - 1 and writes each token's checkpoint.
            ssm_state_indices = idx32.reshape(-1, 1) * checkpoint_stride + torch.arange(
                checkpoint_stride, dtype=torch.int32, device=idx32.device
            )
            if context is not None:
                context.layer_shared_cache[self] = ssm_state_indices

        qsl_key = (num_seqs, rows_per_seq, device)
        qsl_buf = self._kda_query_start_locs.get(qsl_key)
        if qsl_buf is None:
            qsl_buf = torch.arange(num_seqs + 1, dtype=torch.int32, device=device) * rows_per_seq
            self._kda_query_start_locs[qsl_key] = qsl_buf

        conv_input = mixed_qkv.transpose(1, 2).reshape(-1, conv_dim).contiguous()
        conv_out = kernels.causal_conv1d_update_v2(
            conv_input,
            conv_weight.t(),
            conv_cache,
            idx32,
            qsl_buf,
            rows_per_seq,
            num_accepted_tokens,
            bias=None,
            activation=activation in ("silu", "swish"),
        )
        c_split = conv_out.split(qkv_dim, dim=-1)
        q = c_split[0].reshape(-1, nh, head_dim).to(torch.bfloat16)
        k = c_split[1].reshape(-1, nh, head_dim).to(torch.bfloat16)
        v = c_split[2].reshape(-1, nh, head_dim).to(torch.bfloat16)
        gate_arg = gate.reshape(-1, nh, head_dim).float().contiguous()
        beta_flat = beta.reshape(-1, nh).float().contiguous()

        ret = recurrent_kda(
            q,
            k,
            v,
            gate_arg,
            beta_flat,
            initial_state=ssm_cache,
            cu_seqlens=qsl_buf,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            layout="TND",
            scale=scale,
            output_final_state=True,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=False,
            use_gate_in_kernel=False,
            state_v_first=True,
        )
        core_out = ret[0] if isinstance(ret, tuple) else ret
        return core_out.view(1, num_seqs * rows_per_seq, nh, head_dim)
