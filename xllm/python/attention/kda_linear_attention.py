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

import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from xllm.python.attention.kda_constants import (
    _KDA_NO_COORD,
    _KDA_SEQWISE,
    _KDA_VERIFY_V2,
    _KDA_VERIFY_V3,
    _MTP_FULL_COMMIT,
    _MTP_TRACE,
)
from xllm.python.model_executor.forward_context import (
    get_execution_buffer,
    get_forward_context_or_none,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


def _in_acl_graph() -> bool:
    """Whether the current forward runs under ACL graph warmup/capture.

    The decode graph runner always passes an ``execution_state`` (warmup and
    capture) and an ``acl_graph`` capture context (capture only); the eager
    runner sets neither, so eager paths stay byte-identical.
    """
    ctx = get_forward_context_or_none()
    return ctx is not None and (ctx.acl_graph is not None or ctx.execution_state is not None)


class KdaLinearAttentionMixin:
    """KDA linear-attention + MTP spec-verify state I/O, mixed into NpuPagedAttentionBackend."""

    def snapshot_kda_v2_state(self, idx: torch.Tensor):
        """Snapshot the V2 stash rows the graph warmup/capture will consume.

        Mirrors ``_snapshot_linear_state`` for the backend-owned spec-verify
        stash: the capture runs advance the stash just like the conv/ssm
        caches, so the entry contents must be restored afterwards.
        """
        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        snap = []
        for st in self.__dict__.get("_kda_v2", {}).values():
            if "armed_buf" not in st:
                continue
            snap.append(
                (
                    st,
                    idx64,
                    st["conv_out"].index_select(0, idx64).clone(),
                    st["g_raw"].index_select(0, idx64).clone(),
                    st["b_raw"].index_select(0, idx64).clone(),
                    st["tails"].index_select(1, idx64).clone(),
                    st["kv_prev"].index_select(0, idx64).clone(),
                    st["armed_buf"].index_select(0, idx64).clone(),
                )
            )
        return snap or None

    @staticmethod
    def restore_kda_v2_state(snap) -> None:
        if not snap:
            return
        for st, idx64, co, g, b, t, kv, ar in snap:
            st["conv_out"].index_copy_(0, idx64, co)
            st["g_raw"].index_copy_(0, idx64, g)
            st["b_raw"].index_copy_(0, idx64, b)
            st["tails"].index_copy_(1, idx64, t)
            st["kv_prev"].index_copy_(0, idx64, kv)
            st["armed_buf"].index_copy_(0, idx64, ar)

    def disarm_kda_v2_slots(self, idx: torch.Tensor) -> None:
        """Mark slots' V2 stash invalid (prefill restarts the chain)."""
        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        for st in self.__dict__.get("_kda_v2", {}).values():
            if "armed_buf" in st:
                st["armed_buf"].index_fill_(0, idx64, False)

    def disarm_kda_v3_slots(self, idx: torch.Tensor) -> None:
        """Mark slots' V3 combined-pool state invalid (prefill restart)."""
        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        for st in self.__dict__.get("_kda_v3", {}).values():
            if "armed_buf" in st:
                st["armed_buf"].index_fill_(0, idx64, False)

    def snapshot_kda_v3_state(self, idx: torch.Tensor):
        """Snapshot V3 combined-pool rows the graph warmup/capture mutates."""
        idx64 = idx if idx.dtype == torch.int64 else idx.to(torch.int64)
        snap = []
        for st in self.__dict__.get("_kda_v3", {}).values():
            if "armed_buf" not in st:
                continue
            # nslots is the C++ pool capacity (armed_buf/kv_prev are sized to
            # it); the combined pool holds rows_per_seq = R slots per seq
            # (base + R-1 drafts), so snapshot ALL R slots — graph capture
            # mutates every slot's conv/ssm state.
            nslots = st["armed_buf"].shape[0]
            rslots = st["combined_conv"].shape[0] // nslots
            slot_idx = [idx64 + j * nslots for j in range(rslots)]
            snap.append(
                (
                    st,
                    idx64,
                    nslots,
                    rslots,
                    [st["combined_conv"].index_select(0, s).clone() for s in slot_idx],
                    [st["combined_ssm"].index_select(0, s).clone() for s in slot_idx],
                    st["kv_prev"].index_select(0, idx64).clone(),
                    st["armed_buf"].index_select(0, idx64).clone(),
                )
            )
        return snap or None

    @staticmethod
    def restore_kda_v3_state(snap) -> None:
        if not snap:
            return
        for st, idx64, nslots, rslots, conv_snaps, ssm_snaps, kv, ar in snap:
            for j in range(rslots):
                st["combined_conv"].index_copy_(0, idx64 + j * nslots, conv_snaps[j])
                st["combined_ssm"].index_copy_(0, idx64 + j * nslots, ssm_snaps[j])
            st["kv_prev"].index_copy_(0, idx64, kv)
            st["armed_buf"].index_copy_(0, idx64, ar)

    def execute_linear(
        self,
        mixed_qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        """KDA delta-rule over framework conv/ssm slots.

        Returns ``[B, S, num_heads_local, head_dim]``. The conv1d + delta-rule
        math is identical to the model-layer self-contained path (validated
        against the transformers reference); only the state I/O moved here so
        both attention layer types dispatch through the backend.
        """
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda

        from xllm.python.models.glm5_next import (
            _causal_conv1d_fn,
            _causal_conv1d_update,
            _l2norm,
        )

        metadata = self._metadata
        assert metadata is not None, "execute_linear called before prepare()"
        layer_cache = self._kv_caches[layer.layer_id]
        conv_cache = layer_cache.conv
        ssm_cache = layer_cache.ssm
        assert conv_cache is not None and ssm_cache is not None, (
            "execute_linear requires a linear-attention layer cache (conv/ssm)"
        )

        batch_size, _, seq_len = mixed_qkv.shape
        conv_kernel_size = layer.conv_kernel_size
        conv_state_len = conv_kernel_size - 1
        head_dim = layer.head_dim
        num_heads_local = layer.num_heads_local
        qkv_dim = layer.qkv_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)

        # Raw (pre-conv) projections for the lazy-commit stash, kept alive
        # across the branches below (mixed_qkv is reassigned post-conv), and
        # the per-seq read-only chain bookkeeping for plain steps (see the
        # non-verify branch).
        raw_mixed, raw_gate, raw_beta = mixed_qkv, gate, beta
        chain_seqs: dict = {}

        idx = metadata.linear_state_indices
        num_seqs = idx.shape[0] if idx is not None else batch_size
        # ACL-graph decode with a flattened batch: the model forward unsqueezes
        # the 1-D ``[num_seqs]`` decode input into ``[1, num_seqs]``, so
        # mixed_qkv arrives as ``[1, conv_dim, num_seqs]`` with idx
        # ``[num_seqs]``. The eager multi-sequence branch (a Python loop over
        # q_cu_seq_lens with host syncs) is not graph-capturable; decode is
        # exactly one token per sequence, so reshape to ``[num_seqs, conv_dim,
        # 1]`` and take the simple per-sequence path with static shapes. gate
        # ``[1, T, nh, hd]`` / beta ``[1, T, nh]`` follow the same transpose.
        in_graph = _in_acl_graph()
        is_decode = not metadata.is_prefill and not metadata.is_chunked_prefill
        flatten_graph_decode = (
            in_graph
            and is_decode
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
            gate = gate.transpose(0, 1)  # [1, T, nh, hd] -> [T, 1, nh, hd]
            beta = beta.transpose(0, 1)  # [1, T, nh] -> [T, 1, nh]
        if idx is None:
            device = mixed_qkv.device
            conv_state = torch.zeros(
                batch_size,
                layer.conv_dim,
                conv_state_len,
                dtype=layer.conv1d.weight.dtype,
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
            if idx.dtype != torch.int64:
                idx = idx.to(torch.int64)
            # MTP spec-verify expands one logical sequence into consecutive
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
            merged_q_cu: Optional[torch.Tensor] = None
            merged_row0: list = []
            q_cu_raw = metadata.q_cu_seq_lens
            if (
                (_KDA_VERIFY_V2 or _KDA_VERIFY_V3)
                and in_graph
                and getattr(metadata, "expanded_decode_metadata", None) is not None
                and q_cu_raw is not None
            ):
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
                    from fla_npu.ops.ascendc import (
                        recurrent_kda as _rk,
                    )

                    _fn = self._spec_verify_v3 if _KDA_VERIFY_V3 else self._spec_verify_v2
                    return _fn(mixed_qkv, raw_gate, raw_beta, layer, group_idx, metadata, conv_cache, ssm_cache, _rk)
            # The row-merge + lazy-commit bookkeeping below is MTP-spec-verify
            # tracking that relies on device->host syncs (.item()/.tolist()),
            # which are forbidden on a captured ACL-graph stream. The graph
            # decode path is not spec-verify (merged_q_cu stays None and the
            # simple path below uses the capture-safe conv), so skip it whole.
            if os.environ.get("XLLM_KDA_DISPATCH_TRACE") == "1" and in_graph:
                _tn = getattr(self, "_kda_tr_n", 0)
                if _tn < 400:
                    self._kda_tr_n = _tn + 1
                    try:
                        _ea = bool(self.__dict__.get("_kda_v3", {}).get(layer.layer_id, {}).get("ever_armed"))
                    except Exception:
                        _ea = "n/a"
                    _shp = tuple(mixed_qkv.shape)
                    with open("/tmp/kda_dispatch.log", "a") as _fh:
                        _fh.write(
                            f"#{_tn} L={getattr(layer, 'layer_id', '?')} "
                            f"ing={in_graph} dec={is_decode} "
                            f"ns={num_seqs} idx={idx.numel() if idx is not None else 0} "
                            f"shp={_shp} qcu={(None if q_cu_raw is None else int(q_cu_raw.numel()) - 1)} "
                            f"exp={getattr(metadata, 'expanded_decode_metadata', None) is not None} "
                            f"armed={_ea} flat={flatten_graph_decode}\n"
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
                row_lengths = (per_row_cu[1:] - per_row_cu[:-1]).tolist()
                merged_lengths: list = []
                merged_slots: list = []
                for row, (slot, length) in enumerate(zip(per_row_idx.tolist(), row_lengths)):
                    if merged_slots and slot == merged_slots[-1]:
                        merged_lengths[-1] += length
                    else:
                        merged_slots.append(slot)
                        merged_lengths.append(length)
                        merged_row0.append(row)
                merged_cu = [0]
                for length in merged_lengths:
                    merged_cu.append(merged_cu[-1] + length)
                merged_q_cu = torch.tensor(merged_cu, dtype=torch.int64, device=idx.device)
                idx = torch.tensor(merged_slots, dtype=torch.int64, device=idx.device)
                num_seqs = idx.shape[0]
                if _KDA_VERIFY_V2 or _KDA_VERIFY_V3:
                    # Graph-shaped verify path: fixed-shape ops and device
                    # tensors only (see _spec_verify_v2 / _spec_verify_v3).
                    from fla_npu.ops.ascendc import recurrent_kda as _rk

                    _fn = self._spec_verify_v3 if _KDA_VERIFY_V3 else self._spec_verify_v2
                    try:
                        return _fn(mixed_qkv, raw_gate, raw_beta, layer, idx, metadata, conv_cache, ssm_cache, _rk)
                    except Exception:
                        import traceback

                        with open("/tmp/v2dbg.log", "a") as _fh:
                            _fh.write(traceback.format_exc() + "\n")
                        raise
            else:
                if _KDA_VERIFY_V3:
                    _v3_states = self.__dict__.get("_kda_v3", {})
                    if metadata.is_prefill or metadata.is_chunked_prefill:
                        if idx is not None:
                            self.disarm_kda_v3_slots(idx)
                    elif (
                        _v3_states.get(layer.layer_id, {}).get("ever_armed")
                        and idx is not None
                        and idx.numel() > 0
                        and mixed_qkv.dim() == 3
                        and mixed_qkv.shape[2] >= idx.numel()
                        and mixed_qkv.shape[2] % idx.numel() == 0
                    ):
                        # Plain (rejection-bootstrap) step inside an open V3
                        # chain: uniform path (rows_per_seq=1) advances the
                        # single confirmed token in one fused call.
                        from fla_npu.ops.ascendc import recurrent_kda as _rk

                        if os.environ.get("XLLM_KDA_DISPATCH_TRACE") == "1":
                            with open("/tmp/kda_dispatch.log", "a") as _fh:
                                _fh.write(f"#{getattr(self, '_kda_tr_n', 0)} BRANCH=V3_reject\n")
                        try:
                            return self._spec_verify_v3(
                                mixed_qkv, raw_gate, raw_beta, layer, idx, metadata, conv_cache, ssm_cache, _rk
                            )
                        except Exception:
                            import traceback

                            with open("/tmp/v2dbg.log", "a") as _fh:
                                _fh.write(traceback.format_exc() + "\n")
                            raise
                if _KDA_VERIFY_V2:
                    _v2_states = self.__dict__.get("_kda_v2", {})
                    if metadata.is_prefill or metadata.is_chunked_prefill:
                        # Prefill restarts the chain from a fresh state —
                        # per-slot, so concurrent decode sequences keep
                        # their stashes.
                        if idx is not None:
                            self.disarm_kda_v2_slots(idx)
                    elif (
                        _v2_states.get(layer.layer_id, {}).get("ever_armed")
                        and idx is not None
                        and idx.numel() > 0
                        and mixed_qkv.dim() == 3
                        and mixed_qkv.shape[2] >= idx.numel()
                        and mixed_qkv.shape[2] % idx.numel() == 0
                    ):
                        # Plain (rejection-bootstrap) step inside an open V2
                        # chain: same protocol — advance the stashed rows,
                        # chain this row read-only, re-stash it.
                        from fla_npu.ops.ascendc import recurrent_kda as _rk

                        try:
                            return self._spec_verify_v2(
                                mixed_qkv, raw_gate, raw_beta, layer, idx, metadata, conv_cache, ssm_cache, _rk
                            )
                        except Exception:
                            import traceback

                            with open("/tmp/v2dbg.log", "a") as _fh:
                                _fh.write(traceback.format_exc() + "\n")
                            raise
                # Non-verify path (plain decode / prefill): states are
                # committed wholesale the regular way, plus lazy-commit
                # handshakes - but ONLY once this backend has seen a
                # spec-verify batch, so a pure-decode engine (no MTP) keeps
                # its exact pre-MTP behavior. (1) A reject interrupts the
                # verify chain with a bootstrap plain step while the chain's
                # last confirmed token is still stashed un-advanced - consume
                # that stash FIRST (advance the live caches), or the token is
                # silently lost from the linear state at every rejection.
                # (2) A plain step with an open chain is processed
                # READ-ONLY: its row is stashed and committed by the next
                # verify's advance, so both its own output and the re-
                # processed row0 in the next verify match a plain decode
                # exactly (committing here would make the next verify's row0
                # run from a state already containing it). (3) Other plain
                # steps record a coverage marker (kv length, empty stash) so
                # the next verify does not re-advance them. Prefill resets
                # the chain (the slot restarts from a fresh state).
                if not _MTP_FULL_COMMIT and not in_graph:
                    _pending = getattr(self, "_mtp_pending", None)
                    if _pending is None:
                        _pending = {}
                        self._mtp_pending = _pending
                    _lp = _pending.setdefault(layer.layer_id, {})
                    # NOTE: kv_seq_lens.tolist() / idx.tolist() are device->host
                    # syncs forbidden on a captured stream; the graph path uses
                    # static metadata and the capture-safe conv in the simple
                    # path below, so this lazy-commit bookkeeping is eager-only.
                    _kv_list = metadata.kv_seq_lens.tolist() if metadata.kv_seq_lens is not None else None
                    _active = getattr(self, "_mtp_seen_verify", False)
                    _is_pf = metadata.is_prefill or metadata.is_chunked_prefill
                    if (
                        _MTP_TRACE
                        and layer.layer_id == 0
                        and not _is_pf
                        and getattr(self, "_mtp_stats", None) is not None
                    ):
                        # Non-prefill plain steps inside the decode phase are
                        # the rejection bootstraps - their share of all steps
                        # is the acceptance-rate signal.
                        self._mtp_stats["plain"] += num_seqs
                    _cw = layer.conv1d.weight.squeeze(1)
                    for s, slot in enumerate(idx.tolist()):
                        slot = int(slot)
                        if _is_pf or _kv_list is None or s >= len(_kv_list):
                            _lp.pop(slot, None)
                            continue
                        prev = _lp.get(slot)
                        if not _active or prev is None:
                            _lp[slot] = (
                                int(_kv_list[s]),
                                mixed_qkv[:, :, 0:0],
                                gate[:, 0:0],
                                beta[:, 0:0],
                                _cw,
                                layer.activation,
                            )
                            continue
                        (prev_base, prev_seg, prev_g, prev_b, _pw, _pact) = prev
                        _m = int(_kv_list[s]) - 1 - prev_base
                        if 0 < _m <= prev_seg.shape[2]:
                            _cs = conv_cache[slot].transpose(0, 1).unsqueeze(0).contiguous()
                            _cin = torch.cat([_cs, prev_seg[:, :, :_m]], dim=-1)
                            _adv_qkv = _causal_conv1d_fn(
                                _cin,
                                _cw,
                                layer.activation,
                            )[:, :, -_m:]
                            conv_cache[slot] = _cin[0, :, -conv_state_len:].transpose(0, 1)
                            _sp = torch.split(_adv_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1)
                            _, _st = recurrent_kda(
                                _sp[0].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous(),
                                _sp[1].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous(),
                                _sp[2].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous(),
                                prev_g[:, :_m].reshape(-1, num_heads_local, head_dim).to(torch.float32).contiguous(),
                                prev_b[:, :_m].reshape(-1, num_heads_local).to(torch.float32).contiguous(),
                                initial_state=ssm_cache[slot].unsqueeze(0),
                                cu_seqlens=torch.tensor([0, _m], dtype=torch.int32, device=mixed_qkv.device),
                                layout="TND",
                                scale=1.0 / (head_dim**0.5),
                                output_final_state=True,
                                inplace_final_state=False,
                                use_qk_l2norm_in_kernel=True,
                                use_gate_in_kernel=False,
                                use_beta_sigmoid_in_kernel=False,
                                state_v_first=True,
                            )
                            ssm_cache[slot] = _st[0].to(ssm_cache.dtype)
                        if 0 <= _m <= prev_seg.shape[2]:
                            # Open chain: read-only plain; stash at the
                            # tail write-back.
                            _lp.pop(slot, None)
                            chain_seqs[s] = slot
                            continue
                        _lp[slot] = (
                            int(_kv_list[s]),
                            mixed_qkv[:, :, 0:0],
                            gate[:, 0:0],
                            beta[:, 0:0],
                            _cw,
                            layer.activation,
                        )
            if (
                not _MTP_FULL_COMMIT
                and merged_q_cu is not None
                and not in_graph
                and not _KDA_NO_COORD
                and getattr(self, "_mtp_pending", None)
                and layer.layer_id == min(self._mtp_pending.keys())
            ):
                # Batched cross-layer lazy advance. Every KDA layer's
                # verify-step advance is independent (own weights/states,
                # same confirmed-token count), so the FIRST KDA layer of the
                # step advances them ALL here - one conv per layer plus ONE
                # recurrent_kda for the whole batch - instead of one extra
                # recurrent call inside each layer (~33 extra kernel
                # launches per step otherwise). The coordinator runs before
                # any layer reads the caches, so every layer (including this
                # one) reads its already-advanced state below. Non-consumable
                # entries (coverage markers, m == 0 open chains) are left in
                # place for the per-layer logic.
                _slot_base = {}
                _kv_all = metadata.kv_seq_lens.tolist() if metadata.kv_seq_lens is not None else None
                if _kv_all is not None:
                    for _si, _sl in enumerate(idx.tolist()):
                        _slot_base.setdefault(int(_sl), _kv_all[merged_row0[_si]] - 1)
                _bq, _bk, _bv, _bg, _bb = [], [], [], [], []
                _b_init, _b_scatter = [], []
                _b_cu = [0]
                for _lid, _lp2 in self._mtp_pending.items():
                    _lc = self._kv_caches[_lid]
                    for _sl, (_pb, _seg, _pg, _pbeta, _pw, _pact) in list(_lp2.items()):
                        _base = _slot_base.get(_sl)
                        if _base is None:
                            continue
                        _bm = _base - _pb
                        if not (0 < _bm <= _seg.shape[2]):
                            continue
                        del _lp2[_sl]
                        _cs = _lc.conv[_sl].transpose(0, 1).unsqueeze(0).contiguous()
                        _cin = torch.cat([_cs, _seg[:, :, :_bm]], dim=-1)
                        _cout = _causal_conv1d_fn(_cin, _pw, _pact)[:, :, -_bm:]
                        _lc.conv[_sl] = _cin[0, :, -conv_state_len:].transpose(0, 1)
                        _sp = torch.split(_cout.transpose(1, 2), [qkv_dim] * 3, dim=-1)
                        _bq.append(_sp[0].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16))
                        _bk.append(_sp[1].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16))
                        _bv.append(_sp[2].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16))
                        _bg.append(_pg[:, :_bm].reshape(-1, num_heads_local, head_dim).to(torch.float32))
                        _bb.append(_pbeta[:, :_bm].reshape(-1, num_heads_local).to(torch.float32))
                        _b_init.append(_lc.ssm[_sl].unsqueeze(0))
                        _b_scatter.append((_lid, _sl))
                        _b_cu.append(_b_cu[-1] + _bm)
                if _bq:
                    _, _bst = recurrent_kda(
                        torch.cat(_bq).contiguous(),
                        torch.cat(_bk).contiguous(),
                        torch.cat(_bv).contiguous(),
                        torch.cat(_bg).contiguous(),
                        torch.cat(_bb).contiguous(),
                        initial_state=torch.cat(_b_init, dim=0).contiguous(),
                        cu_seqlens=torch.tensor(_b_cu, dtype=torch.int32, device=mixed_qkv.device),
                        layout="TND",
                        scale=1.0 / (head_dim**0.5),
                        output_final_state=True,
                        inplace_final_state=False,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False,
                        state_v_first=True,
                    )
                    for _bi, (_lid, _sl) in enumerate(_b_scatter):
                        self._kv_caches[_lid].ssm[_sl] = _bst[_bi].to(self._kv_caches[_lid].ssm.dtype)
            conv_i = conv_cache.index_select(0, idx)
            conv_i = conv_i.transpose(1, 2).contiguous()
            ssm_i = ssm_cache.index_select(0, idx)
            his = metadata.has_initial_state
            if his is not None and len(his) == num_seqs:
                if not isinstance(his, torch.Tensor):
                    his = torch.tensor(his, dtype=torch.int64, device=conv_i.device)
                warm = his.to(torch.bool).view(num_seqs, 1, 1)
                conv_i = torch.where(warm, conv_i, torch.zeros_like(conv_i))
                ssm_i = torch.where(
                    warm.view(num_seqs, 1, 1, 1),
                    ssm_i,
                    torch.zeros_like(ssm_i),
                )
            conv_state, ssm_state = conv_i, ssm_i.contiguous()
            if chain_seqs:
                # Entry snapshots (post consume-advance) for the read-only
                # plain chain rows: restored at the tail write-back so this
                # step's own token is not committed here.
                _entry_conv = {i: conv_state[i].clone() for i in chain_seqs}
                _entry_ssm = {i: ssm_state[i].clone() for i in chain_seqs}

        conv_weight = layer.conv1d.weight.squeeze(1)
        activation = layer.activation
        scale = 1.0 / (head_dim**0.5)
        # Route on metadata, not seq_len: MTP/spec decode can carry multiple
        # tokens per sequence (seq_len > 1) but is still a decode step; the
        # seq_len heuristic would wrongly send it to the chunked prefill path.
        is_prefill = metadata.is_prefill or metadata.is_chunked_prefill
        device = mixed_qkv.device
        # Spec-verify (merged same-slot rows) commits lazily via the
        # kv-delta scheme in the flattened branch; see the comments there.
        commit_first_only = False
        # A merged spec-verify batch must take the flattened branch even when
        # it collapses to ONE sequence (single-stream verify: 2 rows -> 1
        # merged seq == batch_size): the simple path full-commits both rows,
        # which inserts a phantom draft token into the state on every
        # rejection (the corrected token overwrites the draft's position, so
        # only the bonus row of the previous step is confirmable here - see
        # the lazy-commit scheme below).
        if num_seqs == batch_size and merged_q_cu is None:
            # Simple path: mixed_qkv is already [B, conv_dim, S] (one sequence
            # per batch row, or a single flattened sequence).
            if seq_len == 1:
                if in_graph:
                    # F.conv1d is an aclop NPUGraph cannot capture; the manual
                    # depthwise mul-add is capture-safe. Eager keeps F.conv1d.
                    from xllm.python.models.glm5_next import (
                        _causal_conv1d_update_graph,
                    )

                    mixed_qkv = _causal_conv1d_update_graph(mixed_qkv, conv_state, conv_weight, activation)
                else:
                    mixed_qkv = _causal_conv1d_update(mixed_qkv, conv_state, conv_weight, activation)
            else:
                conv_in = torch.cat([conv_state, mixed_qkv], dim=-1)
                mixed_qkv = _causal_conv1d_fn(conv_in, conv_weight, activation)[:, :, -seq_len:]
                conv_state = conv_in[..., -conv_state_len:]
                if conv_state.shape[-1] < conv_state_len:
                    conv_state = F.pad(
                        conv_state,
                        (conv_state_len - conv_state.shape[-1], 0),
                        value=0,
                    )
        else:
            # Flattened multi-sequence: mixed_qkv is [1, conv_dim, T] (T = sum
            # of per-seq token counts). Variable-length (MTP/spec decode: each
            # sequence may carry a different token count) is supported by a
            # per-sequence conv1d loop (pure-torch F.conv1d is batched and
            # requires equal lengths) followed by a single varlen recurrent_kda
            # call (cu_seqlens does the per-seq split inside the kernel).
            q_cu = merged_q_cu if merged_q_cu is not None else (metadata.q_cu_seq_lens)
            assert q_cu is not None, "multi-sequence linear attention needs q_cu_seq_lens"
            q_cu = q_cu.to(torch.int64)
            q_cu_list = q_cu.tolist()
            commit_first_only = merged_q_cu is not None
            if commit_first_only:
                # First spec-verify batch arms the plain-step handshakes (a
                # pure-decode backend never sets this and keeps the exact
                # pre-MTP plain-path behavior).
                self._mtp_seen_verify = True
                if _MTP_TRACE and layer.layer_id == 0:
                    _st = getattr(self, "_mtp_stats", None)
                    if _st is None:
                        _st = {"verify": 0, "plain": 0}
                        self._mtp_stats = _st
                    _st["verify"] += num_seqs
                    if _st["verify"] >= 50:
                        _mh = getattr(self, "_mtp_mhist", {})
                        _mh_str = " ".join(f"m{k}={v}" for k, v in sorted(_mh.items()))
                        with open("/tmp/mtp_accept.log", "a") as _fh:
                            _fh.write(
                                f"[accept] pid={os.getpid()} "
                                f"verify={_st['verify']} "
                                f"plain(reject)={_st['plain']} "
                                f"accept_rate="
                                f"{_st['verify'] / max(1, _st['verify'] + _st['plain']):.2f} "
                                f"{_mh_str}\n"
                            )
                        _st["verify"] = 0
                        _st["plain"] = 0
                        self._mtp_mhist = {}
                # ---- MTP spec-verify steps (observed layout, k=1) ----
                # Rows are [last-confirmed token (re-processed each step to
                # judge the next draft), drafted token]; the C++ commits
                # exactly one token per step (the draft on accept, the
                # corrected argmax on reject) and the next step's row0 is
                # that token at the previous row1's position, so kv grows
                # +1 per step. Alternative scheme (full,
                # GLM5_MTP_COMMIT=full): commit BOTH rows every step - the
                # row0 re-write is near-idempotent under the delta rule but
                # a rejected draft leaves a persistent phantom write.
                #
                # Default scheme (lazy, see _MTP_FULL_COMMIT): commit
                # NOTHING here; (1) ADVANCE the live state by the PREVIOUS
                # step's confirmed tokens - their count m (kv_seq_lens
                # growth over the previous verify) with the stashed raw
                # qkv/gate rows; (2) process the current rows read-only
                # (outputs only); (3) stash the current rows for the next
                # step's advance. The state then tracks the true token
                # stream exactly, independent of accept/reject.
                kv_rows = metadata.kv_seq_lens.tolist()
                pending = getattr(self, "_mtp_pending", None)
                if pending is None:
                    pending = {}
                    self._mtp_pending = pending
                # The stash is PER LAYER: all KDA layers of a model share this
                # backend instance, and each layer's stashed qkv/gate rows are
                # only valid for that layer's own conv/recurrent advance.
                # Keying by slot alone made every layer overwrite the others'
                # stashes (layer N's advance consumed the last layer's rows -
                # total state corruption under multi-layer models).
                pending = pending.setdefault(layer.layer_id, {})
                live_slots = {int(x) for x in idx.tolist()}
                for slot in list(pending.keys()):
                    if slot not in live_slots:
                        del pending[slot]
                gate_raw = gate.view(1, -1, -1) if gate.dim() == 2 else gate
                beta_raw = beta.view(1, -1, -1) if beta.dim() == 2 else beta
                adv_seg: list = []  # raw qkv [1, conv_dim, m]
                adv_g: list = []
                adv_b: list = []
                adv_seq: list = []  # merged-seq index per segment
                for s in range(num_seqs):
                    slot = int(idx[s])
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    base_now = int(kv_rows[merged_row0[s]]) - 1
                    prev = pending.get(slot)
                    lead = 0
                    if prev is not None and not _MTP_FULL_COMMIT:
                        (prev_base, prev_seg, prev_g, prev_b, _pw, _pact) = prev
                        m = base_now - prev_base
                        if _MTP_TRACE and layer.layer_id == 0:
                            _mh = getattr(self, "_mtp_mhist", None)
                            if _mh is None:
                                _mh = {}
                                self._mtp_mhist = _mh
                            _mh[m] = _mh.get(m, 0) + 1
                        if 0 < m <= prev_seg.shape[2]:
                            adv_seg.append(prev_seg[:, :, :m])
                            adv_g.append(prev_g[:, :m])
                            adv_b.append(prev_b[:, :m])
                            adv_seq.append(s)
                        # Rows already covered by the recorded coverage
                        # (e.g. the plain step before the first verify) must
                        # not be stashed again - trim them off the front.
                        lead = max(0, min(prev_base - base_now, t1 - t0))
                    if not _MTP_FULL_COMMIT:
                        pending[slot] = (
                            base_now + lead,
                            mixed_qkv[:, :, t0 + lead : t1].clone(),
                            gate_raw[:, t0 + lead : t1].clone(),
                            beta_raw[:, t0 + lead : t1].clone(),
                            conv_weight,
                            activation,
                        )
                if adv_seg:
                    # conv-advance each segment sequentially (per slot) and
                    # collect post-conv qkv for the recurrent advance.
                    adv_conv_out = []
                    for s, seg_raw in zip(adv_seq, adv_seg):
                        cs = conv_state[s : s + 1].contiguous()
                        seg_raw = seg_raw.contiguous()
                        cin = torch.cat([cs, seg_raw], dim=-1)
                        adv_conv_out.append(_causal_conv1d_fn(cin, conv_weight, activation)[:, :, -seg_raw.shape[2] :])
                        conv_state[s] = cin[0, :, -conv_state_len:]
                    adv_qkv = torch.cat(adv_conv_out, dim=-1)
                    a_lengths = [seg.shape[2] for seg in adv_seg]
                    a_cu = [0]
                    for length in a_lengths:
                        a_cu.append(a_cu[-1] + length)
                    a_total = a_cu[-1]
                    a_split = torch.split(adv_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1)
                    aq = a_split[0].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
                    ak = a_split[1].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
                    av = a_split[2].reshape(-1, num_heads_local, head_dim).to(torch.bfloat16).contiguous()
                    ag = torch.cat(adv_g, dim=1).reshape(-1, num_heads_local, head_dim).to(torch.float32).contiguous()
                    ab = torch.cat(adv_b, dim=1).reshape(-1, num_heads_local).to(torch.float32).contiguous()
                    adv_idx = torch.tensor(adv_seq, dtype=torch.long, device=ssm_state.device)
                    if _KDA_SEQWISE and len(adv_seq) > 1:
                        # Same seq-wise rationale as the read-only branch: keep
                        # every advance call at the single-sequence shape so
                        # concurrent batches advance state bit-identically to a
                        # single-request run.
                        for _si, _slot in enumerate(adv_seq):
                            _a0, _a1 = a_cu[_si], a_cu[_si + 1]
                            _asel = slice(int(_a0), int(_a1))
                            _, _st = recurrent_kda(
                                aq[_asel].contiguous(),
                                ak[_asel].contiguous(),
                                av[_asel].contiguous(),
                                ag[_asel].contiguous(),
                                ab[_asel].contiguous(),
                                initial_state=ssm_state.index_select(0, adv_idx[[_si]]),
                                cu_seqlens=torch.tensor([0, int(_a1 - _a0)], dtype=torch.int32, device=device),
                                layout="TND",
                                scale=scale,
                                output_final_state=True,
                                inplace_final_state=False,
                                use_qk_l2norm_in_kernel=True,
                                use_gate_in_kernel=False,
                                use_beta_sigmoid_in_kernel=False,
                                state_v_first=True,
                            )
                            ssm_state[adv_idx[_si]] = _st.to(ssm_state.dtype)
                    else:
                        _, adv_state = recurrent_kda(
                            aq,
                            ak,
                            av,
                            ag,
                            ab,
                            initial_state=ssm_state.index_select(0, adv_idx),
                            cu_seqlens=torch.tensor(a_cu, dtype=torch.int32, device=device),
                            layout="TND",
                            scale=scale,
                            output_final_state=True,
                            inplace_final_state=False,
                            use_qk_l2norm_in_kernel=True,
                            use_gate_in_kernel=False,
                            use_beta_sigmoid_in_kernel=False,
                            state_v_first=True,
                        )
                        ssm_state[adv_idx] = adv_state.to(ssm_state.dtype)
                outs = []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    seg = mixed_qkv[:, :, t0:t1].contiguous()
                    # contiguous: a row-view of the batched cache has a
                    # different layout than a single-request's whole cache
                    # and the conv kernel picks its tiling off that layout.
                    cs = conv_state[s : s + 1].contiguous()
                    cin = torch.cat([cs, seg], dim=-1)
                    outs.append(
                        _causal_conv1d_fn(
                            cin,
                            conv_weight,
                            activation,
                        )[:, :, -(t1 - t0) :]
                    )
                    if _MTP_FULL_COMMIT:
                        # Full-commit scheme keeps the live conv state at the
                        # tail of BOTH rows (mirrors the pre-merge simple
                        # path); lazy mode leaves it at the advance boundary.
                        conv_state[s] = cin[0, :, -conv_state_len:]
                mixed_qkv = torch.cat(outs, dim=-1)
            else:
                outs = []
                for s in range(num_seqs):
                    t0, t1 = q_cu_list[s], q_cu_list[s + 1]
                    seg = mixed_qkv[:, :, t0:t1]  # [1, conv_dim, seg_len]
                    cs = conv_state[s : s + 1]  # [1, conv_dim, state_len]
                    seg_len = t1 - t0
                    if seg_len == 1:
                        outs.append(_causal_conv1d_update(seg, cs, conv_weight, activation))
                    else:
                        cin = torch.cat([cs, seg], dim=-1)
                        outs.append(_causal_conv1d_fn(cin, conv_weight, activation)[:, :, -seg_len:])
                        conv_state[s] = cin[0, :, -conv_state_len:]
                        if conv_state.shape[-1] < conv_state_len:
                            conv_state[s] = F.pad(
                                conv_state[s],
                                (conv_state_len - conv_state.shape[-1], 0),
                                value=0,
                            )
                mixed_qkv = torch.cat(outs, dim=-1)  # [1, conv_dim, T]
            # TND packed layout for recurrent_kda: [T, nh, hd] per channel group.
            seq_len = int(q_cu_list[-1])
            hidden_shape = (1, seq_len, -1, head_dim)

        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [qkv_dim] * 3, dim=-1)
        query = query.view(hidden_shape)
        key = key.view(hidden_shape)
        value = value.view(hidden_shape)

        g = gate if num_seqs == batch_size else gate.view(hidden_shape)
        # ``beta`` arrives as [B, S, nh] and is already correct for both
        # layouts: per-sequence [num_seqs, per_seq_len, nh] when the batch rows
        # map 1:1 to sequences, and flattened [1, T, nh] (T = sum of q_cu) for
        # the varlen multi-sequence path. Re-viewing it as
        # (num_seqs, seq_len, nh) assumes a uniform per-seq length == the
        # flattened total and crashes on multi-sequence decode batches
        # (e.g. 2 concurrent requests: view [2, 2, 4] on 8 elements).
        b = beta
        # fla_npu KDA ops require fp32 gate/beta (the pure-torch reference also
        # upcasts them); the model hands them in bf16.
        g = g.to(torch.float32)
        b = b.to(torch.float32)
        if not is_prefill:
            # decode (incl. MTP multi-token-per-seq varlen): recurrent_kda on
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
            if commit_first_only and not is_prefill and not _MTP_FULL_COMMIT:
                # Spec-verify: the state was already ADVANCED by the
                # previous step's confirmed tokens (see the flattened
                # branch). Chain the current rows read-only from that state
                # for their outputs; the tail write-back persists exactly
                # the advanced state (never the drafted rows).
                # state_v_first=True matches the V-first accumulation order of
                # the KDA reference and the prefill/advance paths so every
                # read here is layout-consistent with the advanced state.
                # inplace_final_state stays False on purpose: the lazy-advance
                # scheme owns ssm_state mutations (advance step writes, verify
                # reads); an in-place write here would clobber the advanced
                # state before the tail write-back persists it.
                if _KDA_SEQWISE and num_seqs > 1 and num_seqs != batch_size:
                    # Seq-wise dispatch: one single-segment recurrent call per
                    # sequence, exactly matching the single-request verify call
                    # shape. The flattened multi-segment call picks a different
                    # device tiling inside the engine process (verified
                    # bit-exact offline but not in-process), leaking a 1-ULP
                    # drift into the KDA output that the next layer's mHC
                    # sinkhorn amplifies ~16x per layer until argmax flips —
                    # concurrent outputs then diverge from single-request ones.
                    _ros = []
                    for s in range(num_seqs):
                        _t0, _t1 = q_cu_list[s], q_cu_list[s + 1]
                        _sel = slice(int(_t0), int(_t1))
                        _ro = recurrent_kda(
                            q_tnd[_sel].contiguous(),
                            k_tnd[_sel].contiguous(),
                            v_tnd[_sel].contiguous(),
                            g_tnd[_sel].contiguous(),
                            b_tnd[_sel].contiguous(),
                            initial_state=ssm_state[s : s + 1].contiguous(),
                            cu_seqlens=torch.tensor([0, int(_t1 - _t0)], dtype=torch.int32, device=device),
                            layout="TND",
                            scale=scale,
                            output_final_state=False,
                            inplace_final_state=False,
                            use_qk_l2norm_in_kernel=True,
                            use_gate_in_kernel=False,
                            use_beta_sigmoid_in_kernel=False,
                            state_v_first=True,
                        )
                        _ros.append(_ro[0] if isinstance(_ro, tuple) else _ro)
                    core_attn_out = torch.cat(_ros, dim=0)
                    final_state = ssm_state
                else:
                    ro = recurrent_kda(
                        q_tnd,
                        k_tnd,
                        v_tnd,
                        g_tnd,
                        b_tnd,
                        initial_state=ssm_state,
                        cu_seqlens=cu_seqlens,
                        layout="TND",
                        scale=scale,
                        output_final_state=False,
                        inplace_final_state=False,
                        use_qk_l2norm_in_kernel=True,
                        use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False,
                        state_v_first=True,
                    )
                    # fla_npu returns a tuple even with
                    # output_final_state=False.
                    core_attn_out = ro[0] if isinstance(ro, tuple) else ro
                    final_state = ssm_state
            else:
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
                    use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False,
                    state_v_first=True,
                )
            # recurrent_kda returns the packed TND [T, nh, hd] layout of its
            # inputs; restore the [B, S, nh, hd] grouping the model layer
            # expects (o_norm gates per head). For T == 1 the flat layout
            # happens to broadcast identically, which masked this for
            # single-stream decode; a multi-sequence decode batch (2
            # concurrent requests flattened to [1, 2, ...]) surfaced it.
            core_attn_out = core_attn_out.to(query.dtype).reshape(hidden_shape)
        else:
            q_in = _l2norm(query.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
            k_in = _l2norm(key.float(), dim=-1, eps=1e-6).to(torch.bfloat16).contiguous()
            v_in = value.to(torch.bfloat16).contiguous()
            cu_seqlens = (
                q_cu.to(torch.int32)
                if num_seqs != batch_size
                else torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            )
            if _KDA_SEQWISE and cu_seqlens.numel() > 2:
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
                        use_gate_in_kernel=False,
                        return_intermediate_states=False,
                        state_v_first=True,
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
                    use_gate_in_kernel=False,
                    return_intermediate_states=False,
                    state_v_first=True,
                )
                core_attn_out = result[0].to(query.dtype)
                final_state = result[1]

        if idx is not None:
            if chain_seqs:
                # Read-only plain chain rows: restore the entry (post
                # consume-advance) state and stash this step's own raw rows;
                # the next verify's advance commits them exactly once.
                _pend = self._mtp_pending[layer.layer_id]
                _kv_tail = metadata.kv_seq_lens.tolist() if metadata.kv_seq_lens is not None else None
                for i, slot in chain_seqs.items():
                    conv_state[i] = _entry_conv[i]
                    final_state[i] = _entry_ssm[i]
                    if num_seqs != batch_size:
                        t0, t1 = q_cu_list[i], q_cu_list[i + 1]
                        seg = raw_mixed[:, :, t0:t1]
                        gseg = raw_gate[:, t0:t1]
                        bseg = raw_beta[:, t0:t1]
                    else:
                        seg = raw_mixed[i : i + 1]
                        gseg = raw_gate[i : i + 1]
                        bseg = raw_beta[i : i + 1]
                    _pend[slot] = (
                        int(_kv_tail[i]) - seg.shape[2],
                        seg.clone(),
                        gseg.clone(),
                        bseg.clone(),
                        layer.conv1d.weight.squeeze(1),
                        layer.activation,
                    )
            conv_cache.index_copy_(0, idx, conv_state.transpose(1, 2).contiguous())
            ssm_cache.index_copy_(0, idx, final_state.float().contiguous())
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
