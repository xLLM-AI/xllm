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

"""DeepSeek-V4 DSA metadata construction (faithful Python port of the C++
``DSAMetadataBuilder`` in ``core/layers/common/dsa_metadata_builder.cpp``).

This is the Python-path counterpart of the C++ ``build_dsa_fields`` step: it
turns the framework-allocated ``multi_block_tables`` (per-manager block tables,
exposed through ``AttentionMetadataView``) plus the per-layer
``caches_info`` / ``group_infos`` (rebuilt from ``compress_ratios`` +
``window_size``) into the per-layer ``block_tables`` / ``slot_mappings`` the
DSA attention kernel consumes, along with the c4/c128 compressed positions,
sequence-length metadata, and RoPE tables.

The C++ model forward builds this inside ``DeepseekV4ModelImpl``; under
``--model_impl python`` the C++ forward never runs, so the Python DSA attention
backend builds it here from the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

# ---------------------------------------------------------------------------
# Cache-type enum (mirrors ``DSACacheType`` in dsa_metadata.h).
# ---------------------------------------------------------------------------
DSA_CACHE_TOKEN = 0
DSA_CACHE_SEQUENCE = 1
DSA_CACHE_SLIDING_WINDOW = 2


@dataclass
class DSACacheInfo:
    """Per-cache descriptor: which group it belongs to and its own shape."""

    group_id: int
    cache_type: int
    ratio: int
    block_size: int


@dataclass
class DSAGroupInfo:
    """Per-group descriptor: one block-manager pool."""

    cache_type: int
    ratio: int
    block_size: int


@dataclass
class DsaMetadata:
    """Per-forward DSA metadata, shared across all layers of one model."""

    # Current layer selected by the model loop. Mirrors C++
    # DSAMetadata::layer_id and is updated immediately before each layer runs.
    layer_id: int

    # Sequence lengths (host, int32).
    seq_lens: torch.Tensor
    seq_lens_q: torch.Tensor
    actual_seq_lengths_kv: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    kv_cu_seq_lens: torch.Tensor
    max_seqlen_kv: torch.Tensor
    max_seqlen_q: torch.Tensor
    max_query_len: int
    max_seq_len: int

    # Positions.
    input_positions: torch.Tensor
    c4_pad_positions: torch.Tensor
    c128_pad_positions: torch.Tensor
    start_pos: torch.Tensor

    # RoPE base tables.
    cos_table: torch.Tensor | None = None
    sin_table: torch.Tensor | None = None
    # Per-ratio compressed RoPE tables (C++ DeepseekV4RotaryEmbedding c4/c128
    # groups, compress_rope_theta, no mscale). Populated per-request by indexing
    # the compress RoPE cache with c4/c128_pad_positions.
    c4_cos: torch.Tensor | None = None
    c4_sin: torch.Tensor | None = None
    c128_cos: torch.Tensor | None = None
    c128_sin: torch.Tensor | None = None

    # block_tables / slot_mappings: [n_layers][n_caches_per_layer]; caches in the
    # same group share the same underlying tensor (no copy).
    block_tables: list[list[torch.Tensor]] = field(default_factory=list)
    slot_mappings: list[list[torch.Tensor]] = field(default_factory=list)

    # Precomputed AICPU tiling metadata (filled by the backend, not the builder).
    c1_metadata: torch.Tensor | None = None
    c4_metadata: torch.Tensor | None = None
    c128_metadata: torch.Tensor | None = None
    qli_metadata: torch.Tensor | None = None
    hadamard: torch.Tensor | None = None

    # Keep AICPU metadata-builder inputs alive until all asynchronously
    # enqueued kernels (and, for ACL graph, the captured graph entry) are done.
    # C++ gets this lifetime from the owning DSAMetadata fields; Python creates
    # additional empty optional tensors and device copies while precomputing.
    precomputed_metadata_inputs: tuple[torch.Tensor, ...] = ()

    # Owns the NPU storage for request-shaped metadata packed in prepare().
    # Tensor fields above may be views into this buffer, matching C++
    # DSAMetadata::packed_metadata_buffer.
    packed_metadata_buffer: torch.Tensor | None = None

    is_acl_graph: bool = False

    # Per-forward DeepSeek-V4 context-parallel state. This is the Python
    # counterpart of DSAMetadata::v4_cp_context; it is populated only for
    # prefill when cp_size > 1 and must never survive into a later forward.
    v4_cp_context: object | None = None


def _normalize_compress_ratio(ratio: int) -> int:
    """Mirrors ``deepseek_v4_normalize_compress_ratio``."""
    return 1 if ratio <= 1 else ratio


def build_cache_specs(
    compress_ratios: Sequence[int],
    window_size: int,
    n_layers: int,
) -> tuple[list[list[DSACacheInfo]], list[DSAGroupInfo]]:
    """Python port of ``deepseek_v4_build_cache_specs`` (deepseek_v4.h:332).

    Builds the per-layer ``caches_info`` and the deduplicated ``group_infos``
    from ``compress_ratios`` + ``window_size``. Group 0 is always the SWA
    (sliding-window) group; TOKEN groups for ratios {4, 128} are registered in
    the order they first appear.
    """
    base_block_size = 128
    group_infos: list[DSAGroupInfo] = []
    group_key_map: dict[tuple[int, int, int], int] = {}

    def register_group(cache_type: int, ratio: int, block_size: int) -> int:
        key = (ratio, cache_type, block_size)
        gid = group_key_map.get(key)
        if gid is not None:
            return gid
        gid = len(group_infos)
        group_key_map[key] = gid
        group_infos.append(DSAGroupInfo(cache_type, ratio, block_size))
        return gid

    register_group(DSA_CACHE_SLIDING_WINDOW, 1, window_size)
    for ratio in compress_ratios:
        cr = _normalize_compress_ratio(ratio)
        if cr in (4, 128):
            register_group(DSA_CACHE_TOKEN, cr, base_block_size)

    caches_info: list[list[DSACacheInfo]] = [[] for _ in range(n_layers)]
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 1
        cr = _normalize_compress_ratio(cr)

        if cr == 1:
            entries = [(DSA_CACHE_SLIDING_WINDOW, 1, window_size)]
        elif cr == 4:
            # cmp_kv, cmp_index, swa, kv_state, score_state, idx_kv,
            # idx_score, indexer_scale.
            entries = [
                (DSA_CACHE_TOKEN, 4, base_block_size),
                (DSA_CACHE_TOKEN, 4, base_block_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_TOKEN, 4, base_block_size),
            ]
        elif cr == 128:
            entries = [
                (DSA_CACHE_TOKEN, 128, base_block_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
                (DSA_CACHE_SLIDING_WINDOW, 1, window_size),
            ]
        else:
            entries = []

        for cache_type, ratio, block_size in entries:
            gid = register_group(cache_type, ratio, block_size)
            caches_info[layer_id].append(DSACacheInfo(gid, cache_type, ratio, block_size))

    return caches_info, group_infos


class DsaMetadataBuilder:
    """Faithful Python port of ``DSAMetadataBuilder`` (dsa_metadata_builder.cpp).

    Construct once per model (with the static ``caches_info`` / ``group_infos``),
    then call :meth:`build` every forward to expand the per-manager block tables
    into per-layer ``block_tables`` / ``slot_mappings``.
    """

    def __init__(
        self,
        caches_info: list[list[DSACacheInfo]],
        group_infos: list[DSAGroupInfo],
    ) -> None:
        self.caches_info = caches_info
        self.group_infos = group_infos

    # -- public API ---------------------------------------------------------

    def build(
        self,
        multi_block_tables: Sequence[torch.Tensor],
        kv_seq_lens: Sequence[int],
        q_seq_lens: Sequence[int] | None,
        positions: torch.Tensor,
        dsa_cos_sin: torch.Tensor | None,
        is_prefill: bool,
        is_chunked_prefill: bool,
        enable_graph: bool = False,
        graph_block_table_capacity_cols: int = 0,
        max_query_len: int = 0,
        max_seq_len: int = 0,
    ) -> DsaMetadata:
        batch_size = len(kv_seq_lens)
        if q_seq_lens is None or len(q_seq_lens) != batch_size:
            if is_prefill or is_chunked_prefill:
                q_lens = list(kv_seq_lens)
            else:
                q_lens = [1] * batch_size
        else:
            q_lens = list(q_seq_lens)

        dsa = self._build_seq_lengths(
            kv_seq_lens,
            q_lens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
        )
        dsa.input_positions = positions
        dsa.is_acl_graph = enable_graph
        if dsa_cos_sin is not None and dsa_cos_sin.numel() > 0:
            cos_sin_chunks = dsa_cos_sin.chunk(2, dim=-1)
            dsa.cos_table = cos_sin_chunks[0].contiguous()
            dsa.sin_table = cos_sin_chunks[1].contiguous()
        if positions is not None and positions.numel() > 0:
            self._build_positions(dsa, kv_seq_lens, q_lens, enable_graph)
        dsa.start_pos = (dsa.actual_seq_lengths_kv - dsa.seq_lens_q).to(torch.int32)

        self._build_block_tables_and_slots(
            multi_block_tables,
            kv_seq_lens,
            q_lens,
            batch_size,
            positions,
            enable_graph,
            graph_block_table_capacity_cols,
            dsa,
        )
        return dsa

    # -- step 1: sequence lengths (build_seq_lengths, cpp:574-661) ----------

    def _build_seq_lengths(
        self,
        kv_seq_lens: Sequence[int],
        q_lens: Sequence[int],
        *,
        max_query_len: int,
        max_seq_len: int,
    ) -> DsaMetadata:
        device = torch.device("cpu")
        kv = torch.tensor(kv_seq_lens, dtype=torch.int32, device=device)
        q = torch.tensor(q_lens, dtype=torch.int32, device=device)
        zeros_prefix = torch.zeros(1, dtype=torch.int32, device=device)
        actual_seq_lengths_query = torch.cat([zeros_prefix, q.cumsum(0).to(torch.int32)])
        kv_cu = torch.cat([zeros_prefix, kv.cumsum(0).to(torch.int32)])
        max_kv = kv.max().to(torch.int32) if kv.numel() else torch.zeros(1, dtype=torch.int32, device=device)
        max_q = q.max().to(torch.int32) if q.numel() else torch.zeros(1, dtype=torch.int32, device=device)
        max_query_len = max(int(max_query_len), max((int(value) for value in q_lens), default=0))
        max_seq_len = max(
            int(max_seq_len),
            max((int(value) for value in kv_seq_lens), default=0),
        )
        return DsaMetadata(
            layer_id=-1,
            seq_lens=kv,
            seq_lens_q=q,
            actual_seq_lengths_kv=kv,
            actual_seq_lengths_query=actual_seq_lengths_query,
            kv_cu_seq_lens=kv_cu,
            max_seqlen_kv=max_kv,
            max_seqlen_q=max_q,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            input_positions=torch.empty(0),
            c4_pad_positions=torch.empty(0, dtype=torch.int64),
            c128_pad_positions=torch.empty(0, dtype=torch.int64),
            start_pos=torch.empty(0),
        )

    # -- step 2: positions (build_positions, cpp:663-777) ------------------

    def _build_positions(
        self,
        dsa: DsaMetadata,
        kv_seq_lens: Sequence[int],
        q_lens: Sequence[int],
        enable_graph: bool,
    ) -> None:
        """Collect c4/c128 compressed RoPE positions.

        For each query token at absolute ``pos``, when ``(pos + 1) % ratio == 0``
        the compressed RoPE needs the position ``next_pos - ratio``.
        """
        total_tokens = int(dsa.input_positions.numel())
        c4_positions: list[int] = []
        c128_positions: list[int] = []
        for seq, kv_len in enumerate(kv_seq_lens):
            q_len = min(q_lens[seq], kv_len)
            start_pos = kv_len - q_len
            for i in range(q_len):
                pos = start_pos + i
                next_pos = pos + 1
                if next_pos % 4 == 0:
                    c4_positions.append(next_pos - 4)
                if next_pos % 128 == 0:
                    c128_positions.append(next_pos - 128)

        def _pad(positions: list[int], ratio: int) -> torch.Tensor:
            if enable_graph:
                # Graph mode pads to total_tokens so the tensor address is stable
                # across bucket sizes. C++ vector::resize() zero-fills the tail.
                out = torch.zeros(total_tokens, dtype=dsa.input_positions.dtype)
                for idx, p in enumerate(positions):
                    out[idx] = p
                return out
            # Non-graph: resize to min(total_tokens, total_tokens//ratio + batch_size)
            # with 0 padding, matching C++ dsa_metadata_builder.cpp:717
            # (c4_target = min(num_tokens, num_tokens/4 + batch_size)).
            batch_size = len(kv_seq_lens)
            target = min(total_tokens, total_tokens // ratio + batch_size)
            out = torch.zeros(target, dtype=dsa.input_positions.dtype)
            for idx, p in enumerate(positions):
                if idx >= target:
                    break
                out[idx] = p
            return out

        dsa.c4_pad_positions = _pad(c4_positions, 4)
        dsa.c128_pad_positions = _pad(c128_positions, 128)

    # -- step 3: block_tables / slot_mappings (build_dsa_fields, cpp:169-264)

    def _build_block_tables_and_slots(
        self,
        multi_block_tables: Sequence[torch.Tensor],
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        positions: torch.Tensor,
        enable_graph: bool,
        graph_block_table_capacity_cols: int,
        dsa: DsaMetadata,
    ) -> None:
        if not multi_block_tables or not self.caches_info:
            return

        active = list(multi_block_tables)
        manager_num = len(active)
        # Packed [manager, blocks] auto-unpack when batch_size == 1.
        if (
            manager_num == 1
            and batch_size == 1
            and active[0].dim() == 2
            and active[0].size(0) > 1
            and active[0].size(0) <= len(self.group_infos)
        ):
            packed = active[0].contiguous()
            active = [packed[m].unsqueeze(0).contiguous() for m in range(packed.size(0))]
            manager_num = len(active)

        if manager_num > len(self.group_infos):
            raise ValueError(f"manager count {manager_num} exceeds group count {len(self.group_infos)}")
        if enable_graph and graph_block_table_capacity_cols > 0:
            for manager_id, block_table in enumerate(active):
                if block_table.dim() != 2:
                    raise ValueError(
                        f"ACL graph multi_block_tables must be 2-D: manager {manager_id} has rank {block_table.dim()}"
                    )
                if block_table.size(1) > graph_block_table_capacity_cols:
                    raise ValueError(
                        "ACL graph block table exceeds bucket capacity: "
                        f"manager {manager_id} requires {block_table.size(1)} "
                        f"columns, capacity is {graph_block_table_capacity_cols}"
                    )

        graph_slot_capacity = int(positions.numel()) if enable_graph and positions.numel() > 0 else 0
        total_tokens = sum(int(x) for x in ctx_lens)

        proc_bt: list[torch.Tensor] = [torch.empty(0)] * manager_num
        proc_slots: list[torch.Tensor] = [torch.empty(0)] * manager_num
        for m in range(manager_num):
            gi = self.group_infos[m]
            proc_bt[m], proc_slots[m] = self._process_group(
                active[m],
                gi,
                ctx_lens,
                q_lens,
                batch_size,
                total_tokens,
                graph_slot_capacity,
                graph_block_table_capacity_cols,
            )

        n_layers = len(self.caches_info)
        dsa.block_tables = [[] for _ in range(n_layers)]
        dsa.slot_mappings = [[] for _ in range(n_layers)]
        for lid in range(n_layers):
            for ci in range(len(self.caches_info[lid])):
                gid = self.caches_info[lid][ci].group_id
                if gid < manager_num:
                    dsa.block_tables[lid].append(proc_bt[gid])
                    dsa.slot_mappings[lid].append(proc_slots[gid])
                else:
                    dsa.block_tables[lid].append(torch.empty(0))
                    dsa.slot_mappings[lid].append(torch.empty(0))

    # -- per-group processing (process_group, cpp:323-362) -----------------

    def _process_group(
        self,
        raw_bt: torch.Tensor,
        gi: DSAGroupInfo,
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        total_tokens: int,
        graph_slot_capacity: int,
        graph_block_table_capacity_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gi.cache_type == DSA_CACHE_TOKEN:
            return self._process_token_group(
                raw_bt,
                gi.ratio,
                gi.block_size,
                ctx_lens,
                q_lens,
                batch_size,
                graph_slot_capacity,
                graph_block_table_capacity_cols,
            )
        if gi.cache_type == DSA_CACHE_SLIDING_WINDOW:
            return self._process_swa_group(
                raw_bt,
                gi.block_size,
                ctx_lens,
                q_lens,
                batch_size,
                graph_slot_capacity,
                graph_block_table_capacity_cols,
            )
        # SEQUENCE: expand the whole context.
        return self._expand_blocks_to_slots(raw_bt, gi, ctx_lens, batch_size, total_tokens)

    # -- TOKEN group (process_token_group, cpp:364-464) --------------------
    # Commits only the compressed rows crossed by the current forward step.

    def _process_token_group(
        self,
        raw_bt: torch.Tensor,
        ratio: int,
        block_size: int,
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        graph_slot_capacity: int,
        graph_block_table_capacity_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        committed_rows = 0
        for seq in range(batch_size):
            ctx_len = int(ctx_lens[seq])
            q_len = max(0, min(int(q_lens[seq]), ctx_len))
            prev_ctx_len = ctx_len - q_len
            committed_rows += ctx_len // ratio - prev_ctx_len // ratio

        out_slot_rows = max(graph_slot_capacity, committed_rows) if graph_slot_capacity > 0 else committed_rows
        out_slots = torch.full((out_slot_rows,), -1, dtype=torch.int32, device=raw_bt.device)
        semantic_cols = int(raw_bt.size(1))

        def slot_for_compressed_index(seq: int, compressed_idx: int) -> int:
            if seq >= raw_bt.size(0) or semantic_cols <= 0:
                return -1
            block_idx = compressed_idx // block_size
            if block_idx >= semantic_cols:
                return -1
            block_id = int(raw_bt[seq, block_idx].item())
            if block_id < 0:
                return -1
            block_offset = compressed_idx % block_size
            return block_id * block_size + block_offset

        write_idx = 0
        slots_list = out_slots.tolist()
        for seq in range(batch_size):
            ctx_len = int(ctx_lens[seq])
            q_len = max(0, min(int(q_lens[seq]), ctx_len))
            prev_ctx_len = ctx_len - q_len
            prev_committed = prev_ctx_len // ratio
            committed = ctx_len // ratio
            new_committed = committed - prev_committed
            for i in range(new_committed):
                slots_list[write_idx] = slot_for_compressed_index(seq, prev_committed + i)
                write_idx += 1
        out_slots = torch.tensor(slots_list, dtype=torch.int32, device=raw_bt.device)

        out_bt = raw_bt
        if graph_slot_capacity > 0 and graph_block_table_capacity_cols > 0:
            cap = max(graph_block_table_capacity_cols, int(raw_bt.size(1)))
            out_bt = self._pad_block_table(raw_bt, batch_size, cap, -1)
        return out_bt, out_slots

    # -- SWA group (process_swa_group, cpp:466-572) ------------------------
    # Writes only the current forward's query tokens, ring-indexed by position.

    def _process_swa_group(
        self,
        raw_bt: torch.Tensor,
        block_size: int,
        ctx_lens: Sequence[int],
        q_lens: Sequence[int],
        batch_size: int,
        graph_slot_capacity: int,
        graph_block_table_capacity_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_total_tokens = 0
        for seq in range(batch_size):
            query_total_tokens += max(0, min(int(q_lens[seq]), int(ctx_lens[seq])))

        out_slot_rows = max(graph_slot_capacity, query_total_tokens) if graph_slot_capacity > 0 else query_total_tokens
        out_slots = torch.full((out_slot_rows,), -1, dtype=torch.int32, device=raw_bt.device)
        semantic_cols = int(raw_bt.size(1))

        def slot_for_position(seq: int, pos: int) -> int:
            if semantic_cols <= 0 or seq >= raw_bt.size(0):
                return -1
            block_idx = (pos // block_size) % semantic_cols
            block_id = int(raw_bt[seq, block_idx].item())
            if block_id < 0:
                return -1
            block_offset = pos % block_size
            return block_id * block_size + block_offset

        write_idx = 0
        slots_list = out_slots.tolist()
        for seq in range(batch_size):
            ctx_len = int(ctx_lens[seq])
            q_len = max(0, min(int(q_lens[seq]), ctx_len))
            if seq >= raw_bt.size(0):
                write_idx += q_len
                continue
            q_start = ctx_len - q_len
            for i in range(q_len):
                slots_list[write_idx] = slot_for_position(seq, q_start + i)
                write_idx += 1
        out_slots = torch.tensor(slots_list, dtype=torch.int32, device=raw_bt.device)

        # Rebuild the read-side block table: keep only the SWA window columns,
        # right-aligned.
        dst_lens = [(max(int(ctx_lens[s]), 0) + block_size - 1) // block_size for s in range(batch_size)]
        max_dst_len = max(max(dst_lens) if dst_lens else 0, semantic_cols)
        if graph_slot_capacity > 0 and graph_block_table_capacity_cols > 0:
            storage_cols = max(graph_block_table_capacity_cols, int(raw_bt.size(1)))
            max_dst_len = max(max_dst_len, storage_cols)
        new_bt = torch.full(
            (batch_size, max_dst_len),
            -1,
            dtype=torch.int32,
            device=raw_bt.device,
        )
        for s in range(batch_size):
            if s >= raw_bt.size(0):
                continue
            retained_cols = min(semantic_cols, dst_lens[s])
            start_col = dst_lens[s] - retained_cols
            for j in range(retained_cols):
                logical_col = start_col + j
                physical_col = logical_col % semantic_cols
                new_bt[s, logical_col] = raw_bt[s, physical_col]
        return new_bt, out_slots

    # -- SEQUENCE group (expand_blocks_to_slots, cpp:270-307) --------------

    def _expand_blocks_to_slots(
        self,
        raw_bt: torch.Tensor,
        gi: DSAGroupInfo,
        ctx_lens: Sequence[int],
        batch_size: int,
        total_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = gi.block_size
        slots = torch.full((total_tokens,), -1, dtype=torch.int32, device=raw_bt.device)
        max_blocks = int(raw_bt.size(1))
        start_idx = 0
        for seq in range(batch_size):
            token_len = int(ctx_lens[seq])
            slot_num = self._compute_slot_num(gi, token_len)
            if seq >= raw_bt.size(0):
                start_idx += token_len
                continue
            filled = 0
            for blk in range(max_blocks):
                if filled >= slot_num:
                    break
                block_id = int(raw_bt[seq, blk].item())
                if block_id < 0:
                    break
                for off in range(block_size):
                    if filled >= slot_num:
                        break
                    slots[start_idx + filled] = block_id * block_size + off
                    filled += 1
            start_idx += token_len
        # Replace -1 padding with 0 (C++ does torch::where(eq(-1), 0, raw)).
        slots = torch.where(slots.eq(-1), torch.zeros_like(slots), slots)
        return raw_bt, slots

    @staticmethod
    def _compute_slot_num(gi: DSAGroupInfo, token_len: int) -> int:
        if gi.cache_type == DSA_CACHE_TOKEN:
            return token_len // gi.ratio
        # SLIDING_WINDOW
        block_size = gi.block_size
        if token_len > block_size:
            return token_len % block_size + block_size
        remainder = token_len % block_size
        return block_size if (remainder == 0 and token_len > 0) else remainder

    @staticmethod
    def _pad_block_table(
        raw_bt: torch.Tensor,
        batch_size: int,
        capacity_cols: int,
        pad_value: int,
    ) -> torch.Tensor:
        cols = max(capacity_cols, int(raw_bt.size(1)))
        out = torch.full((batch_size, cols), pad_value, dtype=torch.int32, device=raw_bt.device)
        rows = min(batch_size, int(raw_bt.size(0)))
        copy_cols = min(int(raw_bt.size(1)), cols)
        out[:rows, :copy_cols] = raw_bt[:rows, :copy_cols]
        return out
