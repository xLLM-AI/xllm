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


import triton
import triton.language as tl


@triton.jit
def _combine_head_sums(
    sum_a: tl.tensor,
    error_a: tl.tensor,
    sum_b: tl.tensor,
    error_b: tl.tensor,
) -> tuple[tl.tensor, tl.tensor]:
    total = sum_a + sum_b
    virtual_b = total - sum_a
    error = (sum_a - (total - virtual_b)) + (sum_b - virtual_b)
    # An infinite partial sum has no finite rounding error to recover.
    error = tl.where(tl.abs(total) == float("inf"), 0.0, (error_a + error_b) + error)
    return total, error


@triton.jit
def _sum_weighted_heads(values: tl.tensor, axis: tl.constexpr) -> tl.tensor:
    # Preserve low-order terms when positive and negative heads cancel.
    total, error = tl.reduce((values, tl.full(values.shape, 0, tl.float32)), axis, _combine_head_sums)
    return total + error


@triton.jit
def _globalize_slots(
    slots: tl.tensor,
    PAGE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    SHARDS: tl.constexpr,
    RANK: tl.constexpr,
) -> tl.tensor:
    source = slots.to(tl.int32) if PAGE * SHARDS <= 2147483647 else slots.to(tl.int64)
    if PAGE > 2147483647:
        page = tl.full(slots.shape, 0, tl.int32)
    elif PAGE & (PAGE - 1) == 0:
        page = source >> PAGE_SHIFT
    else:
        page = source // PAGE
    return (source + page * (PAGE * (SHARDS - 1)) + RANK * PAGE).to(tl.int32)


@triton.jit
def _score_owner_loop(
    Q: tl.tensor,
    W: tl.tensor,
    Cache: tl.tensor,
    Slots: tl.tensor,
    Counts: tl.tensor,
    Scores: tl.tensor,
    Global: tl.tensor,
    rows: tl.tensor,
    ROW_OWNED: tl.constexpr,
    SMALL_ROWS: tl.constexpr,
    PRELOAD_COUNTS: tl.constexpr,
    PREFETCH_ROWS: tl.constexpr,
    NARROW: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    CS0: tl.constexpr,
    SHARDS: tl.constexpr,
    RANK: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CORE_COUNT: tl.constexpr,
    SLOT_CAP: tl.constexpr,
    resident_counts: tl.tensor,
    FULL_COUNT: tl.constexpr,
    resident_slots: tl.tensor,
    PROGRAM_VALID: tl.constexpr,
) -> None:
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    n = tl.arange(0, BLOCK_N)
    tiles = tl.cdiv(K, BLOCK_N)
    PRELOAD_SLOTS: tl.constexpr = ROW_OWNED and SMALL_ROWS and K <= 4096
    # W/cache axis products must widen before multiplication, not after
    # adding the row address. Their element offsets can exceed INT32 too.
    head_offsets = h.to(tl.int32) if NARROW else h.to(tl.int64)
    dim_offsets = d.to(tl.int32) if NARROW else d.to(tl.int64)
    EARLY_GLOBAL: tl.constexpr = not SMALL_ROWS and S > 4096
    if PREFETCH_ROWS:
        first_row = tl.program_id(0).to(tl.int32) if NARROW else tl.program_id(0).to(tl.int64)
        first_query_ptr = tl.make_block_ptr(Q + first_row * QS0, (32, 128), (QS1, QS2), (0, 0), (32, 128), (1, 0))
        prefetch_query = tl.load(first_query_ptr)
        prefetch_weights = tl.load(W + first_row * WS0 + head_offsets * WS1, cache_modifier=".ca")
    # Widen before owner arithmetic, including the row/tile product.
    rows = rows.to(tl.int64)
    owners = rows if ROW_OWNED else rows * tiles
    BATCH_ROWS: tl.constexpr = 4 if ROW_OWNED and not SMALL_ROWS else 1
    # Runtime INT32 cdiv would overflow near INT_MAX before division.
    owner_span = owners - tl.program_id(0)
    owner_jobs = owner_span // tl.num_programs(0) + (owner_span % tl.num_programs(0) != 0).to(tl.int32)
    owner_groups = owner_jobs // BATCH_ROWS + (owner_jobs % BATCH_ROWS != 0).to(tl.int32)
    for owner_group in range(owner_groups):
        if BATCH_ROWS == 4:
            batch_index = owner_group * 4 + tl.arange(0, 4)
            batch_index_wide = batch_index.to(tl.int32) if NARROW else batch_index.to(tl.int64)
            batch_row_id = tl.minimum(tl.program_id(0) + batch_index_wide * tl.num_programs(0), rows - 1)
            batch_row = batch_row_id.to(tl.int32) if NARROW else batch_row_id.to(tl.int64)
            if rows % (CORE_COUNT * 4) == 0:
                first_index = owner_group.to(tl.int32) if NARROW else owner_group.to(tl.int64)
                first_row = tl.program_id(0) + first_index * 4 * CORE_COUNT
                q_block = tl.make_block_ptr(
                    Q + first_row * QS0,
                    (4, 32, 128),
                    (CORE_COUNT * QS0, QS1, QS2),
                    (0, 0, 0),
                    (4, 32, 128),
                    (2, 1, 0),
                )
                w_block = tl.make_block_ptr(
                    W + first_row * WS0,
                    (4, 32),
                    (CORE_COUNT * WS0, WS1),
                    (0, 0),
                    (4, 32),
                    (1, 0),
                )
                query_batch = tl.load(q_block).to(tl.float32)
                weight_batch = tl.load(w_block, cache_modifier=".ca").to(tl.float32)
            else:
                query_batch = tl.load(
                    Q
                    + batch_row[:, None, None] * QS0
                    + head_offsets[None, :, None] * QS1
                    + dim_offsets[None, None, :] * QS2
                ).to(tl.float32)
                weight_batch = tl.load(
                    W + batch_row[:, None] * WS0 + head_offsets[None, :] * WS1,
                    cache_modifier=".ca",
                ).to(tl.float32)
            batch_jobs = tl.minimum(4, owner_jobs - owner_group * 4)
        else:
            batch_jobs = 1
        for step in range(batch_jobs):
            owner = tl.program_id(0) + (owner_group * BATCH_ROWS + step) * tl.num_programs(0)
            row_id = owner if ROW_OWNED else owner // tiles
            row = row_id.to(tl.int32) if NARROW else row_id.to(tl.int64)
            if BATCH_ROWS == 4:
                query = tl.reshape(query_batch[step : step + 1, :, :], (32, 128))
                weights = tl.reshape(weight_batch[step : step + 1, :], (32,))
            elif PREFETCH_ROWS:
                query = prefetch_query.to(tl.float32)
                weights = prefetch_weights.to(tl.float32)
                next_row_id = row_id + tl.num_programs(0)
                if next_row_id < rows:
                    next_row = next_row_id.to(tl.int32) if NARROW else next_row_id.to(tl.int64)
                    next_query_ptr = tl.make_block_ptr(
                        Q + next_row * QS0,
                        (32, 128),
                        (QS1, QS2),
                        (0, 0),
                        (32, 128),
                        (1, 0),
                    )
                    prefetch_query = tl.load(next_query_ptr)
                    prefetch_weights = tl.load(W + next_row * WS0 + head_offsets * WS1, cache_modifier=".ca")
            else:
                query_ptr = tl.make_block_ptr(Q + row * QS0, (32, 128), (QS1, QS2), (0, 0), (32, 128), (1, 0))
                query = tl.load(query_ptr).to(tl.float32)
                weights = tl.load(W + row * WS0 + head_offsets * WS1, cache_modifier=".ca").to(tl.float32)
            if FULL_COUNT:
                count = tl.full((), K, tl.int32)
            elif PRELOAD_COUNTS:
                count_index = row_id // CORE_COUNT
                count = tl.sum(resident_counts[count_index : count_index + 1], 0)
            else:
                count = tl.load(Counts + row * CS0)
            if PRELOAD_SLOTS:
                SLOT_BLOCK: tl.constexpr = SLOT_CAP if SLOT_CAP > BLOCK_N else BLOCK_N
                slot_columns = tl.arange(0, SLOT_BLOCK)
                if PROGRAM_VALID:
                    owned_index = row_id // CORE_COUNT
                    row_slots = resident_slots[owned_index : owned_index + 1, :].reshape((SLOT_BLOCK,))
                    all_valid = True
                else:
                    row_slots = tl.load(
                        Slots + row * SS0 + (slot_columns.to(tl.int32) if NARROW else slot_columns.to(tl.int64)) * SS1,
                        slot_columns < K,
                        other=-1,
                    )
                    all_valid = (count >= K) & (tl.min(tl.where(slot_columns < K, row_slots, 0), 0) >= 0)
                row_valid = (slot_columns < K) & (slot_columns < count) & (row_slots >= 0)
                row_global = _globalize_slots(row_slots, PAGE, PAGE_SHIFT, SHARDS, RANK)
                if all_valid:
                    tl.store(Global + row * K + slot_columns, row_global, slot_columns < K)
                else:
                    tl.store(
                        Global + row * K + slot_columns,
                        tl.where(row_valid, row_global, -1),
                        slot_columns < K,
                    )
            else:
                all_valid = False
            active_columns = tl.minimum(tl.maximum(count, 0), K)
            # Avoid the signed INT32 addition in (active_columns + BLOCK_N - 1).
            active_tiles = (
                (active_columns // BLOCK_N + (active_columns % BLOCK_N != 0).to(tl.int32)) if ROW_OWNED else 1
            )
            if PRELOAD_SLOTS and all_valid:
                # Valid resident slots need no per-slot score mask.
                for local_tile in range(tiles):
                    col = local_tile * BLOCK_N + n
                    slot_begin = local_tile * BLOCK_N
                    slot = row_slots[slot_begin : slot_begin + BLOCK_N]
                    original = slot.to(tl.int32) if NARROW else slot.to(tl.int64)
                    safe = tl.minimum(tl.maximum(original, 0), S - 1)
                    keys = tl.load(Cache + safe[:, None] * KS0 + dim_offsets[None, :] * KS1).to(tl.float32)
                    per_head = tl.dot(keys, tl.trans(query), allow_tf32=False, out_dtype=tl.float32)
                    score = _sum_weighted_heads(tl.maximum(per_head, 0) * weights[None, :], 1)
                    tl.store(Scores + row * K + col, score, col < K)
            else:
                for local_tile in range(active_tiles):
                    column_tile = local_tile if ROW_OWNED else owner % tiles
                    col = column_tile * BLOCK_N + n
                    in_bounds = tl.full((BLOCK_N,), True, tl.int1) if K % BLOCK_N == 0 else col < K
                    if PRELOAD_SLOTS:
                        slot_begin = column_tile * BLOCK_N
                        slot = row_slots[slot_begin : slot_begin + BLOCK_N]
                    else:
                        slot = tl.load(
                            Slots + row * SS0 + (col.to(tl.int32) if NARROW else col.to(tl.int64)) * SS1,
                            in_bounds,
                            other=-1,
                        )
                    valid = in_bounds & (col < count) & (slot >= 0)
                    original = slot
                    original = original.to(tl.int32) if NARROW else original.to(tl.int64)
                    if not PRELOAD_SLOTS and EARLY_GLOBAL:
                        global_slot = _globalize_slots(original, PAGE, PAGE_SHIFT, SHARDS, RANK)
                        tl.store(
                            Global + row * K + col,
                            tl.where(valid, global_slot, -1),
                            in_bounds,
                        )
                    if SMALL_ROWS:
                        safe = tl.minimum(tl.maximum(original, 0), S - 1)
                    else:
                        safe = tl.minimum(tl.where(valid, original, col), S - 1)
                    keys = tl.load(
                        Cache + safe[:, None] * KS0 + dim_offsets[None, :] * KS1,
                        cache_modifier=".ca" if not SMALL_ROWS else "",
                    ).to(tl.float32)
                    if SMALL_ROWS:
                        per_head = tl.dot(
                            keys,
                            tl.trans(query),
                            allow_tf32=False,
                            out_dtype=tl.float32,
                        )
                        score = _sum_weighted_heads(tl.maximum(per_head, 0) * weights[None, :], 1)
                    else:
                        per_head = tl.dot(
                            query,
                            tl.trans(keys),
                            allow_tf32=False,
                            out_dtype=tl.float32,
                        )
                        score = _sum_weighted_heads(tl.maximum(per_head, 0) * weights[:, None], 0)
                    if not PRELOAD_SLOTS and not EARLY_GLOBAL:
                        global_slot = _globalize_slots(original, PAGE, PAGE_SHIFT, SHARDS, RANK)
                    out = row * K + col
                    tl.store(Scores + out, tl.where(valid, score, -float("inf")), in_bounds)
                    if not PRELOAD_SLOTS and not EARLY_GLOBAL:
                        tl.store(Global + out, tl.where(valid, global_slot, -1), in_bounds)
            if ROW_OWNED:
                if not SMALL_ROWS and K <= 4096:
                    if active_tiles < tiles:
                        tail_col = tl.arange(0, SLOT_CAP)
                        tail_mask = (tail_col >= active_tiles * BLOCK_N) & (tail_col < K)
                        tail_out = row * K + tail_col
                        tl.store(Scores + tail_out, -float("inf"), tail_mask)
                        tl.store(Global + tail_out, -1, tail_mask)
                else:
                    for tail_tile in range(active_tiles, tiles):
                        tail_col = tail_tile * BLOCK_N + n
                        tail_out = row * K + tail_col
                        tl.store(Scores + tail_out, -float("inf"), tail_col < K)
                        if not PRELOAD_SLOTS:
                            tl.store(Global + tail_out, -1, tail_col < K)


@triton.jit(do_not_specialize=["rows"])
def tmo_dcp_score_candidates_kernel(
    Q: tl.tensor,
    W: tl.tensor,
    Cache: tl.tensor,
    Slots: tl.tensor,
    Counts: tl.tensor,
    Scores: tl.tensor,
    Global: tl.tensor,
    rows: tl.tensor,
    ROW_CAP: tl.constexpr,
    ROW_OWNED: tl.constexpr,
    SMALL_ROWS: tl.constexpr,
    PRELOAD_COUNTS: tl.constexpr,
    PREFETCH_ROWS: tl.constexpr,
    PROGRAM_PROOF: tl.constexpr,
    NARROW: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    CS0: tl.constexpr,
    SHARDS: tl.constexpr,
    RANK: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CORE_COUNT: tl.constexpr,
    SLOT_CAP: tl.constexpr,
) -> None:
    if PRELOAD_COUNTS:
        OWN_COUNT_CAP: tl.constexpr = triton.next_power_of_2(triton.cdiv(ROW_CAP, CORE_COUNT))
        # ROW_OWNED guarantees the grid has CORE_COUNT programs. Keep only
        # this program's cyclic rows, while preserving the full address proof.
        count_rows = tl.program_id(0) + tl.arange(0, OWN_COUNT_CAP) * CORE_COUNT
        safe_count_rows = tl.minimum(count_rows, rows - 1)
        count_offsets = safe_count_rows.to(tl.int32) if NARROW else safe_count_rows.to(tl.int64)
        resident_counts = tl.load(Counts + count_offsets * CS0)
    else:
        resident_counts = tl.full((1,), 0, tl.int32)
    unused_slots = tl.full((1, 1), 0, tl.int32)
    if PROGRAM_PROOF:
        OWNER_CAP: tl.constexpr = triton.next_power_of_2(triton.cdiv(ROW_CAP, CORE_COUNT))
        PROOF_SLOT_CAP: tl.constexpr = SLOT_CAP if SLOT_CAP > BLOCK_N else BLOCK_N
        owner_rows = tl.program_id(0) + tl.arange(0, OWNER_CAP) * CORE_COUNT
        safe_rows = tl.minimum(owner_rows, rows - 1)
        owner_offsets = safe_rows.to(tl.int32) if NARROW else safe_rows.to(tl.int64)
        proof_columns = tl.arange(0, PROOF_SLOT_CAP)
        column_offsets = proof_columns.to(tl.int32) if NARROW else proof_columns.to(tl.int64)
        owner_counts = tl.load(Counts + owner_offsets * CS0, owner_rows < rows, other=0)
        owner_slots = tl.load(
            Slots + owner_offsets[:, None] * SS0 + column_offsets[None, :] * SS1,
            (owner_rows[:, None] < rows) & (proof_columns[None, :] < K),
            other=0,
        )
        # Both padding dimensions are neutral. Check original signs before
        # score-address clamping, and inspect every row owned by this program.
        counts_valid = tl.min(tl.where(owner_rows < rows, owner_counts >= K, True).to(tl.int32), 0) != 0
        slots_valid = tl.min(owner_slots.reshape((OWNER_CAP * PROOF_SLOT_CAP,)), 0) >= 0
        program_all_valid = counts_valid & slots_valid
        if program_all_valid:
            _score_owner_loop(
                Q,
                W,
                Cache,
                Slots,
                Counts,
                Scores,
                Global,
                rows,
                ROW_OWNED,
                SMALL_ROWS,
                PRELOAD_COUNTS,
                PREFETCH_ROWS,
                NARROW,
                K,
                S,
                QS0,
                QS1,
                QS2,
                WS0,
                WS1,
                KS0,
                KS1,
                SS0,
                SS1,
                CS0,
                SHARDS,
                RANK,
                PAGE,
                PAGE_SHIFT,
                BLOCK_N,
                CORE_COUNT,
                SLOT_CAP,
                resident_counts,
                True,
                owner_slots,
                True,
            )
        else:
            _score_owner_loop(
                Q,
                W,
                Cache,
                Slots,
                Counts,
                Scores,
                Global,
                rows,
                ROW_OWNED,
                SMALL_ROWS,
                PRELOAD_COUNTS,
                PREFETCH_ROWS,
                NARROW,
                K,
                S,
                QS0,
                QS1,
                QS2,
                WS0,
                WS1,
                KS0,
                KS1,
                SS0,
                SS1,
                CS0,
                SHARDS,
                RANK,
                PAGE,
                PAGE_SHIFT,
                BLOCK_N,
                CORE_COUNT,
                SLOT_CAP,
                resident_counts,
                False,
                unused_slots,
                False,
            )
    elif PRELOAD_COUNTS and K <= 2147483647 and S > 4096:
        # A Count-only proof retains each slot's original validity check.
        proof_counts = tl.where(count_rows < rows, resident_counts, K)
        all_full = tl.min(proof_counts, 0) >= K
        if all_full:
            _score_owner_loop(
                Q,
                W,
                Cache,
                Slots,
                Counts,
                Scores,
                Global,
                rows,
                ROW_OWNED,
                SMALL_ROWS,
                PRELOAD_COUNTS,
                PREFETCH_ROWS,
                NARROW,
                K,
                S,
                QS0,
                QS1,
                QS2,
                WS0,
                WS1,
                KS0,
                KS1,
                SS0,
                SS1,
                CS0,
                SHARDS,
                RANK,
                PAGE,
                PAGE_SHIFT,
                BLOCK_N,
                CORE_COUNT,
                SLOT_CAP,
                resident_counts,
                True,
                unused_slots,
                False,
            )
        else:
            _score_owner_loop(
                Q,
                W,
                Cache,
                Slots,
                Counts,
                Scores,
                Global,
                rows,
                ROW_OWNED,
                SMALL_ROWS,
                PRELOAD_COUNTS,
                PREFETCH_ROWS,
                NARROW,
                K,
                S,
                QS0,
                QS1,
                QS2,
                WS0,
                WS1,
                KS0,
                KS1,
                SS0,
                SS1,
                CS0,
                SHARDS,
                RANK,
                PAGE,
                PAGE_SHIFT,
                BLOCK_N,
                CORE_COUNT,
                SLOT_CAP,
                resident_counts,
                False,
                unused_slots,
                False,
            )
    else:
        _score_owner_loop(
            Q,
            W,
            Cache,
            Slots,
            Counts,
            Scores,
            Global,
            rows,
            ROW_OWNED,
            SMALL_ROWS,
            PRELOAD_COUNTS,
            PREFETCH_ROWS,
            NARROW,
            K,
            S,
            QS0,
            QS1,
            QS2,
            WS0,
            WS1,
            KS0,
            KS1,
            SS0,
            SS1,
            CS0,
            SHARDS,
            RANK,
            PAGE,
            PAGE_SHIFT,
            BLOCK_N,
            CORE_COUNT,
            SLOT_CAP,
            resident_counts,
            False,
            unused_slots,
            False,
        )
