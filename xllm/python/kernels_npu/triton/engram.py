# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Device hashing and quantized row lookup for DeepSeek-V4.1 Engram."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["tokens", "table_cols", "table_stride"])
def _prepare_unit_query_hash_kernel(
    Ids,
    Positions,
    Dead,
    Slots,
    Blocks,
    Map,
    Compressed,
    Starts,
    tokens,
    table_cols,
    table_stride,
    col_stride: tl.constexpr,
    vocab: tl.constexpr,
    block_size: tl.constexpr,
    ngram: tl.constexpr,
    tile: tl.constexpr,
):
    row = tl.program_id(0) * tile + tl.arange(0, tile)
    valid = row < tokens
    position = tl.load(Positions + row, valid, 0).to(tl.int64)
    token = tl.load(Ids + row, valid, 0)
    token = tl.minimum(tl.maximum(token, 0), vocab - 1)
    compact = tl.load(Map + token, valid, 0)
    dead = tl.load(Dead + row, valid, 0).to(tl.int1)
    tl.store(Compressed + row, tl.where(dead, -1, compact), valid)

    # Only ngram - 1 earlier tokens can enter this hash. Each link must
    # match both the logical position and the physical SWA slot of its predecessor.
    start = position
    linked = valid
    for distance in tl.static_range(1, ngram):
        # Ascend can lower a masked contiguous load to a vector transfer.
        # Keep its base inside the allocation even at the first token of a graph.
        previous_row = tl.minimum(tl.maximum(row - distance, 0), tokens - 1)
        table_row = tl.minimum(tl.maximum(row - distance + 1, 0), tokens - 1)
        linked = linked & (row >= distance)
        previous_pos = tl.load(Positions + previous_row).to(tl.int64)
        previous_slot = tl.load(Slots + previous_row).to(tl.int64)
        prior = tl.maximum(position - distance, 0)
        col = prior // block_size % table_cols
        block = tl.load(Blocks + table_row * table_stride + col * col_stride).to(tl.int64)
        linked = linked & (previous_pos == position - distance) & (previous_slot >= 0)
        linked = linked & (previous_slot == block * block_size + prior % block_size)
        start = tl.where(linked, previous_pos, start)
    tl.store(Starts + row, start, valid)


def prepare_unit_query_hash(
    ids: torch.Tensor,
    positions: torch.Tensor,
    dead: torch.Tensor,
    slots: torch.Tensor,
    blocks: torch.Tensor,
    block_size: int,
    token_map: torch.Tensor,
    ngram: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = ids.numel()
    compressed = torch.empty(tokens, dtype=torch.int32, device=ids.device)
    starts = torch.empty(tokens, dtype=positions.dtype, device=ids.device)
    _prepare_unit_query_hash_kernel[(triton.cdiv(tokens, 16),)](
        ids.contiguous(),
        positions.contiguous(),
        dead.contiguous(),
        slots.contiguous(),
        blocks,
        token_map,
        compressed,
        starts,
        tokens,
        blocks.size(1),
        blocks.stride(0),
        blocks.stride(1),
        token_map.numel(),
        block_size,
        ngram,
        16,
    )
    return compressed, starts


@triton.jit(do_not_specialize=["tokens"])
def _write_unit_query_history_kernel(Cache, Compressed, Slots, tokens, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    slot = tl.load(Slots + row, row < tokens, -1).to(tl.int64)
    compact = tl.load(Compressed + row, row < tokens, 0)
    tl.store(Cache + slot, compact, (row < tokens) & (slot >= 0))


def write_unit_query_history(cache: torch.Tensor, compressed: torch.Tensor, slots: torch.Tensor) -> None:
    # Hashing must finish reading the old ring before any program writes current tokens.
    tokens = compressed.numel()
    _write_unit_query_history_kernel[(triton.cdiv(tokens, 64),)](cache, compressed, slots.contiguous(), tokens, 64)


@triton.jit(do_not_specialize=["tokens", "table_rows", "table_cols", "table_row_stride", "table_col_stride"])
def _hash_kernel(
    Comp,
    Positions,
    Requests,
    Starts,
    History,
    Blocks,
    Replay,
    ReplayRows,
    Multipliers,
    Primes,
    Offsets,
    Out,
    tokens,
    LAYERS: tl.constexpr,
    NGRAM: tl.constexpr,
    HEADS: tl.constexpr,
    PAD: tl.constexpr,
    table_rows,
    table_cols,
    table_row_stride,
    table_col_stride,
    BLOCK_SIZE: tl.constexpr,
    SLOTS: tl.constexpr,
    REPLAY_WIDTH: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
) -> None:
    row = tl.program_id(0) * TOKEN_TILE + tl.arange(0, TOKEN_TILE)
    layer = tl.program_id(1)
    valid = row < tokens
    position = tl.load(Positions + row, valid, 0)
    request = tl.load(Requests + row, valid, 0)
    start = tl.load(Starts + row, valid, 0)
    rolling = tl.full((TOKEN_TILE,), 0, tl.int64)
    blocked = tl.full((TOKEN_TILE,), False, tl.int1)
    head = tl.arange(0, HEAD_TILE)
    head_cols: tl.constexpr = (NGRAM - 1) * HEADS
    if REPLAY_WIDTH:
        replay_row = tl.load(ReplayRows + request, valid, 0) > 0
    for shift in tl.static_range(NGRAM):
        lookback = position - shift
        in_batch = lookback >= start
        batch_index = tl.maximum(row - shift, 0)
        source = tl.load(Comp + batch_index, valid & in_batch, 0)
        if SLOTS:
            previous = tl.maximum(lookback, 0)
            column = previous // BLOCK_SIZE % table_cols
            table_row = tl.minimum(request, table_rows - 1)
            block = tl.load(Blocks + table_row * table_row_stride + column * table_col_stride, valid & ~in_batch, 0).to(
                tl.int64
            )
            address = tl.minimum(tl.maximum(block * BLOCK_SIZE + previous % BLOCK_SIZE, 0), SLOTS - 1)
            cached = tl.load(History + address, valid & ~in_batch, 0)
            source = tl.where(in_batch, source, cached)
        if REPLAY_WIDTH:
            column = tl.minimum(tl.maximum(lookback - start + REPLAY_WIDTH, 0), REPLAY_WIDTH - 1)
            override = valid & replay_row & ~in_batch
            replay_id = tl.load(Replay + request * REPLAY_WIDTH + column, override, 0)
            source = tl.where(override, replay_id, source)
        blocked = blocked | (lookback < 0) | (source == -1)
        value = tl.where(blocked, PAD, source).to(tl.int64)
        weight = tl.load(Multipliers + layer * NGRAM + shift)
        rolling = rolling ^ (value * weight)
        if shift > 0:
            columns = (shift - 1) * HEADS + head
            param = layer * head_cols + columns
            prime = tl.load(Primes + param, head < HEADS, 1)
            offset = tl.load(Offsets + param, head < HEADS, 0)
            remainder = rolling[:, None] % prime[None, :]
            remainder = tl.where(remainder < 0, remainder + prime[None, :], remainder)
            result = remainder + offset[None, :]
            output = (row.to(tl.int64) * LAYERS + layer) * head_cols
            tl.store(Out + output[:, None] + columns[None, :], result, valid[:, None] & (head[None, :] < HEADS))


def hash_rows(
    compressed_ids: torch.Tensor,
    positions: torch.Tensor,
    requests: torch.Tensor,
    starts: torch.Tensor,
    history: torch.Tensor | None,
    block_table: torch.Tensor | None,
    block_size: int,
    replay: torch.Tensor | None,
    replay_rows: torch.Tensor | None,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    tokens = compressed_ids.numel()
    layers, ngrams, heads = primes.shape
    output = torch.empty((tokens, layers, ngrams * heads), dtype=torch.int32, device=positions.device)
    use_history = history is not None and block_table is not None
    _hash_kernel[(triton.cdiv(tokens, 16), layers)](
        compressed_ids,
        positions,
        requests,
        starts,
        history,
        block_table,
        replay,
        replay_rows,
        multipliers,
        primes,
        offsets,
        output,
        tokens,
        layers,
        ngrams + 1,
        heads,
        pad_id,
        block_table.shape[0] if use_history else 0,
        block_table.shape[1] if use_history else 0,
        block_table.stride(0) if use_history else 0,
        block_table.stride(1) if use_history else 0,
        block_size,
        history.numel() if use_history else 0,
        replay.size(1) if replay is not None else 0,
        16,
        triton.next_power_of_2(heads),
    )
    return output


@triton.jit
def _lookup_kernel(
    Indices,
    Weight,
    Scale,
    Out,
    INDEX_STRIDE_TOKEN: tl.constexpr,
    INDEX_STRIDE_HEAD: tl.constexpr,
    HEAD_START: tl.constexpr,
    HEADS: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    WIDTH: tl.constexpr,
    COLUMN_TILE: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    HOST: tl.constexpr,
    E8M0: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    token = row // LOCAL_HEADS
    head = HEAD_START + row % LOCAL_HEADS
    index = tl.load(Indices + token * INDEX_STRIDE_TOKEN + head * INDEX_STRIDE_HEAD, head < HEADS, -1)
    owned = (index >= START) & (index < END)
    local = tl.where(owned, index.to(tl.int64) - START, 0)
    if HOST:
        weight = tl.load(Weight).to(tl.pointer_type(tl.int8))
        if E8M0:
            scale = tl.load(Scale).to(tl.pointer_type(tl.uint8))
        else:
            scale = tl.load(Scale).to(tl.pointer_type(tl.float32))
    else:
        weight = Weight
        scale = Scale
    col = tl.arange(0, COLUMN_TILE)
    valid_col = col < WIDTH
    value = tl.load(weight + local * WIDTH + col, owned & valid_col, 0).to(tl.float32)
    group = col // QUANT_BLOCK
    scale_value = tl.load(scale + local * (WIDTH // QUANT_BLOCK) + group, owned & valid_col, 0)
    if E8M0:
        bits = scale_value.to(tl.int32) << 23
        bits = tl.where(scale_value == 0, 0x00400000, bits)
        scale_value = bits.to(tl.float32, bitcast=True)
    tl.store(Out + row * WIDTH + col, (value * scale_value).to(tl.bfloat16), valid_col)


def lookup_int8(
    indices: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    head_start: int,
    local_heads: int,
    vocab_start: int,
    vocab_end: int,
    width: int,
    quant_block: int,
    *,
    mapped_host: bool = False,
    e8m0: bool,
) -> torch.Tensor:
    output = torch.empty((indices.size(0), local_heads, width), dtype=torch.bfloat16, device=indices.device)
    if output.numel():
        _lookup_kernel[(indices.size(0) * local_heads,)](
            indices,
            weight,
            scale,
            output,
            indices.stride(0),
            indices.stride(1),
            head_start,
            indices.size(1),
            local_heads,
            vocab_start,
            vocab_end,
            width,
            triton.next_power_of_2(width),
            quant_block,
            mapped_host,
            e8m0,
        )
    return output


@triton.jit
def _engram_gate_kernel(
    residual,
    gate_hidden,
    projected,
    query_weight,
    key_weight,
    mask,
    output,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    col = tl.arange(0, TILE)
    keep = col < DIM
    offset = row * DIM + col
    base = token * (HEADS + 1) * DIM
    hidden = tl.load(gate_hidden + offset, keep, 0).to(tl.float32)
    key = tl.load(projected + base + head * DIM + col, keep, 0).to(tl.float32)
    qw = tl.load(query_weight + head * DIM + col, keep, 0).to(tl.float32)
    kw = tl.load(key_weight + head * DIM + col, keep, 0).to(tl.float32)
    dot = tl.sum(((hidden * qw) * kw) * key, axis=0)
    hidden_rms = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / DIM + EPS)
    key_rms = tl.rsqrt(tl.sum(key * key, axis=0) / DIM + EPS)
    score = ((dot * hidden_rms) * key_rms) * (DIM**-0.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(score), CLAMP))
    gate = tl.sigmoid(tl.where(score < 0, -magnitude, magnitude))
    gate = tl.where(tl.load(mask + token), gate, 0.0)
    value = tl.load(projected + base + HEADS * DIM + col, keep, 0).to(tl.float32)
    previous = tl.load(residual + offset, keep, 0).to(tl.float32)
    tl.store(output + offset, previous + gate * value, keep)


def engram_gate(
    residual: torch.Tensor,
    gate_hidden: torch.Tensor,
    projected: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    clamp: float,
) -> torch.Tensor:
    tokens, heads, dim = residual.shape
    output = torch.empty_like(residual)
    _engram_gate_kernel[(tokens * heads,)](
        residual,
        gate_hidden,
        projected,
        query_weight,
        key_weight,
        mask,
        output,
        eps,
        clamp,
        heads,
        dim,
        triton.next_power_of_2(dim),
        enable_fp_fusion=False,
    )
    return output
