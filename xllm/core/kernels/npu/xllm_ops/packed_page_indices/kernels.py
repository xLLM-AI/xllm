# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Masked device kernels; counts and storage addresses stay dynamic."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2, 3])
def _translate_pages(
    x: tl.tensor,
    mapping: tl.tensor,
    cu: tl.tensor,
    output: tl.tensor,
    columns: tl.constexpr,
    layers: tl.constexpr,
    blocks: tl.constexpr,
    total: tl.constexpr,
    column_tile: tl.constexpr,
    layer_tile: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    layer = tl.program_id(1) * layer_tile + tl.arange(0, layer_tile)
    col = tl.program_id(2) * column_tile + tl.arange(0, column_tile)
    start = tl.load(cu + row)
    end = tl.load(cu + row + 1)
    active = (col < columns) & (col < end - start)
    page = tl.load(x + row * columns + col, active, other=-1)
    safe_page = tl.maximum(page, 0)
    physical = tl.load(
        mapping + layer[:, None] * blocks + safe_page[None, :],
        (layer[:, None] < layers) & active[None, :] & (page[None, :] >= 0),
        other=-1,
    )
    tl.store(
        output + layer[:, None] * total + start + col[None, :], physical, (layer[:, None] < layers) & active[None, :]
    )


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2, 3, 4])
def _pack_pages(
    x: tl.tensor,
    mapping: tl.tensor,
    cu: tl.tensor,
    pages: tl.tensor,
    padded: tl.tensor,
    rows: tl.constexpr,
    columns: tl.constexpr,
    layers: tl.constexpr,
    blocks: tl.constexpr,
    column_tile: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    tiles: tl.constexpr = (columns + column_tile - 1) // column_tile
    row = program // tiles
    col = program % tiles * column_tile + tl.arange(0, column_tile)
    start = tl.load(cu + row, row < rows, other=0)
    end = tl.load(cu + row + 1, row < rows, other=0)
    active = (row < rows) & (col < columns) & (col < end - start)
    page = tl.load(x + row * columns + col, active, other=-1)
    tl.store(pages + start + col, tl.where(page < 0, blocks, page), active)
    stride: tl.constexpr = ((blocks + 32) // 32) * 32
    mapping_tile: tl.constexpr = 4096
    mapping_tiles: tl.constexpr = (stride + mapping_tile - 1) // mapping_tile
    layer = program // mapping_tiles
    mapping_col = program % mapping_tiles * mapping_tile + tl.arange(0, mapping_tile)
    if layers * stride > 2147483647:
        index = layer.to(tl.int64) * stride + mapping_col
    else:
        index = layer * stride + mapping_col
    value = tl.load(mapping + layer * blocks + mapping_col, (layer < layers) & (mapping_col < blocks), other=-1)
    tl.store(padded + index, value, (layer < layers) & (mapping_col < stride))
