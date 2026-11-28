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

"""Sparse page-capacity tiers for device-length attention graphs."""

from __future__ import annotations

from collections.abc import Sequence

# Match the C++ non-causal decode profile context before query lookahead.
NON_CAUSAL_WARMUP_CONTEXT_PAGES = 16


def non_causal_kv_capacity_buckets(
    page_size: int, max_pages: int, query_width: int, max_kv_length: int = 0
) -> tuple[int, ...]:
    """Enumerate the existing non-causal decode graph capacities, largest first.

    A positive length selects the bucket covering that bound, not a new
    nonstandard bucket. Zero covers the whole page table; minus one disables
    proactive capture without changing runtime admission or model capacity.
    """
    if any(type(value) is not int or value <= 0 for value in (page_size, max_pages, query_width)):
        raise ValueError("Non-causal decode graph page and query capacities must be positive integers")
    if type(max_kv_length) is not int or max_kv_length < -1:
        raise ValueError("Non-causal decode graph warmup KV limit must be -1, zero or a positive integer")
    if max_kv_length == -1:
        return ()
    minimum_pages = NON_CAUSAL_WARMUP_CONTEXT_PAGES + (query_width + page_size - 1) // page_size
    pages = min(1 << (minimum_pages - 1).bit_length(), max_pages)
    requested_pages = max_pages if max_kv_length == 0 else (max_kv_length + page_size - 1) // page_size
    maximum = min(1 << (max(pages, requested_pages) - 1).bit_length(), max_pages)
    buckets = [pages]
    while pages < maximum:
        pages = min(pages * 2, maximum)
        buckets.append(pages)
    return tuple(reversed(buckets))


def kv_capacity_buckets(max_pages: int) -> tuple[int, ...]:
    """Retain full context with four-page, eightfold-growing graph tiers.

    A tier within a factor of two of the final capacity is folded into that
    final tier, including the ordinary runner's extra lookahead page.
    """
    if type(max_pages) is not int or max_pages <= 0:
        raise ValueError("Graph block-table capacity must be positive")
    pages = 4
    buckets = []
    while pages * 2 <= max_pages:
        buckets.append(pages)
        pages *= 8
    return (*buckets, max_pages)


def kv_capacity_bucket(kv_bounds: Sequence[int] | None, page_size: int, max_pages: int) -> int:
    """Select by scheduler-provided Host upper bounds without Device reads."""
    if type(page_size) is not int or page_size <= 0:
        raise ValueError("Graph page size must be positive")
    buckets = kv_capacity_buckets(max_pages)
    if not kv_bounds or any(type(bound) is not int or bound <= 0 for bound in kv_bounds):
        raise ValueError("KV capacity selection requires positive Host KV upper bounds")
    required_pages = (max(kv_bounds) + page_size - 1) // page_size
    if required_pages > max_pages:
        raise ValueError("Host KV upper bound exceeds the full block-table capacity")
    return next(pages for pages in buckets if pages >= required_pages)
