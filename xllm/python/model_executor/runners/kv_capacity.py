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
