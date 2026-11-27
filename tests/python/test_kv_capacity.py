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

"""Host-only KV capacity tier selection contracts."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("max_pages", "limit", "expected"),
    [
        (1, 0, (1,)),
        (17, 0, (17,)),
        (32, 0, (32,)),
        (65, 0, (65, 64, 32)),
        (2048, 0, (2048, 1024, 512, 256, 128, 64, 32)),
        (2048, -1, ()),
        (2048, 1, (32,)),
        (2048, 4096, (32,)),
        (2048, 4097, (64, 32)),
        (2048, 12000, (128, 64, 32)),
        (65, 65536, (65, 64, 32)),
    ],
)
def test_block_kv_warmup_covers_binary_buckets_largest_first(
    max_pages: int, limit: int, expected: tuple[int, ...]
) -> None:
    from xllm.python.model_executor.runners.kv_capacity import non_causal_kv_capacity_buckets

    assert non_causal_kv_capacity_buckets(128, max_pages, 7, limit) == expected


@pytest.mark.parametrize(
    ("page_size", "max_pages", "width", "limit"),
    [(0, 64, 7, 0), (128, 0, 7, 0), (128, 64, 0, 0), (128, 64, 7, -2), (128, 64, 7, True)],
)
def test_block_kv_warmup_rejects_invalid_geometry(page_size: int, max_pages: int, width: int, limit: int) -> None:
    from xllm.python.model_executor.runners.kv_capacity import non_causal_kv_capacity_buckets

    with pytest.raises(ValueError):
        non_causal_kv_capacity_buckets(page_size, max_pages, width, limit)


@pytest.mark.parametrize(
    ("max_pages", "expected"),
    [
        (1, (1,)),
        (4, (4,)),
        (5, (5,)),
        (8, (4, 8)),
        (31, (4, 31)),
        (32, (4, 32)),
        (33, (4, 33)),
        (65, (4, 32, 65)),
        (2048, (4, 32, 256, 2048)),
        (2049, (4, 32, 256, 2049)),
    ],
)
def test_kv_capacity_tiers_retain_full_context(max_pages: int, expected: tuple[int, ...]) -> None:
    from xllm.python.model_executor.runners.kv_capacity import kv_capacity_buckets

    assert kv_capacity_buckets(max_pages) == expected


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [([1, 511], 4), ([512, 1], 4), ([513, 512], 32), ([4096, 1], 32), ([4097, 1], 65), ([8192, 8320], 65)],
)
def test_kv_capacity_bucket_covers_every_host_bound(bounds: list[int], expected: int) -> None:
    from xllm.python.model_executor.runners.kv_capacity import kv_capacity_bucket

    assert kv_capacity_bucket(bounds, 128, 65) == expected


@pytest.mark.parametrize("bounds", [None, [], [0], [-1], [513, -1], [1.5], [True], [8321]])
def test_kv_capacity_rejects_invalid_or_out_of_context_bounds(bounds: list[int] | None) -> None:
    from xllm.python.model_executor.runners.kv_capacity import kv_capacity_bucket

    with pytest.raises(ValueError):
        kv_capacity_bucket(bounds, 128, 65)


@pytest.mark.parametrize(("page_size", "max_pages"), [(0, 65), (-1, 65), (128, 0), (128, -1)])
def test_kv_capacity_rejects_invalid_page_capacity(page_size: int, max_pages: int) -> None:
    from xllm.python.model_executor.runners.kv_capacity import kv_capacity_bucket

    with pytest.raises(ValueError):
        kv_capacity_bucket([1], page_size, max_pages)
