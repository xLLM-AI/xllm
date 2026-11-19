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

"""CPU-only address proof/check; explicitly not a substitute for NPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reference"))
import golden

TILE = int(re.search(r"kTileElements\s*=\s*(\d+)", (ROOT / "common/t5_plan.h").read_text()).group(1))


def _fragments(b: Any, h: Any, cache_length: Any, d: Any, output: Any) -> Any:
    hd = h * d
    qsize = b * hd
    size = qsize if output == 0 else qsize * (cache_length + 1)
    row = (cache_length + 1) * d
    old = cache_length * d
    for start in range(0, size, TILE):
        end = min(start + TILE, size)
        pos = start
        while pos < end:
            if output == 0 or cache_length == 0:
                batch, within = divmod(pos, hd)
                n = min(end - pos, hd - within)
                source = 0
                src = batch * 3 * hd + output * hd + within
            else:
                head, within = divmod(pos, row)
                if within < old:
                    n = min(end - pos, old - within)
                    source = output
                    src = head * old + within
                else:
                    n = min(end - pos, row - within)
                    source = 0
                    batch, head_index = divmod(head, h)
                    src = batch * 3 * hd + output * hd + head_index * d + within - old
            yield start // TILE, pos, n, source, src
            pos += n


def _oracle_address(shape: Any, output: Any, offset: Any) -> Any:
    b, h, cache_length, d = shape
    index_d = offset % d
    token = (offset // d) % (1 if output == 0 else cache_length + 1)
    head = offset // (d * (1 if output == 0 else cache_length + 1))
    if output and token < cache_length:
        return output, (head * cache_length + token) * d + index_d
    batch, index_h = divmod(head, h)
    return 0, ((batch * 3 + output) * h + index_h) * d + index_d


def _check(shape: Any, *, bitwise: Any = False) -> Any:
    b, h, cache_length, d = shape
    input_sizes = [b * 3 * h * d, b * h * cache_length * d, b * h * cache_length * d]
    output_sizes = [b * h * d, b * h * (cache_length + 1) * d, b * h * (cache_length + 1) * d]
    assert max(input_sizes + output_sizes) < 2**32
    count = 0
    outputs = []
    if bitwise:
        inputs = [(np.arange(n, dtype=np.uint32) * 997 + i * 7919).astype(np.uint16) for i, n in enumerate(input_sizes)]
    for output, size in enumerate(output_sizes):
        cursor = 0
        tile_ids = set()
        if bitwise:
            actual = np.empty(size, dtype=np.uint16)
        for tile, pos, n, source, src in _fragments(*shape, output):
            tile_ids.add(tile)
            assert pos == cursor and n > 0 and src >= 0 and src + n <= input_sizes[source]
            assert pos // TILE == tile and (pos + n - 1) // TILE == tile
            if output and size >= 16 * 1024 * 1024 and d % 16 == 0:
                # The gathered fast path uses plain DMA into disjoint UB spans.
                assert (pos - tile * TILE) % 16 == src % 16 == n % 16 == 0
                assert pos - tile * TILE + n <= TILE
            for index in {0, n // 2, n - 1}:
                assert (source, src + index) == _oracle_address(shape, output, pos + index)
            if bitwise:
                actual[pos : pos + n] = inputs[source][src : src + n]
            cursor += n
            count += 1
        assert cursor == size
        expected_tiles = (size + TILE - 1) // TILE
        assert tile_ids == set(range(expected_tiles)), (shape, output, len(tile_ids), expected_tiles)
        if bitwise:
            outputs.append(actual)
    if bitwise:
        arrays = [
            inputs[0].reshape(b, 1, 3 * h * d),
            inputs[1].reshape(b, h, cache_length, d),
            inputs[2].reshape(b, h, cache_length, d),
        ]
        expected = golden.calculate(arrays, {})
        for actual, wanted in zip(outputs, expected):
            assert np.array_equal(actual, wanted.reshape(-1))
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cpu_address_model.json")
    args = parser.parse_args()
    # The 192 KiB configuration was rejected by GetWorkspaceSize (561002).
    depth_match = re.search(r"kQueueDepth = (\d+)", (ROOT / "op_kernel/t5_copy.h").read_text())
    assert depth_match is not None, "queue depth must be explicit for UB accounting"
    queue_depth = int(depth_match.group(1))
    ub_bytes = queue_depth * TILE * np.dtype(np.uint16).itemsize
    assert ub_bytes <= 160 * 1024, ("queue UB budget", queue_depth, ub_bytes)
    start = time.perf_counter()
    records = []
    cases = json.loads((ROOT / "reference/case.json").read_text(encoding="utf-8"))
    for case in cases:
        shape = case["input_desc"][1]["shape"]
        b, h, cache_length, d = shape
        small = b * h * d * (2 * cache_length + 3) <= 2**23
        count = _check(shape, bitwise=small)
        records.append(
            {
                "case_name": case["case_name"],
                "shape": shape,
                "status": "passed",
                "fragment_count": count,
                "address_ranges_checked": "all",
                "bitwise_cpu_arrays": small,
                "npu_executed": False,
            }
        )
    rng = random.Random(20261028)
    for _ in range(120):
        shape = (rng.randint(1, 4), rng.randint(1, 8), rng.choice([0, 1, 2, 31, 63, 128, 257]), rng.randint(1, 256))
        _check(shape, bitwise=True)
    report = {
        "status": "passed",
        "test_type": "CPU address-model checks only",
        "npu_executed": False,
        "public_shapes": len(records),
        "additional_random_shapes": 120,
        "cases": records,
        "seconds": time.perf_counter() - start,
    }
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
