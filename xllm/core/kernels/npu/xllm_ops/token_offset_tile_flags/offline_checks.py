# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU emulation of the actual kernel body; NOT an NPU compilation/acceptance test."""

from __future__ import annotations

import ast
import copy
import datetime as dt
import importlib.util
import json
import random
import types
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from fixtures import ensure_fixtures
from statistics_utils import paired_statistics
from validation import validate_attributes, validate_values

ROOT = Path(__file__).resolve().parent
ensure_fixtures(ROOT)


def _require(condition: Any, message: str) -> None:
    # Emulator bounds checks must also raise under `python -O` / PYTHONOPTIMIZE.
    if not condition:
        raise AssertionError(message)


class Array(np.ndarray):
    def to(self, dtype: Any) -> Any:
        return self.astype(dtype).view(Array)


def _array(value: Any, dtype: Any = None) -> Any:
    return np.asarray(value, dtype=dtype).view(Array)


class Pointer:
    def __init__(self, buffer: Any, index: Any = 0) -> None:
        self.buffer, self.index = buffer, index

    def __add__(self, index: Any) -> Any:
        return Pointer(self.buffer, self.index + index)


class Language:
    int32, int64, uint8, constexpr = np.int32, np.int64, np.uint8, object
    tensor = Array
    pid = 0
    row = 0

    def program_id(self, axis: Any) -> Any:
        return _array(self.pid if axis == 0 else self.row, np.int32)

    @staticmethod
    def arange(start: Any, stop: Any) -> Any:
        return _array(np.arange(start, stop, dtype=np.int32))

    @staticmethod
    def load(pointer: Any, mask: Any = True, other: Any = 0) -> Any:
        ids, mask = np.broadcast_arrays(pointer.index, mask)
        selected = ids[mask]
        _require((selected >= 0).all() and (selected < len(pointer.buffer)).all(), "unmasked out-of-bounds read")
        result = np.full(ids.shape, other, dtype=pointer.buffer.dtype)
        result[mask] = pointer.buffer[selected]
        return _array(result)

    @staticmethod
    def store(pointer: Any, value: Any, mask: Any = True) -> Any:
        ids, mask, value = np.broadcast_arrays(pointer.index, mask, value)
        selected = ids[mask]
        _require((selected >= 0).all() and (selected < len(pointer.buffer)).all(), "out-of-bounds write")
        pointer.buffer[selected] = value[mask]

    @staticmethod
    def sum(value: Any, axis: Any) -> Any:
        return _array(np.sum(value, axis=axis, dtype=np.int32))

    @staticmethod
    def reshape(value: Any, shape: Any) -> Any:
        return _array(np.reshape(value, shape))

    @staticmethod
    def where(condition: Any, a: Any, b: Any) -> Any:
        return _array(np.where(condition, a, b))

    @staticmethod
    def min(value: Any, axis: Any) -> Any:
        return _array(np.min(value, axis=axis))

    @staticmethod
    def max(value: Any, axis: Any) -> Any:
        return _array(np.max(value, axis=axis))

    @staticmethod
    def minimum(a: Any, b: Any) -> Any:
        return _array(np.minimum(a, b))


def _load_kernel(name: Any) -> Any:
    tree = ast.parse((ROOT / "kernels.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {"tl": Language()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "kernels.py", "exec"), ns)
    return ns[name], ns["tl"]


KERNEL, TL = _load_kernel("tile_flags")
LARGE_KERNEL, LARGE_TL = _load_kernel("tile_flags_large")
ROW_KERNEL, ROW_TL = _load_kernel("tile_flags_row")


def _emulate(values: Any, attrs: Any) -> Any:
    t, k = validate_attributes(attrs)
    rows = validate_values(values, t, k)
    lanes = 1 << (t - 1).bit_length()
    group = 32 if t == 128 else min(32, max(1, 1024 // lanes))
    out = np.full(rows * k, 255, dtype=np.uint8)
    pointers = [Pointer(np.asarray(x, dtype=np.int32)) for x in values]
    kernel, language = (LARGE_KERNEL, LARGE_TL) if group == 1 else (KERNEL, TL)
    programs = (len(out) + group - 1) // group
    if t == 128:
        kernel, language = ROW_KERNEL, ROW_TL
        programs = rows * ((k + group - 1) // group)
    grid = [(pid, 0) for pid in range(programs)]
    for pid, row in grid:
        language.pid = pid
        language.row = row
        kernel(pointers[0], pointers[1], pointers[2], Pointer(out), len(values[0]), rows * k, t, k, lanes, group)
    _require(not (out == 255).any(), "unwritten output")
    return out.tolist()


def _load(name: Any, path: Any) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads((ROOT / "official/case.json").read_text(encoding="utf-8"))
        cls.generate = staticmethod(_load("generator", ROOT / "official/generate_data.py").generate)
        cls.golden = staticmethod(_load("oracle", ROOT / "official/golden.py").calculate)
        cls.public = []

    def test_public_kernel_emulation(self) -> None:
        for case in self.fixture["input_cases"]:
            with self.subTest(case=case["id"]):
                values, attrs = self.generate(case)
                actual, expected = _emulate(values, attrs), self.golden(values, attrs)
                self.assertEqual(actual, expected)
                if "expected" in case:
                    self.assertEqual(actual, case["expected"])
                self.public.append({"id": case["id"], "status": "PASS_CPU_EMULATION", "outputs": len(actual)})

    def test_randomized_kernel_emulation(self) -> None:
        rng = random.Random(20261117)
        for iteration in range(350):
            tile = rng.choice([1, 2, 3, 5, 7, 31, 127, 128, 129, 255, 256, 257, 4095, 4096])
            rows, cap = rng.randrange(0, 8), rng.randrange(1, 8)
            values = [[], [0], [], [i * cap for i in range(rows + 1)]]
            for _ in range(rows):
                length = rng.randrange(cap * tile + 1)
                segment = []
                for i in range((length + tile - 1) // tile):
                    base = rng.randrange(0, 1000) * tile
                    if rng.random() < 0.2:
                        base = ((2**31 - 1) // tile - 1) * tile
                    if rng.random() < 0.3:
                        base += 1
                    chunk = list(range(base, base + tile))
                    if rng.random() < 0.4:
                        chunk[rng.randrange(tile)] = rng.randrange(100)
                    segment.extend(chunk)
                segment = segment[:length] + [999] * rng.randrange(4)
                values[0].extend(segment)
                values[1].append(len(values[0]))
                values[2].append(length)
            attrs = {"tile_size": tile, "max_tiles_per_seq": cap}
            with self.subTest(iteration=iteration, tile=tile):
                self.assertEqual(_emulate(values, attrs), self.golden(values, attrs))

    def test_values_rejected(self) -> None:
        base = [[0, 1, 2, 3], [0, 4], [4], [0, 2]]
        changes = [(0, 0, -1), (1, 0, 1), (1, 1, 3), (2, 0, -1), (2, 0, 5), (3, 1, 1)]
        for a, b, value in changes:
            values = copy.deepcopy(base)
            values[a][b] = value
            with self.assertRaises(ValueError):
                validate_values(values, 4, 2)
        with self.assertRaises(ValueError):
            validate_values(base, 1, 2)
        with self.assertRaises(ValueError):
            validate_values(base, 4, 2**31)
        with self.assertRaises(ValueError):
            validate_values([[], [], [], []], 1, 1)

    def test_attributes_rejected(self) -> None:
        for tile in [0, -1, True, 1.0, "4", 4097]:
            with self.assertRaises(ValueError):
                validate_attributes({"tile_size": tile, "max_tiles_per_seq": 2})
        for cap in [0, -1, False, 2.0]:
            with self.assertRaises(ValueError):
                validate_attributes({"tile_size": 4, "max_tiles_per_seq": cap})
        for attrs in [{}, {"tile_size": 4}, {"tile_size": 4, "max_tiles_per_seq": 2, "unknown": 1}]:
            with self.assertRaises(ValueError):
                validate_attributes(attrs)

    def test_paired_statistics(self) -> None:
        result = paired_statistics([100, 101, 102, 103, 104, 105], [60, 61, 62, 63, 64, 65], 85)
        self.assertEqual(result["paired_mean_difference_ci95_us"], [40.0, 40.0])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(paired_statistics([100] * 6, [90] * 6, 85)["status"], "FAIL")
        self.assertEqual(paired_statistics([10] * 6, [10] * 6, 85)["status"], "FAIL")
        # Pair preservation matters: both arrays vary, their differences do not.
        self.assertEqual(
            paired_statistics([20, 40, 60, 80, 100, 120], [10, 30, 50, 70, 90, 110])["paired_mean_difference_ci95_us"],
            [10.0, 10.0],
        )
        with self.assertRaises(ValueError):
            paired_statistics([1], [1])
        with self.assertRaises(ValueError):
            paired_statistics([float("nan")] * 6, [1] * 6)


if __name__ == "__main__":
    directory = ROOT / "results/offline"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "checks.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        )
    report = {
        "status": "PASS_CPU_ONLY" if result.wasSuccessful() else "FAIL",
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "public_cases": Checks.public,
        "randomized_cases": 350,
        "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Actual kernel Python body emulated with NumPy, pure input-value validation and paired statistics. "
        "No Triton compilation, torch/NPU API, stream execution, performance or allocator verification.",
        "npu_acceptance": "NOT_RUN",
    }
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
