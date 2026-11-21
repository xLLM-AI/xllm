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
"""Smoke tests for the packed page-indices operator."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch_npu  # noqa: F401,E402

from solution import prepare  # noqa: E402

NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
pytestmark = pytest.mark.skipif(not NPU_AVAILABLE, reason="Ascend NPU required")


def _ref(x, m, cu):
    single = m.dim() == 1
    rows = [m] if single else [m[i] for i in range(m.shape[0])]
    out = []
    for row in rows:
        result = []
        for r in range(x.shape[0]):
            count = int(cu[r + 1] - cu[r])
            for j in range(count):
                value = int(x[r, j])
                result.append(-1 if value < 0 else int(row[value]))
        out.append(result)
    return out[0] if single else out


def _npu(value, shape=None):
    t = torch.tensor(value, dtype=torch.int32)
    if shape is not None:
        t = t.reshape(shape)
    return t.contiguous().npu()


def _check(xv, mv, cuv, xshape=None):
    x = _npu(xv, xshape)
    m = _npu(mv)
    cu = _npu(cuv)
    prepared = prepare([x, m, cu], {})
    actual = prepared.run()
    torch.npu.synchronize()
    expected = _ref(x.cpu(), m.cpu(), cu.cpu())
    assert actual.cpu().tolist() == expected
    assert actual.dtype == torch.int32
    assert actual.device.type == "npu"
    assert actual.is_contiguous()
    return prepared, x, m, cu


def test_literal():
    _check(
        [[0, 2, -1], [3, 1, -1], [1, -1, -1]],
        [[3, 0, 6, 1], [5, 2, 7, 4]],
        [0, 2, 2, 3],
    )


def test_prefix_padding():
    _check(
        [[0, -7, 2], [3, 1, -1]],
        [[3, 0, 6, 1], [5, 2, 7, 4]],
        [0, 3, 4],
    )


def test_single_layer():
    _check([[3, 1]], [3, 0, 6, 1], [0, 2])


def test_empty_requests():
    _check([], [[0] * 31, [0] * 31], [0], xshape=(0, 3))


def test_empty_columns():
    _check([], [[0] * 31, [0] * 31], [0, 0, 0, 0], xshape=(3, 0))


def test_dynamic_counts_same_total():
    prepared, x, m, cu = _check(
        [[0, 2, -1], [3, 1, -1], [1, -1, -1]],
        [[3, 0, 6, 1], [5, 2, 7, 4]],
        [0, 2, 2, 3],
    )
    cu.copy_(torch.tensor([0, 1, 1, 3], dtype=torch.int32).npu())
    actual = prepared.run()
    torch.npu.synchronize()
    expected = _ref(x.cpu(), m.cpu(), cu.cpu())
    assert actual.cpu().tolist() == expected


def test_storage_rebinding_offset():
    prepared, x, m, cu = _check(
        [[0, 2, -1], [3, 1, -1], [1, -1, -1]],
        [[3, 0, 6, 1], [5, 2, 7, 4]],
        [0, 2, 2, 3],
    )
    old = x.detach()
    backing = torch.empty(x.numel() + 17, dtype=torch.int32, device=x.device)
    view = backing[1 : x.numel() + 1].reshape(x.shape)
    view.copy_(x)
    x.set_(view)
    assert x.storage_offset() == 1
    actual = prepared.run()
    torch.npu.synchronize()
    expected = _ref(x.cpu(), m.cpu(), cu.cpu())
    assert actual.cpu().tolist() == expected
    x.set_(old)


if __name__ == "__main__":
    test_literal()
    test_prefix_padding()
    test_single_layer()
    test_empty_requests()
    test_empty_columns()
    test_dynamic_counts_same_total()
    test_storage_rebinding_offset()
    print("all smoke tests passed")
