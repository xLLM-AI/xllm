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

from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.parametrize(
    "slots, keep_rows, slot_capacity",
    [
        ([4, 1, 4, -1, 1, 2], [4, 5, 2], 8),
        ([-1, -1], [], 8),
        ([3], [0], 8),
        ([-1], [], 8),
        ([], [], 8),
        ([3, 1, 2], [1, 2, 0], 8),
        ([16777214, 16777215, 16777214, -1], [2, 1], 2**24),
        ([16777216, 16777217, 16777216, -1], [2, 1], 2**24 + 2),
        ([16777216, 16777217, 16777216, -1], [2, 1], None),
    ],
)
def test_last_ring_writes_preserves_exact_npu_slots(
    slots: list[int], keep_rows: list[int], slot_capacity: int | None
) -> None:
    if os.environ.get("XLLM_TEST_NPU") != "1":
        pytest.skip("set XLLM_TEST_NPU=1 after allocating an NPU")
    import torch_npu  # noqa: F401

    from xllm.python.attention.dsa_metadata import last_ring_writes

    torch.npu.set_device(0)
    physical_slots = torch.tensor(slots, dtype=torch.int64, device="npu")
    values = torch.arange(len(slots) * 2, dtype=torch.int32, device="npu").reshape(-1, 2)
    actual_slots, actual_values = last_ring_writes(physical_slots, values, slot_capacity=slot_capacity)
    expected_rows = torch.tensor(keep_rows, dtype=torch.int64)
    assert actual_slots.dtype == physical_slots.dtype
    assert torch.equal(actual_slots.cpu(), physical_slots.cpu()[expected_rows])
    assert torch.equal(actual_values.cpu(), values.cpu()[expected_rows])
