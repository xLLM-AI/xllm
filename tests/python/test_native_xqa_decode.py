# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real host paging contracts; only the CUDA launch boundary is replaced."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python.attention import xqa_decode


def _cpu_plan() -> xqa_decode.XQADecodePlan:
    # Bypass the CUDA-only allocation, then exercise the real preparation code
    # on real CPU tensors. This does not emulate kernel results or Graph replay.
    plan = xqa_decode.XQADecodePlan.__new__(xqa_decode.XQADecodePlan)
    plan._table = torch.empty((2, 2), dtype=torch.int32)
    plan._lengths = torch.empty(2, dtype=torch.uint32)
    plan._batch_size = 2
    plan._maximum = 256
    plan._page_size = 128
    plan._metadata_addresses = None
    return plan


def _metadata() -> SimpleNamespace:
    return SimpleNamespace(
        paged_kv_indptr_host=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_last_page_len_host=torch.tensor([1, 1], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1, 1], dtype=torch.int32),
    )


@pytest.mark.parametrize("field", ["paged_kv_indptr", "paged_kv_indices", "paged_kv_last_page_len"])
@pytest.mark.parametrize("layout", ["strided", "matrix"])
def test_csr_requires_flat_contiguous_buffers(field: str, layout: str) -> None:
    metadata = _metadata()
    tensor = getattr(metadata, field)
    if layout == "strided":
        storage = torch.full((2 * tensor.numel(),), 77, dtype=torch.int32)
        storage[::2] = tensor
        tensor = storage[::2]
    else:
        tensor = tensor.reshape(1, -1)
    setattr(metadata, field, tensor)
    with patch.object(xqa_decode, "_convert_pages", MagicMock()) as kernel:
        with pytest.raises(ValueError, match="contiguous.*int32"):
            _cpu_plan().prepare(metadata, graph_mode=False)
        kernel.assert_not_called()


@pytest.mark.parametrize(
    "batch,page,maximum",
    [(0, 128, 256), (-1, 128, 256), (2, 0, 256), (2, 8, 256), (2, 256, 512), (2, 128, 0), (2, 128, 129)],
)
def test_invalid_geometry_fails_before_device_query_or_allocation(batch: int, page: int, maximum: int) -> None:
    with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)) as capability:
        with (
            patch.object(torch, "empty", side_effect=AssertionError("allocation reached")),
            pytest.raises(ValueError, match="XQA requires"),
        ):
            xqa_decode.XQADecodePlan(batch, page, maximum, torch.device("cuda"))
        capability.assert_not_called()


@pytest.mark.parametrize(
    "indptr,last",
    [([0, 1, 2], [0, 1]), ([0, 1, 2], [129, 1]), ([0, 1, 1], [1, 2]), ([0, 2, 1], [1, 1]), ([0, 1, 4], [1, 1])],
)
def test_invalid_host_pages_fail_before_launch(indptr: list[int], last: list[int]) -> None:
    metadata = _metadata()
    metadata.paged_kv_indptr_host = torch.tensor(indptr, dtype=torch.int32)
    metadata.paged_kv_last_page_len_host = torch.tensor(last, dtype=torch.int32)
    with patch.object(xqa_decode, "_convert_pages", MagicMock()) as kernel:
        with pytest.raises(ValueError):
            _cpu_plan().prepare(metadata, graph_mode=False)
        kernel.assert_not_called()


def test_graph_padding_is_accepted_but_changed_addresses_are_rejected() -> None:
    metadata = _metadata()
    metadata.paged_kv_indptr_host.copy_(torch.tensor([0, 1, 1], dtype=torch.int32))
    metadata.paged_kv_indptr.copy_(metadata.paged_kv_indptr_host)
    plan = _cpu_plan()
    with patch.object(xqa_decode, "_convert_pages", MagicMock()) as kernel:
        plan.prepare(metadata, graph_mode=True)
        addresses = plan._metadata_addresses
        metadata.paged_kv_indices = metadata.paged_kv_indices.clone()
        with pytest.raises(RuntimeError, match="addresses changed"):
            plan.prepare(metadata, graph_mode=True)
        assert plan._metadata_addresses == addresses
        kernel.assert_called_once()
