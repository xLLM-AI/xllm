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

"""Exercise logical all-gather with real NPU storage formats and CP shards."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from xllm.python.distributed import collectives
from xllm.python.model_executor.cp_utils import build_cp_context, cp_merge_rows, cp_shard_rows

torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture
def device() -> Iterator[torch.device]:
    if not torch.npu.is_available():
        pytest.skip("Collective storage tests require an Ascend NPU")
    # This configuration has a setter only; read the native option to restore
    # the process-wide setting after each test.
    internal_format_enabled = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == b"enable"
    torch.npu.config.allow_internal_format = True
    try:
        yield torch.device("npu", torch.npu.current_device())
    finally:
        torch.npu.config.allow_internal_format = internal_format_enabled


def _assert_nd_buffers(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.is_contiguous() and output.is_contiguous()
    assert torch_npu.get_npu_format(input) == torch_npu.Format.ND
    assert torch_npu.get_npu_format(output) == torch_npu.Format.ND
    assert input.data_ptr() != output.data_ptr()


@pytest.mark.parametrize("layout", ["nd", "nchw", "nchw_view", "strided", "nz"])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_gather_normalizes_storage_without_changing_values(
    monkeypatch: pytest.MonkeyPatch, device: torch.device, layout: str, dim: int, dtype: torch.dtype
) -> None:
    source = torch.arange(1024, dtype=torch.float32, device=device).to(dtype).view(4, 2, 4, 32)
    if layout == "nz":
        source = torch_npu.npu_format_cast(source.view(32, 32), torch_npu.Format.FRACTAL_NZ)
    elif layout in ("nchw", "nchw_view"):
        source = torch_npu.npu_format_cast(source, torch_npu.Format.NCHW)
        if layout == "nchw_view":
            source = source.view(4, 8, 32)
        assert torch_npu.get_npu_format(source) == torch_npu.Format.NCHW
    else:
        source = torch_npu.npu_format_cast(source, torch_npu.Format.ND)
        if layout == "strided":
            source = source.transpose(0, 1)

    original = source.cpu()
    original_stride = source.stride()
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setitem(collectives._groups, ("cp", str(device)), group)

    def gather(input: torch.Tensor, output: torch.Tensor, group: object) -> None:
        _assert_nd_buffers(input, output)
        torch.testing.assert_close(input.cpu(), original)
        if layout == "nd":
            assert input.data_ptr() == source.data_ptr(), "ND inputs must not be copied"
        output[0].copy_(input)
        output[1].copy_(input + 1024)

    monkeypatch.setattr(collectives, "_all_gather", gather)
    actual = collectives.all_gather(source, dim, 2, "cp")
    expected = torch.cat((original, original + 1024), dim=dim)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(source.cpu(), original, rtol=0, atol=0)
    assert source.shape == original.shape
    assert source.stride() == original_stride


def test_variable_all_gather_normalizes_four_dimensional_padding(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> None:
    peers = [torch.full((3, 2, 1, 8), rank + 1.0, device=device) for rank in range(3)]
    counts = [2, 0, 3]
    monkeypatch.setitem(collectives._groups, ("dp", str(device)), SimpleNamespace(size=lambda: 3))

    def gather(input: torch.Tensor, output: torch.Tensor, group: object) -> None:
        _assert_nd_buffers(input, output)
        assert input[2].count_nonzero().item() == 0
        for rank, peer in enumerate(peers):
            output[rank].copy_(peer)

    monkeypatch.setattr(collectives, "_all_gather", gather)
    actual = collectives.all_gather_variable(peers[0][:2], counts, 0, "dp")
    expected = torch.cat((peers[0][:2], peers[2]))
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("cp_size,query_lengths", [(2, [7, 3]), (4, [1])])
def test_cp_merge_restores_padded_rope_views(
    monkeypatch: pytest.MonkeyPatch, device: torch.device, cp_size: int, query_lengths: list[int]
) -> None:
    values = torch.arange(sum(query_lengths) * 8, dtype=torch.float32, device=device).view(-1, 1, 8)
    plans = [build_cp_context(query_lengths, query_lengths, cp_size, rank, device) for rank in range(cp_size)]
    shards = [
        torch_npu.npu_format_cast(cp_shard_rows(values, plan).unsqueeze(1), torch_npu.Format.NCHW).squeeze(1)
        for plan in plans
    ]
    monkeypatch.setitem(collectives._groups, ("cp", str(device)), SimpleNamespace(size=lambda: cp_size))

    def gather(input: torch.Tensor, output: torch.Tensor, group: object) -> None:
        _assert_nd_buffers(input, output)
        for rank, shard in enumerate(shards):
            output[rank].copy_(shard)

    monkeypatch.setattr(collectives, "_all_gather", gather)
    for shard, plan in zip(shards, plans, strict=True):
        assert torch_npu.get_npu_format(shard) == torch_npu.Format.NCHW
        actual = cp_merge_rows(shard, plan)
        torch.testing.assert_close(actual.cpu(), values.cpu(), rtol=0, atol=0)
