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

"""Data-parallel contracts for the shared CUDA fused-MoE layer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python import distributed, kernels
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.model_executor.forward_context import ForwardContext, forward_context


def _make_moe(dp_rank: int, monkeypatch: pytest.MonkeyPatch) -> FusedMoE:
    monkeypatch.setattr(kernels, "supports_cutlass_moe", MagicMock(return_value=True), raising=False)
    layer = FusedMoE(
        hidden_size=8,
        intermediate_size=4,
        num_experts=2,
        top_k=1,
        renormalize=True,
        moe_tp_size=2,
        moe_tp_rank=0,
        ep_size=1,
        ep_rank=0,
        dp_size=2,
        dp_rank=dp_rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    layer.gate = torch.nn.Identity()
    return layer


@pytest.mark.parametrize(
    "dp_rank,token_counts,expected_start",
    (
        pytest.param(1, (3, 0), 3, id="empty-rank1"),
        pytest.param(0, (0, 5), 0, id="empty-rank0"),
        pytest.param(1, (3, 4), 3, id="nonempty-rank1"),
        pytest.param(0, (3, 4), 0, id="nonempty-rank0"),
    ),
)
def test_dp_gather_reduces_then_slices_execution_rows(
    monkeypatch: pytest.MonkeyPatch, dp_rank: int, token_counts: tuple[int, int], expected_start: int
) -> None:
    layer = _make_moe(dp_rank, monkeypatch)
    # Empty ranks still execute a dummy row; metadata contains execution counts.
    execution_counts = tuple(max(count, 1) for count in token_counts)
    gathered_rows = sum(execution_counts)
    gathered = torch.zeros(gathered_rows, 8)
    unreduced = torch.arange(gathered_rows * 8, dtype=torch.float32).view(gathered_rows, 8)
    expert_output = unreduced.clone()
    expected_reduced = unreduced * 2 + 13
    gather = MagicMock(return_value=(gathered, expected_start))
    monkeypatch.setattr(distributed, "gather_dp_execution_tokens", gather, raising=False)
    topk = MagicMock(return_value=(torch.ones(gathered_rows, 1), torch.zeros(gathered_rows, 1, dtype=torch.int32)))
    monkeypatch.setattr(kernels, "moe_fused_topk", topk, raising=False)
    monkeypatch.setattr(kernels, "cutlass_fused_moe", MagicMock(return_value=expert_output), raising=False)

    def _reduce(output: torch.Tensor) -> None:
        # Reduction must see all gathered rows, not an already-scattered slice.
        torch.testing.assert_close(output, unreduced, rtol=0, atol=0)
        output.mul_(2).add_(13)

    reducer = MagicMock(side_effect=_reduce)
    monkeypatch.setattr(distributed, "moe_tp_all_reduce", reducer, raising=False)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=execution_counts),
        layer_caches=[],
    )
    local_rows = execution_counts[dp_rank]
    local_input = torch.zeros(local_rows, 8)
    with forward_context(context):
        output = layer(local_input)
    torch.testing.assert_close(output, expected_reduced[expected_start : expected_start + local_rows], rtol=0, atol=0)
    gather.assert_called_once_with(local_input, execution_counts, dp_rank)
    reducer.assert_called_once_with(expert_output)
