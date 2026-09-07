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
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)


def _make_moe(
    dp_rank: int,
    monkeypatch: pytest.MonkeyPatch,
) -> FusedMoE:
    monkeypatch.setattr(
        kernels,
        "supports_cutlass_moe",
        MagicMock(return_value=True),
        raising=False,
    )
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


def _context(
    execution_token_counts: tuple[int, int],
) -> ForwardContext:
    return ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(
            dp_execution_token_counts=execution_token_counts,
        ),
        layer_caches=[],
    )


@pytest.mark.parametrize(
    ("dp_rank", "token_counts", "expected_start"),
    (
        (1, (3, 0), 3),
        (0, (0, 5), 0),
    ),
)
def test_empty_dp_rank_uses_execution_counts_and_keeps_dummy_row(
    monkeypatch: pytest.MonkeyPatch,
    dp_rank: int,
    token_counts: tuple[int, int],
    expected_start: int,
) -> None:
    layer = _make_moe(dp_rank, monkeypatch)
    execution_counts = tuple(max(count, 1) for count in token_counts)
    gathered_rows = sum(execution_counts)
    gathered = torch.zeros(gathered_rows, 8)
    expert_output = torch.arange(
        gathered_rows * 8,
        dtype=torch.float32,
    ).view(gathered_rows, 8)
    gather = MagicMock(return_value=(gathered, expected_start))
    monkeypatch.setattr(
        distributed,
        "gather_dp_execution_tokens",
        gather,
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(gathered_rows, 1),
                torch.zeros(gathered_rows, 1, dtype=torch.int32),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "cutlass_fused_moe",
        MagicMock(return_value=expert_output),
        raising=False,
    )
    monkeypatch.setattr(
        distributed,
        "moe_tp_all_reduce",
        MagicMock(),
        raising=False,
    )

    context = _context(execution_counts)
    local_input = torch.zeros(execution_counts[dp_rank], 8)
    with forward_context(context):
        output = layer(local_input)

    torch.testing.assert_close(
        output,
        expert_output[expected_start : expected_start + 1],
    )
    gather.assert_called_once_with(
        local_input,
        execution_counts,
        dp_rank,
    )


def test_moe_gather_depends_only_on_execution_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_moe(dp_rank=1, monkeypatch=monkeypatch)
    gathered = torch.zeros(7, 8)
    expert_output = torch.arange(56, dtype=torch.float32).view(7, 8)
    gather = MagicMock(return_value=(gathered, 3))
    monkeypatch.setattr(
        distributed,
        "gather_dp_execution_tokens",
        gather,
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(7, 1),
                torch.zeros(7, 1, dtype=torch.int32),
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        kernels,
        "cutlass_fused_moe",
        MagicMock(return_value=expert_output),
        raising=False,
    )
    monkeypatch.setattr(
        distributed,
        "moe_tp_all_reduce",
        MagicMock(),
        raising=False,
    )

    local_input = torch.zeros(4, 8)
    with forward_context(_context((3, 4))):
        output = layer(local_input)

    torch.testing.assert_close(output, expert_output[3:7])
    gather.assert_called_once_with(local_input, (3, 4), 1)
