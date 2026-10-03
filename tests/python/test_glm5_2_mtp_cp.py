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

"""Exercise real MTP row transforms with only CP transport replaced."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.model_executor.cp_utils import CpContext, build_cp_context
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models import deepseek_v32, glm5_2_mtp


class _RowEmbedding(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack((tokens.float(), tokens.float() + 1), dim=-1)


class _FuseHidden(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden[:, :2] + 2 * hidden[:, 2:]


class _RowDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._reuse_flags: list[bool] = []

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor],
        topk_indices: torch.Tensor | None,
        reuse_topk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del residual, half_rope_cos, half_rope_sin, rope_cos, rope_sin, query_cos_sin
        self._reuse_flags.append(reuse_topk)
        if not reuse_topk:
            # Distinct row values make both an incorrect shard and a missing
            # global restore observable in the following draft invocation.
            topk_indices = hidden.to(torch.int32).unsqueeze(1) + 7
        assert topk_indices is not None
        assert topk_indices.shape == (hidden.shape[0], 1, 2)
        output = hidden + topk_indices.sum(dim=(1, 2)).view(-1, 1)
        return output, hidden * 3, topk_indices + 1


def _mtp_body() -> glm5_2_mtp.Glm52MtpModel:
    body = glm5_2_mtp.Glm52MtpModel.__new__(glm5_2_mtp.Glm52MtpModel)
    nn.Module.__init__(body)
    body.cfg = SimpleNamespace(
        index_share_for_mtp_iteration=True,
        indexer_rope_interleave=True,
        enable_attn_dp_weight_sharding=False,
    )
    body.embed_tokens = _RowEmbedding()
    body.eh_proj = _FuseHidden()
    body.rot = nn.Identity()
    body.enorm = nn.Identity()
    body.hnorm = nn.Identity()
    body.layers = nn.ModuleList([_RowDecoder()])
    body.rotary = deepseek_v32.DeepseekYarnRotaryEmbedding(
        4, 32, 1.0, 10000.0, 32, 1, 1.0, 1.0, dtype=torch.float32, device=torch.device("cpu")
    )
    body.enable_rot = False
    body._reuse_topk_by_layer = (True,)
    return body


def _forward(
    body: glm5_2_mtp.Glm52MtpModel,
    plan: CpContext | None,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    hidden: torch.Tensor,
    topk: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    with (
        forward_context(ForwardContext(None, torch.device("cpu"), None, [], cp_context=plan)),
        patch.object(glm5_2_mtp, "record_layer_event"),
    ):
        output, auxiliary, indices = body(tokens, positions, hidden, topk)
    assert auxiliary is None
    assert indices is not None
    return output, indices


@pytest.mark.parametrize(("cp_size", "query_lengths"), [(2, [5, 1]), (4, [1])])
def test_mtp_cp_restores_hidden_and_topk_across_draft_steps(cp_size: int, query_lengths: list[int]) -> None:
    device = torch.device("cpu")
    kv_lengths = [length + 2 for length in query_lengths]
    plans = [build_cp_context(query_lengths, kv_lengths, cp_size, rank, device) for rank in range(cp_size)]
    rows = sum(query_lengths)
    tokens = torch.arange(1, rows + 1, dtype=torch.int32)
    positions = torch.cat([torch.arange(2, length + 2, dtype=torch.int32) for length in query_lengths])
    hidden = torch.arange(rows * 2, dtype=torch.float32).view(rows, 2) + 10
    topk = None
    selected = torch.tensor(query_lengths, dtype=torch.int64).cumsum(0) - 1
    bodies = [_mtp_body() for _ in plans]
    reference = _mtp_body()
    if cp_size == 4:
        assert sum(plan.query_index.numel() == 0 for plan in plans) == 3

    for step in range(2):
        original_hidden = hidden.clone()
        original_topk = None if topk is None else topk.clone()
        expected_hidden, expected_topk = _forward(reference, None, tokens, positions, hidden, topk)
        shards: list[list[torch.Tensor]] = []

        # First evaluate every rank's actual local model outputs. The temporary
        # gather permits forward to finish; its global result is discarded.
        for body, plan in zip(bodies, plans, strict=True):
            local_outputs: list[torch.Tensor] = []

            def collect_local(
                value: torch.Tensor,
                dim: int,
                world_size: int,
                group_name: str,
                *,
                outputs: list[torch.Tensor] = local_outputs,
            ) -> torch.Tensor:
                assert (dim, world_size, group_name) == (0, cp_size, "cp")
                outputs.append(value.clone())
                return torch.cat([value] * cp_size, dim=0)

            with patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=collect_local):
                _forward(body, plan, tokens, positions, hidden, topk)
            assert len(local_outputs) == 2
            shards.append(local_outputs)

        # Replay each rank against actual peer tensors, leaving production
        # sharding, padding, model forward, and restore-index selection intact.
        for rank, (body, plan) in enumerate(zip(bodies, plans, strict=True)):
            gather_calls = 0

            def gather_peers(
                value: torch.Tensor,
                dim: int,
                world_size: int,
                group_name: str,
                *,
                rank_outputs: list[list[torch.Tensor]] = shards,
                cp_rank: int = rank,
            ) -> torch.Tensor:
                nonlocal gather_calls
                assert (dim, world_size, group_name) == (0, cp_size, "cp")
                torch.testing.assert_close(value, rank_outputs[cp_rank][gather_calls])
                peer_values = [peer[gather_calls] for peer in rank_outputs]
                peer_values[cp_rank] = value
                gather_calls += 1
                return torch.cat(peer_values, dim=0)

            with patch("xllm.python.model_executor.cp_utils.distributed.all_gather", side_effect=gather_peers):
                actual_hidden, actual_topk = _forward(body, plan, tokens, positions, hidden, topk)
            assert gather_calls == 2
            torch.testing.assert_close(actual_hidden, expected_hidden)
            torch.testing.assert_close(actual_topk, expected_topk)
            torch.testing.assert_close(actual_hidden[selected], expected_hidden[selected])
            torch.testing.assert_close(actual_topk[selected], expected_topk[selected])
            assert body.layers[0]._reuse_flags[-2:] == [bool(step)] * 2

        torch.testing.assert_close(hidden, original_hidden)
        if original_topk is not None:
            torch.testing.assert_close(topk, original_topk)
        hidden = expected_hidden + 100
        topk = expected_topk + 20
        tokens = tokens + 1


@pytest.mark.parametrize("topk_shape", [(6,), (5, 1, 2), (7, 1, 2)])
def test_mtp_cp_rejects_non_global_topk_rows_before_layers(topk_shape: tuple[int, ...]) -> None:
    plan = build_cp_context([5, 1], [7, 3], 2, 1, torch.device("cpu"))
    body = _mtp_body()
    with (
        patch("xllm.python.model_executor.cp_utils.distributed.all_gather") as gather,
        pytest.raises(ValueError, match="one leading row per global input token"),
    ):
        _forward(
            body,
            plan,
            torch.arange(6, dtype=torch.int32),
            torch.arange(6, dtype=torch.int32),
            torch.zeros(6, 2),
            torch.zeros(topk_shape, dtype=torch.int32),
        )
    assert body.layers[0]._reuse_flags == []
    gather.assert_not_called()
