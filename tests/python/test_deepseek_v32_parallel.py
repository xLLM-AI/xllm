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

"""Parallel-layout tests for the DeepSeek-V3.2 Python model (DP/EP)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python import distributed, kernels
from xllm.python.model_executor.forward_context import (  # noqa: E402
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models.deepseek_v32 import (  # noqa: E402
    DeepseekV3Config,
    DeepseekV3MoE,
)


@pytest.fixture(autouse=True)
def _mock_parallel_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "grouped_moe", MagicMock(), raising=False)
    monkeypatch.setattr(distributed, "all_gather_variable", MagicMock())
    monkeypatch.setattr(distributed, "all_reduce_", MagicMock())
    monkeypatch.setattr(
        distributed,
        "all_gather",
        MagicMock(side_effect=lambda x, **kw: x.repeat(kw["world_size"], 1)),
    )
    monkeypatch.setattr(distributed, "tp_rank", MagicMock(return_value=0))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _config(**overrides) -> DeepseekV3Config:
    values = {
        "hidden_size": 64,
        "n_layers": 2,
        "n_heads": 4,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 1024,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "index_n_heads": 4,
        "index_head_dim": 16,
        "index_topk": 64,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 16,
        "n_shared_experts": 1,
        "num_experts_per_tok": 4,
        "n_group": 4,
        "topk_group": 2,
        "routed_scaling_factor": 2.5,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "moe_intermediate_size": 32,
        "tp_size": 1,
        "tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "world_size": 1,
    }
    values.update(overrides)
    return DeepseekV3Config.from_dict(values)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestDeepseekV3ConfigValidation:
    def test_pure_tp_valid(self):
        cfg = _config(tp_size=2, ep_size=1, world_size=2)
        cfg.validate()

    def test_ep2_valid(self):
        cfg = _config(ep_size=2, moe_tp_size=1, world_size=2)
        cfg.validate()

    def test_ep_equals_world_size_valid(self):
        cfg = _config(ep_size=4, moe_tp_size=1, world_size=4)
        cfg.validate()

    def test_ep_invalid_not_1_or_world(self):
        cfg = _config(ep_size=3, world_size=4)
        with pytest.raises(ValueError, match="ep_size must be 1 or world_size"):
            cfg.validate()

    def test_ep_experts_not_divisible(self):
        cfg = _config(n_routed_experts=15, ep_size=2, moe_tp_size=1, world_size=2)
        with pytest.raises(ValueError, match="divisible"):
            cfg.validate()

    def test_ep_moe_tp_world_mismatch(self) -> None:
        # EP itself is valid, so only the MoE-TP product guard can reject this.
        cfg = _config(ep_size=2, moe_tp_size=2, world_size=2)
        with pytest.raises(
            ValueError,
            match=r"^world_size \(2\) must equal moe_tp_size \(2\) \* ep_size \(2\)$",
        ):
            cfg.validate()


# ---------------------------------------------------------------------------
# MoE layer construction tests
# ---------------------------------------------------------------------------


def _make_moe(
    ep_size: int = 1,
    ep_rank: int = 0,
    dp_size: int = 1,
    dp_rank: int = 0,
    moe_tp_size: int = 1,
    n_experts: int = 16,
) -> DeepseekV3MoE:
    cfg = _config(
        n_routed_experts=n_experts,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        moe_tp_size=moe_tp_size,
        world_size=max(ep_size, 1) * dp_size,
    )
    moe = DeepseekV3MoE(cfg, layer_id=0, dtype=torch.float32, device=torch.device("cpu"))
    # CPU addmm cannot accumulate bf16 operands into an fp32 output. These
    # layout tests use an ordinary fp32 gate while mocking the expert kernels.
    moe.gate = torch.nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=torch.float32)
    return moe


class TestDeepseekV3MoEConstruction:
    def test_ep1_all_experts_local(self):
        moe = _make_moe(ep_size=1)
        assert moe.num_local_experts == 16
        assert moe.local_expert_start == 0
        assert moe.local_expert_end == 16

    def test_ep2_rank0_first_half(self):
        moe = _make_moe(ep_size=2, ep_rank=0)
        assert moe.num_local_experts == 8
        assert moe.local_expert_start == 0
        assert moe.local_expert_end == 8

    def test_ep2_rank1_second_half(self):
        moe = _make_moe(ep_size=2, ep_rank=1)
        assert moe.num_local_experts == 8
        assert moe.local_expert_start == 8
        assert moe.local_expert_end == 16

    def test_intermediate_tp_sharding(self):
        moe = _make_moe(ep_size=2, ep_rank=0, moe_tp_size=2)
        assert moe.inter_local == 32 // 2  # moe_intermediate_size // moe_tp_size


# ---------------------------------------------------------------------------
# MoE forward call tests
# ---------------------------------------------------------------------------


def _mock_forward_context(
    dp_execution_token_counts=(4,),
    is_graph=False,
    dp_is_decode=None,
):
    metadata = SimpleNamespace(
        dp_execution_token_counts=dp_execution_token_counts,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    if dp_is_decode is not None:
        metadata.dp_is_decode = dp_is_decode
    execution_state = AclGraphExecutionState(persistent_buffers={}) if is_graph else None
    ctx = ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        execution_state=execution_state,
    )
    return ctx


class TestDeepseekV3MoEForward:
    @pytest.mark.parametrize(
        ("ep_size", "ep_rank", "moe_tp_size", "groups"),
        [
            (1, 0, 1, []),
            (2, 0, 1, ["moe_ep"]),
            (2, 1, 1, ["moe_ep"]),
            (2, 0, 2, ["moe_ep", "moe_tp"]),
        ],
    )
    def test_expert_range_and_reduction(self, ep_size: int, ep_rank: int, moe_tp_size: int, groups: list[str]) -> None:
        moe = _make_moe(ep_size=ep_size, ep_rank=ep_rank, moe_tp_size=moe_tp_size)
        hidden = torch.ones(4, 64)
        kernels.grouped_moe.return_value = hidden
        moe.shared_experts.forward = MagicMock(return_value=torch.zeros_like(hidden))
        with forward_context(_mock_forward_context()):
            moe.forward(hidden)
        assert kernels.grouped_moe.call_args.args[12] == [16 * ep_rank // ep_size, 16 * (ep_rank + 1) // ep_size]
        assert [args.args[1] for args in distributed.all_reduce_.call_args_list] == groups
        distributed.all_gather.assert_not_called()

    @pytest.mark.parametrize(
        ("is_graph", "dp_rank", "rows", "offset", "local_tokens"),
        [(True, 0, 8, 0, 3), (True, 1, 8, 4, 4), (False, 0, 7, 0, 3), (False, 1, 7, 3, 4)],
    )
    def test_dp_gather_and_local_slice(
        self, is_graph: bool, dp_rank: int, rows: int, offset: int, local_tokens: int
    ) -> None:
        moe = _make_moe(dp_size=2, dp_rank=dp_rank)
        hidden = torch.ones(local_tokens, 64)
        gathered = torch.arange(rows, dtype=torch.float32).view(-1, 1).expand(-1, 64)
        distributed.all_gather.side_effect = None
        distributed.all_gather.return_value = gathered
        distributed.all_gather_variable.return_value = gathered
        kernels.grouped_moe.return_value = gathered + 10
        moe.shared_experts.forward = MagicMock(return_value=torch.zeros_like(gathered))
        ctx = _mock_forward_context((3, 4), is_graph=is_graph, dp_is_decode=(1, 1))
        with forward_context(ctx):
            result = moe.forward(hidden)
        expected_rows = torch.arange(offset + 10, offset + local_tokens + 10, dtype=torch.float32)
        expected = expected_rows.view(-1, 1).expand(-1, 64)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(kernels.grouped_moe.call_args.args[0], gathered, rtol=0, atol=0)
        if is_graph:
            distributed.all_gather.assert_called_once()
            assert distributed.all_gather.call_args.kwargs == {"dim": 0, "world_size": 2, "group_name": "dp"}
            expected_input = torch.nn.functional.pad(hidden, (0, 0, 0, 4 - local_tokens))
            torch.testing.assert_close(distributed.all_gather.call_args.args[0], expected_input)
            distributed.all_gather_variable.assert_not_called()
        else:
            distributed.all_gather_variable.assert_called_once_with(hidden, [3, 4], dp_rank, "dp")
            distributed.all_gather.assert_not_called()
