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

"""Contracts for the NPU pre-selected grouped MoE path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[2]


def _load_npu_moe_module():
    path = _REPO_ROOT / "xllm/python/kernels_npu/moe.py"
    spec = importlib.util.spec_from_file_location("pr5_npu_moe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_expert_moe_matches_native_call_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python import kernels

    moe = _load_npu_moe_module()

    hidden = torch.empty(3, 16, dtype=torch.bfloat16)
    topk_weights = torch.ones(3, 2, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[4, 0], [5, 9], [7, 6]], dtype=torch.int32)
    expanded = torch.empty(6, 16, dtype=torch.bfloat16)
    row_ids = torch.arange(6, dtype=torch.int32)
    expert_tokens = torch.tensor([1, 3, 5, 6, 7, 8], dtype=torch.int64)
    quantized = torch.empty(6, 16, dtype=torch.int8)
    input_scale = torch.empty(6, dtype=torch.float32)
    gemm1 = torch.empty(6, 32, dtype=torch.int32)
    activated = torch.empty(6, 16, dtype=torch.int8)
    activation_scale = torch.empty(6, dtype=torch.float32)
    gemm2 = torch.empty(6, 16, dtype=torch.bfloat16)
    calls: list[tuple[str, object]] = []

    def init_routing(*args, **kwargs):
        calls.append(("routing", kwargs))
        return expanded, row_ids, expert_tokens, torch.empty(0)

    def dynamic_quant(value):
        assert value is expanded
        calls.append(("dynamic_quant", value))
        return quantized, input_scale

    def dequant_swiglu_quant(**kwargs):
        calls.append(("dequant_swiglu_quant", kwargs))
        return activated, activation_scale

    gemm_calls: list[dict[str, object]] = []

    def group_gemm(**kwargs):
        gemm_calls.append(kwargs)
        return gemm1 if len(gemm_calls) == 1 else gemm2

    def token_unpermute(**kwargs):
        calls.append(("unpermute", kwargs))
        return hidden

    monkeypatch.setattr(moe, "_group_gemm", group_gemm)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_init_routing_v2", init_routing)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_token_unpermute", token_unpermute)
    monkeypatch.setattr(kernels, "dynamic_quant", dynamic_quant, raising=False)
    monkeypatch.setattr(kernels, "dequant_swiglu_quant", dequant_swiglu_quant, raising=False)

    result = moe._grouped_moe_with_selected_experts_impl(
        hidden,
        topk_weights,
        topk_ids,
        torch.empty(4, 16, 32, dtype=torch.int8),
        torch.empty(4, 16, 16, dtype=torch.int8),
        torch.empty(4, 32),
        torch.empty(4, 16),
        num_total_experts=16,
        start_expert_id=4,
        num_experts_per_rank=4,
        swiglu_limit=7.0,
    )

    assert result is hidden
    routing = dict(calls)["routing"]
    assert isinstance(routing, dict)
    assert routing["active_expert_range"] == [4, 8]
    assert routing["expert_num"] == 16
    assert routing["quant_mode"] == -1

    assert len(gemm_calls) == 2
    assert gemm_calls[0]["scale"] is None
    assert gemm_calls[0]["per_token_scale"] is None
    assert gemm_calls[0]["output_dtype"] == torch.int32
    assert gemm_calls[1]["scale"].dtype == torch.bfloat16
    assert gemm_calls[1]["per_token_scale"] is activation_scale
    assert gemm_calls[1]["output_dtype"] == torch.bfloat16
    assert all(torch.equal(call["group_list"], expert_tokens[:4]) for call in gemm_calls)
    assert all(call["group_list"].numel() == 4 for call in gemm_calls)
    assert all(call["group_list_type"] == 1 for call in gemm_calls)

    dequant = dict(calls)["dequant_swiglu_quant"]
    assert isinstance(dequant, dict)
    assert dequant["x"] is gemm1
    assert dequant["activation_scale"] is input_scale
    assert torch.equal(dequant["group_index"], expert_tokens[:4])
    assert dequant["clamp_limit"] == 7.0

    unpermute = dict(calls)["unpermute"]
    assert isinstance(unpermute, dict)
    torch.testing.assert_close(
        unpermute["probs"],
        torch.tensor([[1, 0], [1, 0], [1, 1]], dtype=torch.bfloat16),
    )


def test_selected_expert_moe_rejects_an_invalid_active_range() -> None:
    moe = _load_npu_moe_module()

    with pytest.raises(ValueError, match="active expert range"):
        moe._grouped_moe_with_selected_experts_impl(
            torch.empty(1, 16, dtype=torch.bfloat16),
            torch.ones(1, 1, dtype=torch.bfloat16),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.empty(4, 16, 32, dtype=torch.int8),
            torch.empty(4, 16, 16, dtype=torch.int8),
            torch.empty(4, 32),
            torch.empty(4, 16),
            num_total_experts=16,
            start_expert_id=14,
            num_experts_per_rank=4,
        )


def test_qwen35_bf16_grouped_moe_uses_native_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xllm.python import kernels

    moe = _load_npu_moe_module()

    hidden = torch.empty(3, 16, dtype=torch.bfloat16)
    topk_weights = torch.ones(3, 2, dtype=torch.bfloat16)
    topk_ids = torch.tensor([[4, 0], [5, 9], [7, 6]], dtype=torch.int32)
    w13 = torch.empty(4, 16, 32, dtype=torch.bfloat16)
    w2 = torch.empty(4, 16, 16, dtype=torch.bfloat16)
    expanded = torch.empty(6, 16, dtype=torch.bfloat16)
    row_ids = torch.arange(6, dtype=torch.int32)
    expert_tokens = torch.tensor([1, 3, 5, 6], dtype=torch.int64)
    gate_up = torch.cat(
        (
            torch.zeros(6, 16, dtype=torch.bfloat16),
            torch.ones(6, 16, dtype=torch.bfloat16),
        ),
        dim=-1,
    )
    expert_output = torch.empty(6, 16, dtype=torch.bfloat16)
    expected = torch.empty_like(hidden)
    init_routing = MagicMock(return_value=(expanded, row_ids, expert_tokens, torch.empty(0)))
    group_gemm = MagicMock(side_effect=(gate_up, expert_output))
    token_unpermute = MagicMock(return_value=expected)
    silu_and_mul = MagicMock(return_value=torch.zeros(6, 16, dtype=torch.bfloat16))
    monkeypatch.setattr(moe, "_group_gemm", group_gemm)
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_init_routing_v2",
        init_routing,
    )
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_token_unpermute",
        token_unpermute,
    )
    monkeypatch.setattr(kernels, "silu_and_mul", silu_and_mul, raising=False)

    result = moe.grouped_moe_bf16(
        hidden,
        topk_weights,
        topk_ids,
        w13,
        w2,
        16,
        4,
        4,
    )

    assert result is expected
    assert init_routing.call_args.kwargs["active_expert_range"] == [4, 8]
    assert init_routing.call_args.kwargs["quant_mode"] == -1
    assert group_gemm.call_count == 2
    first_gemm = group_gemm.call_args_list[0].kwargs
    second_gemm = group_gemm.call_args_list[1].kwargs
    assert first_gemm["weight"] is w13
    assert second_gemm["weight"] is w2
    silu_and_mul.assert_called_once_with(gate_up)
    assert first_gemm["group_list_type"] == 1
    assert second_gemm["group_list_type"] == 1
    torch.testing.assert_close(
        second_gemm["x"],
        torch.zeros(6, 16, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        token_unpermute.call_args.kwargs["probs"],
        torch.tensor([[1, 0], [1, 0], [1, 1]], dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(("renormalize", "expected_renorm"), ((False, 0), (True, 1)))
def test_npu_softmax_topk_uses_graph_safe_native_op(
    monkeypatch: pytest.MonkeyPatch,
    renormalize: bool,
    expected_renorm: int,
) -> None:
    moe = _load_npu_moe_module()
    logits = torch.zeros(2, 4, dtype=torch.bfloat16)
    weights = torch.tensor([[0.3, 0.2], [0.4, 0.1]], dtype=torch.bfloat16)
    expert_ids = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    native_topk = MagicMock(return_value=(weights, expert_ids, torch.empty_like(expert_ids)))
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_gating_top_k_softmax_v2",
        native_topk,
    )

    actual_weights, actual_ids = moe.moe_fused_topk(
        logits,
        2,
        renormalize,
        "softmax",
    )

    native_topk.assert_called_once_with(
        logits,
        k=2,
        finished=None,
        renorm=expected_renorm,
        output_softmax=False,
    )
    torch.testing.assert_close(actual_weights, weights)
    torch.testing.assert_close(actual_ids, expert_ids)
