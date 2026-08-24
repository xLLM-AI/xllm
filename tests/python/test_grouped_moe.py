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
