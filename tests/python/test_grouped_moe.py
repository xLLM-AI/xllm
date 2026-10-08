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

"""Contracts for the NPU grouped MoE paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.kernels_npu import moe
from xllm.python.model_executor.forward_context import ForwardContext, forward_context


def test_unpermute_probs_keeps_fp32_without_a_dtype_copy() -> None:
    output = torch.empty(3, 16, dtype=torch.bfloat16)
    fp32_probs = torch.ones(3, 2, dtype=torch.float32)
    assert moe._unpermute_probs(fp32_probs, output) is fp32_probs

    bf16_probs = fp32_probs.to(torch.bfloat16)
    converted = moe._unpermute_probs(bf16_probs, output)
    assert converted is bf16_probs


@pytest.mark.parametrize("prob_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_moe_expert_compute_with_routing_preserves_metadata_without_unpermute(
    monkeypatch: pytest.MonkeyPatch,
    prob_dtype: torch.dtype,
) -> None:
    hidden = torch.empty(2, 4, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=prob_dtype)
    topk_ids = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    sorted_hidden = torch.empty(4, 4, dtype=torch.int8)
    sorted_indices = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    group_list = torch.tensor([1, 1, 1, 1], dtype=torch.int64)
    per_token_scale = torch.ones(4, dtype=torch.float32)
    permuted_output = torch.arange(16, dtype=torch.bfloat16).view(4, 4)
    weights = torch.empty(4, 4, 8, dtype=torch.int8)
    down_weights = torch.empty(4, 4, 4, dtype=torch.int8)
    weight_scale = torch.empty(4, 8, dtype=torch.float32)
    down_scale = torch.empty(4, 4, dtype=torch.bfloat16)

    monkeypatch.setattr(moe, "supports_fused_moe_gmm1", lambda device: False)
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_init_routing_v2",
        MagicMock(return_value=(sorted_hidden, sorted_indices, group_list, per_token_scale)),
    )
    monkeypatch.setattr(
        moe,
        "_grouped_matmul_swiglu_quant_v2",
        MagicMock(return_value=(torch.empty(4, 4, dtype=torch.int8), per_token_scale)),
    )
    monkeypatch.setattr(
        moe,
        "_grouped_matmul_gmm2",
        MagicMock(return_value=permuted_output),
    )
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_token_unpermute",
        MagicMock(side_effect=AssertionError("metadata path must not unpermute")),
    )

    output, probs, indices = moe.moe_expert_compute_with_routing(
        hidden,
        topk_weights,
        topk_ids,
        weights,
        down_weights,
        weight_scale,
        down_scale,
        2,
    )

    assert output is permuted_output
    if prob_dtype in (torch.float32, torch.bfloat16):
        assert probs is topk_weights
    else:
        assert torch.equal(probs, topk_weights.to(output.dtype))
    assert torch.equal(indices, sorted_indices)


def test_moe_routing_metadata_keeps_probabilities_during_fake_capture() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        hidden = torch.empty(2, 4, dtype=torch.bfloat16)
        probs = torch.empty(2, 2, dtype=torch.float32)
        ids = torch.empty(2, 2, dtype=torch.int32)
        output, retained_probs, indices = moe.moe_expert_compute_with_routing(
            hidden,
            probs,
            ids,
            torch.empty(4, 4, 8, dtype=torch.int8),
            torch.empty(4, 4, 4, dtype=torch.int8),
            torch.empty(4, 8, dtype=torch.float32),
            torch.empty(4, 4, dtype=torch.bfloat16),
            2,
        )
        assert retained_probs is probs
        assert output.shape == (4, 4)
        assert output.dtype == torch.bfloat16
        assert indices.shape == (4,)
        assert indices.dtype == torch.int32


@pytest.mark.parametrize("active_range", [None, [0, 4], [1, 3]])
def test_grouped_moe_retains_partial_expert_probability_mask(
    monkeypatch: pytest.MonkeyPatch,
    active_range: list[int] | None,
) -> None:
    hidden = torch.empty(2, 4, dtype=torch.bfloat16)
    probs = torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32)
    ids = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    permuted = torch.empty(4, 4, dtype=torch.bfloat16)
    indices = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_gating_top_k", MagicMock(return_value=(probs, ids, None)))
    monkeypatch.setattr(moe, "_moe_expert_compute_impl", MagicMock(return_value=(permuted, indices)))
    output, retained_probs, retained_indices = moe._grouped_moe_impl(
        hidden,
        torch.empty(2, 4),
        hidden,
        hidden,
        hidden,
        hidden,
        None,
        2,
        1,
        1,
        True,
        1.0,
        active_range,
    )
    assert output is permuted
    assert retained_indices is indices
    if active_range == [1, 3]:
        torch.testing.assert_close(retained_probs, torch.tensor([[0.2, 0.0], [0.7, 0.0]]))
    else:
        assert retained_probs is probs


@pytest.mark.parametrize("execution_path", ("grouped", "expert", "split"))
@pytest.mark.parametrize("routing_dtype", (torch.bfloat16, torch.float32))
def test_grouped_moe_uses_caller_prepared_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    execution_path: str,
    routing_dtype: torch.dtype,
) -> None:
    hidden = torch.ones(2, 8, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.1234, 0.8766], [0.5432, 0.4568]], dtype=routing_dtype)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    quantized = torch.zeros(4, 8, dtype=torch.int8)
    row_ids = torch.arange(4, dtype=torch.int32)
    expert_tokens = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    input_scale = torch.ones(4)
    activated = torch.zeros(4, 16, dtype=torch.int8)
    activation_scale = torch.ones(4)
    down_scale = torch.ones(4, 8, dtype=torch.bfloat16)
    expert_output = torch.zeros(4, 8, dtype=torch.bfloat16)
    expected = torch.zeros_like(hidden)
    group_matmul = MagicMock(return_value=[expert_output])
    for tensor in (down_scale, topk_ids, expert_tokens):
        monkeypatch.setattr(
            tensor,
            "to",
            MagicMock(side_effect=AssertionError("MoE must consume prepared scales and native routing dtypes")),
        )
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_gating_top_k",
        MagicMock(return_value=(topk_weights, topk_ids, None)),
        raising=False,
    )
    monkeypatch.setattr(
        moe.torch_npu,
        "npu_moe_init_routing_v2",
        MagicMock(return_value=(quantized, row_ids, expert_tokens, input_scale)),
        raising=False,
    )
    monkeypatch.setattr(moe, "_grouped_matmul_swiglu_quant_v2", MagicMock(return_value=(activated, activation_scale)))
    monkeypatch.setattr(torch.ops.npu, "npu_grouped_matmul", group_matmul, raising=False)
    token_unpermute = MagicMock(return_value=expected)
    monkeypatch.setattr(moe.torch_npu, "npu_moe_token_unpermute", token_unpermute, raising=False)

    w13 = torch.zeros(4, 8, 32, dtype=torch.int8)
    w2 = torch.zeros(4, 16, 8, dtype=torch.int8)
    w13_scale = torch.ones(4, 32)
    context = ForwardContext(None, torch.device("cpu"), SimpleNamespace(), [])
    with forward_context(context):
        if execution_path == "grouped":
            result = moe.grouped_moe(
                hidden,
                torch.zeros(2, 4, dtype=routing_dtype),
                w13,
                w2,
                w13_scale,
                down_scale,
                None,
                topk=2,
                topk_group=1,
                num_expert_groups=1,
                renormalize=True,
                routed_scaling_factor=1.0,
                expert_tokens_num_type=1,
                group_list_type=1,
            )
        elif execution_path == "expert":
            result = moe.moe_expert_compute(hidden, topk_weights, topk_ids, w13, w2, w13_scale, down_scale, 2)
        else:
            dispatched, indices, groups, input_scales = moe.moe_token_dispatch(hidden, topk_ids, 2, 4)
            activations, scales = moe.moe_gmm1(dispatched, w13, w13_scale, input_scales, groups)
            result = moe.moe_gmm2_combine(activations, scales, w2, down_scale, groups, indices, topk_weights)

    group_matmul.assert_called_once()
    assert group_matmul.call_args.kwargs["scale"][0] is down_scale
    assert group_matmul.call_args.kwargs["per_token_scale"][0] is activation_scale
    assert group_matmul.call_args.kwargs["group_list"] is expert_tokens
    assert token_unpermute.call_args.kwargs["probs"] is topk_weights
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("routing_dtype", (torch.bfloat16, torch.float32))
def test_selected_expert_moe_matches_native_call_contract(
    monkeypatch: pytest.MonkeyPatch,
    routing_dtype: torch.dtype,
) -> None:
    from xllm.python import kernels

    hidden = torch.empty(3, 16, dtype=torch.bfloat16)
    topk_weights = torch.tensor([[0.1234, 0.8766], [0.5432, 0.4568], [0.2345, 0.7655]], dtype=routing_dtype)
    topk_ids = torch.tensor([[4, 0], [5, 9], [7, 6]], dtype=torch.int32)
    monkeypatch.setattr(topk_ids, "to", MagicMock(side_effect=AssertionError("Routing IDs already use INT32")))
    expanded = torch.empty(6, 16, dtype=torch.bfloat16)
    row_ids = torch.arange(6, dtype=torch.int32)
    expert_tokens = torch.tensor([1, 3, 5, 6, 7, 8], dtype=torch.int64)
    quantized = torch.empty(6, 16, dtype=torch.int8)
    input_scale = torch.empty(6, dtype=torch.float32)
    gemm1 = torch.empty(6, 32, dtype=torch.int32)
    activated = torch.empty(6, 16, dtype=torch.int8)
    activation_scale = torch.empty(6, dtype=torch.float32)
    gemm2 = torch.empty(6, 16, dtype=torch.bfloat16)
    down_scale = torch.ones(4, 16, dtype=torch.bfloat16)
    monkeypatch.setattr(
        down_scale,
        "to",
        MagicMock(side_effect=AssertionError("MoE must not convert the caller-prepared down scale")),
    )
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
        down_scale,
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
    assert gemm_calls[1]["scale"] is down_scale
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
        topk_weights * torch.tensor([[True, False], [True, False], [True, True]]),
    )


def test_selected_expert_moe_rejects_an_invalid_active_range() -> None:
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

    with monkeypatch.context() as cast_guard:
        cast_guard.setattr(
            torch.Tensor,
            "to",
            MagicMock(side_effect=AssertionError("BF16 MoE must consume native routing dtypes without casts")),
        )
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


@pytest.mark.parametrize("noncontiguous", (False, True))
def test_encode_mega_moe_scale_uses_prepared_fp32_inputs(
    monkeypatch: pytest.MonkeyPatch,
    noncontiguous: bool,
) -> None:
    scale = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    offset = torch.zeros_like(scale)
    if noncontiguous:
        scale = scale.transpose(0, 1)
        offset = offset.transpose(0, 1)
    for tensor in (scale, offset):
        monkeypatch.setattr(tensor, "to", MagicMock(side_effect=AssertionError("MegaMoE inputs already use FP32")))
    encoded = torch.arange(6, dtype=torch.int64)
    trans_quant_param = MagicMock(return_value=encoded)
    monkeypatch.setattr(moe.torch_npu, "npu_trans_quant_param", trans_quant_param, raising=False)

    result = moe.encode_mega_moe_scale(scale, offset)

    actual_scale, actual_offset = trans_quant_param.call_args.args
    torch.testing.assert_close(actual_scale, scale.reshape(-1))
    torch.testing.assert_close(actual_offset, offset.reshape(-1))
    assert actual_scale.is_contiguous() and actual_offset.is_contiguous()
    assert trans_quant_param.call_args.kwargs["round_mode"] == 0
    torch.testing.assert_close(result, encoded.reshape(scale.shape))


@pytest.mark.parametrize("renormalize", (False, True))
def test_npu_softmax_topk_uses_graph_safe_native_op(
    monkeypatch: pytest.MonkeyPatch,
    renormalize: bool,
) -> None:
    logits = torch.zeros(2, 4, dtype=torch.bfloat16)
    weights = torch.tensor([[0.3, 0.2], [0.4, 0.1]], dtype=torch.bfloat16)
    expert_ids = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    native_topk = MagicMock(return_value=(weights, expert_ids))
    monkeypatch.setattr(
        torch.ops.xllm_ops,
        "moe_gating_top_k_softmax",
        native_topk,
        raising=False,
    )

    actual_weights, actual_ids = moe.moe_fused_topk(
        logits,
        2,
        renormalize,
        "softmax",
    )

    native_topk.assert_called_once_with(logits, 2, renormalize)
    torch.testing.assert_close(actual_weights, weights)
    torch.testing.assert_close(actual_ids, expert_ids)


def test_npu_softmax_topk_fake_preserves_bfloat16_dtype() -> None:
    from xllm.python.kernels_npu import _custom_op as custom_op

    logits = torch.empty((2, 4), dtype=torch.bfloat16, device="meta")

    weights, expert_ids = custom_op._moe_gating_top_k_softmax_fake(
        logits,
        topk=2,
        normalize=True,
    )

    assert weights.shape == (2, 2)
    assert weights.dtype == torch.bfloat16
    assert weights.device.type == "meta"
    assert expert_ids.shape == (2, 2)
    assert expert_ids.dtype == torch.int32
    assert expert_ids.device.type == "meta"


def test_grouped_matmul_swiglu_quant_v2_requests_int8_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        torch.empty(0, dtype=torch.int8),
        torch.empty(0, dtype=torch.float32),
    )
    calls: list[dict[str, object]] = []

    def grouped_matmul_swiglu_quant_v2(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        moe.torch_npu,
        "npu_grouped_matmul_swiglu_quant_v2",
        grouped_matmul_swiglu_quant_v2,
    )

    result = moe._grouped_matmul_swiglu_quant_v2(
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
    )

    assert result[0] is expected[0]
    assert result[1] is expected[1]
    assert calls[0]["quant_dtype"] == 1


def test_moe_weight_format_cast_enables_internal_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(allow_internal_format=False)
    monkeypatch.setattr(moe.torch, "npu", SimpleNamespace(config=config), raising=False)

    calls: list[tuple[torch.Tensor, int]] = []

    def format_cast(weight: torch.Tensor, acl_format: int) -> torch.Tensor:
        calls.append((weight, acl_format))
        return weight

    monkeypatch.setattr(moe.torch_npu, "npu_format_cast", format_cast)

    weight = torch.empty(2, 4, 8, dtype=torch.int8)
    assert moe.format_cast_nz(weight) is weight
    assert config.allow_internal_format is True
    assert calls == [(weight, 29)]


@pytest.mark.parametrize("group_list_type", [0, 2])
def test_gmm2_preserves_routing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    group_list_type: int,
) -> None:
    activations = torch.ones(3, 16, dtype=torch.int8)
    activation_scale = torch.ones(3, dtype=torch.float32)
    weight = torch.ones(4, 16, 32, dtype=torch.int8)
    weight_scale = torch.ones(4, 32, dtype=torch.bfloat16)
    # Include empty/non-consecutive experts to distinguish counts from offsets.
    groups = torch.tensor(
        [[1, 1], [3, 2]] if group_list_type == 2 else [0, 1, 1, 3],
        dtype=torch.int64,
    )
    expected = torch.full((3, 32), 7, dtype=torch.bfloat16)
    native_gmm = MagicMock(return_value=[expected])
    monkeypatch.setattr(moe.torch.ops.npu, "npu_grouped_matmul", native_gmm, raising=False)

    with forward_context(ForwardContext(None, activations.device, None, [])):
        actual = moe._grouped_matmul_gmm2(
            act_i8=activations,
            act_pertoken_scale=activation_scale,
            weight=weight,
            weight_scale=weight_scale,
            group_list=groups,
            group_list_type=group_list_type,
        )

    native_gmm.assert_called_once()
    kwargs = native_gmm.call_args.kwargs
    assert kwargs["group_list"] is groups
    assert kwargs["group_list_type"] == group_list_type
    assert kwargs["per_token_scale"][0] is activation_scale
    assert actual is expected
