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

"""CUDA direct-Triton operator regression tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA Triton tests require CUDA", allow_module_level=True)

_PYTHON_ROOT = Path(__file__).parents[2] / "xllm" / "python"

# conftest.py stands in for xllm.python. Stand in for the CUDA kernel package
# too, so that its __init__ does not run while testing the Triton operators.
_kernels_package = types.ModuleType("xllm.python.kernels_cuda")
_kernels_package.__path__ = [str(_PYTHON_ROOT / "kernels_cuda")]
sys.modules["xllm.python.kernels_cuda"] = _kernels_package

from xllm.python.kernels_cuda.triton.causal_conv1d import (
    causal_conv1d_decode as kernel_causal_conv1d_decode,
)
from xllm.python.kernels_cuda.triton.causal_conv1d import (
    causal_conv1d_prefill as kernel_causal_conv1d_prefill,
)
from xllm.python.kernels_cuda.triton.fused_moe import fused_moe as kernel_fused_moe
from xllm.python.kernels_cuda.triton.gated_delta_net import (
    fused_recurrent_gated_delta_rule_packed_decode as kernel_gdn_decode,
)
from xllm.python.kernels_cuda.triton.gdn_prefill import (
    fused_gdn_prefill_post_conv as kernel_gdn_post_conv,
)
from xllm.python.kernels_cuda.triton.l2_norm import l2_norm as kernel_l2_norm

_CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@_CUDA_REQUIRED
def test_fused_moe_matches_stage_rounding_reference() -> None:
    torch.manual_seed(13)
    dtype = torch.bfloat16
    num_tokens, num_experts, hidden_size, intermediate_size, top_k = 3, 8, 128, 64, 3
    hidden = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
    w13 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, device="cuda", dtype=dtype)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device="cuda", dtype=dtype)
    topk_ids = torch.tensor([[0, 3, 7], [5, 2, 1], [6, 4, 0]], device="cuda", dtype=torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1).float()

    expected_rows: list[torch.Tensor] = []
    for token in range(num_tokens):
        expected = torch.zeros(hidden_size, device="cuda", dtype=torch.float32)
        for choice in range(top_k):
            expert = int(topk_ids[token, choice])
            w13_output = torch.mv(w13[expert].float(), hidden[token].float()).to(dtype)
            up, gate = w13_output.chunk(2)
            activated = (torch.nn.functional.silu(gate.float()).to(dtype) * up).to(dtype)
            w2_output = torch.mv(w2[expert].float(), activated.float())
            w2_output *= topk_weights[token, choice]
            expected += w2_output.to(dtype).float()
        expected_rows.append(expected.to(dtype))
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(
        kernel_fused_moe(hidden, topk_ids, topk_weights, w13, w2),
        expected,
        rtol=0,
        atol=0,
    )


def _causal_conv1d_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for sequence in range(slots.numel()):
        start = int(cu_seqlens[sequence])
        end = int(cu_seqlens[sequence + 1])
        slot = int(slots[sequence])
        use_history = has_initial_state is None or bool(has_initial_state[sequence])
        history = state[slot].float() if use_history else torch.zeros_like(state[slot], dtype=torch.float32)
        for token in value[start:end].float():
            window = torch.cat((history, token.unsqueeze(-1)), dim=-1)
            convolution = (window * weight.float()).sum(dim=-1)
            output.append(torch.nn.functional.silu(convolution).to(value.dtype))
            history = window[:, 1:]
        state[slot].copy_(history.to(state.dtype))
    return torch.stack(output)


@_CUDA_REQUIRED
@pytest.mark.parametrize("state_dim_first", [True, False])
def test_causal_conv1d_prefill_decode_matches_reference(
    state_dim_first: bool,
) -> None:
    torch.manual_seed(19)
    dtype = torch.bfloat16
    lengths = (5, 2, 7)
    cu_seqlens = torch.tensor([0, 5, 7, 14], device="cuda", dtype=torch.int64)
    slots = torch.tensor([2, 1, 3], device="cuda", dtype=torch.int32)
    channels, width = 96, 4
    value = torch.randn(sum(lengths), channels, device="cuda", dtype=dtype)
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
    reference_state = torch.randn(5, channels, width - 1, device="cuda", dtype=dtype)
    kernel_state = reference_state.clone() if state_dim_first else reference_state.transpose(1, 2).contiguous()

    expected = _causal_conv1d_reference(value, weight, reference_state, slots, cu_seqlens)
    has_initial_state = torch.ones(len(lengths), device="cuda", dtype=torch.bool)
    actual = kernel_causal_conv1d_prefill(value, weight, kernel_state, slots, has_initial_state, cu_seqlens)
    actual_state = kernel_state if state_dim_first else kernel_state.transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)

    decode_value = torch.randn(len(lengths), channels, device="cuda", dtype=dtype)
    decode_cu_seqlens = torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int64)
    expected = _causal_conv1d_reference(decode_value, weight, reference_state, slots, decode_cu_seqlens)
    actual = kernel_causal_conv1d_decode(decode_value, weight, kernel_state, slots)
    actual_state = kernel_state if state_dim_first else kernel_state.transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=4e-2)
    torch.testing.assert_close(actual_state, reference_state, rtol=0, atol=0)


@_CUDA_REQUIRED
def test_causal_conv1d_ops_preserve_null_slot() -> None:
    torch.manual_seed(23)
    value = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(2, 64, 3, device="cuda", dtype=torch.bfloat16)
    original_state = state.clone()
    slots = torch.tensor([1, 0], device="cuda", dtype=torch.int32)

    output = kernel_causal_conv1d_decode(value, weight, state, slots)

    torch.testing.assert_close(state[0], original_state[0], rtol=0, atol=0)
    assert torch.count_nonzero(output[1]) == 0


@_CUDA_REQUIRED
def test_fused_gdn_prefill_post_conv_matches_reference() -> None:
    torch.manual_seed(9)
    dtype = torch.bfloat16
    num_tokens, num_key_heads, num_value_heads = 37, 16, 32
    key_head_dim = value_head_dim = 128
    mixed_qkv = torch.randn(
        num_tokens,
        2 * num_key_heads * key_head_dim + num_value_heads * value_head_dim,
        device="cuda",
        dtype=dtype,
    )
    a = torch.randn(num_tokens, num_value_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_value_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_value_heads, device="cuda", dtype=dtype)

    actual = kernel_gdn_post_conv(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        num_key_heads,
        key_head_dim,
        value_head_dim,
    )

    expected_q, expected_k, expected_v = mixed_qkv.split(
        [
            num_key_heads * key_head_dim,
            num_key_heads * key_head_dim,
            num_value_heads * value_head_dim,
        ],
        dim=-1,
    )
    expected_q = kernel_l2_norm(expected_q.view(num_tokens, num_key_heads, key_head_dim).contiguous())
    expected_k = kernel_l2_norm(expected_k.view(num_tokens, num_key_heads, key_head_dim).contiguous())
    expected_v = expected_v.view(num_tokens, num_value_heads, value_head_dim).contiguous()
    softplus_input = a.float() + dt_bias.float()
    expected_softplus = torch.where(
        softplus_input > 0,
        softplus_input + torch.log1p(torch.exp(-softplus_input)),
        torch.log1p(torch.exp(softplus_input)),
    )
    expected_softplus = torch.where(softplus_input <= 20.0, expected_softplus, softplus_input)
    expected = (
        expected_q,
        expected_k,
        expected_v,
        -torch.exp(a_log) * expected_softplus,
        torch.sigmoid(b.float()),
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual[4], expected[4], rtol=2e-6, atol=2e-6)


@_CUDA_REQUIRED
def test_fused_gdn_decode_skips_null_slot_and_matches_torch() -> None:
    torch.manual_seed(31)
    dtype = torch.bfloat16
    batch, num_key_heads, num_value_heads = 2, 2, 4
    key_dim = value_dim = 128
    mixed_qkv = torch.randn(
        batch,
        2 * num_key_heads * key_dim + num_value_heads * value_dim,
        device="cuda",
        dtype=dtype,
    )
    a = torch.randn(batch, num_value_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_value_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_value_heads, device="cuda", dtype=dtype)
    state = torch.randn(
        3,
        num_value_heads,
        value_dim,
        key_dim,
        device="cuda",
        dtype=torch.float32,
    )
    original_state = state.clone()
    state_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    scale = key_dim**-0.5

    query, key, value = mixed_qkv[0].split(
        [
            num_key_heads * key_dim,
            num_key_heads * key_dim,
            num_value_heads * value_dim,
        ]
    )
    query = query.view(num_key_heads, key_dim).float()
    key = key.view(num_key_heads, key_dim).float()
    value = value.view(num_value_heads, value_dim).float()
    query /= torch.linalg.vector_norm(query, dim=-1, keepdim=True)
    key /= torch.linalg.vector_norm(key, dim=-1, keepdim=True)
    query = query.repeat_interleave(num_value_heads // num_key_heads, dim=0)
    key = key.repeat_interleave(num_value_heads // num_key_heads, dim=0)
    expected_state = original_state[2].clone()
    decay = -torch.exp(a_log) * torch.nn.functional.softplus(a[0].float() + dt_bias.float())
    expected_state *= torch.exp(decay)[:, None, None]
    delta = value - torch.einsum("hvk,hk->hv", expected_state, key)
    delta *= torch.sigmoid(b[0]).float()[:, None]
    expected_state += torch.einsum("hv,hk->hvk", delta, key)
    expected_output = torch.einsum("hvk,hk->hv", expected_state, query * scale).to(dtype)

    actual = kernel_gdn_decode(
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_indices,
        scale,
    )

    torch.testing.assert_close(actual[0, 0], expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state[2], expected_state, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state[0], original_state[0], rtol=0, atol=0)
    assert torch.count_nonzero(actual[1]) == 0
