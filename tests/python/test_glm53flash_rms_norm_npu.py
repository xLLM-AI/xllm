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

"""NPU tests for the GLM-5.3-Flash sigmoid-gated RMSNorm kernel."""

from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip(
    "torch_npu",
    reason="GLM-5.3-Flash fused RMSNorm tests require torch_npu",
)

from xllm.python.kernels_npu.triton.rms_norm import (  # noqa: E402
    rms_norm_sigmoid_gated,
)


def _rms_fp32(value: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMS-normalized value (no weight, no gate, no down-cast)."""
    value_fp32 = value.float()
    return value_fp32 * torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + eps)


def _reference(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return (_rms_fp32(value, eps) * weight.float() * torch.sigmoid(gate.float())).to(value.dtype)


@pytest.mark.parametrize("seq_len", [1, 2, 6, 8])
def test_sigmoid_gated_rms_norm_is_batch_invariant(seq_len: int) -> None:
    torch_npu.npu.set_device(0)
    torch.manual_seed(2026)
    eps = 1e-5
    base_value = torch.randn(
        1,
        1,
        8,
        128,
        dtype=torch.bfloat16,
        device="npu",
    )
    base_gate = torch.randn_like(base_value)
    weight = torch.randn(128, dtype=torch.bfloat16, device="npu")

    single = rms_norm_sigmoid_gated(
        base_value,
        base_gate,
        weight,
        eps,
    )
    value = base_value.expand(1, seq_len, -1, -1).contiguous()
    gate = base_gate.expand_as(value).contiguous()
    output = rms_norm_sigmoid_gated(value, gate, weight, eps)
    expected = _reference(value, gate, weight, eps)
    torch_npu.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    for token_index in range(seq_len):
        assert torch.equal(
            output[:, token_index].cpu(),
            single[:, 0].cpu(),
        )


def test_sigmoid_gated_rms_norm_does_not_apply_silu_gate() -> None:
    torch_npu.npu.set_device(0)
    eps = 1e-5
    value = torch.ones(1, 8, 128, dtype=torch.bfloat16, device="npu")
    gate = torch.zeros_like(value)
    weight = torch.ones(128, dtype=torch.bfloat16, device="npu")

    output = rms_norm_sigmoid_gated(value, gate, weight, eps)
    expected = _reference(value, gate, weight, eps)
    torch_npu.npu.synchronize()

    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
    assert torch.count_nonzero(output).item() == output.numel()
