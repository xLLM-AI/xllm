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

"""Dispatch and semantic tests for GLM-5.3-Flash RMS normalization."""

from __future__ import annotations

import pytest
import torch

import xllm.python.models.glm5_next as glm5_next


def test_rms_norm_uses_vendor_rms_norm_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = glm5_next.Glm5NextRMSNorm(
        hidden_size=8,
        eps=1e-5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    value = torch.randn(1, 6, 8)
    expected = torch.empty_like(value)
    calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def _rms_norm(
        input_value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        calls.append((input_value, weight, eps))
        return expected

    monkeypatch.setattr(glm5_next.kernels, "rms_norm", _rms_norm, raising=False)

    output = layer(value)

    assert output is expected
    assert len(calls) == 1
    assert calls[0][0] is value
    assert calls[0][1] is layer.weight
    assert calls[0][2] == layer.variance_epsilon


def test_gated_rms_norm_uses_sigmoid_fused_kernel_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = glm5_next._RMSNormGated(
        hidden_size=4,
        eps=1e-5,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    value = torch.randn(1, 6, 2, 4)
    gate = torch.randn_like(value)
    expected = torch.empty_like(value)
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    def _rms_norm_sigmoid_gated(
        input_value: torch.Tensor,
        input_gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        calls.append((input_value, input_gate, weight, eps))
        return expected

    def _unexpected_silu_kernel(*args: object) -> None:
        del args
        raise AssertionError("GLM sigmoid gate must not use the SiLU-gated op")

    monkeypatch.setattr(
        glm5_next.kernels,
        "rms_norm_sigmoid_gated",
        _rms_norm_sigmoid_gated,
        raising=False,
    )
    monkeypatch.setattr(
        glm5_next.kernels,
        "rms_norm_gated",
        _unexpected_silu_kernel,
        raising=False,
    )
    output = layer(value, gate)

    assert output is expected
    assert len(calls) == 1
    assert calls[0][0] is value
    assert calls[0][1] is gate
    assert calls[0][2] is layer.weight
    assert calls[0][3] == layer.variance_epsilon


def test_unweighted_rms_norm_pure_torch_semantics() -> None:
    layer = glm5_next._UnweightedRMSNorm(eps=1e-5)
    value = torch.randn(1, 6, 8)

    output = layer(value)
    value_fp32 = value.float()
    expected = (value_fp32 * torch.rsqrt(value_fp32.square().mean(dim=-1, keepdim=True) + layer.variance_epsilon)).to(
        value.dtype
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
