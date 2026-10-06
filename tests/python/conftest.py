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

"""Initialize the real native operators and Python kernels before collection."""

import pytest
import torch

from xllm import xllm_export  # noqa: F401
from xllm.python import initialize_runtime

initialize_runtime()


def _rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CPU reference for weighted RMSNorm over the last dimension."""
    value_fp32 = value.float()
    variance = value_fp32.square().mean(dim=-1, keepdim=True)
    normalized = value_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(value.dtype)


def _rms_norm_sigmoid_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """CPU reference for kernels_npu rms_norm_sigmoid_gated.

    Matches the Triton kernel contract: RMSNorm over the last dim, scaled by
    ``weight`` and gated by ``sigmoid(gate)``. Used by pure-Python model
    tests (glm5_next KDA o_norm) that cannot link the NPU kernels.
    """
    input_dtype = value.dtype
    x = value.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.to(torch.float32) * gate.sigmoid()).to(input_dtype)


@pytest.fixture(autouse=True)
def _glm_cpu_normalization(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use CPU references only in GLM unit tests; keep real NPU dispatch."""
    if request.module.__name__.rsplit(".", 1)[-1] not in {
        "test_glm5_next_kda",
        "test_glm5_next_dsa",
        "test_glm53flash_rms_norm",
    }:
        return
    from xllm.python import kernels

    native_rms_norm = kernels.rms_norm
    native_gated_rms_norm = kernels.rms_norm_sigmoid_gated

    def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        operation = _rms_norm if value.device.type == "cpu" else native_rms_norm
        return operation(value, weight, eps)

    def gated_rms_norm(value: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        operation = _rms_norm_sigmoid_gated if value.device.type == "cpu" else native_gated_rms_norm
        return operation(value, gate, weight, eps)

    monkeypatch.setattr(kernels, "rms_norm", rms_norm)
    monkeypatch.setattr(kernels, "rms_norm_sigmoid_gated", gated_rms_norm)
