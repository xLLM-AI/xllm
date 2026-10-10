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

"""Native RMSNorm adapter: GLM-5.3 shapes, aliases, fake, streams and ACLGraph."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from xllm_kernel import initialize, prepare_rms_norm
from xllm_kernel.ops import rms_norm

from xllm.python import initialize_runtime, kernels
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_npu(), reason="NPU native adapter")


def _reference(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    value_fp32 = value.cpu().float()
    return value_fp32 * torch.rsqrt(value_fp32.square().mean(-1, keepdim=True) + eps) * weight.cpu().float()


def _check_reference(output: torch.Tensor, value: torch.Tensor, weight: torch.Tensor) -> None:
    tolerance = {torch.bfloat16: 2e-2, torch.float16: 2e-3, torch.float32: 2e-5}[value.dtype]
    torch.testing.assert_close(output.cpu().float(), _reference(value, weight), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [512, 2048, 6144])
@pytest.mark.parametrize("rows", [1, 17, 128])
def test_matches_native_and_reference_without_mutation(dtype: torch.dtype, width: int, rows: int) -> None:
    torch.manual_seed(5309)
    value = torch.randn(rows, width, dtype=dtype, device="npu")
    weight = torch.randn(width, dtype=dtype, device="npu")
    before, weight_before = value.clone(), weight.clone()
    output = rms_norm(value, weight, 1e-5)
    expected = torch.ops.xllm_ops.rms_norm(value, weight, 1e-5)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(value, before, rtol=0, atol=0)
    torch.testing.assert_close(weight, weight_before, rtol=0, atol=0)
    assert output.data_ptr() not in (value.data_ptr(), weight.data_ptr())
    _check_reference(output, value, weight)


@pytest.mark.parametrize("shape", [(512,), (2, 3, 512), (0, 512)])
def test_native_boundary_shapes(shape: tuple[int, ...]) -> None:
    value = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(shape[-1], dtype=value.dtype, device="npu")
    torch.testing.assert_close(
        rms_norm(value, weight, 1e-5), torch.ops.xllm_ops.rms_norm(value, weight, 1e-5), rtol=0, atol=0
    )


def test_strided_glm_projection_view() -> None:
    value = torch.randn(17, 576, dtype=torch.bfloat16, device="npu")[:, :512]
    weight = torch.randn(1024, dtype=value.dtype, device="npu")[::2]
    assert not value.is_contiguous() and not weight.is_contiguous()
    output = rms_norm(value, weight, 1e-5)
    torch.testing.assert_close(output, torch.ops.xllm_ops.rms_norm(value, weight, 1e-5), rtol=0, atol=0)
    _check_reference(output, value, weight)


def test_legacy_shim_and_glm_shared_layer_use_same_binding() -> None:
    from xllm.python.kernels_npu import mla, normalization

    assert kernels.rms_norm is rms_norm
    assert normalization.rms_norm is rms_norm
    assert mla.rms_norm is rms_norm
    layer = RMSNorm(6144, 1e-5, dtype=torch.bfloat16, device="npu")
    value = torch.randn(3, 6144, dtype=torch.bfloat16, device="npu")
    torch.testing.assert_close(layer(value), torch.ops.xllm_ops.rms_norm(value, layer.weight, 1e-5), rtol=0, atol=0)


def test_initialization_is_idempotent_and_policy_is_fixed() -> None:
    plan = prepare_rms_norm()
    initialize_runtime()
    assert initialize(device="npu") is plan
    assert plan.function is torch.ops.xllm_ops.rms_norm
    with pytest.raises(RuntimeError, match="restart"):
        initialize(device="cuda")
    with pytest.raises(RuntimeError, match="restart"):
        initialize(device="npu", implementation="missing")
    importlib.import_module("xllm.python.kernels_npu._custom_op")
    assert prepare_rms_norm() is plan


def test_fake_preserves_existing_native_contract() -> None:
    with FakeTensorMode():
        value = torch.empty(7, 576, dtype=torch.bfloat16, device="npu")[:, :512]
        weight = torch.empty(512, dtype=value.dtype, device="npu")
        output = rms_norm(value, weight, 1e-5)
        expected = torch.ops.xllm_ops.rms_norm(value, weight, 1e-5)
        assert output is not value and output is not weight
        assert output._base is None
        assert (output.shape, output.dtype, output.device, output.stride()) == (
            expected.shape,
            expected.dtype,
            expected.device,
            expected.stride(),
        )


def test_nondefault_stream_producer_and_consumer() -> None:
    stream = torch.npu.Stream()
    results = []
    with torch.npu.stream(stream):
        weight = torch.randn(6144, dtype=torch.bfloat16, device="npu")
        for step in range(8):
            value = torch.randn(17, 6144, dtype=weight.dtype, device="npu").mul(0.25).add(step / 8)
            results.append((value, rms_norm(value, weight, 1e-5).mul(2)))
    torch.npu.current_stream().wait_stream(stream)
    for value, output in results:
        _check_reference(output / 2, value, weight)


@pytest.mark.parametrize("width", [512, 2048, 6144])
def test_aclgraph_replay_updates_inputs_and_keeps_slots_independent(width: int) -> None:
    entries = []
    for rows in (1, 4, 16):
        value = torch.randn(rows, width, dtype=torch.bfloat16, device="npu")
        weight = torch.randn(width, dtype=value.dtype, device="npu")
        for _ in range(3):
            rms_norm(value + 0.125, weight, 1e-5)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = rms_norm(value + 0.125, weight, 1e-5) * 2
        entries.append((value, weight, graph, output))
    for step in range(3):
        for index in (2, 0, 1, 0, 2):
            value, weight, graph, output = entries[index]
            value.copy_(torch.randn_like(value) + step)
            weight.copy_(torch.randn_like(weight))
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                output, torch.ops.xllm_ops.rms_norm(value + 0.125, weight, 1e-5) * 2, rtol=0, atol=0
            )
            _check_reference(output / 2, value + 0.125, weight)
