# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU regression tests for shared normalization and output ownership.

Load the standalone validation library before collecting these tests, or use an
xLLM build that registers the ACLNN operator. Direct tests require that library.
"""

import gc

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.fixture(params=[torch.float16, torch.bfloat16])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


def _inputs(shape: tuple[int, int], dtype: torch.dtype) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    return [
        torch.randn(s, generator=generator, dtype=dtype).npu()
        for s in [shape, (shape[1],), (shape[1],), (shape[1],), (shape[1],)]
    ]


def _run(inputs: list[torch.Tensor], epsilon: float = 1e-5, direct: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    function = torch.ops.xllm_ops.dual_affine_layer_norm_direct if direct else torch.ops.xllm_ops.dual_affine_layer_norm
    return function(*inputs, epsilon)


def _check(inputs: list[torch.Tensor], outputs: tuple[torch.Tensor, torch.Tensor], epsilon: float) -> None:
    x, g1, b1, g2, b2 = [t.cpu().double() for t in inputs]
    centered = x - x.mean(-1, keepdim=True)
    norm = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + epsilon)
    tolerance = 2e-3 if inputs[0].dtype == torch.float16 else 2e-2
    pointers = {t.untyped_storage().data_ptr() for t in inputs}
    for output, gamma, beta in zip(outputs, (g1, g2), (b1, b2)):
        assert output.device == inputs[0].device
        assert output.dtype == inputs[0].dtype and output.is_contiguous()
        pointer = output.untyped_storage().data_ptr()
        assert pointer not in pointers
        pointers.add(pointer)
        torch.testing.assert_close(
            output.cpu().float(), (norm * gamma + beta).to(output.dtype).float(), atol=tolerance, rtol=tolerance
        )


@pytest.mark.parametrize(
    "shape", [(1, 1), (7, 1), (3, 7), (65, 65), (17, 127), (9, 1537), (128, 4096), (3, 8191), (512, 8192)]
)
@pytest.mark.parametrize("direct", [False, True])
def test_offsets_and_independent_outputs(dtype: torch.dtype, shape: tuple[int, int], direct: bool) -> None:
    inputs = _inputs(shape, dtype)
    inputs = [
        torch.empty(t.numel() + 16, device=t.device, dtype=t.dtype)[1 : 1 + t.numel()].view(t.shape).copy_(t)
        for t in inputs
    ]
    originals = [t.cpu().clone() for t in inputs]
    outputs = _run(inputs, direct=direct)
    _check(inputs, outputs, 1e-5)
    for tensor, original in zip(inputs, originals):
        assert torch.equal(tensor.cpu(), original)


@pytest.mark.parametrize("epsilon", [1e-7, 1e-5, 1e-2])
def test_constant_rows_return_beta_exactly(dtype: torch.dtype, epsilon: float) -> None:
    inputs = _inputs((3, 8192), dtype)
    inputs[0].fill_(100)
    for direct in [False, True]:
        outputs = _run(inputs, epsilon, direct)
        for output, beta in zip(outputs, (inputs[2], inputs[4])):
            assert torch.equal(output.cpu(), beta.cpu().expand_as(output.cpu()))


@pytest.mark.parametrize("direct", [False, True])
def test_stream_input_and_output_lifetimes(dtype: torch.dtype, direct: bool) -> None:
    inputs = _inputs((32, 1024), dtype)
    reference = [t.cpu() for t in inputs]
    ready = torch.npu.Event()
    ready.record()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        kept = [_run(inputs, direct=direct) for _ in range(8)]
    del inputs
    gc.collect()
    pressure = [torch.empty((32, 1024), device="npu", dtype=dtype).fill_(31) for _ in range(32)]
    stream.synchronize()
    pointers = [t.untyped_storage().data_ptr() for pair in kept for t in pair]
    assert len(set(pointers)) == 16
    inputs = [t.npu() for t in reference]
    for outputs in kept:
        _check(inputs, outputs, 1e-5)
    del pressure


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), -1, 0, 1e-8, 0.1])
def test_invalid_epsilon(dtype: torch.dtype, epsilon: float) -> None:
    inputs = _inputs((1, 768), dtype)
    for direct in [False, True]:
        with pytest.raises(RuntimeError, match="epsilon"):
            _run(inputs, epsilon, direct)
