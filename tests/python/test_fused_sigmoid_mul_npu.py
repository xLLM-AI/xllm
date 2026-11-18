# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Real-device coverage for output gating, view addressing and graph replay."""

from __future__ import annotations

import pytest
import torch

from xllm.python import kernels

pytestmark = pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU required")


def _case(
    tokens: int, heads: int, head_dim: int, dtype: torch.dtype, layout: str, device: str = "npu:0"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    width = heads * head_dim
    x_storage = torch.randn(tokens, width + 32, device=device, dtype=dtype)
    x = x_storage[:, 16 : width + 16]
    # Include K/V columns in the row stride, and a nonzero storage offset.
    qkv = torch.randn(tokens, 2 * width + 2 * head_dim + 32, device=device, dtype=dtype)
    if layout == "interleaved":
        gate = qkv[:, 16 : 2 * width + 16].view(tokens, heads, 2 * head_dim)[:, :, head_dim:]
    else:
        gate = qkv[:, width + 16 : 2 * width + 16]
        if layout == "contiguous":
            gate = gate.contiguous()
    return x, gate, x_storage, qkv


def _assert_close(actual: torch.Tensor, reference: torch.Tensor) -> None:
    tolerance = (1e-2, 2e-2) if actual.dtype == torch.bfloat16 else (2e-3, 2e-3)
    torch.testing.assert_close(actual, reference, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["contiguous", "grouped", "interleaved"])
@pytest.mark.parametrize(
    "tokens,heads,head_dim",
    [
        (0, 4, 128),
        (1, 4, 17),
        (2, 4, 256),
        (3, 3, 33),
        (16, 32, 256),
        (65, 4, 1025),
        (1024, 8, 256),
        (1, 12, 256),
        (8, 12, 256),
        (16, 12, 256),
        (128, 12, 256),
        (512, 12, 256),
        (1024, 12, 256),
    ],
)
def test_values_strides_and_no_mutation(
    tokens: int, heads: int, head_dim: int, dtype: torch.dtype, layout: str
) -> None:
    torch.manual_seed(42)
    x, gate, x_storage, qkv = _case(tokens, heads, head_dim, dtype, layout)
    saved_x, saved_qkv, saved_gate = x_storage.clone(), qkv.clone(), gate.clone()
    reference = x * torch.sigmoid(gate.reshape(tokens, heads * head_dim))
    output = kernels.fused_sigmoid_mul(x, gate)
    _assert_close(output, reference)
    assert output.is_contiguous()
    assert output.device == x.device and output.dtype == x.dtype
    if tokens:
        assert output.data_ptr() != x.data_ptr()
        assert output.data_ptr() != gate.data_ptr()
    torch.testing.assert_close(x_storage, saved_x, rtol=0, atol=0)
    torch.testing.assert_close(qkv, saved_qkv, rtol=0, atol=0)
    torch.testing.assert_close(gate, saved_gate, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_reference_layout_and_dtype_paths(dtype: torch.dtype) -> None:
    x = torch.randn(3, 514, device="npu", dtype=dtype)[:, ::2]
    gate = torch.randn_like(x.T).T
    _assert_close(kernels.fused_sigmoid_mul(x, gate), x * torch.sigmoid(gate))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_saturation_and_zero(dtype: torch.dtype) -> None:
    gate = torch.tensor([-100, -20, -1, -0.01, 0, 0.01, 1, 20, 100], device="npu", dtype=dtype).repeat(5, 1)
    x = torch.linspace(-8, 8, 45, device="npu", dtype=dtype).reshape(5, 9)
    _assert_close(kernels.fused_sigmoid_mul(x, gate), x * torch.sigmoid(gate))
    _assert_close(kernels.fused_sigmoid_mul(x, torch.zeros_like(gate)), x * 0.5)


@pytest.mark.parametrize("layout", ["contiguous", "grouped", "interleaved"])
@pytest.mark.parametrize("heads", [4, 12])
def test_graph_replay_reads_new_inputs(layout: str, heads: int) -> None:
    x, gate, _, _ = _case(16, heads, 256, torch.bfloat16, layout)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            kernels.fused_sigmoid_mul(x, gate)
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        result = kernels.fused_sigmoid_mul(x, gate)
    for step in range(5):
        x.fill_(step - 2)
        gate.fill_(step * 0.25 - 1)
        graph.replay()
        torch.npu.synchronize()
        _assert_close(result, x * torch.sigmoid(gate.reshape_as(x)))


def test_current_stream_and_device_guard() -> None:
    device = "npu:1" if torch.npu.device_count() > 1 else "npu:0"
    x, gate, _, _ = _case(7, 4, 256, torch.bfloat16, "interleaved", device)
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        x.fill_(3)
        result = kernels.fused_sigmoid_mul(x, gate)
        reference = x * torch.sigmoid(gate.reshape_as(x))
    stream.synchronize()
    _assert_close(result, reference)


def test_operator_schema_alias_contract() -> None:
    x, gate, _, _ = _case(3, 4, 256, torch.bfloat16, "interleaved")
    torch.library.opcheck(
        torch.ops.xllm_python.fused_sigmoid_mul.default,
        (x, gate),
        test_utils=("test_schema",),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nonfinite_values(dtype: torch.dtype) -> None:
    x = torch.tensor([[0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")]], device="npu", dtype=dtype)
    gate = torch.tensor([[-float("inf"), float("inf"), float("nan"), 0.0, 0.0, 1.0]], device="npu", dtype=dtype)
    actual = kernels.fused_sigmoid_mul(x, gate)
    reference = x * torch.sigmoid(gate)
    torch.testing.assert_close(actual, reference, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_three_dimensional_gate(dtype: torch.dtype) -> None:
    x = torch.randn(8, 8192, device="npu", dtype=dtype)
    gate = torch.randn(8, 32, 256, device="npu", dtype=dtype)
    _assert_close(kernels.fused_sigmoid_mul(x, gate), x * torch.sigmoid(gate.view_as(x)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [1024, 2048, 4096, 8192])
@pytest.mark.parametrize("layout", ["contiguous", "grouped"])
def test_row_coalescing_and_tp_widths(dtype: torch.dtype, width: int, layout: str) -> None:
    x = torch.randn(127, width, device="npu", dtype=dtype)
    qkv = torch.randn(127, 2 * width + 256, device="npu", dtype=dtype)
    gate = qkv[:, width : 2 * width]
    if layout == "contiguous":
        gate = gate.contiguous()
    _assert_close(kernels.fused_sigmoid_mul(x, gate), x * torch.sigmoid(gate))


@pytest.mark.parametrize("heads", [4, 8, 16, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_interleaved_tile_sizes(heads: int, head_dim: int) -> None:
    x, gate, _, _ = _case(128, heads, head_dim, torch.bfloat16, "interleaved")
    _assert_close(kernels.fused_sigmoid_mul(x, gate), x * torch.sigmoid(gate.reshape_as(x)))


def test_large_packed_buffer_uses_final_rounding() -> None:
    x = torch.randn(256, 8192, device="npu", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    actual = kernels.fused_sigmoid_mul(x, gate)
    _assert_close(actual, (x.float() * torch.sigmoid(gate.float())).to(x.dtype))


def test_internal_storage_format_is_converted_to_nd() -> None:
    import torch_npu

    x = torch.randn(32, 256, device="npu", dtype=torch.float16)
    gate = torch.randn_like(x)
    x_nz = torch_npu.npu_format_cast(x, 29)
    gate_nz = torch_npu.npu_format_cast(gate, 29)
    _assert_close(kernels.fused_sigmoid_mul(x_nz, gate_nz), x * torch.sigmoid(gate))


@pytest.mark.parametrize("heads", [4, 8, 16])
def test_large_strided_gate_uses_fp32_intermediates(heads: int) -> None:
    x, gate, _, _ = _case(2048, heads, 256, torch.bfloat16, "interleaved")
    _assert_close(
        kernels.fused_sigmoid_mul(x, gate), (x.float() * torch.sigmoid(gate.float().reshape_as(x))).to(x.dtype)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 8, 128, 129, 512, 1024])
@pytest.mark.parametrize("gate_rank", [2, 3])
def test_every_gate_batch_calls_triton_kernel(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, tokens: int, gate_rank: int
) -> None:
    from xllm.python.kernels_npu.triton import fused_sigmoid_mul as implementation

    x = torch.randn(tokens, 3072, device="npu", dtype=dtype)
    backing = torch.randn(tokens, 6656, device="npu", dtype=dtype)
    gate = backing[:, 3072:6144]
    if gate_rank == 3:
        gate = gate.view(tokens, 12, 256)
    saved = backing.clone()
    expected = (x.float() * torch.sigmoid(gate.float().reshape_as(x))).to(dtype)
    original = implementation.fused_sigmoid_mul
    calls: list[tuple[torch.Size, torch.Size]] = []

    def record_kernel(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        calls.append((value.shape, weight.shape))
        return original(value, weight)

    monkeypatch.setattr(implementation, "fused_sigmoid_mul", record_kernel)
    actual = kernels.fused_sigmoid_mul(x, gate)
    _assert_close(actual, expected)
    assert calls == [(gate.shape, gate.shape)]
    torch.testing.assert_close(backing, saved, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_single_rounding_matches_fp32_reference(dtype: torch.dtype) -> None:
    x = torch.full((1, 1), 3.0, device="npu", dtype=dtype)
    gate = torch.full_like(x, 0.3)
    expected = (x.float() * torch.sigmoid(gate.float())).to(dtype)
    actual = kernels.fused_sigmoid_mul(x, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, x * torch.sigmoid(gate))


def test_python_custom_op_is_registered() -> None:
    assert hasattr(torch.ops.xllm_python, "fused_sigmoid_mul")


@pytest.mark.parametrize(
    "input_shape,gate_shape,input_dtype,gate_dtype",
    [
        ((3,), (3,), torch.bfloat16, torch.bfloat16),
        ((2, 4), (2, 1, 1, 4), torch.bfloat16, torch.bfloat16),
        ((2, 4), (3, 4), torch.bfloat16, torch.bfloat16),
        ((2, 4), (2, 3), torch.bfloat16, torch.bfloat16),
        ((2, 0), (2, 0), torch.bfloat16, torch.bfloat16),
        ((2, 4), (2, 4), torch.int32, torch.int32),
        ((2, 4), (2, 4), torch.bfloat16, torch.float16),
    ],
)
def test_invalid_gate_is_rejected_before_launch(
    input_shape: tuple[int, ...],
    gate_shape: tuple[int, ...],
    input_dtype: torch.dtype,
    gate_dtype: torch.dtype,
) -> None:
    x = torch.zeros(input_shape, device="npu", dtype=input_dtype)
    gate = torch.zeros(gate_shape, device="npu", dtype=gate_dtype)
    with pytest.raises(RuntimeError):
        kernels.fused_sigmoid_mul(x, gate)


def test_mixed_devices_are_rejected_before_launch() -> None:
    x = torch.zeros(2, 4, device="npu", dtype=torch.bfloat16)
    gate = torch.zeros(2, 4, device="cpu", dtype=x.dtype)
    with pytest.raises(RuntimeError):
        kernels.fused_sigmoid_mul(x, gate)
