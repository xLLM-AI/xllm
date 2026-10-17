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

"""Real-binding cases launched by glm5_next_moe_test, not CPU pytest.

The C++ driver registers the production xllm_ops library and initializes its
Python runtime before loading this file. No native binding is replaced here.
References use CPU integer matmuls and explicit FP32 dequant/clamp/quant math.
These are local-expert tests, not distributed or full-model ON/ON acceptance.
"""

from __future__ import annotations

from types import ModuleType

import torch

_LIMIT = 10.0
_SCALE_RTOL = 2e-4
_SCALE_ATOL = 1e-5
_MOE_RTOL = 0.02
_MOE_ATOL = 0.1


def _require_real_binding() -> ModuleType:
    from xllm.python import kernels

    assert kernels.dequant_swiglu_quant.__module__ == "xllm.python.kernels_npu.moe"
    assert torch._C._dispatch_has_kernel_for_dispatch_key("xllm_ops::dequant_swiglu_quant", "PrivateUse1")
    return kernels


def _quantize_reference(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    scale = value.abs().amax(dim=-1) / 127.0
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(value / divisor.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return quantized, scale


def _activation_reference(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.float().chunk(2, dim=-1)
    return torch.nn.functional.silu(gate.clamp_max(_LIMIT)) * up.clamp(-_LIMIT, _LIMIT)


def _assert_quantized(
    actual: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output, scale = actual
    expected, expected_scale = reference
    assert output.dtype == torch.int8
    assert scale.dtype == torch.float32
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=_SCALE_RTOL, atol=_SCALE_ATOL)
    error = (output.cpu().int() - expected.int()).abs()
    assert int(error.max()) <= 1, f"INT8 activation differs by {int(error.max())} quantization levels"


@torch.inference_mode()
def check_v2_clamped_activation() -> None:
    kernels = _require_real_binding()
    for intermediate in (128, 2048):
        # Deliberately cross both clamp boundaries; this is not an agreement
        # check between two invocations of the same kernel.
        gate = torch.tensor([-20, -5, -1, 1, 5, 20, -12, 12], dtype=torch.int32)
        up = torch.tensor([3, 15, -20, 7, -2, 20, 1, -15], dtype=torch.int32)
        gate_up = torch.cat((gate.repeat(intermediate // 8), up.repeat(intermediate // 8))).repeat(3, 1)
        weight_scale = torch.ones(3, 2 * intermediate, dtype=torch.float32)
        weight_scale[2, :intermediate] = 0.75
        weight_scale[2, intermediate:] = 1.25
        activation_scale = torch.tensor([1.0, 0.5, 1.5], dtype=torch.float32)
        counts = torch.tensor([2, 0, 1], dtype=torch.int64)
        row_experts = torch.tensor([0, 0, 2])
        dequantized = gate_up.float() * weight_scale[row_experts] * activation_scale[:, None]
        expected = _quantize_reference(_activation_reference(dequantized))
        plain_gate, plain_up = dequantized.chunk(2, dim=-1)
        plain = torch.nn.functional.silu(plain_gate) * plain_up
        assert not torch.allclose(plain, _activation_reference(dequantized))

        # V2 provider: custom_xllm_math/op_api/lib/libcust_opapi.so.
        actual = kernels.dequant_swiglu_quant(
            gate_up.to("npu:0"),
            weight_scale.to("npu:0"),
            activation_scale.to("npu:0"),
            group_index=counts.to("npu:0"),
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=_LIMIT,
            glu_alpha=1.0,
            glu_bias=0.0,
        )
        torch.npu.synchronize()
        _assert_quantized(actual, expected)


def _make_case(rows: int, *, quantization_safe: bool = True) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(53 + rows)
    case = {
        "hidden": (torch.randn(rows, 128, generator=generator) * 4).to(torch.bfloat16),
        "w13": torch.randint(-16, 17, (4, 128, 256), dtype=torch.int8, generator=generator),
        "w2": torch.randint(-16, 17, (4, 128, 128), dtype=torch.int8, generator=generator),
        "s13": torch.full((4, 256), 0.05),
        "s2": torch.full((4, 128), 0.02, dtype=torch.bfloat16),
        "ids": ((torch.arange(rows).unsqueeze(1) + torch.tensor([0, 2])) % 4).to(torch.int32),
        "probs": torch.tensor([0.25, 0.75]).expand(rows, 2).contiguous(),
    }
    if quantization_safe:
        # Exact BF16 multiples of 1/16 with an exact maximum of 127/16.
        # Input quantization therefore maps to integers, a distance of 0.5
        # from every rounding tie. Do not confuse these ties with the GLM
        # activation clamp boundaries exercised by the V2 test above.
        codes = (case["hidden"].float() * 8).round().clamp(-126, 126)
        codes[:, 0] = 127
        case["hidden"] = (codes / 16).to(torch.bfloat16)
        # Also separate the activation requantization from half ties. The
        # first input coordinate is 127/16; these power-of-two scales give
        # gate=+/-15.875 and up=k*127/256, with a clamped positive maximum
        # fixed at 10*silu(10). Nonzero normalized values are k*16129/2560;
        # for |k|<=24 their distance from a half tie exceeds 0.0019.
        case["w13"].zero_()
        case["w13"][:, 0, :128] = 32
        case["w13"][:, 0, 2:128:8] = -32
        case["w13"][:, 0, 128:] = torch.randint(-24, 25, (4, 128), dtype=torch.int8, generator=generator)
        case["w13"][:, 0, 128] = 64
        case["w13"][:, 0, 129] = -64
        case["s13"].fill_(1 / 16)
        case["s2"].fill_(1 / 32)
    return case


def _grouped_reference(case: dict[str, torch.Tensor], start: int, end: int) -> torch.Tensor:
    hidden = case["hidden"]
    x_int8, x_scale = _quantize_reference(hidden)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    probabilities = case["probs"].to(hidden.dtype).float()
    for expert in range(start, end):
        rows, slots = torch.where(case["ids"] == expert)
        if rows.numel() == 0:
            continue
        accumulated = x_int8[rows].to(torch.int64) @ case["w13"][expert].to(torch.int64)
        gate_up = accumulated.float() * x_scale[rows, None] * case["s13"][expert]
        act_int8, act_scale = _quantize_reference(_activation_reference(gate_up))
        down = act_int8.to(torch.int64) @ case["w2"][expert].to(torch.int64)
        down_scale = case["s2"][expert].to(hidden.dtype).float()
        projected = (down.float() * act_scale[:, None] * down_scale).to(hidden.dtype)
        output.index_add_(0, rows, projected.float() * probabilities[rows, slots, None])
    return output.to(hidden.dtype)


def _device_case(case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    from xllm.python import kernels

    result = {name: tensor.to("npu:0") for name, tensor in case.items() if name not in ("w13", "w2")}
    # Format each weight partition independently, as in real expert loading.
    for start, end in ((0, 2), (2, 4)):
        result[f"w13_{start}"] = kernels.format_cast_nz(case["w13"][start:end].to("npu:0"))
        result[f"w2_{start}"] = kernels.format_cast_nz(case["w2"][start:end].to("npu:0"))
    return result


def _run_shard(case: dict[str, torch.Tensor], start: int) -> torch.Tensor:
    from xllm.python import kernels

    return kernels.grouped_moe_with_selected_experts(
        case["hidden"],
        case["probs"],
        case["ids"],
        case[f"w13_{start}"],
        case[f"w2_{start}"],
        case["s13"][start : start + 2],
        case["s2"][start : start + 2],
        num_total_experts=4,
        start_expert_id=start,
        num_experts_per_rank=2,
        swiglu_limit=_LIMIT,
    )


def _check_input_quantization_boundary() -> None:
    kernels = _require_real_binding()
    # Original graph step 1: preserve its half ties and validate quantization
    # itself, independently of any amplified downstream matmul difference.
    hidden = (_make_case(16, quantization_safe=False)["hidden"].float() + 0.25).to(torch.bfloat16)
    expected, expected_scale = _quantize_reference(hidden)
    normalized = hidden.float() / expected_scale[:, None]
    tie_distance = (normalized - normalized.floor() - 0.5).abs()
    assert (tie_distance < 1e-4).any()
    actual, actual_scale = kernels.dynamic_quant(hidden.to("npu:0"))
    torch.npu.synchronize()
    torch.testing.assert_close(
        actual_scale.cpu().view_as(expected_scale), expected_scale, rtol=_SCALE_RTOL, atol=_SCALE_ATOL
    )
    codes = actual.cpu().int()
    errors = (codes - expected.int()).abs()
    assert int(errors.max()) <= 1
    changed = errors != 0
    assert torch.all(tie_distance[changed] < 1e-4), "input quantization changed a non-boundary code"
    assert torch.all((codes.float() - normalized).abs() <= 0.5001), "input quantization exceeds nearest-rounding error"


@torch.inference_mode()
def check_grouped_local_experts() -> None:
    _require_real_binding()
    _check_input_quantization_boundary()
    for rows in (1, 7, 16):
        case = _make_case(rows)
        _, input_scale = _quantize_reference(case["hidden"])
        normalized = case["hidden"].float() / input_scale[:, None]
        torch.testing.assert_close(normalized, normalized.round(), rtol=0, atol=0)
        x_int8, _ = _quantize_reference(case["hidden"])
        for expert in range(4):
            accumulated = x_int8.to(torch.int64) @ case["w13"][expert].to(torch.int64)
            activated = _activation_reference(accumulated.float() * input_scale[:, None] * case["s13"][expert])
            _, scale = _quantize_reference(activated)
            normalized_activation = activated / scale[:, None]
            assert torch.all((normalized_activation - normalized_activation.floor() - 0.5).abs() > 1e-3)
        device = _device_case(case)
        for start in (0, 2):
            result = _run_shard(device, start)
            torch.npu.synchronize()
            expected = _grouped_reference(case, start, start + 2)
            torch.testing.assert_close(result.cpu(), expected, rtol=_MOE_RTOL, atol=_MOE_ATOL)
        case["ids"] = torch.tensor([0, 1], dtype=torch.int32).expand(rows, 2).contiguous()
        device["ids"].copy_(case["ids"])
        empty = _run_shard(device, 2)
        torch.npu.synchronize()
        torch.testing.assert_close(empty.cpu(), torch.zeros_like(case["hidden"]), rtol=0, atol=0)


@torch.inference_mode()
def check_local_shard_graph_buffers() -> None:
    _require_real_binding()
    # Preserve the original seed and input updates, including input-quantizer
    # half ties. This test checks replay against eager, not CPU rounding
    # choices. Independent numerical acceptance is covered separately above.
    base = _make_case(16, quantization_safe=False)
    buffers = [_device_case(base), _device_case(base)]
    graphs = []
    outputs = []
    try:
        for device in buffers:
            _run_shard(device, 0)
            _run_shard(device, 2)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            graphs.append(graph)
            with torch.npu.graph(graph):
                outputs.append((_run_shard(device, 0), _run_shard(device, 2)))
        addresses = [tuple(output.data_ptr() for output in pair) for pair in outputs]
        # Each buffer sees both local ranges become empty and populated.
        for step, route in enumerate((0, 2, 2, 0, 0, 2, 2, 0)):
            slot = step % 2
            case = dict(base)
            case["ids"] = torch.tensor([route, route + 1], dtype=torch.int32).expand(16, 2).contiguous()
            case["hidden"] = (base["hidden"].float() + step * 0.25).to(torch.bfloat16)
            buffers[slot]["hidden"].copy_(case["hidden"])
            buffers[slot]["ids"].copy_(case["ids"])
            graphs[slot].replay()
            torch.npu.synchronize()
            assert tuple(output.data_ptr() for output in outputs[slot]) == addresses[slot]
            for index, start in enumerate((0, 2)):
                captured = outputs[slot][index].cpu()
                direct = _run_shard(buffers[slot], start).cpu()
                torch.testing.assert_close(
                    captured, direct, rtol=0, atol=0, msg=f"step={step}, slot={slot}, start={start}"
                )
                if start != route:
                    torch.testing.assert_close(captured, torch.zeros_like(captured), rtol=0, atol=0)
    finally:
        # Destroy graphs while all referenced inputs, weights and outputs live.
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
