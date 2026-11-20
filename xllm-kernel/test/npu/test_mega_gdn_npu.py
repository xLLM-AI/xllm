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

"""Real fused GDN operators with independent Conv and SSM row spaces."""

from __future__ import annotations

import pytest
import torch
from xllm_kernel.ops.attention.gdn.mega import (
    has_mega_gdn_mtp_decode_varlen,
    mega_gdn_decode,
    mega_gdn_mtp_decode,
    mega_gdn_mtp_decode_varlen,
    mega_gdn_prefill,
)


def _state_caches(width: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    # Conv indices intentionally exceed the former operator's 1024-slot limit.
    # The SSM arena has a different capacity and unrelated checkpoint bases.
    # Two exactly representable BF16 components distinguish all 1100 slots.
    conv_rows = torch.arange(1100, dtype=torch.int64)
    conv_values = torch.stack(((conv_rows % 128 + 128) / 4096, (conv_rows // 128 + 16) / 512), dim=-1)
    conv = conv_values[:, None, :].repeat(1, width + 2, 192).to(device=device, dtype=torch.bfloat16)
    ssm_rows = (torch.arange(53, dtype=torch.float32) + 8) / 512
    ssm = ssm_rows[:, None, None, None].expand(-1, 1, 128, 128).contiguous().to(device=device)
    dense_conv = torch.zeros(5, width + 2, 384, dtype=torch.bfloat16, device=device)
    dense_ssm = torch.zeros(5 * width, 1, 128, 128, device=device)
    return conv, ssm, dense_conv, dense_ssm


@pytest.mark.parametrize("width", [1, 3, 5], ids=["decode", "verify3", "verify5"])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_split_state_indices_match_compact_state_and_preserve_unowned_rows(width: int, graph_mode: bool) -> None:
    torch.manual_seed(1808)
    device = torch.device("npu", torch.npu.current_device())
    conv, ssm, dense_conv, dense_ssm = _state_caches(width, device)
    conv_initial, ssm_initial = conv.clone(), ssm.clone()
    dense_initial_conv, dense_initial_ssm = dense_conv.clone(), dense_ssm.clone()
    qkv = torch.randn(3, width, 384, dtype=torch.bfloat16, device=device) * 0.1
    z = torch.randn(3, width, 1, 128, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(3, width, 1, dtype=torch.bfloat16, device=device) * 0.1
    a = torch.randn_like(b) * 0.1
    weight = torch.randn(4, 384, dtype=torch.bfloat16, device=device) * 0.1
    a_log = torch.tensor([-0.5], device=device)
    dt_bias = torch.zeros(1, device=device)
    norm_weight = torch.ones(128, dtype=torch.bfloat16, device=device)
    read = torch.tensor([[1043, 7], [1049, 18], [-1, -1]], dtype=torch.int32, device=device)
    write = torch.tensor([[1095, 30], [1097, 40], [-1, -1]], dtype=torch.int32, device=device)
    dense_read = torch.tensor([1, 2], dtype=torch.int32, device=device)
    dense_write = torch.tensor([3, 4], dtype=torch.int32, device=device)
    accepted = torch.tensor([1, width, 1], dtype=torch.int32, device=device)

    def _actual() -> torch.Tensor:
        if width == 1:
            return mega_gdn_decode(
                qkv[:, 0],
                z[:, 0],
                b[:, 0],
                a[:, 0],
                weight,
                conv,
                a_log,
                dt_bias,
                ssm,
                read,
                write,
                norm_weight,
            )
        return mega_gdn_mtp_decode(
            qkv,
            z,
            b,
            a,
            weight,
            conv,
            a_log,
            dt_bias,
            ssm,
            read,
            write,
            accepted,
            norm_weight,
        )

    def _restore() -> None:
        conv.copy_(conv_initial)
        ssm.copy_(ssm_initial)
        dense_conv.copy_(dense_initial_conv)
        dense_ssm.copy_(dense_initial_ssm)
        for sequence in range(2):
            dense_conv[sequence + 1].copy_(conv_initial[read[sequence, 0].item()])
            base = read[sequence, 1].item()
            dense_ssm[(sequence + 1) * width : (sequence + 2) * width].copy_(ssm_initial[base : base + width])

    _restore()
    if graph_mode:
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            for _ in range(3):
                _actual()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                actual = _actual()
        torch.npu.current_stream(device).wait_stream(stream)
        torch.npu.synchronize()

    for iteration in range(2):
        if iteration:
            # Update the same index storage after capture: both row spaces
            # move independently without changing graph nodes or pointers.
            read.copy_(torch.tensor([[1049, 11], [1043, 22], [-1, -1]], dtype=torch.int32, device=device))
            write.copy_(torch.tensor([[1097, 35], [1095, 45], [-1, -1]], dtype=torch.int32, device=device))
            qkv.normal_(std=0.1)
            accepted.copy_(torch.tensor([width, 1, 1], dtype=torch.int32, device=device))
        _restore()
        if graph_mode:
            torch.npu.synchronize()
            graph.replay()
        else:
            actual = _actual()
        if width == 1:
            expected = mega_gdn_decode(
                qkv[:2, 0],
                z[:2, 0],
                b[:2, 0],
                a[:2, 0],
                weight,
                dense_conv,
                a_log,
                dt_bias,
                dense_ssm,
                dense_read,
                dense_write,
                norm_weight,
            )
        else:
            expected = mega_gdn_mtp_decode(
                qkv[:2],
                z[:2],
                b[:2],
                a[:2],
                weight,
                dense_conv,
                a_log,
                dt_bias,
                dense_ssm,
                dense_read,
                dense_write,
                accepted[:2],
                norm_weight,
            )
        torch.npu.synchronize()
        torch.testing.assert_close(actual[:2], expected, rtol=0.02, atol=0.002)
        torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]), rtol=0, atol=0)
        conv_unowned = torch.ones(conv.shape[0], dtype=torch.bool, device=device)
        ssm_unowned = torch.ones(ssm.shape[0], dtype=torch.bool, device=device)
        for sequence in range(2):
            conv_row = write[sequence, 0].item()
            ssm_row = write[sequence, 1].item()
            torch.testing.assert_close(conv[conv_row], dense_conv[sequence + 3], rtol=0, atol=0)
            torch.testing.assert_close(
                ssm[ssm_row : ssm_row + width],
                dense_ssm[(sequence + 3) * width : (sequence + 4) * width],
                rtol=0.0001,
                atol=0.00001,
            )
            conv_unowned[conv_row] = False
            ssm_unowned[ssm_row : ssm_row + width] = False
        torch.testing.assert_close(conv[conv_unowned], conv_initial[conv_unowned], rtol=0, atol=0)
        torch.testing.assert_close(ssm[ssm_unowned], ssm_initial[ssm_unowned], rtol=0, atol=0)


@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_split_state_varlen_uses_physical_bases_and_leaves_cold_checkpoints(graph_mode: bool) -> None:
    if not has_mega_gdn_mtp_decode_varlen():
        pytest.skip("packed varlen fused verifier requires Ascend A2/A3")
    torch.manual_seed(1809)
    device = torch.device("npu", torch.npu.current_device())
    width = 5
    conv, ssm, dense_conv, dense_ssm = _state_caches(width, device)
    indices = torch.tensor([[1043, 7], [1049, 18], [-1, -1]], dtype=torch.int32, device=device)
    for sequence, (conv_row, ssm_row) in enumerate(((1043, 7), (1049, 18))):
        dense_conv[sequence + 1].copy_(conv[conv_row])
        dense_ssm[(sequence + 1) * width : (sequence + 2) * width].copy_(ssm[ssm_row : ssm_row + width])
    qkv = torch.randn(6, 384, dtype=torch.bfloat16, device=device) * 0.1
    z = torch.randn(6, 1, 128, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(6, 1, dtype=torch.bfloat16, device=device) * 0.1
    a = torch.randn_like(b) * 0.1
    weight = torch.randn(4, 384, dtype=torch.bfloat16, device=device) * 0.1
    a_log, dt_bias = torch.tensor([-0.5], device=device), torch.zeros(1, device=device)
    accepted = torch.tensor([2, 4, 1], dtype=torch.int32, device=device)
    norm_weight = torch.ones(128, dtype=torch.bfloat16, device=device)
    endpoints = torch.tensor([0, 1, 4, 6], dtype=torch.int32, device=device)
    actual = torch.zeros_like(z)
    initial = tuple(tensor.clone() for tensor in (conv, ssm, dense_conv, dense_ssm))

    def run() -> None:
        mega_gdn_mtp_decode_varlen(
            qkv, z, b, a, weight, conv, a_log, dt_bias, ssm, indices, accepted, norm_weight, endpoints, actual
        )

    if graph_mode:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()

    pointer = actual.data_ptr()
    for step in range(3):
        for tensor, snapshot in zip((conv, ssm, dense_conv, dense_ssm), initial, strict=True):
            tensor.copy_(snapshot)
        actual.zero_()
        if step:
            qkv.normal_(std=0.1)
            accepted.copy_(torch.tensor([step, width - step, 1], dtype=torch.int32, device=device))
        if graph_mode:
            graph.replay()
        else:
            run()
        expected = torch.zeros_like(z[:4])
        mega_gdn_mtp_decode_varlen(
            qkv[:4],
            z[:4],
            b[:4],
            a[:4],
            weight,
            dense_conv,
            a_log,
            dt_bias,
            dense_ssm,
            torch.tensor([1, 2], dtype=torch.int32, device=device),
            accepted[:2],
            norm_weight,
            endpoints[:3],
            expected,
        )
        torch.npu.synchronize()
        assert actual.data_ptr() == pointer
        torch.testing.assert_close(actual[:4], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[4:], torch.zeros_like(actual[4:]), rtol=0, atol=0)
        for sequence, (conv_row, ssm_row) in enumerate(((1043, 7), (1049, 18))):
            torch.testing.assert_close(conv[conv_row], dense_conv[sequence + 1], rtol=0, atol=0)
            torch.testing.assert_close(
                ssm[ssm_row : ssm_row + width],
                dense_ssm[(sequence + 1) * width : (sequence + 2) * width],
                rtol=0,
                atol=0,
            )


def _prefill_reference(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
    initial: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    window = torch.cat((history, qkv))
    projected = torch.nn.functional.conv1d(window.float().t().unsqueeze(0), weight.float().t().unsqueeze(1), groups=384)
    q, k, v = torch.nn.functional.silu(projected[0].t()).bfloat16().float().chunk(3, dim=-1)
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    g = -a_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = b.float().sigmoid()
    state = initial.float()
    outputs = []
    for i in range(qkv.shape[0]):
        state = state * g[i].exp()
        residual = (v[i] - k[i] @ state) * beta[i]
        state = state + k[i, :, None] * residual[None, :]
        value = (q[i] @ state) * 128**-0.5
        normalized = value * torch.rsqrt(value.square().mean() + 1e-6)
        outputs.append(normalized * torch.nn.functional.silu(z[i, 0].float()))
    return torch.stack(outputs).bfloat16().unsqueeze(1), window[-3:], state


@torch.inference_mode()
def test_mega_gdn_prefill_matches_reference_and_writes_only_selected_rows() -> None:
    generator = torch.Generator().manual_seed(2801)
    qkv = (torch.randn((7, 384), generator=generator) * 0.15).bfloat16()
    z = (torch.randn((7, 1, 128), generator=generator) * 0.15).bfloat16()
    b = (torch.randn((7, 1), generator=generator) * 0.2).bfloat16()
    a = (torch.randn((7, 1), generator=generator) * 0.2).bfloat16()
    weight = (torch.randn((4, 384), generator=generator) * 0.12).bfloat16()
    conv = (torch.randn((5, 3, 384), generator=generator) * 0.08).bfloat16()
    ssm = torch.randn((5, 1, 128, 128), generator=generator) * 0.004
    a_log = torch.tensor([-0.5])
    dt_bias = torch.tensor([0.1])
    conv_read, conv_write = (3, -1), (0, 4)
    ssm_read, ssm_write = (1, -1), (3, 2)
    read_and_write = tuple(
        torch.tensor(indices, dtype=torch.int32).npu() for indices in (conv_read, conv_write, ssm_read, ssm_write)
    )
    conv_npu, ssm_npu = conv.npu(), ssm.npu()
    output = mega_gdn_prefill(
        qkv.npu(),
        b.npu(),
        a.npu(),
        z.npu(),
        weight.npu(),
        conv_npu,
        a_log.npu(),
        dt_bias.npu(),
        *read_and_write,
        ssm_npu,
        torch.tensor([0, 3, 7], dtype=torch.int32).npu(),
        torch.ones(128, dtype=torch.bfloat16).npu(),
        2,
    )

    expected_conv, expected_ssm = conv.clone(), ssm.clone()
    outputs = []
    for i, token_slice in enumerate((slice(0, 3), slice(3, 7))):
        history = conv[conv_read[i]] if conv_read[i] >= 0 else torch.zeros_like(conv[0])
        initial = ssm[ssm_read[i], 0] if ssm_read[i] >= 0 else torch.zeros_like(ssm[0, 0])
        result, history_out, final = _prefill_reference(
            qkv[token_slice], z[token_slice], b[token_slice], a[token_slice], weight, history, initial, a_log, dt_bias
        )
        outputs.append(result)
        expected_conv[conv_write[i]] = history_out
        expected_ssm[ssm_write[i], 0] = final
    torch.testing.assert_close(output.cpu(), torch.cat(outputs), atol=0.005, rtol=0.04)
    torch.testing.assert_close(conv_npu.cpu(), expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(ssm_npu.cpu(), expected_ssm, atol=4e-4, rtol=0.03)
