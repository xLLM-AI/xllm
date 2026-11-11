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

import os

import pytest
import torch

from xllm.python.kernels_npu.gated_delta_net import (
    mega_gdn_decode,
    mega_gdn_mtp_decode,
    mega_gdn_mtp_decode_varlen,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
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
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(1808)
    device = torch.device("npu:0")
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


@torch.inference_mode()
def test_split_state_varlen_uses_physical_bases_and_leaves_cold_checkpoints() -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    if "Ascend910B" not in torch.npu.get_device_name(0) and "Ascend910_93" not in torch.npu.get_device_name(0):
        pytest.skip("packed varlen fused verifier requires Ascend A2/A3")
    torch.manual_seed(1809)
    device = torch.device("npu:0")
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
    mega_gdn_mtp_decode_varlen(
        qkv,
        z,
        b,
        a,
        weight,
        conv,
        a_log,
        dt_bias,
        ssm,
        indices,
        accepted,
        norm_weight,
        endpoints,
        actual,
    )
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
