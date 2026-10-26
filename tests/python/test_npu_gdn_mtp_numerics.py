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

"""Real fused-MTP checkpoint/rollback comparisons against one-token GDN.

Run only after reserving an idle device through the container test launcher.
The reference advances the production single-token operator independently;
it does not use the speculative checkpoint bank to construct expected states.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("torch_npu", reason="requires the real NPU backend")

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)

_OUTPUT_RTOL = 0.02
_OUTPUT_ATOL = 1e-3
_STATE_RTOL = 0.005
# Match the FP32 checkpoint tolerance of the same-version fused operator's
# reference test; Conv history remains exact and wrong-state controls must fail.
_STATE_ATOL = 2.5e-5


@pytest.mark.parametrize("draft_width", [1, 2, 3, 4, 16])
@pytest.mark.parametrize("graph", [False, True])
def test_mtp_rejected_prefix_matches_single_token_recurrence(draft_width: int, graph: bool) -> None:
    width = draft_width + 1
    device = torch.device("npu:0")
    generator = torch.Generator().manual_seed(90210 + width)
    batch_size, slots, heads, dimension = 2, 3, 1, 128
    channels = 3 * dimension

    def random(shape: tuple[int, ...], *, fp32: bool = False) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.1).to(
            device=device, dtype=torch.float32 if fp32 else torch.bfloat16
        )

    conv_weight = random((4, channels))
    a_log = random((heads,), fp32=True)
    dt_bias = torch.zeros(heads, device=device, dtype=torch.float32)
    norm_weight = torch.ones(dimension, device=device, dtype=torch.bfloat16)
    indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
    cache_indices = indices.to(torch.long)
    no_state = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
    prefix = (
        random((14, channels)),
        random((14, heads, dimension)),
        random((14, heads)),
        random((14, heads)),
    )
    boundaries = torch.tensor([0, 7, 14], device=device, dtype=torch.int32)

    def prefill(stride: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv = torch.zeros(slots, stride + 2, channels, device=device, dtype=torch.bfloat16)
        ssm = torch.zeros(slots * stride, heads, dimension, dimension, device=device, dtype=torch.float32)
        qkv, z, b, a = prefix
        result = torch.ops.xllm_ops.mega_gdn_prefill(
            qkv,
            b,
            a,
            z,
            conv_weight,
            conv,
            a_log,
            dt_bias,
            no_state,
            indices,
            no_state,
            indices * stride,
            ssm,
            boundaries,
            norm_weight,
            batch_size * heads,
        )
        return conv, ssm, result

    conv, ssm, prefill_output = prefill(width)
    reference_conv, reference_ssm, reference_prefill = prefill(1)
    torch.testing.assert_close(prefill_output, reference_prefill, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
    inputs = [
        random((batch_size, width, channels)),
        random((batch_size, width, heads, dimension)),
        random((batch_size, width, heads)),
        random((batch_size, width, heads)),
    ]
    accepted = torch.ones(batch_size, device=device, dtype=torch.int32)

    def verify() -> torch.Tensor:
        qkv, z, b, a = inputs
        return torch.ops.xllm_ops.mega_gdn_mtp_decode(
            qkv,
            z,
            b,
            a,
            conv_weight,
            conv,
            a_log,
            dt_bias,
            ssm,
            indices,
            indices,
            accepted,
            norm_weight,
            True,
        )

    captured = None
    graph_output = None
    if graph:
        # Both warmup and capture mutate recurrent state. Restore the exact
        # prefill state before the first measured replay, including Conv history.
        initial_conv, initial_ssm = conv.clone(), ssm.clone()
        verify()
        torch.npu.synchronize()
        conv.copy_(initial_conv)
        ssm.copy_(initial_ssm)
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            graph_output = verify()
        torch.npu.synchronize()
        conv.copy_(initial_conv)
        ssm.copy_(initial_ssm)

    prior_conv: list[torch.Tensor] = []
    prior_ssm: list[torch.Tensor] = []
    acceptance_cases = ([1, 1], [1, width], [max(1, width // 2), width])
    for round_index, accepted_host in enumerate(acceptance_cases):
        if round_index:
            for sequence, prefix_length in enumerate(accepted_host):
                slot = sequence + 1
                reference_conv[slot].copy_(prior_conv[prefix_length - 1][slot])
                reference_ssm[slot].copy_(prior_ssm[prefix_length - 1][slot])
            for value in inputs:
                value.copy_(random(tuple(value.shape)))
        # MegaGdnMtpDecode preserves the selected history's final two raw
        # rows, then stores the current raw QKV tokens at offsets [2, S+2).
        # Construct that window before the single-token reference advances.
        expected_conv = torch.cat((reference_conv.index_select(0, cache_indices)[:, -2:], inputs[0]), dim=1)
        accepted.copy_(torch.tensor(accepted_host, device=device, dtype=torch.int32))
        if captured is not None:
            captured.replay()
            actual = graph_output
        else:
            actual = verify()
        torch.testing.assert_close(conv.index_select(0, cache_indices), expected_conv, rtol=0, atol=0)

        expected = []
        prior_conv, prior_ssm = [], []
        qkv, z, b, a = inputs
        for step in range(width):
            expected.append(
                torch.ops.xllm_ops.mega_gdn_decode(
                    qkv[:, step].contiguous(),
                    z[:, step].contiguous(),
                    b[:, step].contiguous(),
                    a[:, step].contiguous(),
                    conv_weight,
                    reference_conv,
                    a_log,
                    dt_bias,
                    reference_ssm,
                    indices,
                    indices,
                    norm_weight,
                    True,
                )
            )
            prior_conv.append(reference_conv.clone())
            prior_ssm.append(reference_ssm.clone())
        expected_output = torch.stack(expected, dim=1)
        torch.testing.assert_close(actual, expected_output, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        for sequence in range(batch_size):
            slot = sequence + 1
            checkpoints = torch.stack([state[slot] for state in prior_ssm])
            torch.testing.assert_close(
                ssm[slot * width : (slot + 1) * width], checkpoints, rtol=_STATE_RTOL, atol=_STATE_ATOL
            )
            # Verify that these inputs and tolerances reject a missing state
            # bank and an off-by-one choice of the committed prefix.
            checkpoints_cpu = checkpoints.cpu()
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    torch.zeros_like(checkpoints_cpu), checkpoints_cpu, rtol=_STATE_RTOL, atol=_STATE_ATOL
                )
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    torch.roll(checkpoints_cpu, shifts=1, dims=0), checkpoints_cpu, rtol=_STATE_RTOL, atol=_STATE_ATOL
                )
    torch.npu.synchronize()
