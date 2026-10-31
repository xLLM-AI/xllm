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
from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LinearVerifyGraphInputs
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner
from xllm.python.platform import current_platform

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


@pytest.mark.parametrize("draft_width", [1, 2, 3, 4, 5, 7, 16])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
def test_mtp_rejected_prefix_matches_single_token_recurrence(draft_width: int, graph: bool, ragged: bool) -> None:
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
    short_width = max(1, width // 2)
    query_lengths = [width, short_width] if ragged else [width, width]
    packed_inputs = [value.flatten(0, 1).clone() for value in inputs]
    cu_seqlens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    packed_output = torch.zeros_like(packed_inputs[1])
    use_varlen = current_platform.get_ascend_soc_generation() == "a3"
    owners = torch.full((batch_size * width,), -1, device=device, dtype=torch.int32)
    columns = torch.zeros_like(owners)
    packed_metadata = SimpleNamespace(
        control=torch.tensor([batch_size], device=device, dtype=torch.int64),
        linear_state_indices=indices,
        num_accepted_tokens=accepted,
        offsets=cu_seqlens,
        owners=owners,
        columns=columns,
    )
    dense_graph = LinearVerifyGraphInputs(
        state_indices=torch.empty_like(indices),
        num_accepted_tokens=torch.empty_like(accepted),
        query_width=width,
        packed_to_dense=torch.zeros(batch_size * width, device=device, dtype=torch.int64),
        dense_to_packed=torch.zeros(batch_size * width, device=device, dtype=torch.int64),
    )

    def pack_inputs() -> None:
        for source, packed in zip(inputs, packed_inputs):
            packed.zero_()
            values = torch.cat([source[row, :length] for row, length in enumerate(query_lengths)])
            packed[: values.shape[0]].copy_(values)
        cu_seqlens.copy_(torch.tensor([0, query_lengths[0], sum(query_lengths)], device=device, dtype=torch.int32))
        owners.fill_(-1)
        columns.zero_()
        offset = 0
        for row, length in enumerate(query_lengths):
            owners[offset : offset + length].fill_(row)
            columns[offset : offset + length].copy_(torch.arange(length, device=device, dtype=torch.int32))
            offset += length

    pack_inputs()

    def verify() -> torch.Tensor:
        if ragged and use_varlen:
            qkv, z, b, a = packed_inputs
            torch.ops.xllm_ops.mega_gdn_mtp_decode_varlen(
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
                accepted,
                norm_weight,
                cu_seqlens,
                packed_output,
                True,
            )
            return packed_output
        qkv, z, b, a = inputs
        if ragged:
            PreparedAclGraphRunner._refresh_linear_verify_graph(packed_metadata, dense_graph, batch_size * width)
            qkv, z, b, a = [
                torch.cat((value, torch.zeros_like(value[:1])))
                .index_select(0, dense_graph.dense_to_packed)
                .view(batch_size, width, *value.shape[1:])
                for value in packed_inputs
            ]
        output = torch.ops.xllm_ops.mega_gdn_mtp_decode(
            qkv,
            z,
            b,
            a,
            conv_weight,
            conv,
            a_log,
            dt_bias,
            ssm,
            dense_graph.state_indices if ragged else indices,
            dense_graph.state_indices if ragged else indices,
            dense_graph.num_accepted_tokens if ragged else accepted,
            norm_weight,
            True,
        )
        if ragged:
            output = output.flatten(0, 1)
            output = torch.cat((output, torch.zeros_like(output[:1]))).index_select(0, dense_graph.packed_to_dense)
        return output

    if ragged and not use_varlen:
        live_conv, live_ssm = conv[1:].clone(), ssm[width:].clone()
        packed_metadata.control.fill_(batch_size | (1 << 32))
        verify()
        torch.testing.assert_close(conv[1:], live_conv, rtol=0, atol=0)
        torch.testing.assert_close(ssm[width:], live_ssm, rtol=0, atol=0)
        packed_metadata.control.fill_(batch_size)

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
    acceptance_cases = ([1, 1], [1, short_width if ragged else width], [short_width, width])
    for round_index, accepted_host in enumerate(acceptance_cases):
        if round_index:
            for sequence, prefix_length in enumerate(accepted_host):
                slot = sequence + 1
                reference_conv[slot].copy_(prior_conv[prefix_length - 1][slot])
                reference_ssm[slot].copy_(prior_ssm[prefix_length - 1][slot])
            for value in inputs:
                value.copy_(random(tuple(value.shape)))
        if ragged:
            query_lengths = ([width, short_width], [short_width, width], [width, 1])[round_index]
            pack_inputs()
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
        for sequence, length in enumerate(query_lengths):
            torch.testing.assert_close(
                conv[sequence + 1, : length + 2], expected_conv[sequence, : length + 2], rtol=0, atol=0
            )

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
        if ragged:
            expected_output = torch.cat([expected_output[row, :length] for row, length in enumerate(query_lengths)])
            actual = actual[: sum(query_lengths)]
        torch.testing.assert_close(actual, expected_output, rtol=_OUTPUT_RTOL, atol=_OUTPUT_ATOL)
        for sequence in range(batch_size):
            slot = sequence + 1
            length = query_lengths[sequence]
            checkpoints = torch.stack([state[slot] for state in prior_ssm[:length]])
            torch.testing.assert_close(
                ssm[slot * width : slot * width + length], checkpoints, rtol=_STATE_RTOL, atol=_STATE_ATOL
            )
            # Verify that these inputs and tolerances reject a missing state
            # bank and an off-by-one choice of the committed prefix.
            checkpoints_cpu = checkpoints.cpu()
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    torch.zeros_like(checkpoints_cpu), checkpoints_cpu, rtol=_STATE_RTOL, atol=_STATE_ATOL
                )
            if length > 1:
                with pytest.raises(AssertionError):
                    torch.testing.assert_close(
                        torch.roll(checkpoints_cpu, shifts=1, dims=0),
                        checkpoints_cpu,
                        rtol=_STATE_RTOL,
                        atol=_STATE_ATOL,
                    )
    torch.npu.synchronize()
