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

"""NPU tests for Triton DCP packed AllToAll pack + fused LSE combine.

Source: vllm-ascend ``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_cp_a2a.py``.
"""

from __future__ import annotations

import pytest
import torch


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _atol_rtol(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-2, 1e-2


def _decode_packed_lse(send: torch.Tensor, head_dim: int) -> torch.Tensor:
    packed = send[..., head_dim:].float()
    if packed.shape[-1] == 1:
        return packed[..., 0]
    exponent_code = packed[..., 0]
    significand_hi = packed[..., 1]
    significand_mid = packed[..., 2]
    significand_lo = packed[..., 3]
    packed_valid = exponent_code != 0
    sign = torch.where(exponent_code < 0, -1.0, 1.0)
    exponent_magnitude = exponent_code.abs()
    significand = significand_hi * 65536.0 + significand_mid * 256.0 + significand_lo
    lse = sign * significand * torch.exp2(exponent_magnitude - 128.0 - 23.0)
    return torch.where(packed_valid, lse, torch.full_like(lse, float("nan")))


def _reference_merge(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(lse)
    safe_lse = lse.masked_fill(~finite, float("-inf"))
    weights = torch.nan_to_num(torch.softmax(safe_lse, dim=0), nan=0.0)
    safe_output = torch.where(finite.unsqueeze(-1), output.float(), 0.0)
    return (safe_output * weights.unsqueeze(-1)).sum(0).to(output.dtype)


def _simulate_all_to_all(packed_by_rank: list[torch.Tensor], my_rank: int) -> torch.Tensor:
    return torch.stack([packed[my_rank] for packed in packed_by_rank], dim=0)


def test_pack_rejects_cpu_tensors() -> None:
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import pack_dcp_output_lse

    output = torch.randn(8, 16, 512, dtype=torch.bfloat16)
    softmax_lse = torch.randn(8, 16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="NPU"):
        pack_dcp_output_lse(output, softmax_lse, 4, scatter_dim=1)


def _shard_for_rank(
    outputs: list[torch.Tensor],
    lses: list[torch.Tensor],
    dcp_size: int,
    scatter_dim: int,
    my_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scatter_dim == 1:
        h_local = outputs[0].shape[1] // dcp_size
        start = my_rank * h_local
        end = start + h_local
        shard_out = torch.stack([out[:, start:end] for out in outputs], dim=0)
        shard_lse = torch.stack([lse[:, start:end] for lse in lses], dim=0)
        return shard_out, shard_lse
    t_local = outputs[0].shape[0] // dcp_size
    start = my_rank * t_local
    end = start + t_local
    shard_out = torch.stack([out[start:end] for out in outputs], dim=0)
    shard_lse = torch.stack([lse[start:end] for lse in lses], dim=0)
    return shard_out, shard_lse


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_pack_layout_and_lse_roundtrip(dtype: torch.dtype, scatter_dim: int, head_dim: int) -> None:
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import pack_dcp_output_lse

    device = torch.device("npu")
    dcp_size = 4
    num_tokens = 8
    num_heads = 16
    torch.manual_seed(0)
    output = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    softmax_lse = torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device)
    send = pack_dcp_output_lse(output, softmax_lse, dcp_size, scatter_dim)
    torch.npu.synchronize()

    local_scatter = (num_tokens if scatter_dim == 0 else num_heads) // dcp_size
    for dest in range(dcp_size):
        for scatter_idx in range(local_scatter):
            if scatter_dim == 1:
                head_idx = dest * local_scatter + scatter_idx
                packed_out = send[dest, scatter_idx]
                expected_out = output[:, head_idx]
                expected_lse = softmax_lse[:, head_idx]
            else:
                token_idx = dest * local_scatter + scatter_idx
                packed_out = send[dest, scatter_idx]
                expected_out = output[token_idx]
                expected_lse = softmax_lse[token_idx]
            torch.testing.assert_close(packed_out[..., :head_dim], expected_out, rtol=0, atol=0)
            decoded = _decode_packed_lse(packed_out.unsqueeze(0), head_dim).squeeze(0)
            torch.testing.assert_close(decoded, expected_lse, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_fused_combine_matches_softmax_merge(dtype: torch.dtype, scatter_dim: int, head_dim: int) -> None:
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import (
        fused_dcp_lse_combine,
        pack_dcp_output_lse,
    )

    device = torch.device("npu")
    dcp_size = 4
    num_tokens = 8
    num_heads = 16
    my_rank = 1
    torch.manual_seed(1)
    outputs = [torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device) for _ in range(dcp_size)]
    lses = [torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device) * 8.0 for _ in range(dcp_size)]
    packed = [pack_dcp_output_lse(out, lse, dcp_size, scatter_dim) for out, lse in zip(outputs, lses)]
    recv = _simulate_all_to_all(packed, my_rank)
    merged = fused_dcp_lse_combine(recv, head_dim, scatter_dim)
    torch.npu.synchronize()
    shard_out, shard_lse = _shard_for_rank(outputs, lses, dcp_size, scatter_dim, my_rank)
    expected = _reference_merge(shard_out, shard_lse)
    atol, rtol = _atol_rtol(dtype)
    torch.testing.assert_close(merged, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_fused_combine_skips_invalid_lse() -> None:
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import (
        fused_dcp_lse_combine,
        pack_dcp_output_lse,
    )

    device = torch.device("npu")
    dcp_size = 4
    num_tokens = 4
    num_heads = 8
    head_dim = 128
    scatter_dim = 1
    torch.manual_seed(2)
    outputs = [
        torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device) for _ in range(dcp_size)
    ]
    lses = [torch.randn(num_tokens, num_heads, dtype=torch.float32, device=device) for _ in range(dcp_size)]
    lses[0][:] = float("nan")
    lses[1][0, 0] = float("inf")
    lses[2][1, 1] = float("-inf")
    packed = [pack_dcp_output_lse(out, lse, dcp_size, scatter_dim) for out, lse in zip(outputs, lses)]
    recv = _simulate_all_to_all(packed, my_rank=0)
    merged = fused_dcp_lse_combine(recv, head_dim, scatter_dim)
    torch.npu.synchronize()
    shard_out, shard_lse = _shard_for_rank(outputs, lses, dcp_size, scatter_dim, my_rank=0)
    expected = _reference_merge(shard_out, shard_lse)
    torch.testing.assert_close(merged, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_pack_accepts_noncontiguous_and_graph_lse() -> None:
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import pack_dcp_output_lse

    device = torch.device("npu")
    dcp_size = 4
    num_tokens = 8
    num_heads = 16
    head_dim = 512
    scatter_dim = 1
    torch.manual_seed(3)
    base = torch.randn(num_heads, num_tokens, head_dim, dtype=torch.bfloat16, device=device)
    output = base.permute(1, 0, 2)
    assert not output.is_contiguous()
    softmax_lse = torch.randn(1, num_tokens, num_heads, 1, dtype=torch.float32, device=device)[0]
    assert tuple(softmax_lse.shape) == (num_tokens, num_heads, 1)
    send = pack_dcp_output_lse(output, softmax_lse, dcp_size, scatter_dim)
    torch.npu.synchronize()
    h_local = num_heads // dcp_size
    torch.testing.assert_close(
        send[0, 0, :, :head_dim],
        output[:, 0],
        rtol=0,
        atol=0,
    )
    decoded = _decode_packed_lse(send[1, 3], head_dim)
    torch.testing.assert_close(decoded, softmax_lse[:, 1 * h_local + 3, 0], rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_pack_preserves_lse_outside_fp16_range() -> None:
    from xllm.python.kernels_npu.triton.dcp_packed_a2a import pack_dcp_output_lse

    device = torch.device("npu")
    dcp_size = 2
    num_tokens = 2
    num_heads = 4
    head_dim = 128
    output = torch.zeros(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    softmax_lse = torch.full((num_tokens, num_heads), 70000.0, dtype=torch.float32, device=device)
    softmax_lse[0, 1] = -70000.0
    send = pack_dcp_output_lse(output, softmax_lse, dcp_size, scatter_dim=1)
    torch.npu.synchronize()
    decoded = _decode_packed_lse(send[0, 0], head_dim)
    torch.testing.assert_close(decoded, softmax_lse[:, 0], rtol=1e-5, atol=1.0)
    decoded_neg = _decode_packed_lse(send[0, 1], head_dim)
    torch.testing.assert_close(decoded_neg, softmax_lse[:, 1], rtol=1e-5, atol=1.0)
