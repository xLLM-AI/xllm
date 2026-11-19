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

"""NPU tests for fused MLA preprocessing and weight packing."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from functools import cache
from typing import Any

import pytest
import torch
from xllm_kernel.ops.attention.mla import has_mla_preprocess_v2, mla_preprocess_v2
from xllm_kernel.ops.attention.mla.npu.weights import prepare_mla_preprocess_v2_q_b

torch_npu = pytest.importorskip("torch_npu")

_DTYPE = torch.bfloat16
_TOKEN_NUM = 2
_HIDDEN_SIZE = 7168
_Q_LORA_RANK = 1536
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_NUM_HEADS = 2
_BLOCK_NUM = 2
_BLOCK_SIZE = 128


@pytest.fixture(autouse=True)
def _allow_internal_format() -> Iterator[None]:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    if not has_mla_preprocess_v2():
        pytest.skip("aclnnMlaPreprocessV2 is not available in the loaded runtime")
    # Fused MLA weights require NZ storage. Restore the process setting afterward.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    yield
    torch.npu.config.allow_internal_format = previous in ("enable", b"enable")


@cache
def _runtime_library() -> ctypes.CDLL:
    library = ctypes.CDLL("libascendcl.so")
    library.aclrtMemset.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t)
    library.aclrtMemset.restype = ctypes.c_int
    return library


def _packed_nz(shape: tuple[int, ...], dtype: torch.dtype, byte_value: int) -> torch.Tensor:
    tensor = torch_npu.empty_with_format(shape, dtype=dtype, device="npu", acl_format=torch_npu.Format.FRACTAL_NZ)
    size = tensor.untyped_storage().nbytes()
    # A uniform byte pattern initializes the physical NZ storage directly.
    assert _runtime_library().aclrtMemset(tensor.data_ptr(), size, byte_value, size) == 0
    return tensor


def _make_inputs(nz_cache: bool = False) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(239)
    qkv_width = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK
    q_width = _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM)

    def _ones(shape: int | tuple[int, ...], dtype: torch.dtype = _DTYPE) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype).npu()

    def _zeros(shape: int | tuple[int, ...], dtype: torch.dtype = _DTYPE) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype).npu()

    def _cache(width: int) -> torch.Tensor:
        shape = (_BLOCK_NUM, _BLOCK_SIZE, width)
        if nz_cache:
            return _packed_nz(shape, _DTYPE, 255)
        return torch.full(shape, float("nan"), dtype=_DTYPE).npu()

    return {
        "hidden": torch.rand((_TOKEN_NUM, _HIDDEN_SIZE), generator=generator).to(_DTYPE).npu(),
        "input_norm_weight": _ones(_HIDDEN_SIZE),
        "input_norm_bias": _zeros(_HIDDEN_SIZE),
        "qkv_input_scale": _ones(1),
        "qkv_input_offset": _zeros(1, torch.int8),
        "qkv_weight": _packed_nz((qkv_width, _HIDDEN_SIZE), torch.int8, 1),
        "qkv_deq_scale": _ones(qkv_width, torch.float32),
        "qkv_quant_bias": _zeros(qkv_width, torch.int32),
        "q_norm_weight": _ones(_Q_LORA_RANK),
        "q_norm_bias": _zeros(_Q_LORA_RANK),
        "q_b_input_scale": _ones(1),
        "q_b_input_offset": _zeros(1, torch.int8),
        "q_b_weight": _packed_nz((q_width, _Q_LORA_RANK), torch.int8, 1),
        "q_b_deq_scale": _ones(q_width, torch.float32),
        "q_b_quant_bias": _zeros(q_width, torch.int32),
        "kv_norm_weight": _ones(_KV_LORA_RANK),
        "rope_cos": _ones((_TOKEN_NUM, _QK_ROPE_HEAD_DIM)),
        "rope_sin": _zeros((_TOKEN_NUM, _QK_ROPE_HEAD_DIM)),
        "w_uk": torch.rand((_NUM_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK), generator=generator).to(_DTYPE).npu(),
        "kv_cache": _cache(_KV_LORA_RANK),
        "rope_cache": _cache(_QK_ROPE_HEAD_DIM),
        "slot_mapping": torch.arange(_TOKEN_NUM, dtype=torch.int32).npu(),
        "cache_mode": 3 if nz_cache else 1,
        "q_lora_rank": _Q_LORA_RANK,
        "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
        "norm_epsilon": 1e-5,
    }


def test_mla_preprocess_v2_q_b_preserves_logical_weight() -> None:
    device = torch.device("npu")
    head_dim = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
    rows = _NUM_HEADS * head_dim
    weight = torch.arange(rows * _Q_LORA_RANK, device=device).remainder(127).to(torch.int8).view(rows, _Q_LORA_RANK)
    descale = torch.arange(rows, dtype=torch.float32, device=device)
    bias = torch.arange(rows, dtype=torch.int32, device=device)
    prepared_weight, prepared_descale, prepared_bias = prepare_mla_preprocess_v2_q_b(
        weight, descale, bias, _NUM_HEADS, _QK_NOPE_HEAD_DIM, _QK_ROPE_HEAD_DIM
    )

    # NZ changes the storage layout, not the logical matrix dimensions.
    assert prepared_weight.shape == weight.shape
    assert torch_npu.get_npu_format(prepared_weight) == torch_npu.Format.FRACTAL_NZ
    expected_rows = torch.arange(rows).view(_NUM_HEADS, head_dim)
    expected_rows = torch.cat(
        (
            expected_rows[:, :_QK_NOPE_HEAD_DIM],
            expected_rows[:, _QK_NOPE_HEAD_DIM::2],
            expected_rows[:, _QK_NOPE_HEAD_DIM + 1 :: 2],
        ),
        dim=1,
    ).flatten()
    torch.testing.assert_close(prepared_weight.cpu(), weight.cpu()[expected_rows])
    torch.testing.assert_close(prepared_descale.cpu(), descale.cpu()[expected_rows])
    torch.testing.assert_close(prepared_bias.cpu(), bias.cpu()[expected_rows])


def test_mla_preprocess_v2_writes_outputs_and_selected_cache_slots() -> None:
    inputs = _make_inputs()
    q_c, q_latent, q_pe = mla_preprocess_v2(**inputs)

    assert q_latent.shape == (_TOKEN_NUM, _NUM_HEADS, _KV_LORA_RANK)
    assert q_pe.shape == (_TOKEN_NUM, _NUM_HEADS, _QK_ROPE_HEAD_DIM)
    assert q_c.shape == (_TOKEN_NUM, _Q_LORA_RANK)
    for output in (q_c, q_latent, q_pe):
        host_output = output.cpu()
        assert torch.isfinite(host_output).all() and torch.count_nonzero(host_output)
    for name in ("kv_cache", "rope_cache"):
        actual = inputs[name].cpu()
        assert torch.isfinite(actual[0, :_TOKEN_NUM]).all()
        assert torch.isnan(actual[0, _TOKEN_NUM:]).all() and torch.isnan(actual[1]).all()


def test_mla_preprocess_v2_nz_cache_matches_nd() -> None:
    nd_inputs = _make_inputs()
    nz_inputs = _make_inputs(nz_cache=True)
    nd_outputs = mla_preprocess_v2(**nd_inputs)
    nz_outputs = mla_preprocess_v2(**nz_inputs)

    torch.testing.assert_close(nz_outputs, nd_outputs, rtol=0, atol=0)
    for name in ("kv_cache", "rope_cache"):
        torch.testing.assert_close(nz_inputs[name].cpu(), nd_inputs[name].cpu(), rtol=0, atol=0, equal_nan=True)


def test_mla_preprocess_v2_aclgraph_replays_q_and_kv() -> None:
    graph_inputs = _make_inputs()
    eager_inputs = _make_inputs()
    mla_preprocess_v2(**graph_inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_outputs = mla_preprocess_v2(**graph_inputs)

    generator = torch.Generator().manual_seed(239)
    for block in (1, 0, 1):
        hidden = torch.randn(graph_inputs["hidden"].shape, generator=generator).to(_DTYPE)
        slots = torch.arange(_TOKEN_NUM, dtype=torch.int32) + block * _BLOCK_SIZE
        for inputs in (graph_inputs, eager_inputs):
            inputs["hidden"].copy_(hidden)
            inputs["slot_mapping"].copy_(slots)
            for name in ("kv_cache", "rope_cache"):
                inputs[name].copy_(torch.full(inputs[name].shape, float("nan"), dtype=_DTYPE))
        graph.replay()
        torch.npu.synchronize()
        eager_outputs = mla_preprocess_v2(**eager_inputs)
        torch.testing.assert_close(graph_outputs, eager_outputs, rtol=0, atol=0)
        for name in ("kv_cache", "rope_cache"):
            actual, expected = graph_inputs[name].cpu(), eager_inputs[name].cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            assert torch.isfinite(actual[block, :_TOKEN_NUM]).all()
            assert torch.isnan(actual[1 - block]).all()
