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

"""NPU smoke and integration tests for the fused MLA preprocess path."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest
import torch

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
_CACHE_MODE = 1  # krope_ctkv
_QUANT_MODE = 0  # per_tensor_quant_asymm


@pytest.fixture(name="mla")
def _mla_with_internal_format() -> Iterator[ModuleType]:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from xllm.python.kernels_npu import mla

    if not mla.has_mla_preprocess_v2():
        pytest.skip("aclnnMlaPreprocessV2 is not available in the loaded runtime")
    # Fused MLA weights require NZ storage. Restore the process setting afterward.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    yield mla
    torch.npu.config.allow_internal_format = previous in ("enable", b"enable")


def _make_inputs(mla: ModuleType) -> dict[str, torch.Tensor | int | float | bool]:
    device = torch.device("npu")
    torch.manual_seed(0)

    qkv_weight = torch.randint(
        0,
        7,
        (_KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK, _HIDDEN_SIZE),
        dtype=torch.int8,
        device=device,
    )
    qkv_deq_scale = torch.ones(
        _KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK,
        dtype=torch.float32,
        device=device,
    )
    qkv_quant_bias = torch.zeros(
        _KV_LORA_RANK + _QK_ROPE_HEAD_DIM + _Q_LORA_RANK,
        dtype=torch.int32,
        device=device,
    )
    qkv_weight, qkv_deq_scale, qkv_quant_bias = mla.prepare_mla_preprocess_v2_qkv(
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        _KV_LORA_RANK,
        _QK_ROPE_HEAD_DIM,
    )

    q_b_weight = torch.randint(
        0,
        7,
        (_NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM), _Q_LORA_RANK),
        dtype=torch.int8,
        device=device,
    )
    q_b_deq_scale = torch.ones(
        _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    q_b_quant_bias = torch.zeros(
        _NUM_HEADS * (_QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM),
        dtype=torch.int32,
        device=device,
    )
    q_b_weight, q_b_deq_scale, q_b_quant_bias = mla.prepare_mla_preprocess_v2_q_b(
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        _NUM_HEADS,
        _QK_NOPE_HEAD_DIM,
        _QK_ROPE_HEAD_DIM,
    )

    w_uk = torch.randn(
        (_NUM_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK),
        dtype=_DTYPE,
        device=device,
    )
    w_uk = torch_npu.npu_format_cast(w_uk, 29)

    inputs: dict[str, torch.Tensor | int | float | bool] = {
        "hidden": torch.randn((_TOKEN_NUM, _HIDDEN_SIZE), dtype=_DTYPE, device=device),
        "input_norm_weight": torch.ones(_HIDDEN_SIZE, dtype=_DTYPE, device=device),
        "input_norm_bias": torch.zeros(_HIDDEN_SIZE, dtype=_DTYPE, device=device),
        "qkv_input_scale": torch.tensor([0.25], dtype=_DTYPE, device=device),
        "qkv_input_offset": torch.zeros(1, dtype=torch.int8, device=device),
        "qkv_weight": qkv_weight,
        "qkv_deq_scale": qkv_deq_scale,
        "qkv_quant_bias": qkv_quant_bias,
        "q_norm_weight": torch.ones(_Q_LORA_RANK, dtype=_DTYPE, device=device),
        "q_norm_bias": torch.zeros(_Q_LORA_RANK, dtype=_DTYPE, device=device),
        "q_b_input_scale": torch.tensor([0.25], dtype=_DTYPE, device=device),
        "q_b_input_offset": torch.zeros(1, dtype=torch.int8, device=device),
        "q_b_weight": q_b_weight,
        "q_b_deq_scale": q_b_deq_scale,
        "q_b_quant_bias": q_b_quant_bias,
        "kv_norm_weight": torch.ones(_KV_LORA_RANK, dtype=_DTYPE, device=device),
        "rope_cos": torch.randn((_TOKEN_NUM, _QK_ROPE_HEAD_DIM), dtype=_DTYPE, device=device),
        "rope_sin": torch.randn((_TOKEN_NUM, _QK_ROPE_HEAD_DIM), dtype=_DTYPE, device=device),
        "w_uk": w_uk,
        "kv_cache": torch.full(
            (_BLOCK_NUM, _BLOCK_SIZE, _KV_LORA_RANK),
            float("nan"),
            dtype=_DTYPE,
            device=device,
        ),
        "rope_cache": torch.full(
            (_BLOCK_NUM, _BLOCK_SIZE, _QK_ROPE_HEAD_DIM),
            float("nan"),
            dtype=_DTYPE,
            device=device,
        ),
        "slot_mapping": torch.tensor([0, 1], dtype=torch.int32, device=device),
        "kv_lora_rank": _KV_LORA_RANK,
        "q_lora_rank": _Q_LORA_RANK,
        "qk_rope_head_dim": _QK_ROPE_HEAD_DIM,
        "norm_epsilon": 1e-5,
    }
    return inputs


def _run_low_level_op(inputs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    num_tokens = inputs["hidden"].shape[0]
    return torch.ops.xllm_ops.mla_preprocess_v2(
        inputs["hidden"],
        inputs["input_norm_weight"],
        inputs["input_norm_bias"],
        inputs["qkv_input_scale"],
        inputs["qkv_input_offset"],
        inputs["qkv_weight"],
        inputs["qkv_deq_scale"],
        inputs["qkv_quant_bias"],
        inputs["q_norm_weight"],
        inputs["q_norm_bias"],
        inputs["q_b_input_scale"],
        inputs["q_b_input_offset"],
        inputs["q_b_weight"],
        inputs["q_b_deq_scale"],
        inputs["q_b_quant_bias"],
        inputs["kv_norm_weight"],
        inputs["rope_cos"].view(num_tokens, -1),
        inputs["rope_sin"].view(num_tokens, -1),
        inputs["w_uk"],
        inputs["kv_cache"],
        inputs["rope_cache"],
        inputs["slot_mapping"],
        inputs["qkv_input_scale"],
        inputs["q_b_input_scale"],
        inputs["q_lora_rank"],
        inputs["qk_rope_head_dim"],
        inputs["qk_rope_head_dim"],
        inputs["norm_epsilon"],
        2,
        2,
        True,
        True,
        True,
        _CACHE_MODE,
        _QUANT_MODE,
        False,
        1,
        True,
    )


def test_mla_preprocess_v2_q_b_preserves_logical_weight(mla: ModuleType) -> None:
    device = torch.device("npu")
    head_dim = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
    rows = _NUM_HEADS * head_dim
    weight = torch.arange(rows * _Q_LORA_RANK, device=device).remainder(127).to(torch.int8).view(rows, _Q_LORA_RANK)
    descale = torch.arange(rows, dtype=torch.float32, device=device)
    bias = torch.arange(rows, dtype=torch.int32, device=device)
    prepared_weight, prepared_descale, prepared_bias = mla.prepare_mla_preprocess_v2_q_b(
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


def test_mla_preprocess_v2_writes_outputs_and_selected_cache_slots(mla: ModuleType) -> None:
    inputs = _make_inputs(mla)

    q_latent, kv_cache, q_pe, rope_cache, q_c = _run_low_level_op(inputs)
    torch.npu.synchronize()

    assert q_latent.shape == (_TOKEN_NUM, _NUM_HEADS, _KV_LORA_RANK)
    assert q_pe.shape == (_TOKEN_NUM, _NUM_HEADS, _QK_ROPE_HEAD_DIM)
    assert q_c.shape == (_TOKEN_NUM, _Q_LORA_RANK)
    assert torch.isfinite(q_latent).all()
    assert torch.isfinite(q_pe).all()
    assert torch.isfinite(q_c).all()
    assert torch.isfinite(kv_cache[0, :_TOKEN_NUM]).all()
    assert torch.isfinite(rope_cache[0, :_TOKEN_NUM]).all()
    assert torch.isnan(kv_cache[1]).all()
    assert torch.isnan(rope_cache[1]).all()


def test_mla_preprocess_v2_python_wrapper_matches_low_level_op(mla: ModuleType) -> None:
    low_level_inputs = _make_inputs(mla)
    # Rebuild with the same seed: clone() can discard private NPU formats.
    wrapper_inputs = _make_inputs(mla)

    low_level_outputs = _run_low_level_op(low_level_inputs)
    wrapper_outputs = mla.deepseek_mla_preprocess_decode_v2(
        wrapper_inputs["hidden"],
        wrapper_inputs["input_norm_weight"],
        wrapper_inputs["input_norm_bias"],
        wrapper_inputs["qkv_input_scale"],
        wrapper_inputs["qkv_input_offset"],
        wrapper_inputs["qkv_weight"],
        wrapper_inputs["qkv_deq_scale"],
        wrapper_inputs["qkv_quant_bias"],
        wrapper_inputs["q_norm_weight"],
        wrapper_inputs["q_norm_bias"],
        wrapper_inputs["q_b_input_scale"],
        wrapper_inputs["q_b_input_offset"],
        wrapper_inputs["q_b_weight"],
        wrapper_inputs["q_b_deq_scale"],
        wrapper_inputs["q_b_quant_bias"],
        wrapper_inputs["kv_norm_weight"],
        wrapper_inputs["rope_cos"],
        wrapper_inputs["rope_sin"],
        wrapper_inputs["w_uk"],
        wrapper_inputs["kv_cache"],
        wrapper_inputs["rope_cache"],
        wrapper_inputs["slot_mapping"],
        _KV_LORA_RANK,
        _Q_LORA_RANK,
        _QK_ROPE_HEAD_DIM,
        1e-5,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(wrapper_outputs[0], low_level_outputs[4], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(wrapper_outputs[1], low_level_outputs[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(wrapper_outputs[2], low_level_outputs[2], rtol=1e-2, atol=1e-2)
