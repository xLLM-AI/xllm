# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shape and dtype contracts for the NPU DeepSeek-V4 fake kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = Path(__file__).parents[2]


def _load_fake_kernels(monkeypatch: pytest.MonkeyPatch):
    """Load fake implementations without requiring compiled NPU schemas."""

    class _OperatorNamespace:
        def __getattr__(self, name: str) -> object:
            del name
            return object()

    real_ops = torch.ops
    monkeypatch.setattr(
        torch,
        "ops",
        SimpleNamespace(xllm_ops=_OperatorNamespace()),
    )
    monkeypatch.setattr(
        torch.library,
        "register_fake",
        lambda qualname: (lambda function: function),
    )

    path = _REPO_ROOT / "xllm/python/kernels_npu/_custom_op.py"
    spec = importlib.util.spec_from_file_location("npu_fake_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch, "ops", real_ops)
    return module


def test_dsv4_fake_tensor_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _load_fake_kernels(monkeypatch)

    with torch._subclasses.fake_tensor.FakeTensorMode():
        hidden = torch.empty(8, 16)
        normalized, norm_scale = fake._rms_norm_dynamic_quant_fake(hidden, torch.empty(16), 1e-6)
        assert normalized.shape == hidden.shape
        assert normalized.dtype == torch.int8
        assert norm_scale.shape == (8,)
        assert norm_scale.dtype == torch.float32

        rotary_input = torch.empty(8, 2, 128)
        assert (
            fake._npu_inplace_partial_rotary_mul_fake(
                rotary_input,
                torch.empty(8, 64),
                torch.empty(8, 64),
                "interleave",
                [64, 128],
            )
            is None
        )

        hc_input = torch.empty(8, 4, 16)
        attn_input, post, comb = fake._hc_pre_fake(
            hc_input,
            torch.empty(24, 64),
            torch.empty(3),
            torch.empty(24),
            4,
            20,
            1e-6,
            1e-6,
        )
        assert attn_input.shape == (8, 16)
        assert post.shape == (8, 4)
        assert comb.shape == (8, 4, 4)
        assert fake._hc_post_fake(attn_input, hc_input, post, comb).shape == hc_input.shape

        compressor_out = fake._compressor_fake(
            torch.empty(1, 8, 16),
            torch.empty(8, 16),
            torch.empty(8, 16),
            torch.empty(1, 128, 8),
            torch.empty(1, 128, 8),
            torch.empty(4, 8),
            torch.empty(16),
            torch.empty(2, 4),
            torch.empty(2, 4),
            None,
            None,
            None,
            None,
            None,
            4,
            4,
            1,
            1e-6,
            1,
            False,
        )
        assert compressor_out[0].shape == (1, 2, 16)
        assert all(tensor.numel() == 0 for tensor in compressor_out[1:])

        query = torch.empty(1, 4, 64, 512)
        sparse_out, sparse_lse = fake._sparse_attn_sharedkv_fake(
            query,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.empty(1024, dtype=torch.int32),
            1.0,
            1,
            4,
            3,
            127,
            0,
            "BSND",
            "PA_ND",
            False,
        )
        assert sparse_out.shape == query.shape
        assert sparse_lse.shape == (0,)
        assert sparse_lse.dtype == torch.float32

        _, sparse_lse = fake._sparse_attn_sharedkv_fake(
            query,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.empty(1024, dtype=torch.int32),
            1.0,
            1,
            4,
            3,
            127,
            0,
            "BSND",
            "PA_ND",
            True,
        )
        assert sparse_lse.shape == (1, 4, 64, 1)
        assert sparse_lse.dtype == torch.float32

        seq_lens = torch.empty(1, dtype=torch.int32)
        metadata = fake._sparse_attn_sharedkv_metadata_fake(
            64,
            1,
            512,
            None,
            None,
            None,
            seq_lens,
            seq_lens,
            1,
            4,
            16,
            0,
            0,
            1,
            4,
            3,
            127,
            0,
            "BSND",
            "PA_ND",
            True,
            False,
        )
        assert metadata.shape == (1024,)
        assert metadata.dtype == torch.int32

        qli_metadata = fake._quant_lightning_indexer_metadata_fake(
            64,
            1,
            128,
            0,
            0,
            seq_lens,
            seq_lens,
            1,
            8,
            8,
            "TND",
            "PA_BSND",
            512,
            3,
            2**63 - 1,
            2**63 - 1,
            4,
            "cpu",
        )
        assert qli_metadata.shape == (1024,)
        assert qli_metadata.dtype == torch.int32

        qli_indices, qli_values = fake._quant_lightning_indexer_fake(
            torch.empty(8, 64, 128, dtype=torch.int8),
            torch.empty(1, 128, 1, 128, dtype=torch.int8),
            torch.empty(8, 64),
            torch.empty(8, 64),
            torch.empty(1, 128, 1),
            0,
            0,
            seq_lens,
            seq_lens,
            torch.empty(1, 1),
            qli_metadata,
            "TND",
            "PA_BSND",
            512,
            3,
            2**63 - 1,
            2**63 - 1,
            4,
            False,
        )
        assert qli_indices.shape == (8, 1, 512)
        assert qli_indices.dtype == torch.int32
        assert qli_values.numel() == 0

        gate_weights, expert_ids, gate_output = fake._moe_gating_top_k_hash_fake(
            torch.empty(8, 256),
            6,
            None,
            None,
            None,
            1,
            1,
            1.0,
            1e-20,
            1,
            0,
            2,
            False,
        )
        assert gate_weights.shape == (8, 6)
        assert expert_ids.shape == (8, 6)
        assert expert_ids.dtype == torch.int32
        assert gate_output.shape == (8, 256)
        assert gate_output.dtype == torch.float32

        swiglu_out, swiglu_scale = fake._dequant_swiglu_quant_fake(
            torch.empty(2, 8, 32, dtype=torch.int32),
            None,
            None,
            None,
            None,
            None,
            None,
            True,
            1,
            1,
            0.0,
            1.0,
            0.0,
        )
        assert swiglu_out.shape == (2, 8, 16)
        assert swiglu_out.dtype == torch.int8
        assert swiglu_scale.shape == (2, 8)
        assert swiglu_scale.dtype == torch.float32
