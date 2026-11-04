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

from __future__ import annotations

import pytest
import torch

from xllm.python.models.qwen3 import Qwen3ForCausalLM


@pytest.fixture
def model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        {
            "hidden_size": 4,
            "n_layers": 0,
            "n_heads": 1,
            "n_kv_heads": 1,
            "head_dim": 4,
            "intermediate_size": 8,
            "vocab_size": 8,
            "max_position_embeddings": 8,
            "dtype": "float32",
            "device": "cpu",
        }
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_pooler_selects_then_normalizes_without_mutating_hidden(
    model: Qwen3ForCausalLM, dtype: torch.dtype, index_dtype: torch.dtype
) -> None:
    hidden = torch.tensor([[3, 0, -3], [4, 5, -4]], dtype=dtype).t()
    original = hidden.clone()
    indices = torch.tensor([2, 0, 2], dtype=index_dtype)
    expected = torch.tensor([[-0.6, -0.8], [0.6, 0.8], [-0.6, -0.8]], dtype=dtype)
    result = model.pooler(hidden, indices)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_none_and_empty_indices_have_distinct_semantics(model: Qwen3ForCausalLM, dtype: torch.dtype) -> None:
    hidden = torch.tensor([[0, 5], [0, 0]], dtype=dtype)
    torch.testing.assert_close(model.pooler(hidden, None), torch.tensor([[0, 1], [0, 0]], dtype=dtype))
    empty = model.pooler(hidden, torch.empty(0, dtype=torch.int64))
    assert empty.shape == (0, 2)
    assert empty.dtype == dtype


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_selected_eos_row_is_not_confused_with_padding(model: Qwen3ForCausalLM, padding_side: str) -> None:
    # EOS and padding may share an ID. The input builder owns validity, so
    # pooling must obey its row selection instead of interpreting token IDs.
    eos = torch.tensor([3.0, 4.0])
    padding = torch.tensor([100.0, 0.0])
    token = torch.tensor([0.0, 2.0])
    rows = [padding, token, eos] if padding_side == "left" else [token, eos, padding]
    index = 2 if padding_side == "left" else 1
    batch = torch.stack(rows + [token, token, eos])
    pooled = model.pooler(batch, torch.tensor([index, 5]))
    single = model.pooler(torch.stack([token, eos]), torch.tensor([1]))
    torch.testing.assert_close(pooled, single.expand(2, -1), rtol=0, atol=0)


def test_pooling_keeps_generation_logits_unchanged(model: Qwen3ForCausalLM) -> None:
    hidden = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    selected = torch.tensor([1, 0])
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.arange(32).reshape(8, 4).float() / 32)
        before = model.compute_logits(hidden, selected)
        model.pooler(hidden, selected)
        after = model.compute_logits(hidden, selected)
    expected = torch.nn.functional.linear(hidden.index_select(0, selected), model.lm_head.weight)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)
