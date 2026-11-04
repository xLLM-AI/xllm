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

"""Qwen3 embedding selection, normalization, and generation regressions."""

from __future__ import annotations

import pytest
import torch

from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3 import Qwen3ForCausalLM


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _model(*, embedding_mode: bool = False, tie_word_embeddings: bool = False) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        {
            "hidden_size": 4,
            "num_hidden_layers": 0,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 8,
            "max_position_embeddings": 8,
            "vocab_size": 8,
            "dtype": "float32",
            "device": "cpu",
            "embedding_mode": embedding_mode,
            "tie_word_embeddings": tie_word_embeddings,
        }
    )


@pytest.mark.parametrize("device", ["cpu", "npu:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_pooler_preserves_selected_order_and_ignores_batch_padding(
    device: str, dtype: torch.dtype, index_dtype: torch.dtype
) -> None:
    if device.startswith("npu") and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        pytest.skip("requires Ascend NPU")
    model = _model(embedding_mode=True)
    # Two packed prompts end at rows 1 and 4. Row 4 represents an explicit
    # EOS; row 5 is execution padding, deliberately pointing elsewhere.
    hidden = torch.tensor(
        [[1, 1, 1, 1], [3, 4, 0, 0], [2, 2, 2, 2], [1, 2, 3, 4], [0, 0, 5, 12], [99, 99, 99, 99]],
        dtype=dtype,
        device=device,
    )
    before = hidden.clone()
    selected = torch.tensor([4, 1, 4], dtype=index_dtype, device=device)

    pooled = model.pooler(hidden, selected)

    expected = torch.tensor([[0, 0, 5 / 13, 12 / 13], [3 / 5, 4 / 5, 0, 0], [0, 0, 5 / 13, 12 / 13]])
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(pooled.float().cpu(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(pooled.float().norm(dim=1).cpu(), torch.ones(3), rtol=tolerance, atol=tolerance)
    for row, index in enumerate((4, 1, 4)):
        single = model.pooler(hidden[index : index + 1], None)
        torch.testing.assert_close(pooled[row : row + 1], single, rtol=0, atol=0)
    torch.testing.assert_close(hidden, before, rtol=0, atol=0)
    assert pooled.dtype == dtype
    assert pooled.device == hidden.device


@pytest.mark.parametrize("indices", [None, [], [2, 0]])
def test_default_pooler_keeps_unnormalized_values(indices: list[int] | None) -> None:
    hidden = torch.tensor([[3.0, 4.0], [5.0, 12.0], [8.0, 15.0]])
    selected = None if indices is None else torch.tensor(indices, dtype=torch.int32)

    pooled = PyModelBase().pooler(hidden, selected)

    expected = hidden if indices is None else hidden[indices]
    torch.testing.assert_close(pooled, expected, rtol=0, atol=0)


def test_qwen3_pooler_distinguishes_none_from_empty_indices() -> None:
    model = _model(embedding_mode=True)
    hidden = torch.tensor([[3.0, 4.0, 0.0, 0.0]])

    torch.testing.assert_close(model.pooler(hidden, None), torch.tensor([[0.6, 0.8, 0.0, 0.0]]))
    assert model.pooler(hidden, torch.empty(0, dtype=torch.int32)).shape == (0, 4)
    assert model.pooler(torch.empty(0, 4), None).shape == (0, 4)


def test_qwen3_pooler_matches_atb_epsilon_for_zero_and_small_norms() -> None:
    hidden = torch.tensor([[0.0, 0.0, 0.0, 0.0], [3e-14, 4e-14, 0.0, 0.0]])

    pooled = _model(embedding_mode=True).pooler(hidden, None)

    torch.testing.assert_close(pooled, torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.03, 0.04, 0.0, 0.0]]))


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_embedding_checkpoint_loads_without_output_head(tie_word_embeddings: bool) -> None:
    model = _model(embedding_mode=True, tie_word_embeddings=tie_word_embeddings)
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    norm = torch.tensor([1.0, 2.0, 3.0, 4.0])

    model.load_weights([_StateDict({"embed_tokens.weight": embedding, "norm.weight": norm})], 0, 1)

    assert model.lm_head is None
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.model.norm.weight, norm)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_embedding_checkpoint_with_odd_vocab_loads_hidden_shards(tp_rank: int) -> None:
    config = {
        "hidden_size": 4,
        "num_hidden_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "intermediate_size": 8,
        "max_position_embeddings": 8,
        "vocab_size": 9,
        "tp_size": 2,
        "tp_rank": tp_rank,
        "dtype": "float32",
        "device": "cpu",
        "embedding_mode": True,
    }
    model = Qwen3ForCausalLM(config)
    embedding = torch.arange(36, dtype=torch.float32).reshape(9, 4)

    model.load_weights([_StateDict({"embed_tokens.weight": embedding, "norm.weight": torch.ones(4)})], tp_rank, 2)

    assert model.lm_head is None
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding[:, tp_rank * 2 : (tp_rank + 1) * 2])
    with pytest.raises(AssertionError):
        Qwen3ForCausalLM(config | {"embedding_mode": False})


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_generation_head_and_logits_are_unchanged_by_pooling(tie_word_embeddings: bool) -> None:
    model = _model(tie_word_embeddings=tie_word_embeddings)
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    head = embedding.flip(0)
    state = {"model.embed_tokens.weight": embedding, "model.norm.weight": torch.ones(4)}
    if not tie_word_embeddings:
        state["lm_head.weight"] = head
    model.load_weights([_StateDict(state)], 0, 1)
    hidden = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 12.0]])
    selected = torch.tensor([1, 0], dtype=torch.int64)
    expected = hidden[selected] @ (embedding if tie_word_embeddings else head).t()

    before = model.compute_logits(hidden, selected)
    model.pooler(hidden, selected)
    after = model.compute_logits(hidden, selected)

    torch.testing.assert_close(before, expected)
    torch.testing.assert_close(after, expected)


def test_generation_checkpoint_still_requires_untied_output_head() -> None:
    model = _model()
    state = _StateDict({"model.embed_tokens.weight": torch.zeros(8, 4), "model.norm.weight": torch.ones(4)})

    with pytest.raises(KeyError, match="lm_head.weight"):
        model.load_weights([state], 0, 1)
