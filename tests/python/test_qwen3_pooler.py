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


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


@pytest.fixture
def config() -> dict[str, object]:
    return {
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


@pytest.fixture
def model(config: dict[str, object]) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(config)


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


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize("prefix", ["", "model."])
def test_embedding_checkpoint_loads_without_output_head(
    config: dict[str, object], tie_word_embeddings: bool, prefix: str
) -> None:
    model = Qwen3ForCausalLM(config | {"embedding_mode": True, "tie_word_embeddings": tie_word_embeddings})
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    norm = torch.tensor([1.0, 2.0, 3.0, 4.0])

    model.load_weights([_StateDict({f"{prefix}embed_tokens.weight": embedding, f"{prefix}norm.weight": norm})], 0, 1)

    assert model.lm_head is None
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)
    torch.testing.assert_close(model.model.norm.weight, norm)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_embedding_checkpoint_with_odd_vocab_loads_hidden_shards(config: dict[str, object], tp_rank: int) -> None:
    model = Qwen3ForCausalLM(
        config
        | {
            "embedding_mode": True,
            "vocab_size": 9,
            "tp_size": 2,
            "tp_rank": tp_rank,
            "n_heads": 2,
            "n_kv_heads": 2,
            "head_dim": 2,
        }
    )
    embedding = torch.arange(36, dtype=torch.float32).reshape(9, 4)

    model.load_weights([_StateDict({"embed_tokens.weight": embedding, "norm.weight": torch.ones(4)})], tp_rank, 2)

    assert model.lm_head is None
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding[:, tp_rank * 2 : (tp_rank + 1) * 2])


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_generation_checkpoint_still_loads_output_head_and_computes_logits(
    config: dict[str, object], tie_word_embeddings: bool
) -> None:
    model = Qwen3ForCausalLM(config | {"embedding_mode": False, "tie_word_embeddings": tie_word_embeddings})
    embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    head = embedding.flip(0)
    tensors = {"model.embed_tokens.weight": embedding, "model.norm.weight": torch.ones(4)}
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = head

    model.load_weights([_StateDict(tensors)], 0, 1)

    assert model.lm_head is not None
    expected_head = embedding if tie_word_embeddings else head
    torch.testing.assert_close(model.lm_head.weight, expected_head)
    hidden = torch.tensor([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 12.0]])
    selected = torch.tensor([1, 0])
    torch.testing.assert_close(model.compute_logits(hidden, selected), hidden[selected] @ expected_head.t())


def test_generation_checkpoint_still_requires_untied_output_head(config: dict[str, object]) -> None:
    model = Qwen3ForCausalLM(config | {"tie_word_embeddings": False})
    state = _StateDict({"model.embed_tokens.weight": torch.zeros(8, 4), "model.norm.weight": torch.ones(4)})

    with pytest.raises(KeyError, match="lm_head.weight"):
        model.load_weights([state], 0, 1)


def test_generation_still_requires_vocab_divisible_by_tp(config: dict[str, object]) -> None:
    with pytest.raises(AssertionError):
        Qwen3ForCausalLM(config | {"vocab_size": 9, "tp_size": 2})
