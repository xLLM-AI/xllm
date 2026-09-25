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

from xllm.python import kernels, registry
from xllm.python.layers import DFlash2GroupedConv
from xllm.python.models.qwen3_dflash2 import (
    DFlash2CandidateSelector,
    DFlash2Qwen3Config,
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)


def _config_dict(**overrides) -> dict:
    values = {
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 16,
        "sliding_window": 8,
        "vocab_size": 8,
        "draft_vocab_size": 8,
        "dflash2_block_size": 4,
        "dflash2_conv_group_size": 2,
        "dflash2_conv_kernel_size": 2,
        "dflash2_selector_rank": 2,
        "dflash2_selector_top_k": 2,
        "tp_size": 1,
        "tp_rank": 0,
        "dp_size": 1,
        "dp_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def _config(**overrides) -> DFlash2Qwen3Config:
    config = DFlash2Qwen3Config.from_dict(_config_dict(**overrides))
    config.validate()
    return config


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def test_config_accepts_nested_and_reflected_dflash_geometry() -> None:
    reflected = _config()
    nested_values = _config_dict(
        dflash2_block_size=None,
        dflash2_conv_group_size=None,
        dflash2_conv_kernel_size=None,
        dflash2_selector_rank=None,
        dflash2_selector_top_k=None,
        dflash_config={
            "block_size": 4,
            "conv_group_size": 2,
            "conv_kernel_size": 2,
            "selector_rank": 2,
            "selector_top_k": 2,
        },
    )
    nested = DFlash2Qwen3Config.from_dict(nested_values)
    nested.validate()

    speculators_values = _config_dict(
        dflash2_block_size=None,
        dflash2_conv_group_size=None,
        dflash2_conv_kernel_size=None,
        dflash2_selector_rank=None,
        dflash2_selector_top_k=None,
        block_size=4,
        conv_group_size=2,
        conv_kernel_size=2,
        selector_rank=2,
        selector_top_k=2,
    )
    speculators = DFlash2Qwen3Config.from_dict(speculators_values)
    speculators.validate()

    assert nested.block_size == reflected.block_size
    assert nested.conv_group_size == reflected.conv_group_size
    assert nested.conv_kernel_size == reflected.conv_kernel_size
    assert nested.selector_rank == reflected.selector_rank
    assert nested.selector_top_k == reflected.selector_top_k
    assert speculators.block_size == reflected.block_size
    assert speculators.conv_group_size == reflected.conv_group_size
    assert speculators.conv_kernel_size == reflected.conv_kernel_size
    assert speculators.selector_rank == reflected.selector_rank
    assert speculators.selector_top_k == reflected.selector_top_k


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dflash2_block_size": 0}, "geometry values"),
        ({"dflash2_conv_group_size": 3}, "must divide hidden_size"),
        ({"sliding_window": 0}, "positive sliding_window"),
        ({"dflash2_selector_top_k": 9}, "must not exceed vocab_size"),
    ],
)
def test_config_rejects_invalid_dflash2_geometry(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_grouped_convolution_resets_history_at_block_boundaries() -> None:
    convolution = DFlash2GroupedConv(
        hidden_size=4,
        taps=2,
        group_size=2,
        block_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    hidden = torch.arange(32, dtype=torch.float32).view(8, 4)
    with torch.no_grad():
        convolution.kernel_projection.weight.zero_()
        convolution.base_kernel.zero_()
        convolution.base_kernel[0, 0].fill_(1.0)
        convolution.base_kernel[0, 1].fill_(10.0)

    output, coefficients = convolution.prepare(hidden)
    expected = hidden.clone()
    expected[1:4] += 10.0 * hidden[0:3]
    expected[5:8] += 10.0 * hidden[4:7]

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(coefficients, torch.zeros_like(coefficients))


def test_candidate_selector_builds_topk_edge_logits() -> None:
    selector = DFlash2CandidateSelector(
        hidden_size=4,
        vocab_size=4,
        rank=2,
        top_k=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        selector.hidden_projection.weight.zero_()
        selector.predecessor_codebook.zero_()
        selector.successor_codebook.zero_()
    logits = torch.tensor([[[0.0, 3.0, 1.0, 2.0], [4.0, 1.0, 5.0, 0.0]]])

    candidate_ids, edge_logits = selector(
        torch.ones(1, 2, 4),
        logits,
        torch.tensor([1]),
    )

    expected_ids = torch.tensor([[[1, 3], [2, 0]]])
    expected_values = torch.tensor([[[3.0, 2.0], [5.0, 4.0]]])
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(
        edge_logits,
        expected_values.unsqueeze(2).expand(1, 2, 2, 2),
    )


def test_dflash2_model_uses_band_attention_and_shared_target_weights() -> None:
    config = _config()
    model = DFlash2Qwen3Model(config, torch.float32, torch.device("cpu"))
    attention = model.layers[0].self_attn.attn

    assert model.embed_tokens is None
    assert not attention.causal
    assert attention.fia_sparse_mode == 4
    assert attention.fia_pre_tokens == config.sliding_window - 1
    assert attention.fia_next_tokens == config.block_size - 1
    assert attention.fia_use_attention_mask


def test_checkpoint_weights_load_without_draft_embedding_or_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        lambda weight: (weight, False),
        raising=False,
    )
    model = DFlash2Qwen3ForCausalLM(_config_dict())
    tensors = {
        "embed_tokens.weight": torch.full((8, 4), 99.0),
        "lm_head.weight": torch.full((8, 4), 98.0),
        "fc.weight": torch.ones(4, 8),
        "hidden_norm.weight": torch.ones(4),
        "candidate_selector.hidden_projection.weight": torch.full((2, 4), 2.0),
        "candidate_selector.predecessor_codebook": torch.full((8, 2), 3.0),
        "candidate_selector.successor_codebook": torch.full((8, 2), 4.0),
        "layers.0.input_layernorm.weight": torch.ones(4),
        "layers.0.post_attention_layernorm.weight": torch.ones(4),
        "layers.0.self_attn.q_norm.weight": torch.ones(4),
        "layers.0.self_attn.k_norm.weight": torch.ones(4),
        "layers.0.self_attn.q_proj.weight": torch.full((4, 4), 1.0),
        "layers.0.self_attn.k_proj.weight": torch.full((4, 4), 2.0),
        "layers.0.self_attn.v_proj.weight": torch.full((4, 4), 3.0),
        "layers.0.self_attn.o_proj.weight": torch.full((4, 4), 4.0),
        "layers.0.mlp.gate_proj.weight": torch.full((8, 4), 5.0),
        "layers.0.mlp.up_proj.weight": torch.full((8, 4), 6.0),
        "layers.0.mlp.down_proj.weight": torch.full((4, 8), 7.0),
        "layers.0.attention_conv.base_kernel": torch.full((2, 2, 4), 8.0),
        "layers.0.attention_conv.kernel_projection.weight": torch.full((8, 4), 9.0),
        "layers.0.mlp_conv.base_kernel": torch.full((2, 2, 4), 10.0),
        "layers.0.mlp_conv.kernel_projection.weight": torch.full((8, 4), 11.0),
        "norm.weight": torch.ones(4),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    assert model.model.embed_tokens is None
    assert model.lm_head is None
    torch.testing.assert_close(
        model.model.candidate_selector.hidden_projection.weight,
        tensors["candidate_selector.hidden_projection.weight"],
    )
    torch.testing.assert_close(
        model.model.layers[0].attention_conv.base_kernel,
        tensors["layers.0.attention_conv.base_kernel"],
    )
    torch.testing.assert_close(
        model.model.layers[0].mlp_conv.kernel_projection.weight,
        tensors["layers.0.mlp_conv.kernel_projection.weight"],
    )


def test_registry_resolves_dflash2_python_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")

    assert registry.get_model_class("DFlash2DraftModel") is DFlash2Qwen3ForCausalLM
