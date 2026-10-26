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

"""Qwen MTP checkpoint and recurrent-hidden computation contracts."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.python.qwen3_5_test_utils import StateDict
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models import qwen3_5, qwen3_5_mtp
from xllm.python.models.qwen3_5 import Qwen3_5Config


class _GemmaNorm(nn.Module):
    def __init__(self, size: int, eps: float, **kwargs: Any) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, **kwargs))
        self.eps = eps

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        summed = hidden if residual is None else hidden + residual
        output = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + self.eps) * (self.weight + 1)
        return output if residual is None else (output, summed)


class _Decoder(nn.Module):
    normalization_cls = _GemmaNorm

    def __init__(
        self, cfg: Qwen3_5Config, _layer_id: int, dtype: torch.dtype, device: torch.device, _rotary: nn.Module
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=dtype, device=device))

    def forward(
        self, hidden: torch.Tensor, _residual: torch.Tensor | None, _positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden * self.weight, hidden

    def load_weights(self, weights: ScopedWeightLoader, _context: ParallelLoadContext) -> None:
        weights.load_tensor(self.weight, "weight")


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> qwen3_5_mtp.Qwen3_5MtpForCausalLM:
    monkeypatch.setattr(qwen3_5_mtp, "get_qwen3_5_decoder_layer_class", lambda _device: _Decoder)
    monkeypatch.setattr(qwen3_5, "get_qwen3_5_decoder_layer_class", lambda _device: _Decoder)
    return qwen3_5_mtp.Qwen3_5MtpForCausalLM(
        {
            "hidden_size": 4,
            # Native model loading supplies the draft's actual layer geometry.
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "vocab_size": 8,
            "max_position_embeddings": 16,
            "partial_rotary_factor": 1,
            "dtype": "float32",
            "device": "cpu",
        }
    )


def _checkpoint() -> dict[str, torch.Tensor]:
    return {
        "pre_fc_norm_embedding.weight": torch.tensor([0.2, -0.1, 0.3, 0.1]),
        "pre_fc_norm_hidden.weight": torch.tensor([-0.2, 0.4, 0.1, 0.2]),
        "fc.weight": torch.arange(32, dtype=torch.float32).view(4, 8) / 16,
        "layers.0.weight": torch.tensor([0.5, 0.7, 0.2, 0.4]),
        "norm.weight": torch.tensor([0.1, -0.2, 0.3, 0.4]),
    }


def _weights(checkpoint: dict[str, torch.Tensor], prefix: str = "mtp.") -> dict[str, torch.Tensor]:
    return {
        **{prefix + key: value for key, value in checkpoint.items()},
        "lm_head.weight": torch.ones(8, 4),
        "model.language_model.embed_tokens.weight": torch.ones(8, 4),
    }


@pytest.mark.parametrize("prefix", ["mtp.", "model.mtp."])
def test_mtp_loads_one_complete_checkpoint_root_and_shared_checkpoint_endpoints(
    model: qwen3_5_mtp.Qwen3_5MtpForCausalLM, prefix: str
) -> None:
    checkpoint = _checkpoint()
    weights = _weights(checkpoint, prefix)
    model.load_weights([StateDict(weights)], 0, 1)
    assert model.cfg.n_layers == 1 and model.cfg.layer_types == ["full_attention"]
    torch.testing.assert_close(model.model.embed_tokens.weight, weights["model.language_model.embed_tokens.weight"])
    torch.testing.assert_close(model.lm_head.weight, weights["lm_head.weight"])
    torch.testing.assert_close(model.model.fc.weight, checkpoint["fc.weight"])
    torch.testing.assert_close(model.model.norm.weight, checkpoint["norm.weight"])


@pytest.mark.parametrize("missing", ["pre_fc_norm_hidden.weight", "fc.weight", "layers.0.weight", "norm.weight"])
def test_mtp_rejects_missing_weights_instead_of_mixing_checkpoint_roots(
    model: qwen3_5_mtp.Qwen3_5MtpForCausalLM, missing: str
) -> None:
    checkpoint = _checkpoint()
    weights = _weights({key: value for key, value in checkpoint.items() if key != missing})
    weights["model." + missing] = checkpoint[missing]
    with pytest.raises(KeyError):
        model.load_weights([StateDict(weights)], 0, 1)


def test_mtp_matches_native_fusion_and_returns_post_norm_recurrent_hidden(
    model: qwen3_5_mtp.Qwen3_5MtpForCausalLM,
) -> None:
    checkpoint = _checkpoint()
    model.load_weights([StateDict(_weights(checkpoint))], 0, 1)
    embedding = nn.Embedding(8, 4)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(32).view(8, 4) / 7 + 0.1)
    model.model.embed_tokens = embedding
    model.lm_head = nn.Linear(4, 8, bias=False)
    ids = torch.tensor([1, 2, 3])
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    recurrent = torch.arange(12, dtype=torch.float32).view(3, 4) / 11 - 0.2

    def reference_norm(value: torch.Tensor, name: str) -> torch.Tensor:
        return value * (value.square().mean(-1, keepdim=True) + model.cfg.rms_norm_eps).rsqrt() * (1 + checkpoint[name])

    fused = torch.cat(
        (
            reference_norm(embedding(ids), "pre_fc_norm_embedding.weight"),
            reference_norm(recurrent, "pre_fc_norm_hidden.weight"),
        ),
        dim=-1,
    )
    projected = torch.nn.functional.linear(fused, checkpoint["fc.weight"])
    expected = reference_norm(projected * (1 + checkpoint["layers.0.weight"]), "norm.weight")
    actual = model.model(ids, positions, recurrent)
    torch.testing.assert_close(actual, expected)
    assert actual[0].abs().sum() > 0  # Qwen does not apply GLM's position-zero masking.
    torch.testing.assert_close(model.compute_logits(actual, torch.tensor([2, 0])), model.lm_head(expected[[2, 0]]))
    with pytest.raises(ValueError, match="one carried hidden state"):
        model.model(ids, positions, recurrent[:1])


@pytest.mark.parametrize("prefix", ["", "model."])
def test_mtp_rejects_unscoped_draft_checkpoint_roots(model: qwen3_5_mtp.Qwen3_5MtpForCausalLM, prefix: str) -> None:
    with pytest.raises(KeyError):
        model.load_weights([StateDict(_weights(_checkpoint(), prefix))], 0, 1)


def test_mtp_owns_usable_embedding_and_vocabulary_endpoints(model: qwen3_5_mtp.Qwen3_5MtpForCausalLM) -> None:
    model.load_weights([StateDict(_weights(_checkpoint()))], 0, 1)
    hidden = model.model(torch.tensor([1]), torch.tensor([0]), torch.ones(1, 4))
    assert hidden.shape == (1, 4)
    assert model.compute_logits(hidden, None).shape == (1, 8)
