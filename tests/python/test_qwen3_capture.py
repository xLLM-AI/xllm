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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import xllm.python.models.qwen3 as qwen3_module
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.qwen3 import Qwen3Config, Qwen3Model


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        values = input_ids.to(torch.float32)
        return torch.stack((values, values + 10.0), dim=-1)


class _ResidualLayer(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions, cos_sin_cache, cos, sin, mrope_section
        residual = hidden if residual is None else hidden + residual
        return torch.full_like(hidden, self.delta), residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (hidden if residual is None else hidden + residual), residual


def _config(*, layers_to_capture: tuple[int, ...]) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=2,
        n_layers=3,
        n_heads=1,
        n_kv_heads=1,
        head_dim=2,
        intermediate_size=4,
        max_position_embeddings=8,
        vocab_size=4,
        layers_to_capture=layers_to_capture,
    )


def _model(monkeypatch: pytest.MonkeyPatch, layers_to_capture: tuple[int, ...]) -> Qwen3Model:
    monkeypatch.setattr(
        qwen3_module,
        "get_forward_context",
        lambda: SimpleNamespace(cp_context=None),
    )
    model = Qwen3Model(_config(layers_to_capture=layers_to_capture), torch.float32, torch.device("cpu"))
    model.embed_tokens = _Embedding()
    model.layers = nn.ModuleList([_ResidualLayer(1.0), _ResidualLayer(2.0), _ResidualLayer(3.0)])
    model.norm = _FinalNorm()
    return model


def test_qwen3_config_reads_capture_layers() -> None:
    config = Qwen3Config.from_dict({"layers_to_capture": [3, 1]})

    assert config.layers_to_capture == (3, 1)


def test_qwen3_model_returns_captured_residual_streams_in_config_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch, layers_to_capture=(2, 1))
    embedded = model.embed_tokens(torch.tensor([1, 2]))

    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, tuple)
    hidden, aux_hidden = output
    torch.testing.assert_close(hidden, embedded + 6.0)
    torch.testing.assert_close(aux_hidden, torch.cat((embedded + 3.0, embedded + 1.0), dim=-1))


def test_qwen3_model_returns_tensor_when_capture_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(monkeypatch, layers_to_capture=())

    output = model(torch.tensor([1, 2]), torch.tensor([0, 1]))

    assert isinstance(output, torch.Tensor)


def test_aux_hidden_capture_snapshots_hidden_without_residual() -> None:
    capture = AuxHiddenCapture((0,))
    hidden = torch.tensor([[1.0, 2.0]])
    captured: dict[int, torch.Tensor] = {}

    capture.capture_layer(0, hidden, None, captured)
    hidden.add_(10.0)
    _, aux_hidden = capture.finalize(hidden, captured)

    torch.testing.assert_close(aux_hidden, torch.tensor([[1.0, 2.0]]))
