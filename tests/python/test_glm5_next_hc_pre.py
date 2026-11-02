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

import pytest
import torch

from xllm.python.models import glm5_next


@pytest.fixture
def hc_pre(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, ...]]:
    calls = []

    def run(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        hc_mult: int,
        iterations: int,
        norm_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append((weight, scale, base))
        assert weight.dtype == scale.dtype == base.dtype == torch.float32
        assert hc_mult == 4 and iterations == 20
        assert norm_eps == 1e-5 and hc_eps == 1e-6
        return hidden[:, :, 0], scale, base

    monkeypatch.setattr(glm5_next.kernels, "hc_pre", run)
    return calls


def _layer(dtype: torch.dtype = torch.bfloat16) -> glm5_next.Glm5NextHyperConnection:
    config = glm5_next.Glm5NextConfig(hidden_size=16)
    layer = glm5_next.Glm5NextHyperConnection(config, torch.bfloat16, torch.device("cpu")).to(dtype)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.123456)
    return layer


def _forward(layer: glm5_next.Glm5NextHyperConnection) -> None:
    layer(torch.zeros(1, 4, 4, 16, dtype=layer.fn.dtype))


@torch.inference_mode()
def test_hc_pre_caches_rounded_nonpersistent_weight(hc_pre: list[tuple[torch.Tensor, ...]]) -> None:
    layer = _layer()
    layer.process_weights_after_loading()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        _forward(layer)
        _forward(layer)
    assert not {event.key for event in trace.key_averages()} & {"aten::to", "aten::_to_copy", "aten::copy_"}
    weight = hc_pre[-1][0]
    assert weight is hc_pre[-2][0]
    assert weight.data_ptr() != layer.fn.data_ptr()
    assert set(layer.state_dict()) == {"fn", "scale", "base"}
    torch.testing.assert_close(weight, layer.fn.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("assign", [False, True])
@torch.inference_mode()
def test_hc_pre_reload_keeps_graph_addresses(
    hc_pre: list[tuple[torch.Tensor, ...]], assign: bool, dtype: torch.dtype
) -> None:
    layer = _layer(dtype)
    layer.process_weights_after_loading()
    _forward(layer)
    before = hc_pre[-1]
    state = {name: torch.full_like(value, 0.654321) for name, value in layer.state_dict().items()}
    layer.load_state_dict(state, assign=assign)
    _forward(layer)
    for name, previous, actual in zip(("fn", "scale", "base"), before, hc_pre[-1], strict=True):
        assert previous is actual
        torch.testing.assert_close(actual, getattr(layer, name).float(), rtol=0, atol=0)


@torch.inference_mode()
def test_hc_pre_dtype_move_refreshes_weight(hc_pre: list[tuple[torch.Tensor, ...]]) -> None:
    layer = _layer()
    layer.process_weights_after_loading()
    layer.to(torch.float32)
    layer.fn.fill_(0.7654321)
    layer.to(torch.bfloat16)
    _forward(layer)
    torch.testing.assert_close(hc_pre[-1][0], layer.fn.float(), rtol=0, atol=0)
