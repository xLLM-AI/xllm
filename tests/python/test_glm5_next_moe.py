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

from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_loader import W8A8WeightLoader
from xllm.python.models import glm5_next


@pytest.mark.parametrize("shape", [(1, 8), (4, 8), (1, 4, 8)])
def test_quantized_expert_scales_are_loaded_in_kernel_dtypes(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    config = glm5_next.Glm5NextConfig(
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        tp_size=1,
    )
    module = glm5_next.Glm5NextMoE(config, torch.bfloat16, torch.device("cpu"))
    module.shared_experts = torch.nn.Identity()
    module.use_w8a8 = True
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = module
    model.model.layers = torch.nn.ModuleList([layer])
    original = torch.linspace(0.001, 1.0, 32).view(4, 8, 1)
    gate_up_scale = torch.linspace(0.001, 1.0, 16).view(16, 1)
    tensors: dict[str, torch.Tensor] = {}
    for expert_index in range(config.n_routed_experts):
        expert_tensors = {
            "gate_proj.weight": torch.ones(16, 8, dtype=torch.int8),
            "gate_proj.weight_scale": gate_up_scale,
            "gate_proj.weight_offset": torch.zeros(16, 1),
            "up_proj.weight": torch.ones(16, 8, dtype=torch.int8),
            "up_proj.weight_scale": gate_up_scale,
            "up_proj.weight_offset": torch.zeros(16, 1),
            "down_proj.weight": torch.ones(8, 16, dtype=torch.int8),
            "down_proj.weight_scale": original[expert_index],
            "down_proj.weight_offset": torch.zeros(8, 1),
        }
        tensors.update(
            {f"model.layers.0.mlp.experts.{expert_index}.{name}": value for name, value in expert_tensors.items()}
        )
    loader = W8A8WeightLoader(
        model,
        [SimpleNamespace(has=tensors.__contains__, get_tensor=tensors.__getitem__)],
        tp_size=1,
        tp_rank=0,
    )
    monkeypatch.setattr(glm5_next.kernels, "format_cast_nz", lambda weight: weight, raising=False)

    glm5_next.Glm5NextForCausalLM._load_experts_w8a8(model, loader, "model.layers.0.mlp.", config.n_routed_experts)

    assert module.experts_w13_scale.dtype == torch.float32
    assert module.experts_w2_scale.dtype == torch.bfloat16
    torch.testing.assert_close(module.experts_w2_scale, original.bfloat16(), rtol=0, atol=0)
    module.process_weights_after_loading()

    expected_w13_scale = torch.cat((gate_up_scale, gate_up_scale)).view(1, 32).expand(4, 32)
    torch.testing.assert_close(module.experts_w13_scale, expected_w13_scale, rtol=0, atol=0)
    torch.testing.assert_close(module.experts_w2_scale, original.view(4, 8).bfloat16(), rtol=0, atol=0)
    assert module.get_buffer("experts_w2_scale") is module.experts_w2_scale
    assert not hasattr(module, "experts_w2_scale_compute")
    torch.testing.assert_close(module.state_dict()["experts_w2_scale"], original.view(4, 8).bfloat16(), rtol=0, atol=0)
    calls: list[object] = []

    def grouped_moe(hidden: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        assert args[3] is module.experts_w13_scale
        calls.append(args[4])
        return torch.zeros_like(hidden)

    monkeypatch.setattr(glm5_next.kernels, "grouped_moe", grouped_moe, raising=False)
    hidden = torch.ones(shape, dtype=torch.bfloat16)
    for _ in range(2):
        torch.testing.assert_close(module(hidden), hidden, rtol=0, atol=0)
    assert len(calls) == 2
    assert all(scale is module.experts_w2_scale for scale in calls)


def test_nonquantized_experts_do_not_allocate_quantization_scales() -> None:
    config = glm5_next.Glm5NextConfig(
        hidden_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        tp_size=1,
    )
    module = glm5_next.Glm5NextMoE(config, torch.bfloat16, torch.device("cpu"))
    module.shared_experts = torch.nn.Identity()

    module.process_weights_after_loading()

    assert not hasattr(module, "experts_w13_scale")
    assert not hasattr(module, "experts_w2_scale")
    assert not hasattr(module, "experts_w2_scale_compute")
