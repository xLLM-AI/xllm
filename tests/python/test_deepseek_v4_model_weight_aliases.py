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

from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM


@pytest.mark.parametrize("source_prefix", ["", "model."])
@pytest.mark.parametrize("canonical_endpoints", [False, True])
def test_model_loader_accepts_native_endpoint_aliases(source_prefix: str, canonical_endpoints: bool) -> None:
    class StateDict:
        tensors = {
            f"{source_prefix}embed_tokens.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            f"{source_prefix}final_layernorm.weight": torch.arange(2, dtype=torch.float32),
            "model.hc_head_fn": torch.full((2,), 1.0),
            "model.hc_head_base": torch.full((2,), 2.0),
            "model.hc_head_scale": torch.full((2,), 3.0),
            "model.head.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "head.weight": torch.full((4, 2), -1.0),
        }

        def has(self, name: str) -> bool:
            return name in self.tensors

        def get_tensor(self, name: str) -> torch.Tensor:
            return self.tensors[name]

    owner = torch.nn.Module()
    owner.cfg = SimpleNamespace(tp_size=2, tp_rank=1, n_layers=0)
    owner.model = torch.nn.Module()
    owner.model.embed_tokens = torch.nn.Embedding(4, 2)
    owner.model.norm = torch.nn.LayerNorm(2, elementwise_affine=True, bias=False)
    owner.model.register_parameter("hc_head_fn", torch.nn.Parameter(torch.empty(2)))
    owner.model.register_parameter("hc_head_base", torch.nn.Parameter(torch.empty(2)))
    owner.model.register_parameter("hc_head_scale", torch.nn.Parameter(torch.empty(2)))
    owner.lm_head = torch.nn.Linear(2, 2, bias=False)

    state = StateDict()
    state.tensors = dict(state.tensors)
    embed_key = f"{source_prefix}embed_tokens.weight"
    norm_key = f"{source_prefix}final_layernorm.weight"
    if canonical_endpoints:
        embed_key = f"{source_prefix}embed.weight"
        norm_key = f"{source_prefix}norm.weight"
        state.tensors[embed_key] = torch.full((4, 4), 9.0)
        state.tensors[norm_key] = torch.full((2,), 8.0)
    DeepseekV4ForCausalLM.load_weights(owner, [state], tp_rank=1, tp_size=2)

    torch.testing.assert_close(
        owner.model.embed_tokens.weight,
        state.tensors[embed_key][:, 2:],
    )
    torch.testing.assert_close(owner.model.norm.weight, state.tensors[norm_key])
    torch.testing.assert_close(owner.model.hc_head_fn, StateDict.tensors["model.hc_head_fn"])
    torch.testing.assert_close(owner.model.hc_head_base, StateDict.tensors["model.hc_head_base"])
    torch.testing.assert_close(owner.model.hc_head_scale, StateDict.tensors["model.hc_head_scale"])
    torch.testing.assert_close(owner.lm_head.weight, StateDict.tensors["model.head.weight"][2:])
