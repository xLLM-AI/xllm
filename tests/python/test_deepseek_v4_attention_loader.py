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

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_loader.module_loaders import load_w8a8_dynamic_projection
from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM


@pytest.mark.parametrize("module_name", ["attn", "self_attn"])
def test_attention_loader_accepts_supported_prefixes_and_sink_alias(module_name: str) -> None:
    layer_prefix = "layers.0."
    prefix = f"layers.0.{module_name}."
    tensors = {
        prefix + "wq_a.weight": torch.arange(12).reshape(4, 3),
        prefix + "wq_a.weight_scale": torch.arange(4).reshape(4, 1),
        prefix + "wq_a.weight_offset": torch.zeros(4, 1),
        prefix + "wq_b.weight": torch.arange(12, 24).reshape(4, 3),
        prefix + "wq_b.weight_scale": torch.arange(4, 8).reshape(4, 1),
        prefix + "wq_b.weight_offset": torch.zeros(4, 1),
        prefix + "wkv.weight": torch.arange(6).reshape(2, 3),
        prefix + "wo_a.weight": torch.arange(12, 24).reshape(4, 3),
        prefix + "wo_b.weight": torch.arange(12).reshape(3, 4),
        prefix + "q_norm.weight": torch.arange(3),
        prefix + "kv_norm.weight": torch.arange(3, 6),
        prefix + "attn_sink.weight": torch.arange(4),
        layer_prefix + "attn_norm.weight": torch.arange(3),
        layer_prefix + "ffn_norm.weight": torch.arange(3),
        layer_prefix + "hc_attn_fn": torch.arange(3),
        layer_prefix + "hc_attn_scale": torch.arange(3),
        layer_prefix + "hc_attn_base": torch.arange(3),
        layer_prefix + "hc_ffn_fn": torch.arange(3),
        layer_prefix + "hc_ffn_scale": torch.arange(3),
        layer_prefix + "hc_ffn_base": torch.arange(3),
    }

    class FakeLoader:
        def __init__(self) -> None:
            self.loaded: dict[str, torch.Tensor] = {}

        def has(self, name: str) -> bool:
            return name in tensors

        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

        def shard(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            return tensor.chunk(2, dim=dim)[1].contiguous()

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    attention = SimpleNamespace(attn_sink_loaded=False)
    owner = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(owner)
    owner.cfg = SimpleNamespace(n_heads=4, tp_size=2, tp_rank=1)
    owner.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    loader = FakeLoader()

    loaded_attention, resolved_prefix = DeepseekV4ForCausalLM._load_dsv4_attention(
        owner,
        loader,
        checkpoint_prefix="layers.0.",
        parameter_prefix="model.layers.0.",
        layer_id=0,
        w8a8_loader=partial(load_w8a8_dynamic_projection, loader),
    )

    assert loaded_attention is attention
    assert resolved_prefix == prefix
    assert attention.attn_sink_loaded
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.q_b_proj.weight"],
        tensors[prefix + "wq_b.weight"][2:],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.o_a_proj.weight"],
        tensors[prefix + "wo_a.weight"][2:],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.o_b_proj.weight"],
        tensors[prefix + "wo_b.weight"][:, 2:],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.attn_sink"],
        tensors[prefix + "attn_sink.weight"][2:],
    )


def test_attention_loader_rejects_missing_attention_weights() -> None:
    owner = SimpleNamespace()
    loader = SimpleNamespace(has=lambda _name: False)

    with pytest.raises(KeyError, match="DeepSeek-V4 layer 0 attention weights not found"):
        DeepseekV4ForCausalLM._load_dsv4_attention(
            owner,
            loader,
            checkpoint_prefix="layers.0.",
            parameter_prefix="model.layers.0.",
            layer_id=0,
            w8a8_loader=partial(load_w8a8_dynamic_projection, loader),
        )
