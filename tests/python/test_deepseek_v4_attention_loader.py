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
from unittest.mock import create_autospec

import pytest
import torch

from xllm.python.model_loader import ModelWeightLoader
from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM


def _compressor_tensors(prefix: str) -> dict[str, torch.Tensor]:
    return {
        prefix + "compressor.wkv.weight": torch.arange(8).reshape(4, 2),
        prefix + "compressor.wgate.weight": torch.arange(4, 8).reshape(2, 2),
        prefix + "compressor.ape": torch.arange(4).reshape(2, 2),
        prefix + "compressor.norm.weight": torch.arange(2, 4).reshape(2, 1),
    }


@pytest.mark.parametrize("module_name", ["attn", "self_attn"])
@pytest.mark.parametrize("with_indexer", [False, True])
def test_attention_loader_accepts_supported_prefixes_and_sink_alias(module_name: str, with_indexer: bool) -> None:
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
        prefix + "wkv.weight_scale": torch.ones(2, 1),
        prefix + "wkv.weight_offset": torch.zeros(2, 1),
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
    tensors.update(_compressor_tensors(prefix))
    if with_indexer:
        tensors.update(
            {
                prefix + "indexer.wq_b.weight": torch.arange(6).reshape(2, 3),
                prefix + "indexer.wq_b.weight_scale": torch.ones(2, 1),
                prefix + "indexer.wq_b.weight_offset": torch.zeros(2, 1),
                prefix + "indexer.weights_proj.weight": torch.arange(6).reshape(3, 2),
            }
        )
        tensors.update(_compressor_tensors(prefix + "indexer."))

    class FakeLoader(ModelWeightLoader):
        def __init__(self) -> None:
            self.loaded: dict[str, torch.Tensor] = {}

        def has(self, name: str) -> bool:
            return name in tensors

        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

        def shard(self, tensor: torch.Tensor, dim: int, *, world=None, rank=None) -> torch.Tensor:
            return tensor.chunk(2, dim=dim)[1].contiguous()

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    attention = SimpleNamespace(
        attn_sink_loaded=False, indexer=SimpleNamespace() if with_indexer else None, cmp_wkv=None
    )
    owner = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(owner)
    owner.cfg = SimpleNamespace(n_heads=4, tp_size=2, tp_rank=1)
    owner.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])
    loader = FakeLoader()

    loaded_attention = DeepseekV4ForCausalLM._load_dsv4_attention(
        owner,
        loader,
        checkpoint_prefix="layers.0.",
        parameter_prefix="model.layers.0.",
        layer_id=0,
    )

    assert loaded_attention is attention
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
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_wkv.weight"],
        tensors[prefix + "compressor.wkv.weight"],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_wgate.weight"],
        tensors[prefix + "compressor.wgate.weight"],
    )
    if with_indexer:
        torch.testing.assert_close(
            loader.loaded["model.layers.0.self_attn.indexer.wq_b.weight"],
            tensors[prefix + "indexer.wq_b.weight"],
        )
        torch.testing.assert_close(
            loader.loaded["model.layers.0.self_attn.indexer.compressor_wkv.weight"],
            tensors[prefix + "indexer.compressor.wkv.weight"],
        )
    else:
        assert "model.layers.0.self_attn.indexer.compressor_wkv.weight" not in loader.loaded


def test_attention_loader_rejects_missing_attention_weights() -> None:
    owner = SimpleNamespace()
    loader = create_autospec(ModelWeightLoader, instance=True)
    loader.has.return_value = False

    with pytest.raises(KeyError, match="DeepSeek-V4 layer 0 attention weights not found"):
        DeepseekV4ForCausalLM._load_dsv4_attention(
            owner,
            loader,
            checkpoint_prefix="layers.0.",
            parameter_prefix="model.layers.0.",
            layer_id=0,
        )
