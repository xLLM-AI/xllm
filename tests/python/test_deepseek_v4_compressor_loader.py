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
from xllm.python.models import deepseek_v4
from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM


def test_compressor_loader_accepts_split_native_projection_aliases() -> None:
    prefix = "layers.0.self_attn.compress."
    tensors = {
        prefix + "k_proj.weight": torch.full((2, 3), 1.0),
        prefix + "v_proj.weight": torch.full((4, 3), 2.0),
        prefix + "score_proj.weight": torch.full((6, 3), 3.0),
        prefix + "position_bias": torch.full((4, 6), 4.0),
        prefix + "rms_norm.weight": torch.full((3,), 5.0),
    }

    class Loader:
        def __init__(self) -> None:
            self.loaded = {}

        def has(self, name: str) -> bool:
            return name in tensors

        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    loader = Loader()
    DeepseekV4ForCausalLM._load_dsv4_compressor(
        loader,
        (prefix,),
        "model.layers.0.self_attn.cmp_",
        "compressor weights not found",
    )

    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_wkv.weight"],
        torch.cat([tensors[prefix + "k_proj.weight"], tensors[prefix + "v_proj.weight"]]),
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_wgate.weight"],
        tensors[prefix + "score_proj.weight"],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_ape"],
        tensors[prefix + "position_bias"],
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_norm.weight"],
        tensors[prefix + "rms_norm.weight"],
    )


@pytest.mark.parametrize("indexer_module_name", ["compressor", "compress"])
def test_load_weights_routes_compressor_aliases_through_loader(
    monkeypatch: pytest.MonkeyPatch,
    indexer_module_name: str,
) -> None:
    attention_prefix = "layers.0.attn.compress."
    indexer_prefix = f"layers.0.attn.indexer.{indexer_module_name}."
    tensors = {
        attention_prefix + "k_proj.weight": torch.full((2, 3), 1.0),
        attention_prefix + "v_proj.weight": torch.full((4, 3), 2.0),
        attention_prefix + "score_proj.weight": torch.full((6, 3), 3.0),
        attention_prefix + "position_bias": torch.full((4, 6), 4.0),
        attention_prefix + "rms_norm.weight": torch.full((3,), 5.0),
        indexer_prefix + "kv_proj.weight": torch.full((6, 3), 6.0),
        indexer_prefix + "score_proj.weight": torch.full((6, 3), 7.0),
        indexer_prefix + "position_bias": torch.full((4, 6), 8.0),
        indexer_prefix + "rms_norm.weight": torch.full((3,), 9.0),
        "layers.0.attn.indexer.wq_b.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }

    class Loader(W8A8WeightLoader):
        def __init__(self, *_args: object) -> None:
            self.tp_size = 1
            self.tp_rank = 0
            self.loaded: dict[str, torch.Tensor] = {}

        def has(self, name: str) -> bool:
            return name in tensors

        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors.get(name, torch.ones(1))

        def shard(self, tensor: torch.Tensor, dim: int, **_kwargs: object) -> torch.Tensor:
            return tensor

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    loader = Loader()
    monkeypatch.setattr(
        deepseek_v4,
        "W8A8WeightLoader",
        lambda *_args, **_kwargs: loader,
    )
    attention = SimpleNamespace(
        indexer=object(),
        cmp_wkv=object(),
        process_weights_after_loading=lambda: None,
    )
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(tp_size=1, tp_rank=0, n_layers=1, n_heads=1)
    model.model = SimpleNamespace(layers=[layer])

    model.load_weights({}, tp_rank=0, tp_size=1)

    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.cmp_wkv.weight"],
        torch.cat(
            [
                tensors[attention_prefix + "k_proj.weight"],
                tensors[attention_prefix + "v_proj.weight"],
            ]
        ),
    )
    torch.testing.assert_close(
        loader.loaded["model.layers.0.self_attn.indexer.compressor_wkv.weight"],
        tensors[indexer_prefix + "kv_proj.weight"],
    )
