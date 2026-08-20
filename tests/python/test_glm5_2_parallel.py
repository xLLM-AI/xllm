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

"""Parallel-layout tests for the GLM-5.2 Python NPU model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.models import glm5_2
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM


def _config(**overrides) -> dict:
    values = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "vocab_size": 32,
        "max_position_embeddings": 16,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "first_k_dense_replace": 0,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 8,
        "tp_size": 2,
        "tp_rank": 0,
        "dp_size": 2,
        "dp_rank": 0,
        "cp_size": 1,
        "cp_rank": 0,
        "world_size": 4,
        "moe_tp_size": 1,
        "moe_tp_rank": 0,
        "ep_size": 4,
        "ep_rank": 0,
        "dtype": "float32",
        "device": "cpu",
    }
    values.update(overrides)
    return values


def test_full_world_ep_partitions_glm_experts() -> None:
    cfg = Glm52Config.from_dict(_config(ep_rank=3))
    cfg.validate()

    model = Glm52ForCausalLM(_config(ep_rank=3))
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 6
    assert moe.local_expert_end == 8
    assert moe.experts_w13.shape == (2, 16, 16)
    assert moe.experts_w2.shape == (2, 16, 8)


def test_glm_parallel_world_size_defaults_to_tp_dp_product() -> None:
    values = _config()
    values.pop("world_size")

    cfg = Glm52Config.from_dict(values)

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 4


def test_glm_parallel_world_size_includes_context_parallel() -> None:
    cfg = Glm52Config.from_dict(_config(cp_size=2, cp_rank=1, world_size=8, ep_size=8))

    cfg.validate()

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 8
    assert cfg.cp_rank == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ep_size": 2}, "ep_size must be 1 or world_size"),
        ({"world_size": 8}, r"world_size must equal tp_size \* dp_size \* cp_size"),
        ({"cp_rank": 2}, "cp_rank must be in"),
        ({"n_routed_experts": 10}, "n_routed_experts must be divisible by ep_size"),
        ({"moe_tp_size": 2}, r"moe_tp_size \* ep_size"),
        ({"ep_rank": 4}, "ep_rank must be in"),
    ],
)
def test_invalid_glm_parallel_topology_is_rejected(overrides: dict, message: str) -> None:
    cfg = Glm52Config.from_dict(_config(**overrides))

    with pytest.raises(ValueError, match=message):
        cfg.validate()


class _RecordingLoader:
    latest: _RecordingLoader | None = None

    def __init__(self, _model, _state_dicts, tp_size: int, tp_rank: int) -> None:
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.loaded: list[str] = []
        self.shared_shards: list[tuple[str, int, int]] = []
        type(self).latest = self

    def load_tensor(self, name: str) -> torch.Tensor:
        self.loaded.append(name)
        if ".mlp.experts." not in name:
            return torch.zeros(32, 32)
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            return torch.zeros(8, 16, dtype=torch.int8)
        if name.endswith(("gate_proj.weight_scale", "up_proj.weight_scale")):
            return torch.zeros(8, 1)
        if name.endswith(("gate_proj.weight_offset", "up_proj.weight_offset")):
            return torch.zeros(8, 1)
        if name.endswith("down_proj.weight"):
            return torch.zeros(16, 8, dtype=torch.int8)
        if name.endswith(("down_proj.weight_scale", "down_proj.weight_offset")):
            return torch.zeros(16, 1)
        raise AssertionError(f"unexpected expert tensor: {name}")

    def shard(
        self,
        tensor: torch.Tensor,
        dim: int,
        world: int | None = None,
        rank: int | None = None,
    ) -> torch.Tensor:
        world = self.tp_size if world is None else world
        rank = self.tp_rank if rank is None else rank
        if world <= 1:
            return tensor
        size = tensor.size(dim) // world
        return tensor.narrow(dim, rank * size, size).contiguous()

    def copy_in(self, name: str, tensor: torch.Tensor) -> None:
        self.loaded.append(name)
        assert tensor.is_contiguous()

    def load_w8a8_a(self, prefix: str, proj: str, _shard_dims: dict | None = None) -> None:
        self.loaded.append(prefix + proj)

    def load_w8a8_b(self, prefix: str) -> None:
        self.loaded.append(prefix)
        if ".shared_experts." in prefix:
            self.shared_shards.append((prefix, self.tp_size, self.tp_rank))


def test_glm_weight_loader_reads_only_local_ep_experts(monkeypatch) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(".experts.4." in name or ".experts.5." in name for name in expert_names)
    assert loader.tp_size == 2
    assert loader.tp_rank == 0
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 1, 0)]
