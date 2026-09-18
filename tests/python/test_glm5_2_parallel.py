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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from xllm.python.models import glm5_2
from xllm.python.models.glm5_2 import Glm52Config, Glm52ForCausalLM
from xllm.python.models.weight_utils import W8A8WeightLoader


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
    assert moe.num_local_experts == 2

    moe.allocate_experts_w13_for_loading()
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w13.shape == (2, 16, 16)
    assert moe.experts_w2.shape == (2, 16, 8)


def test_proper_divisor_ep_partitions_experts_and_moe_intermediate() -> None:
    values = _config(ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    cfg = Glm52Config.from_dict(values)

    cfg.validate()
    model = Glm52ForCausalLM(values)
    moe = model.model.layers[0].mlp

    assert moe.local_expert_start == 4
    assert moe.local_expert_end == 8
    assert moe.num_local_experts == 4
    assert moe.inter_local == 4

    moe.allocate_experts_w13_for_loading()
    moe.allocate_experts_w2_for_loading()
    assert moe.experts_w13.shape == (4, 8, 16)
    assert moe.experts_w2.shape == (4, 16, 4)


def test_glm_ep8_moe_tp2_topology_is_valid() -> None:
    cfg = Glm52Config.from_dict(
        _config(
            num_attention_heads=64,
            n_routed_experts=256,
            moe_intermediate_size=2048,
            tp_size=8,
            dp_size=2,
            world_size=16,
            ep_size=8,
            ep_rank=7,
            moe_tp_size=2,
            moe_tp_rank=1,
        )
    )

    cfg.validate()


def test_glm_parallel_world_size_defaults_to_tp_dp_product() -> None:
    values = _config()
    values.pop("world_size")

    cfg = Glm52Config.from_dict(values)

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 4


@pytest.mark.parametrize(("configured", "expected"), [(None, True), (False, False)])
def test_glm_mlapo_config_defaults_on_and_can_be_disabled(
    configured: bool | None,
    expected: bool,
) -> None:
    values = _config()
    if configured is not None:
        values["enable_mlapo"] = configured

    cfg = Glm52Config.from_dict(values)

    assert cfg.enable_mlapo is expected


@pytest.mark.parametrize(
    ("enable_mlapo", "device_type", "runtime_supported", "expected"),
    [
        (True, "npu", True, True),
        (True, "privateuseone", True, True),
        (False, "npu", True, False),
        (True, "cpu", True, False),
        (True, "npu", False, False),
    ],
)
def test_glm_mlapo_requires_config_device_and_runtime_support(
    monkeypatch,
    enable_mlapo: bool,
    device_type: str,
    runtime_supported: bool,
    expected: bool,
) -> None:
    cfg = Glm52Config.from_dict(_config(enable_mlapo=enable_mlapo))
    supports = MagicMock(return_value=runtime_supported)
    monkeypatch.setattr(glm5_2.kernels, "supports_mla_preprocess_v2", supports, raising=False)

    actual = glm5_2._can_use_mlapo_v2(cfg, MagicMock(type=device_type))

    assert actual is expected
    if enable_mlapo and device_type in ("npu", "privateuseone"):
        supports.assert_called_once_with(cfg.kv_lora_rank, cfg.qk_rope_head_dim)
    else:
        supports.assert_not_called()


def test_glm_parallel_world_size_includes_context_parallel() -> None:
    cfg = Glm52Config.from_dict(_config(cp_size=2, cp_rank=1, world_size=8, ep_size=8))

    cfg.validate()

    assert cfg.world_size == cfg.tp_size * cfg.dp_size * cfg.cp_size == 8
    assert cfg.cp_rank == 1


def test_glm_layerwise_split_rank_is_validated() -> None:
    cfg = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=1))
    cfg.validate()
    assert cfg.layerwise_split_rank == 1

    invalid = Glm52Config.from_dict(_config(layerwise_split_size=2, layerwise_split_rank=2))
    with pytest.raises(ValueError, match="layerwise_split_rank"):
        invalid.validate()


def test_glm_layerwise_split_cannot_overlap_context_parallel() -> None:
    cfg = Glm52Config.from_dict(
        _config(
            cp_size=2,
            cp_rank=0,
            world_size=8,
            ep_size=8,
            layerwise_split_size=2,
        )
    )
    with pytest.raises(ValueError, match="CP and layerwise"):
        cfg.validate()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ep_size": 3}, "ep_size must divide world_size"),
        ({"ep_size": 2}, r"moe_tp_size \* ep_size"),
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


class _RecordingLoader(W8A8WeightLoader):
    latest: _RecordingLoader | None = None
    dynamic_activation = False

    def __init__(self, model, state_dicts, tp_size: int, tp_rank: int) -> None:
        super().__init__(model, state_dicts, tp_size, tp_rank)
        self.loaded: list[str] = []
        self.fused_projections: list[tuple[str, str, tuple[str, ...]]] = []
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

    def copy_in(self, name: str, tensor: torch.Tensor) -> None:
        self.loaded.append(name)
        assert tensor.is_contiguous()

    def load_compatible_w8a8_projection(
        self,
        prefix: str,
        proj: str,
        _shard_dims: dict | None = None,
        dynamic_activation: bool | None = None,
    ) -> bool:
        self.loaded.append(prefix + proj)
        assert dynamic_activation is self.dynamic_activation
        return self.dynamic_activation

    def w8a8_projection_uses_dynamic_activation(self, prefix: str, proj: str) -> bool:
        return self.dynamic_activation

    def load_fused_w8a8_projection(
        self,
        prefix: str,
        target_proj: str,
        source_projs: tuple[str, ...],
    ) -> None:
        self.fused_projections.append((prefix, target_proj, source_projs))

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        self.loaded.append(prefix)
        if ".shared_experts." in prefix:
            self.shared_shards.append((prefix, world, rank))


class _TensorStateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


@pytest.mark.parametrize("dynamic_activation", [False, True])
def test_compatible_attention_loader_allocates_only_selected_quantization(dynamic_activation: bool) -> None:
    projection = glm5_2._W8A8AttentionLinear(4, 6, torch.device("cpu"))
    model = torch.nn.Module()
    model.proj = projection
    tensors = {"proj.weight": torch.zeros(6, 4, dtype=torch.int8)}
    if dynamic_activation:
        tensors.update(
            {
                "proj.weight_scale": torch.ones(6, 1),
                "proj.weight_offset": torch.zeros(6, 1),
            }
        )
    else:
        tensors.update(
            {
                "proj.deq_scale": torch.ones(6),
                "proj.quant_bias": torch.zeros(6, dtype=torch.int32),
                "proj.input_scale": torch.ones(1, dtype=torch.bfloat16),
                "proj.input_offset": torch.zeros(1, dtype=torch.bfloat16),
            }
        )
    loader = W8A8WeightLoader(model, [_TensorStateDict(tensors)], tp_size=1, tp_rank=0)

    selected_dynamic = loader.w8a8_projection_uses_dynamic_activation("", "proj")
    projection._set_dynamic_activation(selected_dynamic)
    loader.load_compatible_w8a8_projection("", "proj", dynamic_activation=selected_dynamic)

    assert selected_dynamic is dynamic_activation
    assert torch.equal(projection.weight, tensors["proj.weight"])
    if dynamic_activation:
        assert projection.weight_scale.numel() == projection.out_features
        assert projection.weight_offset.numel() == projection.out_features
        assert projection.deq_scale.numel() == 0
        assert projection.quant_bias.numel() == 0
        assert projection.input_scale.numel() == 0
        assert projection.input_offset.numel() == 0
    else:
        assert projection.weight_scale.numel() == 0
        assert projection.weight_offset.numel() == 0
        assert projection.deq_scale.numel() == projection.out_features
        assert projection.quant_bias.numel() == projection.out_features
        assert projection.input_scale.numel() == 1
        assert projection.input_offset.numel() == 1


def test_dynamic_attention_prepares_one_fused_qkv_projection() -> None:
    attention = glm5_2.Glm52MLAAttention.__new__(glm5_2.Glm52MLAAttention)
    torch.nn.Module.__init__(attention)
    attention._use_fused_mla_decode = True
    attention._use_mlapo_v2 = False
    attention._fused_mla_ready = False
    attention._dynamic_mla_ready = False

    q_weight = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int8)
    kv_weight = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int8)
    q_scale = torch.tensor([[0.1], [0.2]])
    kv_scale = torch.tensor([[0.3], [0.4]])

    def dynamic_projection(weight: torch.Tensor, scale: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            _dynamic_activation=True,
            weight=SimpleNamespace(data=weight),
            weight_scale=scale,
            process_weights_after_loading=MagicMock(),
        )

    attention.q_a_proj = dynamic_projection(q_weight, q_scale)
    attention.kv_a_proj_with_mqa = dynamic_projection(kv_weight, kv_scale)
    attention.q_b_proj = dynamic_projection(torch.empty(2, 2, dtype=torch.int8), torch.ones(2, 1))
    attention.o_proj = SimpleNamespace(process_weights_after_loading=MagicMock())
    attention.kv_b_proj = SimpleNamespace(weight=SimpleNamespace(data=torch.ones(3, 1)))
    attention.num_heads_local = 1
    attention.qk_nope_head_dim = 1
    attention.v_head_dim = 2
    attention.kv_lora_rank = 1
    attention.W_UK = torch.empty(1, 1, 1)
    attention.W_UV = torch.empty(1, 1, 2)
    attention.indexer = None

    with patch.object(
        glm5_2.kernels,
        "prepare_quant_weight",
        side_effect=lambda weight: weight.transpose(0, 1).contiguous(),
        create=True,
    ):
        attention.process_weights_after_loading()

    expected_weight = torch.cat((kv_weight, q_weight), dim=0).transpose(0, 1).contiguous()
    expected_scale = torch.cat((kv_scale.flatten(), q_scale.flatten()))
    torch.testing.assert_close(attention._dynamic_qkv_weight, expected_weight)
    torch.testing.assert_close(attention._dynamic_qkv_weight_scale, expected_scale)
    assert attention._dynamic_mla_ready is True
    assert attention._fused_mla_ready is True


def test_glm_weight_loader_reads_only_local_ep_experts(monkeypatch) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(".experts.4." in name or ".experts.5." in name for name in expert_names)
    assert loader.tp_size == 2
    assert loader.tp_rank == 0
    attention_prefix = "model.layers.0.self_attn."
    attention_projections = [name for name in loader.loaded if name.startswith(attention_prefix)]
    assert all(
        name in attention_projections
        for name in [
            attention_prefix + "q_a_proj",
            attention_prefix + "q_b_proj",
            attention_prefix + "kv_a_proj_with_mqa",
        ]
    )
    assert loader.fused_projections == []
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 1, 0)]


def test_glm_weight_loader_shards_proper_divisor_ep_with_moe_tp(monkeypatch) -> None:
    values = _config(ep_size=2, ep_rank=1, moe_tp_size=2, moe_tp_rank=1)
    model = Glm52ForCausalLM(values)
    model.model.layers[0].self_attn.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    loader = _RecordingLoader.latest
    assert loader is not None
    expert_names = [name for name in loader.loaded if ".mlp.experts." in name]
    assert expert_names
    assert all(any(f".experts.{expert}." in name for expert in range(4, 8)) for name in expert_names)
    assert loader.shared_shards == [("model.layers.0.mlp.shared_experts.", 2, 1)]


@pytest.mark.parametrize("dynamic_activation", [False, True])
def test_glm_attention_selects_checkpoint_quantization(
    dynamic_activation: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = Glm52ForCausalLM(_config(ep_rank=2))
    attention = model.model.layers[0].self_attn
    attention.process_weights_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w13_after_loading = MagicMock()
    model.model.layers[0].mlp.process_experts_w2_after_loading = MagicMock()
    model.model.layers[0].mlp.shared_experts.process_weights_after_loading = MagicMock()
    monkeypatch.setattr(_RecordingLoader, "dynamic_activation", dynamic_activation)
    monkeypatch.setattr(glm5_2, "W8A8WeightLoader", _RecordingLoader)

    model.load_weights([], tp_rank=0, tp_size=2)

    assert attention.q_a_proj._dynamic_activation is dynamic_activation
    assert attention.q_b_proj._dynamic_activation is dynamic_activation
    assert attention.kv_a_proj_with_mqa._dynamic_activation is dynamic_activation
    assert attention.o_proj._dynamic_activation is dynamic_activation
    assert attention.indexer is not None
    assert attention.indexer.wq_b._dynamic_activation is dynamic_activation
    projections = (
        attention.q_a_proj,
        attention.q_b_proj,
        attention.kv_a_proj_with_mqa,
        attention.o_proj,
        attention.indexer.wq_b,
    )
    for projection in projections:
        if dynamic_activation:
            assert projection.weight_scale.numel() == projection.out_features
            assert projection.weight_offset.numel() == projection.out_features
            assert projection.deq_scale.numel() == 0
            assert projection.quant_bias.numel() == 0
            assert projection.input_scale.numel() == 0
            assert projection.input_offset.numel() == 0
        else:
            assert projection.weight_scale.numel() == 0
            assert projection.weight_offset.numel() == 0
            assert projection.deq_scale.numel() == projection.out_features
            assert projection.quant_bias.numel() == projection.out_features
            assert projection.input_scale.numel() == 1
            assert projection.input_offset.numel() == 1
