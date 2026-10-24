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

from types import MethodType, SimpleNamespace

import pytest
import torch

from xllm.python.models import deepseek_v4 as deepseek_v4_module
from xllm.python.models.deepseek_v4 import DeepseekV4ForCausalLM

# Quantization buffers that _load_dsv4_moe resolves via owner.get_buffer.
_MOE_QUANT_BUFFER_NAMES = [
    "experts_w13_scale",
    "experts_w13_scale_second",
    "experts_w2_scale",
    "experts_w2_scale_second",
    "experts_w13_offset",
    "experts_w2_offset",
    "experts_w13_scale_bias",
    "experts_w2_scale_bias",
]


def _moe_quant_buffers() -> dict[str, torch.Tensor]:
    return {"model.layers.0.mlp." + name: torch.empty(0) for name in _MOE_QUANT_BUFFER_NAMES}


@pytest.mark.parametrize(
    ("hash_layer", "metadata_suffix", "parameter_name"),
    [
        (True, "tid2eid.weight", "model.layers.0.mlp.tid2eid"),
        (False, "e_score_correction_bias", "model.layers.0.mlp.e_score_correction_bias"),
    ],
)
def test_moe_routing_loader_accepts_native_metadata_aliases(
    hash_layer: bool,
    metadata_suffix: str,
    parameter_name: str,
) -> None:
    gate_prefix = "layers.0.gate."
    tensors = {
        gate_prefix + "weight": torch.arange(6).reshape(2, 3),
        gate_prefix + metadata_suffix: torch.arange(2),
    }

    class FakeLoader:
        def __init__(self) -> None:
            self.loaded: dict[str, torch.Tensor] = {}

        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

        def has(self, name: str) -> bool:
            return name in tensors

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            self.loaded[name] = tensor

    loader = FakeLoader()
    DeepseekV4ForCausalLM._load_dsv4_moe_routing(
        loader,
        gate_prefix,
        "model.layers.0.",
        SimpleNamespace(hash_layer=hash_layer),
        layer_id=0,
    )

    torch.testing.assert_close(
        loader.loaded["model.layers.0.mlp.gate.weight"],
        tensors[gate_prefix + "weight"],
    )
    torch.testing.assert_close(loader.loaded[parameter_name], tensors[gate_prefix + metadata_suffix])


def test_moe_loader_uses_native_projection_names_and_local_expert_range() -> None:
    prefix = "layers.0.mlp."
    expert_prefix = prefix + "experts.3."
    shared_prefix = prefix + "shared_experts."
    tensors = {
        prefix + "gate.weight": torch.arange(6).reshape(2, 3),
        prefix + "gate.tid2eid": torch.arange(2),
        expert_prefix + "gate_proj.weight": torch.arange(12, dtype=torch.int8).reshape(4, 3),
        expert_prefix + "up_proj.weight": torch.arange(12, 24, dtype=torch.int8).reshape(4, 3),
        expert_prefix + "down_proj.weight": torch.arange(12, dtype=torch.int8).reshape(3, 4),
        expert_prefix + "gate_proj.weight_scale": torch.ones(4, 1),
        expert_prefix + "up_proj.weight_scale": torch.ones(4, 1),
        expert_prefix + "down_proj.weight_scale": torch.ones(3, 1),
        shared_prefix + "gate_proj.weight": torch.arange(12, dtype=torch.int8).reshape(4, 3),
        shared_prefix + "up_proj.weight": torch.arange(12, 24, dtype=torch.int8).reshape(4, 3),
        shared_prefix + "down_proj.weight": torch.arange(12, dtype=torch.int8).reshape(3, 4),
    }

    copied: dict[str, torch.Tensor] = {}
    loader = SimpleNamespace(
        has=tensors.__contains__,
        get_tensor=tensors.__getitem__,
        shard=lambda tensor, dim, world=None, rank=None: tensor.chunk(world or 2, dim=dim)[
            1 if rank is None else rank
        ].contiguous(),
        copy_in=copied.__setitem__,
    )
    mlp = SimpleNamespace(
        hash_layer=True,
        moe_tp_size=2,
        moe_tp_rank=1,
        start_expert_id=3,
        num_experts_per_rank=1,
    )
    parameters = {
        "model.layers.0.mlp.experts_w13": torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False),
        "model.layers.0.mlp.experts_w2": torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False),
    }
    buffers = _moe_quant_buffers()
    owner = SimpleNamespace(
        cfg=SimpleNamespace(moe_intermediate_size=4, hidden_size=3),
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
        get_parameter=lambda name: parameters[name],
        get_buffer=lambda name: buffers[name],
        _load_dsv4_moe_routing=DeepseekV4ForCausalLM._load_dsv4_moe_routing,
    )

    DeepseekV4ForCausalLM._load_dsv4_moe(
        owner,
        loader,
        "layers.0.",
        "model.layers.0.",
        layer_id=0,
        source_prefix=prefix,
    )

    torch.testing.assert_close(
        parameters["model.layers.0.mlp.experts_w13"][0],
        torch.cat([tensors[expert_prefix + "gate_proj.weight"][2:], tensors[expert_prefix + "up_proj.weight"][2:]]),
    )
    torch.testing.assert_close(
        copied["model.layers.0.mlp.shared_experts.gate_up_proj.weight"],
        torch.cat(
            [
                tensors[shared_prefix + "gate_proj.weight"][2:],
                tensors[shared_prefix + "up_proj.weight"][2:],
            ]
        ),
    )
    torch.testing.assert_close(
        copied["model.layers.0.mlp.shared_experts.down_proj.weight"],
        tensors[shared_prefix + "down_proj.weight"][:, 2:],
    )


@pytest.mark.parametrize("gate_location", ["nested", "layer"])
@pytest.mark.parametrize("module_name", ["ffn", "mlp"])
@pytest.mark.parametrize(
    ("gate_name", "up_name", "down_name"),
    [
        ("gate_proj", "up_proj", "down_proj"),
        ("w1", "w3", "w2"),
    ],
)
def test_load_weights_reaches_routed_expert_aliases(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    gate_location: str,
    gate_name: str,
    up_name: str,
    down_name: str,
) -> None:
    prefix = f"layers.0.{module_name}."
    expert_prefix = prefix + "experts.0."
    gate_prefix = prefix + "gate." if gate_location == "nested" else "layers.0.gate."
    tensors = {
        "embed.weight": torch.zeros(1),
        "norm.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
        gate_prefix + "weight": torch.arange(6).reshape(2, 3),
        gate_prefix + "tid2eid": torch.arange(2),
        expert_prefix + gate_name + ".weight": torch.arange(12, dtype=torch.int8).reshape(4, 3),
        expert_prefix + up_name + ".weight": torch.arange(12, 24, dtype=torch.int8).reshape(4, 3),
        expert_prefix + down_name + ".weight": torch.arange(12, dtype=torch.int8).reshape(3, 4),
        expert_prefix + gate_name + ".weight_scale": torch.ones(4, 1),
        expert_prefix + up_name + ".weight_scale": torch.ones(4, 1),
        expert_prefix + down_name + ".weight_scale": torch.ones(3, 1),
    }

    class FakeWeightLoader:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def has(self, name: str) -> bool:
            return name in tensors

        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors.get(name, torch.zeros(1))

        def shard(
            self,
            tensor: torch.Tensor,
            dim: int,
            world: int | None = None,
            rank: int | None = None,
        ) -> torch.Tensor:
            return tensor

        def copy_in(self, name: str, tensor: torch.Tensor) -> None:
            pass

        def copy_shard(self, name: str, dim: int) -> None:
            self.copy_in(name, self.shard(self.get_tensor(name), dim))

    monkeypatch.setattr(deepseek_v4_module, "W8A8WeightLoader", FakeWeightLoader)
    process_calls = 0

    def _record_process() -> None:
        nonlocal process_calls
        process_calls += 1

    mlp = SimpleNamespace(
        hash_layer=True,
        moe_tp_size=1,
        moe_tp_rank=0,
        start_expert_id=0,
        num_experts_per_rank=1,
        experts_w13=torch.empty(0),
        process_weights_after_loading=_record_process,
    )
    attention = SimpleNamespace(
        indexer=None,
        process_weights_after_loading=lambda: None,
    )

    def _load_attention(*_args: object, **_kwargs: object) -> tuple[SimpleNamespace, str]:
        return attention, "layers.0.attn."

    parameters = {
        "model.layers.0.mlp.experts_w13": torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False),
        "model.layers.0.mlp.experts_w2": torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False),
    }
    buffers = _moe_quant_buffers()
    owner = SimpleNamespace(
        cfg=SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            n_layers=1,
            n_heads=1,
            moe_intermediate_size=4,
            hidden_size=3,
        ),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention, mlp=mlp)]),
        get_parameter=parameters.__getitem__,
        get_buffer=buffers.__getitem__,
        _load_dsv4_attention=_load_attention,
        _load_dsv4_moe_routing=DeepseekV4ForCausalLM._load_dsv4_moe_routing,
    )
    owner._load_dsv4_moe = MethodType(DeepseekV4ForCausalLM._load_dsv4_moe, owner)

    DeepseekV4ForCausalLM.load_weights(owner, [], tp_rank=0, tp_size=1)

    torch.testing.assert_close(
        parameters["model.layers.0.mlp.experts_w13"][0],
        torch.cat(
            [
                tensors[expert_prefix + gate_name + ".weight"],
                tensors[expert_prefix + up_name + ".weight"],
            ]
        ),
    )
    assert process_calls == 1


def _minimal_moe_loader_state(tensors: dict[str, torch.Tensor]) -> tuple[SimpleNamespace, SimpleNamespace]:
    copied: dict[str, torch.Tensor] = {}
    loader = SimpleNamespace(
        has=tensors.__contains__,
        get_tensor=tensors.__getitem__,
        shard=lambda tensor, dim, world=None, rank=None: tensor,
        copy_in=copied.__setitem__,
    )
    mlp = SimpleNamespace(
        hash_layer=True,
        moe_tp_size=1,
        moe_tp_rank=0,
        start_expert_id=0,
        num_experts_per_rank=1,
        w4a8_dynamic=False,
    )
    parameters = {
        "model.layers.0.mlp.experts_w13": torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False),
        "model.layers.0.mlp.experts_w2": torch.nn.Parameter(torch.empty(0, dtype=torch.int8), requires_grad=False),
    }
    buffers = _moe_quant_buffers()
    owner = SimpleNamespace(
        cfg=SimpleNamespace(moe_intermediate_size=4, hidden_size=4),
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]),
        get_parameter=parameters.__getitem__,
        get_buffer=buffers.__getitem__,
        _load_dsv4_moe_routing=DeepseekV4ForCausalLM._load_dsv4_moe_routing,
    )
    return owner, loader


@pytest.mark.parametrize(
    ("gate_weight", "error"),
    [
        (torch.ones(4, 4), "must be a two-dimensional INT8 tensor"),
        (torch.ones(3, 4, dtype=torch.int8), "rows must match"),
        (torch.ones(4, 3, dtype=torch.int8), "input width 4"),
    ],
)
def test_moe_loader_rejects_invalid_quantized_weight_layout(
    gate_weight: torch.Tensor,
    error: str,
) -> None:
    prefix = "layers.0.mlp."
    expert_prefix = prefix + "experts.0."
    tensors = {
        prefix + "gate.weight": torch.ones(2, 4),
        prefix + "gate.tid2eid": torch.arange(2),
        expert_prefix + "gate_proj.weight": gate_weight,
        expert_prefix + "up_proj.weight": torch.ones(4, 4, dtype=torch.int8),
        expert_prefix + "down_proj.weight": torch.ones(4, 4, dtype=torch.int8),
    }
    owner, loader = _minimal_moe_loader_state(tensors)

    with pytest.raises(ValueError, match=error):
        DeepseekV4ForCausalLM._load_dsv4_moe(
            owner,
            loader,
            "layers.0.",
            "model.layers.0.",
            layer_id=0,
            source_prefix=prefix,
        )


def test_moe_loader_rejects_missing_quantization_scale() -> None:
    prefix = "layers.0.mlp."
    expert_prefix = prefix + "experts.0."
    tensors = {
        prefix + "gate.weight": torch.ones(2, 4),
        prefix + "gate.tid2eid": torch.arange(2),
        expert_prefix + "gate_proj.weight": torch.ones(4, 4, dtype=torch.int8),
        expert_prefix + "up_proj.weight": torch.ones(4, 4, dtype=torch.int8),
        expert_prefix + "down_proj.weight": torch.ones(4, 4, dtype=torch.int8),
        expert_prefix + "gate_proj.weight_scale": torch.ones(4, 1),
        expert_prefix + "up_proj.weight_scale": torch.ones(4, 1),
    }
    owner, loader = _minimal_moe_loader_state(tensors)

    with pytest.raises(KeyError, match="expert 0 down scale not found"):
        DeepseekV4ForCausalLM._load_dsv4_moe(
            owner,
            loader,
            "layers.0.",
            "model.layers.0.",
            layer_id=0,
            source_prefix=prefix,
        )


def test_moe_loader_loads_packed_w4a8_expert_tensors() -> None:
    prefix = "layers.0.mlp."
    expert_prefix = prefix + "experts.0."
    tensors = {
        prefix + "gate.weight": torch.ones(2, 4),
        prefix + "gate.tid2eid": torch.arange(2),
        expert_prefix + "gate_proj.weight": torch.ones(2, 4, dtype=torch.int8),
        expert_prefix + "up_proj.weight": torch.ones(2, 4, dtype=torch.int8),
        expert_prefix + "down_proj.weight": torch.ones(2, 4, dtype=torch.int8),
        expert_prefix + "gate_proj.weight_scale": torch.ones(4, 1),
        expert_prefix + "up_proj.weight_scale": torch.ones(4, 1),
        expert_prefix + "down_proj.weight_scale": torch.ones(4, 1),
        expert_prefix + "gate_proj.weight_scale_second": torch.ones(4, 1),
        expert_prefix + "up_proj.weight_scale_second": torch.ones(4, 1),
        expert_prefix + "down_proj.weight_scale_second": torch.ones(4, 1),
        expert_prefix + "gate_proj.scale_bias": torch.ones(4, 1),
        expert_prefix + "up_proj.scale_bias": torch.ones(4, 1),
        expert_prefix + "down_proj.scale_bias": torch.ones(4, 1),
    }
    owner, loader = _minimal_moe_loader_state(tensors)

    DeepseekV4ForCausalLM._load_dsv4_moe(
        owner,
        loader,
        "layers.0.",
        "model.layers.0.",
        layer_id=0,
        source_prefix=prefix,
    )

    mlp = owner.model.layers[0].mlp
    assert mlp.w4a8_dynamic
    assert owner.get_parameter("model.layers.0.mlp.experts_w13").shape == (1, 4, 4)
    assert owner.get_parameter("model.layers.0.mlp.experts_w2").shape == (1, 2, 4)
    for name, shape in (
        ("experts_w13_scale", (1, 8, 1)),
        ("experts_w13_scale_second", (1, 8, 1)),
        ("experts_w13_scale_bias", (1, 8, 1)),
        ("experts_w2_scale", (1, 4, 1)),
        ("experts_w2_scale_second", (1, 4, 1)),
        ("experts_w2_scale_bias", (1, 4, 1)),
    ):
        assert owner.get_buffer("model.layers.0.mlp." + name).shape == shape


def test_moe_loader_rejects_packed_w4a8_with_odd_hidden_size() -> None:
    prefix = "layers.0.mlp."
    expert_prefix = prefix + "experts.0."
    tensors = {
        prefix + "gate.weight": torch.ones(2, 5),
        prefix + "gate.tid2eid": torch.arange(2),
        expert_prefix + "gate_proj.weight": torch.ones(2, 5, dtype=torch.int8),
        expert_prefix + "up_proj.weight": torch.ones(2, 5, dtype=torch.int8),
        expert_prefix + "down_proj.weight": torch.ones(2, 4, dtype=torch.int8),
    }
    owner, loader = _minimal_moe_loader_state(tensors)
    owner.cfg.hidden_size = 5

    with pytest.raises(ValueError, match="requires an even hidden size"):
        DeepseekV4ForCausalLM._load_dsv4_moe(
            owner,
            loader,
            "layers.0.",
            "model.layers.0.",
            layer_id=0,
            source_prefix=prefix,
        )


def test_moe_loader_preserves_legacy_shared_up_projection_scale() -> None:
    prefix = "layers.0.ffn."
    expert_prefix = prefix + "experts.0."
    tensors = {
        prefix + "gate.weight": torch.ones(2, 4),
        prefix + "gate.tid2eid": torch.arange(2),
        expert_prefix + "w1.weight": torch.ones(4, 4, dtype=torch.int8),
        expert_prefix + "w3.weight": torch.full((4, 4), 2, dtype=torch.int8),
        expert_prefix + "w2.weight": torch.full((4, 4), 3, dtype=torch.int8),
        expert_prefix + "w1.weight_scale": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        expert_prefix + "w2.weight_scale": torch.ones(4, 1),
    }
    owner, loader = _minimal_moe_loader_state(tensors)

    DeepseekV4ForCausalLM._load_dsv4_moe(
        owner, loader, "layers.0.", "model.layers.0.", layer_id=0, source_prefix=prefix
    )

    torch.testing.assert_close(
        owner.get_buffer("model.layers.0.mlp.experts_w13_scale")[0],
        torch.tensor([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0], [3.0], [4.0]]),
    )
