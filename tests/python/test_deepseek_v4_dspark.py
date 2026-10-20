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

"""DeepSeek-V4 Python DSpark model tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xllm.python import registry
from xllm.python.model_executor.executor import _is_deepseek_v4_model_type
from xllm.python.models import deepseek_v4_dspark, deepseek_v32
from xllm.python.models.deepseek_v4_dspark import (
    DeepseekV4DSparkConfig,
    DeepseekV4DSparkForCausalLM,
)
from xllm.python.registry import get_model_class

_DSPARK_CONFIG = {
    "model_type": "deepseek_v4",
    "hidden_size": 8,
    "num_hidden_layers": 43,
    "num_attention_heads": 2,
    "head_dim": 4,
    "vocab_size": 16,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 16,
    "original_max_position_embeddings": 16,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 1,
        "original_max_position_embeddings": 16,
        "type": "yarn",
    },
    "q_lora_rank": 4,
    "qk_rope_head_dim": 2,
    "o_lora_rank": 4,
    "o_groups": 1,
    "compress_ratios": [0] * 43,
    "window_size": 8,
    "index_head_dim": 4,
    "index_n_heads": 2,
    "index_topk": 0,
    "n_activated_experts": 1,
    "num_hash_layers": 3,
    "hc_mult": 2,
    "hc_sinkhorn_iters": 2,
    "hc_eps": 1e-6,
    "scoring_func": "sqrtsoftplus",
    "scale_fmt": "ue8m0",
    "n_routed_experts": 2,
    "n_shared_experts": 1,
    "moe_intermediate_size": 4,
    "first_k_dense_replace": 0,
    "tie_word_embeddings": False,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_block_size": 5,
    "dspark_noise_token_id": 15,
    "dspark_markov_rank": 2,
    "tp_size": 1,
    "tp_rank": 0,
    "moe_tp_size": 1,
    "moe_tp_rank": 0,
    "ep_size": 1,
    "ep_rank": 0,
    "device": "cpu",
    "dtype": "float32",
}


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def has(self, name: str) -> bool:
        return name in self._tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


class _ContextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def write_context_kv(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_slots: torch.Tensor,
        swa_cache: torch.Tensor,
    ) -> None:
        self.calls.append((hidden, cos, sin, cache_slots, swa_cache))


class _LayerSynchronizer:
    def __init__(self, failed_layer: int | None = None) -> None:
        self.failed_layer = failed_layer
        self.layers: list[int] = []

    def record_event(self, layer_id: int) -> bool:
        self.layers.append(layer_id)
        return layer_id != self.failed_layer


def test_registry_resolves_deepseek_v4_dspark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry.current_platform, "device_type", lambda: "npu")
    assert get_model_class("deepseek_v4_dspark") is DeepseekV4DSparkForCausalLM
    assert _is_deepseek_v4_model_type("deepseek_v4_dspark")
    assert _is_deepseek_v4_model_type("deepseek_v4_mtp")


def test_dspark_config_builds_only_draft_layers() -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)

    assert isinstance(model.cfg, DeepseekV4DSparkConfig)
    assert model.cfg.n_layers == 3
    assert model.cfg.n_hash_layers == 0
    assert model.cfg.compress_ratios == [1, 1, 1]
    assert model.cfg.dspark_use_native_sas is False
    assert len(model.model.layers) == 3
    assert model.model.main_proj.in_features == 24
    assert model.model.main_proj.out_features == 8


def test_dspark_requires_draft_layers_and_markov_rank() -> None:
    with pytest.raises(ValueError, match="dspark_num_layers > 0"):
        DeepseekV4DSparkForCausalLM(
            {
                **_DSPARK_CONFIG,
                "dspark_target_layer_ids": [],
                "dspark_num_layers": 0,
            }
        )
    with pytest.raises(ValueError, match="dspark_markov_rank > 0"):
        DeepseekV4DSparkForCausalLM(
            {
                **_DSPARK_CONFIG,
                "dspark_markov_rank": 0,
            }
        )
    with pytest.raises(ValueError, match="target layer count"):
        DeepseekV4DSparkConfig.from_dict(
            {
                **_DSPARK_CONFIG,
                "dspark_num_layers": 2,
                "dspark_target_layer_ids": [1],
                "dspark_markov_rank": 4,
            }
        ).validate()


def test_dspark_confidence_head_is_biasless() -> None:
    model = DeepseekV4DSparkForCausalLM(
        {
            **_DSPARK_CONFIG,
            "enable_confidence_head": True,
        }
    )

    assert model.confidence_head is not None
    assert model.confidence_head.proj.bias is None
    assert model.confidence_head.proj.in_features == 10


def test_dspark_writes_projected_context_to_every_layer() -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)
    layers = nn.ModuleList([_ContextLayer() for _ in range(3)])
    model.model.layers = layers
    model.model.main_norm = nn.Identity()
    model.model.main_proj.resolve_quant(False)
    with torch.no_grad():
        model.model.main_proj.weight.zero_()
        model.model.main_proj.weight[:, :8].copy_(torch.eye(8))

    target_hidden = torch.arange(48, dtype=torch.float32).view(2, 24)
    positions = torch.tensor([1, 3])
    cache_slots = torch.tensor([2, 7])
    swa_caches = [torch.zeros(2, 4) for _ in range(3)]
    kv_caches = [(None, None, None, None, None, swa) for swa in swa_caches]
    synchronizer = _LayerSynchronizer()

    projected = model.write_context_kv(
        target_hidden,
        positions,
        cache_slots,
        kv_caches,
        synchronizer,
    )

    assert projected is not None
    torch.testing.assert_close(projected, target_hidden[:, :8])
    assert synchronizer.layers == [0, 1, 2]
    for layer, swa_cache in zip(layers, swa_caches, strict=True):
        assert len(layer.calls) == 1
        hidden, cos, sin, slots, written_cache = layer.calls[0]
        torch.testing.assert_close(hidden, projected)
        assert cos.shape == (2, 2)
        assert sin.shape == (2, 2)
        assert slots is cache_slots
        assert written_cache is swa_cache


def test_dspark_stops_context_write_when_event_recording_fails() -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)
    layers = nn.ModuleList([_ContextLayer() for _ in range(3)])
    model.model.layers = layers
    model.model.main_norm = nn.Identity()
    model.model.main_proj.resolve_quant(False)
    kv_caches = [(None, None, None, None, None, torch.zeros(2, 4)) for _ in range(3)]
    synchronizer = _LayerSynchronizer(failed_layer=1)

    result = model.write_context_kv(
        torch.zeros(2, 24),
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        kv_caches,
        synchronizer,
    )

    assert result is None
    assert synchronizer.layers == [0, 1]
    assert len(layers[0].calls) == 1
    assert len(layers[1].calls) == 1
    assert len(layers[2].calls) == 0


def test_dspark_loads_dedicated_vocab_and_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DeepseekV4DSparkForCausalLM(
        {
            **_DSPARK_CONFIG,
            "enable_confidence_head": True,
        }
    )
    monkeypatch.setattr(
        DeepseekV4DSparkForCausalLM,
        "_load_dsv4_decoder_layer",
        lambda self, loader, checkpoint_prefix, parameter_prefix, layer_id: None,
    )
    tensors: dict[str, torch.Tensor] = {}
    for layer_id in range(3):
        tensors[f"model.mtp.{layer_id}.attn.wq_a.weight"] = torch.ones(1)
    tensors.update(
        {
            "model.mtp.0.main_proj.weight": torch.full((8, 24), 1.0),
            "model.mtp.0.main_norm.weight": torch.full((8,), 2.0),
            "model.mtp.0.embed.weight": torch.full((16, 8), 3.0),
            "model.embed.weight": torch.full((16, 8), 30.0),
            "model.mtp.2.norm.weight": torch.full((8,), 4.0),
            "model.mtp.2.hc_head_fn": torch.full((2, 16), 5.0),
            "model.mtp.2.hc_head_base": torch.full((2,), 6.0),
            "model.mtp.2.hc_head_scale": torch.full((1,), 7.0),
            "model.mtp.2.markov_head.markov_w1.weight": torch.full((16, 2), 8.0),
            "model.mtp.2.markov_head.markov_w2.weight": torch.full((16, 2), 9.0),
            "model.mtp.2.confidence_head.proj.weight": torch.full((1, 10), 10.0),
            "model.mtp.2.head.weight": torch.full((16, 8), 11.0),
            "model.head.weight": torch.full((16, 8), 110.0),
        }
    )

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    torch.testing.assert_close(model.model.main_proj.weight, tensors["model.mtp.0.main_proj.weight"])
    torch.testing.assert_close(model.model.main_norm.weight, tensors["model.mtp.0.main_norm.weight"])
    torch.testing.assert_close(model.model.embed_tokens.weight, tensors["model.mtp.0.embed.weight"])
    torch.testing.assert_close(model.model.norm.weight, tensors["model.mtp.2.norm.weight"])
    torch.testing.assert_close(model.model.hc_head_fn, tensors["model.mtp.2.hc_head_fn"])
    torch.testing.assert_close(
        model.markov_head.markov_w1.weight,
        tensors["model.mtp.2.markov_head.markov_w1.weight"],
    )
    assert model.confidence_head is not None
    torch.testing.assert_close(
        model.confidence_head.proj.weight,
        tensors["model.mtp.2.confidence_head.proj.weight"],
    )
    torch.testing.assert_close(model.lm_head.weight, tensors["model.mtp.2.head.weight"])


def test_dspark_loads_quantized_main_projection_and_head(monkeypatch: pytest.MonkeyPatch) -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)
    monkeypatch.setattr(
        DeepseekV4DSparkForCausalLM,
        "_load_dsv4_decoder_layer",
        lambda self, loader, checkpoint_prefix, parameter_prefix, layer_id: None,
    )
    monkeypatch.setattr(
        deepseek_v32.kernels,
        "prepare_quant_weight",
        lambda weight: weight.transpose(0, 1).contiguous(),
        raising=False,
    )
    tensors: dict[str, torch.Tensor] = {
        "model.mtp.0.main_proj.weight": torch.arange(8 * 24, dtype=torch.int8).reshape(8, 24),
        "model.mtp.0.main_proj.weight_scale": torch.arange(1, 9, dtype=torch.float32).view(8, 1),
        "model.mtp.0.main_norm.weight": torch.ones(8),
        "model.mtp.0.embed.weight": torch.ones(16, 8),
        "model.mtp.2.norm.weight": torch.ones(8),
        "model.mtp.2.hc_head_fn": torch.ones(2, 16),
        "model.mtp.2.hc_head_base": torch.ones(2),
        "model.mtp.2.hc_head_scale": torch.ones(1),
        "model.mtp.2.markov_head.markov_w1.weight": torch.ones(16, 2),
        "model.mtp.2.markov_head.markov_w2.weight": torch.ones(16, 2),
        "model.mtp.2.head.weight": torch.arange(16 * 8, dtype=torch.int8).reshape(16, 8),
        "model.mtp.2.head.weight_scale": torch.arange(1, 17, dtype=torch.float32).view(16, 1),
    }
    for layer_id in range(3):
        tensors[f"model.mtp.{layer_id}.attn.wq_a.weight"] = torch.ones(1)

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    assert model.model.main_proj.use_w8a8 is True
    assert model.model.main_proj.weight is None
    assert model.model.main_proj._w8a8 is not None
    torch.testing.assert_close(
        model.model.main_proj._w8a8.weight,
        tensors["model.mtp.0.main_proj.weight"].transpose(0, 1),
    )
    torch.testing.assert_close(
        model.model.main_proj._w8a8.weight_scale,
        tensors["model.mtp.0.main_proj.weight_scale"].flatten(),
    )
    assert model.lm_head.use_w8a8 is True
    assert model.lm_head.weight is None
    assert model.lm_head._w8a8 is not None
    torch.testing.assert_close(
        model.lm_head._w8a8.weight_scale,
        tensors["model.mtp.2.head.weight_scale"].flatten(),
    )


def test_dspark_attention_loader_requires_weight_scale() -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)
    tensors = {
        "model.mtp.0.attn.wq_a.weight": torch.ones(4, 8, dtype=torch.int8),
    }
    loader = deepseek_v4_dspark.W8A8WeightLoader(
        model,
        [_StateDict(tensors)],
        tp_size=1,
        tp_rank=0,
        src_prefixes=("", "model."),
    )

    with pytest.raises(KeyError, match="wq_a.weight_scale"):
        model._load_dsv4_decoder_layer(
            loader,
            "mtp.0.",
            "model.layers.0.",
            layer_id=0,
        )


def test_dspark_moe_loader_receives_explicit_checkpoint_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)
    loader = _StateDict({"mtp.0.ffn.experts.0.w1.weight": torch.ones(1)})
    attention = model.model.layers[0].self_attn
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        model,
        "_load_dsv4_attention",
        lambda *args, **kwargs: (attention, None),
    )
    monkeypatch.setattr(attention, "process_weights_after_loading", lambda: None)
    monkeypatch.setattr(
        model.model.layers[0].mlp,
        "process_weights_after_loading",
        lambda: None,
    )
    monkeypatch.setattr(model, "_load_dsv4_moe", lambda *args: calls.append(args))

    model._load_dsv4_decoder_layer(
        loader,
        "mtp.0.",
        "model.layers.0.",
        layer_id=0,
    )

    assert calls == [
        (
            loader,
            "mtp.0.",
            "model.layers.0.",
            0,
            "mtp.0.ffn.",
        )
    ]


@pytest.mark.parametrize(
    ("gate_name", "up_name", "down_name"),
    [
        ("gate_proj", "up_proj", "down_proj"),
        ("w1", "w3", "w2"),
    ],
)
def test_dspark_loads_moe_from_mlp_checkpoint_prefix(
    monkeypatch: pytest.MonkeyPatch,
    gate_name: str,
    up_name: str,
    down_name: str,
) -> None:
    model = DeepseekV4DSparkForCausalLM(_DSPARK_CONFIG)
    source_prefix = "mtp.0.mlp."
    tensors = {
        source_prefix + "gate.weight": torch.arange(16, dtype=torch.float32).reshape(2, 8),
        source_prefix + "gate.bias": torch.tensor([0.25, -0.5]),
    }
    for expert_id in range(2):
        expert_prefix = source_prefix + f"experts.{expert_id}."
        tensors.update(
            {
                expert_prefix + gate_name + ".weight": torch.full((4, 8), expert_id + 1, dtype=torch.int8),
                expert_prefix + gate_name + ".weight_scale": torch.ones(4, 1),
                expert_prefix + up_name + ".weight": torch.full((4, 8), expert_id + 3, dtype=torch.int8),
                expert_prefix + up_name + ".weight_scale": torch.ones(4, 1),
                expert_prefix + down_name + ".weight": torch.full((8, 4), expert_id + 5, dtype=torch.int8),
                expert_prefix + down_name + ".weight_scale": torch.ones(8, 1),
            }
        )
    loader = deepseek_v4_dspark.W8A8WeightLoader(
        model,
        [_StateDict(tensors)],
        tp_size=1,
        tp_rank=0,
        src_prefixes=("", "model."),
    )
    attention = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp
    monkeypatch.setattr(
        model,
        "_load_dsv4_attention",
        lambda *args, **kwargs: (attention, "mtp.0.attn."),
    )
    monkeypatch.setattr(attention, "process_weights_after_loading", lambda: None)
    monkeypatch.setattr(mlp, "process_weights_after_loading", lambda: None)

    model._load_dsv4_decoder_layer(
        loader,
        "mtp.0.",
        "model.layers.0.",
        layer_id=0,
    )

    torch.testing.assert_close(mlp.gate.weight, tensors[source_prefix + "gate.weight"])
    torch.testing.assert_close(mlp.e_score_correction_bias, tensors[source_prefix + "gate.bias"])
    for expert_id in range(2):
        expert_prefix = source_prefix + f"experts.{expert_id}."
        torch.testing.assert_close(
            mlp.experts_w13[expert_id],
            torch.cat(
                [
                    tensors[expert_prefix + gate_name + ".weight"],
                    tensors[expert_prefix + up_name + ".weight"],
                ],
                dim=0,
            ),
        )
        torch.testing.assert_close(
            mlp.experts_w2[expert_id],
            tensors[expert_prefix + down_name + ".weight"],
        )
