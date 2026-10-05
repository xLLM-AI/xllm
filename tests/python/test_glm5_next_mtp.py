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

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip("torch_npu")

from xllm.python.models import glm5_next_mtp


class _StateDict:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def has(self, name: str) -> bool:
        return name in self.tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]


@pytest.mark.parametrize("prefix", ["model.language_model.layers.45.", "model.layers.45."])
def test_native_mtp_weights_do_not_resolve_target_weights(prefix: str) -> None:
    names = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.enorm.weight": "enorm.weight",
        "model.hnorm.weight": "hnorm.weight",
        "model.eh_proj.weight": "eh_proj.weight",
        "model.norm.weight": "shared_head.norm.weight",
        "lm_head.weight": "shared_head.head.weight",
        "model.layers.0.input_layernorm.weight": "input_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight": "self_attn.q_b_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale": "mlp.experts.0.gate_proj.weight_scale",
    }
    target = _StateDict({name: torch.tensor(-1.0) for name in names})
    draft = _StateDict({prefix + suffix: torch.tensor(2.0) for suffix in names.values()})
    loader = glm5_next_mtp.QLinearWeightLoader(
        torch.nn.Module(),
        [target, draft],
        1,
        0,
        name_aliases=glm5_next_mtp._get_mtp_name_aliases([target, draft], 45),
    )
    for name, suffix in names.items():
        assert loader.find(name) is draft
        assert loader.get_tensor(name) is draft.get_tensor(prefix + suffix)


def test_exported_mtp_weights_remain_unchanged() -> None:
    state_dicts = [_StateDict({"model.enorm.weight": torch.ones(4)})]
    assert glm5_next_mtp._get_mtp_name_aliases(state_dicts, -1) is None


def test_native_mtp_missing_layer_fails_instead_of_using_target() -> None:
    state_dicts = [_StateDict({"model.enorm.weight": torch.ones(4)})]
    with pytest.raises(ValueError, match="45"):
        glm5_next_mtp._get_mtp_name_aliases(state_dicts, 45)


def test_native_mtp_missing_head_does_not_use_target_head() -> None:
    source = _StateDict(
        {
            "model.language_model.layers.45.enorm.weight": torch.ones(4),
            "lm_head.weight": torch.ones(4, 4),
        }
    )
    loader = glm5_next_mtp.QLinearWeightLoader(
        torch.nn.Module(),
        [source],
        1,
        0,
        name_aliases=glm5_next_mtp._get_mtp_name_aliases([source], 45),
    )
    assert not loader.has("lm_head.weight")


def test_native_mtp_quant_probe_and_tp_sharding() -> None:
    prefix = "model.language_model.layers.45."
    weights = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    source = _StateDict(
        {
            prefix + "enorm.weight": torch.ones(4),
            prefix + "eh_proj.weight": weights,
            prefix + "mlp.experts.0.gate_proj.weight_scale": torch.ones(4, 1),
        }
    )
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.eh_proj = torch.nn.Linear(8, 2, bias=False)
    loader = glm5_next_mtp.QLinearWeightLoader(
        model,
        [source],
        2,
        1,
        name_aliases=glm5_next_mtp._get_mtp_name_aliases([source], 45),
    )
    assert loader.probe_quant("model.layers.0.mlp.experts.0.", "gate_proj")
    assert not loader.probe_quant("model.layers.0.self_attn.", "q_a_proj")
    loader.copy_shard("model.eh_proj.weight", dim=0)
    torch.testing.assert_close(model.model.eh_proj.weight, weights[2:])


class _SafeTensorStateDict:
    def __init__(self, source: Any) -> None:
        self.source = source
        self.keys = set(source.keys())
        self.read_keys: set[str] = set()

    def has(self, name: str) -> bool:
        return name in self.keys

    def get_tensor(self, name: str) -> torch.Tensor:
        self.read_keys.add(name)
        return self.source.get_tensor(name)


@pytest.mark.skipif(not os.getenv("XLLM_GLM_MTP_MODEL_PATH"), reason="real checkpoint not configured")
@pytest.mark.parametrize("tp_rank", [0, 7])
def test_real_native_mtp_checkpoint_loads(tp_rank: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Check real tensor loading on CPU, excluding the device-only NZ layout cast."""
    from safetensors import safe_open

    from xllm.python import kernels
    from xllm.python.kernels_npu.linear import prepare_quant_weight

    def _cpu_format_cast(tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.device.type == "cpu"
        return tensor

    monkeypatch.setattr(kernels, "format_cast_nz", _cpu_format_cast, raising=False)
    monkeypatch.setattr(kernels, "prepare_quant_weight", prepare_quant_weight, raising=False)
    model_path = Path(os.environ["XLLM_GLM_MTP_MODEL_PATH"])
    config = json.loads((model_path / "config.json").read_text())
    start_layer = config["text_config"]["num_hidden_layers"]
    config.update(
        model_type="glm5_next_mtp",
        n_layers=1,
        first_k_dense_replace=0,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["sparse"],
        indexer_types=["full"],
        mtp_start_layer_idx=start_layer,
        device="cpu",
        dtype="bfloat16",
        tp_size=8,
        tp_rank=tp_rank,
    )
    with ExitStack() as stack:
        state_dicts = [
            _SafeTensorStateDict(stack.enter_context(safe_open(str(path), framework="pt")))
            for path in sorted(model_path.glob("*.safetensors"))
        ]
        model = glm5_next_mtp.Glm5NextMtpForCausalLM(config)
        model.load_weights(state_dicts, tp_rank=tp_rank, tp_size=8)
        prefix = f"model.language_model.layers.{start_layer}."
        read_keys = set().union(*(source.read_keys for source in state_dicts))
        assert read_keys
        assert all(name.startswith(prefix) for name in read_keys)
        assert prefix + "shared_head.head.weight" in read_keys
        assert model.model.layers[0].mlp.use_w8a8
        assert model.model.layers[0].mlp.experts_w13.dtype == torch.int8
        for model_name, checkpoint_name, dimension in (
            ("model.embed_tokens.weight", "embed_tokens.weight", 1),
            ("lm_head.weight", "shared_head.head.weight", 0),
            ("model.eh_proj.weight", "eh_proj.weight", 0),
        ):
            source = next(source for source in state_dicts if source.has(prefix + checkpoint_name))
            tensor = source.get_tensor(prefix + checkpoint_name)
            shard_size = tensor.shape[dimension] // 8
            expected = tensor.narrow(dimension, tp_rank * shard_size, shard_size)
            torch.testing.assert_close(model.get_parameter(model_name), expected, rtol=0, atol=0)


def _make_cfg(**overrides: Any) -> glm5_next_mtp.Glm5NextConfig:
    """Real one-layer MTP configuration for the fail-fast prelude."""
    defaults: dict[str, Any] = {
        "hidden_size": 16,
        "vocab_size": 32,
        "rms_norm_eps": 1e-6,
        "n_layers": 1,
        "first_k_dense_replace": 0,
        "tp_size": 1,
        "tp_rank": 0,
        "layer_types": ["deepseek_sparse_attention"],
        "mlp_layer_types": ["sparse"],
        "indexer_types": ["full"],
    }
    defaults.update(overrides)
    return glm5_next_mtp.Glm5NextConfig(**defaults)


def _instantiate_model(cfg: glm5_next_mtp.Glm5NextConfig) -> None:
    """Trigger fail-fast checks before module construction."""
    glm5_next_mtp.Glm5NextMtpModel(cfg, torch.float32, torch.device("cpu"))


def test_rejects_non_divisible_hidden_size() -> None:
    cfg = _make_cfg(hidden_size=17, tp_size=2)
    with pytest.raises(ValueError, match="hidden_size"):
        _instantiate_model(cfg)


def test_rejects_non_divisible_vocab_size() -> None:
    cfg = _make_cfg(vocab_size=33, tp_size=2)
    with pytest.raises(ValueError, match="vocab_size"):
        _instantiate_model(cfg)


def test_rejects_multi_layer_config() -> None:
    cfg = _make_cfg(n_layers=45)
    with pytest.raises(ValueError, match="1 layer"):
        _instantiate_model(cfg)


def test_rejects_dense_layer_0() -> None:
    cfg = _make_cfg()
    cfg.mlp_layer_types = ["dense"]
    with pytest.raises(ValueError, match="must be MoE"):
        _instantiate_model(cfg)


def test_rejects_shared_indexer_layer_0() -> None:
    cfg = _make_cfg()
    cfg.indexer_types = ["shared"]
    with pytest.raises(ValueError, match="full indexer"):
        _instantiate_model(cfg)
