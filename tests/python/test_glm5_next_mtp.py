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

from typing import Any

import pytest
import torch

pytest.importorskip("torch_npu")

from xllm.python.models import glm5_next_mtp


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
