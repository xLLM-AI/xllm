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

"""Qwen3 DSpark draft model for the Python NPU executor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.qwen3_dflash import (
    DFlashQwen3Config,
    DFlashQwen3Model,
)


@dataclass
class Qwen3DSparkConfig(DFlashQwen3Config):
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3DSparkConfig:
        base = DFlashQwen3Config.from_dict(d)
        return cls(
            **base.__dict__,
            markov_rank=int(d.get("markov_rank", 0)),
            enable_confidence_head=bool(d.get("enable_confidence_head", False)),
            confidence_head_with_markov=bool(d.get("confidence_head_with_markov", False)),
        )

    def validate(self) -> None:
        super().validate()
        if self.markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")


class Qwen3DSparkModel(DFlashQwen3Model):
    """DFlash draft backbone used with DSpark-specific output heads."""


class Qwen3DSparkForCausalLM(DSparkForCausalLMBase):
    def __init__(self, config: dict) -> None:
        cfg = Qwen3DSparkConfig.from_dict(config)
        cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu"))
        super().__init__(
            vocab_size=cfg.vocab_size,
            draft_vocab_size=cfg.draft_vocab_size,
            markov_rank=cfg.markov_rank,
            hidden_size=cfg.hidden_size,
            enable_confidence_head=cfg.enable_confidence_head,
            confidence_head_with_markov=cfg.confidence_head_with_markov,
            dtype=dtype,
            device=device,
        )
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.model = Qwen3DSparkModel(cfg, dtype, device)
        self.lm_head: nn.Module | None = None

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        self.model.load_weights(state_dicts, tp_rank, tp_size)

        def find(name: str):
            candidates = (name, f"model.{name}")
            for candidate in candidates:
                for state_dict in state_dicts:
                    if state_dict.has(candidate):
                        return state_dict, candidate
            return None

        def load_tensor(name: str) -> torch.Tensor:
            found = find(name)
            if found is None:
                raise KeyError(f"checkpoint tensor not found: {name}")
            state_dict, key = found
            return state_dict.get_tensor(key)

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            param = self.get_parameter(param_name)
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

        markov_w1 = load_tensor("markov_head.markov_w1.weight")
        markov_w2 = load_tensor("markov_head.markov_w2.weight")
        copy_in("markov_head.markov_w1.weight", markov_w1)
        copy_in("markov_head.markov_w2.weight", markov_w2)

        if self.confidence_head is not None:
            confidence_weight = load_tensor("confidence_head.proj.weight")
            confidence_bias = load_tensor("confidence_head.proj.bias")
            copy_in("confidence_head.proj.weight", confidence_weight)
            copy_in("confidence_head.proj.bias", confidence_bias)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        return self.model.write_context_kv(
            target_hidden,
            positions,
            cache_slots,
            kv_caches,
            layer_synchronizer,
        )
