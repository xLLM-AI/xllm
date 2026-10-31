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
from typing import ClassVar

import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_loader import (
    ParallelLoadContext,
    load_draft_lm_head_if_present,
    load_missing_draft_vocab_from_quarot_target,
)
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.qwen3_dflash import (
    DFlashQwen3Config,
    DFlashQwen3Model,
)


@dataclass
class Qwen3DSparkConfig(DFlashQwen3Config):
    # DSpark remaps draft ids to target ids (checkpoint "d2t" table), so a
    # draft vocabulary smaller than the target's is supported.
    supports_reduced_draft_vocab: ClassVar[bool] = True

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
    model: Qwen3DSparkModel

    def __init__(self, config: dict) -> None:
        cfg = Qwen3DSparkConfig.from_dict(config)
        cfg.validate()
        causal = cfg.sliding_window > 0 and not config.get("sliding_window_non_causal", False)
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
        self.model = Qwen3DSparkModel(cfg, dtype, device, causal=causal)  # pyright: ignore[reportIncompatibleVariableOverride]
        self.lm_head: nn.Module | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        all_weights = self.model.load_weights(state_dicts, tp_rank, tp_size)
        load_draft_lm_head_if_present(
            self,
            all_weights,
            context=ParallelLoadContext(tp_rank, tp_size),
        )
        all_weights.load_tensor(self.markov_head.markov_w1.weight, "markov_head.markov_w1.weight")
        all_weights.load_tensor(self.markov_head.markov_w2.weight, "markov_head.markov_w2.weight")
        if self.confidence_head is not None:
            all_weights.load_tensor(self.confidence_head.proj.weight, "confidence_head.proj.weight")
            all_weights.load_tensor(self.confidence_head.proj.bias, "confidence_head.proj.bias")
        # target id = d2t[draft_id] + draft_id; a same-width permutation is a
        # valid mapping too. Absent = draft and target share the vocabulary.
        if all_weights.has("d2t"):
            d2t = all_weights.get_tensor("d2t")
            if d2t.dim() != 1:
                raise ValueError("d2t table must be 1D")
            if d2t.numel() != self.cfg.draft_vocab_size:
                raise ValueError(
                    f"d2t table has {d2t.numel()} entries but draft_vocab_size is {self.cfg.draft_vocab_size}"
                )
            mapped_ids = d2t.to(dtype=torch.int64) + torch.arange(d2t.numel(), device=d2t.device, dtype=torch.int64)
            if torch.unique(mapped_ids).numel() != mapped_ids.numel():
                raise ValueError("d2t mapped target ids must be unique")
            if int(mapped_ids.min()) < 0 or int(mapped_ids.max()) >= self.cfg.vocab_size:
                raise ValueError(f"d2t mapped target ids must be in [0, {self.cfg.vocab_size})")
            self.hot_token_id = mapped_ids.to(device=self.device)
        elif self.cfg.draft_vocab_size != self.cfg.vocab_size:
            raise ValueError("reduced-vocabulary DSpark draft checkpoints must provide a d2t table")
        else:
            self.hot_token_id = None

    def adapt_weights_for_reference_model(
        self,
        reference_model_path: str,
    ) -> None:
        rotation = self.model.adapt_weights_for_reference_model(reference_model_path)
        if rotation is not None:
            load_missing_draft_vocab_from_quarot_target(self, reference_model_path, rotation)

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
