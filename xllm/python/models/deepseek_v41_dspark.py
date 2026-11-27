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

"""DeepSeek-V4.1's checkpoint-contained DSpark draft and context projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from xllm.python import kernels
from xllm.python.attention.backend import LayerCacheInput, normalize_layer_caches
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.model_loader import W8A8WeightLoader
from xllm.python.models.deepseek_v41 import (
    V41_QUANT_ASCEND,
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
    _detect_v41_quant_mode,
)
from xllm.python.models.dspark import DSparkForCausalLMBase


@dataclass
class DeepseekV41DSparkConfig(DeepseekV41Config):
    dspark_num_layers: int = 3
    markov_rank: int = 256

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DeepseekV41DSparkConfig:
        cfg = super().from_dict(config)
        count = int(config.get("dspark_num_layers", config.get("num_nextn_predict_layers", 3)))
        if count <= 0:
            raise ValueError("DeepSeek-V4.1 DSpark requires at least one draft layer")
        return replace(
            cfg,
            model_type="deepseek_v41_dspark",
            n_layers=count,
            dspark_num_layers=count,
            markov_rank=int(config.get("markov_rank", config.get("dspark_markov_rank", 256))),
            compress_ratios=[0] * count,
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            engram_layer_ids=[],
            engram_num_embeddings=[],
            layers_to_capture=(),
            first_k_dense_replace=0,
        )


class DeepseekV41DSparkModel(DeepseekV41Model):
    def __init__(
        self, cfg: DeepseekV41DSparkConfig, dtype: torch.dtype, device: torch.device, config: dict[str, Any]
    ) -> None:
        super().__init__(cfg, dtype, device, V41_QUANT_ASCEND, config)
        self.main_proj = nn.Linear(
            cfg.hidden_size * cfg.dspark_num_layers, cfg.hidden_size, bias=False, dtype=dtype, device=device
        )
        self.main_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.register_buffer("_context_wkv_weight", None, persistent=False)

    @property
    def context_projection_weight(self) -> torch.Tensor:
        return self.main_proj.weight

    def pack_context_wkv(self) -> None:
        projections = [layer.self_attn.kv_proj for layer in self.layers]
        if len(projections) <= 1 or not all(isinstance(proj, nn.Linear) for proj in projections):
            return
        weights = [proj.weight.detach() for proj in projections]
        if self._context_wkv_weight is None:
            self._context_wkv_weight = torch.cat(weights, dim=0)
        else:
            # Captured context-write graphs retain the packed weight address.
            torch.cat(weights, dim=0, out=self._context_wkv_weight)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[LayerCacheInput],
        layer_synchronizer: LayerSynchronizer | None,
        *,
        graph_mode: bool = False,
    ) -> torch.Tensor | None:
        from xllm.python.attention.csa2_attention import apply_interleaved_rope
        from xllm.python.attention.csa_attention import _scatter_by_slot

        projected = self.main_norm(self.main_proj(target_hidden))
        all_kv = None
        if self._context_wkv_weight is not None:
            all_kv = F.linear(projected, self._context_wkv_weight).unflatten(-1, (self.cfg.n_layers, self.cfg.head_dim))
            # RMSNorm and cache scatter consume contiguous rows for one layer.
            all_kv = all_kv.transpose(0, 1).contiguous()
        for layer_id, (layer, cache) in enumerate(zip(self.layers, normalize_layer_caches(kv_caches), strict=True)):
            if cache.swa is None:
                raise ValueError("DeepSeek-V4.1 DSpark context requires an SWA cache per draft layer")
            attention = layer.self_attn
            kv = attention.kv_proj(projected) if all_kv is None else all_kv[layer_id]
            kv = attention.kv_a_layernorm(kv)
            kv = apply_interleaved_rope(kv, positions, self.rotary.cos_sin_cache)
            if graph_mode:
                # The A3 scatter skips negative slots, including all warmup
                # rows and the unused tail of a replay bucket.
                kernels.scatter_nd_update(
                    cache.swa.view(-1, kv.shape[-1]), cache_slots.view(-1, 1), kv.to(cache.swa.dtype)
                )
            else:
                _scatter_by_slot(cache.swa, cache_slots, kv)
            if layer_synchronizer is not None and not layer_synchronizer.record_event(layer_id):
                return None
        return projected


class DeepseekV41DSparkForCausalLM(DeepseekV41ForCausalLM, DSparkForCausalLMBase):
    def __init__(self, config: dict[str, Any]) -> None:
        cfg = DeepseekV41DSparkConfig.from_dict(config)
        if _detect_v41_quant_mode(config) != V41_QUANT_ASCEND:
            raise ValueError("DeepSeek-V4.1 DSpark currently requires an Ascend W8A8 checkpoint")
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        DSparkForCausalLMBase.__init__(
            self,
            vocab_size=cfg.vocab_size,
            draft_vocab_size=cfg.vocab_size,
            markov_rank=cfg.markov_rank,
            hidden_size=cfg.hidden_size,
            enable_confidence_head=False,
            confidence_head_with_markov=True,
            confidence_head_bias=False,
            dtype=dtype,
            device=device,
        )
        self.cfg = cfg
        self.quant_mode = V41_QUANT_ASCEND
        self.model = DeepseekV41DSparkModel(cfg, dtype, device, config)
        self.lm_head = ColumnParallelLinear(
            cfg.hidden_size, cfg.vocab_size // cfg.tp_size, cfg.tp_size, gather_output=True, dtype=dtype, device=device
        )

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        return super().compute_logits(self.model.norm(hidden), selected_idxes)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[LayerCacheInput],
        layer_synchronizer: LayerSynchronizer | None,
        *,
        graph_mode: bool = False,
    ) -> torch.Tensor | None:
        return self.model.write_context_kv(
            target_hidden, positions, cache_slots, kv_caches, layer_synchronizer, graph_mode=graph_mode
        )

    def load_weights(self, state_dicts: list[Any], tp_rank: int, tp_size: int) -> None:
        last = f"mtp.{self.cfg.n_layers - 1}."
        self._load_weights_ascend(
            state_dicts, layer_prefix="mtp", embedding_key="mtp.0.embed.weight", norm_key=last + "norm.weight"
        )
        self.model.pack_context_wkv()
        loader = W8A8WeightLoader(self, state_dicts, tp_size, tp_rank)
        for name in ("main_proj.weight", "main_norm.weight"):
            loader.copy_in("model." + name, loader.get_tensor("mtp.0." + name))
        loader.copy_in("markov_head.markov_w1.weight", loader.get_tensor(last + "markov_head.embed.weight"))
        loader.copy_in("markov_head.markov_w2.weight", loader.get_tensor(last + "markov_head.head.weight"))

    def _load_lm_head(self, loader: W8A8WeightLoader) -> None:
        key = f"mtp.{self.cfg.n_layers - 1}.head.weight"
        loader.copy_in("lm_head.weight", loader.shard(loader.get_tensor(key), dim=0))
