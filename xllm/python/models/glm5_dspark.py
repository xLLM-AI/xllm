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

"""Dense MLA DSpark draft with compressed latent/RoPE KV caches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from xllm_kernel.ops import reshape_paged_cache

from xllm.python.layers import Attention, GatedMLP, RMSNorm, RotaryEmbedding, RowParallelLinear
from xllm.python.layers.rotary_embedding import apply_rotary_half, gather_half_rope_cos_sin
from xllm.python.model_executor.forward_context import LayerSynchronizer, get_forward_context, record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    load_causal_lm_weights,
    load_draft_embedding_if_present,
    load_draft_lm_head_if_present,
    load_missing_draft_vocab_from_quarot_target,
    mla_head_split,
)
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.qwen3_dflash import DFlashContextProjection, _load_reference_quarot_rotation
from xllm.python.models.qwen3_dspark import Qwen3DSparkConfig


@dataclass
class Glm5DSparkConfig(Qwen3DSparkConfig):
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    rope_interleave: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> Glm5DSparkConfig:
        values = dict(config.get("transformer_layer_config") or {})
        values.update({key: value for key, value in config.items() if value is not None})
        values["head_dim"] = int(values.get("qk_nope_head_dim", 192)) + int(values.get("qk_rope_head_dim", 64))
        values["n_kv_heads"] = 1
        values["sliding_window"] = max(int(values.get("sliding_window") or 0), 0)
        values.setdefault("tie_word_embeddings", False)
        base = Qwen3DSparkConfig.from_dict(values)
        return cls(
            **base.__dict__,
            q_lora_rank=int(values.get("q_lora_rank", 2048)),
            kv_lora_rank=int(values.get("kv_lora_rank", 512)),
            qk_nope_head_dim=int(values.get("qk_nope_head_dim", 192)),
            qk_rope_head_dim=int(values.get("qk_rope_head_dim", 64)),
            v_head_dim=int(values.get("v_head_dim", 256)),
            rope_interleave=bool(values.get("rope_interleave", True)),
        )

    def head_split(self) -> tuple[int, int]:
        return mla_head_split(self.n_heads, self.tp_size)

    def validate(self) -> None:
        super().validate()
        if min(self.q_lora_rank, self.kv_lora_rank, self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim) <= 0:
            raise ValueError("MLA DSpark projection dimensions must be positive")
        if self.qk_rope_head_dim % 2:
            raise ValueError("MLA DSpark RoPE dimension must be even")
        if self.attention_bias or self.sliding_window:
            raise ValueError("MLA DSpark supports bias-free full attention only")


def _apply_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        return apply_rotary_half(value, cos, sin)
    pairs = value.reshape(*value.shape[:-1], -1, 2)
    first, second = pairs.unbind(-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1).flatten(-2)


class Glm5DSparkAttention(Attention):
    def __init__(self, cfg: Glm5DSparkConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        heads, _ = cfg.head_split()
        super().__init__(
            heads, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.head_dim**-0.5, 0, layer_id, causal=False
        )
        self.cfg = cfg
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.q_a_proj = nn.Linear(cfg.hidden_size, cfg.q_lora_rank, bias=False, dtype=dtype, device=device)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, heads * cfg.head_dim, bias=False, dtype=dtype, device=device)
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False, dtype=dtype, device=device
        )
        self.kv_a_layernorm = RMSNorm(cfg.kv_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank, heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), bias=False, dtype=dtype, device=device
        )
        self.o_proj = RowParallelLinear(
            heads * cfg.v_head_dim, cfg.hidden_size, cfg.tp_size, dtype=dtype, device=device
        )

    def _kv_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Views of the loaded TP shard: do not retain a second expanded KV weight.
        weight = self.kv_b_proj.weight.view(
            self.num_heads, self.cfg.qk_nope_head_dim + self.cfg.v_head_dim, self.cfg.kv_lora_rank
        )
        return weight.split((self.cfg.qk_nope_head_dim, self.cfg.v_head_dim), dim=1)

    def project_context(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent, rope = self.kv_a_proj_with_mqa(hidden).split((self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim), dim=-1)
        return self.kv_a_layernorm(latent).unsqueeze(1), _apply_rope(
            rope.unsqueeze(1), cos, sin, interleaved=self.cfg.rope_interleave
        )

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        query = query.view(hidden.shape[0], self.num_heads, self.cfg.head_dim)
        q_nope, q_rope = query.split((self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim), dim=-1)
        key_weight, value_weight = self._kv_weights()
        q_latent = torch.einsum("thd,hdr->thr", q_nope, key_weight)
        q_pe = _apply_rope(q_rope, cos, sin, interleaved=self.cfg.rope_interleave)
        latent, key_pe = self.project_context(hidden, cos, sin)
        output = get_forward_context().attention_backend.execute_mla(q_latent.contiguous(), q_pe, latent, key_pe, self)
        values = torch.einsum("thr,hvr->thv", output, value_weight)
        return self.o_proj(values.reshape(hidden.shape[0], -1))

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        for name in ("q_a_proj", "q_a_layernorm", "kv_a_proj_with_mqa", "kv_a_layernorm"):
            weights.load_tensor(getattr(self, name).weight, f"{name}.weight")
        for name in ("q_b_proj", "kv_b_proj"):
            weights.load_tensor(
                getattr(self, name).weight, f"{name}.weight", dim=0, rank=context.tp_rank, world_size=context.tp_size
            )
        weights.load_tensor(
            self.o_proj.weight, "o_proj.weight", dim=1, rank=context.tp_rank, world_size=context.tp_size
        )
        self.o_proj.process_weights_after_loading()


class Glm5DSparkDecoderLayer(nn.Module):
    def __init__(self, cfg: Glm5DSparkConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Glm5DSparkAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size, cfg.tp_size, dtype, device)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights.load_tensor(getattr(self, name).weight, f"{name}.weight")
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        self.mlp.process_weights_after_loading()


class Glm5DSparkModel(nn.Module):
    def __init__(self, cfg: Glm5DSparkConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg, self.dtype, self.device = cfg, dtype, device
        self.embed_tokens = None
        self.context_proj = DFlashContextProjection(cfg.hidden_size, cfg.tp_size, dtype, device)
        self.context_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = RotaryEmbedding(cfg.qk_rope_head_dim, cfg.max_position_embeddings, cfg.rope_theta, dtype, device)
        self.layers = nn.ModuleList(Glm5DSparkDecoderLayer(cfg, index, dtype, device) for index in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)

    @property
    def context_projection_weight(self) -> torch.Tensor:
        return self.context_proj.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("MLA DSpark embedding weights are not loaded")
        hidden = self.embed_tokens(input_ids)
        cos, sin = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions.to(torch.long))
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
            record_layer_event(layer.layer_id)
        return self.norm(hidden)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        if (
            len(kv_caches) != self.cfg.n_layers
            or positions.numel() != target_hidden.shape[0]
            or cache_slots.shape != positions.shape
        ):
            raise ValueError("MLA context hidden, positions, slots and cache layers must agree")
        hidden = self.context_norm(self.context_proj(target_hidden))
        cos, sin = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions.to(torch.long))
        for layer, cache in zip(self.layers, kv_caches):
            if cache[0] is None or cache[1] is None:
                raise RuntimeError(f"MLA draft KV cache is missing for layer {layer.layer_id}")
            latent, rope = layer.self_attn.project_context(hidden, cos, sin)
            reshape_paged_cache(cache_slots, latent, rope, cache[0], cache[1])
            if layer_synchronizer is not None and layer_synchronizer.record_event(layer.layer_id) is False:
                return None
        return hidden

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> ScopedWeightLoader:
        weights = ScopedWeightLoader(state_dicts, src_prefixes=("", "model."))
        context = ParallelLoadContext(tp_rank, tp_size)
        backbone = load_causal_lm_weights(self, None, weights, context, tie_word_embeddings=False, load_embedding=False)
        self.context_proj.load_weight(backbone.get_tensor("context_proj.weight"), tp_rank)
        backbone.load_tensor(self.context_norm.weight, "context_norm.weight")
        load_draft_embedding_if_present(self, weights, context=context)
        return weights


class Glm5DSparkForCausalLM(DSparkForCausalLMBase):
    def __init__(self, config: dict) -> None:
        if int(config.get("kv_split_size", 1)) > 1 or int(config.get("cp_size", 1)) > 1:
            raise NotImplementedError(
                "Python MLA DSpark requires unsharded KV caches; use kv_split_size=1 and cp_size=1"
            )
        cfg = Glm5DSparkConfig.from_dict(config)
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
        self.cfg, self.dtype, self.device = cfg, dtype, device
        self.model = Glm5DSparkModel(cfg, dtype, device)
        self.lm_head = None

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        weights = self.model.load_weights(state_dicts, tp_rank, tp_size)
        load_draft_lm_head_if_present(self, weights, context=ParallelLoadContext(tp_rank, tp_size))
        for name in ("markov_w1", "markov_w2"):
            weights.load_tensor(getattr(self.markov_head, name).weight, f"markov_head.{name}.weight")
        if self.confidence_head is not None:
            weights.load_tensor(self.confidence_head.proj.weight, "confidence_head.proj.weight")
            weights.load_tensor(self.confidence_head.proj.bias, "confidence_head.proj.bias")

    def adapt_weights_for_reference_model(self, reference_model_path: str) -> None:
        rotation = _load_reference_quarot_rotation(reference_model_path)
        if rotation is not None:
            self.model.context_proj.apply_input_rotation(rotation)
            load_missing_draft_vocab_from_quarot_target(self, reference_model_path, rotation)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        return self.model.write_context_kv(target_hidden, positions, cache_slots, kv_caches, layer_synchronizer)
