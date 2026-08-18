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
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.layers import RMSNorm, RotaryEmbedding
from xllm.python.model_executor.forward_context import record_layer_event
from xllm.python.models.dspark import DSparkForCausalLMBase
from xllm.python.models.qwen3 import Qwen3Config, Qwen3DecoderLayer


@dataclass
class Qwen3DSparkConfig(Qwen3Config):
    draft_vocab_size: int = 0
    markov_rank: int = 0
    enable_confidence_head: bool = False
    confidence_head_with_markov: bool = False
    mask_token_id: int | None = None
    block_size: int = 0

    @classmethod
    def from_dict(cls, config: dict) -> Qwen3DSparkConfig:
        base = Qwen3Config.from_dict(config)
        draft_vocab_size = int(config.get("draft_vocab_size") or base.vocab_size)
        mask_token_id = config.get("mask_token_id")
        return cls(
            **base.__dict__,
            draft_vocab_size=draft_vocab_size,
            markov_rank=int(config.get("markov_rank", 0)),
            enable_confidence_head=bool(config.get("enable_confidence_head", False)),
            confidence_head_with_markov=bool(config.get("confidence_head_with_markov", False)),
            mask_token_id=None if mask_token_id is None else int(mask_token_id),
            block_size=int(config.get("dspark_block_size") or config.get("block_size") or 0),
        )

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("invalid Qwen3 DSpark dimensions")
        if self.markov_rank <= 0:
            raise ValueError("Qwen3 DSpark requires markov_rank > 0")
        if self.hidden_size % self.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        if self.vocab_size % self.tp_size:
            raise ValueError("vocab_size must be divisible by tp_size")
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError("reduced-vocabulary DSpark drafts are not supported")
        self.head_split()


class DSparkContextProjection(nn.Module):
    """Replicated projection with checkpoint-defined input width."""

    def __init__(self, out_features: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, 0, dtype=dtype, device=device))

    def load_weight(self, weight: torch.Tensor) -> None:
        if weight.dim() != 2 or weight.size(0) != self.out_features:
            raise ValueError("DSpark fc.weight has an invalid shape")
        self.weight = nn.Parameter(weight.to(dtype=self.weight.dtype, device=self.weight.device).contiguous())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.weight.size(1) == 0:
            raise RuntimeError("DSpark context projection weight is not loaded")
        if hidden.size(-1) != self.weight.size(1):
            raise ValueError("target auxiliary hidden size does not match DSpark fc.weight")
        return F.linear(hidden, self.weight)


class Qwen3DSparkDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        cfg: Qwen3DSparkConfig,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(cfg, layer_id, dtype, device)
        self.self_attn.attn.causal = False


class Qwen3DSparkModel(nn.Module):
    def __init__(self, cfg: Qwen3DSparkConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens: nn.Module | None = None
        self.fc = DSparkContextProjection(cfg.hidden_size, dtype, device)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            Qwen3DSparkDecoderLayer(cfg, layer_id, dtype, device) for layer_id in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.register_buffer("_fused_kv_weight", None, persistent=False)
        self.register_buffer("_fused_kv_bias", None, persistent=False)
        self.register_buffer("_k_norm_weights", None, persistent=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("DSpark target embedding has not been shared")
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            hidden, residual = layer(
                hidden,
                residual,
                positions,
                self.rotary.cos_sin_cache,
                None,
                None,
            )
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        return hidden

    def build_context_kv_buffers(self) -> None:
        kv_weights: list[torch.Tensor] = []
        kv_biases: list[torch.Tensor] = []
        k_norm_weights: list[torch.Tensor] = []
        has_bias = self.layers[0].self_attn.qkv_proj.bias is not None
        for layer in self.layers:
            attention = layer.self_attn
            kv_weights.append(attention.qkv_proj.weight[attention.q_size :])
            if has_bias:
                kv_biases.append(attention.qkv_proj.bias[attention.q_size :])
            k_norm_weights.append(attention.k_norm.weight)
        self._fused_kv_weight = torch.cat(kv_weights, dim=0).detach().contiguous()
        self._fused_kv_bias = torch.cat(kv_biases, dim=0).detach().contiguous() if has_bias else None
        self._k_norm_weights = torch.stack(k_norm_weights, dim=0).detach().view(self.cfg.n_layers, 1, 1, -1)

    def _normalize_context_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if self._k_norm_weights is None:
            raise RuntimeError("DSpark K-norm buffers are not initialized")
        keys_fp32 = keys.float()
        variance = keys_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = keys_fp32 * torch.rsqrt(variance + self.cfg.rms_norm_eps)
        return (normalized * self._k_norm_weights.float()).to(keys.dtype)

    def _apply_context_rope(self, keys: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        num_layers, num_context, num_kv_heads, head_dim = keys.shape
        flat_keys = keys.reshape(num_layers * num_context, num_kv_heads, head_dim)
        repeated_positions = positions.to(torch.long).repeat(num_layers)
        cache = self.rotary.cos_sin_cache.index_select(0, repeated_positions)
        cos_half, sin_half = cache.chunk(2, dim=-1)
        cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
        sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
        first, second = flat_keys.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (flat_keys * cos + rotated * sin).view_as(keys)

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: object | None,
    ) -> torch.Tensor | None:
        if len(kv_caches) != self.cfg.n_layers:
            raise ValueError("DSpark KV cache layer count mismatch")
        if cache_slots.numel() != target_hidden.size(0):
            raise ValueError("DSpark context cache slot count mismatch")
        if self._fused_kv_weight is None:
            self.build_context_kv_buffers()

        projected_hidden = self.hidden_norm(self.fc(target_hidden))
        if not isinstance(projected_hidden, torch.Tensor):
            raise TypeError("DSpark hidden_norm returned an unexpected value")
        all_kv = F.linear(projected_hidden, self._fused_kv_weight, self._fused_kv_bias)
        num_context = projected_hidden.size(0)
        num_kv_heads = self.layers[0].self_attn.num_kv_heads
        all_kv = all_kv.view(
            num_context,
            self.cfg.n_layers,
            2,
            num_kv_heads,
            self.cfg.head_dim,
        ).permute(2, 1, 0, 3, 4)
        all_keys = self._apply_context_rope(self._normalize_context_keys(all_kv[0]), positions)
        all_values = all_kv[1]

        for layer_id, cache in enumerate(kv_caches):
            key_cache, value_cache = cache[0], cache[1]
            if key_cache is None or value_cache is None:
                raise RuntimeError(f"DSpark KV cache is missing for layer {layer_id}")
            kernels.reshape_paged_cache(
                cache_slots,
                all_keys[layer_id].contiguous(),
                all_values[layer_id].contiguous(),
                key_cache,
                value_cache,
            )
            if layer_synchronizer is not None:
                if layer_synchronizer.record_event(layer_id) is False:
                    return None
        return projected_hidden


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
        cfg = self.cfg
        kv_replicas = tp_size // cfg.n_kv_heads if cfg.n_kv_heads < tp_size else 1
        kv_rank = tp_rank // kv_replicas if kv_replicas > 1 else tp_rank
        kv_world = tp_size // kv_replicas if kv_replicas > 1 else tp_size

        def find(name: str):
            candidates = (name, f"model.{name}")
            for candidate in candidates:
                for state_dict in state_dicts:
                    if state_dict.has(candidate):
                        return state_dict, candidate
            return None

        def tensor(name: str, required: bool = True) -> torch.Tensor | None:
            found = find(name)
            if found is None:
                if required:
                    raise KeyError(f"missing required DSpark weight: {name}")
                return None
            state_dict, key = found
            return state_dict.get_tensor(key)

        def shard(name: str, dim: int, kv: bool = False) -> torch.Tensor:
            loaded = tensor(name)
            if loaded is None:
                raise KeyError(name)
            rank = kv_rank if kv else tp_rank
            world = kv_world if kv else tp_size
            if world <= 1:
                return loaded
            chunk_size = loaded.size(dim) // world
            return loaded.narrow(dim, rank * chunk_size, chunk_size).contiguous()

        def copy_in(parameter_name: str, loaded: torch.Tensor) -> None:
            parameter = self.get_parameter(parameter_name)
            parameter.data.copy_(loaded.to(dtype=parameter.dtype, device=parameter.device))

        fc_weight = tensor("fc.weight")
        if fc_weight is None:
            raise KeyError("fc.weight")
        self.model.fc.load_weight(fc_weight)
        hidden_norm = tensor("hidden_norm.weight")
        if hidden_norm is None:
            raise KeyError("hidden_norm.weight")
        copy_in("model.hidden_norm.weight", hidden_norm)

        for layer_id in range(cfg.n_layers):
            prefix = f"layers.{layer_id}."
            target = f"model.layers.{layer_id}."
            for norm_name in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ):
                loaded = tensor(prefix + norm_name)
                if loaded is None:
                    raise KeyError(prefix + norm_name)
                copy_in(target + norm_name, loaded)

            q_weight = shard(prefix + "self_attn.q_proj.weight", dim=0)
            k_weight = shard(prefix + "self_attn.k_proj.weight", dim=0, kv=True)
            v_weight = shard(prefix + "self_attn.v_proj.weight", dim=0, kv=True)
            copy_in(target + "self_attn.qkv_proj.weight", torch.cat((q_weight, k_weight, v_weight)))
            copy_in(target + "self_attn.o_proj.weight", shard(prefix + "self_attn.o_proj.weight", dim=1))

            if cfg.attention_bias:
                q_bias = shard(prefix + "self_attn.q_proj.bias", dim=0)
                k_bias = shard(prefix + "self_attn.k_proj.bias", dim=0, kv=True)
                v_bias = shard(prefix + "self_attn.v_proj.bias", dim=0, kv=True)
                copy_in(target + "self_attn.qkv_proj.bias", torch.cat((q_bias, k_bias, v_bias)))
                output_bias = tensor(prefix + "self_attn.o_proj.bias")
                if output_bias is None:
                    raise KeyError(prefix + "self_attn.o_proj.bias")
                copy_in(target + "self_attn.o_proj.bias", output_bias)

            gate_weight = shard(prefix + "mlp.gate_proj.weight", dim=0)
            up_weight = shard(prefix + "mlp.up_proj.weight", dim=0)
            copy_in(target + "mlp.gate_up_proj.weight", torch.cat((gate_weight, up_weight)))
            copy_in(target + "mlp.down_proj.weight", shard(prefix + "mlp.down_proj.weight", dim=1))

            layer = self.model.layers[layer_id]
            layer.self_attn.o_proj.process_weights_after_loading()
            layer.mlp.down_proj.process_weights_after_loading()

        norm_weight = tensor("norm.weight")
        if norm_weight is None:
            raise KeyError("norm.weight")
        copy_in("model.norm.weight", norm_weight)
        markov_w1 = tensor("markov_head.markov_w1.weight")
        markov_w2 = tensor("markov_head.markov_w2.weight")
        if markov_w1 is None or markov_w2 is None:
            raise KeyError("DSpark Markov head weights are incomplete")
        copy_in("markov_head.markov_w1.weight", markov_w1)
        copy_in("markov_head.markov_w2.weight", markov_w2)

        if self.confidence_head is not None:
            confidence_weight = tensor("confidence_head.proj.weight")
            confidence_bias = tensor("confidence_head.proj.bias")
            if confidence_weight is None or confidence_bias is None:
                raise KeyError("DSpark confidence head weights are incomplete")
            copy_in("confidence_head.proj.weight", confidence_weight)
            copy_in("confidence_head.proj.bias", confidence_bias)

        self.model.build_context_kv_buffers()

    def write_context_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor | None, ...]],
        layer_synchronizer: object | None,
    ) -> torch.Tensor | None:
        return self.model.write_context_kv(
            target_hidden,
            positions,
            cache_slots,
            kv_caches,
            layer_synchronizer,
        )
