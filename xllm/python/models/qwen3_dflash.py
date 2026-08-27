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

"""Qwen3-style DFlash draft model for the Python NPU executor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.layers import RMSNorm
from xllm.python.model_executor.forward_context import LayerSynchronizer
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen3 import Qwen3Config, Qwen3Model


@dataclass
class DFlashQwen3Config(Qwen3Config):
    draft_vocab_size: int = 0
    world_size: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> DFlashQwen3Config:
        normalized_config = dict(d)
        rope_parameters = d.get("rope_parameters") or {}
        if normalized_config.get("rope_theta") is None:
            normalized_config["rope_theta"] = rope_parameters.get("rope_theta")

        base = Qwen3Config.from_dict(normalized_config)
        draft_vocab_size = int(d.get("draft_vocab_size") or base.vocab_size)
        return cls(
            **base.__dict__,
            draft_vocab_size=draft_vocab_size,
            world_size=int(d.get("world_size", base.tp_size * base.dp_size)),
        )

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("invalid Qwen3-style block-diffusion dimensions")
        if min(self.tp_size, self.dp_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if self.hidden_size % self.tp_size:
            raise ValueError("hidden_size must be divisible by tp_size")
        if self.vocab_size % self.tp_size:
            raise ValueError("vocab_size must be divisible by tp_size")
        if self.draft_vocab_size != self.vocab_size:
            raise ValueError("reduced-vocabulary block-diffusion drafts are not supported")
        if self.layers_to_capture:
            raise ValueError("DFlash draft model does not support layers_to_capture")
        self.head_split()


class DFlashContextProjection(nn.Module):
    """Column-parallel projection with checkpoint-defined input width."""

    def __init__(
        self,
        out_features: int,
        tp_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.tp_size = tp_size
        self.weight = nn.Parameter(torch.empty(out_features // tp_size, 0, dtype=dtype, device=device))

    def load_weight(self, weight: torch.Tensor, tp_rank: int) -> None:
        if weight.dim() != 2 or weight.size(0) != self.out_features:
            raise ValueError("DFlash fc.weight has an invalid shape")
        local_out_features = self.out_features // self.tp_size
        weight = weight.narrow(
            0,
            tp_rank * local_out_features,
            local_out_features,
        )
        self.weight = nn.Parameter(weight.to(dtype=self.weight.dtype, device=self.weight.device).contiguous())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.weight.size(1) == 0:
            raise RuntimeError("block-diffusion context projection weight is not loaded")
        if hidden.size(-1) != self.weight.size(1):
            raise ValueError("target auxiliary hidden size does not match the draft fc.weight")
        output = F.linear(hidden, self.weight)
        if self.tp_size > 1:
            output = distributed.all_gather(
                output,
                dim=-1,
                world_size=self.tp_size,
            )
        return output


class DFlashQwen3Model(Qwen3Model):
    def __init__(self, cfg: DFlashQwen3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(
            cfg,
            dtype,
            device,
            causal=False,
            create_embedding=False,
        )
        self.cfg = cfg
        self.fc = DFlashContextProjection(
            cfg.hidden_size,
            cfg.tp_size,
            dtype,
            device,
        )
        self.hidden_norm = RMSNorm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.register_buffer("_fused_kv_weight", None, persistent=False)
        self.register_buffer("_fused_kv_bias", None, persistent=False)
        self.register_buffer("_k_norm_weights", None, persistent=False)

    def _build_context_kv_buffers(self) -> None:
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
        self._k_norm_weights = torch.stack(k_norm_weights, dim=0).detach().float().view(self.cfg.n_layers, 1, 1, -1)

    def _normalize_context_keys(self, keys: torch.Tensor) -> torch.Tensor:
        if self._k_norm_weights is None:
            raise RuntimeError("draft K-norm buffers are not initialized")
        keys_fp32 = keys.float()
        variance = keys_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = keys_fp32 * torch.rsqrt(variance + self.cfg.rms_norm_eps)
        return (normalized * self._k_norm_weights).to(keys.dtype)

    def _apply_context_rope(
        self,
        keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
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
        layer_synchronizer: LayerSynchronizer | None,
    ) -> torch.Tensor | None:
        if len(kv_caches) != self.cfg.n_layers:
            raise ValueError("draft KV cache layer count mismatch")
        if cache_slots.numel() != target_hidden.size(0):
            raise ValueError("draft context cache slot count mismatch")
        if self._fused_kv_weight is None or self._k_norm_weights is None:
            raise RuntimeError("draft context KV buffers are not initialized")

        projected_hidden = self.hidden_norm(self.fc(target_hidden))
        all_kv = F.linear(
            projected_hidden,
            self._fused_kv_weight,
            self._fused_kv_bias,
        )
        num_context = projected_hidden.size(0)
        num_kv_heads = self.layers[0].self_attn.num_kv_heads
        all_kv = all_kv.view(
            num_context,
            self.cfg.n_layers,
            2,
            num_kv_heads,
            self.cfg.head_dim,
        ).permute(2, 1, 0, 3, 4)
        all_keys = self._apply_context_rope(
            self._normalize_context_keys(all_kv[0]),
            positions,
        )
        all_values = all_kv[1].contiguous()

        for layer_id, cache in enumerate(kv_caches):
            key_cache, value_cache = cache[0], cache[1]
            if key_cache is None or value_cache is None:
                raise RuntimeError(f"draft KV cache is missing for layer {layer_id}")
            kernels.reshape_paged_cache(
                cache_slots,
                all_keys[layer_id].contiguous(),
                all_values[layer_id],
                key_cache,
                value_cache,
            )
            if layer_synchronizer is not None:
                if layer_synchronizer.record_event(layer_id) is False:
                    return None
        return projected_hidden

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

        def load_tensor(name: str) -> torch.Tensor:
            found = find(name)
            if found is None:
                raise KeyError(f"checkpoint tensor not found: {name}")
            state_dict, key = found
            return state_dict.get_tensor(key)

        def shard(name: str, dim: int, kv: bool = False) -> torch.Tensor:
            tensor = load_tensor(name)
            rank = kv_rank if kv else tp_rank
            world = kv_world if kv else tp_size
            if world <= 1:
                return tensor
            chunk_size = tensor.size(dim) // world
            return tensor.narrow(dim, rank * chunk_size, chunk_size).contiguous()

        def copy_in(param_name: str, tensor: torch.Tensor) -> None:
            param = self.get_parameter(param_name)
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))

        self.fc.load_weight(load_tensor("fc.weight"), tp_rank)
        copy_in("hidden_norm.weight", load_tensor("hidden_norm.weight"))

        for layer_id in range(cfg.n_layers):
            prefix = f"layers.{layer_id}."
            target = f"layers.{layer_id}."
            for norm_name in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ):
                copy_in(target + norm_name, load_tensor(prefix + norm_name))

            q_weight = shard(prefix + "self_attn.q_proj.weight", dim=0)
            k_weight = shard(prefix + "self_attn.k_proj.weight", dim=0, kv=True)
            v_weight = shard(prefix + "self_attn.v_proj.weight", dim=0, kv=True)
            copy_in(
                target + "self_attn.qkv_proj.weight",
                torch.cat((q_weight, k_weight, v_weight)),
            )
            copy_in(
                target + "self_attn.o_proj.weight",
                shard(prefix + "self_attn.o_proj.weight", dim=1),
            )

            if cfg.attention_bias:
                q_bias = shard(prefix + "self_attn.q_proj.bias", dim=0)
                k_bias = shard(prefix + "self_attn.k_proj.bias", dim=0, kv=True)
                v_bias = shard(prefix + "self_attn.v_proj.bias", dim=0, kv=True)
                copy_in(
                    target + "self_attn.qkv_proj.bias",
                    torch.cat((q_bias, k_bias, v_bias)),
                )
                copy_in(
                    target + "self_attn.o_proj.bias",
                    load_tensor(prefix + "self_attn.o_proj.bias"),
                )

            gate_weight = shard(prefix + "mlp.gate_proj.weight", dim=0)
            up_weight = shard(prefix + "mlp.up_proj.weight", dim=0)
            copy_in(
                target + "mlp.gate_up_proj.weight",
                torch.cat((gate_weight, up_weight)),
            )
            copy_in(
                target + "mlp.down_proj.weight",
                shard(prefix + "mlp.down_proj.weight", dim=1),
            )

            layer = self.layers[layer_id]
            layer.self_attn.o_proj.process_weights_after_loading()
            layer.mlp.down_proj.process_weights_after_loading()

        copy_in("norm.weight", load_tensor("norm.weight"))
        self._build_context_kv_buffers()


class DFlashQwen3ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DFlashQwen3Config.from_dict(config)
        self.cfg.validate()
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        self.device = torch.device(config.get("device", "npu"))
        self.model = DFlashQwen3Model(self.cfg, self.dtype, self.device)
        self.lm_head: nn.Module | None = None

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        self.model.load_weights(state_dicts, tp_rank, tp_size)

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
