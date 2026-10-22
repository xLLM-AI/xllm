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

"""OLMo2 with full-projection Q/K normalization and post-norm residuals.

The initial execution contract is the unquantized, untied MHA OLMo2 checkpoint,
with TP=1/2 and DP=CP=PP=1. Attention/cache execution belongs to the scoped
backend; this module owns the model computation and checkpoint transforms.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python import distributed
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RotaryEmbedding,
    RowParallelLinear,
)
from xllm.python.layers.rotary_embedding import apply_rotary_half, gather_half_rope_cos_sin
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event
from xllm.python.model_loader import (
    ParallelLoadContext,
    ScopedWeightLoader,
    load_causal_lm_weights,
    load_gqa_fused_attention,
)
from xllm.python.model_loader.scoped_weight_loader import StateDictLike
from xllm.python.models.base import PyModelBase


@dataclass(frozen=True)
class Olmo2Config:
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Olmo2Config:
        def _pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if config.get(name) is not None:
                    return config[name]
            if default is None:
                raise ValueError(f"OLMo2 requires config field: {names[0]}")
            return default

        hidden = int(_pick("hidden_size"))
        heads = int(_pick("n_heads", "num_attention_heads"))
        if hidden <= 0 or heads <= 0 or hidden % heads:
            raise ValueError("OLMo2 hidden_size must be positive and divisible by n_heads")
        rope_parameters = config.get("rope_parameters")
        if rope_parameters is None:
            rope_parameters = {}
        if not isinstance(rope_parameters, dict):
            raise ValueError("OLMo2 rope_parameters must be an object")
        cfg = cls(
            hidden_size=hidden,
            n_layers=int(_pick("n_layers", "num_hidden_layers")),
            n_heads=heads,
            n_kv_heads=int(_pick("n_kv_heads", "num_key_value_heads", default=heads)),
            head_dim=int(_pick("head_dim", default=hidden // heads)),
            intermediate_size=int(_pick("intermediate_size")),
            vocab_size=int(_pick("vocab_size")),
            max_position_embeddings=int(_pick("max_position_embeddings")),
            rms_norm_eps=float(_pick("rms_norm_eps", default=1e-5)),
            rope_theta=float(rope_parameters.get("rope_theta", _pick("rope_theta", default=10000.0))),
            tp_size=int(_pick("tp_size", default=1)),
            tp_rank=int(_pick("tp_rank", default=0)),
        )
        if min(cfg.n_layers, cfg.intermediate_size, cfg.vocab_size, cfg.max_position_embeddings) <= 0:
            raise ValueError("OLMo2 model dimensions must be positive")
        if cfg.head_dim != hidden // heads or cfg.head_dim % 2:
            raise ValueError("OLMo2 requires an even head_dim equal to hidden_size / n_heads")
        if cfg.n_kv_heads != heads:
            raise ValueError("OLMo2 currently requires MHA without replicated KV heads")
        if cfg.tp_size not in (1, 2) or not 0 <= cfg.tp_rank < cfg.tp_size:
            raise ValueError("OLMo2 requires TP=1/2 and a rank in [0, tp_size)")
        if any(size % cfg.tp_size for size in (heads, hidden, cfg.intermediate_size, cfg.vocab_size)):
            raise ValueError("OLMo2 heads, hidden, intermediate and vocabulary must divide evenly across TP")
        if int(config.get("world_size", cfg.tp_size)) != cfg.tp_size or any(
            int(config.get(name, 1)) != 1 for name in ("dp_size", "cp_size", "pp_size", "layerwise_split_size")
        ):
            raise ValueError("OLMo2 currently requires DP=CP=PP=1")
        if any(not math.isfinite(value) or value <= 0 for value in (cfg.rms_norm_eps, cfg.rope_theta)):
            raise ValueError("OLMo2 norm epsilon and RoPE theta must be positive and finite")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("OLMo2 currently requires SiLU")
        if config.get("attention_bias", False) or config.get("qkv_bias", False):
            raise ValueError("OLMo2 currently requires bias-free attention")
        if config.get("tie_word_embeddings", False):
            raise ValueError("OLMo2 currently requires untied embedding and LM head weights")
        if config.get("rope_scaling") not in (None, -1) or config.get("rope_scaling_rope_type") not in (
            None,
            "",
            "default",
        ):
            raise ValueError("OLMo2 currently requires unscaled RoPE")
        if rope_parameters:
            # Transformers prefers rope_type, falling back to the legacy type.
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            if rope_type != "default" or rope_parameters.get("factor") is not None:
                raise ValueError("OLMo2 currently requires unscaled RoPE")
        if config.get("layers_to_capture") or config.get("rope_scaling_mrope_section") or config.get("mrope_section"):
            raise ValueError("OLMo2 auxiliary hidden capture and mRoPE are not supported")
        if config.get("quant_method") not in (None, ""):
            raise ValueError("OLMo2 currently requires unquantized weights")
        return cfg


def _bf16_linear_fp32(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Preserve FP32 activations using two BF16 products with FP32 outputs.

    Weights remain in the checkpoint dtype. Rounding an FP32 residual to one
    BF16 input discards small updates; the second product recovers that input
    rounding error before post-attention/post-FF normalization.
    """
    low = value.to(torch.bfloat16)
    output = torch.mm(low, weight, out_dtype=torch.float32)
    if value.dtype == torch.float32:
        correction = (value - low.float()).to(torch.bfloat16)
        # The first product owns its storage; avoid a third full-size FP32
        # output allocation when adding the correction during long prefill.
        output.add_(torch.mm(correction, weight, out_dtype=torch.float32))
    return output


class _Olmo2ColumnParallelLinear(ColumnParallelLinear):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != torch.bfloat16 or self.weight.device.type != "cuda":
            return super().forward(value)
        output = _bf16_linear_fp32(value, self.weight.t())
        if self.bias is not None:
            output = output + self.bias.float()
        if self.gather_output and self.tp_size > 1:
            output = distributed.tp_all_gather(output, dim=-1, world_size=self.tp_size)
        return output


class _Olmo2RowParallelLinear(RowParallelLinear):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != torch.bfloat16 or self.weight.device.type != "cuda":
            return super().forward(value)
        weight = self.weight if self._weight_is_transposed else self.weight.t()
        output = _bf16_linear_fp32(value, weight)
        if self.tp_size > 1 and self.reduce_results:
            distributed.tp_all_reduce(output)
        if self.bias is not None:
            output = output + self.bias.float()
        return output


class _Olmo2MLP(GatedMLP):
    def __init__(self, cfg: Olmo2Config, dtype: torch.dtype, device: torch.device) -> None:
        nn.Module.__init__(self)
        local_intermediate = cfg.intermediate_size // cfg.tp_size
        self.gate_up_proj = _Olmo2ColumnParallelLinear(
            cfg.hidden_size, 2 * local_intermediate, cfg.tp_size, dtype=dtype, device=device
        )
        self.down_proj = _Olmo2RowParallelLinear(
            local_intermediate, cfg.hidden_size, cfg.tp_size, dtype=dtype, device=device
        )


class Olmo2RMSNorm(nn.Module):
    """Keep statistics and gamma multiplication in FP32 until the final cast."""

    def __init__(self, size: int, eps: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(size, dtype=dtype, device=device))

    def forward(self, value: torch.Tensor, mean_square: torch.Tensor | None = None) -> torch.Tensor:
        value_fp32 = value.float()
        if mean_square is None:
            mean_square = value_fp32.square().mean(dim=-1, keepdim=True)
        normalized = value_fp32 * torch.rsqrt(mean_square + self.eps)
        return (normalized * self.weight.float()).to(value.dtype)


class Olmo2Attention(nn.Module):
    def __init__(self, cfg: Olmo2Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.n_heads // cfg.tp_size
        self.head_dim = cfg.head_dim
        self.local_size = self.num_heads * self.head_dim
        self.qkv_proj = _Olmo2ColumnParallelLinear(
            cfg.hidden_size, 3 * self.local_size, cfg.tp_size, dtype=dtype, device=device
        )
        self.o_proj = _Olmo2RowParallelLinear(self.local_size, cfg.hidden_size, cfg.tp_size, dtype=dtype, device=device)
        self.q_norm = Olmo2RMSNorm(self.local_size, cfg.rms_norm_eps, dtype, device)
        self.k_norm = Olmo2RMSNorm(self.local_size, cfg.rms_norm_eps, dtype, device)
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_dim=cfg.head_dim,
            scale=cfg.head_dim**-0.5,
            sliding_window=0,
            layer_id=layer_id,
            # Match the prefill tensor-core accumulation path for BF16 decode;
            # small rounding changes otherwise compound through post-norms.
            use_tensor_core_decode=True if dtype == torch.bfloat16 else None,
        )

    def _normalize_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Norm spans the full projection, before splitting into heads. Pack two
        # independent statistics in one collective, then divide by global size.
        stats = torch.cat(
            (q.float().square().sum(dim=-1, keepdim=True), k.float().square().sum(dim=-1, keepdim=True)), dim=-1
        )
        if self.cfg.tp_size > 1:
            distributed.tp_all_reduce(stats)
        global_size = self.cfg.n_heads * self.head_dim
        return self.q_norm(q, stats[:, :1] / global_size), self.k_norm(k, stats[:, 1:] / global_size)

    def forward(self, hidden: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(hidden).split(self.local_size, dim=-1)
        q, k = self._normalize_qk(q, k)
        # Match the reference's FP32 rotation, with no BF16 table quantization.
        shape = (-1, self.num_heads, self.head_dim)
        q = apply_rotary_half(q.reshape(shape).float(), cos_half, sin_half).to(q.dtype).flatten(1)
        k = apply_rotary_half(k.reshape(shape).float(), cos_half, sin_half).to(k.dtype).flatten(1)
        # Keep backend Q/K/V and paged caches in the checkpoint compute dtype;
        # the higher-precision projections/norms do not change the cache format.
        dtype = self.qkv_proj.weight.dtype
        return self.o_proj(self.attn(q.to(dtype), k.to(dtype), v.to(dtype).contiguous()))

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        load_gqa_fused_attention(self, weights, context, self.cfg.n_kv_heads, qk_norm=False)
        # Unlike per-head Qwen norms, OLMo2 gamma is itself TP-sharded.
        for name, norm in (("q_norm.weight", self.q_norm), ("k_norm.weight", self.k_norm)):
            weights.load_tensor(norm.weight, name, dim=0, rank=context.tp_rank, world_size=context.tp_size)
        self.o_proj.process_weights_after_loading()


class Olmo2DecoderLayer(nn.Module):
    def __init__(self, cfg: Olmo2Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.self_attn = Olmo2Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.post_feedforward_layernorm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.mlp = _Olmo2MLP(cfg, dtype, device)
        self._preserve_bf16_input = dtype == torch.bfloat16 and device.type == "cuda"

    def forward(self, hidden: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
        # Row-parallel projections have already reduced their partial outputs.
        # Preserve small post-norm updates across layers in the FP32 residual
        # stream. BF16 projections preserve this input via compensated products;
        # FP16 retains the established projection path and attention dtype.
        dtype = self.self_attn.qkv_proj.weight.dtype
        attention_input = hidden if self._preserve_bf16_input else hidden.to(dtype)
        attention = self.self_attn(attention_input, cos_half, sin_half)
        hidden = hidden.float() + self.post_attention_layernorm(attention)
        mlp_input = hidden if self._preserve_bf16_input else hidden.to(dtype)
        return hidden + self.post_feedforward_layernorm(self.mlp(mlp_input))

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None:
        self.self_attn.load_weights(weights.with_prefix("self_attn."), context)
        weights.load_tensor(self.post_attention_layernorm.weight, "post_attention_layernorm.weight")
        weights.load_tensor(self.post_feedforward_layernorm.weight, "post_feedforward_layernorm.weight")
        self.mlp.load_weights(weights.with_prefix("mlp."), context)
        self.mlp.process_weights_after_loading()


class Olmo2Model(nn.Module):
    def __init__(self, cfg: Olmo2Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // cfg.tp_size, cfg.tp_size, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta, dtype=torch.float32, device=device
        )
        self.layers = nn.ModuleList(Olmo2DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers))
        self.norm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self._preserve_bf16_input = dtype == torch.bfloat16 and device.type == "cuda"

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if get_forward_context().cp_context is not None:
            raise ValueError("OLMo2 context parallelism is not supported")
        hidden = self.embed_tokens(input_ids)
        cos_half, sin_half = gather_half_rope_cos_sin(self.rotary.cos_sin_cache, positions.to(torch.long))
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, cos_half, sin_half)
            record_layer_event(i)
        normalized = self.norm(hidden)
        return normalized if self._preserve_bf16_input else normalized.to(self.embed_tokens.weight.dtype)


def _checkpoint_shapes(cfg: Olmo2Config) -> dict[str, tuple[int, ...]]:
    hidden, intermediate = cfg.hidden_size, cfg.intermediate_size
    shapes = {
        "model.embed_tokens.weight": (cfg.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (cfg.vocab_size, hidden),
    }
    layer_shapes = {
        "self_attn.q_proj.weight": (hidden, hidden),
        "self_attn.k_proj.weight": (hidden, hidden),
        "self_attn.v_proj.weight": (hidden, hidden),
        "self_attn.o_proj.weight": (hidden, hidden),
        "self_attn.q_norm.weight": (hidden,),
        "self_attn.k_norm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "post_feedforward_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    for layer in range(cfg.n_layers):
        shapes.update({f"model.layers.{layer}.{name}": shape for name, shape in layer_shapes.items()})
    return shapes


class _Olmo2Checkpoint:
    """Validate the complete checkpoint and account for every consumed tensor."""

    def __init__(self, states: Sequence[StateDictLike], cfg: Olmo2Config) -> None:
        self._states: dict[str, StateDictLike] = {}
        self._shapes = _checkpoint_shapes(cfg)
        self._consumed: set[str] = set()
        for state in states:
            keys = getattr(state, "keys", None)
            if not callable(keys):
                raise TypeError("OLMo2 checkpoint shards must expose keys()")
            for name in keys():
                if name in self._states:
                    raise ValueError(f"duplicate OLMo2 checkpoint tensor: {name}")
                self._states[name] = state
        actual, expected = set(self._states), set(self._shapes)
        if actual != expected:
            raise ValueError(
                f"OLMo2 checkpoint keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    def keys(self) -> list[str]:
        return list(self._states)

    def has(self, name: str) -> bool:
        return name in self._states

    def get_tensor(self, name: str) -> torch.Tensor:
        value = self._states[name].get_tensor(name)
        if tuple(value.shape) != self._shapes[name]:
            raise ValueError(
                f"OLMo2 checkpoint tensor {name} has shape {tuple(value.shape)}, expected {self._shapes[name]}"
            )
        self._consumed.add(name)
        return value

    def verify_consumed(self) -> tuple[str, ...]:
        missing = set(self._states) - self._consumed
        if missing:
            raise ValueError(f"unconsumed OLMo2 checkpoint tensors: {sorted(missing)}")
        return tuple(sorted(self._consumed))


class Olmo2ForCausalLM(PyModelBase):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = Olmo2Config.from_dict(config)
        self.dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("OLMo2 requires float32, float16 or bfloat16")
        self.device = torch.device(config.get("device", "cuda"))
        self.model = Olmo2Model(self.cfg, self.dtype, self.device)
        self.lm_head = _Olmo2ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )
        self.loaded_checkpoint_keys: tuple[str, ...] = ()

    def load_weights(self, state_dicts: list[StateDictLike], tp_rank: int, tp_size: int) -> None:
        if (tp_rank, tp_size) != (self.cfg.tp_rank, self.cfg.tp_size):
            raise ValueError("OLMo2 weight-loading TP topology differs from model construction")
        checkpoint = _Olmo2Checkpoint(state_dicts, self.cfg)
        weights = ScopedWeightLoader([checkpoint], src_prefixes=("model.", ""))
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            weights,
            ParallelLoadContext(tp_rank, tp_size),
            tie_word_embeddings=False,
        )
        self.loaded_checkpoint_keys = checkpoint.verify_consumed()
        logger.info(
            f"Loaded {len(self.loaded_checkpoint_keys)} OLMo2 checkpoint tensors on TP rank {tp_rank}/{tp_size}; "
            f"config={self.cfg}, dtype={self.dtype}, device={self.device}"
        )
