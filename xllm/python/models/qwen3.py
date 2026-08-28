# Copyright 2025-2026 The xLLM Authors.
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

"""Qwen3 dense causal LM (Python model executor target).

Architecture: fused add + RMSNorm carrying (hidden, residual) between layers,
QK-norm before RoPE, gated-SiLU MLP. Tensor parallelism when tp_size>1.

Attention is delegated to the FlashInferBackend via the scoped ForwardContext.
The model does not import FlashInfer, own wrappers, or call plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    GatedMLP,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)
from xllm.python.model_executor.cp_utils import cp_merge_rows, cp_shard_positions, cp_shard_rows
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    record_layer_event,
)  # noqa: F401
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase
from xllm.python.models.weight_utils import WeightLoader, gqa_head_split, kv_replica_shard


@dataclass
class Qwen3Config:
    hidden_size: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 40960
    vocab_size: int = 151936
    tie_word_embeddings: bool = True
    sliding_window: int = 0
    attention_bias: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    layers_to_capture: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3Config:
        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        hidden = int(pick("hidden_size", default=1024))
        n_heads = int(pick("n_heads", "num_attention_heads", default=16))
        return cls(
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=28)),
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            head_dim=int(pick("head_dim", default=hidden // n_heads)),
            intermediate_size=int(pick("intermediate_size", default=3072)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-6)),
            rope_theta=float(pick("rope_theta", default=1e6)),
            max_position_embeddings=int(pick("max_position_embeddings", default=40960)),
            vocab_size=int(pick("vocab_size", default=151936)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=True)),
            sliding_window=int(pick("sliding_window", default=0)),
            attention_bias=bool(pick("attention_bias", default=False)),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            dp_size=int(pick("dp_size", default=1)),
            dp_rank=int(pick("dp_rank", default=0)),
            layers_to_capture=tuple(int(layer_id) for layer_id in pick("layers_to_capture", default=[])),
        )

    def head_split(self) -> tuple[int, int]:
        """Per-rank ``(num_heads, num_kv_heads)``."""
        return gqa_head_split(self.n_heads, self.n_kv_heads, self.tp_size)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        num_heads, num_kv_heads = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            bias=cfg.attention_bias,
            dtype=dtype,
            device=device,
        )
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.attn = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            scale=self.head_dim**-0.5,
            sliding_window=cfg.sliding_window,
            layer_id=layer_id,
            causal=causal,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)

        if mrope_section is not None and positions.dim() == 2:
            # mRoPE prefill: per-head Q/K RMSNorm (same math as the fused
            # kernel) then kernels.mrope, which does the time/height/width
            # section combination + rotation in one op.
            # cos_sin_cache here is the [max_pos, head_dim]=[cos_half|sin_half]
            # table; q/k stay 2D [N, num_heads*head_dim] as npu_mrope requires.
            num_tokens = qkv.size(0)
            q = torch.ops.xllm_ops.rms_norm(
                qkv[:, : self.q_size].reshape(num_tokens * self.num_heads, self.head_dim),
                self.q_norm.weight,
                self.q_norm.eps,
            ).view(num_tokens, self.q_size)
            k = torch.ops.xllm_ops.rms_norm(
                qkv[:, self.q_size : self.q_size + self.kv_size].reshape(num_tokens * self.num_kv_heads, self.head_dim),
                self.k_norm.weight,
                self.k_norm.eps,
            ).view(num_tokens, self.kv_size)
            v = qkv[:, self.q_size + self.kv_size :]
            q, k = kernels.mrope(
                positions,
                q,
                k,
                cos_sin_cache,
                self.head_dim,
                mrope_section=list(mrope_section),
                rotary_mode="half",
                cache_mode="interleave",
            )
        else:
            q, k, v = kernels.fused_qk_norm_rope(
                qkv,
                num_heads_q=self.num_heads,
                num_heads_k=self.num_kv_heads,
                num_heads_v=self.num_kv_heads,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                cos_sin_cache=cos_sin_cache,
                position_ids=positions,
                cos=cos,
                sin=sin,
            )

        attn_out = self.attn(q, k, v)
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Qwen3Attention(cfg, layer_id, dtype, device, causal=causal)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.mlp = GatedMLP(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.tp_size,
            dtype,
            device,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        mrope_section: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(positions, hidden, cos_sin_cache, cos, sin, mrope_section)

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        dtype: torch.dtype,
        device: torch.device,
        *,
        causal: bool = True,
        create_embedding: bool = True,
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.embed_tokens: HiddenParallelEmbedding | None = None
        if create_embedding:
            self.embed_tokens = HiddenParallelEmbedding(
                cfg.vocab_size,
                cfg.hidden_size // tp,
                tp,
                dtype=dtype,
                device=device,
            )
        self.rotary = RotaryEmbedding(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(cfg, i, dtype, device, causal=causal) for i in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.aux_hidden_capture = AuxHiddenCapture(cfg.layers_to_capture)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mrope_section: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.embed_tokens is None:
            raise RuntimeError("input embedding has not been initialized")
        hidden = self.embed_tokens(input_ids)
        # The fused QK-norm+RoPE kernel requires int64 position ids, but C++
        # passes them as int32. Cast once here instead of once per layer. In the
        # captured decode graph this single cast is recorded inside the graph
        # (its output lives in the graph memory pool), so replay re-casts the
        # updated static_positions correctly.
        positions = positions.to(torch.int64).contiguous()
        # Context-Parallel: shard the sequence across the CP group after embed
        # and merge back before the final norm (model-side CP semantics). Only
        # active on prefill with cp_size>1; cp_context is None otherwise.
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            if self.aux_hidden_capture.enabled:
                raise NotImplementedError("aux hidden capture does not support context parallelism")
            hidden = cp_shard_rows(hidden, cp_context)
            positions = cp_shard_positions(positions, cp_context).contiguous()
        residual: torch.Tensor | None = None
        aux_hidden_buffer = self.aux_hidden_capture.create_buffer(hidden)
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(
                hidden,
                residual,
                positions,
                self.rotary.cos_sin_cache,
                None,
                None,
                mrope_section,
            )
            self.aux_hidden_capture.capture_layer(i, hidden, residual, aux_hidden_buffer)
            record_layer_event(i)
        hidden, _ = self.norm(hidden, residual)
        if cp_context is not None:
            hidden = cp_merge_rows(hidden, cp_context)
        return self.aux_hidden_capture.finalize(hidden, aux_hidden_buffer)


def load_qwen3_attention(
    loader: WeightLoader,
    src: str,
    dst: str,
    *,
    kv_world: int,
    kv_rank: int,
    attention_bias: bool = False,
) -> None:
    """Load one GQA self-attention block: q/k norm, fused qkv, o_proj, optional bias."""
    loader.copy_in(dst + "self_attn.q_norm.weight", loader.load_tensor(src + "self_attn.q_norm.weight"))
    loader.copy_in(dst + "self_attn.k_norm.weight", loader.load_tensor(src + "self_attn.k_norm.weight"))

    q = loader.load_shard(src + "self_attn.q_proj.weight", 0)
    k = loader.load_shard(src + "self_attn.k_proj.weight", 0, world=kv_world, rank=kv_rank)
    v = loader.load_shard(src + "self_attn.v_proj.weight", 0, world=kv_world, rank=kv_rank)
    loader.copy_in(dst + "self_attn.qkv_proj.weight", torch.cat([q, k, v], dim=0))
    loader.copy_in(dst + "self_attn.o_proj.weight", loader.load_shard(src + "self_attn.o_proj.weight", 1))

    if attention_bias:
        qb = loader.load_shard(src + "self_attn.q_proj.bias", 0)
        kb = loader.load_shard(src + "self_attn.k_proj.bias", 0, world=kv_world, rank=kv_rank)
        vb = loader.load_shard(src + "self_attn.v_proj.bias", 0, world=kv_world, rank=kv_rank)
        loader.copy_in(dst + "self_attn.qkv_proj.bias", torch.cat([qb, kb, vb], dim=0))
        # o_proj bias is replicated and added after the all-reduce, so every
        # rank loads the full (unsharded) bias.
        loader.copy_in(dst + "self_attn.o_proj.bias", loader.load_tensor(src + "self_attn.o_proj.bias"))


def load_qwen3_backbone(
    loader: WeightLoader,
    layers: nn.ModuleList,
    *,
    kv_world: int,
    kv_rank: int,
    attention_bias: bool = False,
    dst_prefix: str = "model.",
) -> None:
    """Load every Qwen3-dense decoder layer (norms, attention, MLP), then finalize.

    ``dst_prefix``: ``"model."`` for a top-level CausalLM, ``""`` when the loader
    targets the decoder module directly.
    """
    for i, layer in enumerate(layers):
        src = f"layers.{i}."
        dst = f"{dst_prefix}layers.{i}."
        loader.copy_in(dst + "input_layernorm.weight", loader.load_tensor(src + "input_layernorm.weight"))
        loader.copy_in(
            dst + "post_attention_layernorm.weight",
            loader.load_tensor(src + "post_attention_layernorm.weight"),
        )
        load_qwen3_attention(loader, src, dst, kv_world=kv_world, kv_rank=kv_rank, attention_bias=attention_bias)
        loader.load_gated_mlp(dst + "mlp.", src + "mlp.")

        layer.self_attn.o_proj.process_weights_after_loading()
        layer.mlp.down_proj.process_weights_after_loading()


class Qwen3ForCausalLM(PyModelBase):
    """Top-level entry the C++ PyCausalLM drives."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Qwen3Config.from_dict(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        tp = self.cfg.tp_size
        dp = self.cfg.dp_size
        if tp * dp != int(config.get("world_size", tp * dp)):
            raise ValueError("world_size must equal tp_size * dp_size")
        if not 0 <= self.cfg.dp_rank < dp:
            raise ValueError("dp_rank must be in [0, dp_size)")
        assert self.cfg.vocab_size % tp == 0
        self.model = Qwen3Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    # -- weight loading ---------------------------------------------------
    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg

        kv_world, kv_rank = kv_replica_shard(cfg.n_kv_heads, tp_rank, tp_size)
        loader = WeightLoader(self, state_dicts, tp_size, tp_rank, src_prefixes=("model.", ""))

        loader.copy_shard("model.embed_tokens.weight", dim=1)

        load_qwen3_backbone(
            loader,
            self.model.layers,
            kv_world=kv_world,
            kv_rank=kv_rank,
            attention_bias=cfg.attention_bias,
        )

        loader.copy_replicated("model.norm.weight")

        lm_name = "embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
        loader.copy_in("lm_head.weight", loader.load_shard(lm_name, dim=0))
