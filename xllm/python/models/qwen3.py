# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3 dense causal LM (Python model executor target).

Layer structure and residual flow mirror xLLM's C++ Qwen2/Qwen3 decoder layer:
fused add + RMSNorm carrying (hidden, residual) between layers, QK-norm before
RoPE (the Qwen3 differentiator), gated-SiLU MLP.

Tensor parallelism (WS1): when ``tp_size>1`` every projection/embedding holds a
per-partition weight and the parallel layers insert the same all-reduce /
all-gather the native C++ path uses. Weights are sharded on this rank by the C++
bridge (``PyCausalLM::load_model``) executing ``sharding_plan`` via the native
``get_sharded_tensor``; ``load_assembled_weights`` then copies them in.
At ``tp_size==1`` everything degrades to full-size weights with no collectives,
so the single-card parity path is byte-identical to the first version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .. import ops
from ..layers import (
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)
from .base import PyModelBase, ShardingPlan


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
    # Tensor parallelism (passed from C++ build_config_dict; default = single
    # card). Weights/heads are split per rank exactly as the native path.
    tp_size: int = 1
    tp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Qwen3Config":
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
            max_position_embeddings=int(
                pick("max_position_embeddings", default=40960)
            ),
            vocab_size=int(pick("vocab_size", default=151936)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=True)),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
        )

    def head_split(self) -> Tuple[int, int, int]:
        """Per-rank ``(num_heads, num_kv_heads, num_kv_head_replicas)``,
        replicating native ``qwen2_attention.cpp:47-65``: heads split evenly;
        for GQA with ``n_kv_heads < tp_size`` a single KV head is replicated
        across the ranks that share it.
        """
        tp = self.tp_size
        assert self.n_heads % tp == 0, (
            f"n_heads {self.n_heads} not divisible by tp_size {tp}"
        )
        num_heads = self.n_heads // tp
        if self.n_kv_heads >= tp:
            assert self.n_kv_heads % tp == 0, (
                f"n_kv_heads {self.n_kv_heads} not divisible by tp_size {tp}"
            )
            num_kv_heads = self.n_kv_heads // tp
            replicas = 1
        else:
            assert tp % self.n_kv_heads == 0, (
                f"tp_size {tp} not divisible by n_kv_heads {self.n_kv_heads}"
            )
            num_kv_heads = 1
            replicas = tp // self.n_kv_heads
        return num_heads, num_kv_heads, replicas


class Qwen3MLP(nn.Module):
    def __init__(
        self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.intermediate_size % tp == 0, (
            f"intermediate_size {cfg.intermediate_size} not divisible by "
            f"tp_size {tp}"
        )
        inter_per_rank = cfg.intermediate_size // tp
        # gate/up fused ColumnParallel: per-rank weight is
        # cat([gate_shard, up_shard], dim=0) -> [2*inter_per_rank, hidden].
        self.gate_up_proj = ColumnParallelLinear(
            cfg.hidden_size,
            2 * inter_per_rank,
            tp,
            dtype=dtype,
            device=device,
        )
        # down RowParallel: input already partitioned to inter_per_rank; output
        # is SUM all-reduced back to full hidden.
        self.down_proj = RowParallelLinear(
            inter_per_rank,
            cfg.hidden_size,
            tp,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        act = ops.silu_and_mul(gate_up)
        return self.down_proj(act)


class Qwen3Attention(nn.Module):
    def __init__(
        self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        num_heads, num_kv_heads, replicas = cfg.head_split()
        tp = cfg.tp_size
        self.num_heads = num_heads  # per-rank
        self.num_kv_heads = num_kv_heads  # per-rank
        self.num_kv_head_replicas = replicas
        self.head_dim = cfg.head_dim
        self.q_size = num_heads * self.head_dim  # per-rank
        self.kv_size = num_kv_heads * self.head_dim  # per-rank

        # qkv fused ColumnParallel: per-rank weight is
        # cat([q_shard, k_shard, v_shard], dim=0). No gather here — the o_proj
        # RowParallel all-reduce combines the per-rank attention outputs.
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_size,
            self.q_size + 2 * self.kv_size,
            tp,
            dtype=dtype,
            device=device,
        )
        # o_proj RowParallel: input is the per-rank head slice (q_size), output
        # is SUM all-reduced back to full hidden.
        self.o_proj = RowParallelLinear(
            self.q_size,
            cfg.hidden_size,
            tp,
            dtype=dtype,
            device=device,
        )
        # Qwen3 QK-norm: RMSNorm over head_dim, applied per head before RoPE.
        # Replicated on every rank (not sharded).
        self.q_norm = RMSNorm(
            self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.k_norm = RMSNorm(
            self.head_dim, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.rotary = RotaryEmbedding(
            self.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)

        # Fused per-head QK-RMSNorm + RoPE on the packed qkv — the same
        # xllm_ops.fused_qk_norm_rope the C++ qwen3 path uses, with per-rank
        # head counts. q/k are normed+roped in place; v is untouched.
        qkv = ops.fused_qk_norm_rope(
            qkv,
            num_heads_q=self.num_heads,
            num_heads_k=self.num_kv_heads,
            num_heads_v=self.num_kv_heads,
            head_dim=self.head_dim,
            eps=self.q_norm.eps,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin_cache=self.rotary.cos_sin_cache,
            position_ids=positions,
        )
        q = qkv[:, : self.q_size]
        k = qkv[:, self.q_size : self.q_size + self.kv_size]
        v = qkv[:, self.q_size + self.kv_size :]

        attn_out = ops.attention(q, k, v, layer_id)
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Qwen3Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.self_attn = Qwen3Attention(cfg, dtype, device)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )
        self.mlp = Qwen3MLP(cfg, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)

        hidden = self.self_attn(positions, hidden, self.layer_id)

        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class Qwen3Model(nn.Module):
    def __init__(
        self, cfg: Qwen3Config, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0, (
            f"hidden_size {cfg.hidden_size} not divisible by tp_size {tp}"
        )
        # Embedding sharded on the hidden dim (dim 1) + all-gather, matching
        # native WordEmbedding.
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(cfg, i, dtype, device)
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual: Optional[torch.Tensor] = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions)
        hidden, _ = self.norm(hidden, residual)
        return hidden


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
        assert self.cfg.vocab_size % tp == 0, (
            f"vocab_size {self.cfg.vocab_size} not divisible by tp_size {tp}"
        )
        self.model = Qwen3Model(self.cfg, dtype, device)
        # lm_head ColumnParallel over vocab (dim 0) + gather_output all-gather to
        # reconstruct the full [tokens, vocab] logits, matching native LmHead.
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        # Wrap the pure-GPU graph (self.model) in the shared graph runner. Base
        # owns forward / compute_logits / load_assembled_weights; only the
        # model-specific sharding_plan lives here.
        self._init_runner()

    # -- weight loading ---------------------------------------------------
    def sharding_plan(self) -> ShardingPlan:
        """Declare how each parameter is assembled from checkpoint tensors.

        Returns a list of ``(target_param_name, sources, cat_dim)`` where
        ``sources`` is a list of ``(candidate_names, shard_dim, is_kv)``:
          * ``candidate_names`` — checkpoint tensor names tried in order;
          * ``shard_dim`` — dim to shard on this rank, or ``< 0`` for replicated;
          * ``is_kv`` — shard with the GQA KV-replica ``(rank, world)`` coords.

        The C++ bridge (``PyCausalLM::load_model``) EXECUTES this plan via the
        native ``StateDict::get_sharded_tensor`` and feeds back per-rank tensors.
        Keeping the sharding RULES here (the model knows its own layout) while
        the EXECUTION reuses the validated native chunk means Python no longer
        re-implements sharding — there is a single chunk path, removing the
        two-implementation parity risk. ``cat_dim`` concatenates fused sources
        (QKV / gate-up) exactly as the native fused-weight loader does.
        """
        cfg = self.cfg
        plan = []
        # Embedding: shard the hidden dim (1) — matches HiddenParallelEmbedding.
        plan.append(
            (
                "model.embed_tokens.weight",
                [(["model.embed_tokens.weight", "embed_tokens.weight"], 1, False)],
                0,
            )
        )
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            # Norms are replicated (full) on every rank.
            plan.append(
                (
                    p + "input_layernorm.weight",
                    [([p + "input_layernorm.weight"], -1, False)],
                    0,
                )
            )
            plan.append(
                (
                    p + "post_attention_layernorm.weight",
                    [([p + "post_attention_layernorm.weight"], -1, False)],
                    0,
                )
            )
            # QKV fused on dim 0; K/V use the KV-replica coords (is_kv=True).
            plan.append(
                (
                    p + "self_attn.qkv_proj.weight",
                    [
                        ([p + "self_attn.q_proj.weight"], 0, False),
                        ([p + "self_attn.k_proj.weight"], 0, True),
                        ([p + "self_attn.v_proj.weight"], 0, True),
                    ],
                    0,
                )
            )
            # o_proj RowParallel: shard the input dim (1).
            plan.append(
                (
                    p + "self_attn.o_proj.weight",
                    [([p + "self_attn.o_proj.weight"], 1, False)],
                    0,
                )
            )
            # q/k norm over head_dim: replicated (full).
            plan.append(
                (
                    p + "self_attn.q_norm.weight",
                    [([p + "self_attn.q_norm.weight"], -1, False)],
                    0,
                )
            )
            plan.append(
                (
                    p + "self_attn.k_norm.weight",
                    [([p + "self_attn.k_norm.weight"], -1, False)],
                    0,
                )
            )
            # gate/up fused on dim 0.
            plan.append(
                (
                    p + "mlp.gate_up_proj.weight",
                    [
                        ([p + "mlp.gate_proj.weight"], 0, False),
                        ([p + "mlp.up_proj.weight"], 0, False),
                    ],
                    0,
                )
            )
            # down_proj RowParallel: shard the input dim (1).
            plan.append(
                (
                    p + "mlp.down_proj.weight",
                    [([p + "mlp.down_proj.weight"], 1, False)],
                    0,
                )
            )
        plan.append(
            (
                "model.norm.weight",
                [(["model.norm.weight", "norm.weight"], -1, False)],
                0,
            )
        )
        # lm_head: ColumnParallel over vocab (dim 0). Tied -> the source is the
        # embedding, but sharded on dim 0 here (vs the embedding's dim-1 shard).
        if cfg.tie_word_embeddings:
            lm_src = ["model.embed_tokens.weight", "embed_tokens.weight"]
        else:
            lm_src = ["lm_head.weight"]
        plan.append(("lm_head.weight", [(lm_src, 0, False)], 0))
        return plan

