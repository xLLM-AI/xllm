# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""GLM-5.2 (model_type=glm_moe_dsa) causal LM, adapted from DeepSeek-V3.2.

Shared machinery is imported from ``deepseek_v32``: W8A8 linears, dense MLP,
MoE, YaRN RoPE, and the MLA RoPE helpers. Only the
GLM-5.2 structural deltas live here:

  * cross-layer top-k sharing -- ``indexer_types`` marks full/shared layers;
    shared layers skip the indexer and reuse the previous full layer's top-k.
  * indexer ``wq_b`` is W8A8 (not bf16 ``nn.Linear``).
  * indexer RoPE is configurable (``indexer_rope_interleave``); DSV3.2's
    indexer uses half-rotate only.
  * per-layer MLP type comes from ``mlp_layer_types`` (not a single
    ``first_k_dense_replace`` threshold).
  * the YaRN ``cos_sin_cache`` is built once at the model level and threaded
    through ``forward`` (no per-layer rotary module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from xllm.python import distributed, kernels
from xllm.python.attention.backend import MlaIndexContext

# The AICPU tiling of ``aclnnQuantLightningIndexer`` requires
# ``num_heads_q / num_heads_k == 64``. GLM-5.2 uses ``index_n_heads=32`` and
# ``num_heads_k=1``, so we pad Q / q_scale / weights along the head axis to
# 64 heads (padded weights are zero, keeping ``score = sum_h w[h]*q[h]*k``
# mathematically identical). Pure-Python workaround limited to this file.
from xllm.python.layers import (
    Attention,
    ColumnParallelLinear,
    HiddenParallelEmbedding,
    RMSNorm,
)
from xllm.python.model_executor.cp_utils import (
    cp_gather_kv,
    cp_merge_rows,
    cp_shard_positions,
    cp_shard_rows,
)
from xllm.python.model_executor.forward_context import get_forward_context, record_layer_event
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import (
    DeepseekV3MLP,
    DeepseekV3MoE,
    W8A8StaticLinear,
    _apply_half_rope,
    _create_hadamard_matrix,
    _gather_interleave_cos_sin,
    _interleave_rope_with,
    _swiglu_with_clamp,
    _tp_rank_from_device,
    _yarn_get_mscale,
)
from xllm.python.models.deepseek_v32 import (
    DeepseekYarnRotaryEmbedding as Glm52YarnRotaryEmbedding,
)
from xllm.python.models.weight_utils import W8A8WeightLoader, effective_moe_tp, mla_head_split

# aclnnQuantLightningIndexer tiling hard-requires n_heads_q / n_heads_k == 64
# (G_SIZE_LIMIT in third_party/xllm_ops/.../quant_lightning_indexer_tiling.h). The
# xllm_ops kernel fixes matmul M=256 tile with head_dim=128, kv_head=1 for DSV3.2/DSV4
# shapes; other gSize values are neither exposed nor tested. GLM-5.2 is 32/1 and pads Q
# up to 64 heads at the caller (see Glm52Indexer._pad_q_heads_to_kernel_gsize).
_QLI_KERNEL_GSIZE = 64


@dataclass
class Glm52Config:
    """GLM-5.2 (glm_moe_dsa) architecture parameters."""

    model_type: str = "glm_moe_dsa"
    hidden_size: int = 6144
    n_layers: int = 78
    n_heads: int = 64
    head_dim: int = 0
    intermediate_size: int = 12288
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1.0e6
    max_position_embeddings: int = 202752
    original_max_position_embeddings: int = 202752
    rope_scaling_factor: float = 1.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    tie_word_embeddings: bool = False
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    first_k_dense_replace: int = 3
    moe_layer_freq: int = 1
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 2048
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    layerwise_split_size: int = 1
    layerwise_split_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    world_size: int = 1
    indexer_types: list | None = None
    mlp_layer_types: list | None = None
    index_skip_topk_offset: int = 2
    index_topk_freq: int = 1
    index_topk_pattern: list | None = None
    indexer_rope_interleave: bool = True
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False
    layers_to_capture: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, d: dict) -> Glm52Config:
        def pick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        rs_raw = d.get("rope_scaling")
        rs = rs_raw if isinstance(rs_raw, dict) else {}
        if not rs:
            rp = d.get("rope_parameters")
            if isinstance(rp, dict):
                rs = rp

        def rpick(*keys: str, default: Any = None) -> Any:
            for k in keys:
                if isinstance(rs, dict) and k in rs and rs[k] is not None:
                    return rs[k]
                fk = f"rope_scaling_{k}"
                if fk in d and d[fk] is not None:
                    return d[fk]
                if k in d and d[k] is not None:
                    return d[k]
            return default

        def rpick_nz(*keys: str, default: Any) -> Any:
            v = rpick(*keys, default=None)
            if v is None or v == 0 or v == -1 or v == "":
                return default
            return v

        hidden = int(pick("hidden_size", default=6144))
        n_heads = int(pick("n_heads", "num_attention_heads", default=64))
        max_pe = int(pick("max_position_embeddings", default=202752))
        tp_size = int(pick("tp_size", default=1))
        dp_size = int(pick("dp_size", default=1))
        cp_size = int(pick("cp_size", default=1))
        world_size = int(pick("world_size", default=tp_size * dp_size * cp_size))
        rope_scaling_factor = float(rpick_nz("factor", "rope_scaling_factor", default=1.0))
        original_max = int(rpick_nz("original_max_position_embeddings", default=max_pe))

        cfg = cls(
            model_type=str(pick("model_type", default="glm_moe_dsa")),
            hidden_size=hidden,
            n_layers=int(pick("n_layers", "num_hidden_layers", default=78)),
            n_heads=n_heads,
            head_dim=int(pick("head_dim", default=hidden // n_heads if n_heads else 0)),
            intermediate_size=int(pick("intermediate_size", default=12288)),
            vocab_size=int(pick("vocab_size", default=154880)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            rope_theta=float(pick("rope_theta", default=1.0e6)),
            max_position_embeddings=max_pe,
            original_max_position_embeddings=original_max,
            rope_scaling_factor=rope_scaling_factor,
            rope_beta_fast=int(rpick_nz("beta_fast", default=32)),
            rope_beta_slow=int(rpick_nz("beta_slow", default=1)),
            rope_mscale=float(rpick_nz("mscale", default=1.0)),
            rope_mscale_all_dim=float(rpick_nz("mscale_all_dim", default=1.0)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            q_lora_rank=int(pick("q_lora_rank", default=2048)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            index_n_heads=int(pick("index_n_heads", default=32)),
            index_head_dim=int(pick("index_head_dim", default=128)),
            index_topk=int(pick("index_topk", default=2048)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=192)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=64)),
            qk_head_dim=int(pick("qk_head_dim", default=256)),
            v_head_dim=int(pick("v_head_dim", default=256)),
            first_k_dense_replace=int(pick("first_k_dense_replace", default=3)),
            moe_layer_freq=int(pick("moe_layer_freq", default=1)),
            n_routed_experts=int(pick("n_routed_experts", "num_local_experts", "num_experts", default=256)),
            n_shared_experts=int(pick("n_shared_experts", default=1)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=8)),
            n_group=int(pick("n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=2.5)),
            topk_method=str(pick("topk_method", default="noaux_tc")),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            moe_intermediate_size=int(pick("moe_intermediate_size", default=2048)),
            tp_size=tp_size,
            tp_rank=int(pick("tp_rank", default=0)),
            ep_size=int(pick("ep_size", default=1)),
            ep_rank=int(pick("ep_rank", default=0)),
            dp_size=dp_size,
            dp_rank=int(pick("dp_rank", default=0)),
            cp_size=cp_size,
            cp_rank=int(pick("cp_rank", default=0)),
            layerwise_split_size=int(pick("layerwise_split_size", default=1)),
            layerwise_split_rank=int(pick("layerwise_split_rank", default=0)),
            moe_tp_size=int(pick("moe_tp_size", default=1)),
            moe_tp_rank=int(pick("moe_tp_rank", default=0)),
            world_size=world_size,
            indexer_types=pick("indexer_types", default=None) or None,
            mlp_layer_types=pick("mlp_layer_types", default=None) or None,
            index_skip_topk_offset=int(pick("index_skip_topk_offset", default=2)),
            index_topk_freq=int(pick("index_topk_freq", default=1)),
            index_topk_pattern=pick("index_topk_pattern", default=None),
            indexer_rope_interleave=bool(pick("indexer_rope_interleave", default=True)),
            num_nextn_predict_layers=int(pick("num_nextn_predict_layers", default=0)),
            index_share_for_mtp_iteration=bool(pick("index_share_for_mtp_iteration", default=False)),
            layers_to_capture=tuple(int(layer_id) for layer_id in pick("layers_to_capture", default=[])),
        )
        cfg._resolve_indexer_types()
        cfg._resolve_mlp_layer_types()
        return cfg

    def validate(self) -> None:
        """Validate the orthogonal attention/DP and MoE EP topology."""
        if min(self.tp_size, self.ep_size, self.dp_size, self.cp_size, self.moe_tp_size) <= 0:
            raise ValueError("parallel sizes must be positive")
        if self.tp_size * self.dp_size * self.cp_size != self.world_size:
            raise ValueError("world_size must equal tp_size * dp_size * cp_size")
        if self.ep_size not in (1, self.world_size):
            raise ValueError(f"ep_size must be 1 or world_size ({self.world_size})")
        if self.ep_size > 1:
            if self.n_routed_experts % self.ep_size:
                raise ValueError("n_routed_experts must be divisible by ep_size")
            if self.moe_tp_size * self.ep_size != self.world_size:
                raise ValueError("world_size must equal moe_tp_size * ep_size")
        if self.moe_intermediate_size % effective_moe_tp(self):
            raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError("dp_rank must be in [0, dp_size)")
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError("cp_rank must be in [0, cp_size)")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("ep_rank must be in [0, ep_size)")
        if not 0 <= self.moe_tp_rank < self.moe_tp_size:
            raise ValueError("moe_tp_rank must be in [0, moe_tp_size)")
        if self.layerwise_split_size <= 0 or self.tp_size % self.layerwise_split_size:
            raise ValueError("layerwise_split_size must be a positive divisor of tp_size")
        if not 0 <= self.layerwise_split_rank < self.layerwise_split_size:
            raise ValueError("layerwise_split_rank must be in [0, layerwise_split_size)")
        if self.layerwise_split_size > 1 and self.cp_size > 1:
            raise ValueError("GLM5.2 Python does not support CP and layerwise split together")

    def _resolve_indexer_types(self) -> None:
        """Derive per-layer indexer mode (full/shared)."""
        if self.indexer_types is not None:
            return
        pattern = self.index_topk_pattern
        if pattern:
            if isinstance(pattern, str):
                self.indexer_types = [{"F": "full", "S": "shared"}[c] for c in pattern]
            else:
                self.indexer_types = list(pattern)
            return
        freq = max(self.index_topk_freq, 1)
        offset = self.index_skip_topk_offset
        self.indexer_types = [
            "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.n_layers)
        ]

    def _resolve_mlp_layer_types(self) -> None:
        """Derive per-layer MLP mode (dense/sparse)."""
        if self.mlp_layer_types is not None:
            return
        n_dense = min(self.first_k_dense_replace, self.n_layers)
        self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.n_layers - n_dense)

    def head_split(self) -> tuple[int, int]:
        """Per-rank (num_heads_local, num_kv_heads_local=1) — MLA has one latent KV head per rank."""
        return mla_head_split(self.n_heads, self.tp_size)


class Glm52MLP(DeepseekV3MLP):
    """GLM dense MLP with numerically stable FP32 TP reduction."""

    def _reduce_output(
        self,
        out: torch.Tensor,
        tp_reduce_add: torch.Tensor | None,
    ) -> torch.Tensor:
        if tp_reduce_add is not None:
            out = out + tp_reduce_add
        if self.tp > 1 and (not self.skip_tp_reduce or tp_reduce_add is not None):
            output_dtype = out.dtype
            out = out.to(torch.float32)
            distributed.tp_all_reduce(out)
            out = out.to(output_dtype)
        return out

    def _forward_gate_up(
        self,
        gate_up: torch.Tensor,
        tp_reduce_add: torch.Tensor | None,
    ) -> torch.Tensor:
        act = _swiglu_with_clamp(gate_up, self.swiglu_limit)
        out = self.down_proj(act)
        return self._reduce_output(out, tp_reduce_add)

    def forward_dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_int8, pertoken = kernels.dynamic_quant(x)
        gate_up = self.gate_up_proj.forward_accumulated(x_int8)
        act_int8, act_scale = kernels.dequant_swiglu_quant(
            gate_up,
            self.gate_up_proj.weight_scale,
            pertoken,
        )
        out = self.down_proj.forward_quantized(act_int8, act_scale)
        return self._reduce_output(out, tp_reduce_add)


class Glm52MLAAttention(Attention):
    """Absorbed-MLA attention with per-layer full/shared DSA indexer."""

    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp = cfg.tp_size
        num_heads, _ = cfg.head_split()
        kv_lora = cfg.kv_lora_rank
        qk_nope = cfg.qk_nope_head_dim
        qk_rope = cfg.qk_rope_head_dim
        v_head = cfg.v_head_dim
        scale = (qk_nope + qk_rope) ** -0.5
        attn_mscale = _yarn_get_mscale(cfg.rope_scaling_factor, cfg.rope_mscale_all_dim)
        scale = scale * attn_mscale * attn_mscale
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=kv_lora,
            scale=scale,
            sliding_window=0,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.qk_nope_head_dim = qk_nope
        self.qk_rope_head_dim = qk_rope
        self.v_head_dim = v_head
        self.kv_lora_rank = kv_lora
        self.num_heads_local = num_heads

        self.q_a_proj = W8A8StaticLinear(cfg.hidden_size, cfg.q_lora_rank, device)
        self.kv_a_proj_with_mqa = W8A8StaticLinear(cfg.hidden_size, kv_lora + qk_rope, device)
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.kv_a_layernorm = RMSNorm(kv_lora, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.q_b_proj = W8A8StaticLinear(cfg.q_lora_rank, num_heads * (qk_nope + qk_rope), device)
        self.kv_b_proj = ColumnParallelLinear(
            kv_lora,
            num_heads * (qk_nope + v_head),
            tp,
            dtype=dtype,
            device=device,
        )
        self.o_proj = W8A8StaticLinear(num_heads * v_head, cfg.hidden_size, device, row_parallel=True)
        self.register_buffer(
            "W_UK",
            torch.empty(num_heads, qk_nope, kv_lora, dtype=dtype, device=device),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.empty(num_heads, kv_lora, v_head, dtype=dtype, device=device),
            persistent=False,
        )
        self.is_shared = (
            cfg.indexer_types is not None
            and layer_id < len(cfg.indexer_types)
            and cfg.indexer_types[layer_id] == "shared"
        )
        self.indexer = None
        if not self.is_shared:
            self.indexer = Glm52Indexer(cfg, dtype, device)

    def process_weights_after_loading(self) -> None:
        self.q_a_proj.process_weights_after_loading()
        self.kv_a_proj_with_mqa.process_weights_after_loading()
        self.q_b_proj.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        w = self.kv_b_proj.weight.data
        w = w.view(
            self.num_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = w.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.W_UK.copy_(w_uk.contiguous())
        self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())
        if self.indexer is not None:
            self.indexer.process_weights_after_loading()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden.shape[0]
        q_a = self.q_a_proj(hidden)
        q_c = self.q_a_layernorm(q_a)
        forward_ctx = get_forward_context()
        backend = forward_ctx.attention_backend
        cp_context = forward_ctx.cp_context
        layerwise = self.cfg.layerwise_split_size > 1 and not (
            forward_ctx.metadata.is_prefill or forward_ctx.metadata.is_chunked_prefill
        )
        layer_owner = self.layer_id % self.cfg.layerwise_split_size
        owns_layer_cache = self.cfg.layerwise_split_rank == layer_owner
        if self.indexer is not None:
            ctx = backend.mla_index_context(self)
            if layerwise:
                if owns_layer_cache:
                    topk = self.indexer.select_qli(hidden, q_c, positions, ctx, cos_sin_cache)
                else:
                    topk = torch.empty(
                        (num_tokens, ctx.index_cache.size(2), self.cfg.index_topk),
                        dtype=torch.int32,
                        device=hidden.device,
                    )
                distributed.broadcast_(topk, layer_owner, "layerwise")
            elif cp_context is None:
                topk = self.indexer.select_qli(hidden, q_c, positions, ctx, cos_sin_cache)
            else:
                # Indexer queries are packed to real CP-owned rows.  The key
                # side is all-gathered inside the indexer so the paged index
                # cache remains globally addressable.
                query_index = cp_context.query_index
                topk = self.indexer.select_qli(
                    hidden.index_select(0, query_index),
                    q_c.index_select(0, query_index),
                    positions.index_select(0, query_index),
                    ctx,
                    cos_sin_cache,
                    cache_hidden=hidden,
                    cache_positions=positions,
                )
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous "
                    "full indexer layer (prev_topk_indices is None)."
                )
            topk = prev_topk_indices
        q = self.q_b_proj(q_c)
        q = q.view(
            num_tokens,
            self.num_heads_local,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
        )
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_latent = torch.bmm(q_nope.transpose(0, 1), self.W_UK).transpose(0, 1)
        cos, sin = _gather_interleave_cos_sin(cos_sin_cache, positions)
        q_pe = _interleave_rope_with(q_rope, cos, sin)
        kv = self.kv_a_proj_with_mqa(hidden)
        k_latent_raw, k_rope_raw = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_pe = _interleave_rope_with(k_rope_raw.unsqueeze(1), cos, sin)
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)
        k_pe_3d = k_pe.view(num_tokens, 1, self.qk_rope_head_dim)

        if layerwise:
            local_query = torch.cat((q_latent, q_pe), dim=-1)
            gathered_query = distributed.all_gather(
                local_query,
                dim=1,
                world_size=self.cfg.layerwise_split_size,
                group_name="layerwise",
            )
            latent_width = q_latent.shape[-1]
            gathered_q_latent = gathered_query[..., :latent_width]
            gathered_q_pe = gathered_query[..., latent_width:]
            if owns_layer_cache:
                gathered_attn_out = backend.execute_mla(
                    gathered_q_latent,
                    gathered_q_pe,
                    k_latent_3d,
                    k_pe_3d,
                    self,
                    topk=topk,
                )
            else:
                gathered_attn_out = torch.empty_like(gathered_q_latent)
            distributed.broadcast_(gathered_attn_out, layer_owner, "layerwise")
            head_offset = self.cfg.layerwise_split_rank * self.num_heads_local
            attn_out = gathered_attn_out.narrow(1, head_offset, self.num_heads_local)
        else:
            attn_out = backend.execute_mla(q_latent, q_pe, k_latent_3d, k_pe_3d, self, topk=topk)
        v_full = kernels.batch_matmul_transpose(
            attn_out.transpose(0, 1),
            self.W_UV,
        )
        v_full = v_full.reshape(num_tokens, self.num_heads_local * self.v_head_dim)
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            output_dtype = o.dtype
            o = o.to(torch.float32)
            distributed.all_reduce_(o)
            o = o.to(output_dtype)
        return o, topk


class Glm52Indexer(nn.Module):
    """GLM-5.2 DSA lightning indexer (wq_b W8A8, configurable RoPE)."""

    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.n_head = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self.indexer_rope_interleave = cfg.indexer_rope_interleave
        self.wq_b = W8A8StaticLinear(cfg.q_lora_rank, self.n_head * self.head_dim, device)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim, bias=False, dtype=dtype, device=device)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_head, bias=False, dtype=dtype, device=device)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.register_buffer(
            "hadamard",
            _create_hadamard_matrix(self.head_dim, dtype, device),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        self.wq_b.process_weights_after_loading()

    def _pad_q_heads_to_kernel_gsize(
        self,
        q: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        required_q_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # aclnnQuantLightningIndexer tiling hard-requires n_heads_q/n_heads_k == 64
        # (G_SIZE_LIMIT in quant_lightning_indexer_tiling.h; xllm_ops kernel hard-codes
        # matmul M=256 tile, head_dim=128 and kv_head=1 for DSV3.2/DSV4 shapes). GLM-5.2
        # is index_n_heads=32 / kv_head=1, so pad Q from 32 to 64 heads to satisfy the
        # kernel. The score sum_h(w_h * q_h . k) is mathematically unchanged: padded Q
        # rows are zero, so q_h . k = 0 for h >= n_head; padded weights are zero so those
        # zero terms cannot contribute even if the kernel processed them differently.
        pad_heads = required_q_heads - self.n_head
        if pad_heads == 0:
            return q, q_scale, weights
        if pad_heads < 0:
            raise RuntimeError(f"Glm52Indexer expected index_n_heads<={required_q_heads}, got {self.n_head}")
        q = torch.cat(
            [q, torch.zeros((q.size(0), pad_heads, q.size(2)), dtype=q.dtype, device=q.device)],
            dim=1,
        )
        q_scale = torch.cat(
            [q_scale, torch.zeros((q_scale.size(0), pad_heads), dtype=q_scale.dtype, device=q_scale.device)],
            dim=1,
        )
        weights = torch.cat(
            [weights, torch.zeros((weights.size(0), pad_heads), dtype=weights.dtype, device=weights.device)],
            dim=1,
        )
        return q, q_scale, weights

    def select_qli(
        self,
        hidden: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        ctx: MlaIndexContext,
        cos_sin_cache: torch.Tensor,
        cache_hidden: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actual_seq_q = ctx.actual_seq_q
        actual_seq_kv = ctx.actual_seq_kv
        q = self.wq_b(qr).view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        cache_hidden = hidden if cache_hidden is None else cache_hidden
        cache_positions = positions if cache_positions is None else cache_positions
        k = self.wk(cache_hidden)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        if self.indexer_rope_interleave:
            cos, sin = _gather_interleave_cos_sin(cos_sin_cache, positions)
            q_pe = _interleave_rope_with(q_pe, cos, sin)
            k_cos, k_sin = _gather_interleave_cos_sin(cos_sin_cache, cache_positions)
            k_pe = _interleave_rope_with(k_pe.unsqueeze(1), k_cos, k_sin).squeeze(1)
        else:
            q_pe = _apply_half_rope(cos_sin_cache, q_pe, positions)
            k_pe = _apply_half_rope(cos_sin_cache, k_pe.unsqueeze(1), cache_positions).squeeze(1)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        weights = self.weights_proj(hidden)
        if ctx.cp_context is not None:
            q = cp_gather_kv(q, ctx.cp_context).contiguous()
            k = cp_gather_kv(k, ctx.cp_context).contiguous()
            weights = cp_gather_kv(weights, ctx.cp_context).contiguous()

        index_cache = ctx.index_cache
        index_cache_scale = ctx.index_cache_scale
        use_quant_indexer = index_cache.dtype == torch.int8 and index_cache_scale is not None
        if use_quant_indexer:
            rotation_scale = self.head_dim**-0.5
            q = torch.matmul(q, self.hadamard) * rotation_scale
            k = torch.matmul(k, self.hadamard) * rotation_scale
            q, q_scale = kernels.dynamic_quant(q)
            k, k_scale = kernels.dynamic_quant(k)
            assert q_scale is not None
            assert k_scale is not None
            q_scale = q_scale.to(torch.float16)
            k_scale = k_scale.unsqueeze(-1).to(torch.float16)
            ctx.update_index_cache(k, k_scale)
            index_cache, index_cache_scale, block_table = ctx.materialize_index_cache()
            assert index_cache_scale is not None
            weight_scale = self.head_dim**-0.5 * self.n_head**-0.5
            # xLLM stores one index key per source token.
            cmp_ratio = 1

            required_q_heads = index_cache.size(2) * _QLI_KERNEL_GSIZE
            q, q_scale, weights_padded = self._pad_q_heads_to_kernel_gsize(q, q_scale, weights, required_q_heads)

            qli_metadata = ctx.get_quant_indexer_metadata(required_q_heads, self.head_dim, self.topk, cmp_ratio)
            topk = kernels.quant_lightning_indexer(
                q,
                index_cache,
                (weights_padded * weight_scale).to(torch.float16),
                q_scale,
                index_cache_scale,
                qli_metadata,
                actual_seq_q,
                actual_seq_kv,
                block_table,
                self.topk,
                cmp_ratio,
            )
        else:
            ctx.update_index_cache(k, None)
            index_cache, _, block_table = ctx.materialize_index_cache()
            topk = kernels.lightning_indexer(
                q,
                index_cache,
                weights,
                actual_seq_q,
                actual_seq_kv,
                block_table,
                "TND",
                "PA_BSND",
                self.topk,
                3,
                9223372036854775807,
                9223372036854775807,
                False,
            )
        if ctx.cp_context is not None:
            topk = cp_shard_rows(topk, ctx.cp_context)
        return topk


class Glm52MoE(DeepseekV3MoE):
    """EP MoE with CP rows materialized before expert reduction."""

    def _combine_expert_outputs(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_size > 1:
            return super()._combine_expert_outputs(routed, shared)

        final = routed + shared
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(final, "tp")
        return final

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cp_context = get_forward_context().cp_context
        if cp_context is None or self.ep_size == 1:
            return super().forward(hidden)

        global_hidden = cp_gather_kv(hidden, cp_context)
        global_output = super().forward(global_hidden)
        return cp_shard_rows(global_output, cp_context)


class Glm52DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: Glm52Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = Glm52MLAAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        mlp_type = (
            cfg.mlp_layer_types[layer_id]
            if cfg.mlp_layer_types is not None and layer_id < len(cfg.mlp_layer_types)
            else ("dense" if layer_id < cfg.first_k_dense_replace else "sparse")
        )
        if mlp_type == "dense":
            self.mlp = Glm52MLP(cfg, cfg.intermediate_size, dtype, device)
        else:
            self.mlp = Glm52MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        prev_topk_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, topk_indices = self.self_attn(hidden, positions, cos_sin_cache, prev_topk_indices)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual, topk_indices


class Glm52Model(nn.Module):
    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0
        self.cfg = cfg
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList([Glm52DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.rotary = Glm52YarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.aux_hidden_capture = AuxHiddenCapture(cfg.layers_to_capture)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            hidden = cp_shard_rows(hidden, cp_context)
            positions = cp_shard_positions(positions, cp_context).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        residual: torch.Tensor | None = None
        prev_topk: torch.Tensor | None = None
        aux_hidden_buffer = self.aux_hidden_capture.create_buffer(hidden)
        for layer_id, layer in enumerate(self.layers):
            hidden, residual, prev_topk = layer(hidden, residual, positions, cos_sin_cache, prev_topk)
            self.aux_hidden_capture.capture_layer(layer_id, hidden, residual, aux_hidden_buffer)
            record_layer_event(layer_id)
        hidden, _ = self.norm(hidden, residual)
        if cp_context is not None:
            hidden = cp_merge_rows(hidden, cp_context)
        return self.aux_hidden_capture.finalize(hidden, aux_hidden_buffer)


class Glm52ForCausalLM(PyModelBase):
    """GLM-5.2 causal LM. Registered under ``model_type='glm_moe_dsa'``."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = Glm52Config.from_dict(config)
        self.cfg.tp_size = int(config.get("tp_size", 1))
        self.cfg.tp_rank = int(config.get("tp_rank", _tp_rank_from_device(config.get("device", "npu:0"))))
        self.cfg.ep_size = int(config.get("ep_size", 1))
        self.cfg.ep_rank = int(config.get("ep_rank", 0))
        self.cfg.dp_size = int(config.get("dp_size", 1))
        self.cfg.dp_rank = int(config.get("dp_rank", 0))
        self.cfg.cp_size = int(config.get("cp_size", 1))
        self.cfg.cp_rank = int(config.get("cp_rank", 0))
        self.cfg.layerwise_split_size = int(config.get("layerwise_split_size", 1))
        self.cfg.layerwise_split_rank = int(config.get("layerwise_split_rank", 0))
        self.cfg.moe_tp_size = int(config.get("moe_tp_size", 1))
        self.cfg.moe_tp_rank = int(config.get("moe_tp_rank", 0))
        self.cfg.world_size = int(config.get("world_size", self.cfg.tp_size * self.cfg.dp_size * self.cfg.cp_size))
        self.cfg.validate()
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device
        tp = self.cfg.tp_size
        assert self.cfg.vocab_size % tp == 0
        self.model = Glm52Model(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        loader.copy_shard("model.embed_tokens.weight", dim=1)

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            loader.copy_replicated(p + "input_layernorm.weight")
            loader.copy_replicated(p + "post_attention_layernorm.weight")
            attn = p + "self_attn."
            loader.load_w8a8_projection(attn, "q_a_proj")
            loader.copy_replicated(attn + "q_a_layernorm.weight")
            loader.load_w8a8_projection(attn, "q_b_proj", {"weight": 0, "deq_scale": 0, "quant_bias": 0})
            loader.load_w8a8_projection(attn, "kv_a_proj_with_mqa")
            loader.copy_replicated(attn + "kv_a_layernorm.weight")
            loader.copy_shard(attn + "kv_b_proj.weight", dim=0)
            loader.load_w8a8_projection(attn, "o_proj", {"weight": 1})
            if not self.model.layers[i].self_attn.is_shared:
                idx = attn + "indexer."
                loader.load_w8a8_projection(idx, "wq_b")
                loader.copy_replicated(idx + "wk.weight")
                loader.copy_replicated(idx + "k_norm.weight")
                loader.copy_replicated(idx + "k_norm.bias")
                loader.copy_replicated(idx + "weights_proj.weight")
            self.model.layers[i].self_attn.process_weights_after_loading()

            self.model.layers[i].mlp.load_from_checkpoint(loader, p + "mlp.")

        loader.copy_replicated("model.norm.weight")
        loader.copy_shard("lm_head.weight", dim=0)
