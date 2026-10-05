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
  * YaRN coefficients are prepared once per model forward and shared across
    attention and indexer consumers (no per-layer rotary module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import distributed, kernels
from xllm.python.attention.attn_dp_collectives import (
    attention_latent_all_to_all_dp_tp,
    o_all_reduce_dp_tp,
    q_head_all_to_all_dp_tp,
    quantized_value_all_to_all_dp_tp,
)
from xllm.python.attention.attn_dp_layout import AttnDpLayout
from xllm.python.attention.backend import AttentionBackend, MlaIndexContext
from xllm.python.device_stream import get_device_stream
from xllm.python.layers import ColumnParallelLinear
from xllm.python.model_executor.cp_utils import (
    CpContext,
    cp_gather_kv,
    cp_shard_rows,
)
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_loader import W8A8WeightLoader, mla_head_split, moe_shard
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import (
    DeepseekV3DecoderLayer,
    DeepseekV3Indexer,
    DeepseekV3MLAAttention,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    W8A8AttentionLinear,
    _interleave_rope_with,
    _tp_rank_from_device,
    _validate_rope_cos_sin,
)
from xllm.python.models.deepseek_v32 import (
    DeepseekYarnRotaryEmbedding as Glm52YarnRotaryEmbedding,
)

_MLAPO_V2_Q_LORA_RANK = 1536
_MLAPO_V2_KV_LORA_RANK = 512
_MLAPO_V2_QK_NOPE_HEAD_DIM = 128
_MLAPO_V2_QK_ROPE_HEAD_DIM = 64
_MLAPO_V2_V_HEAD_DIM = 128


def _can_use_mlapo_v2(cfg: Glm52Config, device: torch.device) -> bool:
    if not cfg.enable_mlapo or device.type not in ("npu", "privateuseone"):
        return False
    if (
        cfg.q_lora_rank != _MLAPO_V2_Q_LORA_RANK
        or cfg.kv_lora_rank != _MLAPO_V2_KV_LORA_RANK
        or cfg.qk_nope_head_dim != _MLAPO_V2_QK_NOPE_HEAD_DIM
        or cfg.qk_rope_head_dim != _MLAPO_V2_QK_ROPE_HEAD_DIM
        or cfg.v_head_dim != _MLAPO_V2_V_HEAD_DIM
    ):
        return False
    return kernels.supports_mla_preprocess_v2(
        cfg.kv_lora_rank,
        cfg.qk_rope_head_dim,
    )


def _load_w8a8_attention_projection(
    loader: W8A8WeightLoader,
    module: W8A8AttentionLinear,
    prefix: str,
    proj: str,
    shard_dims: dict[str, int] | None = None,
    *,
    world: int | None = None,
    rank: int | None = None,
) -> None:
    dynamic_activation = loader.w8a8_projection_uses_dynamic_activation(prefix, proj)
    module._set_dynamic_activation(dynamic_activation)
    loader.load_compatible_w8a8_projection(
        prefix,
        proj,
        shard_dims,
        dynamic_activation=dynamic_activation,
        world=world,
        rank=rank,
    )


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
    enable_dsa_multi_stream: bool = False
    num_nextn_predict_layers: int = 0
    index_share_for_mtp_iteration: bool = False
    enable_mlapo: bool = True
    enable_attn_dp_weight_sharding: bool = False
    layers_to_capture: tuple[int, ...] = ()
    enable_mega_moe: bool = False
    mega_moe_context: torch.Tensor | None = None
    mega_moe_ccl_buffer_size: int = 0
    mega_moe_num_max_tokens_per_rank: int = 0

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
            rope_theta=float(rpick("rope_theta", default=1.0e6)),
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
            enable_dsa_multi_stream=bool(pick("enable_dsa_multi_stream", default=False)),
            num_nextn_predict_layers=int(pick("num_nextn_predict_layers", default=0)),
            index_share_for_mtp_iteration=bool(pick("index_share_for_mtp_iteration", default=False)),
            enable_mlapo=bool(pick("enable_mlapo", default=True)),
            enable_attn_dp_weight_sharding=bool(pick("enable_attn_dp_weight_sharding", default=False)),
            layers_to_capture=tuple(int(layer_id) for layer_id in pick("layers_to_capture", default=[])),
            enable_mega_moe=bool(pick("enable_mega_moe", default=False)),
            mega_moe_context=pick("mega_moe_context", default=None),
            mega_moe_ccl_buffer_size=int(pick("mega_moe_ccl_buffer_size", default=0)),
            mega_moe_num_max_tokens_per_rank=int(pick("mega_moe_num_max_tokens_per_rank", default=0)),
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
        if self.world_size % self.ep_size:
            raise ValueError(f"ep_size must divide world_size: ep_size={self.ep_size}, world_size={self.world_size}")
        if self.ep_size > 1:
            if self.n_routed_experts % self.ep_size:
                raise ValueError("n_routed_experts must be divisible by ep_size")
            if self.moe_tp_size * self.ep_size != self.world_size:
                raise ValueError("world_size must equal moe_tp_size * ep_size")
        if self.moe_intermediate_size % moe_shard(self)[0]:
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
        if self.enable_attn_dp_weight_sharding:
            if self.cp_size != 1 or self.layerwise_split_size != 1:
                raise ValueError("attention DP sharding requires cp_size=1 and layerwise_split_size=1")
            if self.dp_size <= 1:
                raise ValueError("attention DP sharding requires dp_size > 1")
            if self.n_heads % (self.tp_size * self.dp_size):
                raise ValueError("attention heads must be divisible by tp_size * dp_size")

    def attention_weight_shard(self) -> tuple[int, int]:
        """Keep each TP owner's head range contiguous across its DP peers."""
        if self.enable_attn_dp_weight_sharding:
            return self.tp_size * self.dp_size, self.tp_rank * self.dp_size + self.dp_rank
        return self.tp_size, self.tp_rank

    def _resolve_indexer_types(self) -> None:
        """Derive per-layer indexer mode (full/shared)."""
        if self.indexer_types is not None:
            if len(self.indexer_types) == self.n_layers:
                return
            self.indexer_types = None
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
            if len(self.mlp_layer_types) == self.n_layers:
                return
            self.mlp_layer_types = None
        n_dense = min(self.first_k_dense_replace, self.n_layers)
        self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (self.n_layers - n_dense)

    def head_split(self) -> tuple[int, int]:
        """Per-rank (num_heads_local, num_kv_heads_local=1) — MLA has one latent KV head per rank."""
        return mla_head_split(self.n_heads, self.tp_size)


def _attn_dp_execution_counts(cfg: Glm52Config) -> tuple[int, ...]:
    counts = tuple(get_forward_context().metadata.dp_execution_token_counts)
    if len(counts) != cfg.dp_size or any(count <= 0 for count in counts):
        raise ValueError("attention DP requires positive execution token counts for every DP rank")
    return counts


def _attn_dp_owner_rows(value: torch.Tensor, cfg: Glm52Config) -> torch.Tensor:
    counts = _attn_dp_execution_counts(cfg)
    padded_tokens = max(counts)
    if value.shape[0] != cfg.dp_size * padded_tokens:
        raise ValueError("attention DP output must contain all padded DP rows")
    return value.narrow(0, cfg.dp_rank * padded_tokens, counts[cfg.dp_rank])


def _attn_dp_gather_inputs(
    hidden: torch.Tensor, positions: torch.Tensor, cfg: Glm52Config
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = _attn_dp_execution_counts(cfg)
    local_tokens = counts[cfg.dp_rank]
    if hidden.shape[0] != local_tokens or positions.shape[0] != local_tokens:
        raise ValueError("attention DP execution counts must match local hidden and positions")
    pad = max(counts) - local_tokens
    hidden = F.pad(hidden, (0, 0, 0, pad))
    positions = F.pad(positions.to(torch.int64), (0, pad))
    hidden = distributed.all_gather(hidden, dim=0, world_size=cfg.dp_size, group_name="dp")
    positions = distributed.all_gather(positions, dim=0, world_size=cfg.dp_size, group_name="dp")
    return hidden, positions


class Glm52MLAAttention(DeepseekV3MLAAttention):
    """Checkpoint, index-sharing and CP/layerwise adapters for common MLA."""

    _linear_type = W8A8AttentionLinear

    def __init__(self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, layer_id, dtype, device)
        self._attn_dp_layout: AttnDpLayout | None = None
        self.register_buffer("W_UV_owner", torch.empty(0, dtype=dtype, device=device), persistent=False)
        if not cfg.enable_attn_dp_weight_sharding:
            return
        world, _ = cfg.attention_weight_shard()
        self.weight_heads_local = cfg.n_heads // world
        self._attn_dp_layout = AttnDpLayout(
            world_size=world,
            num_heads=cfg.n_heads,
            q_head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            kv_lora_rank=cfg.kv_lora_rank,
            hidden_size=cfg.hidden_size,
            owner_heads=self.num_heads_local,
        )
        # Backend/cache metadata still describes TP-local owner heads. Only
        # the head-dependent projection weights are sharded over TP * DP.
        heads = self.weight_heads_local
        self.q_b_proj = W8A8AttentionLinear(
            cfg.q_lora_rank, heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), device
        )
        self.kv_b_proj = ColumnParallelLinear(
            cfg.kv_lora_rank,
            heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            world,
            dtype=dtype,
            device=device,
        )
        self.o_proj = W8A8AttentionLinear(heads * cfg.v_head_dim, cfg.hidden_size, device, row_parallel=True)
        self.W_UK = torch.empty(heads, cfg.qk_nope_head_dim, cfg.kv_lora_rank, dtype=dtype, device=device)
        self.W_UV = torch.empty(heads, cfg.kv_lora_rank, cfg.v_head_dim, dtype=dtype, device=device)
        self._use_fused_mla_decode = False
        self._use_mlapo_v2 = False

    def process_weights_after_loading(self, owner_kv_b_proj: torch.Tensor | None = None) -> None:
        if getattr(self, "_attn_dp_layout", None) is None:
            super().process_weights_after_loading()
            return
        if self.o_proj._dynamic_activation and owner_kv_b_proj is None:
            raise ValueError("dynamic attention DP requires the owner TP shard of kv_b_proj")
        self._prepare_separate_a_projections()
        self.q_b_proj.process_weights_after_loading()
        self.o_proj.process_weights_after_loading()
        if not self.o_proj._dynamic_activation and self.cfg.dp_rank != 0:
            # The inherited row-parallel linear adds bias only on tp_rank=0.
            # C4 spans DP as well, so only DP0 may contribute that bias.
            self.o_proj.quant_bias.zero_()
        weight = self.kv_b_proj.weight.data.view(
            self.weight_heads_local,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk, w_uv = weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=1)
        self.W_UK.copy_(w_uk.contiguous())
        if self.o_proj._dynamic_activation:
            owner_weight = owner_kv_b_proj.view(
                self.num_heads_local,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank,
            )
            self.W_UV_owner = (
                owner_weight[:, self.qk_nope_head_dim :]
                .transpose(1, 2)
                .to(device=self.W_UV.device, dtype=self.W_UV.dtype)
                .contiguous()
            )
            self.W_UV = self.W_UV.new_empty(0)
        else:
            self.W_UV.copy_(w_uv.transpose(1, 2).contiguous())
        # Preserve the parameter identity held by the checkpoint loader while
        # releasing the staging storage after absorbing UK/UV.
        self.kv_b_proj.weight.data = self.kv_b_proj.weight.new_empty(0)
        self._prepare_indexer_weights()

    def _mlapo_enabled(self, cfg: Glm52Config, device: torch.device) -> bool:
        return _can_use_mlapo_v2(cfg, device)

    def _init_a_projections(self, cfg: Glm52Config, device: torch.device) -> None:
        self._combined_qkv: W8A8AttentionLinear | None = None
        self.q_a_proj = W8A8AttentionLinear(cfg.hidden_size, cfg.q_lora_rank, device)
        self.kv_a_proj_with_mqa = W8A8AttentionLinear(cfg.hidden_size, cfg.kv_lora_rank + cfg.qk_rope_head_dim, device)

    def _make_indexer(
        self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> Glm52Indexer | None:
        # A draft shared layer still needs K/cache updates, and full selection
        # on the first step. Target shared layers have no indexer at all.
        is_mtp = cfg.model_type.endswith("_mtp") and cfg.index_share_for_mtp_iteration
        self.is_shared = (
            not is_mtp
            and cfg.indexer_types is not None
            and layer_id < len(cfg.indexer_types)
            and cfg.indexer_types[layer_id] == "shared"
        )
        return None if self.is_shared else Glm52Indexer(cfg, dtype, device, layer_id)

    def _prepare_a_projection(self) -> W8A8AttentionLinear | None:
        self._combined_qkv = W8A8AttentionLinear.combine(self.kv_a_proj_with_mqa, self.q_a_proj)
        return self._combined_qkv

    def _prepare_separate_a_projections(self) -> None:
        self.q_a_proj.process_weights_after_loading()
        self.kv_a_proj_with_mqa.process_weights_after_loading()

    def _prepare_indexer_weights(self) -> None:
        if self.indexer is not None:
            self.indexer.process_weights_after_loading()

    def _a_projection(self) -> W8A8AttentionLinear | None:
        return self._combined_qkv

    def _project_separate_a(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_a_proj(hidden), self.kv_a_proj_with_mqa(hidden)

    def _can_fuse_input_norm_quant(self) -> bool:
        projection = self._a_projection()
        return (
            self._use_fused_mla_decode
            and not self.cfg.model_type.endswith("_mtp")
            and self.is_shared
            and self.indexer is None
            and getattr(self, "_attn_dp_layout", None) is None
            and self._dynamic_mla_ready
            and isinstance(projection, W8A8AttentionLinear)
            and projection._dynamic_activation is True
        )

    def _can_fuse_decode(self) -> bool:
        ctx = get_forward_context()
        layerwise = self.cfg.layerwise_split_size > 1 and not (
            ctx.metadata.is_prefill or ctx.metadata.is_chunked_prefill
        )
        return super()._can_fuse_decode() and ctx.cp_context is None and not layerwise

    def _select_topk(
        self,
        hidden: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        backend: AttentionBackend,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        prev_topk: torch.Tensor | None,
        reuse_topk: bool,
    ) -> torch.Tensor:
        assert query_cos_sin is not None
        key_cos_sin = (rope_cos, rope_sin) if self.cfg.indexer_rope_interleave else (half_rope_cos, half_rope_sin)
        ctx = get_forward_context()
        cp_context = ctx.cp_context
        layerwise = self.cfg.layerwise_split_size > 1 and not (
            ctx.metadata.is_prefill or ctx.metadata.is_chunked_prefill
        )
        layer_owner = self.layer_id % self.cfg.layerwise_split_size
        owns_layer_cache = self.cfg.layerwise_split_rank == layer_owner
        if reuse_topk:
            if prev_topk is None:
                raise ValueError("MTP DSA top-k reuse requires indices from the previous draft step")
            if self.indexer is not None:
                ctx = backend.mla_index_context(self)
                if not layerwise or owns_layer_cache:
                    self.indexer._update_index_cache(hidden, ctx, key_cos_sin)
            topk = prev_topk
        elif self.indexer is not None:
            ctx = backend.mla_index_context(self)
            if layerwise:
                if owns_layer_cache:
                    topk = self.indexer.select_qli(hidden, q_c, ctx, query_cos_sin, key_cos_sin)
                else:
                    topk = torch.empty(
                        (hidden.shape[0], ctx.index_cache.size(2), self.cfg.index_topk),
                        dtype=torch.int32,
                        device=hidden.device,
                    )
                distributed.broadcast_(topk, layer_owner, "layerwise")
            elif cp_context is None:
                topk = self.indexer.select_qli(hidden, q_c, ctx, query_cos_sin, key_cos_sin)
            else:
                # Indexer queries are packed to real CP-owned rows.  The key
                # side is all-gathered inside the indexer so the paged index
                # cache remains globally addressable.
                query_index = cp_context.query_index
                local_q_c = (
                    tuple(value.index_select(0, query_index) for value in q_c)
                    if isinstance(q_c, tuple)
                    else q_c.index_select(0, query_index)
                )
                topk = self.indexer.select_qli(
                    hidden.index_select(0, query_index),
                    local_q_c,
                    ctx,
                    query_cos_sin,
                    key_cos_sin,
                    cache_hidden=hidden,
                )
        else:
            if prev_topk is None:
                raise ValueError(
                    "Shared DSA layers require top-k indices from a previous full indexer layer (prev_topk is None)."
                )
            topk = prev_topk
        return topk

    def _execute_attention(
        self,
        backend: AttentionBackend,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        topk: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx = get_forward_context()
        layerwise = self.cfg.layerwise_split_size > 1 and not (
            ctx.metadata.is_prefill or ctx.metadata.is_chunked_prefill
        )
        layer_owner = self.layer_id % self.cfg.layerwise_split_size
        owns_layer_cache = self.cfg.layerwise_split_rank == layer_owner
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
        return attn_out

    def _forward_with_topk(
        self,
        hidden: torch.Tensor,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk: torch.Tensor | None = None,
        reuse_topk: bool = False,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if getattr(self, "_attn_dp_layout", None) is None:
            return super()._forward_with_topk(
                hidden,
                half_rope_cos,
                half_rope_sin,
                rope_cos,
                rope_sin,
                query_cos_sin,
                prev_topk,
                reuse_topk,
                hidden_scale,
                slot_mapping_int64,
            )
        self._validate_hidden_scale(hidden, hidden_scale)
        if slot_mapping_int64 is not None:
            raise ValueError("attention DP separate projections do not consume prepared MLA INT64 slots")
        layout = self._attn_dp_layout
        counts = _attn_dp_execution_counts(self.cfg)
        local_tokens = counts[self.cfg.dp_rank]
        padded_tokens = max(counts)
        if hidden.shape[0] != self.cfg.dp_size * padded_tokens:
            raise ValueError("attention DP hidden must contain all padded DP rows")
        offset = self.cfg.dp_rank * padded_tokens
        local_hidden = hidden.narrow(0, offset, local_tokens)
        local_rope = tuple(
            value.narrow(0, offset, local_tokens) for value in (half_rope_cos, half_rope_sin, rope_cos, rope_sin)
        )
        if query_cos_sin is None:
            raise ValueError("attention DP requires indexer RoPE coefficients")
        local_query_cos_sin = tuple(value.narrow(0, offset, local_tokens) for value in query_cos_sin)
        backend = get_forward_context().attention_backend

        q_c, q = self._normalize_and_project_query(self.q_a_proj(hidden))
        local_q_c = (
            tuple(value.narrow(0, offset, local_tokens) for value in q_c)
            if isinstance(q_c, tuple)
            else q_c.narrow(0, offset, local_tokens)
        )
        q = q.view(hidden.shape[0], self.weight_heads_local, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_latent = kernels.atb_matmul_ein_sum(q_nope, self.W_UK)
        owner_query = q_head_all_to_all_dp_tp(
            torch.cat((q_latent, q_rope), dim=-1),
            layout,
            local_tokens,
            padded_tokens,
            self.cfg.tp_size,
            self.cfg.dp_size,
            self.cfg.tp_rank,
            self.cfg.dp_rank,
        )
        q_latent, q_rope = owner_query.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        q_pe = _interleave_rope_with(q_rope, local_rope[2], local_rope[3])
        kv = self.kv_a_proj_with_mqa(local_hidden)
        k_latent_raw, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_latent = self.kv_a_layernorm(k_latent_raw).unsqueeze(1)
        k_pe = _interleave_rope_with(k_rope.unsqueeze(1), local_rope[2], local_rope[3])
        topk = self._select_topk(
            local_hidden, local_q_c, backend, *local_rope, local_query_cos_sin, prev_topk, reuse_topk
        )
        attn_out = backend.execute_mla(q_latent.contiguous(), q_pe, k_latent, k_pe, self, topk=topk)
        if self.o_proj._dynamic_activation:
            if self.W_UV_owner.numel() == 0:
                raise RuntimeError("dynamic attention DP requires full owner-local W_UV weights")
            owner_value = kernels.atb_matmul_ein_sum(attn_out, self.W_UV_owner)
            owner_value = owner_value.reshape(local_tokens, self.num_heads_local * self.v_head_dim)
            value_i8, scale = kernels.dynamic_quant(owner_value)
            weight_value, weight_scale = quantized_value_all_to_all_dp_tp(
                value_i8,
                scale,
                layout,
                self.v_head_dim,
                local_tokens,
                padded_tokens,
                self.cfg.tp_size,
                self.cfg.dp_size,
                self.cfg.tp_rank,
                self.cfg.dp_rank,
            )
            output = self.o_proj.forward_quantized(
                weight_value.reshape(hidden.shape[0], self.weight_heads_local * self.v_head_dim), weight_scale
            )
        else:
            weight_latent = attention_latent_all_to_all_dp_tp(
                attn_out,
                layout,
                local_tokens,
                padded_tokens,
                self.cfg.tp_size,
                self.cfg.dp_size,
                self.cfg.tp_rank,
                self.cfg.dp_rank,
            )
            value = kernels.atb_matmul_ein_sum(weight_latent, self.W_UV)
            output = self.o_proj(value.reshape(hidden.shape[0], self.weight_heads_local * self.v_head_dim))
        output_dtype = output.dtype
        output = o_all_reduce_dp_tp(output.float(), layout, padded_tokens, self.cfg.dp_size)
        return output.to(output_dtype), topk

    def forward(
        self,
        hidden: torch.Tensor,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        indexer_query_cos_sin: tuple[torch.Tensor, torch.Tensor],
        prev_topk_indices: torch.Tensor | None = None,
        reuse_topk_indices: bool = False,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_hidden_scale(hidden, hidden_scale)
        rope_dtype = self.q_a_layernorm.weight.dtype if hidden_scale is not None else hidden.dtype
        _validate_rope_cos_sin((rope_cos, rope_sin), hidden, self.qk_rope_head_dim, True, "attention", rope_dtype)
        return self._forward_with_topk(
            hidden,
            half_rope_cos,
            half_rope_sin,
            rope_cos,
            rope_sin,
            indexer_query_cos_sin,
            prev_topk_indices,
            reuse_topk_indices,
            hidden_scale,
            slot_mapping_int64,
        )


class Glm52Indexer(DeepseekV3Indexer):
    """GLM checkpoint projections and optional DSA projection streams."""

    def _uses_interleaved_rope(self, cfg: Glm52Config) -> bool:
        return cfg.indexer_rope_interleave

    def _init_streams(self, cfg: Glm52Config, device: torch.device) -> None:
        super()._init_streams(cfg, device)
        if cfg.enable_dsa_multi_stream:
            self._weights_stream = get_device_stream(device, "dsa_indexer_weights")
            if not self.indexer_rope_interleave:
                self._q_stream = get_device_stream(device, "dsa_indexer_q")

    def _init_projections(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        self.wq_b = W8A8AttentionLinear(cfg.q_lora_rank, self.n_head * self.head_dim, device)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim, bias=False, dtype=dtype, device=device)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_head, bias=False, dtype=dtype, device=device)
        self.register_buffer(
            "_wk_weights_proj_weight",
            torch.empty(self.head_dim + self.n_head, cfg.hidden_size, dtype=dtype, device=device),
            persistent=False,
        )
        self._wk_weights_proj_ready = False

    def process_weights_after_loading(self) -> None:
        self.wq_b.process_weights_after_loading()
        with torch.no_grad():
            self._wk_weights_proj_weight.copy_(torch.cat((self.wk.weight, self.weights_proj.weight), dim=0))
        self._wk_weights_proj_ready = True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._wk_weights_proj_ready = False

    def _project_k_and_weights(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._wk_weights_proj_ready:
            projected = F.linear(hidden, self._wk_weights_proj_weight)
            return projected[..., : self.head_dim], projected[..., self.head_dim :].contiguous()
        return self.wk(hidden), self.weights_proj(hidden)

    def _project_key(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.wk(hidden)

    def _project_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.weights_proj(hidden)

    def select_qli(
        self,
        hidden: torch.Tensor,
        qr: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ctx: MlaIndexContext,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor],
        key_cos_sin: tuple[torch.Tensor, torch.Tensor],
        cache_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._select_qli(hidden, qr, ctx, query_cos_sin, key_cos_sin, cache_hidden)


class Glm52MoE(DeepseekV3MoE):
    """EP MoE with CP rows materialized before expert reduction."""

    def _combine_expert_outputs(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        use_mega_moe: bool,
    ) -> torch.Tensor:
        if self.ep_size > 1:
            return super()._combine_expert_outputs(routed, shared, use_mega_moe)

        final = routed + shared
        if getattr(self.cfg, "enable_attn_dp_weight_sharding", False):
            if self.moe_tp_size > 1:
                distributed.all_reduce_(final, "moe_tp")
            return final
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(final, "tp")
        return final

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.enable_attn_dp_weight_sharding:
            use_mega_moe = self._should_use_mega_moe()
            # The model already materialized padded DP rows for all layers.
            if self._fine_overlap_enabled:
                return self._forward_fine_grained_parallel(hidden)
            if self._expert_parallel_enabled:
                return self._forward_parallel(hidden, use_mega_moe)
            return self._combine_expert_outputs(
                self._run_routed_experts(hidden, use_mega_moe), self._run_shared_experts(hidden), use_mega_moe
            )
        cp_context = get_forward_context().cp_context
        if cp_context is None or self.ep_size == 1:
            return super().forward(hidden)

        global_hidden = cp_gather_kv(hidden, cp_context)
        global_output = super().forward(global_hidden)
        return cp_shard_rows(global_output, cp_context)


class Glm52DecoderLayer(DeepseekV3DecoderLayer):
    def _make_attention(
        self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> Glm52MLAAttention:
        return Glm52MLAAttention(cfg, layer_id, dtype, device)

    def _make_mlp(
        self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> DeepseekV3MLP | DeepseekV3MoE:
        mlp_type = (
            cfg.mlp_layer_types[layer_id]
            if cfg.mlp_layer_types is not None and layer_id < len(cfg.mlp_layer_types)
            else ("dense" if layer_id < cfg.first_k_dense_replace else "sparse")
        )
        if mlp_type == "dense":
            return DeepseekV3MLP(cfg, cfg.intermediate_size, dtype, device)
        return Glm52MoE(cfg, layer_id, dtype, device)

    def _attention(
        self,
        hidden: torch.Tensor,
        half_rope_cos: torch.Tensor,
        half_rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        query_cos_sin: tuple[torch.Tensor, torch.Tensor] | None,
        prev_topk: torch.Tensor | None,
        reuse_topk: bool,
        hidden_scale: torch.Tensor | None = None,
        slot_mapping_int64: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert query_cos_sin is not None
        return self.self_attn(
            hidden,
            half_rope_cos,
            half_rope_sin,
            rope_cos,
            rope_sin,
            query_cos_sin,
            prev_topk,
            reuse_topk,
            hidden_scale,
            slot_mapping_int64,
        )


class Glm52Model(DeepseekV3Model):
    def __init__(self, cfg: Glm52Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(cfg, dtype, device)
        self.aux_hidden_capture = AuxHiddenCapture(cfg.layers_to_capture)

    def _make_decoder(
        self, cfg: Glm52Config, layer_id: int, dtype: torch.dtype, device: torch.device
    ) -> Glm52DecoderLayer:
        return Glm52DecoderLayer(cfg, layer_id, dtype, device)

    def _record_layer_event(self, layer_id: int) -> None:
        record_layer_event(layer_id)

    def _cp_context(self) -> CpContext | None:
        return get_forward_context().cp_context

    def _indexer_interleaved(self) -> bool:
        return self.cfg.indexer_rope_interleave

    def _prepare_mla_slots(self) -> torch.Tensor | None:
        if len(self.layers) <= 1 or self.cfg.model_type.endswith("_mtp"):
            return None
        attention = self.layers[0].self_attn
        if not attention._dynamic_mla_ready or not attention._can_fuse_decode():
            return None
        context = get_forward_context().attention_backend.mla_preprocess_context(attention)
        if context is None or not kernels.supports_mla_kv_cache_slot_reuse(context.kv_cache):
            return None
        # Captured once before the layer loop, and refreshed from live slots on every replay.
        return context.slot_mapping.to(torch.int64)

    def _prepare_layer_inputs(
        self, hidden: torch.Tensor, positions: torch.Tensor, cp_context: CpContext | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]:
        if self.cfg.enable_attn_dp_weight_sharding:
            if cp_context is not None:
                raise ValueError("attention DP sharding does not support CP")
            hidden, positions = _attn_dp_gather_inputs(hidden, positions, self.cfg)
        return super()._prepare_layer_inputs(hidden, positions, cp_context)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output = super().forward(input_ids, positions)
        if not self.cfg.enable_attn_dp_weight_sharding:
            return output
        if isinstance(output, tuple):
            return tuple(_attn_dp_owner_rows(value, self.cfg) for value in output)
        return _attn_dp_owner_rows(output, self.cfg)


class Glm52ForCausalLM(PyModelBase):
    """GLM-5.2 causal LM. Registered under ``model_type='glm_moe_dsa'``."""

    def __init__(self, config: dict, build_model: bool = True) -> None:
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
        self.model: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        if build_model:
            self._build_model()

    def _build_model(self) -> None:
        tp = self.cfg.tp_size
        self.model = Glm52Model(self.cfg, self.dtype, self.device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=self.dtype,
            device=self.device,
        )

    def load_weights(
        self,
        state_dicts: list,
        tp_rank: int,
        tp_size: int,
        load_lm_head: bool = True,
        load_embedding: bool = True,
        loader: W8A8WeightLoader | None = None,
    ) -> None:
        cfg = self.cfg
        if loader is None:
            loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)
        if self.model is None:
            raise RuntimeError("GLM model body must be built before loading weights")

        if load_embedding:
            loader.copy_shard("model.embed_tokens.weight", dim=1)

        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            loader.copy_replicated(p + "input_layernorm.weight")
            loader.copy_replicated(p + "post_attention_layernorm.weight")
            attn = p + "self_attn."
            attention = self.model.layers[i].self_attn
            attn_world, attn_rank = cfg.attention_weight_shard()
            _load_w8a8_attention_projection(loader, attention.q_a_proj, attn, "q_a_proj")
            loader.copy_replicated(attn + "q_a_layernorm.weight")
            _load_w8a8_attention_projection(
                loader,
                attention.q_b_proj,
                attn,
                "q_b_proj",
                {
                    "weight": 0,
                    "deq_scale": 0,
                    "quant_bias": 0,
                    "weight_scale": 0,
                    "weight_offset": 0,
                },
                world=attn_world,
                rank=attn_rank,
            )
            _load_w8a8_attention_projection(
                loader,
                attention.kv_a_proj_with_mqa,
                attn,
                "kv_a_proj_with_mqa",
            )
            loader.copy_replicated(attn + "kv_a_layernorm.weight")
            loader.copy_in(
                attn + "kv_b_proj.weight",
                loader.load_shard(attn + "kv_b_proj.weight", dim=0, world=attn_world, rank=attn_rank),
            )
            _load_w8a8_attention_projection(
                loader, attention.o_proj, attn, "o_proj", {"weight": 1}, world=attn_world, rank=attn_rank
            )
            if not attention.is_shared:
                idx = attn + "indexer."
                assert attention.indexer is not None
                _load_w8a8_attention_projection(loader, attention.indexer.wq_b, idx, "wq_b")
                loader.copy_replicated(idx + "wk.weight")
                loader.copy_replicated(idx + "k_norm.weight")
                loader.copy_replicated(idx + "k_norm.bias")
                loader.copy_replicated(idx + "weights_proj.weight")
            if cfg.enable_attn_dp_weight_sharding and attention.o_proj._dynamic_activation:
                # Quantize the complete owner TP row before splitting its V
                # features; shard-local scales would change W8A8 numerics.
                owner_kv_b = loader.load_shard(attn + "kv_b_proj.weight", dim=0, world=cfg.tp_size, rank=cfg.tp_rank)
                attention.process_weights_after_loading(owner_kv_b)
                del owner_kv_b
            else:
                attention.process_weights_after_loading()

            self.model.layers[i].mlp.load_from_checkpoint(loader, p + "mlp.")

        loader.copy_replicated("model.norm.weight")
        if load_lm_head:
            loader.copy_shard("lm_head.weight", dim=0)
