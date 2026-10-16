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

"""glm5_next (model_type=glm5_next) causal LM — Python model executor target.

Hybrid decoder: 3 of every 4 layers are KDA (Kimi Delta Attention) linear
attention; the 4th is MLA full-attention driven by a DSA ``kPool`` sparse
indexer. MLP is dense SwiGLU for the first ``first_k_dense_replace`` layers and
DeepSeek-V2-style MoE (sigmoid + noaux_tc) thereafter. bf16 throughout, matching
the patched HuggingFace ``Glm5NextForCausalLM`` reference for tensor alignment.

KDA runs through the attention backend's fused NPU operators. The executor
binds per-sequence convolution and recurrent-state cache slots, and the
backend manages their reads, updates, and speculative verification.
The mHC residual follows the patched transformers reference.
"""

from __future__ import annotations

import functools
import importlib.util
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None  # type: ignore[assignment]

from xllm.python import distributed, kernels
from xllm.python.attention.backend import MlaIndexContext
from xllm.python.attention.mla_index_context import (
    KPoolMlaIndexContext,
    KPoolMlaIndexContextBuilder,
    is_compact_kpool_tail,
)
from xllm.python.model_executor.cp_utils import (
    cp_merge_rows,
    cp_shard_positions,
    cp_shard_rows,
    cp_zero_fill_local_rows,
)
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    get_forward_context_or_none,
    in_acl_graph,
    record_layer_event,
)

_has_mhc_fused = hasattr(kernels, "hc_pre") and kernels.hc_pre is not None


@functools.cache
def _load_compact_kpool_update_op() -> Callable[..., None] | None:
    """Load the NPU-only compact KPool updater without affecting other backends."""
    if torch_npu is None or importlib.util.find_spec("triton") is None:
        return None
    from xllm.python.kernels_npu.triton.kpool_compress import (
        update_compact_kpool_fast,
    )

    return update_compact_kpool_fast


def _compact_kpool_triton_query_len(
    query_lens: Sequence[int],
    rate: int,
) -> int | None:
    """The caller separately checks compact-tail capacity."""
    if not query_lens:
        return None
    query_len = int(query_lens[0])
    if rate <= 0 or query_len <= 0 or any(int(length) != query_len for length in query_lens[1:]):
        return None
    return query_len


# Per-chunk cap (bytes) for the indexer scores slab [B, sub, n_heads, n_pools]
# in Glm5NextIndexer.select_topk. The unchunked tensor grows with n_pools
# (= kv_len / index_kpool): 8 GiB at 32K context, 16 GiB at 64K — beyond the
# activation headroom and a hard device OOM mid prefill.
_SCORES_SLAB_CAP_BYTES = int(1.5 * 1024**3)
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.kda import KdaForgetGate
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.layers.linear import ColumnParallelLinear
from xllm.python.layers.moe_dp import dp_gather_tokens, reduce_and_scatter
from xllm.python.layers.qlinear import QLinear
from xllm.python.models.base import PyModelBase
from xllm.python.models.glm5_next_kpool import (
    append_causal_tail,
    read_pools,
    update_compressed_kpool,
)
from xllm.python.models.glm5_next_kpool import pooled_states as _kpool_pooled_states

# xllm Attention base — present in the real engine (full xllm.python package).
# Under the standalone stub-loader align path the package is not wired, so fall
# back to a stub nn.Module that accepts (and ignores) the same __init__ kwargs;
# the layers still work as plain modules there (the executor's
# isinstance(module, Attention) check is never reached on that path).
try:
    from xllm.python.layers.attention import Attention
except ImportError:  # pragma: no cover - stub-loader path

    class Attention(nn.Module):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            super().__init__()


# Paged pool cache: write-time incremental compression + direct pool read
# (see glm5_next_kpool.py).


class _UnweightedRMSNorm(nn.Module):
    """Unweighted RMSNorm (transformers Glm5NextTextUnweightedRMSNorm).

    Used inside the mHC input projection: no weight parameter, just rescale by
    the fp32 RMS then cast back (input_norm in the reference HyperConnection).
    The vendor npu_rms_norm needs a weight and this norm only runs on the
    non-fused-mHC fallback (fused hc_pre folds the mHC rsqrt in), so keep it
    pure-torch.
    """

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(input_dtype)


class _RMSNormGated(nn.Module):
    """RMSNorm + sigmoid gate (transformers Glm5NextRMSNormGated)."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Fused sigmoid-gated kernel (one launch over all rows). Must be the
        # sigmoid-gated kernel — never kernels.rms_norm_gated, which is SiLU.
        return kernels.rms_norm_sigmoid_gated(x, gate, self.weight, self.variance_epsilon)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Glm5NextConfig:
    """glm5_next architecture parameters (transformers schema)."""

    model_type: str = "glm5_next"
    hidden_size: int = 4096
    n_layers: int = 45
    n_heads: int = 64
    n_kv_heads: int = 64
    intermediate_size: int = 12288
    vocab_size: int = 154880
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1104096
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    # MLA
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    # KDA
    kda_num_heads: int = 64
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    linear_lower_bound: Optional[float] = -5.0
    swiglu_limit: float = 10.0
    # MoE
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 3
    # DSA / kPool indexer
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 1
    index_kpool_compress: bool = False
    index_kpool_always_select_tail: bool = False
    # C++ resolves this cache-layout capability from the actual model args.
    # None preserves standalone Python model construction compatibility.
    requires_kpool_tail_cache: Optional[bool] = None
    # mHC (multi-stream hyper-connection residual) — always on, per reference
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    # derived
    layer_types: list = field(default_factory=list)  # "linear_attention" / "deepseek_sparse_attention"
    mlp_layer_types: list = field(default_factory=list)  # "dense" / "sparse"
    indexer_types: list = field(default_factory=list)  # "full" / "shared"
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    moe_tp_size: Optional[int] = None
    moe_tp_rank: Optional[int] = None
    expert_parallel_degree: int = 0
    enable_mega_moe: bool = False
    enable_fused_mc2: bool = False

    def __post_init__(self) -> None:
        # Native EP callers supply the MoE group axes.
        if self.moe_tp_size is None:
            self.moe_tp_size = self.tp_size
        if self.moe_tp_rank is None:
            self.moe_tp_rank = self.tp_rank

    @classmethod
    def from_dict(cls, d: dict) -> Glm5NextConfig:
        # Multimodal full-weight configs nest text-model fields under
        # "text_config"; single-model configs are flat. Merge text_config into
        # the top level (without clobbering top-level overrides) so the flat
        # picks below work for both layouts. Mirrors JsonReader::resolve.
        tc = d.get("text_config")
        if isinstance(tc, dict):
            d = {**tc, **d}

        def pick(*keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        hidden = int(pick("hidden_size", default=4096))
        n_heads = int(pick("n_heads", "num_attention_heads", default=64))
        n_layers = int(pick("n_layers", "num_hidden_layers", default=45))
        first_k_dense = int(pick("first_k_dense_replace", default=3))
        requires_kpool_tail_cache = pick("requires_kpool_tail_cache", default=None)

        # KDA linear_attn_config (transformers stores it as a dict)
        lac = pick("linear_attn_config", default=None) or {}
        kda_heads = int(lac.get("num_heads", 64))
        kda_dim = int(lac.get("head_dim", 128))
        conv_k = int(lac.get("short_conv_kernel_size", 4))
        full_attn_layers = lac.get("full_attn_layers")
        if full_attn_layers is None:
            full_attn_layers = [i for i in range(n_layers) if i % 4 == 3]

        # Forget-gate lower-bound resolution — mirrors reference config __post_init__:
        # the dict key is ``gate_lower_bound`` (NOT ``lower_bound``); field default
        # is -5.0; if safe_gate (default True) and the bound is None, force -5.0.
        lower_bound = lac.get("gate_lower_bound", -5.0)
        if lac.get("safe_gate", True) and lower_bound is None:
            lower_bound = -5.0

        cfg = cls(
            model_type=str(pick("model_type", default="glm5_next")),
            hidden_size=hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=int(pick("n_kv_heads", "num_key_value_heads", default=n_heads)),
            intermediate_size=int(pick("intermediate_size", default=12288)),
            vocab_size=int(pick("vocab_size", default=154880)),
            rms_norm_eps=float(pick("rms_norm_eps", default=1e-5)),
            rope_theta=float(pick("rope_theta", default=10000.0)),
            max_position_embeddings=int(pick("max_position_embeddings", default=1104096)),
            hidden_act=str(pick("hidden_act", default="silu")),
            attention_bias=bool(pick("attention_bias", default=False)),
            tie_word_embeddings=bool(pick("tie_word_embeddings", default=False)),
            q_lora_rank=int(pick("q_lora_rank", default=1536)),
            kv_lora_rank=int(pick("kv_lora_rank", default=512)),
            qk_nope_head_dim=int(pick("qk_nope_head_dim", default=256)),
            qk_rope_head_dim=int(pick("qk_rope_head_dim", default=0)),
            v_head_dim=int(pick("v_head_dim", default=256)),
            kda_num_heads=kda_heads,
            kda_head_dim=kda_dim,
            short_conv_kernel_size=conv_k,
            linear_lower_bound=lower_bound,
            swiglu_limit=float(pick("swiglu_limit", default=10.0)) or 10.0,
            moe_intermediate_size=int(pick("moe_intermediate_size", default=2048)),
            n_routed_experts=int(pick("n_routed_experts", "num_local_experts", "num_experts", default=288)),
            n_shared_experts=int(pick("n_shared_experts", default=1)),
            num_experts_per_tok=int(pick("num_experts_per_tok", default=8)),
            n_group=int(pick("n_group", default=1)),
            topk_group=int(pick("topk_group", default=1)),
            routed_scaling_factor=float(pick("routed_scaling_factor", default=2.5)),
            norm_topk_prob=bool(pick("norm_topk_prob", default=True)),
            first_k_dense_replace=first_k_dense,
            index_n_heads=int(pick("index_n_heads", default=32)),
            index_head_dim=int(pick("index_head_dim", default=128)),
            index_topk=int(pick("index_topk", default=2048)),
            index_kpool=int(pick("index_kpool", default=1)),
            index_kpool_compress=bool(pick("index_kpool_compress", default=False)),
            index_kpool_always_select_tail=bool(pick("index_kpool_always_select_tail", default=False)),
            requires_kpool_tail_cache=None if requires_kpool_tail_cache is None else bool(requires_kpool_tail_cache),
            tp_size=int(pick("tp_size", default=1)),
            tp_rank=int(pick("tp_rank", default=0)),
            ep_size=int(pick("ep_size", default=1)),
            ep_rank=int(pick("ep_rank", default=0)),
            dp_size=int(pick("dp_size", default=1)),
            dp_rank=int(pick("dp_rank", default=0)),
            moe_tp_size=pick("moe_tp_size"),
            moe_tp_rank=pick("moe_tp_rank"),
            expert_parallel_degree=int(pick("expert_parallel_degree", default=0)),
            enable_mega_moe=bool(pick("enable_mega_moe", default=False)),
            enable_fused_mc2=bool(pick("enable_fused_mc2", default=False)),
            # mHC fields: ModelArgs may emit a 0 default (un-plumbed); treat 0
            # /None as unset and fall back to the real 300B defaults.
            hc_mult=(int(pick("hc_mult", default=4)) or 4),
            hc_eps=(float(pick("hc_eps", default=1e-6)) or 1e-6),
            hc_sinkhorn_iters=(int(pick("hc_sinkhorn_iters", default=20)) or 20),
        )
        cfg._resolve_schedules(full_attn_layers, d)
        cfg._validate_moe_parallelism()
        return cfg

    def _validate_moe_parallelism(self) -> None:
        if self.expert_parallel_degree not in (0, 1):
            raise ValueError("GLM-5.3-Flash supports ordinary EP level 1 only; expert_parallel_degree must be 0 or 1")
        if self.enable_mega_moe:
            raise ValueError("GLM-5.3-Flash ordinary EP does not support enable_mega_moe")
        if self.enable_fused_mc2:
            raise ValueError("GLM-5.3-Flash ordinary EP does not support enable_fused_mc2")
        for name in ("tp", "ep", "dp", "moe_tp"):
            size = getattr(self, f"{name}_size")
            rank = getattr(self, f"{name}_rank")
            if size <= 0 or not 0 <= rank < size:
                raise ValueError(f"invalid {name} parallel size/rank: {size}/{rank}")
        if self.n_routed_experts % self.ep_size:
            raise ValueError("n_routed_experts must be divisible by ep_size")
        if self.moe_intermediate_size % self.moe_tp_size:
            raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
        if self.ep_size == 1 and self.dp_size == 1:
            if (self.moe_tp_size, self.moe_tp_rank) != (self.tp_size, self.tp_rank):
                raise ValueError("TP-only MoE must use the attention TP size and rank")
        elif self.ep_size * self.moe_tp_size != self.dp_size * self.tp_size:
            raise ValueError("EP size times MoE-TP size must equal DP size times attention TP size")

    def _resolve_schedules(self, full_attn_layers: list, d: dict) -> None:
        n = self.n_layers
        lt = d.get("layer_types")
        if isinstance(lt, list) and lt:
            self.layer_types = list(lt)
        else:
            self.layer_types = [
                "deepseek_sparse_attention" if i in full_attn_layers else "linear_attention" for i in range(n)
            ]
        mlt = d.get("mlp_layer_types")
        if isinstance(mlt, list) and mlt:
            self.mlp_layer_types = list(mlt)
        else:
            n_dense = min(self.first_k_dense_replace, n)
            self.mlp_layer_types = ["dense"] * n_dense + ["sparse"] * (n - n_dense)
        it = d.get("indexer_types")
        if isinstance(it, list) and it:
            self.indexer_types = list(it)
        else:
            offset = int(d.get("index_skip_topk_offset", 1))
            freq = int(d.get("index_topk_freq", 1))
            self.indexer_types = [
                "full" if (max(i - offset + 1, 0) % max(freq, 1)) == 0 else "shared" for i in range(n)
            ]

    def is_dsa(self, layer_id: int) -> bool:
        return layer_id < len(self.layer_types) and self.layer_types[layer_id] == "deepseek_sparse_attention"

    def is_moe(self, layer_id: int) -> bool:
        if layer_id < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer_id] == "sparse"
        return layer_id >= self.first_k_dense_replace

    def indexer_shared(self, layer_id: int) -> bool:
        return layer_id < len(self.indexer_types) and self.indexer_types[layer_id] == "shared"


# ---------------------------------------------------------------------------
# KDA (Kimi Delta Attention) linear-attention layer
# ---------------------------------------------------------------------------
_KDA_FG_BMM_MAX_TOKENS = 32

_KDA_IN_PROJ = (
    ("q_proj", "qkv_dim", 0),
    ("k_proj", "qkv_dim", 0),
    ("v_proj", "qkv_dim", 0),
    ("b_proj", "num_heads_local", 0),
    ("forget_gate.f_a_proj", "head_dim", None),
    ("g_a_proj", "head_dim", None),
)


def _stable_pack(dst: torch.Tensor | None, packed: torch.Tensor) -> torch.Tensor:
    """Keep a packed weight buffer at a stable storage address across reloads.

    A captured decode ACL graph records the buffer's address, so a plain
    reallocation on every ``process_weights_after_loading`` (e.g. weight
    hot-reload) leaves the graph replaying stale weights. Mirror the W_UK/W_UV
    discipline: allocate once, then copy in place when the layout is unchanged.
    """
    if dst is None or dst.shape != packed.shape or dst.dtype != packed.dtype:
        return packed
    dst.copy_(packed)
    return dst


class Glm5NextKdaAttention(Attention):
    """KDA linear-attention layer (conv1d + delta-rule + gated norm + o_proj).

    Subclasses xllm ``Attention`` (reports num_heads/head_dim/scale for the
    executor's bookkeeping) but overrides forward to dispatch the conv1d +
    delta-rule through the attention backend
    (``NpuPagedAttentionBackend.execute_linear``), which owns the per-layer
    conv/ssm state slots (conv_cache: ``[num_slots, conv_state_len, conv_dim]``,
    ssm_cache: ``[num_slots, nh, k_hd, v_hd]`` fp32) and the per-sequence
    slot/cold-start view (``linear_state_indices`` / ``has_initial_state``).

    KDA math is dispatched through ``NpuPagedAttentionBackend`` to fused
    ``fla_npu`` operators.
    """

    def __init__(self, cfg: Glm5NextConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(
            num_heads=cfg.kda_num_heads,
            num_kv_heads=cfg.kda_num_heads,
            head_dim=cfg.kda_head_dim,
            scale=cfg.kda_head_dim**-0.5,
            sliding_window=0,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.hidden_size = cfg.hidden_size
        # Head-sharded TP (mirrors DSA). super().__init__ reports the FULL
        # kda_num_heads (the executor's C++ cache shape divides it by tp via
        # linear_num_key_heads/world_size — see kv_cache_shape.cpp), so the
        # framework conv/ssm slots are already this rank's head-subset
        # ([slots, conv_dim/tp, len] / [slots, nh/tp, k_hd, v_hd]). The model
        # therefore computes ONLY its local heads so mixed_qkv ([B, conv_dim/tp,
        # S]) matches the framework conv_state ([B, conv_dim/tp, len]) — without
        # this the cat at the conv window mismatches ([conv_dim/tp] vs
        # [conv_dim]) at tp>1. At tp==1 num_heads_local==num_heads (no-op), so
        # the standalone align path is unchanged.
        self.tp = cfg.tp_size
        self.num_heads = cfg.kda_num_heads  # full (reported to base / bookkeeping)
        self.num_heads_local = cfg.kda_num_heads // cfg.tp_size
        self.head_dim = cfg.kda_head_dim
        self.qkv_dim = self.head_dim * self.num_heads_local  # local per-rank
        self.conv_kernel_size = cfg.short_conv_kernel_size
        self.conv_dim = self.qkv_dim * 3  # local conv_dim = 3 * qkv_dim_local
        self.activation = cfg.hidden_act
        if self.activation not in ("identity", "silu"):
            raise ValueError(f"Unsupported KDA convolution activation: {self.activation}")
        self.eps = cfg.rms_norm_eps

        projection_sizes = tuple(getattr(self, size_attr) for _, size_attr, _ in _KDA_IN_PROJ)
        # Follow checkpoint row order; equal-sized f_a/g_a blocks must not be exchanged.
        self.input_projection_sizes = (sum(projection_sizes[:3]), projection_sizes[3], sum(projection_sizes[4:]))
        self.in_proj_qkvbfg_a = nn.Linear(self.hidden_size, sum(self.input_projection_sizes), bias=False)
        # conv1d: depthwise over the LOCAL conv_dim (groups=conv_dim_local); the
        # loader shards each of q/k/v_conv1d by head then cats so the channel
        # order [q_loc|k_loc|v_loc] matches mixed_qkv. fp32 in transformers.
        self.conv_weight = nn.Parameter(torch.empty(self.conv_dim, 1, self.conv_kernel_size, dtype=torch.float32))
        self.register_buffer("conv_weight_t", None, persistent=False)
        self.forget_gate = KdaForgetGate(cfg.kda_head_dim, self.num_heads_local, cfg.linear_lower_bound)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.register_buffer("_fg_b_weight", None, persistent=False)
        self.o_norm = _RMSNormGated(self.head_dim, self.eps, dtype, device)
        # o_proj: row-parallel + all_reduce. With KDA now head-sharded, each
        # rank's attention output is its head-subset's partial sum over
        # qkv_dim_local; o_proj ([hidden, qkv_dim_local]) produces a partial
        # hidden that must be all-reduced across ranks (mirrors DSA o_proj).
        self.o_proj = QLinear(
            self.qkv_dim,
            self.hidden_size,
            device=device,
            dtype=dtype,
            kind="static",
            row_parallel=True,
        )

    def process_weights_after_loading(self) -> None:
        conv_weight = self.conv_weight.squeeze(1).t().to(self.in_proj_qkvbfg_a.weight.dtype).contiguous()
        self.conv_weight_t = _stable_pack(self.conv_weight_t, conv_weight)
        packed = torch.stack((self.forget_gate.f_b_proj.weight.detach(), self.g_b_proj.weight.detach()))
        self._fg_b_weight = _stable_pack(self._fg_b_weight, packed)
        self.forget_gate.f_b_proj.weight.data = self._fg_b_weight[0]
        self.g_b_proj.weight.data = self._fg_b_weight[1]

    def _project_fg(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch the two distinct F/G inputs as in SGLang's batched linear."""
        hidden_shape = (*latents.shape[:2], self.num_heads_local, self.head_dim)
        # Keep batched projection to small matrices; larger batches retain the
        # separate GEMMs to avoid prefill and captured-graph regressions.
        if self._fg_b_weight is None or hidden_shape[0] * hidden_shape[1] > _KDA_FG_BMM_MAX_TOKENS:
            forget_latent, output_latent = latents.split(self.head_dim, dim=-1)
            return self.forget_gate.raw_projection(forget_latent), self.g_b_proj(output_latent).view(hidden_shape)
        projected = torch.bmm(latents.view(-1, 2, self.head_dim).transpose(0, 1), self._fg_b_weight.transpose(-1, -2))
        return projected[0].view(hidden_shape), projected[1].view(hidden_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        projected = self.in_proj_qkvbfg_a(hidden_states)
        mixed_qkv, beta_raw, fg_latents = projected.split(self.input_projection_sizes, dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        g_raw, gate = self._project_fg(fg_latents)
        beta = torch.sigmoid(beta_raw.float())

        # KDA conv1d + delta-rule + conv/ssm state is owned by the backend
        # (NpuPagedAttentionBackend.execute_linear). No self-contained fallback
        # — KDA must run inside the engine with a bound backend.
        ctx = get_forward_context_or_none()
        backend = ctx.attention_backend if ctx is not None else None
        if backend is None:
            raise RuntimeError(
                "Glm5NextKdaAttention requires an attention backend with execute_linear; run inside the engine."
            )
        cp_context = ctx.cp_context if ctx is not None else None
        if cp_context is not None:
            mixed_qkv = cp_merge_rows(mixed_qkv.transpose(1, 2).reshape(-1, self.conv_dim), cp_context)
            mixed_qkv = mixed_qkv.unsqueeze(0).transpose(1, 2).contiguous()
            g_raw = cp_merge_rows(
                g_raw.reshape(-1, self.num_heads_local, self.head_dim),
                cp_context,
            ).unsqueeze(0)
            beta = cp_merge_rows(beta.reshape(-1, self.num_heads_local), cp_context).unsqueeze(0)

        core_attn_out = backend.execute_linear(
            mixed_qkv,
            beta,
            self.layer_id,
            self.conv_weight_t,
            self.forget_gate,
            self.activation,
            raw_gate_proj=g_raw,
        )

        if cp_context is not None:
            core_attn_out = cp_shard_rows(
                core_attn_out.reshape(-1, self.num_heads_local, self.head_dim),
                cp_context,
            ).unsqueeze(0)
        output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
        # KDA is head-sharded: each rank's o_proj (row-parallel, input
        # qkv_dim_local) yields a partial hidden summed over its head-subset;
        # all-reduce across ranks to assemble the full hidden (mirrors DSA
        # o_proj). At tp==1 this is a no-op.
        o = self.o_proj(output)
        if self.tp > 1:
            distributed.all_reduce_(o)
        if cp_context is not None:
            o = cp_zero_fill_local_rows(o, cp_context, leading_seq_dim=True)
        return o


def _current_q_seq_lens(num_seqs: int, num_tokens: int) -> list[int]:
    """Per-sequence query lengths of the current forward's varlen batch.

    The engine flattens every batch to ``[1, T, ...]``; the real sequence
    boundaries live in the attention metadata's ``q_cu_seq_lens`` (cumulative
    offsets, same source the KDA layers use).
    """
    q_cu = get_forward_context().metadata.q_cu_seq_lens
    if q_cu is None:
        raise RuntimeError(
            f"varlen batch without q_cu_seq_lens: cannot split num_tokens={num_tokens} across num_seqs={num_seqs}"
        )
    ends = q_cu[1:].cpu().tolist()
    if len(ends) != num_seqs or ends[-1] != num_tokens:
        raise RuntimeError(
            f"q_cu_seq_lens does not describe this batch: ends={ends}, num_seqs={num_seqs}, num_tokens={num_tokens}"
        )
    return [ends[0]] + [ends[i] - ends[i - 1] for i in range(1, len(ends))]


# Forward-context cache key for the request-level KPool query spans shared by
# every full indexer layer of one forward.
_KPOOL_LAYOUT_CACHE_KEY = (__name__, "kpool_query_lens")
_CP_MERGED_POSITIONS_KEY = (__name__, "cp_merged_positions")
_CP_ATTENTION_MASK_KEY = (__name__, "cp_attention_mask")


def _span_lens(num_rows: int, num_tokens: int) -> list[int]:
    """Fallback request spans when no explicit query lengths are available."""
    if num_rows == 1:
        return [num_tokens]
    if num_rows == num_tokens:
        return [1] * num_tokens
    return _current_q_seq_lens(num_rows, num_tokens)


def _kpool_update_query_lens(ctx: KPoolMlaIndexContext, num_tokens: int) -> list[int]:
    """Resolve logical request spans for request-owned KPool tail updates."""
    kpool = ctx.kpool
    if kpool is not None and kpool.query_lens:
        query_lens = [int(length) for length in kpool.query_lens]
        if any(length <= 0 for length in query_lens):
            raise RuntimeError(f"kPool query spans must be positive: spans={query_lens}")
        if sum(query_lens) != num_tokens:
            raise RuntimeError(
                f"kPool query spans must cover the current token rows: spans={query_lens}, num_tokens={num_tokens}"
            )
        return query_lens
    if ctx.block_table is None:
        raise RuntimeError("kPool update requires a block table")
    return _span_lens(ctx.block_table.shape[0], num_tokens)


def _kpool_logical_rows(
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    query_lens: list[int],
    query_lens_device: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse token-expanded MTP metadata to one row per request."""
    num_requests = len(query_lens)
    num_tokens = sum(query_lens)
    starts = ends = None
    if block_table.shape[0] == num_requests:
        logical_block_table = block_table
    elif block_table.shape[0] >= num_tokens:
        starts, ends = _kpool_request_offsets(query_lens, query_lens_device, block_table.device)
        logical_block_table = block_table.index_select(0, starts)
    else:
        raise RuntimeError(
            "kPool block table does not cover every logical request or query row: "
            f"block_rows={block_table.shape[0]}, requests={num_requests}, query_rows={num_tokens}"
        )

    if kv_lens.numel() == num_requests:
        logical_kv_lens = kv_lens.reshape(-1)
    elif kv_lens.numel() >= num_tokens:
        if ends is None or ends.device != kv_lens.device:
            _, ends = _kpool_request_offsets(query_lens, query_lens_device, kv_lens.device)
        logical_kv_lens = kv_lens.reshape(-1).index_select(0, ends)
    else:
        raise RuntimeError(
            "kPool KV lengths do not cover every logical request or query row: "
            f"kv_rows={kv_lens.numel()}, requests={num_requests}, query_rows={num_tokens}"
        )
    return logical_block_table, logical_kv_lens


def _request_row_starts(query_lens: list[int]) -> list[int]:
    """First flat row of every request in a ``[1, T, ...]`` varlen batch."""
    return list(accumulate(query_lens, initial=0))[:-1]


def _kpool_request_offsets(
    query_lens: list[int],
    query_lens_device: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_requests = len(query_lens)
    if query_lens_device is not None:
        if query_lens_device.numel() != num_requests:
            raise RuntimeError("device KPool query lengths must match the logical request count")
        cumulative_lengths = F.pad(
            torch.cumsum(query_lens_device.reshape(-1), dim=0, dtype=torch.int32),
            (1, 0),
        )
        return (
            cumulative_lengths[:-1].to(torch.int64),
            cumulative_lengths[1:].to(torch.int64) - 1,
        )

    starts = torch.tensor(_request_row_starts(query_lens), dtype=torch.int64, device=device)
    return starts, starts + torch.tensor(query_lens, dtype=torch.int64, device=device) - 1


def _kpool_logical_state_indices(
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    query_lens: list[int],
    query_lens_device: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse token-expanded KPool state slots to one id per request."""
    num_requests = len(query_lens)
    num_tokens = sum(query_lens)

    def _collapse(indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == num_requests:
            return indices
        if indices.numel() < num_tokens:
            raise RuntimeError(
                "kPool state indices do not cover every logical request or query row: "
                f"state_rows={indices.numel()}, requests={num_requests}, query_rows={num_tokens}"
            )
        starts, _ = _kpool_request_offsets(query_lens, query_lens_device, indices.device)
        return indices.reshape(-1).index_select(0, starts)

    if read_indices is write_indices:
        logical_indices = _collapse(write_indices)
        return logical_indices, logical_indices
    return _collapse(read_indices), _collapse(write_indices)


def _kpool_valid_rows(attention_mask: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
    """Exclude padding and speculative placeholder rows from KPool updates."""
    if slot_mapping.numel() != attention_mask.numel():
        raise RuntimeError(
            "kPool slot mapping must contain one entry per token row: "
            f"slots={slot_mapping.numel()}, rows={attention_mask.numel()}"
        )
    return attention_mask.to(torch.bool) & (slot_mapping.reshape(attention_mask.shape) > 0)


# ---------------------------------------------------------------------------
# kPool DSA indexer (assembled from small ops, faithful to transformers)
# ---------------------------------------------------------------------------
class Glm5NextIndexer(nn.Module):
    """GLM KPool indexer with compressed-tail and packed fallback layouts."""

    def __init__(self, cfg: Glm5NextConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.topk = cfg.index_topk
        self.index_kpool = cfg.index_kpool
        self.index_kpool_compress = cfg.index_kpool_compress
        self.index_kpool_always_select_tail = cfg.index_kpool_always_select_tail
        self._uses_npu_compressed_tail = (
            cfg.requires_kpool_tail_cache
            if cfg.requires_kpool_tail_cache is not None
            else (
                device.type in ("npu", "privateuseone")
                and self.index_kpool > 1
                and self.index_kpool_compress
                and self.index_kpool_always_select_tail
            )
        )
        self._update_compact_kpool = _load_compact_kpool_update_op() if self._uses_npu_compressed_tail else None
        self.softmax_scale = self.head_dim**-0.5
        self.wq_b = nn.Linear(cfg.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False)
        self.register_buffer("_wk_weights_weight", None, persistent=False)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=dtype, device=device)
        )
        # raw Parameter (F.linear), matching transformers' state-dict key
        # ``indexer.index_kpool_compress_gate`` (no ``.weight`` suffix).
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, cfg.hidden_size, dtype=dtype, device=device)
        )
        self._compact_kpool_model_compatible = (
            self._update_compact_kpool is not None
            and dtype == torch.bfloat16
            and self.index_kpool_compress_ape.is_contiguous()
        )

    def process_weights_after_loading(self) -> None:
        prev_weight = self._wk_weights_weight
        self._wk_weights_weight = None
        if self.wk.weight.dtype != self.weights_proj.weight.dtype:
            return
        packed = torch.cat((self.wk.weight.detach(), self.weights_proj.weight.detach()), dim=0)
        self._wk_weights_weight = _stable_pack(prev_weight, packed)
        self.wk.weight.data = self._wk_weights_weight[: self.head_dim]
        self.weights_proj.weight.data = self._wk_weights_weight[self.head_dim :]

    def _project_key_weights(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._wk_weights_weight is None:
            return self.wk(hidden_states), self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype))
        projected = F.linear(hidden_states, self._wk_weights_weight)
        return projected.split((self.head_dim, self.n_heads), dim=-1)

    def get_pooled_states(self, packed_states: torch.Tensor, key_valid: torch.Tensor):
        pool_keys, pool_indices, pool_valid = _kpool_pooled_states(
            packed_states,
            key_valid,
            self.index_kpool_compress_ape,
            self.head_dim,
            self.index_kpool,
        )
        if in_acl_graph():
            # Graph branch (fixed shapes): keep ALL pools. The boolean
            # ``pool_keys[:, keep]`` filter below produces a data-dependent
            # output shape (aclnnNonzeroV2) that ACL graph capture cannot
            # record. select_topk masks invalid pools before selection and
            # fills their output slots with -1. Tied valid scores can select
            # different indices when the candidate width changes, so cached
            # and rebuilt decode pools must use the same uncompressed width.
            return pool_keys, pool_indices, pool_valid
        keep = pool_valid.any(0)
        return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]

    def get_packed_states(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, key: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-token indexer cache row ``[B, S, head_dim*2+1]`` =
        [k(128), gate(128), valid(1)]. NoPE: no RoPE applied to k."""
        k, gate_scores = self.get_kpool_states(hidden_states, key)
        valid_channel = attention_mask.to(k.dtype).unsqueeze(-1)
        return torch.cat([k, gate_scores, valid_channel], dim=-1)

    def get_kpool_states(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the raw K/gate rows retained only by the incomplete tail."""
        key = self.wk(hidden_states) if key is None else key
        return self.k_norm(key), F.linear(hidden_states, self.index_kpool_compress_gate)

    def _select_topk_fused_pa(
        self,
        query: torch.Tensor,
        projected_weights: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_cache: torch.Tensor,
        pool_block_table: torch.Tensor,
        query_positions: torch.Tensor | None = None,
        pool_query_block_table: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Run PoolKeyIndexer through the paged PA_BBND pool cache."""
        # A shared tail length cannot describe every prefill/verify query.
        if pool_cache.dtype != query.dtype:
            return None

        batch_size, query_len = query.shape[:2]
        if query_len == 1:
            fused_query = query
            fused_weights = projected_weights
            token_count = kv_seq_lens.reshape(-1).to(torch.int32)
            fused_block_table = pool_block_table
        else:
            if query_positions is None or query_positions.shape != (batch_size, query_len):
                return None
            # PoolKeyIndexer only receives one pool-tail length per batch. An
            # MTP verify spans several causal positions, so treating its rows
            # as one S1 sequence loses the historical partial pool whenever the
            # verify window crosses a pool boundary. Present every query row as
            # an independent one-token batch with its own visible KV length.
            fused_query = query.reshape(batch_size * query_len, 1, *query.shape[2:])
            fused_weights = projected_weights.reshape(batch_size * query_len, 1, *projected_weights.shape[2:])
            token_count = (query_positions.reshape(-1) + 1).to(torch.int32)
            token_count = torch.where(
                attention_mask.reshape(-1),
                token_count,
                torch.zeros_like(token_count),
            )
            if pool_query_block_table is not None and pool_query_block_table.shape[0] >= batch_size * query_len:
                fused_block_table = pool_query_block_table[: batch_size * query_len]
            else:
                fused_block_table = (
                    pool_block_table[:, None, :].expand(batch_size, query_len, -1).reshape(batch_size * query_len, -1)
                )
        # PA_BBND addresses complete pooled keys and uses pool_tail_k for the
        # remaining tokens in the current tail pool.
        indices, _ = kernels.pool_key_indexer(
            fused_query,
            pool_cache,
            fused_weights,
            torch.remainder(token_count, self.index_kpool).contiguous(),
            self.topk,
            self.index_kpool,
            return_value=False,
            actual_seq_k=torch.div(token_count, self.index_kpool, rounding_mode="floor"),
            block_table=fused_block_table.to(dtype=torch.int32).contiguous(),
            mask_mode=kernels.POOL_KEY_INDEXER_MASK_MODE,
            layout_k="PA_BBND",
        )

        output_width = self.topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
        indices = indices.reshape(batch_size, query_len, -1)[..., :output_width]
        return indices.masked_fill(~attention_mask[..., None], -1).long()

    def select_topk(
        self,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_len: int,
        current_length: int,
        packed_states: torch.Tensor | None = None,
        pool_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        key_valid: torch.Tensor | None = None,
        pool_block_table: torch.Tensor | None = None,
        pool_cache: torch.Tensor | None = None,
        kv_seq_lens: torch.Tensor | None = None,
        projected_weights: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        pool_query_block_table: torch.Tensor | None = None,
        append_unscored_tail: bool = False,
    ) -> torch.Tensor:
        """Top-k pool selection over the FULL packed index history.

        ``packed_states`` is ``[B, kv_len, head_dim*2+1]`` (k, gate, valid) for
        every kv token accumulated so far (the reference keeps this in the
        indexer cache and pools over the whole of it each step); ``q_resid`` /
        ``hidden_states`` are the CURRENT query tokens ``[B, S_q, ...]``.
        Alternatively, pass ``pool_data``/``key_valid`` from the paged pool
        cache (read_pools) to skip both the dense gather and the per-step
        re-pooling. Paged decode can pass ``kv_seq_lens`` instead of a dense
        ``key_valid`` mask; only the non-fused fallback materializes that mask,
        while ``query_positions`` preserves exact causal visibility for packed
        or variable-length query rows.
        Returns absolute kv-position top-k indices
        ``[B, S_q, topk]`` (int64, -1 = invalid), matching the reference
        indexer output.
        """
        batch_size, seq_len = q_resid.shape[:2]
        device = q_resid.device

        q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)

        raw_key_cache = packed_states is not None and packed_states.shape[-1] == self.head_dim
        if raw_key_cache and self.index_kpool != 1:
            raise ValueError("index_kpool_compress=false requires index_kpool=1")

        if pool_data is not None:
            # Pool-cache path: pools were compressed at write time; per-pool
            # visibility is derived later (at gather time) from key_valid,
            # mirroring gather_index_history's row_valid.
            pool_keys, pool_indices, pool_valid = pool_data
        else:
            if key_valid is None and kv_seq_lens is None:
                if raw_key_cache:
                    key_valid = torch.ones(
                        batch_size,
                        packed_states.shape[1],
                        dtype=torch.bool,
                        device=device,
                    )
                else:
                    key_valid = packed_states[..., -1].gt(0)

        weights = (
            self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype))
            if projected_weights is None
            else projected_weights
        )
        if (
            self.index_kpool_compress
            and q.device.type == "npu"
            and kernels.supports_pool_key_indexer(self.head_dim, self.topk, self.index_kpool)
            and pool_cache is not None
            and pool_block_table is not None
        ):
            scaled_weights = (weights * (self.n_heads**-0.5)).to(q.dtype)
            fused_indices = self._select_topk_fused_pa(
                q,
                scaled_weights,
                kv_seq_lens if kv_seq_lens is not None else key_valid.sum(-1, dtype=torch.int32),
                attention_mask,
                pool_cache,
                pool_block_table,
                query_positions,
                pool_query_block_table,
            )
            if fused_indices is not None:
                return fused_indices
        if key_valid is None:
            # kv_len is padded to a whole number of pools by the decode caller;
            # a narrower mask would misalign the gather at pool_start below and
            # silently drop the tail pool's validity bit.
            assert kv_len % self.index_kpool == 0, (
                f"kv_len {kv_len} must be a multiple of index_kpool {self.index_kpool}"
            )
            # This dense [B, kv_len] mask scales with kv_len and is rebuilt each
            # step; only decode (seq_len=1) reaches here with kv_seq_lens, so it
            # stays cheap. Long-kv prefill takes the fused / pool_data paths above
            # rather than materializing this mask.
            key_valid = torch.arange(kv_len, device=device)[None] < kv_seq_lens.reshape(-1, 1)
        # Resolve query positions only after the fused decode fast path. This
        # keeps graph decode from materializing any history-sized helper tensor.
        if query_positions is None:
            q_pos = current_length - seq_len + torch.arange(seq_len, device=device)
            q_pos = q_pos.unsqueeze(0).expand(batch_size, -1)
        else:
            q_pos = query_positions.to(device=device, dtype=torch.int64)
            if q_pos.shape != (batch_size, seq_len):
                raise ValueError(
                    "query_positions must match the logical query shape: "
                    f"positions={tuple(q_pos.shape)}, queries={(batch_size, seq_len)}"
                )
        if pool_data is None:
            if pool_cache is not None and pool_block_table is not None:
                kv_lens = kv_seq_lens if kv_seq_lens is not None else key_valid.sum(-1, dtype=torch.int64)
                n_pools = (kv_len + self.index_kpool - 1) // self.index_kpool
                pool_keys, pool_indices, pool_valid = read_pools(
                    pool_cache,
                    pool_block_table,
                    kv_lens,
                    n_pools,
                    self.index_kpool,
                )
            else:
                if raw_key_cache:
                    pool_keys = packed_states
                    if pool_keys.dim() == 4:
                        pool_keys = pool_keys.squeeze(-2)
                    pool_indices = (
                        torch.arange(
                            pool_keys.shape[1],
                            device=device,
                        )
                        .expand(batch_size, -1)
                        .unsqueeze(-1)
                    )
                    pool_valid = key_valid
                else:
                    pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states, key_valid)
        weights = weights.float().mul_(self.n_heads**-0.5)
        # Query-dim sub-chunking: each query row's logits are
        # q[row] @ pool_keys.T and its pool score a head-weighted sum of those
        # logits — independent of every other row — so slicing the query dim
        # is bit-exact while keeping each transient scores slab under
        # _SCORES_SLAB_CAP_BYTES. Without it the full [B, S, n_heads, n_pools]
        # fp32 scores tensor grows linearly with n_pools (= kv_len /
        # index_kpool) and OOMs mid prefill. When the full tensor already fits
        # (decode: S=1; short prefills) sub == seq_len and the loop degenerates
        # to the original single matmul.
        n_pools = pool_keys.shape[1]
        pool_keys_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        q_f = q.float()
        slab_elems = self.n_heads * n_pools
        sub = seq_len if slab_elems == 0 else max(1, min(seq_len, _SCORES_SLAB_CAP_BYTES // (slab_elems * 4)))
        pool_scores = torch.empty(batch_size, seq_len, n_pools, dtype=torch.float32, device=device)
        for start in range(0, seq_len, sub):
            end = min(start + sub, seq_len)
            slab = torch.matmul(q_f[:, start:end], pool_keys_t)
            # In-place relu/scale: bit-identical elementwise math, but avoids a
            # second [B, sub, n_heads, n_pools] fp32 allocation.
            slab = torch.relu_(slab)
            slab *= self.softmax_scale
            pool_scores[:, start:end] = torch.einsum("bshp,bsh->bsp", slab, weights[:, start:end])
            del slab

        if pool_keys.shape[1] != 0:
            # The NPU compressed-tail layout scores only complete pools and
            # appends the current incomplete pool separately. Existing packed
            # and non-tail layouts retain pool-start visibility.
            pool_anchor = pool_indices[..., -1 if append_unscored_tail else 0].clamp(0, kv_len - 1)
            pool_key_valid = key_valid.gather(1, pool_anchor)  # [B, n_pools]
            pool_visible = (
                (pool_anchor[:, None, :] <= q_pos[:, :, None]) & pool_key_valid[:, None, :] & attention_mask[:, :, None]
            )  # [B, seq_len, n_pools]
            candidate_valid = (pool_visible & pool_valid[:, None]).to(torch.bool)
            pool_scores.masked_fill_(~candidate_valid, torch.finfo(pool_scores.dtype).min)
        else:
            candidate_valid = pool_valid[:, None].expand(batch_size, seq_len, -1).to(torch.bool)

        group_budget = self.topk // self.index_kpool
        select_k = min(group_budget, pool_scores.shape[-1])
        if select_k == 0:
            topk_indices = torch.empty(batch_size, seq_len, 0, dtype=torch.long, device=device)
        else:
            selected = pool_scores.topk(select_k, dim=-1).indices
            selected_valid = candidate_valid.gather(-1, selected)
            # AICore-native gather replaces fancy indexing
            # pool_indices[batch_pool_idx, selected] which falls back to
            # aclnnIndex on AI_CPU. pool_indices is [B, n_pools, rate]; index
            # dim=1 by selected [B, S_q, select_k] -> [B, S_q, select_k, rate].
            # Flatten (n_pools*rate) rows and gather, matching the
            # get_pooled_states pattern.
            rate = self.index_kpool
            rate_off = torch.arange(rate, device=device)
            sel_flat = (selected[..., None] * rate + rate_off).reshape(batch_size, -1)
            selected_indices = (
                pool_indices.reshape(batch_size, -1).gather(1, sel_flat).reshape(batch_size, seq_len, select_k, rate)
            )
            topk_indices = selected_indices.flatten(-2)
            topk_indices.masked_fill_(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2),
                -1,
            )
            # Keep the per-position filter as a defensive guard for invalid or
            # externally supplied pool metadata.
            safe_pos = topk_indices.clamp(0, kv_len - 1)  # [B, seq_len, select_k*rate]
            # Per-position causal filter (same bit-exact expression as
            # pool_visible above), evaluated directly at the selected kv
            # positions instead of gathering the dense [B, seq_len, kv_len] mask.
            flat = safe_pos.reshape(batch_size, -1)
            selected_key_valid = key_valid.gather(1, flat).reshape_as(safe_pos)
            pos_visible = (safe_pos <= q_pos[:, :, None]) & selected_key_valid & attention_mask[:, :, None]
            topk_indices.masked_fill_(~pos_visible, -1)

        if append_unscored_tail:
            topk_indices = append_causal_tail(topk_indices, q_pos, self.index_kpool)
        output_width = self.topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
        if topk_indices.shape[-1] < output_width:
            topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        topk_indices = topk_indices[..., :output_width]
        topk_indices.masked_fill_(~attention_mask[..., None], -1)
        return topk_indices.long()

    def select_qli(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        ctx: MlaIndexContext,
        layer: Attention,
        backend,
    ) -> torch.Tensor:
        """Update the selected KPool layout and return SFA token indices."""
        batch_size, seq_len = hidden_states.shape[:2]
        num_tokens = batch_size * seq_len
        key, weights = self._project_key_weights(hidden_states)
        if ctx.block_table is None:
            raise RuntimeError("GLM-5.3-Flash kPool requires a block table")

        kpool = ctx.kpool if isinstance(ctx, KPoolMlaIndexContext) else None
        if self._uses_npu_compressed_tail and kpool is None:
            raise TypeError("compressed KPool requires KPoolMlaIndexContext from the attention backend")
        uses_compressed_tail = self._uses_npu_compressed_tail and is_compact_kpool_tail(
            kpool.tail if kpool is not None else None
        )
        # Request-level KPool layout (query spans, logical rows) is identical
        # for every indexer layer of this forward; compute it once in the
        # forward context instead of per layer.
        forward_cache = get_forward_context_or_none()
        kpool_query_lens = None
        if uses_compressed_tail:
            kpool_query_lens = (
                None if forward_cache is None else forward_cache.layer_shared_cache.get(_KPOOL_LAYOUT_CACHE_KEY)
            )
            if kpool_query_lens is None:
                kpool_query_lens = _kpool_update_query_lens(ctx, num_tokens)
                if forward_cache is not None:
                    forward_cache.layer_shared_cache[_KPOOL_LAYOUT_CACHE_KEY] = kpool_query_lens
        query_lens_device = kpool.query_lens_device if kpool is not None else None
        if query_lens_device is not None and (
            query_lens_device.device != hidden_states.device
            or query_lens_device.dtype != torch.int32
            or query_lens_device.numel() != len(kpool_query_lens or ())
        ):
            query_lens_device = None
        if uses_compressed_tail and in_acl_graph() and query_lens_device is None:
            raise RuntimeError("ACL graph compressed KPool requires device query lengths matching its request spans")
        kpool_block_table = ctx.block_table
        pool_query_block_table = kpool_block_table
        kpool_kv_lens = ctx.actual_seq_kv.reshape(-1)
        if kpool_query_lens is not None:
            kpool_block_table, kpool_kv_lens = _kpool_logical_rows(
                kpool_block_table,
                kpool_kv_lens,
                kpool_query_lens,
                query_lens_device,
            )
        if self.index_kpool_compress:
            raw_k, gate_scores = self.get_kpool_states(hidden_states, key)
            if uses_compressed_tail:
                if kpool is None or kpool.tail_read_indices is None or kpool.tail_write_indices is None:
                    raise RuntimeError("compressed kPool requires linear-state read/write indices")
                kpool_tail_read_indices, kpool_tail_write_indices = _kpool_logical_state_indices(
                    kpool.tail_read_indices,
                    kpool.tail_write_indices,
                    kpool_query_lens,
                    query_lens_device,
                )
                kpool_valid_rows = _kpool_valid_rows(attention_mask, ctx.slot_mapping)
                triton_query_len = _compact_kpool_triton_query_len(kpool_query_lens, self.index_kpool)
                # Decode/spec-verify resolves read/write to the same tensor. A
                # distinct prefix-restore destination needs the torch path's
                # explicit old-tail snapshot and copy semantics.
                use_triton_update = (
                    self._compact_kpool_model_compatible
                    and kpool.triton_compatible
                    and triton_query_len is not None
                    and kpool_tail_read_indices is kpool_tail_write_indices
                    and kpool.tail.shape[2] >= self.index_kpool + triton_query_len - 1
                )
                if use_triton_update:
                    self._update_compact_kpool(
                        raw_k.contiguous(),
                        gate_scores.contiguous(),
                        kpool_valid_rows.contiguous(),
                        positions.contiguous(),
                        ctx.index_cache,
                        kpool.tail,
                        kpool_tail_write_indices.contiguous(),
                        kpool_block_table.contiguous(),
                        triton_query_len,
                        self.index_kpool_compress_ape,
                        self.index_kpool,
                    )
                else:
                    update_compressed_kpool(
                        raw_k,
                        gate_scores,
                        kpool_valid_rows,
                        positions,
                        ctx.index_cache,
                        kpool.tail,
                        kpool_tail_read_indices,
                        kpool_tail_write_indices,
                        kpool_block_table,
                        kpool_query_lens,
                        self.index_kpool_compress_ape,
                        self.index_kpool,
                        graph_mode=in_acl_graph(),
                    )
                pool_cache = ctx.index_cache
                packed = None
            else:
                pool_cache = None
                packed = torch.cat(
                    [
                        raw_k,
                        gate_scores,
                        attention_mask.to(raw_k.dtype).unsqueeze(-1),
                    ],
                    dim=-1,
                )
        else:
            pool_cache = None
            # The non-compressed cache stores one raw K per token. This is the
            # original small-operator path (index_kpool=1), so no gate/valid
            # channels are written to the narrower cache.
            packed = self.k_norm(key).view(hidden_states.shape[0], hidden_states.shape[1], -1, self.head_dim).squeeze(2)
        if packed is not None and ctx.index_cache is not None and ctx.slot_mapping is not None:
            # kPool index cache is unquantized (no scale side-channel).
            ctx.update_index_cache(packed.reshape(num_tokens, -1), None)

        packed_history = None
        if pool_cache is not None:
            num_seqs = len(kpool_query_lens)
        else:
            packed_history = backend.gather_index_history(layer)
            num_seqs = packed_history.shape[0]
        if kpool_query_lens is not None:
            q_lens = kpool_query_lens
        else:
            q_lens = _span_lens(num_seqs, num_tokens)
        max_q = max(q_lens)
        is_varlen = max_q * num_seqs != num_tokens
        if not is_varlen:
            # Uniform per-sequence lengths (single sequence, packed
            # one-token decode, equal-length batch): a plain view is exact
            # and allocation-free — the decode hot path stays untouched.
            qr_bsd = qr.view(num_seqs, max_q, -1)
            hidden_bsd = hidden_states.view(num_seqs, max_q, -1)
            mask_bsd = attention_mask.view(num_seqs, max_q)
            positions_bsd = positions.reshape(num_seqs, max_q)
            weights_bsd = weights.view(num_seqs, max_q, self.n_heads)
        else:
            # Varlen batch (unequal prompts prefilled together): the engine
            # flattens the batch to [1, T, D]; scatter tokens to a padded
            # [num_seqs, max_q, ...] layout by the real sequence boundaries
            # and mask the pad rows; the top-k rows are gathered back to the
            # flat [T, ...] order below.
            starts_t = torch.tensor(_request_row_starts(q_lens), device=qr.device, dtype=torch.int64)
            lens_t = torch.tensor(q_lens, device=qr.device, dtype=torch.int64)
            q_offsets = torch.arange(max_q, device=qr.device, dtype=torch.int64)
            src = starts_t[:, None] + q_offsets[None, :]
            valid = q_offsets[None, :] < lens_t[:, None]
            src_flat = src.clamp(max=num_tokens - 1).reshape(-1)
            qr_bsd = qr.index_select(0, src_flat).view(num_seqs, max_q, -1)
            hidden_bsd = hidden_states.reshape(num_tokens, -1).index_select(0, src_flat).view(num_seqs, max_q, -1)
            mask_bsd = attention_mask.reshape(-1).index_select(0, src_flat).view(num_seqs, max_q) & valid
            positions_bsd = positions.reshape(-1).index_select(0, src_flat).view(num_seqs, max_q)
            weights_bsd = weights.reshape(num_tokens, self.n_heads).index_select(0, src_flat).view(num_seqs, max_q, -1)

        max_kv_cap = getattr(backend, "graph_index_history_max_kv", None)
        if in_acl_graph() and max_kv_cap is not None:
            token_page_size = (
                pool_cache.shape[1] * self.index_kpool if pool_cache is not None else ctx.index_cache.shape[1]
            )
            kv_len = min(ctx.block_table.shape[1] * token_page_size, max_kv_cap)
        elif pool_cache is not None:
            kv_lens_host = ctx.actual_seq_kv_host
            if kv_lens_host is not None and len(kv_lens_host) == kpool_kv_lens.numel():
                kv_len = max(max(int(length) for length in kv_lens_host), self.index_kpool)
            else:
                kv_len = ctx.block_table.shape[1] * pool_cache.shape[1] * self.index_kpool
            kv_len = ((kv_len + self.index_kpool - 1) // self.index_kpool) * self.index_kpool
        else:
            kv_len = packed_history.shape[1]
        logical_kv_lens = kpool_kv_lens if pool_cache is not None else ctx.actual_seq_kv[:num_seqs].to(torch.int64)
        if pool_cache is not None:
            key_valid = None
            kv_seq_lens = logical_kv_lens.clamp(max=kv_len)
        else:
            key_valid = torch.arange(kv_len, device=packed_history.device)[None, :] < logical_kv_lens[:, None]
            kv_seq_lens = None
        topk_indices = self.select_topk(
            qr_bsd,
            hidden_bsd,
            mask_bsd,
            kv_len=kv_len,
            current_length=kv_len,
            packed_states=packed_history,
            projected_weights=weights_bsd,
            key_valid=key_valid,
            kv_seq_lens=kv_seq_lens,
            pool_cache=pool_cache,
            pool_block_table=kpool_block_table,
            query_positions=positions_bsd,
            pool_query_block_table=pool_query_block_table,
            append_unscored_tail=uses_compressed_tail and self.index_kpool_always_select_tail,
        )
        if is_varlen:
            topk_indices = topk_indices[valid]
        return topk_indices.reshape(num_tokens, 1, -1).to(torch.int32)


# ---------------------------------------------------------------------------
# MLA (absorbed, NoPE) + DSA sparse attention via the SFA op
# ---------------------------------------------------------------------------
class Glm5NextMlaAttention(Attention):
    """Absorbed MLA (NoPE, qk_rope=0) + DSA sparse attention via SFA op.

    Mirrors ``DeepseekV3MlaAttention``'s forward: q_latent = bmm(q_nope,
    W_UK); k_latent into the paged nope cache; SFA over the kPool topk;
    v_full = bmm(attn_out, W_UV). The only divergence is the indexer: kPool
    stores completed compressed pools in the index cache and keeps the current
    incomplete pool in request state, with packed INDEX as the compatibility
    fallback. NoPE uses rope_dim=0 and q_pe/k_pe=None.
    """

    # Marker so the Python executor can identify GLM-Next DSA layers without
    # importing this module (the import would pull KDA kernel transitive deps
    # and fail on builds without them).
    is_glm_next_mla: bool = True
    supports_cp_nope_prefill: bool = True
    mla_index_context_builder = KPoolMlaIndexContextBuilder

    def __init__(self, cfg: Glm5NextConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(
            num_heads=cfg.n_heads,
            num_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
            scale=(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5,
            sliding_window=0,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.hidden_size = cfg.hidden_size
        tp = self.cfg.tp_size
        assert self.cfg.n_heads % tp == 0, f"n_heads {self.cfg.n_heads} not divisible by tp_size {tp}"
        num_heads = self.cfg.n_heads // tp  # per-rank head count
        self.num_heads_local = num_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.eps = cfg.rms_norm_eps
        shared = cfg.indexer_shared(layer_id)
        self.indexer = None if shared else Glm5NextIndexer(cfg, layer_id, dtype, device)

        dev, dt = device, dtype
        # q_a_proj: replicated (hidden -> q_lora)
        self.q_a_proj = QLinear(
            self.hidden_size, self.q_lora_rank, device=dev, dtype=dt, kind="static", bias=cfg.attention_bias
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, self.eps, dtype, device)
        # q_b_proj: column-parallel (out = num_heads_local * (qk_nope + qk_rope))
        self.q_b_proj = QLinear(self.q_lora_rank, num_heads * self.qk_head_dim, device=dev, dtype=dt, kind="static")
        # kv_a: replicated (latent + rope)
        self.kv_a_proj_with_mqa = QLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            device=dev,
            dtype=dt,
            kind="static",
            bias=cfg.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, self.eps, dtype, device)
        self.register_buffer("_qkv_a_weight", None, persistent=False)
        self.register_buffer("_qkv_a_bias", None, persistent=False)
        # kv_b: column-parallel fp (absorbed split reads .weight; stays fp, see
        # design §3)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            tp,
            dtype=dt,
            device=dev,
        )
        # o_proj: row-parallel + all_reduce (out = hidden_size)
        self.o_proj = QLinear(
            num_heads * self.v_head_dim,
            self.hidden_size,
            device=dev,
            dtype=dt,
            kind="static",
            row_parallel=True,
            bias=cfg.attention_bias,
        )
        # Absorbed-MLA: kv_b_proj weight = [num_heads*(qk_nope+v_hd), kv_lora].
        # Split into W_UK [H, qk_nope, kv_lora] and W_UV [H, kv_lora, v_hd]
        # (W_UV stored transposed, matching deepseek_v32 / glm5_2: bmm with the
        # [H, T, kv_lora] attn_out yields [H, T, v_hd]). Dead until Task 7
        # routes DSA through backend.execute_mla.
        self.register_buffer(
            "W_UK",
            torch.zeros(
                num_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "W_UV",
            torch.zeros(
                num_heads,
                self.kv_lora_rank,
                self.v_head_dim,
                dtype=dtype,
                device=device,
            ),
            persistent=False,
        )

    def process_weights_after_loading(self) -> None:
        prev_weight, prev_bias = self._qkv_a_weight, self._qkv_a_bias
        self._qkv_a_weight = None
        self._qkv_a_bias = None
        if self.q_a_proj.use_w8a8 is False and self.kv_a_proj_with_mqa.use_w8a8 is False:
            packed = torch.cat((self.q_a_proj.weight.detach(), self.kv_a_proj_with_mqa.weight.detach()), dim=0)
            self._qkv_a_weight = _stable_pack(prev_weight, packed)
            self.q_a_proj.weight.data = self._qkv_a_weight[: self.q_lora_rank]
            self.kv_a_proj_with_mqa.weight.data = self._qkv_a_weight[self.q_lora_rank :]
            if self.cfg.attention_bias:
                packed_bias = torch.cat((self.q_a_proj.bias.detach(), self.kv_a_proj_with_mqa.bias.detach()), dim=0)
                self._qkv_a_bias = _stable_pack(prev_bias, packed_bias)
                self.q_a_proj.bias.data = self._qkv_a_bias[: self.q_lora_rank]
                self.kv_a_proj_with_mqa.bias.data = self._qkv_a_bias[self.q_lora_rank :]
        # Split kv_b_proj.weight into absorbed W_UK / W_UV (mirrors
        # deepseek_v32.process_weights_after_loading split). glm5_next is NoPE
        # (qk_rope_head_dim=0), so qk_head_dim == qk_nope_head_dim.
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

    def _project_qkv_a(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse the shared-input Q/KV-A linear layout used by vLLM-Ascend."""
        # W8A8 can't fuse: each QLinear owns a per-tensor act scale and a repacked
        # int8 weight, so only the float path cats into the single GEMM below.
        if self._qkv_a_weight is None:
            return self.q_a_proj(hidden), self.kv_a_proj_with_mqa(hidden)
        projected = F.linear(hidden, self._qkv_a_weight, self._qkv_a_bias)
        return projected.split((self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Absorbed MLA forward (mirrors DeepseekV3MlaAttention.forward).

        NoPE (qk_rope_head_dim=0): q_pe/k_pe are None and rope is skipped. The
        kPool indexer updates the selected cache layout, runs ``select_topk``,
        and adapts the result to the SFA op's ``sparse_indices`` contract.
        """
        forward_context = get_forward_context()
        cp_context = getattr(forward_context, "cp_context", None)
        if cp_context is not None:
            hidden_states = cp_merge_rows(
                hidden_states.reshape(-1, self.hidden_size),
                cp_context,
            ).unsqueeze(0)
            # Merged positions and the all-ones mask are identical for every
            # MLA layer of one forward; cache them to avoid a per-layer
            # all-gather over the CP group.
            merged_positions = cast(
                "torch.Tensor | None", forward_context.layer_shared_cache.get(_CP_MERGED_POSITIONS_KEY)
            )
            if merged_positions is None:
                merged_positions = cp_merge_rows(
                    position_ids.reshape(-1, 1),
                    cp_context,
                ).reshape(1, -1)
                forward_context.layer_shared_cache[_CP_MERGED_POSITIONS_KEY] = merged_positions
            position_ids = merged_positions
            attention_mask = forward_context.layer_shared_cache.get(_CP_ATTENTION_MASK_KEY)
            if attention_mask is None:
                attention_mask = torch.ones(
                    1,
                    position_ids.shape[1],
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                forward_context.layer_shared_cache[_CP_ATTENTION_MASK_KEY] = attention_mask
            # Only shared-indexer layers consume top-k rows produced on the
            # preceding CP-sharded layer; full indexers compute their own rows.
            if self.indexer is None and prev_topk_indices is not None:
                prev_topk_indices = cp_merge_rows(
                    prev_topk_indices.reshape(prev_topk_indices.shape[0], -1),
                    cp_context,
                )

        num_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        hidden = hidden_states.view(num_tokens, -1)
        q_a, kv = self._project_qkv_a(hidden)
        q_c = self.q_a_layernorm(q_a)
        backend = get_forward_context().attention_backend
        topk = None
        if self.indexer is not None:
            ctx = backend.mla_index_context(self)
            topk = self.indexer.select_qli(hidden_states, q_c, position_ids, attention_mask, ctx, self, backend)
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk = prev_topk_indices.reshape(num_tokens, 1, -1).to(torch.int32)

        q = self.q_b_proj(q_c).view(num_tokens, self.num_heads_local, self.qk_head_dim)
        q_nope = q.narrow(-1, 0, self.qk_nope_head_dim)
        q_latent = torch.bmm(q_nope.transpose(0, 1), self.W_UK).transpose(0, 1)

        k_latent_raw = kv.narrow(-1, 0, self.kv_lora_rank)
        k_latent = self.kv_a_layernorm(k_latent_raw)
        k_latent_3d = k_latent.view(num_tokens, 1, self.kv_lora_rank)

        attn_out = backend.execute_mla(q_latent, None, k_latent_3d, None, self, topk=topk)
        if cp_context is not None:
            attn_out = cp_shard_rows(
                attn_out.reshape(num_tokens, self.num_heads_local, self.kv_lora_rank),
                cp_context,
            )
        v_full = torch.bmm(attn_out.transpose(0, 1), self.W_UV).transpose(0, 1)
        v_full = v_full.reshape(attn_out.shape[0], self.num_heads_local * self.v_head_dim)
        o = self.o_proj(v_full)
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(o)
        if cp_context is not None:
            o = cp_zero_fill_local_rows(o, cp_context, leading_seq_dim=False)
            topk = cp_shard_rows(topk.reshape(num_tokens, -1), cp_context).view(cp_context.total_local, 1, -1)
        return o, topk


# ---------------------------------------------------------------------------
# MLP (dense + MoE)
# ---------------------------------------------------------------------------
class Glm5NextMLP(nn.Module):
    def __init__(
        self,
        cfg: Glm5NextConfig,
        intermediate_size: int,
        dtype: torch.dtype,
        device: torch.device,
        skip_tp_reduce: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.skip_tp_reduce = skip_tp_reduce
        self.swiglu_limit = cfg.swiglu_limit
        # TP: column-parallel SwiGLU (gate_up sharded on the intermediate dim)
        # + row-parallel down_proj (sharded on the input dim, summed via
        # all_reduce_ in forward). At tp==1 inter_local == intermediate_size so
        # the fp graph is byte-identical to the previous nn.Linear trio.
        tp = cfg.tp_size
        inter_local = intermediate_size // tp
        self.gate_up_proj = QLinear(
            cfg.hidden_size,
            2 * inter_local,
            device=device,
            dtype=dtype,
            kind="dynamic",
        )
        self.down_proj = QLinear(
            inter_local,
            cfg.hidden_size,
            device=device,
            dtype=dtype,
            kind="dynamic",
            row_parallel=True,
        )

    def process_weights_after_loading(self) -> None:
        self.gate_up_proj.process_weights_after_loading()
        self.down_proj.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        # GLM-5.3-Flash SwiGLU clamp (matches HF Glm5NextTextMLP).
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        out = self.down_proj(F.silu(gate) * up)
        if self.cfg.tp_size > 1 and not self.skip_tp_reduce:
            distributed.all_reduce_(out)
        return out


def _local_expert_range(cfg: Glm5NextConfig, num_experts: int) -> tuple[int, int, int]:
    """Return (num_local, start, end) for this EP rank's routed experts."""
    num_local = num_experts // cfg.ep_size
    start = cfg.ep_rank * num_local
    return num_local, start, start + num_local


class Glm5NextExperts(nn.Module):
    """Local EP experts with MoE-TP-sharded SwiGLU intermediates.

    Global routing IDs select the rank-local expert range. The parent MoE
    combines the partial down-projection outputs across the expert axes.
    """

    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.num_experts = cfg.n_routed_experts
        self.num_local_experts, self.local_expert_start, self.local_expert_end = _local_expert_range(
            cfg, self.num_experts
        )
        self.swiglu_limit = cfg.swiglu_limit
        # Local shard of the expert intermediate (inter // tp).
        self.intermediate_dim = cfg.moe_intermediate_size // cfg.moe_tp_size
        self.hidden_dim = cfg.hidden_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_dim, self.hidden_dim, dtype=dtype, device=device)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_local_experts, self.hidden_dim, self.intermediate_dim, dtype=dtype, device=device)
        )

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        # Deterministic fp32 accumulation: index_add_ scatters in bf16 and (per
        # the transformers moe.py note) is non-deterministic on accelerator
        # atomicAdd. Accumulate each (token, topk-slot) expert contribution in a
        # fp32 [n_tokens, topk, hidden] buffer and sum the topk axis — stable
        # and matches the reference's eager path exactly.
        #
        # Two paths:
        #  1) Graph-friendly: batched gather + bmm — fixed-shape ops that stay
        #     compatible with ACL graph capture (no dynamic nonzero/torch.where).
        #  2) Eager: per-expert loop with nonzero/torch.where — memory-efficient
        #     for large n_tokens (prefill) where the batched gather would OOM.
        ctx = get_forward_context_or_none()
        if ctx is not None and ctx.acl_graph is not None:
            return self._forward_graph_friendly(hidden_states, top_k_index, top_k_weights)
        return self._forward_eager(hidden_states, top_k_index, top_k_weights)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, topk = top_k_index.shape
        hidden = hidden_states.shape[-1]
        final_f32 = torch.zeros(n_tokens, topk, hidden, dtype=torch.float32, device=hidden_states.device)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            if self.num_local_experts != self.num_experts:
                mask = mask[self.local_expert_start : self.local_expert_end]
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            gate, up = gate_up.chunk(2, dim=-1)
            # GLM-5.3-Flash SwiGLU clamp (matches HF Glm5NextTextExperts).
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            current = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_f32[token_idx, top_k_pos] += current.float()
        return final_f32.sum(1).to(hidden_states.dtype)

    def _forward_graph_friendly(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape batched expert dispatch for ACL graph capture.

        Replaces the eager per-expert loop (nonzero → torch.where → F.linear)
        with gather + bmm whose output shapes are static regardless of which
        experts are selected.  This avoids ``aclnnNonzero`` -triggered stream
        syncs that are illegal inside a captured stream.
        """
        n_tokens, topk = top_k_index.shape
        hidden = hidden_states.shape[-1]

        flat_indices = top_k_index.flatten()  # [N]
        flat_weights = top_k_weights.flatten()  # [N]
        if self.num_local_experts != self.num_experts:
            flat_indices = flat_indices - self.local_expert_start
            local_mask = (flat_indices >= 0) & (flat_indices < self.num_local_experts)
            flat_indices = flat_indices.clamp(0, self.num_local_experts - 1)
            flat_weights = flat_weights * local_mask

        # Repeat each token's hidden state for every expert slot it has.
        h = hidden_states.repeat_interleave(topk, dim=0)  # [N, hidden]

        # Gather the expert weight slices for all assignments at once.
        gate_up_w = self.gate_up_proj[flat_indices]  # [N, 2*inter_dim, hidden]
        down_w = self.down_proj[flat_indices]  # [N, hidden, inter_dim]

        # Batched gate + up projection.
        # [N, 1, hidden] × [N, hidden, 2*inter_dim] → [N, 1, 2*inter_dim]
        gate_up = torch.bmm(
            h.unsqueeze(1),
            gate_up_w.transpose(1, 2),
        ).squeeze(1)  # [N, 2*inter_dim]

        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        # Batched down projection.
        # [N, 1, inter_dim] × [N, inter_dim, hidden] → [N, 1, hidden]
        current = torch.bmm(
            (F.silu(gate) * up).unsqueeze(1),
            down_w.transpose(1, 2),
        ).squeeze(1)  # [N, hidden]

        # Apply router weights and accumulate in fp32 for determinism.
        current = (current * flat_weights.unsqueeze(-1)).float()
        current = current.view(n_tokens, topk, hidden)
        return current.sum(1).to(hidden_states.dtype)


class Glm5NextMoE(nn.Module):
    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        tp = cfg.moe_tp_size
        self.num_experts = cfg.n_routed_experts
        self.num_local_experts, self.local_expert_start, self.local_expert_end = _local_expert_range(
            cfg, self.num_experts
        )
        self.topk = cfg.num_experts_per_tok
        self.n_group = cfg.n_group
        self.topk_group = cfg.topk_group
        self.routed_scaling = cfg.routed_scaling_factor
        self.moe_inter = cfg.moe_intermediate_size
        self.hidden = cfg.hidden_size
        assert self.moe_inter % tp == 0, f"moe_intermediate_size {self.moe_inter} not divisible by tp {tp}"
        self.inter_local = self.moe_inter // tp
        self.use_w8a8: bool = False  # set in load_weights via probe_quant

        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=dtype, device=device)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            persistent=False,
        )

        # --- W8A8 int8 experts (ported from DeepseekV3MoE, deepseek_v32.py:755-796) ---
        # LAZY: the int8 params are registered in _load_experts_w8a8 (the
        # per-layer loader probes the checkpoint first). Eager allocation here
        # costs ~17GB/card at TP16 of idle int8 on the bf16 path, which together
        # with the bf16 experts OOMs the card (KV-cache estimation aborts with
        # 0 available bytes).
        # --- bf16 experts branch: LAZY too (built only in _load_experts_bf16).
        # Eagerly building Glm5NextExperts' 3D bf16 params (~38GB/card at TP8)
        # alongside the int8 experts OOMs the 60GB card.
        self.experts: Optional[Glm5NextExperts] = None

        self.shared_experts = Glm5NextMLP(
            cfg,
            cfg.moe_intermediate_size * cfg.n_shared_experts,
            dtype,
            device,
            skip_tp_reduce=True,
        )

    def process_weights_after_loading(self) -> None:
        """W8A8: transpose + NZ-format expert weights (ported from DeepseekV3MoE).
        bf16: no-op on int8 expert weights, only shared experts processed."""
        if not self.use_w8a8:
            # bf16 branch: int8 experts not loaded, leave untouched; shared goes through its own process
            self.shared_experts.process_weights_after_loading()
            return
        assert torch.all(self.experts_w13_offset == 0), (
            "Glm5NextMoE int8-grouped path needs symmetric int8 experts (experts_w13_offset == 0)"
        )
        assert torch.all(self.experts_w2_offset == 0), (
            "Glm5NextMoE int8-grouped path needs symmetric int8 experts (experts_w2_offset == 0)"
        )
        # Transpose + NZ format-cast with the raw layout released before the
        # cast (mirrors DeepseekV3MoE._format_and_release_expert_weight): ACL
        # graph capture must see the NZ tensor as the sole allocation, not an
        # NCL allocation overlaid by a format view — otherwise the GMM kernel
        # aborts at capture with "weight Format expect FRACTAL_NZ, but got
        # [NCL]". Numerically identical to the previous transpose + cast.
        for param in (self.experts_w13, self.experts_w2):
            transposed = param.data.transpose(1, 2).contiguous()
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            param.data = kernels.format_cast_nz(transposed)
            del transposed
        self.experts_w13_scale.data = self.experts_w13_scale.data.view(self.num_local_experts, -1).contiguous()
        self.experts_w13_offset.data = self.experts_w13_offset.data.view(self.num_local_experts, -1).contiguous()
        self.experts_w2_scale.data = self.experts_w2_scale.data.view(self.num_local_experts, -1).contiguous()
        self.experts_w2_offset.data = self.experts_w2_offset.data.view(self.num_local_experts, -1).contiguous()
        self.shared_experts.process_weights_after_loading()

    def _topk(self, hidden_states: torch.Tensor):
        cfg = self.cfg
        hidden_states = hidden_states.view(-1, cfg.hidden_size)
        router_logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = (
            scores_for_choice.view(-1, cfg.n_group, cfg.n_routed_experts // cfg.n_group).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=cfg.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, cfg.n_group, cfg.n_routed_experts // cfg.n_group)
            .reshape(-1, cfg.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * cfg.routed_scaling_factor
        return router_logits, topk_weights, topk_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_w8a8:
            return self._forward_w8a8(hidden_states)
        if self.cfg.ep_size > 1 or self.cfg.dp_size > 1:
            raise NotImplementedError("GLM-5.3-Flash ordinary EP/DP supports W8A8 expert weights only, not BF16")
        return self._forward_bf16_tp(hidden_states)

    def _forward_bf16_tp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retain the established BF16 TP-only expert and reduction path."""
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden)
        _, topk_weights, topk_indices = self._topk(hidden_states)
        out = self.experts(flat, topk_indices, topk_weights).view(*orig_shape)
        final = out + self.shared_experts(hidden_states)
        if self.cfg.tp_size > 1:
            distributed.all_reduce_(final)
        return final

    def _forward_w8a8(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use clamped local experts and joint reduction for every TP/EP/DP layout."""
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, self.hidden)
        flat, scatter_state = dp_gather_tokens(flat, self.cfg.dp_size, self.cfg.dp_rank)
        logits = self.gate(flat)
        topk_weights, topk_ids = kernels.moe_gate_routing(
            logits,
            self.e_score_correction_bias,
            self.topk,
            self.topk_group,
            self.n_group,
            self.cfg.norm_topk_prob,
            routed_scaling_factor=1.0,
        )
        # The V2 provider consumes [gate | up] columns and retains GLM clamp.
        out = kernels.grouped_moe_with_selected_experts(
            flat,
            topk_weights,
            topk_ids,
            self.experts_w13,
            self.experts_w2,
            self.experts_w13_scale,
            self.experts_w2_scale,
            num_total_experts=self.num_experts,
            start_expert_id=self.local_expert_start,
            num_experts_per_rank=self.num_local_experts,
            swiglu_limit=self.cfg.swiglu_limit,
        )
        out = (out * self.routed_scaling).float()
        # Shared weights remain attention-TP sharded. Each DP group evaluates
        # only its own execution rows, contributing each shared shard once.
        # Reuse the routed buffer's local view: no shared gather, replication
        # factor, or separate TP reduction is needed. Graph capture supplies
        # fixed padded DP execution counts through the same scatter state.
        # Accumulate and reduce in FP32: adding BF16 routed/shared partials
        # first otherwise amplifies rounding when those terms cancel.
        shared = self.shared_experts(hidden_states).view(-1, self.hidden).float()
        scatter_state.scatter(out).add_(shared)
        out = reduce_and_scatter(
            out,
            scatter_state,
            reduce_results=True,
            moe_tp_size=self.cfg.moe_tp_size,
            ep_size=self.cfg.ep_size,
        )
        return out.to(hidden_states.dtype).view(*orig_shape)


# ---------------------------------------------------------------------------
# mHC (Manifold-constrained Hyper-Connection) residual — faithful port of
# transformers Glm5NextTextHyperConnection / Glm5NextTextHyperHead.
# ---------------------------------------------------------------------------
class Glm5NextHyperConnection(nn.Module):
    """4-stream hyper-connection residual (reference 216-292).

    Owns the learned (fn, base, scale) parameters that turn the incoming
    ``hc_mult`` residual streams into collapse/expand weights. ``forward``
    returns ``(post, comb, collapsed)``: ``post`` scales the sublayer output
    per stream, ``comb`` is the Sinkhorn doubly-stochastic 4x4 stream mixer,
    ``collapsed`` is the single-sequence input to feed the sublayer.
    """

    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_sinkhorn_iters = cfg.hc_sinkhorn_iters
        self.hc_eps = cfg.hc_eps
        self.input_norm = _UnweightedRMSNorm(cfg.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * cfg.hidden_size, dtype=torch.float32, device=device))
        self.base = nn.Parameter(torch.empty(mix, dtype=torch.float32, device=device))
        # 3 outputs: pre (collapse), post (placement), comb (mixer) scales.
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32, device=device))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult

        if _has_mhc_fused:
            # --- Fused NPU kernel path (matches DeepSeek V4 C++ decoder layer) ---
            # hc_pre does rsqrt + linear + sinkhorn + weighted-sum-reduce in one
            # fused call.  x: [B, S, hc_mult, D] -> (output [B,S,D], post [B,S,hc_mult],
            # comb [B,S,hc_mult,hc_mult]).
            # .float() guards against weight-loading casting params back to bf16;
            # __init__ creates them in float32 (aligned with vLLM).
            collapsed, post, comb = kernels.hc_pre(
                hidden_streams,
                self.fn,
                self.scale.float(),
                self.base.float(),
                hc,
                self.hc_sinkhorn_iters,
                self.input_norm.variance_epsilon,
                self.hc_eps,
            )
            return post, comb, collapsed
        else:
            # --- Fallback: pure-Python reference implementation ---
            flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
            pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
            pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
            pre_scale, post_scale, comb_scale = self.scale.unbind(0)

            pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
            post = 2 * torch.sigmoid(post_w * post_scale + post_b)
            comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
            comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            for _ in range(self.hc_sinkhorn_iters - 1):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
            return post, comb, collapsed

    def post(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if _has_mhc_fused:
            return kernels.hc_post(hidden_states, residual, post, comb)
        dtype = hidden_states.dtype
        return post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )


class Glm5NextHyperHead(nn.Module):
    """Final mHC stream collapse — unweighted mean over the hc_mult streams."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------
class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, cfg: Glm5NextConfig, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        if cfg.is_dsa(layer_id):
            self.self_attn = Glm5NextMlaAttention(cfg, layer_id, dtype, device)
        else:
            self.self_attn = Glm5NextKdaAttention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        if cfg.is_moe(layer_id):
            self.mlp = Glm5NextMoE(cfg, dtype, device)
        else:
            self.mlp = Glm5NextMLP(cfg, cfg.intermediate_size, dtype, device)
        # mHC residual sites (always on, per reference — `mhc` config is not consulted).
        self.attn_hc = Glm5NextHyperConnection(cfg, dtype, device)
        self.ffn_hc = Glm5NextHyperConnection(cfg, dtype, device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, position_ids, attention_mask, prev_topk_indices)
        if isinstance(attn_out, tuple):
            hidden_states, topk = attn_out
        else:
            hidden_states, topk = attn_out, None
        # MLA returns [B*S, D]; mHC post expects [B, S, D].
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.view(residual.shape[0], residual.shape[1], -1)
        hidden_states = self.attn_hc.post(hidden_states, residual, post, comb)

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ffn_hc.post(hidden_states, residual, post, comb)
        return hidden_states, topk


class Glm5NextModel(nn.Module):
    def __init__(self, cfg: Glm5NextConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size,
            cfg.hidden_size // cfg.tp_size,
            cfg.tp_size,
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList([Glm5NextDecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hc_head = Glm5NextHyperHead()
        # Set externally by the VL composer (get_input_embeddings) before the
        # runner drives forward(); when set, it replaces embed_tokens(input_ids)
        # so image/video embeddings merged into the sequence are used as-is. The
        # mHC 4-stream expand below applies to the merged hidden identically.
        self._inputs_embeds: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The C++ EagerRunner passes the flattened token tensor, which is 1-D
        # ``[num_tokens]`` for a single sequence; normalise to ``[B, S]``.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if self._inputs_embeds is not None:
            hidden = self._inputs_embeds
            self._inputs_embeds = None
        else:
            hidden = self.embed_tokens(input_ids)
        batch_size, seq_len = hidden.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden.device)

        forward_context = get_forward_context()
        cp_context = getattr(forward_context, "cp_context", None)
        if cp_context is not None:
            hidden = cp_shard_rows(hidden.view(-1, self.cfg.hidden_size), cp_context).unsqueeze(0)
            position_ids = cp_shard_positions(position_ids.reshape(-1), cp_context).unsqueeze(0).contiguous()
            attention_mask = cp_context.shard_valid_mask.unsqueeze(0)
            batch_size, seq_len = hidden.shape[:2]

        # Expand embedding to hc_mult residual streams (all streams start
        # identical — reference model forward, modeling line 1521). NoPE: no
        # position embeddings are computed or threaded (reference passes None).
        hidden = hidden.unsqueeze(2).expand(-1, -1, self.cfg.hc_mult, -1).contiguous()
        prev_topk: Optional[torch.Tensor] = None
        for i, layer in enumerate(self.layers):
            hidden, prev_topk = layer(hidden, position_ids, attention_mask, prev_topk)
            record_layer_event(i)
        # Final collapse: unweighted mean over the streams, then RMSNorm
        # (reference `self.norm(self.hc_head(hidden_states))`, line 1537).
        # Flatten [B, S, D] -> [B*S, D]: the engine's compute_logits does
        # ``hidden.index_select(0, selected_idxes)`` where selected_idxes are
        # token ids in the flattened sequence, so a 3-D output would select the
        # wrong (batch) axis and gather out of range for multi-token prefill.
        h = self.norm(self.hc_head(hidden)).view(-1, self.cfg.hidden_size)
        if cp_context is not None:
            h = cp_merge_rows(h, cp_context)
        return h


def _w8a8_shard_dims(fp_dim: Optional[int]) -> Optional[dict]:
    """Static-W8A8 projection shard map from the fp shard dim.

    - ``fp_dim is None`` (replicated, e.g. q_a_proj / kv_a_proj): no shard.
    - ``fp_dim == 0`` (column-parallel, e.g. q_b_proj): weight + deq_scale +
      quant_bias shard dim 0 (the W8A8 dequant/quant buffers are per-output).
    - ``fp_dim == 1`` (row-parallel, e.g. o_proj): only the weight shards dim 1;
      deq_scale/quant_bias are unsharded (the output is gathered, not split).
    Mirrors deepseek_v32's ``load_weights`` shard-dim choices.
    """
    if fp_dim is None:
        return None
    if fp_dim == 1:
        return {"weight": 1}
    return {"weight": 0, "deq_scale": 0, "quant_bias": 0}


_HC_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(attn|ffn)_hc\.(fn|base|scale)$")


def _real_ckpt_aliases(name: str) -> list:
    """Candidate real-checkpoint keys for a model state-dict key.

    The official hf checkpoint nests text weights under ``model.language_model.``
    (multimodal container) and flattens mHC params as ``hc_{site}_{param}``,
    while our model state-dict uses bare ``model.`` and ``{site}_hc.{param}``.
    Tries the alias list in order; the caller falls back through them.
    """
    out = []
    m = _HC_PATTERN.match(name)
    if m:  # mHC: {site}_hc.{param} -> hc_{site}_{param} (+ language_model)
        i, site, param = m.group(1), m.group(2), m.group(3)
        out.append(f"model.language_model.layers.{i}.hc_{site}_{param}")
    # forget_gate nesting: model self_attn.forget_gate.{x} -> real flat self_attn.{x}
    # (f_a_proj/f_b_proj/dt_bias/A_log). Strip the forget_gate. segment.
    if ".self_attn.forget_gate." in name:
        out.append(name.replace(".self_attn.forget_gate.", ".self_attn."))
    if name.startswith("model.") and not name.startswith("model.language_model."):
        out.append("model.language_model." + name[len("model.") :])
    # Also try stripping forget_gate on the prefixed alias (applied after the
    # prefix transform above so order matters: rebuild from the base name).
    if ".self_attn.forget_gate." in name and name.startswith("model."):
        base = name.replace(".self_attn.forget_gate.", ".self_attn.")
        out.append("model.language_model." + base[len("model.") :])
    return out


class Glm5NextForCausalLM(PyModelBase):
    """glm5_next causal LM. Registered under model_type='glm5_next'."""

    def __init__(self, config: dict, build_model: bool = True) -> None:
        super().__init__()
        self.cfg = Glm5NextConfig.from_dict(config)
        for name, val in (
            ("n_heads", self.cfg.n_heads),
            ("intermediate_size", self.cfg.intermediate_size),
            ("moe_intermediate_size", self.cfg.moe_intermediate_size),
            ("vocab_size", self.cfg.vocab_size),
            ("hidden_size", self.cfg.hidden_size),
        ):
            assert val % self.cfg.tp_size == 0, f"{name} {val} not divisible by tp {self.cfg.tp_size}"
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0" if torch_npu else "cpu"))
        self.dtype = dtype
        self.device = device
        if build_model:
            self.model = Glm5NextModel(self.cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // self.cfg.tp_size,
            self.cfg.tp_size,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        # The layer projections are created without an explicit dtype (they
        # default to float32 on the target device); move AND cast the whole
        # graph to the target dtype/device so the engine matches the reference
        # (which runs .to(bf16) over the model). The KDA conv weights are included in
        # this cast — the reference's conv1d is also bf16 after .to(dtype).
        self.to(device=device, dtype=dtype)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids, attention_mask)
        return self.lm_head(hidden)

    # -- weight loading ---------------------------------------------------
    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        from xllm.python.model_loader import QLinearWeightLoader

        L = QLinearWeightLoader(
            self,
            state_dicts,
            tp_size,
            tp_rank,
            name_aliases=lambda name: (name, *_real_ckpt_aliases(name)),
        )

        # embed_tokens: HiddenParallelEmbedding — shard the hidden dim (dim 1).
        L.copy_shard("model.embed_tokens.weight", dim=1)
        for i in range(self.cfg.n_layers):
            p = f"model.layers.{i}."
            L.copy_replicated(p + "input_layernorm.weight")
            L.copy_replicated(p + "post_attention_layernorm.weight")
            attn = p + "self_attn."
            if self.cfg.is_dsa(i):
                self._load_dsa_attn(L, attn, i)
            else:
                self._load_kda_attn(L, attn, i)
            self._load_mlp(L, p + "mlp.", i)
            # mHC residual sites (always on, replicated).
            for hc in ("attn_hc", "ffn_hc"):
                for w in ("fn", "base", "scale"):
                    L.copy_replicated(p + hc + "." + w)
        L.copy_replicated("model.norm.weight")
        # lm_head: ColumnParallelLinear — shard the vocab dim (dim 0).
        L.copy_shard("lm_head.weight", dim=0)

    def _load_qlinear(
        self,
        L,
        prefix: str,
        proj: str,
        fp_dim: Optional[int],
    ) -> None:
        """Probe fp-vs-w8a8 for a QLinear projection, resolve, and load.

        ``fp_dim`` is the TP shard dim for the fp ``.weight`` (None = replicated,
        0 = column-parallel, 1 = row-parallel). The w8a8 path routes static
        projections into the QLinear's ``_w8a8`` submodule through
        ``load_projection``; dynamic ones never reach here (MLP uses
        ``_load_mlp_fp_or_w8a8`` -> ``load_w8a8_mlp``). The w8a8
        branch is NOT exercised until real w8a8 checkpoints exist (Task 10); it
        is written to match deepseek_v32's documented shard-dim contract.
        """
        name = prefix + proj
        mod = self.get_submodule(name)
        assert isinstance(mod, QLinear), f"{name} is not a QLinear"
        if L.probe_quant(prefix, proj):
            mod.resolve_quant(True)
            L.load_projection(
                name,
                name + "._w8a8",
                ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset"),
                _w8a8_shard_dims(fp_dim),
            )
        else:
            mod.resolve_quant(False)
            if fp_dim is None:
                L.copy_replicated(name + ".weight")
            else:
                L.copy_shard(name + ".weight", dim=fp_dim)

    def _load_dsa_attn(self, L, attn: str, i: int) -> None:
        # q_a_proj / kv_a_proj_with_mqa: replicated QLinear (no shard).
        self._load_qlinear(L, attn, "q_a_proj", None)
        self._load_qlinear(L, attn, "kv_a_proj_with_mqa", None)
        # q_b_proj: column-parallel QLinear (shard dim 0).
        self._load_qlinear(L, attn, "q_b_proj", 0)
        # o_proj: row-parallel QLinear (shard dim 1).
        self._load_qlinear(L, attn, "o_proj", 1)
        # kv_b_proj: fp ColumnParallelLinear (NOT a QLinear) — column-parallel.
        L.copy_shard(attn + "kv_b_proj.weight", dim=0)
        L.copy_replicated(attn + "q_a_layernorm.weight")
        L.copy_replicated(attn + "kv_a_layernorm.weight")
        if not self.cfg.indexer_shared(i):
            idx = attn + "indexer."
            # Indexer projections are plain nn.Linear and stay replicated: the
            # sparse kPool mask is shared across TP ranks (each rank attends its
            # head-subset to the SAME mask), so wq_b/wk/weights_proj hold the
            # full index-head tensors on every rank.
            for w in ("wq_b.weight", "wk.weight", "weights_proj.weight"):
                L.copy_replicated(idx + w)
            L.copy_replicated(idx + "index_kpool_compress_gate")
            L.copy_replicated(idx + "k_norm.weight")
            L.copy_replicated(idx + "k_norm.bias")
            L.copy_replicated(idx + "index_kpool_compress_ape")
        # Splits kv_b_proj.weight into absorbed W_UK / W_UV (NoPE path).
        self.model.layers[i].self_attn.process_weights_after_loading()

    def _load_kda_attn(self, L, attn: str, i: int) -> None:
        """Load floating-point KDA inputs; per-projection W8A8 is not supported here."""
        layer = self.get_submodule(attn.rstrip("."))
        input_weights = []
        for projection, size_attr, shard_dim in _KDA_IN_PROJ:
            weight = L.get_tensor(attn + projection + ".weight")
            if not weight.is_floating_point() or L.probe_quant(attn, projection):
                raise ValueError(
                    f"KDA input projection {attn}{projection} requires floating-point weights "
                    f"without W8A8 metadata, got {weight.dtype}"
                )
            if shard_dim is not None:
                weight = L.shard(weight, dim=shard_dim)
            expected_shape = (getattr(layer, size_attr), layer.hidden_size)
            if weight.shape != expected_shape:
                raise ValueError(
                    f"checkpoint tensor {attn}{projection}.weight has shape {tuple(weight.shape)}, expected {expected_shape}"
                )
            input_weights.append(weight)
        L.copy_in(attn + "in_proj_qkvbfg_a.weight", torch.cat(input_weights, dim=0))
        L.copy_shard(attn + "g_b_proj.weight", dim=0)
        # conv1d: depthwise over [q|k|v] (conv_dim = 3*qkv_dim). The model holds
        # the LOCAL conv_dim (3*qkv_dim_local); to shard by head each of q/k/v
        # must be narrowed on dim 0 (its head block) BEFORE cat, so the channel
        # order [q_loc|k_loc|v_loc] matches mixed_qkv. (A single contiguous
        # narrow of the cat'd tensor would cross the q/k boundary.) Fake
        # single-key checkpoint is only run at tp==1, where the shard is a no-op.
        if L.find(attn + "conv1d.weight") is not None:
            conv = L.get_tensor(attn + "conv1d.weight")
            if L.tp_size > 1:
                conv = L.shard(conv, dim=0)
        else:
            parts = [L.get_tensor(attn + n + "_conv1d.weight") for n in ("q", "k", "v")]
            conv = torch.cat([L.shard(p, dim=0) for p in parts] if L.tp_size > 1 else parts, dim=0)
        L.copy_in(attn + "conv_weight", conv)
        # o_proj: row-parallel QLinear — shard the INPUT dim (dim 1, qkv_dim) so
        # each rank's [hidden, qkv_dim_local] weight consumes its head-subset's
        # partial output; the forward all-reduces the partials.
        self._load_qlinear(L, attn, "o_proj", 1)
        # forget_gate: model nests under forget_gate.*, real ckpt is flat
        # (self_attn.f_a_proj / f_b_proj / dt_bias / A_log). load_tensor goes
        # through the alias wrapper, so the bare name resolves to the flat key.
        L.copy_shard(attn + "forget_gate.f_b_proj.weight", dim=0)
        L.copy_shard(attn + "forget_gate.dt_bias", dim=0)
        L.copy_shard(attn + "forget_gate.A_log", dim=0)
        L.copy_replicated(attn + "o_norm.weight")
        self.model.layers[i].self_attn.process_weights_after_loading()

    def _load_mlp_fp_or_w8a8(self, L, mlp_pfx: str) -> None:
        """Load a ``Glm5NextMLP`` (gate_up_proj + down_proj) from the OLD-style
        checkpoint keys ``gate_proj`` / ``up_proj`` / ``down_proj``.

        fp path: cat gate+up on dim 0, shard dim 0 -> ``gate_up_proj.weight``;
        ``down_proj.weight`` shards dim 1 (row-parallel). w8a8 dynamic path:
        ``load_w8a8_mlp`` does the same cat+shard for the quant
        tensors, writing into each QLinear's ``_w8a8`` submodule (NOT exercised
        until real w8a8 checkpoints exist, Task 10).
        """
        gate_mod = self.get_submodule(mlp_pfx + "gate_up_proj")
        down_mod = self.get_submodule(mlp_pfx + "down_proj")
        assert isinstance(gate_mod, QLinear) and isinstance(down_mod, QLinear), (
            f"{mlp_pfx}gate_up_proj/down_proj must be QLinear"
        )
        is_w8a8 = (
            L.probe_quant(mlp_pfx, "gate_proj")
            or L.probe_quant(mlp_pfx, "up_proj")
            or L.probe_quant(mlp_pfx, "down_proj")
        )
        if is_w8a8:
            # dynamic w8a8: route through load_w8a8_mlp (cat
            # gate+up, shard dim 0; down shard dim 1), NOT the per-projection
            # load_projection path above which handles static per-projection
            # tensors. resolve_quant builds the w8a8 submodules; the loader
            # writes into ``_w8a8``.
            gate_mod.resolve_quant(True)
            down_mod.resolve_quant(True)
            L.load_w8a8_mlp(mlp_pfx, destination_suffix="._w8a8")
        else:
            gate_mod.resolve_quant(False)
            down_mod.resolve_quant(False)
            L.copy_in(mlp_pfx + "gate_up_proj.weight", L.pack_gate_up(mlp_pfx))
            L.copy_in(mlp_pfx + "down_proj.weight", L.load_shard(mlp_pfx + "down_proj.weight", 1))

    def _load_experts_w8a8(self, L, mlp: str, moe_mod: Glm5NextMoE) -> None:
        """Load local W8A8 experts with separate EP and MoE-TP sharding."""
        se = mlp + "experts."
        ref = moe_mod.gate.weight
        num_local = moe_mod.num_local_experts
        inter_local = moe_mod.inter_local
        hidden = moe_mod.hidden
        if not hasattr(moe_mod, "experts_w13"):
            moe_mod.experts_w13 = nn.Parameter(
                torch.empty(num_local, 2 * inter_local, hidden, dtype=torch.int8, device=ref.device),
                requires_grad=False,
            )
            moe_mod.register_buffer(
                "experts_w13_scale",
                torch.empty(num_local, 2 * inter_local, 1, dtype=torch.float32, device=ref.device),
            )
            moe_mod.register_buffer(
                "experts_w13_offset",
                torch.zeros(num_local, 2 * inter_local, 1, dtype=torch.float32, device=ref.device),
            )
            moe_mod.experts_w2 = nn.Parameter(
                torch.empty(num_local, hidden, inter_local, dtype=torch.int8, device=ref.device),
                requires_grad=False,
            )
            moe_mod.register_buffer(
                "experts_w2_scale", torch.empty(num_local, hidden, 1, dtype=torch.bfloat16, device=ref.device)
            )
            moe_mod.register_buffer(
                "experts_w2_offset", torch.zeros(num_local, hidden, 1, dtype=torch.float32, device=ref.device)
            )
        w13 = self.get_parameter(mlp + "experts_w13")
        w2 = self.get_parameter(mlp + "experts_w2")
        w13s = self.get_buffer(mlp + "experts_w13_scale")
        w13o = self.get_buffer(mlp + "experts_w13_offset")
        w2s = self.get_buffer(mlp + "experts_w2_scale")
        w2o = self.get_buffer(mlp + "experts_w2_offset")
        world = self.cfg.moe_tp_size
        rank = self.cfg.moe_tp_rank
        for local_index, expert in enumerate(range(moe_mod.local_expert_start, moe_mod.local_expert_end)):
            prefix = se + f"{expert}."
            w13.data[local_index].copy_(L.pack_gate_up(prefix, world=world, rank=rank))
            w13s.data[local_index].copy_(L.pack_gate_up(prefix, "weight_scale", world=world, rank=rank))
            w13o.data[local_index].copy_(L.pack_gate_up(prefix, "weight_offset", world=world, rank=rank))
            down, scale = L.load_w8a8_down(prefix, world=world, rank=rank)
            w2.data[local_index].copy_(down.contiguous())
            w2s.data[local_index].copy_(scale.contiguous())
            w2o.data[local_index].copy_(L.get_tensor(prefix + "down_proj.weight_offset").contiguous())

    def _load_experts_bf16(self, L, mlp: str, moe_mod: Glm5NextMoE) -> None:
        """Load local BF16 experts with separate EP and MoE-TP sharding."""
        start, end = moe_mod.local_expert_start, moe_mod.local_expert_end
        if moe_mod.experts is None:
            ref = moe_mod.shared_experts.gate_up_proj.weight
            moe_mod.experts = Glm5NextExperts(self.cfg, ref.dtype, ref.device)
        if L.find(mlp + "experts.gate_up_proj") is not None:
            gu = L.get_tensor(mlp + "experts.gate_up_proj")[start:end]
            gate_full, up_full = gu.split([gu.size(1) // 2, gu.size(1) // 2], dim=1)
        else:
            gate_full = torch.stack(
                [L.get_tensor(mlp + f"experts.{expert}.gate_proj.weight") for expert in range(start, end)]
            )
            up_full = torch.stack(
                [L.get_tensor(mlp + f"experts.{expert}.up_proj.weight") for expert in range(start, end)]
            )
        gate = L.shard(gate_full, dim=1, world=self.cfg.moe_tp_size, rank=self.cfg.moe_tp_rank)
        up = L.shard(up_full, dim=1, world=self.cfg.moe_tp_size, rank=self.cfg.moe_tp_rank)
        L.copy_in(mlp + "experts.gate_up_proj", torch.cat([gate, up], dim=1))
        if L.find(mlp + "experts.down_proj") is not None:
            down = L.get_tensor(mlp + "experts.down_proj")[start:end]
        else:
            down = torch.stack(
                [L.get_tensor(mlp + f"experts.{expert}.down_proj.weight") for expert in range(start, end)]
            )
        L.copy_in(
            mlp + "experts.down_proj",
            L.shard(down, dim=2, world=self.cfg.moe_tp_size, rank=self.cfg.moe_tp_rank),
        )

    def _load_mlp(self, L, mlp: str, i: int) -> None:
        if self.cfg.is_moe(i):
            moe = self.model.layers[i].mlp
            # Router (FLOAT, shared by both branches).
            L.copy_replicated(mlp + "gate.weight")
            # e_score_correction_bias quirk: checkpoint key lives under
            # ``gate.e_score_correction_bias`` but the buffer is ``e_score_...``.
            e = L.get_tensor(mlp + "gate.e_score_correction_bias")
            L.copy_in(mlp + "e_score_correction_bias", e)
            # Expert branch: probe exp0's gate_proj for a weight_scale tensor.
            # Real W8A8 checkpoints carry weight_scale; bf16 checkpoints do not.
            is_w8a8 = L.find(mlp + "experts.0.gate_proj.weight_scale") is not None
            if not is_w8a8 and (self.cfg.ep_size > 1 or self.cfg.dp_size > 1):
                raise NotImplementedError("GLM-5.3-Flash ordinary EP/DP supports W8A8 expert weights only, not BF16")
            moe.use_w8a8 = is_w8a8
            if is_w8a8:
                self._load_experts_w8a8(L, mlp, moe)
            else:
                self._load_experts_bf16(L, mlp, moe)
            self._load_mlp_fp_or_w8a8(L, mlp + "shared_experts.")
            moe.process_weights_after_loading()
        else:
            self._load_mlp_fp_or_w8a8(L, mlp)
            self.model.layers[i].mlp.process_weights_after_loading()


# Registration is centralised in xllm.python.registry; import there.
