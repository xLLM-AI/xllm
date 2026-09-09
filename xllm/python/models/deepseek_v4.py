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

"""DeepSeek-V4 Python model (DSA sparse attention, TORCH backend).

Structural port of the C++ ``DeepseekV4ModelImpl`` /
``DeepseekV4DecoderLayerImpl`` / ``DSAttentionImpl`` (xllm/models/llm/
deepseek_v4.h, xllm/core/layers/deepseek_v4_decoder_layer.cpp,
xllm/core/layers/npu_torch/deepseek_sparse_attention.cpp). Reuses the W8A8
linear / MLP / MoE / YaRN-RoPE / weight-loader primitives from
``deepseek_v32`` and adds the DeepSeek-V4-specific pieces:

  * ``DeepseekV4Config`` -- reads the DSV4 fields (compress_ratios, window_size,
    o_lora_rank, o_groups, hc_*, index_*).
  * HyperConnection residual path (hc_pre / hc_post).
  * ``DeepseekV4Attention`` -- q_a/kv projections + RoPE, hands q/kv to the
    DSA attention backend (``backend.execute``), two-stage o_a/o_b output proj.
  * ``DeepseekV4Indexer`` -- Hadamard rotation + compressor + quantized
    lightning indexer.
  * ``DeepseekV4ForCausalLM`` -- ``load_weights`` reusing ``W8A8WeightLoader``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn
import torch_npu

from xllm.python.attention.csa_attention import (
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_executor.forward_context import (
    get_forward_context,
    record_layer_event,
)
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v32 import (
    DeepseekV3MLP,
    DeepseekYarnRotaryEmbedding,
    W8A8DynamicLinear,
    _tp_rank_from_device,
)
from xllm.python.models.weight_utils import W8A8WeightLoader

try:
    from xllm.python import distributed
except Exception:  # pragma: no cover - distributed is optional in tests
    distributed = None  # type: ignore[assignment]

try:
    from xllm.python import kernels
except Exception:  # pragma: no cover - kernels need the compiled lib
    kernels = None  # type: ignore[assignment]


def _pick(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _compress_kv(
    hidden: torch.Tensor,
    *,
    kv_state: torch.Tensor | None,
    score_state: torch.Tensor | None,
    dsa: Any,
    layer_id: int,
    kv_cache_idx: int,
    score_cache_idx: int,
    compress_ratio: int,
    rope_head_dim: int,
    wkv: torch.Tensor | None,
    wgate: torch.Tensor | None,
    ape: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> torch.Tensor | None:
    """Run a DSV4 compressor with caller-owned weights and cache state."""
    if kernels is None or kv_state is None or score_state is None:
        return None
    if wkv is None or wgate is None or norm_weight is None:
        raise RuntimeError("compressor weights must be processed before forward")

    if compress_ratio == 4:
        cos_table = dsa.c4_cos
        sin_table = dsa.c4_sin
    elif compress_ratio == 128:
        cos_table = dsa.c128_cos
        sin_table = dsa.c128_sin
    else:
        cos_table = dsa.cos_table
        sin_table = dsa.sin_table
    if cos_table is None or sin_table is None:
        return None

    block_tables = dsa.block_tables
    kv_block_table = _get_layer_cache_tensor(block_tables, layer_id, kv_cache_idx) if block_tables else None
    score_block_table = _get_layer_cache_tensor(block_tables, layer_id, score_cache_idx) if block_tables else None
    sin_view = sin_table.reshape(-1, sin_table.size(-1)) if sin_table.dim() > 2 else sin_table
    cos_view = cos_table.reshape(-1, cos_table.size(-1)) if cos_table.dim() > 2 else cos_table
    if sin_view.size(-1) * 2 == rope_head_dim:
        sin_view = sin_view.repeat_interleave(2, dim=-1)
        cos_view = cos_view.repeat_interleave(2, dim=-1)

    # C++ keeps the tiling metadata on the host. Only tensor inputs consumed by
    # the AICore kernel are converted to the required BF16 device representation.
    seq_q = dsa.actual_seq_lengths_query.contiguous()
    start_pos = dsa.start_pos.contiguous() if dsa.start_pos.numel() > 0 else None
    compressed_kv, _, _, _, _ = kernels.compressor(
        x=hidden.to(torch.bfloat16).contiguous(),
        wkv=wkv,
        wgate=wgate,
        kv_state=kv_state,
        score_state=score_state,
        ape=ape,
        norm_weight=norm_weight,
        rope_sin=sin_view.to(device=hidden.device, dtype=torch.bfloat16).contiguous(),
        rope_cos=cos_view.to(device=hidden.device, dtype=torch.bfloat16).contiguous(),
        kv_block_table=kv_block_table,
        score_block_table=score_block_table,
        cu_seqlens=seq_q,
        seqused=None,
        start_pos=start_pos,
        rope_head_dim=rope_head_dim,
        cmp_ratio=compress_ratio,
        coff=2 if compress_ratio == 4 else 1,
        norm_eps=norm_eps,
        rotary_mode=2,
        enable_grad=False,
    )
    return compressed_kv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4Config:
    """DeepSeek-V4 model config (DSV3.2 MLA fields + DSV4-specific fields)."""

    model_type: str = "deepseek_v4"
    hidden_size: int = 4096
    n_layers: int = 43
    n_heads: int = 64
    head_dim: int = 512
    vocab_size: int = 129280
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1048576
    original_max_position_embeddings: int = 65536
    rope_scaling_factor: float = 16.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    q_lora_rank: int = 1024
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 64
    v_head_dim: int = 0
    # DSV4-specific
    rope_head_dim: int = 64
    o_lora_rank: int = 1024
    o_groups: int = 8
    compress_ratios: list[int] = field(default_factory=list)
    compress_rope_theta: float = 160000.0
    window_size: int = 128
    n_activated_experts: int = 6
    n_hash_layers: int = 3
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    scoring_func: str = "sqrtsoftplus"
    scale_fmt: str = "ue8m0"
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    swiglu_limit: float = 10.0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    topk_method: str = "noaux_tc"
    n_group: int = 0
    topk_group: int = 0
    tie_word_embeddings: bool = False
    tp_size: int = 1
    tp_rank: int = 0
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV4Config:
        rs_raw = d.get("rope_scaling")
        rs = rs_raw if isinstance(rs_raw, dict) else {}

        def rope_value(
            model_arg: str,
            nested_key: str,
            default: float | int,
            *legacy_keys: str,
        ) -> float | int:
            # PyCausalLM reflects ModelArgs into this dict. DSV4 uses factor,
            # beta_fast/beta_slow and rope_scaling_attn_factor directly; older
            # generic aliases may also be present with their zero defaults.
            for key in (model_arg, *legacy_keys):
                value = d.get(key)
                if value not in (None, 0, 0.0):
                    return value
            value = rs.get(nested_key)
            return default if value in (None, 0, 0.0) else value

        n_layers = int(_pick(d, "num_hidden_layers", "n_layers", default=43))
        compress_ratios = [1 if int(ratio) <= 1 else int(ratio) for ratio in d.get("compress_ratios", [])]
        if len(compress_ratios) < n_layers:
            compress_ratios.extend([1] * (n_layers - len(compress_ratios)))

        return cls(
            model_type=_pick(d, "model_type", default="deepseek_v4"),
            hidden_size=int(_pick(d, "hidden_size", default=4096)),
            n_layers=n_layers,
            n_heads=int(_pick(d, "n_heads", "num_attention_heads", default=64)),
            head_dim=int(_pick(d, "head_dim", default=512)),
            vocab_size=int(_pick(d, "vocab_size", default=129280)),
            rms_norm_eps=float(_pick(d, "rms_norm_eps", default=1e-6)),
            rope_theta=float(_pick(d, "rope_theta", default=10000.0)),
            max_position_embeddings=int(_pick(d, "max_position_embeddings", default=1048576)),
            original_max_position_embeddings=int(
                rope_value(
                    "rope_scaling_original_max_position_embeddings",
                    "original_max_position_embeddings",
                    65536,
                )
            ),
            rope_scaling_factor=float(rope_value("factor", "factor", 16.0, "rope_scaling_factor")),
            rope_beta_fast=int(rope_value("beta_fast", "beta_fast", 32, "rope_scaling_beta_fast")),
            rope_beta_slow=int(rope_value("beta_slow", "beta_slow", 1, "rope_scaling_beta_slow")),
            rope_mscale=float(rope_value("rope_scaling_attn_factor", "attn_factor", 1.0)),
            rope_mscale_all_dim=1.0,
            q_lora_rank=int(_pick(d, "q_lora_rank", default=1024)),
            qk_rope_head_dim=int(_pick(d, "qk_rope_head_dim", default=64)),
            rope_head_dim=int(_pick(d, "qk_rope_head_dim", default=64)),
            o_lora_rank=int(_pick(d, "o_lora_rank", default=1024)),
            o_groups=int(_pick(d, "o_groups", default=8)),
            compress_ratios=compress_ratios,
            compress_rope_theta=float(_pick(d, "compress_rope_theta", default=160000.0)),
            window_size=(
                int(v) if (v := _pick(d, "window_size", "sliding_window", default=128)) not in (None, -1, 0) else 128
            ),
            n_activated_experts=int(_pick(d, "n_activated_experts", "num_experts_per_tok", default=6)),
            # PyCausalLM reflects the native ModelArgs field as n_hash_layers;
            # direct Hugging Face config dictionaries use num_hash_layers.
            n_hash_layers=int(_pick(d, "n_hash_layers", "num_hash_layers", default=3)),
            hc_mult=int(_pick(d, "hc_mult", default=4)),
            hc_sinkhorn_iters=int(_pick(d, "hc_sinkhorn_iters", default=20)),
            hc_eps=float(_pick(d, "hc_eps", default=1e-6)),
            scoring_func=_pick(d, "scoring_func", default="sqrtsoftplus"),
            scale_fmt=_pick(d, "scale_fmt", default="ue8m0"),
            index_head_dim=int(_pick(d, "index_head_dim", default=128)),
            index_n_heads=int(_pick(d, "index_n_heads", default=64)),
            index_topk=int(_pick(d, "index_topk", default=512)),
            n_routed_experts=int(_pick(d, "n_routed_experts", default=256)),
            n_shared_experts=int(_pick(d, "n_shared_experts", default=1)),
            moe_intermediate_size=int(_pick(d, "moe_intermediate_size", default=2048)),
            swiglu_limit=float(_pick(d, "swiglu_limit", default=10.0)),
            first_k_dense_replace=int(_pick(d, "first_k_dense_replace", default=0)),
            moe_layer_freq=int(_pick(d, "moe_layer_freq", default=1)),
            norm_topk_prob=bool(_pick(d, "norm_topk_prob", default=True)),
            routed_scaling_factor=float(_pick(d, "routed_scaling_factor", default=1.5)),
            topk_method=_pick(d, "topk_method", default="noaux_tc"),
            n_group=int(_pick(d, "n_group", default=0)),
            topk_group=int(_pick(d, "topk_group", default=0)),
            tie_word_embeddings=bool(_pick(d, "tie_word_embeddings", default=False)),
            tp_size=int(d.get("tp_size", 1)),
            tp_rank=int(d.get("tp_rank", _tp_rank_from_device(d.get("device", "npu:0")))),
            moe_tp_size=int(d.get("moe_tp_size", 1)),
            moe_tp_rank=int(d.get("moe_tp_rank", 0)),
            ep_size=int(d.get("ep_size", d.get("tp_size", 1))),
            ep_rank=int(d.get("ep_rank", d.get("tp_rank", 0))),
            cp_size=int(d.get("cp_size", 1)),
            cp_rank=int(d.get("cp_rank", 0)),
            dp_size=int(d.get("dp_size", 1)),
            dp_rank=int(d.get("dp_rank", 0)),
        )

    def head_split(self) -> tuple[int, int]:
        return self.n_heads // self.tp_size, 1

    # -- aliases so DSV3.2-reused modules (MoE/MLP) read DSV4 config unchanged --
    @property
    def num_experts_per_tok(self) -> int:
        return self.n_activated_experts

    @property
    def intermediate_size(self) -> int:
        return self.moe_intermediate_size


# ---------------------------------------------------------------------------
# DeepSeek-V4 RoPE
# ---------------------------------------------------------------------------


class DeepseekV4RotaryEmbedding(nn.Module):
    """Compact-cache equivalent of C++ DeepseekV4RotaryEmbedding.

    C++ keeps cache length and YaRN's old-context length as separate inputs.
    The generic Python DeepseekYarnRotaryEmbedding derives cache length as
    ``old_context * factor``, which cannot represent the native DSV4 call.
    """

    _cache_lock = threading.Lock()
    _cache_by_descriptor: dict[tuple[Any, ...], torch.Tensor] = {}

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        scaling_factor: float,
        theta: float,
        beta_fast: int,
        beta_slow: int,
        old_context_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        descriptor = (
            rotary_dim,
            max_position_embeddings,
            scaling_factor,
            theta,
            beta_fast,
            beta_slow,
            old_context_len,
            dtype,
            torch.device(device),
        )
        with self._cache_lock:
            cache = self._cache_by_descriptor.get(descriptor)
            if cache is None:
                cache = self._build_cache(
                    rotary_dim,
                    max_position_embeddings,
                    scaling_factor,
                    theta,
                    beta_fast,
                    beta_slow,
                    old_context_len,
                    dtype,
                    device,
                )
                self._cache_by_descriptor[descriptor] = cache
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @staticmethod
    def _build_cache(
        rotary_dim: int,
        max_position_embeddings: int,
        scaling_factor: float,
        theta: float,
        beta_fast: int,
        beta_slow: int,
        old_context_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Match C++ rotary::apply_deepseek_yarn_rope_scaling() and
        # create_cos_sin_tensor(): build the cache in CPU float32, then perform
        # one final transfer/conversion to the model device and dtype. Computing
        # pow/cos/sin directly on NPU can change BF16 cache entries by one ULP.
        cpu = torch.device("cpu")
        inv_freq = DeepseekYarnRotaryEmbedding._yarn_inv_freq(
            scaling_factor,
            rotary_dim,
            theta,
            beta_fast,
            beta_slow,
            old_context_len,
            cpu,
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float32, device=cpu)
        freqs = torch.outer(positions, inv_freq)
        # Keep one value per frequency. Call sites repeat_interleave to the
        # same [position, rotary_dim] interleaved layout C++ stores directly.
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device=device, dtype=dtype)
        return cache.contiguous()


# ---------------------------------------------------------------------------
# HyperConnection
# ---------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    """HyperConnection residual path (hc_pre + hc_post).

    Faithful port of the C++ ``DeepseekV4DecoderLayerImpl::hc_pre``/``hc_post``
    (deepseek_v4_decoder_layer.cpp:238+), which call the registered NPU
    ``hc_pre``/``hc_post`` kernels. hc_pre mixes the hc_mult parallel residual
    streams into one sub-block input (via Sinkhorn); hc_post combines the
    sub-block output with the residual. Weight shapes match the checkpoint:
    ``hc_fn = [mix_hc, hc_dim]`` where ``mix_hc = (2+hc_mult)*hc_mult`` and
    ``hc_dim = hc_mult*hidden``; ``hc_base = [mix_hc]``; ``hc_scale = [3]``.
    """

    def __init__(self, cfg: DeepseekV4Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters
        hidden = cfg.hidden_size
        # hc_mult is a fixed model constant; NOT TP-sharded (C++ matches).
        self.hc_mult_local = cfg.hc_mult
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * hidden
        # hc_fn/scale/base per sub-block. C++ registers these as float32.
        for part in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{part}_fn",
                nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_base",
                nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32, device=device)),
            )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Call the registered hc_pre kernel: x [T, hc_mult, hidden] -> (attn_input, post, comb)."""
        from xllm.python import kernels

        return kernels.hc_pre(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )

    def hc_post(
        self,
        sub_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Call the registered hc_post kernel: combine sub-block output + residual.

        Faithful port of C++ (decoder_layer.cpp:258-272): when x=2D, residual=3D,
        post=2D, comb=3D, unsqueeze(0) all inputs (kernel expects 3D/4D/3D/4D),
        then squeeze(0) the output.
        """
        from xllm.python import kernels

        if sub_out.dim() == 2 and residual.dim() == 3 and post.dim() == 2 and comb.dim() == 3:
            out = kernels.hc_post(
                sub_out.unsqueeze(0),
                residual.unsqueeze(0),
                post.unsqueeze(0),
                comb.unsqueeze(0),
            )
            return out.squeeze(0)
        return kernels.hc_post(sub_out, residual, post, comb)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class DeepseekV4Attention(Attention):
    """DeepSeek-V4 DSA attention.

    Projects q/kv (W8A8), applies RoPE, hands q/kv to the DSA backend
    (``backend.execute``), then runs the two-stage o_a/o_b output projection.
    The compressor + indexer are wired through backend callbacks so the
    attention forward stays a thin orchestrator matching the C++ flow.
    """

    def __init__(
        self,
        cfg: DeepseekV4Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp = cfg.tp_size
        num_heads = cfg.n_heads // tp
        head_dim = cfg.head_dim
        scale = head_dim**-0.5
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            scale=scale,
            sliding_window=cfg.window_size,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.num_heads_local = num_heads
        self.head_dim = head_dim
        self.rope_head_dim = cfg.qk_rope_head_dim
        self.nope_head_dim = head_dim - cfg.qk_rope_head_dim
        self.kv_lora_rank = cfg.q_lora_rank
        # Attention sink (learnable logit bias per head), loaded from
        # ``attn_sink`` in the checkpoint. Registered as float32 (the
        # sparse_attn_sharedkv kernel requires DT_FLOAT sinks).
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.empty(num_heads, dtype=torch.float32, device=device)),
        )
        self.attn_sink_loaded = False
        # q/kv down-projections (W8A8).
        # Native DSV4 keeps dynamic-W8A8 weights in checkpoint [N, K] layout
        # and calls quant_matmul with transpose2=true.
        self.q_a_proj = W8A8DynamicLinear(
            cfg.hidden_size,
            cfg.q_lora_rank,
            device,
            transpose_weight_after_loading=False,
        )
        self.kv_proj = W8A8DynamicLinear(
            cfg.hidden_size,
            head_dim,
            device,
            transpose_weight_after_loading=False,
        )
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.kv_a_layernorm = RMSNorm(head_dim, cfg.rms_norm_eps, dtype=dtype, device=device)
        # q up-projection (W8A8) produces [T, num_heads, head_dim].
        self.q_b_proj = W8A8DynamicLinear(
            cfg.q_lora_rank,
            num_heads * head_dim,
            device,
            transpose_weight_after_loading=False,
        )
        self.register_buffer(
            "q_rms_gamma",
            torch.ones(head_dim, dtype=dtype, device=device),
        )
        # Two-stage output projection: o_a (column) -> grouped low-rank -> o_b (row).
        # o_a input is the per-group head slice = global_num_heads * head_dim / o_groups
        # (uses the GLOBAL head count, not the TP-local one); C++ matches this.
        assert cfg.o_groups % tp == 0
        self.n_local_groups = cfg.o_groups // tp
        self.o_lora_rank = cfg.o_lora_rank
        o_a_in = (cfg.n_heads * head_dim) // cfg.o_groups
        # ColumnParallelLinear takes out_features_PER_PARTITION, so shard the
        # full o_groups*o_lora output by tp (matches the C++ ColumnParallelLinear
        # which hands the per-partition count).
        self.o_a_proj = ColumnParallelLinear(
            o_a_in,
            (cfg.o_groups * cfg.o_lora_rank) // tp,
            tp,
            dtype=dtype,
            device=device,
        )
        self.o_b_proj = RowParallelLinear(
            (cfg.o_groups * cfg.o_lora_rank) // tp,
            cfg.hidden_size,
            tp,
            dtype=dtype,
            device=device,
        )
        compress_ratio = cfg.compress_ratios[layer_id]
        self.indexer: DeepseekV4Indexer | None = (
            DeepseekV4Indexer(cfg, dtype, device) if compress_ratio == 4 and cfg.index_topk > 0 else None
        )
        # Cmp_kv compressor (separate from the indexer compressor). C++ DSA
        # attention has its own CompressorImpl with head_dim_=512 (attention
        # head_dim), distinct from the indexer's head_dim_=128. Weights loaded
        # from attn.compressor.* (not attn.indexer.compressor.*).
        # wkv/wgate out = coff * head_dim (coff=2 for C4, 1 for C128);
        # norm=head_dim; ape=[4, coff*head_dim].
        cmp_hd = head_dim  # attention head_dim=512, NOT index_head_dim=128
        if compress_ratio > 1:
            cmp_coff = 2 if compress_ratio == 4 else 1
            cmp_out = cmp_coff * cmp_hd
            self.cmp_wkv = nn.Linear(
                cfg.hidden_size,
                cmp_out,
                bias=False,
                dtype=torch.float32,
                device=device,
            )
            self.cmp_wgate = nn.Linear(
                cfg.hidden_size,
                cmp_out,
                bias=False,
                dtype=torch.float32,
                device=device,
            )
            self.cmp_ape = nn.Parameter(torch.empty(compress_ratio, cmp_out, dtype=torch.float32, device=device))
            self.cmp_norm = RMSNorm(cmp_hd, cfg.rms_norm_eps, dtype=torch.float32, device=device)
            # The compressor kernel consumes these weights as BF16. Keep the
            # checkpoint-facing FP32 parameters and cache one converted copy
            # after loading instead of allocating one on every forward.
            self.register_buffer("_cmp_wkv_bf16", None, persistent=False)
            self.register_buffer("_cmp_wgate_bf16", None, persistent=False)
            self.register_buffer("_cmp_norm_bf16", None, persistent=False)

    def process_weights_after_loading(self) -> None:
        for m in (self.q_a_proj, self.kv_proj, self.q_b_proj):
            m.process_weights_after_loading()
        # Keep o_b in checkpoint [N, K] layout. Native DSAttention sends this
        # unquantized RowParallelLinear through F.linear(input, weight); the
        # generic NPU preparation transposes it to FRACTAL_NZ and selects a
        # different matmul accumulation path.
        if hasattr(self.o_a_proj, "process_weights_after_loading"):
            self.o_a_proj.process_weights_after_loading()
        if self.indexer is not None:
            self.indexer.process_weights_after_loading()
        if hasattr(self, "cmp_wkv"):
            self._cmp_wkv_bf16 = self.cmp_wkv.weight.to(torch.bfloat16).contiguous()
            self._cmp_wgate_bf16 = self.cmp_wgate.weight.to(torch.bfloat16).contiguous()
            self._cmp_norm_bf16 = self.cmp_norm.weight.to(torch.bfloat16).contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        backend = get_forward_context().attention_backend
        metadata = get_forward_context().metadata
        dsa = getattr(metadata, "dsa_metadata", None)
        kv_hidden = hidden

        # q/kv down + up + RoPE (matches run_dsv4_preprocess_fallback).
        # W8A8 path (C++ deepseek_sparse_attention.cpp:396-413): q_a_proj does
        # dynamic_quant internally; then rms_norm_dynamic_quant fuses the q_a
        # RMSNorm with a second dynamic quant, producing qr (int8) +
        # qr_pertoken_scale; q_b_proj consumes that pre-quantized qr (no
        # re-quant). The indexer reuses the same qr + qr_pertoken_scale in
        # build_query, so stash them on the backend for _run_indexer.
        q_a = self.q_a_proj(hidden)
        from xllm.python import kernels as _k

        qr, qr_pertoken_scale = _k.rms_norm_dynamic_quant(q_a, self.q_a_layernorm.weight, self.cfg.rms_norm_eps)
        q = self.q_b_proj.forward_quantized(qr, qr_pertoken_scale).view(num_tokens, self.num_heads_local, self.head_dim)
        q = _k.rms_norm(q, self.q_rms_gamma, self.cfg.rms_norm_eps)

        cos_sin = cos_sin_cache.index_select(0, positions.long())
        half = cos_sin.size(-1) // 2
        cos = cos_sin[..., :half].repeat_interleave(2, dim=-1).contiguous()
        sin = cos_sin[..., half:].repeat_interleave(2, dim=-1).contiguous()
        _k.npu_inplace_partial_rotary_mul(q, cos, sin, self.nope_head_dim, self.rope_head_dim)

        kv = self.kv_proj(kv_hidden)
        # kv_proj outputs head_dim = nope_head_dim + rope_head_dim; layernorm
        # the whole thing then split for RoPE (matches C++ run_dsv4_preprocess).
        kv = self.kv_a_layernorm(kv)
        kv_tensor = kv.view(kv_hidden.shape[0], 1, self.head_dim)
        kv_cos, kv_sin = cos, sin
        _k.npu_inplace_partial_rotary_mul(
            kv_tensor,
            kv_cos,
            kv_sin,
            self.nope_head_dim,
            self.rope_head_dim,
        )

        # Attach the compressor/indexer callbacks so the backend can invoke them.
        if self.indexer is not None and hasattr(backend, "attach_indexer"):
            backend.attach_indexer(self._run_indexer)
        if hasattr(backend, "attach_compressor"):
            backend.attach_compressor(self._run_compressor)

        # Pass hidden to backend so compressor/indexer can access it.
        backend._current_hidden = hidden
        backend._current_kv_hidden = kv_hidden
        # Stash the W8A8 pre-quantized query (int8 qr + per-token scale) for the
        # indexer build_query path (mirrors C++ select_qli's qr/qr_pertoken_scale).
        backend._current_qr = qr
        backend._current_qr_pertoken_scale = qr_pertoken_scale
        attn_out = backend.execute(q, kv_tensor, kv_tensor, self)
        # Native DSA rotates the attention output back before o_a/o_b.
        _k.npu_inplace_partial_rotary_mul(
            attn_out,
            cos,
            sin,
            self.nope_head_dim,
            self.rope_head_dim,
            inverse=True,
        )
        # Two-stage output projection (o_a -> grouped -> o_b).
        num_tokens = attn_out.size(0)
        out = attn_out.view(num_tokens, self.n_local_groups, -1)
        # Match C++ DSAttentionImpl exactly. A flattened F.linear is
        # mathematically equivalent but selects a different NPU accumulation
        # path and produces layer-by-layer BF16 drift.
        wo_a = self.o_a_proj.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o_low = torch.einsum("tgd,grd->tgr", out, wo_a)
        o = self.o_b_proj(o_low.reshape(num_tokens, -1))
        # o_b_proj is RowParallelLinear with reduce_results=True (default),
        # which internally calls tp_all_reduce. Do NOT call tp_all_reduce again
        # here — that would be a duplicate collective (the expert analysis
        # identified this as a cause of HCCL deadlock / 507015).
        return o

    def _run_compressor(self, layer_id, layer_cache, dsa, mapping, cmp_block_table, compress_ratio):
        """Cmp_kv compressor callback (attention-level, head_dim=512).

        Mirrors C++ DSAttentionImpl's compressor_->forward
        (deepseek_sparse_attention.cpp:840-852): uses the attention's own
        CompressorImpl with head_dim_=512, NOT the indexer's head_dim_=128.
        Weights from attn.compressor.* (loaded separately from indexer's).
        """
        if not hasattr(self, "cmp_wkv"):
            return None
        del cmp_block_table
        backend = get_forward_context().attention_backend
        hidden = getattr(backend, "_current_kv_hidden", None)
        if hidden is None:
            return None
        return _compress_kv(
            hidden,
            kv_state=layer_cache.compress_kv_state,
            score_state=layer_cache.compress_score_state,
            dsa=dsa,
            layer_id=layer_id,
            kv_cache_idx=mapping.kv_state_cache_idx,
            score_cache_idx=mapping.score_state_cache_idx,
            compress_ratio=compress_ratio,
            rope_head_dim=self.cfg.qk_rope_head_dim,
            wkv=self._cmp_wkv_bf16,
            wgate=self._cmp_wgate_bf16,
            ape=self.cmp_ape,
            norm_weight=self._cmp_norm_bf16,
            norm_eps=self.cfg.rms_norm_eps,
        )

    def _run_indexer(self, layer_id, layer_cache, dsa, mapping, q):
        """Indexer callback: returns top-k compressed block indices."""
        if self.indexer is not None:
            backend = get_forward_context().attention_backend
            qr = getattr(backend, "_current_qr", None)
            qr_pertoken_scale = getattr(backend, "_current_qr_pertoken_scale", None)
            hidden = getattr(backend, "_current_hidden", None)
            kv_hidden = getattr(backend, "_current_kv_hidden", hidden)
            return self.indexer.select_qli_dsv4(
                layer_id,
                layer_cache,
                dsa,
                mapping,
                q,
                qr,
                qr_pertoken_scale,
                hidden,
                kv_hidden,
            )
        return None


# ---------------------------------------------------------------------------
# Indexer + Compressor
# ---------------------------------------------------------------------------


class DeepseekV4Indexer(nn.Module):
    """DeepSeek-V4 indexer: Hadamard rotation + compressor + quant lightning.

    Faithful to the C++ ``DeepseekV4IndexerImpl`` (deepseek_v4_indexer.cpp):
    Hadamard-rotates the compressed key, runs the NSA compressor, scatters the
    compressed key into the paged index cache, then runs the quantized lightning
    indexer to pick top-k compressed blocks.
    """

    def __init__(self, cfg: DeepseekV4Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.topk = cfg.index_topk
        self.dtype = dtype
        self.hadamard_scale = cfg.index_head_dim**-0.5 if cfg.index_head_dim else 1.0
        # indexer q projection + scoring weights + compressor (K projection is
        # done by the compressor's wkv, so there is no separate wk/k_norm --
        # matches C++ DeepseekV4IndexerImpl).
        self.wq_b = W8A8DynamicLinear(
            cfg.q_lora_rank,
            self.n_head * self.head_dim,
            device,
            transpose_weight_after_loading=False,
        )
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_head, bias=False, dtype=dtype, device=device)
        # Compressor: wkv (fused wk+wv, unquantized f32) + wgate + ape + norm.
        # wkv out = 2*head_dim (cat of wk, wv); ape = [4, 2*head_dim].
        cmp_out = 2 * self.head_dim
        self.compressor_wkv = nn.Linear(cfg.hidden_size, cmp_out, bias=False, dtype=torch.float32, device=device)
        self.compressor_wgate = nn.Linear(cfg.hidden_size, cmp_out, bias=False, dtype=torch.float32, device=device)
        self.compressor_ape = nn.Parameter(torch.empty(4, cmp_out, dtype=torch.float32, device=device))
        self.compressor_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype=torch.float32, device=device)
        # Keep the checkpoint-facing FP32 parameters while reusing the BF16
        # tensors required by the compressor kernel across forwards.
        self.register_buffer("_compressor_wkv_bf16", None, persistent=False)
        self.register_buffer("_compressor_wgate_bf16", None, persistent=False)
        self.register_buffer("_compressor_norm_bf16", None, persistent=False)

    def process_weights_after_loading(self) -> None:
        self.wq_b.process_weights_after_loading()
        self._compressor_wkv_bf16 = self.compressor_wkv.weight.to(torch.bfloat16).contiguous()
        self._compressor_wgate_bf16 = self.compressor_wgate.weight.to(torch.bfloat16).contiguous()
        self._compressor_norm_bf16 = self.compressor_norm.weight.to(torch.bfloat16).contiguous()

    def select_qli_dsv4(
        self,
        layer_id,
        layer_cache,
        dsa,
        mapping,
        q,
        qr,
        qr_pertoken_scale,
        hidden,
        kv_hidden=None,
    ) -> torch.Tensor:
        """Quantized lightning indexer: returns top-k compressed block indices.

        Faithful port of C++ ``DeepseekV4IndexerImpl::select_qli``
        (deepseek_v4_indexer.cpp:393-555). The C++ path does NOT just read a
        pre-filled index cache -- it rebuilds it every call:
          build_query(qr, qr_pertoken_scale) -> q; partial RoPE + Hadamard(q);
          compress_kv(hidden) -> kv; Hadamard(kv); dynamic_quant_int8(kv) ->
          kv_quant + kv_scale; scatter kv_quant -> index_cache, kv_scale ->
          indexer_scale (via slot_mapping); then quant_lightning_indexer.
        Python mirrors that, otherwise index_cache/indexer_scale hold stale or
        uninitialized data and the returned topk addresses out-of-range blocks
        (507015 aicore in sparse_attn_sharedkv).
        """
        if kernels is None or layer_cache.index is None:
            return torch.empty(0, dtype=torch.int32)
        index_cache = layer_cache.index
        device = index_cache.device
        # --- build_query (C++ 310-335): wq_b W8A8 matmul over pre-quantized qr. ---
        if qr is not None and qr_pertoken_scale is not None:
            q_idx = self.wq_b.forward_quantized(qr, qr_pertoken_scale).view(-1, self.n_head, self.head_dim)
        else:
            q_idx = self.wq_b(qr).view(-1, self.n_head, self.head_dim)
        # --- partial RoPE on q (C++ 417-422): apply_partial_rope over
        # [rope_start_dim:rope_start_dim+rope_head_dim]. Uses the DEFAULT RoPE
        # table (cos/sin) indexed by positions, 2D [M, rope_dim], NOT the
        # compressed c4 table. Mirrors C++ apply_partial_rope ->
        # npu_inplace_partial_rotary_mul (deepseek_sparse_attention.cpp:151-190).
        rope_start_dim = max(self.head_dim - self.rope_dim, 0)
        cos_table = dsa.cos_table
        sin_table = dsa.sin_table
        if cos_table is not None and sin_table is not None and self.rope_dim > 0:
            cos_v = cos_table.reshape(-1, cos_table.size(-1)) if cos_table.dim() > 2 else cos_table
            sin_v = sin_table.reshape(-1, sin_table.size(-1)) if sin_table.dim() > 2 else sin_table
            # Per-token cos/sin indexed by positions: 2D [M, rope_dim/2] (Python
            # DeepseekYarnRotaryEmbedding stores half-dim cos/sin, NOT interleaved).
            pos = dsa.input_positions.to(device).reshape(-1).long()
            cos_sel = cos_v.index_select(0, pos).to(q_idx.dtype)  # [M, rope_dim/2]
            sin_sel = sin_v.index_select(0, pos).to(q_idx.dtype)
            # npu_inplace_partial_rotary_mul (interleave mode) expects cos/sin
            # [M, rope_dim] in C++ interleaved format: freqs.repeat_interleave(2)
            # (rotary_embedding_util.cpp:135-137). The half-dim cos/sin must be
            # repeat_interleave'd to full rope_dim. Local fix only -- do NOT change
            # the global DeepseekYarnRotaryEmbedding cache (the attention main path
            # uses _interleave_rope_with which consumes half-dim).
            if cos_sel.size(-1) * 2 == self.rope_dim:
                cos_sel = cos_sel.repeat_interleave(2, dim=-1).contiguous()
                sin_sel = sin_sel.repeat_interleave(2, dim=-1).contiguous()
            elif cos_sel.size(-1) != self.rope_dim:
                raise RuntimeError(f"QRoPE cos/sin dim mismatch: cos={cos_sel.shape}, rope_dim={self.rope_dim}")
            # In-place partial RoPE: modifies q_idx[...rope_start_dim:rope_dim]
            # via aclnnInplacePartialRotaryMul (interleave mode).
            from xllm.python import kernels as _pk

            _pk.npu_inplace_partial_rotary_mul(q_idx, cos_sel, sin_sel, rope_start_dim, self.rope_dim)
        # --- Hadamard rotation on q (C++ 423-424). ---
        hadamard = self._get_hadamard(device)
        q_idx = _rotate_hadamard(q_idx, hadamard, self.hadamard_scale)
        # --- build_weights(hidden) (C++ 337-340, 456). ---
        softmax_mul = (self.head_dim**-0.5) * (self.n_head**-0.5)
        weights = self.weights_proj(hidden) * softmax_mul
        # --- Rebuild index cache: compress_kv -> Hadamard -> quant -> scatter. ---
        kv = self._indexer_compress_kv(
            kv_hidden if kv_hidden is not None else hidden,
            layer_cache,
            dsa,
            mapping,
            layer_id,
        )
        if kv is not None and kv.numel() > 0:
            kv = _rotate_hadamard(kv, hadamard, self.hadamard_scale)
            kv_quant, kv_scale = kernels.dynamic_quant(kv)
            kv_scale = kv_scale.unsqueeze(-1).to(torch.float16)
            # Scatter kv_quant -> index_cache, kv_scale -> indexer_scale, by slot.
            slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.index_cache_idx)
            if slot is not None and slot.numel() > 0:
                _scatter_by_slot(index_cache, slot, kv_quant)
                if layer_cache.indexer_scale is not None:
                    _scatter_by_slot(layer_cache.indexer_scale, slot, kv_scale)
        # --- dynamic_quant_int8(q) -> q_quant (int8), q_scale (float16). ---
        q_quant, q_scale = kernels.dynamic_quant(q_idx)
        q_scale = q_scale.to(torch.float16)
        # key_dequant_scale = indexer_scale (written above), else ones fallback.
        key_dequant_scale = layer_cache.indexer_scale
        if key_dequant_scale is None or key_dequant_scale.numel() == 0:
            scale_sizes = list(index_cache.shape)
            scale_sizes[-1] = 1
            key_dequant_scale = torch.ones(scale_sizes, dtype=torch.float16, device=device)
        block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.index_cache_idx)
        query_seq_lens = dsa.actual_seq_lengths_query
        if query_seq_lens.dim() > 0 and query_seq_lens.size(0) > 1:
            query_seq_lens = query_seq_lens[1:]
        key_seq_lens = dsa.actual_seq_lengths_kv
        qli_metadata = dsa.qli_metadata
        # C++ packs the current forward's DSA metadata onto the runtime device
        # before building and invoking QLI. Python follows the same ownership
        # contract in DsaAttentionBackend.prepare_dsa_metadata_for_forward();
        # do not create per-call copies here because they hide lifecycle bugs.
        for name, tensor in (
            ("query_seq_lens", query_seq_lens),
            ("key_seq_lens", key_seq_lens),
            ("block_table", block_table),
            ("qli_metadata", qli_metadata),
        ):
            if tensor is None or tensor.numel() == 0:
                raise RuntimeError(f"QLI {name} must be defined and non-empty")
            if tensor.device != device:
                raise RuntimeError(f"QLI {name} must be on {device}, got {tensor.device}")
        topk = kernels.quant_lightning_indexer(
            q_quant,
            index_cache,
            weights.to(torch.float16),
            q_scale,
            key_dequant_scale,
            qli_metadata,
            query_seq_lens,
            key_seq_lens,
            block_table,
            self.topk,
            cmp_ratio=4,
        )
        return topk

    def _get_hadamard(self, device: torch.device) -> torch.Tensor:
        """Build (or cache) the Sylvester Hadamard matrix for head_dim.

        Mirrors C++ ``create_hadamard_matrix`` (index_head_dim_padded = next
        pow2 >= head_dim, normalize=False). Cached on the module.
        """
        cached = getattr(self, "_hadamard_matrix", None)
        if cached is not None and cached.device == device:
            return cached
        n = 1
        while n < self.head_dim:
            n <<= 1
        mat = torch.ones((1, 1), dtype=self.dtype, device=device)
        m = 1
        while m < n:
            top = torch.cat([mat, mat], 1)
            bottom = torch.cat([mat, -mat], 1)
            mat = torch.cat([top, bottom], 0)
            m <<= 1
        self._hadamard_matrix = mat
        return mat

    def _indexer_compress_kv(self, hidden, layer_cache, dsa, mapping, layer_id):
        """Run the compressor against the index-cache states (not cmp_kv).

        Mirrors C++ ``select_qli``'s ``compress_kv(kv_source, ...,
        &indexer_states, &indexer_block_tables, c4_cos, c4_sin, ...)``. Uses
        ``compress_index_kv_state`` / ``compress_index_score_state`` + the
        index-cache block tables, distinct from the cmp_kv compressor callback.
        """
        # The indexer owns different weights and cache state from the attention
        # compressor. Only the common kernel preparation and invocation is shared.
        return _compress_kv(
            hidden,
            kv_state=layer_cache.compress_index_kv_state,
            score_state=layer_cache.compress_index_score_state,
            dsa=dsa,
            layer_id=layer_id,
            kv_cache_idx=mapping.index_kv_state_cache_idx,
            score_cache_idx=mapping.index_score_state_cache_idx,
            compress_ratio=4,
            rope_head_dim=self.cfg.qk_rope_head_dim,
            wkv=self._compressor_wkv_bf16,
            wgate=self._compressor_wgate_bf16,
            ape=self.compressor_ape,
            norm_weight=self._compressor_norm_bf16,
            norm_eps=self.cfg.rms_norm_eps,
        )


def _rotate_hadamard(x: torch.Tensor, hadamard: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply the Hadamard transform to the last dim of ``x``.

    Faithful port of C++ ``rotate_activation_with_hadamard`` ->
    ``hadamard_transform_ref`` (deepseek_v4_indexer.cpp:61-87): pad the last
    dim to ``hadamard.size(0)`` (next pow2), matmul, slice back, scale.
    """
    if hadamard is None or hadamard.numel() == 0:
        return x
    dim = x.size(-1)
    x2d = x.reshape(-1, dim)
    if x2d.dtype != hadamard.dtype:
        raise RuntimeError(f"Hadamard dtype must match input: {hadamard.dtype} != {x2d.dtype}")
    dim_padded = hadamard.size(0)
    if dim != dim_padded:
        x2d = torch.nn.functional.pad(x2d, (0, dim_padded - dim))
    out = torch.nn.functional.linear(x2d, hadamard)
    out = out[:, :dim]
    out = out.reshape(x.shape)
    if scale != 1.0:
        out = out * scale
    return out


# ---------------------------------------------------------------------------
# DSV4 MoE with hash routing
# ---------------------------------------------------------------------------


class DeepseekV4MoE(nn.Module):
    """DeepSeek-V4 MoE with hash routing.

    Uses ``moe_gating_top_k_hash`` for routing (hash layers) or bias-based
    routing (non-hash layers), then ``grouped_moe_with_selected_experts`` for
    expert computation, plus shared experts. Mirrors C++ DeepseekV4GateImpl +
    FusedMoEImpl::forward_with_selected_experts.
    """

    def __init__(self, cfg: DeepseekV4Config, layer_id: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.topk = cfg.n_activated_experts
        self.num_total_experts = cfg.n_routed_experts
        self.routed_scaling = cfg.routed_scaling_factor
        self.n_hash_layers = cfg.n_hash_layers
        self.hash_layer = 0 <= layer_id < cfg.n_hash_layers
        self.scoring_func = cfg.scoring_func
        # EP: each rank holds num_experts_per_rank experts (not all).
        # Mirrors C++ FusedMoEImpl (fused_moe.cpp:420-421).
        # Use ep_size/ep_rank from config dict (set by PyCausalLM::build_config_dict).
        ep_size = cfg.ep_size if cfg.ep_size > 0 else cfg.tp_size
        ep_rank = cfg.ep_rank if cfg.ep_size > 0 else cfg.tp_rank
        # Attention TP and MoE TP are independent under orthogonal CP. C++
        # exposes both process groups; deriving MoE TP from attention TP makes
        # cp=2, ep=8 fail at construction time (attention TP=4, MoE TP=1).
        self.moe_tp_size = cfg.moe_tp_size
        self.moe_tp_rank = cfg.moe_tp_rank
        self.num_experts_per_rank = self.num_total_experts // ep_size
        self.start_expert_id = ep_rank * self.num_experts_per_rank
        inter_local = cfg.moe_intermediate_size // self.moe_tp_size
        self.inter_local = inter_local
        # DSV4 Flash checkpoints use W4A8_DYNAMIC routed experts.  The packed
        # storage shape is discovered from the checkpoint in load_weights;
        # keep expert tensors empty until then to avoid allocating the W8A8
        # fallback shape for all 43 layers.
        self.w4a8_dynamic = False

        # Gate weight [n_total_experts, hidden] float32 (replicated, not sharded).
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=torch.float32, device=device)
        # Hash table for hash layers [vocab, topk] int32 (None for non-hash).
        if self.hash_layer:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    cfg.vocab_size,
                    cfg.n_activated_experts,
                    dtype=torch.int32,
                    device=device,
                ),
                requires_grad=False,
            )
        else:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(
                    cfg.n_routed_experts,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )

        # Expert weights — EP sharded: each rank holds num_experts_per_rank experts
        # (not all n_routed_experts). Mirrors C++ FusedMoEImpl (fused_moe.cpp:605-624).
        nepr = self.num_experts_per_rank
        self.experts_w13 = nn.Parameter(torch.empty(0, dtype=torch.int8, device=device), requires_grad=False)
        self.experts_w2 = nn.Parameter(torch.empty(0, dtype=torch.int8, device=device), requires_grad=False)
        self.register_buffer("experts_w13_scale", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w13_scale_second", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w13_offset", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w2_scale", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w2_scale_second", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w2_offset", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w13_scale_bias", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("experts_w2_scale_bias", torch.empty(0, dtype=torch.float32, device=device))

        # Shared expert uses the orthogonal MoE TP group, matching C++
        # FusedMoEImpl. skip_tp_reduce keeps collective ordering in this class.
        shared_cfg = replace(cfg, tp_size=self.moe_tp_size, tp_rank=self.moe_tp_rank)
        self.shared_experts = DeepseekV3MLP(
            shared_cfg,
            cfg.moe_intermediate_size * cfg.n_shared_experts,
            dtype,
            device,
            skip_tp_reduce=True,
            swiglu_limit=cfg.swiglu_limit,
        )
        # C++ applies this optional gate only when its checkpoint weight is
        # present. Keep the module registered so the parameter ownership and
        # forward structure match FusedMoEImpl without changing ungated
        # checkpoints.
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype, device=device)
        self.shared_expert_gate_is_loaded = False

    def process_weights_after_loading(self) -> None:
        if self.w4a8_dynamic:
            # Match FusedMoEImpl::preprocess_w4a8_dynamic_weights.  Ascend
            # stores each W4 output pair in one int8 row; after transpose the
            # operator consumes NZ/INT32 packed weights and int64 bit-cast
            # per-channel scales plus the summed scale_bias.
            import torch_npu

            def _pack(weight: torch.Tensor) -> torch.Tensor:
                weight = torch_npu.npu_format_cast(weight.transpose(1, 2).contiguous(), 29)
                return weight.view(torch.int32).contiguous()

            def _scale(scale: torch.Tensor, scale_second: torch.Tensor | None) -> torch.Tensor:
                transposed = scale.transpose(1, 2).contiguous()
                if scale_second is not None and scale_second.numel() > 0:
                    groups = scale_second.transpose(1, 2).contiguous()
                    groups = groups.reshape(transposed.size(0), -1, transposed.size(2))
                    # C++ process_scale uses at::kHalf for the intermediate
                    # product before reinterpreting the FP32 bits.
                    transposed = (transposed.float() * groups.float()).to(torch.float16).float()
                # Native preprocessing bit-casts FP32 bits into int64 slots;
                # doing the reinterpretation on CPU avoids backend-specific
                # restrictions on view(dtype) for NPU tensors.
                cpu = transposed.to("cpu")
                # Match fp32_bits_to_int64_tensor: zero-extend the raw
                # uint32 payload into an int64 slot (avoid sign extension).
                bits = cpu.view(torch.int32).to(torch.int64)
                bits = bits & 0xFFFFFFFF
                return bits.to(scale.device).contiguous()

            self.experts_w13.data = _pack(self.experts_w13)
            self.experts_w2.data = _pack(self.experts_w2)
            self.experts_w13_scale.data = _scale(self.experts_w13_scale, self.experts_w13_scale_second)
            self.experts_w2_scale.data = _scale(self.experts_w2_scale, self.experts_w2_scale_second)
            self.experts_w13_scale_bias.data = self.experts_w13_scale_bias.transpose(1, 2).contiguous().sum(1)
            self.experts_w2_scale_bias.data = self.experts_w2_scale_bias.transpose(1, 2).contiguous().sum(1)
        else:
            # W8A8 grouped path: [expert, out, in] -> [expert, in, out].
            self.experts_w13.data = self.experts_w13.data.transpose(1, 2).contiguous()
            self.experts_w2.data = self.experts_w2.data.transpose(1, 2).contiguous()
            self.experts_w13_scale.data = self.experts_w13_scale.data.squeeze(-1).contiguous()
            self.experts_w2_scale.data = self.experts_w2_scale.data.squeeze(-1).contiguous()
        self.shared_experts.gate_up_proj.process_weights_after_loading()
        self.shared_experts.down_proj.process_weights_after_loading()

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        from xllm.python import kernels

        # Prepare input_ids: reshape to 1D + move to hidden's device (C++ :202-216).
        gate_input_ids = None
        if input_ids is not None and input_ids.numel() > 0:
            flat_ids = input_ids.reshape(-1).to(hidden.device)
            token_count = flat_ids.size(0)
            hidden_rows = hidden.size(0)
            if token_count == hidden_rows:
                gate_input_ids = flat_ids
            elif token_count > 0 and hidden_rows % token_count == 0:
                repeat_factor = hidden_rows // token_count
                gate_input_ids = flat_ids.unsqueeze(1).repeat(1, repeat_factor).reshape(hidden_rows)

        # 1) Gate: compute logits + moe_gating_top_k_hash.
        gate_input = hidden.to(torch.float32)
        logits = self.gate(gate_input)
        norm_type = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}.get(self.scoring_func, 2)
        renorm = 0 if norm_type == 2 else 1

        if self.hash_layer and hasattr(self, "tid2eid") and gate_input_ids is not None:
            topk_weights, topk_idx, _ = kernels.moe_gating_top_k_hash(
                x=logits,
                k=self.topk,
                bias=None,
                input_ids=gate_input_ids,
                tid2eid=self.tid2eid,
                k_group=1,
                group_count=1,
                routed_scaling_factor=self.routed_scaling,
                eps=1e-20,
                group_select_mode=1,
                renorm=renorm,
                norm_type=norm_type,
                out_flag=False,
            )
        elif self.hash_layer:
            raise RuntimeError("DeepSeek-V4 hash gate requires input_ids for routing")
        else:
            bias = getattr(self, "e_score_correction_bias", None)
            topk_weights, topk_idx, _ = kernels.moe_gating_top_k_hash(
                x=logits,
                k=self.topk,
                bias=bias,
                input_ids=None,
                tid2eid=None,
                k_group=1,
                group_count=1,
                routed_scaling_factor=self.routed_scaling,
                eps=1e-20,
                group_select_mode=1,
                renorm=renorm,
                norm_type=norm_type,
                out_flag=False,
            )

        # 2) EP: zero out non-local expert weights (C++ fused_moe.cpp:843-850).
        ep_size = self.cfg.ep_size if self.cfg.ep_size > 0 else self.cfg.tp_size
        if ep_size > 1:
            local_mask = (topk_idx >= self.start_expert_id) & (
                topk_idx < self.start_expert_id + self.num_experts_per_rank
            )
            topk_weights = topk_weights * local_mask.to(topk_weights.dtype)

        # 3) Expert computation with pre-selected routing (EP-sharded).
        if self.w4a8_dynamic:
            # W4A8_DYNAMIC uses the same two grouped GEMMs as C++
            # forward_expert: W4 GMM1 -> SwiGLU -> dynamic int8 -> W4 GMM2.
            sorted_hidden, expanded_row_idx, group_list, per_token = torch_npu.npu_moe_init_routing_v2(
                hidden,
                topk_idx.to(torch.int32),
                scale=None,
                active_num=hidden.size(0) * topk_idx.size(-1),
                expert_num=self.num_total_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[self.start_expert_id, self.start_expert_id + self.num_experts_per_rank],
                # Keep expanded activations in model dtype; C++ W4A8 path
                # performs dynamic quantization explicitly after dispatch.
                quant_mode=-1,
            )
            sorted_hidden_i8, input_scale = kernels.dynamic_quant(sorted_hidden)
            group_list = group_list[: self.num_experts_per_rank].to(torch.int64)
            gemm1 = torch.ops.npu.npu_grouped_matmul(
                x=[sorted_hidden_i8],
                weight=[self.experts_w13],
                bias=[self.experts_w13_scale_bias],
                scale=[self.experts_w13_scale],
                per_token_scale=[input_scale],
                split_item=2,
                group_type=0,
                group_list_type=1,
                group_list=group_list,
                output_dtype=torch.bfloat16,
            )[0]
            gate, up = gemm1.chunk(2, dim=-1)
            # Match xllm::layer::Activation::swiglu_with_clamp: perform
            # clamp/SILU/multiply in FP32, then cast back to GEMM1 dtype.
            activation_dtype = gemm1.dtype
            gate = gate.float()
            up = up.float()
            limit = self.cfg.swiglu_limit
            if 0.0 < limit < 1_000_000.0:
                gate = gate.clamp_max(limit)
                up = up.clamp(min=-limit, max=limit)
            act = (torch_npu.npu_silu(gate) * up).to(activation_dtype)
            # C++ FusedMoE normalizes the fused SwiGLU result to the dtype
            # accepted by dynamic_quant before producing the W4A8 GEMM2
            # activation scale. Keep this explicit so the Python path does
            # not depend on the backend's implicit promotion rules.
            if act.dtype != torch.bfloat16:
                act = act.to(torch.bfloat16)
            act_i8, act_scale = kernels.dynamic_quant(act)
            gemm2 = torch.ops.npu.npu_grouped_matmul(
                x=[act_i8],
                weight=[self.experts_w2],
                bias=[self.experts_w2_scale_bias],
                scale=[self.experts_w2_scale],
                per_token_scale=[act_scale],
                split_item=2,
                group_type=0,
                group_list_type=1,
                group_list=group_list,
                output_dtype=hidden.dtype,
            )[0]
            local_mask = (topk_idx >= self.start_expert_id) & (
                topk_idx < self.start_expert_id + self.num_experts_per_rank
            )
            routed_out = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=gemm2,
                sorted_indices=expanded_row_idx.abs(),
                probs=(topk_weights * local_mask).to(gemm2.dtype),
            )
        else:
            routed_out = kernels.grouped_moe_with_selected_experts(
                hidden,
                topk_weights,
                topk_idx.to(torch.int32),
                self.experts_w13,
                self.experts_w2,
                self.experts_w13_scale,
                self.experts_w2_scale,
                self.experts_w13_offset,
                self.experts_w2_offset,
                self.num_total_experts,
                self.start_expert_id,
                self.num_experts_per_rank,
                self.cfg.swiglu_limit,
            )
        # 4) Shared experts + C++-ordered TP/EP reductions.
        shared_out = self.shared_experts(hidden)
        if self.shared_expert_gate_is_loaded:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden))
            shared_out = shared_gate * shared_out

        output = self._reduce_moe_outputs(routed_out, shared_out)
        return output

    def _reduce_moe_outputs(self, routed_out: torch.Tensor, shared_out: torch.Tensor) -> torch.Tensor:
        """Reduce routed/shared results in the C++ ``FusedMoEImpl`` order.

        With both MoE TP and EP enabled, routed and shared outputs are partial
        on different dimensions.  C++ reduces each partial over MoE TP first,
        reduces routed over MoE EP, then reduces shared over MoE TP before
        adding the two results (fused_moe.cpp:2158-2181).  Keeping this order
        also keeps every rank in both process groups in the same collective
        sequence.
        """
        ep_size = self.cfg.ep_size if self.cfg.ep_size > 0 else self.cfg.tp_size
        if distributed is None and (ep_size > 1 or self.moe_tp_size > 1):
            raise RuntimeError("Python distributed collectives are unavailable")

        if ep_size > 1:
            if self.moe_tp_size > 1:
                distributed.moe_tp_all_reduce(routed_out)
            distributed.moe_ep_all_reduce(routed_out)
            if self.moe_tp_size > 1:
                distributed.moe_tp_all_reduce(shared_out)
            return routed_out + shared_out

        out = routed_out + shared_out
        if self.moe_tp_size > 1:
            # EP1: routed and shared partials can be combined before one
            # reduction, matching C++'s reduce(a + b) fast path.
            distributed.moe_tp_all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------


class DeepseekV4DecoderLayer(nn.Module):
    """DeepSeek-V4 decoder layer: HyperConnection(attn) + HyperConnection(ffn)."""

    def __init__(
        self,
        cfg: DeepseekV4Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hc = DeepseekV4HyperConnection(cfg, dtype, device)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        self.self_attn = DeepseekV4Attention(cfg, layer_id, dtype, device)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        # Dense vs MoE by first_k_dense_replace / moe_layer_freq.
        is_dense = (layer_id < cfg.first_k_dense_replace) or (
            cfg.moe_layer_freq > 1 and layer_id % cfg.moe_layer_freq != 0
        )
        if is_dense:
            self.mlp = DeepseekV3MLP(
                cfg,
                cfg.moe_intermediate_size,
                dtype,
                device,
                swiglu_limit=cfg.swiglu_limit,
            )
        else:
            self.mlp = DeepseekV4MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Match DeepseekV4DecoderLayerImpl::forward exactly: HyperConnection
        # selects the 2D sub-input first, then RMSNorm is applied to that input.
        residual_attn = hidden
        attn_input, post_attn, comb_attn = self.hc.hc_pre(
            hidden,
            self.hc.hc_attn_fn,
            self.hc.hc_attn_scale,
            self.hc.hc_attn_base,
        )
        attn_input = self.input_layernorm(attn_input)
        attn_output = self.self_attn(attn_input, positions, cos_sin_cache)
        hidden = self.hc.hc_post(attn_output, residual_attn, post_attn, comb_attn)

        residual_ffn = hidden
        ffn_input, post_ffn, comb_ffn = self.hc.hc_pre(
            hidden,
            self.hc.hc_ffn_fn,
            self.hc.hc_ffn_scale,
            self.hc.hc_ffn_base,
        )
        ffn_input = self.post_attention_layernorm(ffn_input)
        ffn_output = self.mlp(ffn_input, input_ids) if isinstance(self.mlp, DeepseekV4MoE) else self.mlp(ffn_input)
        hidden = self.hc.hc_post(ffn_output, residual_ffn, post_ffn, comb_ffn)
        # Native C++ resets its optional residual at the start of every layer.
        return hidden, None


class DeepseekV4Model(nn.Module):
    """DeepSeek-V4 transformer body."""

    def __init__(self, cfg: DeepseekV4Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.cp_size > 1:
            raise NotImplementedError("DeepSeek-V4 Python CP is reserved for the CP context PR")
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList([DeepseekV4DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=dtype, device=device)
        # Model-level HyperConnection head: merges the hc_mult residual streams
        # back into a single hidden vector before the final norm. Unlike the
        # per-layer hc_pre (which uses mix_hc=(2+mult)*mult), the head uses a
        # plain [hc_mult, hc_dim] hc_fn + [hc_mult] base + [1] scale.
        hc_dim = cfg.hc_mult * cfg.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(cfg.hc_mult, hc_dim, dtype=torch.float32, device=device))
        self.hc_head_base = nn.Parameter(torch.empty(cfg.hc_mult, dtype=torch.float32, device=device))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32, device=device))
        # Native C++ falls back to max_position_embeddings when the flat
        # rope_scaling_original_max_position_embeddings ModelArgs field is 0.
        # The DSV4 loader currently leaves that field at 0, so old_context_len
        # is 1048576 for this checkpoint, not nested rope_scaling's 65536.
        native_old_context_len = cfg.max_position_embeddings
        self.rotary = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            native_old_context_len,
            dtype=dtype,
            device=device,
        )
        # Per-ratio compressed RoPE caches (C++ DeepseekV4RotaryEmbedding c4/c128
        # groups). Same YaRN inv_freq as the default cache but with
        # compress_rope_theta (config=160000) and NO mscale amplitude (C++
        # create_cos_sin_cache does (void)mscale). mscale=1/mscale_all_dim=1 makes
        # rope_mscale = get_mscale(s,1)/get_mscale(s,1) = 1.0 (no amplitude).
        self.compress_rotary_c4 = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.compress_rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            native_old_context_len,
            dtype=dtype,
            device=device,
        )
        self.compress_rotary_c128 = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.compress_rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            native_old_context_len,
            dtype=dtype,
            device=device,
        )

    def attach_rope_tables_to_backend(
        self,
        backend,
        positions: torch.Tensor,
        graph_bt_cols: int = 0,
        metadata=None,
    ) -> None:
        """Attach default + per-ratio compressed RoPE caches to the backend.

        Called inside model forward after embedding, matching the C++ DSA
        metadata construction order. Default cache uses rope_theta; c4/c128 use
        compress_rope_theta with no mscale.
        """
        if backend is None or not hasattr(backend, "attach_rope_tables"):
            raise RuntimeError("DeepSeek-V4 requires a CSA backend with attach_rope_tables")
        positions = positions.to(torch.int64).contiguous()
        backend.attach_rope_tables(
            positions,
            self.rotary.cos_sin_cache,
            graph_bt_cols=graph_bt_cols,
            csa_cos_sin=self.compress_rotary_c4.cos_sin_cache,
            hca_cos_sin=self.compress_rotary_c128.cos_sin_cache,
            metadata=metadata,
        )

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        """Final HyperConnection head.

        Matches C++ DeepseekV4ModelImpl::hc_head: this final merge is not the
        per-layer Sinkhorn hc_pre kernel. Its checkpoint weights have shapes
        hc_head_fn=[hc_mult, hc_mult*hidden], hc_head_base=[hc_mult], and
        hc_head_scale=[1].
        """
        x_float = x.to(torch.float32)
        x_flatten = x_float.flatten(-2, -1)
        rsqrt = torch.rsqrt(x_flatten.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        mixes = torch.matmul(x_flatten, self.hc_head_fn.transpose(0, 1))
        mixes = mixes * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.cfg.hc_eps
        y = (pre.unsqueeze(-1) * x_float).sum(-2)
        return y.to(x.dtype)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        backend.reset_forward(metadata)
        self.attach_rope_tables_to_backend(backend, positions, metadata=metadata)
        prepare_dsa = getattr(backend, "prepare_dsa_metadata_for_forward", None)
        if prepare_dsa is None:
            raise RuntimeError("DeepSeek-V4 requires prepare_dsa_metadata_for_forward")
        prepare_dsa(metadata)
        if self.cfg.cp_size > 1:
            raise NotImplementedError("DeepSeek-V4 Python CP is reserved for the CP context PR")
        # Expand hidden into hc_mult parallel residual streams for the
        # HyperConnection decoder layers (C++ flat_hc does this reshape).
        hidden = hidden.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        residual: torch.Tensor | None = None
        for layer_id, layer in enumerate(self.layers):
            compress_ratio = self.cfg.compress_ratios[layer_id] if layer_id < len(self.cfg.compress_ratios) else 1
            if compress_ratio == 4:
                layer_cos_sin_cache = self.compress_rotary_c4.cos_sin_cache
            elif compress_ratio == 128:
                layer_cos_sin_cache = self.compress_rotary_c128.cos_sin_cache
            else:
                layer_cos_sin_cache = cos_sin_cache
            # The C++ model updates DSAMetadata::cos/sin for every layer before
            # entering the decoder. Layer 2 is the first C4 layer and must not
            # reuse the ratio-1 RoPE table selected during metadata preparation.
            select_layer_rope = getattr(backend, "select_dsa_layer_rope", None)
            if select_layer_rope is None:
                raise RuntimeError("DeepSeek-V4 requires select_dsa_layer_rope")
            select_layer_rope(layer_id, layer_cos_sin_cache, metadata)
            hidden, residual = layer(
                hidden,
                residual,
                positions,
                layer_cos_sin_cache,
                input_ids,
            )
            record_layer_event(layer_id)
        # hc_head: merge the hc_mult streams back into a single hidden vector.
        merged = self._hc_head(residual if residual is not None else hidden)
        hidden = self.norm(merged, None)
        return hidden


class DeepseekV4ForCausalLM(PyModelBase):
    """DeepSeek-V4 causal LM driven by the C++ PyCausalLM bridge."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV4Config.from_dict(config)
        if self.cfg.dp_size > 1:
            raise NotImplementedError("DeepSeek-V4 Python does not support dp_size > 1")
        if self.cfg.cp_size > 1:
            raise NotImplementedError("DeepSeek-V4 Python CP is reserved for the CP context PR")
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = DeepseekV4Model(self.cfg, dtype, device)
        tp = self.cfg.tp_size
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts, tp_rank: int, tp_size: int) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        def _has(name: str) -> bool:
            return loader.has(name)

        def _w8a8(ckpt_prefix: str, param_prefix: str, shard_dims: dict | None = None) -> None:
            """Load a W8A8-dynamic projection.

            DSV4's checkpoint stores ``weight`` (int8) + ``weight_scale`` +
            ``weight_offset`` (per-output-channel), which is the
            ``W8A8DynamicLinear`` format -- NOT the static deq_scale/quant_bias
            format of ``W8A8StaticLinear``.
            """
            for suffix in ("weight", "weight_scale", "weight_offset"):
                ckpt_key = ckpt_prefix + "." + suffix
                if not _has(ckpt_key):
                    continue
                t = loader.load_tensor(ckpt_key)
                dim = (shard_dims or {}).get(suffix)
                if dim is not None:
                    t = loader.shard(t, dim=dim)
                loader.copy_in(param_prefix + "." + suffix, t)

        # --- Embedding (checkpoint: embed.weight). ---
        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.load_tensor("embed.weight"), dim=1),
        )

        # --- Per-layer weights (checkpoint: layers.N.<...>). ---
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."  # checkpoint prefix
            pm = f"model.layers.{i}."  # parameter prefix
            attn = self.model.layers[i].self_attn
            # Attention W8A8 projections (ckpt name -> module name).
            _w8a8(ck + "attn.wq_a", pm + "self_attn.q_a_proj")
            _w8a8(
                ck + "attn.wq_b",
                pm + "self_attn.q_b_proj",
                {"weight": 0, "weight_scale": 0, "weight_offset": 0},
            )
            _w8a8(ck + "attn.wkv", pm + "self_attn.kv_proj")
            # o_a/o_b are bf16 (unquantized) column/row-parallel weights, not W8A8.
            loader.copy_in(
                pm + "self_attn.o_a_proj.weight",
                loader.shard(loader.load_tensor(ck + "attn.wo_a.weight"), dim=0),
            )
            loader.copy_in(
                pm + "self_attn.o_b_proj.weight",
                loader.shard(loader.load_tensor(ck + "attn.wo_b.weight"), dim=1),
            )
            # Attention layernorms + sink.
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.load_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.load_tensor(ck + "attn.kv_norm.weight"),
            )
            # attn_sink (parameter): load either bare tensor or .weight form.
            sink_key = ck + "attn.attn_sink"
            if not _has(sink_key):
                sink_key = ck + "attn.attn_sink.weight"
            if _has(sink_key):
                sink = loader.load_tensor(sink_key)
                if sink.dim() == 1 and sink.size(0) == cfg.n_heads and cfg.tp_size > 1:
                    shard_size = cfg.n_heads // cfg.tp_size
                    sink = sink.narrow(0, cfg.tp_rank * shard_size, shard_size)
                loader.copy_in(pm + "self_attn.attn_sink", sink)
                attn.attn_sink_loaded = True
            # Layer layernorms (ckpt attn_norm/ffn_norm -> input/post_attention).
            loader.copy_in(
                pm + "input_layernorm.weight",
                loader.load_tensor(ck + "attn_norm.weight"),
            )
            loader.copy_in(
                pm + "post_attention_layernorm.weight",
                loader.load_tensor(ck + "ffn_norm.weight"),
            )
            # HyperConnection weights (ckpt layers.N.hc_* -> model.layers.N.hc.hc_*).
            for part in ("attn", "ffn"):
                for suffix in ("fn", "scale", "base"):
                    name = f"hc_{part}_{suffix}"
                    loader.copy_in(pm + "hc." + name, loader.load_tensor(ck + name))
            # Indexer weights (ckpt layers.N.attn.indexer.*).
            if attn.indexer is not None and _has(ck + "attn.indexer.wq_b.weight"):
                # Indexer wq_b (ReplicatedLinear, not sharded) + weights_proj.
                _w8a8(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
                loader.copy_in(
                    pm + "self_attn.indexer.weights_proj.weight",
                    loader.load_tensor(ck + "attn.indexer.weights_proj.weight"),
                )
                # Compressor sub-module: wkv (unquantized f32 fused wk+wv) +
                # wgate + ape + norm (all f32, not W8A8).
                loader.copy_in(
                    pm + "self_attn.indexer.compressor_wkv.weight",
                    loader.load_tensor(ck + "attn.indexer.compressor.wkv.weight"),
                )
                loader.copy_in(
                    pm + "self_attn.indexer.compressor_wgate.weight",
                    loader.load_tensor(ck + "attn.indexer.compressor.wgate.weight"),
                )
                loader.copy_in(
                    pm + "self_attn.indexer.compressor_ape",
                    loader.load_tensor(ck + "attn.indexer.compressor.ape"),
                )
                loader.copy_in(
                    pm + "self_attn.indexer.compressor_norm.weight",
                    loader.load_tensor(ck + "attn.indexer.compressor.norm.weight"),
                )
            # Attention-level cmp_kv compressor (head_dim=512, separate from the
            # indexer compressor at head_dim=128). Ckpt: attn.compressor.*.
            # Mirrors C++ DSAttentionImpl compressor_ (compressor.cpp:590-597).
            if hasattr(attn, "cmp_wkv") and _has(ck + "attn.compressor.wkv.weight"):
                _w = loader.load_tensor(ck + "attn.compressor.wkv.weight")
                loader.copy_in(pm + "self_attn.cmp_wkv.weight", _w)
                loader.copy_in(
                    pm + "self_attn.cmp_wgate.weight",
                    loader.load_tensor(ck + "attn.compressor.wgate.weight"),
                )
                loader.copy_in(
                    pm + "self_attn.cmp_ape",
                    loader.load_tensor(ck + "attn.compressor.ape"),
                )
                loader.copy_in(
                    pm + "self_attn.cmp_norm.weight",
                    loader.load_tensor(ck + "attn.compressor.norm.weight"),
                )
            attn.process_weights_after_loading()
            # MoE / dense MLP weights. MoE layers use hash routing
            # (gate.weight + gate.tid2eid) and per-expert w1/w2/w3, while dense
            # layers use the fused W8A8 MLP loader below.
            mlp = self.model.layers[i].mlp
            if hasattr(mlp, "experts_w13") and _has(ck + "ffn.experts.0.w1.weight"):
                self._load_dsv4_moe(loader, ck, pm, i)
                mlp.process_weights_after_loading()
            elif isinstance(mlp, DeepseekV3MLP):
                self._load_dsv4_dense_mlp(loader, ck, pm, mlp)

        # --- Final norm + hc_head + lm_head. ---
        loader.copy_in("model.norm.weight", loader.load_tensor("norm.weight"))
        loader.copy_in("model.hc_head_fn", loader.load_tensor("hc_head_fn"))
        loader.copy_in("model.hc_head_base", loader.load_tensor("hc_head_base"))
        loader.copy_in("model.hc_head_scale", loader.load_tensor("hc_head_scale"))
        # Match LlmForCausalLMImplBase's non-tied output-head lookup order.  The
        # Flash checkpoint uses ``head.weight`` rather than ``lm_head.weight``.
        lm_head_key = next(
            (
                name
                for name in (
                    "lm_head.weight",
                    "model.lm_head.weight",
                    "model.head.weight",
                    "head.weight",
                )
                if _has(name)
            ),
            None,
        )
        assert lm_head_key is not None, "checkpoint output-head weight not found"
        loader.copy_in(
            "lm_head.weight",
            loader.shard(loader.load_tensor(lm_head_key), dim=0),
        )

    @staticmethod
    def _load_dsv4_dense_mlp(
        loader: W8A8WeightLoader,
        checkpoint_prefix: str,
        parameter_prefix: str,
        mlp: DeepseekV3MLP,
    ) -> None:
        """Load DSV4 ``w1/w3/w2`` tensors into a fused dense W8A8 MLP."""
        gate_up_prefix = parameter_prefix + "mlp.gate_up_proj."
        down_prefix = parameter_prefix + "mlp.down_proj."
        for suffix in ("weight", "weight_scale", "weight_offset"):
            w1 = loader.shard(loader.load_tensor(checkpoint_prefix + "ffn.w1." + suffix), dim=0)
            w3 = loader.shard(loader.load_tensor(checkpoint_prefix + "ffn.w3." + suffix), dim=0)
            loader.copy_in(gate_up_prefix + suffix, torch.cat([w1, w3], dim=0))
            w2 = loader.load_tensor(checkpoint_prefix + "ffn.w2." + suffix)
            if suffix == "weight":
                w2 = loader.shard(w2, dim=1)
            loader.copy_in(down_prefix + suffix, w2)
        mlp.process_weights_after_loading()

    def _load_dsv4_moe(self, loader, ck: str, pm: str, layer_id: int) -> None:
        """Stage DSV4 MoE weights for DeepseekV4MoE (hash routing + EP sharding).

        Loads gate.weight + tid2eid (hash layers) + per-expert w1/w2/w3
        (only local EP experts, fused into w13=w1+w3) + shared_experts.
        Mirrors C++ FusedMoEImpl::load_experts (fused_moe.cpp:1938+).
        """

        def _has(name: str) -> bool:
            return loader.has(name)

        cfg = self.cfg
        mlp = self.model.layers[layer_id].mlp
        # Gate weight [n_total_experts, hidden] float32 (replicated, not EP-sharded).
        loader.copy_in(pm + "mlp.gate.weight", loader.load_tensor(ck + "ffn.gate.weight"))
        if mlp.hash_layer:
            # C++ DeepseekV4GateImpl requires tid2eid for every hash layer.
            tid2eid_key = ck + "ffn.gate.tid2eid"
            if not _has(tid2eid_key):
                tid2eid_key += ".weight"
            assert _has(tid2eid_key), f"hash gate checkpoint tensor not found: {tid2eid_key}"
            loader.copy_in(pm + "mlp.tid2eid", loader.load_tensor(tid2eid_key))
        else:
            # Match DeepseekV4GateImpl::load_state_dict: the correction bias is
            # mandatory for non-hash routing, with the legacy key as fallback.
            bias_key = ck + "ffn.gate.bias"
            if not _has(bias_key):
                bias_key = ck + "ffn.gate.e_score_correction_bias"
            assert _has(bias_key), (
                f"non-hash gate checkpoint tensor not found: {ck}ffn.gate.bias (or e_score_correction_bias)"
            )
            loader.copy_in(
                pm + "mlp.e_score_correction_bias",
                loader.load_tensor(bias_key),
            )
        # Per-expert w1+w3 -> fused w13, w2 -> w2 (int8 + scale).
        # EP: only load local experts [start_expert_id, start_expert_id + num_experts_per_rank).
        tp = mlp.moe_tp_size
        tp_rank = mlp.moe_tp_rank
        start = mlp.start_expert_id
        nepr = mlp.num_experts_per_rank
        w13 = self.get_parameter(pm + "mlp.experts_w13")
        w2 = self.get_parameter(pm + "mlp.experts_w2")
        w13_scale = self.get_buffer(pm + "mlp.experts_w13_scale")
        w13_scale_second = self.get_buffer(pm + "mlp.experts_w13_scale_second")
        w2_scale = self.get_buffer(pm + "mlp.experts_w2_scale")
        w2_scale_second = self.get_buffer(pm + "mlp.experts_w2_scale_second")
        w13_offset = self.get_buffer(pm + "mlp.experts_w13_offset")
        w2_offset = self.get_buffer(pm + "mlp.experts_w2_offset")
        w13_scale_bias = self.get_buffer(pm + "mlp.experts_w13_scale_bias")
        w2_scale_bias = self.get_buffer(pm + "mlp.experts_w2_scale_bias")

        def _shard_fused(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            """Shard w1/w3 independently, matching LOAD_MOE_FUSED_WEIGHT."""
            if tp > 1:
                first = loader.shard(first, dim=0, world=tp, rank=tp_rank)
                second = loader.shard(second, dim=0, world=tp, rank=tp_rank)
            return torch.cat([first, second], dim=0)

        probe = ck + f"ffn.experts.{start}."
        probe_w1 = loader.load_tensor(probe + "w1.weight")
        # W4A8 stores two 4-bit output values per int8 row.  The checkpoint
        # therefore has half as many physical rows as the logical expert
        # intermediate size; W8A8 has one row per logical output value.
        is_w4a8 = probe_w1.shape[0] * 2 == cfg.moe_intermediate_size
        mlp.w4a8_dynamic = is_w4a8
        device = w13.device
        if is_w4a8:
            w13.data = torch.empty(
                nepr, cfg.moe_intermediate_size // tp, cfg.hidden_size, dtype=torch.int8, device=device
            )
            w2.data = torch.empty(
                nepr, cfg.hidden_size // 2, cfg.moe_intermediate_size // tp, dtype=torch.int8, device=device
            )
            w13_scale.data = torch.empty(
                nepr, 2 * (cfg.moe_intermediate_size // tp), 1, dtype=torch.float32, device=device
            )
            w2_scale.data = torch.empty(nepr, cfg.hidden_size, 1, dtype=torch.float32, device=device)
            w13_offset.data = torch.empty_like(w13_scale)
            w2_offset.data = torch.empty_like(w2_scale)
            w13_scale_bias.data = torch.empty(
                nepr, 2 * (cfg.moe_intermediate_size // tp), 1, dtype=torch.float32, device=device
            )
            sb2 = loader.load_tensor(probe + "w2.scale_bias")
            if tp > 1:
                sb2 = loader.shard(sb2, dim=1, world=tp, rank=tp_rank)
            w2_scale_bias.data = torch.empty(nepr, *sb2.shape, dtype=torch.float32, device=device)
        else:
            inter_local = cfg.moe_intermediate_size // tp
            w13.data = torch.empty(nepr, 2 * inter_local, cfg.hidden_size, dtype=torch.int8, device=device)
            w2.data = torch.empty(nepr, cfg.hidden_size, inter_local, dtype=torch.int8, device=device)
            w13_scale.data = torch.empty(nepr, 2 * inter_local, 1, dtype=torch.float32, device=device)
            w2_scale.data = torch.empty(nepr, cfg.hidden_size, 1, dtype=torch.float32, device=device)
            w13_offset.data = torch.zeros_like(w13_scale)
            w2_offset.data = torch.zeros_like(w2_scale)
        for local_idx in range(nepr):
            global_id = start + local_idx
            e = ck + f"ffn.experts.{global_id}."
            w1 = probe_w1 if local_idx == 0 else loader.load_tensor(e + "w1.weight")
            w3 = loader.load_tensor(e + "w3.weight")
            w13_j = _shard_fused(w1, w3)
            w2_j = loader.load_tensor(e + "w2.weight")
            if tp > 1:
                w2_j = loader.shard(w2_j, dim=1, world=tp, rank=tp_rank)
            w13[local_idx].copy_(w13_j.to(w13.dtype))
            w2[local_idx].copy_(w2_j.to(w2.dtype))
            if _has(e + "w1.weight_scale"):
                s1 = loader.load_tensor(e + "w1.weight_scale")
                s3 = loader.load_tensor(e + "w3.weight_scale") if _has(e + "w3.weight_scale") else s1
                s13 = _shard_fused(s1, s3)
                w13_scale[local_idx].copy_(s13)
            if is_w4a8:
                if _has(e + "w1.weight_scale_second"):
                    s1_second = loader.load_tensor(e + "w1.weight_scale_second")
                    s3_second = (
                        loader.load_tensor(e + "w3.weight_scale_second")
                        if _has(e + "w3.weight_scale_second")
                        else s1_second
                    )
                    s13_second = _shard_fused(s1_second, s3_second)
                    if w13_scale_second.numel() == 0:
                        w13_scale_second.data = torch.empty(nepr, *s13_second.shape, dtype=torch.float32, device=device)
                    w13_scale_second[local_idx].copy_(s13_second)
                sb1 = loader.load_tensor(e + "w1.scale_bias")
                sb3 = loader.load_tensor(e + "w3.scale_bias")
                sb13 = _shard_fused(sb1, sb3)
                w13_scale_bias[local_idx].copy_(sb13)
            if _has(e + "w2.weight_scale"):
                w2_scale[local_idx].copy_(loader.load_tensor(e + "w2.weight_scale"))
            if is_w4a8:
                if _has(e + "w2.weight_scale_second"):
                    s2_second = loader.load_tensor(e + "w2.weight_scale_second")
                    if tp > 1:
                        s2_second = loader.shard(s2_second, dim=1, world=tp, rank=tp_rank)
                    if w2_scale_second.numel() == 0:
                        w2_scale_second.data = torch.empty(nepr, *s2_second.shape, dtype=torch.float32, device=device)
                    w2_scale_second[local_idx].copy_(s2_second)
                s2_bias = loader.load_tensor(e + "w2.scale_bias")
                if tp > 1:
                    s2_bias = loader.shard(s2_bias, dim=1, world=tp, rank=tp_rank)
                w2_scale_bias[local_idx].copy_(s2_bias)
        # Shared experts: checkpoint has w1/w2/w3 (W8A8 dynamic), fuse w1+w3 -> gate_up_proj.
        se = ck + "ffn.shared_experts."
        shared_gate_keys = (
            ck + "ffn.shared_expert_gate.weight",
            ck + "ffn.shared_experts_gate.weight",
        )
        for shared_gate_key in shared_gate_keys:
            if _has(shared_gate_key):
                loader.copy_in(
                    pm + "mlp.shared_expert_gate.weight",
                    loader.load_tensor(shared_gate_key),
                )
                mlp.shared_expert_gate_is_loaded = True
                break
        if _has(se + "w1.weight"):
            se_w1 = loader.load_tensor(se + "w1.weight")
            se_w3 = loader.load_tensor(se + "w3.weight")
            se_w13 = _shard_fused(se_w1, se_w3)
            loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight", se_w13)
            if _has(se + "w1.weight_scale"):
                s1 = loader.load_tensor(se + "w1.weight_scale")
                s3 = loader.load_tensor(se + "w3.weight_scale")
                se_s13 = _shard_fused(s1, s3)
                loader.copy_in(pm + "mlp.shared_experts.gate_up_proj.weight_scale", se_s13[: se_w13.size(0)])
            if _has(se + "w1.weight_offset"):
                o1 = loader.load_tensor(se + "w1.weight_offset")
                o3 = loader.load_tensor(se + "w3.weight_offset")
                loader.copy_in(
                    pm + "mlp.shared_experts.gate_up_proj.weight_offset", _shard_fused(o1, o3)[: se_w13.size(0)]
                )
            se_w2 = loader.load_tensor(se + "w2.weight")
            if tp > 1:
                se_w2 = loader.shard(se_w2, dim=1, world=tp, rank=tp_rank)
            loader.copy_in(pm + "mlp.shared_experts.down_proj.weight", se_w2)
            if _has(se + "w2.weight_scale"):
                loader.copy_in(
                    pm + "mlp.shared_experts.down_proj.weight_scale", loader.load_tensor(se + "w2.weight_scale")
                )
            if _has(se + "w2.weight_offset"):
                loader.copy_in(
                    pm + "mlp.shared_experts.down_proj.weight_offset", loader.load_tensor(se + "w2.weight_offset")
                )
            # NOTE: shared_experts.{gate_up,down}_proj.process_weights_after_loading
            # is NOT called here. It is called exactly once via
            # DeepseekV4MoE.process_weights_after_loading (line ~816) at the end of
            # the per-layer load loop (load_weights line ~1187). Calling it here
            # too would transpose the weight twice (process_weights is not
            # idempotent), leaving it in [out, in] layout and tripping quant_matmul
            # "x1 dim[-1] must match x2 dim[-2], got 4096 vs 512".
