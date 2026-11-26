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


from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from xllm_kernel.ops.gemm import matmul_ein_sum

from scripts.logger import logger
from xllm.python import distributed as _distributed
from xllm.python import kernels
from xllm.python.device_stream import get_device_stream
from xllm.python.distributed.collectives import DecodeTpCollectives, EngramTpAllGather, decode_tp_all_reduce
from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.layernorm import RMSNorm
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.layers.moe_parallel import dp_gather_tokens
from xllm.python.layers.rotary_embedding import RotaryEmbedding
from xllm.python.model_executor.forward_context import (
    forward_context,
    get_forward_context,
    record_layer_event,
)
from xllm.python.model_loader import W8A8WeightLoader
from xllm.python.model_loader.module_loaders import load_w8a8_dynamic_projection
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    _DeepseekHyperConnection,
    _pick,
    _require_checkpoint_key,
    _resolve_dsv4_moe_prefix,
)
from xllm.python.models.deepseek_v32 import DeepseekV3MLP, W8A8DynamicLinear
from xllm.python.models.deepseek_v41_engram import (
    _E4M3_SUBNORMAL_STEP,
    ENGRAM_DEAD_ID,
    ENGRAM_STORAGE_FP8,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramHashState,
    EngramLayout,
    _e4m3_to_float,
    _validate_ue8m0_scale_bytes,
    image_sentinel_mask,
    load_quarot_gate_unrotate,
    ue8m0_to_scale,
)

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict

    from xllm.python.attention.csa2_attention import Csa2LayerContext


CSA2_MODE_SWA = "swa"
CSA2_MODE_FULL = "full"
CSA2_MODE_REINDEX = "reindex"
CSA2_MODE_REUSE = "reuse"


@dataclass(frozen=True)
class CSA2LayerMode:
    mode: str
    # 0: SWA only; 1: decoder main KV; 2: pooled encoder main KV.
    compress_ratio: int

    @property
    def is_csa2(self) -> bool:
        return self.mode != CSA2_MODE_SWA

    @property
    def is_full(self) -> bool:
        return self.mode == CSA2_MODE_FULL

    @property
    def is_reindex(self) -> bool:
        return self.mode == CSA2_MODE_REINDEX

    @property
    def is_reuse(self) -> bool:
        return self.mode == CSA2_MODE_REUSE


class CSA2LayerPlan:
    def __init__(self, modes: list[CSA2LayerMode]) -> None:
        self._modes = modes

    def mode(self, layer_id: int) -> CSA2LayerMode:
        if layer_id < 0 or layer_id >= len(self._modes):
            raise IndexError(f"layer {layer_id} outside the CSA2 plan's {len(self._modes)} layers")
        return self._modes[layer_id]

    @classmethod
    def build(
        cls,
        n_layers: int,
        compress_ratios: Sequence[int],
        kv_source_layer_ids: Sequence[int],
        index_source_layer_ids: Sequence[int],
    ) -> CSA2LayerPlan:
        kv_sources = sorted({int(source) for source in kv_source_layer_ids})
        index_sources = sorted({int(source) for source in index_source_layer_ids})
        kv_source_set = set(kv_sources)
        index_source_set = set(index_sources)
        for source in kv_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"kv source layer {source} is outside the model's {n_layers} layers")
        for source in index_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"index source layer {source} is outside the model's {n_layers} layers")
        non_indexed = [source for source in kv_sources if source not in index_source_set]
        if non_indexed:
            raise ValueError(f"kv source layers must also be index sources: {non_indexed}")
        for source in index_sources:
            if not any(kv_source <= source for kv_source in kv_sources):
                raise ValueError(f"index source layer {source} has no preceding kv source layer")
        ratios = [int(ratio) for ratio in compress_ratios]
        for source in kv_sources:
            if source >= len(ratios) or ratios[source] <= 0:
                raise ValueError(f"kv source layer {source} must have a positive compress ratio")

        modes: list[CSA2LayerMode] = []
        for layer_id in range(n_layers):
            ratio = ratios[layer_id] if layer_id < len(ratios) else 0
            kv_source = max((source for source in kv_sources if source <= layer_id), default=-1)
            if ratio <= 0 or kv_source < 0:
                mode = CSA2_MODE_SWA
            elif layer_id in kv_source_set:
                mode = CSA2_MODE_FULL
            elif layer_id in index_source_set:
                mode = CSA2_MODE_REINDEX
            else:
                mode = CSA2_MODE_REUSE
            modes.append(
                CSA2LayerMode(
                    mode=mode,
                    compress_ratio=max(ratio, 0),
                )
            )
        return cls(modes)


def _int_list(d: dict, key: str) -> list[int]:
    return [int(item) for item in _pick(d, key, default=[])]


def _parse_v41_compress_ratios(raw: Sequence[int], n_layers: int) -> list[int]:
    ratios = [int(ratio) for ratio in raw]
    for ratio in ratios:
        if ratio not in (0, 1, 2, 4):
            raise ValueError(f"unsupported DeepSeek-V4.1 compression ratio: {ratio}")
    if len(ratios) < n_layers:
        ratios.extend([0] * (n_layers - len(ratios)))
    return ratios[:n_layers]


@dataclass
class DeepseekV41Config(DeepseekV4Config):
    model_type: str = "deepseek_v41"
    hidden_size: int = 5120
    n_layers: int = 40
    q_lora_rank: int = 1280
    rms_norm_eps: float = 1e-20
    n_routed_experts: int = 384
    moe_intermediate_size: int = 2304
    dense_ffn_intermediate_size: int = 0
    index_n_heads: int = 32
    kv_source_layer_ids: list[int] = field(default_factory=list)
    index_source_layer_ids: list[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_num_embeddings: list[int] = field(default_factory=list)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_compressed_vocab_size: int = 99092
    engram_pad_token_id: int = -1
    image_token_id: int = -1
    mc2_comm_name: str = ""

    @property
    def dense_intermediate_size(self) -> int:
        if self.dense_ffn_intermediate_size > 0:
            return self.dense_ffn_intermediate_size
        if self.moe_intermediate_size > 0:
            return self.moe_intermediate_size
        return max(self.hidden_size * 4, 0)

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV41Config:
        cfg = super().from_dict(d)
        return replace(
            cfg,
            dense_ffn_intermediate_size=int(_pick(d, "intermediate_size", default=cfg.dense_ffn_intermediate_size)),
            # Re-parse: V4's from_dict normalized "ratio <= 1 -> 1".
            compress_ratios=_parse_v41_compress_ratios(list(d.get("compress_ratios") or []), cfg.n_layers),
            # V4.1 removed hash routing (gate.tid2eid); the field stays for
            # the inherited V4 machinery.
            n_hash_layers=0,
            kv_source_layer_ids=_int_list(d, "kv_source_layer_ids"),
            index_source_layer_ids=_int_list(d, "index_source_layer_ids"),
            candidate_source_layer_id=int(_pick(d, "candidate_source_layer_id", default=cfg.candidate_source_layer_id)),
            candidate_topk_blocks=int(_pick(d, "candidate_topk_blocks", default=cfg.candidate_topk_blocks)),
            candidate_block_size=int(_pick(d, "candidate_block_size", default=cfg.candidate_block_size)),
            engram_layer_ids=_int_list(d, "engram_layer_ids"),
            engram_num_embeddings=_int_list(d, "engram_num_embeddings"),
            engram_max_ngram_size=int(_pick(d, "engram_max_ngram_size", default=cfg.engram_max_ngram_size)),
            engram_vocab_size=int(_pick(d, "engram_vocab_size", default=cfg.engram_vocab_size)),
            engram_n_heads=int(_pick(d, "engram_n_heads", default=cfg.engram_n_heads)),
            engram_head_dim=int(_pick(d, "engram_head_dim", default=cfg.engram_head_dim)),
            engram_compressed_vocab_size=int(
                _pick(d, "engram_compressed_vocab_size", default=cfg.engram_compressed_vocab_size)
            ),
            engram_pad_token_id=int(_pick(d, "engram_pad_token_id", default=cfg.engram_pad_token_id)),
            image_token_id=int(_pick(d, "image_token_id", default=cfg.image_token_id)),
            mc2_comm_name=str(d.get("v41_mc2_comm_name", "")),
        )


V41_QUANT_FP8 = "fp8"
V41_QUANT_ASCEND = "ascend"
V41_QUANT_NONE = "none"


def _detect_v41_quant_mode(config: dict) -> str:
    model_path = config.get("model_path")
    if not model_path:
        logger.warning("DeepSeek-V4.1 quant mode detection: model_path missing; defaulting to bf16 (V41_QUANT_NONE)")
        return V41_QUANT_NONE
    if os.path.isfile(os.path.join(model_path, "quant_model_description.json")):
        return V41_QUANT_ASCEND
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return V41_QUANT_NONE
    with open(config_path, encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    quant_config = checkpoint_config.get("quantization_config") or {}
    method = quant_config.get("quant_method")
    if method == "fp8":
        return V41_QUANT_FP8
    if method == "ascend":
        return V41_QUANT_ASCEND
    return V41_QUANT_NONE


def _projection_quant_mode(
    quant_description: dict[str, Any],
    checkpoint_module: str,
    default_mode: str,
) -> str:
    weight_key = checkpoint_module + ".weight"
    quant_type = quant_description.get(weight_key)
    if quant_type is None:
        return default_mode
    if quant_type == "FLOAT":
        return V41_QUANT_NONE
    if quant_type == "W8A8_DYNAMIC":
        return V41_QUANT_ASCEND
    raise ValueError(f"Unsupported DeepSeek-V4.1 quantization {quant_type!r} for {weight_key}")


def _make_v41_linear(
    in_features: int,
    out_features: int,
    quant_mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    if quant_mode == V41_QUANT_ASCEND:
        return W8A8DynamicLinear(
            in_features,
            out_features,
            device,
            transpose_weight_after_loading=False,
        )
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[:, 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return pre, post, comb


class DeepseekV41HyperConnection(_DeepseekHyperConnection):
    def __init__(self, cfg: DeepseekV41Config, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._use_fused_pre = (
            device.type in ("npu", "privateuseone") and self.hc_mult == 4 and torch.ops.xllm_ops.supports_fused_hc_pre()
        )

    def hc_pre_delayed(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._use_fused_pre and x.dtype == torch.bfloat16 and x.numel():
            return torch.ops.xllm_ops.hc_pre_delayed(
                x, hc_fn, hc_scale, hc_base, pre_mix, self.hc_mult, self.sinkhorn_iters, self.norm_eps, self.hc_eps
            )

        pre, post, comb = self.hc_mixes(x, hc_fn, hc_scale, hc_base)
        return self.hc_pre(x, pre_mix), post, comb, pre

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flattened, hc_fn) * rsqrt
        return _hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=-2)
        return y.to(x.dtype)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        if x.numel() == 0:
            # The native kernel requires non-empty dimensions; empty DP
            # ranks still participate in the surrounding collectives.
            return residual
        if self.hc_mult != 4 or x.size(-1) % 64 != 0:
            # The shared NPU kernel tiles 64 hidden elements and four streams.
            expanded = post.float().unsqueeze(-1) * x.float().unsqueeze(-2)
            mixed = torch.einsum("tij,tid->tjd", comb.float(), residual.float())
            return (expanded + mixed).to(x.dtype)
        # Each row of comb maps one input residual stream to all output streams.
        return kernels.hc_post(
            x.unsqueeze(0),
            residual.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        ).squeeze(0)

    @staticmethod
    def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        pre_mix = torch.zeros(x.size(0), hc_mult, dtype=torch.float32, device=x.device)
        pre_mix[:, 0] = 1.0
        return pre_mix


class DeepseekV41Indexer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str,
        owns_k: bool,
    ) -> None:
        super().__init__()
        self.owns_k = owns_k
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.wq_b = _make_v41_linear(
            cfg.q_lora_rank,
            self.n_heads * self.head_dim,
            quant_mode,
            dtype,
            device,
        )
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False, dtype=dtype, device=device)
        if owns_k:
            self.wk = nn.Linear(cfg.head_dim, self.head_dim, bias=False, dtype=dtype, device=device)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype, device)

    def process_weights_after_loading(self) -> None:
        if isinstance(self.wq_b, W8A8DynamicLinear):
            self.wq_b.process_weights_after_loading()


class DeepseekV41Attention(Attention):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
        quant_description: dict[str, Any] | None = None,
    ) -> None:
        tp = cfg.tp_size
        num_heads = cfg.n_heads // tp
        head_dim = cfg.head_dim
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            sliding_window=cfg.window_size,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.mode = mode
        self.quant_mode = quant_mode
        self.num_heads_local = num_heads
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.zeros(num_heads, dtype=torch.float32, device=device)),
        )
        self.attn_sink_loaded = False
        layer_prefix = "mtp" if cfg.model_type == "deepseek_v41_dspark" else "layers"
        checkpoint_prefix = f"{layer_prefix}.{layer_id}.attn."
        quant_description = quant_description or {}
        q_a_quant = _projection_quant_mode(quant_description, checkpoint_prefix + "wq_a", quant_mode)
        kv_quant = _projection_quant_mode(quant_description, checkpoint_prefix + "wkv", quant_mode)
        pack_q_a_kv = (
            quant_mode == V41_QUANT_ASCEND
            and q_a_quant == kv_quant == V41_QUANT_NONE
            and dtype == torch.bfloat16
            and device.type == "npu"
            and torch.npu.get_device_name(device).startswith("Ascend910_93")
        )
        projection_device = torch.device("meta") if pack_q_a_kv else device
        self.q_a_proj = _make_v41_linear(
            cfg.hidden_size,
            cfg.q_lora_rank,
            q_a_quant,
            dtype,
            projection_device,
        )
        self.q_a_layernorm = RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype, device)
        self.q_b_proj = _make_v41_linear(
            cfg.q_lora_rank,
            num_heads * head_dim,
            _projection_quant_mode(quant_description, checkpoint_prefix + "wq_b", quant_mode),
            dtype,
            device,
        )
        self.kv_proj = _make_v41_linear(
            cfg.hidden_size,
            head_dim,
            kv_quant,
            dtype,
            projection_device,
        )
        packed = None
        if pack_q_a_kv:
            # Allocate the shared bank before the checkpoint load to avoid an
            # extra pair of BF16 weights while MoE weights are being converted.
            packed = torch.empty((cfg.q_lora_rank + head_dim, cfg.hidden_size), dtype=dtype, device=device)
            q_weight, kv_weight = packed.split((cfg.q_lora_rank, head_dim), dim=0)
            self.q_a_proj.weight, self.kv_proj.weight = nn.Parameter(q_weight), nn.Parameter(kv_weight)
            self.q_a_proj.reset_parameters()
            self.kv_proj.reset_parameters()
        self.kv_a_layernorm = RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        self.register_buffer("_packed_q_a_kv_weight", packed, persistent=False)
        self._kv_stream = get_device_stream(device, "deepseek_v41_kv") if pack_q_a_kv else None
        if cfg.o_groups % tp != 0:
            raise ValueError(
                f"DeepSeek-V4.1 o_groups ({cfg.o_groups}) must be divisible by tensor-parallel "
                f"size {tp}; the grouped o_a/o_b output projection cannot be sharded evenly"
            )
        self.n_local_groups = cfg.o_groups // tp
        self.o_lora_rank = cfg.o_lora_rank
        o_a_in = (cfg.n_heads * head_dim) // cfg.o_groups
        if quant_mode == V41_QUANT_ASCEND:
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
                reduce_results=False,
                use_checkpoint_layout=True,
            )
        else:
            self.o_a_proj = nn.Linear(
                o_a_in,
                self.n_local_groups * cfg.o_lora_rank,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = nn.Linear(
                self.n_local_groups * cfg.o_lora_rank,
                cfg.hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
        # Ratio-2 pooling stays FP32, including its projection parameters.
        if mode.is_full:
            ratio = mode.compress_ratio
            cmp_dtype = torch.float32 if ratio > 1 else dtype
            self.cmp_wkv = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=cmp_dtype, device=device)
            if ratio > 1:
                self.cmp_wgate = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=torch.float32, device=device)
            self.cmp_norm = RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        self.indexer: DeepseekV41Indexer | None = (
            DeepseekV41Indexer(
                cfg,
                dtype,
                device,
                _projection_quant_mode(quant_description, checkpoint_prefix + "indexer.wq_b", quant_mode),
                owns_k=mode.is_full,
            )
            if mode.is_full or mode.is_reindex
            else None
        )
        self._csa2_ctx: Csa2LayerContext | None = None

    def _pack_o_a_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.view(self.n_local_groups, self.o_lora_rank, -1).transpose(1, 2)

    def process_weights_after_loading(self) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            for module in (self.q_a_proj, self.q_b_proj, self.kv_proj):
                if isinstance(module, W8A8DynamicLinear):
                    module.process_weights_after_loading()
            if self.indexer is not None:
                self.indexer.process_weights_after_loading()
            wo_a = self.o_a_proj.weight
            if (
                wo_a.ndim == 2
                and wo_a.dtype == torch.bfloat16
                and wo_a.device.type == "npu"
                and torch.npu.get_device_name(wo_a.device).startswith("Ascend910_93")
            ):
                # Axis order [groups, input, rank] matches the ND kernel.
                wo_a.data = self._pack_o_a_weight(wo_a.data).contiguous()

    def _project_grouped_output(self, value: torch.Tensor) -> torch.Tensor:
        wo_a = self.o_a_proj.weight
        out = value.view(value.size(0), self.n_local_groups, wo_a.size(1))
        if wo_a.ndim == 3:
            if 0 < out.size(0) < 65536:
                return matmul_ein_sum(out, wo_a)
            wo_a = wo_a.transpose(1, 2)
        return torch.einsum("tgd,grd->tgr", out, wo_a.view(self.n_local_groups, self.o_lora_rank, -1))

    def _project_packed_q_a_kv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from xllm.python.kernels_npu.triton.packed_projection import split_packed_projection

        projected = F.linear(hidden, self._packed_q_a_kv_weight)
        return split_packed_projection(projected, self.cfg.q_lora_rank)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        from xllm.python.attention.csa2_attention import (
            Csa2LayerContext,
            apply_interleaved_rope,
        )

        forward_context = get_forward_context()
        backend = forward_context.attention_backend
        execute_csa2 = getattr(backend, "execute_csa2_layer", None)
        if not callable(execute_csa2):
            raise RuntimeError("DeepSeek-V4.1 requires the CSA2 attention backend")
        num_tokens = hidden.size(0)
        use_graph = forward_context.acl_graph is not None
        if self._packed_q_a_kv_weight is not None and (
            use_graph or getattr(forward_context.metadata, "dsa_graph_mode", False)
        ):
            q_a, kv = self._project_packed_q_a_kv(hidden)
        else:
            q_a, kv = self.q_a_proj(hidden), None
        prepare_swa_kv = getattr(backend, "prepare_swa_kv", None)
        prepare_rope = getattr(backend, "prepare_projection_rope", None)
        kv_stream = None
        if (
            use_graph
            and kv is not None
            and self._kv_stream is not None
            and prepare_swa_kv is not None
            and prepare_rope is not None
            and prepare_rope(kv, positions, cos_sin_cache)
        ):
            kv_stream = self._kv_stream
            kv_stream.wait_for_current()
            with kv_stream.activate():
                prepared_kv, swa_cache_written = prepare_swa_kv(self.kv_a_layernorm(kv), positions, cos_sin_cache, self)
        use_a3 = getattr(backend, "uses_a3_cache_precision", False)
        quantized_q = isinstance(self.q_b_proj, W8A8DynamicLinear)
        quantized_indexer = self.indexer is not None and isinstance(self.indexer.wq_b, W8A8DynamicLinear)
        qr_quantized = None
        qr = None
        if use_graph and use_a3 and quantized_q and (self.indexer is None or quantized_indexer):
            qr_quantized = kernels.rms_norm_dynamic_quant(q_a, self.q_a_layernorm.weight, self.q_a_layernorm.eps)
            q = self.q_b_proj.forward_quantized(*qr_quantized)
        elif quantized_q and quantized_indexer:
            qr = self.q_a_layernorm(q_a)
            qr_quantized = kernels.dynamic_quant(qr)
            q = self.q_b_proj.forward_quantized(*qr_quantized)
        else:
            qr = self.q_a_layernorm(q_a)
            q = self.q_b_proj(qr)
        q = q.view(num_tokens, self.num_heads_local, self.head_dim)
        if kv_stream is None:
            kv = self.kv_a_layernorm(self.kv_proj(hidden) if kv is None else kv)
        apply_rope = getattr(backend, "apply_projection_rope", apply_interleaved_rope)
        q = apply_rope(q, positions, cos_sin_cache)
        if kv_stream is not None:
            kv_stream.join()
            kv_stream.record_on_current(prepared_kv)
            kv = prepared_kv
        elif prepare_swa_kv is None:
            kv, swa_cache_written = apply_rope(kv, positions, cos_sin_cache), False
        else:
            kv, swa_cache_written = prepare_swa_kv(kv, positions, cos_sin_cache, self)
        if not use_a3:
            kv = fp8_act_qdq(kv, 32)
        self._csa2_ctx = Csa2LayerContext(
            x=hidden,
            qr=qr,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            qr_quantized=qr_quantized,
            swa_cache_written=swa_cache_written,
        )
        attn_out = execute_csa2(q, kv, self)
        attn_out = apply_rope(attn_out, positions, cos_sin_cache, inverse=True)
        o_low = self._project_grouped_output(attn_out)
        o = self.o_b_proj(o_low.reshape(num_tokens, -1))
        if self.cfg.tp_size > 1:
            decode_tp_all_reduce(o)
        return o

    def _project_swa_kv(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        uses_a3_cache_precision: bool,
    ) -> torch.Tensor:
        from xllm.python.attention.csa2_attention import apply_interleaved_rope

        kv = self.kv_a_layernorm(self.kv_proj(hidden))
        kv = apply_interleaved_rope(kv, positions, cos_sin_cache)
        return kv if uses_a3_cache_precision else fp8_act_qdq(kv, 32)

    def prepare_decoder_replay_kv(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        from xllm.python.attention.csa2_attention import Csa2LayerContext

        backend = get_forward_context().attention_backend
        ctx = Csa2LayerContext(x=hidden, qr=None, positions=positions, cos_sin_cache=cos_sin_cache)
        backend.prepare_decoder_replay_kv(self, ctx)

    def compress_kv_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode.compress_ratio > 1:
            xf = x.float()
            return self.cmp_wkv(xf), self.cmp_wgate(xf)
        return self.cmp_wkv(x), None

    def compress_rmsnorm(self, v: torch.Tensor) -> torch.Tensor:
        return self.cmp_norm(v)

    def indexer_key(self, latent: torch.Tensor, cos_sin: torch.Tensor | None = None) -> torch.Tensor:
        k = self.indexer.wk(latent)
        norm = self.indexer.k_norm
        if cos_sin is not None:
            return torch.ops.xllm_ops.v41_index_key_transform(k, norm.weight, cos_sin, norm.eps)
        return norm(k)

    def indexer_query(
        self, qr: torch.Tensor | None, qr_quantized: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        wq_b = self.indexer.wq_b
        if qr_quantized is None:
            assert qr is not None
            projected = wq_b(qr)
        else:
            projected = wq_b.forward_quantized(*qr_quantized)
        return projected.view(-1, self.indexer.n_heads, self.indexer.head_dim)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.indexer.head_dim**-0.5 * self.indexer.n_heads**-0.5
        return self.indexer.weights_proj(x).float() * scale


class DeepseekV41TorchMLP(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.swiglu_limit = cfg.swiglu_limit
        inter = cfg.dense_intermediate_size
        self.w1 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(inter, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.w1(x), self.w3(x), self.w2, self.swiglu_limit)


def _swiglu_clamped(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    w2: nn.Module,
    limit: float,
) -> torch.Tensor:
    gate = gate_proj.float()
    up = up_proj.float()
    if 0.0 < limit < 1_000_000.0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    hidden = F.silu(gate) * up
    return w2(hidden.to(x.dtype))


class DeepseekV41MoE(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.topk = cfg.n_activated_experts
        self.n_routed_experts = cfg.n_routed_experts
        self.routed_scaling = cfg.routed_scaling_factor
        self.norm_topk_prob = cfg.norm_topk_prob
        self.swiglu_limit = cfg.swiglu_limit
        self.image_token_id = cfg.image_token_id
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=dtype, device=device)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False
        inter = cfg.moe_intermediate_size
        hidden = cfg.hidden_size
        ep_size = cfg.ep_size if cfg.ep_size > 0 else 1
        ep_rank = cfg.ep_rank if cfg.ep_size > 0 else 0
        if cfg.n_routed_experts % ep_size != 0:
            raise ValueError(
                f"deepseek v4.1 torch MoE requires n_routed_experts divisible by ep_size, "
                f"got {cfg.n_routed_experts} experts and ep_size={ep_size}"
            )
        self.ep_size = ep_size
        self.num_experts_per_rank = cfg.n_routed_experts // ep_size
        self.start_expert_id = ep_rank * self.num_experts_per_rank
        self.experts_w13 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, 2 * inter, hidden, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, hidden, inter, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.shared_w1 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)
        self.shared_w2 = nn.Linear(inter, hidden, bias=False, dtype=dtype, device=device)
        self.shared_w3 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)

    def _selection_logits(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.gate.weight.float())
        scores = F.softplus(scores).sqrt()
        bias = self.e_score_correction_bias
        if input_ids is not None and self.gate_bias_vl_loaded:
            image_mask = input_ids.reshape(-1) == self.image_token_id
            bias = torch.where(image_mask.unsqueeze(-1), self.gate_bias_vl, self.e_score_correction_bias)
        return scores + bias, scores

    def forward(self, hidden: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        shape = hidden.shape
        x = hidden.reshape(-1, self.cfg.hidden_size)
        logits, scores = self._selection_logits(x, input_ids)
        x, scatter_state = dp_gather_tokens(x, self.cfg.dp_size, self.cfg.dp_rank)
        if self.cfg.dp_size > 1:
            gate_values, _ = dp_gather_tokens(torch.cat((logits, scores), dim=-1), self.cfg.dp_size, self.cfg.dp_rank)
            logits, scores = gate_values.chunk(2, dim=-1)
        # Selection bias changes the chosen experts, not their routing weights.
        indices = logits.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling
        y = kernels.grouped_moe_bf16(
            x,
            weights,
            indices.to(torch.int32),
            self.experts_w13.transpose(1, 2),
            self.experts_w2.transpose(1, 2),
            self.n_routed_experts,
            self.start_expert_id,
            self.num_experts_per_rank,
            swiglu_limit=self.swiglu_limit,
            router_weight_before_down=True,
        )
        if self.ep_size > 1:
            _distributed.moe_ep_all_reduce(y)
        y += _swiglu_clamped(x, self.shared_w1(x), self.shared_w3(x), self.shared_w2, self.swiglu_limit).float()
        return scatter_state.scatter(y.type_as(x)).view(shape)


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
        engram_layout: EngramLayout | None = None,
        engram_storage: str = ENGRAM_STORAGE_FP8,
        engram_gate_unrotate: torch.Tensor | None = None,
        quant_description: dict[str, Any] | None = None,
        engram_host_offload: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hc = DeepseekV41HyperConnection(cfg, device)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.self_attn = DeepseekV41Attention(cfg, layer_id, dtype, device, mode, quant_mode, quant_description)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.engram: Engram | None = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(
                engram_layout,
                engram_layout.layer_ids.index(layer_id),
                cfg.hidden_size,
                cfg.hc_mult,
                cfg.rms_norm_eps,
                cfg.tp_size,
                cfg.tp_rank,
                dtype,
                device,
                storage=engram_storage,
                gate_unrotate=engram_gate_unrotate,
                dp_size=cfg.dp_size if engram_storage == ENGRAM_STORAGE_INT8 else 1,
                dp_rank=cfg.dp_rank if engram_storage == ENGRAM_STORAGE_INT8 else 0,
                host_offload=engram_host_offload,
            )
        is_dense = (layer_id < cfg.first_k_dense_replace) or (
            cfg.moe_layer_freq > 1 and layer_id % cfg.moe_layer_freq != 0
        )
        if is_dense:
            if quant_mode == V41_QUANT_ASCEND:
                self.mlp: nn.Module = DeepseekV3MLP(
                    cfg,
                    cfg.dense_intermediate_size,
                    dtype,
                    device,
                    swiglu_limit=cfg.swiglu_limit,
                )
            else:
                self.mlp = DeepseekV41TorchMLP(cfg, dtype, device)
        elif quant_mode == V41_QUANT_ASCEND:
            self.mlp = DeepseekV4MoE(
                cfg,
                layer_id,
                dtype,
                device,
                fuse_shared_expert_activation=True,
                fused_mc2_group=cfg.mc2_comm_name,
                pretranspose_w8a8_weights=True,
            )
        else:
            self.mlp = DeepseekV41MoE(cfg, dtype, device)

    def prepare_decoder_replay_kv(
        self,
        hidden: torch.Tensor,
        pre_mix: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> None:
        x = self.input_layernorm(self.hc.hc_pre(hidden, pre_mix))
        self.self_attn.prepare_decoder_replay_kv(x, positions, cos_sin_cache)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        pre_mix: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        engram_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pre_mix is None:
            pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.hc.hc_mult)
        # Engram injection happens between the previous sublayer's post and
        # this block's pre, on the full hc stream, so the mix coefficients
        # see the injected stream.
        if self.engram is not None and engram_hashes is not None:
            hidden = self.engram(
                hidden,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
                prefetched=engram_rows,
            )
        residual = hidden
        x, attn_post, attn_comb, attn_pre = self.hc.hc_pre_delayed(
            hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base, pre_mix
        )
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, cos_sin_cache)
        hidden = self.hc.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden
        x, ffn_post, ffn_comb, ffn_pre = self.hc.hc_pre_delayed(
            hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base, attn_pre
        )
        if isinstance(self.mlp, DeepseekV4MoE):
            x = self.mlp(x, input_ids, input_norm=self.post_attention_layernorm)
        else:
            x = self.post_attention_layernorm(x)
            x = self.mlp(x, input_ids) if isinstance(self.mlp, DeepseekV41MoE) else self.mlp(x)
        hidden = self.hc.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden, ffn_pre


@dataclass
class _DecoderReplayMetadata:
    source: Any
    dsa_metadata: Any
    dp_execution_token_counts: Sequence[int]
    raw_dp_execution_token_counts: Sequence[int]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


class DeepseekV41Model(nn.Module):
    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str = V41_QUANT_NONE,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        if cfg.cp_size > 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 context parallelism is scheduled with the Phase-6 distributed work"
            )
        self.cfg = cfg
        self.csa2_plan = CSA2LayerPlan.build(
            cfg.n_layers,
            cfg.compress_ratios,
            cfg.kv_source_layer_ids,
            cfg.index_source_layer_ids,
        )
        self._decode_tp_collectives: DecodeTpCollectives | None = None
        self.decoder_replay_cut = cfg.n_layers
        self._decoder_swa_replay_logged = False
        self._decoder_swa_skip_logged = False
        if config and config.get("enable_swa_bounded_replay") and cfg.kv_source_layer_ids:
            last_source = max(cfg.kv_source_layer_ids)
            last_engram = max(cfg.engram_layer_ids, default=-1)
            if last_engram >= last_source:
                raise ValueError("DeepSeek-V4.1 SWA bounded replay needs Engram layers before the final KV source")
            self.decoder_replay_cut = last_source
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: EngramHashState | None = None
        if cfg.engram_layer_ids:
            if cfg.engram_pad_token_id < 0:
                raise ValueError(
                    "DeepSeek-V4.1 engram is enabled (engram_layer_ids="
                    f"{cfg.engram_layer_ids}) but engram_pad_token_id is unset"
                )
            self.engram_layout = EngramLayout(
                cfg.engram_layer_ids,
                cfg.engram_num_embeddings,
                cfg.engram_max_ngram_size,
                cfg.engram_n_heads,
                cfg.engram_head_dim,
                cfg.engram_vocab_size,
                cfg.engram_compressed_vocab_size,
                cfg.engram_pad_token_id,
            )
            self.engram_hash = EngramHashState.from_config(
                self.engram_layout,
                {**(config or {}), "vocab_size": cfg.vocab_size},
            ).to(device)
        # Ascend W8A8 exports use symmetric INT8 tables and rotate the
        # residual stream (QuaRot). The gate dot uses the original basis.
        self.engram_storage = ENGRAM_STORAGE_INT8 if quant_mode == V41_QUANT_ASCEND else ENGRAM_STORAGE_FP8
        self.engram_host_offload = (
            bool(cfg.engram_layer_ids)
            and self.engram_storage == ENGRAM_STORAGE_INT8
            and device.type in ("npu", "privateuseone")
            and (config or {}).get("enable_engram_host_offload", True)
        )
        self._engram_prefetch_stream = get_device_stream(device, "v41_engram") if self.engram_host_offload else None
        self.engram_gate_unrotate = None
        if cfg.engram_layer_ids and quant_mode == V41_QUANT_ASCEND:
            self.engram_gate_unrotate = load_quarot_gate_unrotate(str((config or {}).get("model_path") or ""), device)
        quant_description = {}
        model_path = (config or {}).get("model_path")
        if model_path and quant_mode == V41_QUANT_ASCEND:
            quant_path = os.path.join(model_path, "quant_model_description.json")
            if os.path.isfile(quant_path):
                with open(quant_path, encoding="utf-8") as handle:
                    quant_description = json.load(handle)
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device, use_decode_tp=True
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    cfg,
                    i,
                    dtype,
                    device,
                    self.csa2_plan.mode(i),
                    quant_mode,
                    self.engram_layout,
                    engram_storage=self.engram_storage,
                    engram_gate_unrotate=self.engram_gate_unrotate,
                    quant_description=quant_description,
                    engram_host_offload=self.engram_host_offload,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self._engram_tp_gather: EngramTpAllGather | None = None
        if self._engram_prefetch_stream is not None and tp > 1:
            table = self.layers[cfg.engram_layer_ids[0]].engram.embed_tokens
            # Preserve the normal TP group's stream affinity while warming this
            # independent CCLBuffer on the Engram producer before graph capture.
            with self._engram_prefetch_stream.activate():
                self._engram_tp_gather = EngramTpAllGather(table.weight.device, table.part_n_hash_cols * table.dim)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.aux_hidden_capture = AuxHiddenCapture(cfg.layers_to_capture, transform=lambda hidden: hidden.mean(dim=1))
        self.rotary = RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            dtype=torch.float32,
            device="cpu",
        ).to(device)
        self.compress_rotary = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.compress_rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.original_max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )

    def acl_graph_capture_context(self, input_ids: torch.Tensor) -> AbstractContextManager:
        backend = get_forward_context().attention_backend
        if (
            self.cfg.model_type not in ("deepseek_v41", "deepseek_v41_dspark")
            or self.cfg.tp_size != 4
            or self.layers[0].self_attn.quant_mode != V41_QUANT_ASCEND
            or self.norm.weight.dtype != torch.bfloat16
            or not getattr(backend, "uses_a3_cache_precision", False)
            or os.environ.get("HCCL_OP_EXPANSION_MODE") != "AIV"
        ):
            return nullcontext()
        if self._decode_tp_collectives is None:
            self._decode_tp_collectives = DecodeTpCollectives(
                input_ids.numel(),
                self.cfg.hidden_size,
                input_ids.device,
                buffer_name="xllm_v41_draft_decode_tp"
                if self.cfg.model_type == "deepseek_v41_dspark"
                else "xllm_v41_decode_tp",
            )
        return self._decode_tp_collectives.capture_context()

    @staticmethod
    def _compute_engram_hashes(
        hash_state: EngramHashState,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        replay_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        dsa = getattr(metadata, "dsa_metadata", None)
        swa_group = backend.swa_group_metadata(dsa)
        if swa_group is None or not hash_state.ensure_cache(swa_group.cache):
            return None, None
        q_cu = dsa.actual_seq_lengths_query
        dead = image_sentinel_mask(input_ids)
        hashes = hash_state(
            input_ids,
            positions,
            q_cu,
            dead,
            swa_group.slot_mapping,
            swa_group.block_table,
            swa_group.block_size,
            unit_query_rows=q_cu[:-1] if dsa.query_aligned_token_slots and dsa.max_query_len == 1 else None,
            replay_history=replay_history,
            replay_requests=dsa.compressed_write_floors,
        )
        return hashes, ~dead

    def _decoder_replay_lengths(self, metadata: Any) -> tuple[list[int], list[int], list[int]]:
        dsa = metadata.dsa_metadata
        starts = getattr(metadata, "decoder_replay_starts", ())
        empty_rank = self.cfg.dp_size > 1 and not metadata.raw_dp_execution_token_counts[self.cfg.dp_rank]
        if not starts:
            # Synthetic profiler batches have no request prompt boundaries.
            starts = (
                [max(0, end - self.cfg.window_size) for end in dsa.seq_lens_kv_host]
                if metadata.is_prefill or metadata.is_chunked_prefill
                else [-1] * len(dsa.seq_lens_q_host)
            )
        floors = [max(start, 0) for start in starts]
        lengths = [
            min(query, max(0, end - floor)) if not empty_rank else 0
            for query, end, floor in zip(dsa.seq_lens_q_host, dsa.seq_lens_kv_host, floors, strict=True)
        ]
        if self.cfg.dp_size == 1:
            return lengths, floors, [sum(lengths)]
        return lengths, floors, metadata.decoder_replay_dp_token_counts

    def _decoder_replay_enabled(self, metadata: Any, graph_mode: bool) -> bool:
        if self.decoder_replay_cut >= self.cfg.n_layers or graph_mode or metadata.is_target_verify:
            return False
        if self.cfg.dp_size == 1:
            return metadata.is_prefill or metadata.is_chunked_prefill or metadata.is_mixed
        # The input factory sends the same replay plan to every DP rank.
        return bool(getattr(metadata, "decoder_replay_dp_token_counts", ()))

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        aux_hidden = self.aux_hidden_capture.create_buffer(hidden)
        positions = positions.to(torch.int64).contiguous()
        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        graph_mode = bool(getattr(metadata, "dsa_graph_mode", False))
        preserve_dsa_metadata = (
            graph_mode
            and getattr(metadata, "prepared_attention_state", None) is None
            and getattr(metadata, "dsa_metadata", None) is not None
        )
        # Rebuild layer-shared tensors inside every captured forward. Reusing
        # warmup's SWA tiling tensor leaves the graph borrowing storage that a
        # later batch bucket can release, and omits its live-length update.
        backend.reset_forward(metadata, preserve_dsa_metadata=preserve_dsa_metadata)
        if not preserve_dsa_metadata:
            metadata.dsa_graph_mode = graph_mode
            metadata.dsa_positions = positions
            prepare_dsa = getattr(backend, "prepare_dsa_metadata_for_forward", None)
            if prepare_dsa is None:
                raise RuntimeError("DeepSeek-V4.1 requires prepare_dsa_metadata_for_forward")
            prepare_dsa(metadata)

        replay_history = None
        if not graph_mode:
            replay_starts, replay_ends, history_ids = getattr(metadata, "swa_prefix_replay", ((), (), ()))
            if replay_starts:
                dsa = metadata.dsa_metadata
                dsa.swa_history_floors = tuple(max(0, start) for start in replay_starts)
                dsa.compressed_write_floors = tuple(replay_ends)
                if history_ids:
                    replay_history = input_ids.new_tensor(history_ids).view(len(replay_starts), -1)

        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        prefetched_engram: dict[int, tuple[torch.Tensor, torch.npu.Event]] = {}
        prefetch_stream = self._engram_prefetch_stream
        if self.engram_hash is not None:
            engram_hashes, engram_mask = self._compute_engram_hashes(
                self.engram_hash, backend, metadata, input_ids, positions, replay_history
            )
            lookup_hashes = engram_hashes
            dp_scatter = None
            if self.engram_storage == ENGRAM_STORAGE_INT8 and self.cfg.dp_size > 1 and not self.engram_host_offload:
                if lookup_hashes is None:
                    # Join the DP lookup even when this rank has no Engram
                    # history. Dead IDs select no rows and leave gating off.
                    layout = self.engram_hash.layout
                    lookup_hashes = input_ids.new_full(
                        (input_ids.numel(), len(layout.layer_ids), layout.n_hash_cols),
                        ENGRAM_DEAD_ID,
                        dtype=torch.int32,
                    )
                lookup_hashes, dp_scatter = dp_gather_tokens(lookup_hashes, self.cfg.dp_size, self.cfg.dp_rank)
            if lookup_hashes is not None and prefetch_stream is not None:
                prefetch_stream.wait_for_current()
                with prefetch_stream.activate():
                    stream = torch.npu.current_stream(input_ids.device)
                    lookup_hashes.record_stream(stream)
                    for layer_id in self.cfg.engram_layer_ids:
                        engram = self.layers[layer_id].engram
                        rows = engram.embed_tokens.lookup(
                            lookup_hashes[:, engram.layer_hash_index], tp_gather=self._engram_tp_gather
                        )
                        ready = torch.npu.Event()
                        ready.record(stream)
                        prefetched_engram[layer_id] = (rows, ready)
            elif lookup_hashes is not None:
                for layer in self.layers:
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(lookup_hashes[:, engram.layer_hash_index], dp_scatter)

        hidden = hidden.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.cfg.hc_mult)
        replay = self._decoder_replay_enabled(metadata, graph_mode or context.acl_graph is not None)
        full_rows = hidden.size(0)
        tail_rows = None
        skip_decoder = False
        needs_compaction = False
        dummy_decoder = False
        replay_plan = None
        tail_metadata = None
        if replay:
            dsa = metadata.dsa_metadata
            lengths, floors, dp_counts = self._decoder_replay_lengths(metadata)
            skip_decoder = not any(dp_counts)
            needs_compaction = sum(lengths) != full_rows
            if not skip_decoder:
                dummy_decoder = not any(lengths)
                if dummy_decoder:
                    # MoE collectives still require participation from
                    # shards without real decoder queries.
                    lengths[0] = 1
                    floors[0] = dsa.seq_lens_kv_host[0] - 1
                rows = None
                if needs_compaction:
                    tail_indices = []
                    end = 0
                    for original_len, length in zip(dsa.seq_lens_q_host, lengths, strict=True):
                        end += original_len
                        tail_indices.extend(range(end - length, end))
                    rows = input_ids.new_tensor(tail_indices, dtype=torch.int64)
                    if rows.numel() != full_rows:
                        tail_rows = rows
                replay_plan = backend.prepare_decoder_swa_replay(dsa, lengths, rows, floors, dummy=dummy_decoder)
                tail_metadata = _DecoderReplayMetadata(
                    metadata, replay_plan.metadata, [max(count, 1) for count in dp_counts], dp_counts
                )

        with ExitStack() as stack:
            for layer_id, layer in enumerate(self.layers):
                layer_cos_sin_cache = (
                    self.compress_rotary.cos_sin_cache if layer.mode.is_csa2 else self.rotary.cos_sin_cache
                )
                if replay and layer_id == self.decoder_replay_cut:
                    if skip_decoder or needs_compaction:
                        layer.prepare_decoder_replay_kv(hidden, pre_mix, positions, layer_cos_sin_cache)
                    if skip_decoder:
                        if self.cfg.tp_rank == 0 and not self._decoder_swa_skip_logged:
                            logger.info(
                                "DeepSeek-V4.1 decoder SWA replay: dp_rank=%d, first_tail_layer=%d, "
                                "full_rows=%d, tail_rows=0, cut_layer_kv_rows=%d",
                                self.cfg.dp_rank,
                                layer_id,
                                full_rows,
                                full_rows,
                            )
                            self._decoder_swa_skip_logged = True
                        output = hidden.new_zeros((full_rows, self.cfg.hidden_size))
                        return self.aux_hidden_capture.finalize(
                            output, None if aux_hidden is None else aux_hidden.zero_()
                        )
                    if tail_rows is not None:
                        hidden = hidden.index_select(0, tail_rows)
                        pre_mix = pre_mix.index_select(0, tail_rows)
                        positions = positions.index_select(0, tail_rows)
                        input_ids = input_ids.index_select(0, tail_rows)
                        if aux_hidden is not None:
                            aux_hidden = aux_hidden.index_select(0, tail_rows)
                    stack.enter_context(backend.decoder_swa_replay(replay_plan))
                    stack.enter_context(
                        forward_context(replace(context, metadata=tail_metadata, layer_shared_cache={}))
                    )
                    if self.cfg.tp_rank == 0 and not dummy_decoder and not self._decoder_swa_replay_logged:
                        logger.info(
                            "DeepSeek-V4.1 decoder SWA replay: dp_rank=%d, first_tail_layer=%d, "
                            "full_rows=%d, tail_rows=%d, cut_layer_kv_rows=%d",
                            self.cfg.dp_rank,
                            layer_id,
                            full_rows,
                            hidden.size(0),
                            full_rows,
                        )
                        self._decoder_swa_replay_logged = True
                # V4.1 checkpoint IDs select the residual entering the layer.
                self.aux_hidden_capture.capture_layer(layer_id, hidden, None, aux_hidden)
                layer_args = (hidden, positions, layer_cos_sin_cache, input_ids, pre_mix, engram_hashes, engram_mask)
                if layer_id in prefetched_engram:
                    engram_rows, ready = prefetched_engram[layer_id]
                    torch.npu.current_stream(input_ids.device).wait_event(ready)
                    prefetch_stream.record_on_current(engram_rows)
                    hidden, pre_mix = layer(*layer_args, engram_rows=engram_rows)
                else:
                    hidden, pre_mix = layer(*layer_args)
                record_layer_event(layer_id)
            merged = self.layers[-1].hc.hc_pre(hidden, pre_mix)
            output = merged if self.cfg.model_type == "deepseek_v41_dspark" else self.norm(merged)
        if dummy_decoder:
            output = output.new_zeros((full_rows, output.size(-1)))
            if aux_hidden is not None:
                aux_hidden = aux_hidden.new_zeros((full_rows, aux_hidden.size(-1)))
        elif tail_rows is not None:
            # The sampler and DSpark context publication still use full-batch
            # row addresses. Retained rows own the final output and draft KV.
            output = output.new_zeros((full_rows, output.size(-1))).index_copy_(0, tail_rows, output)
            if aux_hidden is not None:
                aux_hidden = aux_hidden.new_zeros((full_rows, aux_hidden.size(-1))).index_copy_(
                    0, tail_rows, aux_hidden
                )
        return self.aux_hidden_capture.finalize(output, aux_hidden)


class DeepseekV41ForCausalLM(PyModelBase):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV41Config.from_dict(config)
        self.quant_mode = _detect_v41_quant_mode(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = DeepseekV41Model(self.cfg, dtype, device, self.quant_mode, config)
        tp = self.cfg.tp_size
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            self._load_weights_ascend(state_dicts)
        else:
            self._load_weights_torch(state_dicts)

    def _load_weights_torch(self, state_dicts: list[StateDict]) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)
        is_fp8 = self.quant_mode == V41_QUANT_FP8

        def _linear(ckpt_prefix: str, param_prefix: str, shard_dim: int | None = None) -> None:
            weight = loader.get_tensor(ckpt_prefix + ".weight")
            if is_fp8:
                scale = loader.get_tensor(ckpt_prefix + ".scale")
                weight = dequant_fp8_block(weight, scale)
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            loader.copy_in(param_prefix + ".weight", weight)

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.get_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            for ckpt_name, param_name, shard_dim in (
                ("attn.wq_a", "self_attn.q_a_proj", None),
                ("attn.wq_b", "self_attn.q_b_proj", 0),
                ("attn.wkv", "self_attn.kv_proj", None),
                ("attn.wo_a", "self_attn.o_a_proj", 0),
                ("attn.wo_b", "self_attn.o_b_proj", 1),
            ):
                _linear(ck + ckpt_name, pm + param_name, shard_dim)
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.get_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.get_tensor(ck + "attn.kv_norm.weight"),
            )
            self._load_attention_extras(loader, ck, pm, attn, _linear)
            mlp = self.model.layers[i].mlp
            if isinstance(mlp, DeepseekV41MoE):
                self._load_torch_moe(loader, ck, pm, i, is_fp8)
            elif isinstance(mlp, DeepseekV41TorchMLP):
                for name in ("w1", "w2", "w3"):
                    _linear(ck + f"ffn.{name}", pm + f"mlp.{name}")
            eng = self.model.layers[i].engram
            if eng is not None:
                self._load_engram_torch(loader, ck, pm, eng, is_fp8)
        loader.copy_in("model.norm.weight", loader.get_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_torch_moe(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        mlp = self.model.layers[layer_id].mlp
        loader.copy_in(pm + "mlp.gate.weight", loader.get_tensor(ck + "ffn.gate.weight"))
        loader.copy_in(
            pm + "mlp.e_score_correction_bias",
            loader.get_tensor(ck + "ffn.gate.bias"),
        )
        if loader.has(ck + "ffn.gate.bias_vl"):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.get_tensor(ck + "ffn.gate.bias_vl"))
            mlp.gate_bias_vl_loaded = True
        gate_and_up_slots = mlp.experts_w13.data
        inter = gate_and_up_slots.shape[1] // 2
        for local_slot in range(mlp.num_experts_per_rank):
            expert = mlp.start_expert_id + local_slot
            for name, target in (
                ("w1", gate_and_up_slots[local_slot, :inter]),
                ("w2", mlp.experts_w2.data[local_slot]),
                ("w3", gate_and_up_slots[local_slot, inter:]),
            ):
                prefix = ck + f"ffn.experts.{expert}.{name}"
                weight = loader.get_tensor(prefix + ".weight")
                if is_fp8:
                    weight = dequant_fp4_group(
                        weight,
                        loader.get_tensor(prefix + ".scale"),
                    )
                target.copy_(weight.to(target.dtype))
        shared_prefix = ck + "ffn.shared_experts."
        for name, module_name in (
            ("w1", "shared_w1"),
            ("w2", "shared_w2"),
            ("w3", "shared_w3"),
        ):
            prefix = shared_prefix + name
            weight = loader.get_tensor(prefix + ".weight")
            if is_fp8:
                weight = dequant_fp8_block(
                    weight,
                    loader.get_tensor(prefix + ".scale"),
                )
            loader.copy_in(pm + f"mlp.{module_name}.weight", weight)

    @staticmethod
    def _load_engram_torch(
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        eng: Engram,
        is_fp8: bool,
    ) -> None:
        table = loader.get_tensor(ck + "engram.embed.weight")
        table_u8 = table.view(torch.uint8) if table.element_size() == 1 else table
        scale = loader.get_tensor(ck + "engram.embed.scale")
        scale_u8 = scale.view(torch.uint8) if scale.element_size() == 1 else scale
        start = eng.embed_tokens.vocab_start
        rows = eng.embed_tokens.part_num_embeddings
        if table_u8.size(0) < eng.embed_tokens.num_embeddings:
            raise ValueError(
                f"engram table {ck} has {table_u8.size(0)} rows, config declares {eng.embed_tokens.num_embeddings}"
            )
        if rows > 0:
            local_scale = scale_u8.narrow(0, start, rows)
            _validate_ue8m0_scale_bytes(local_scale)
            eng.embed_tokens.weight.data.copy_(table_u8.narrow(0, start, rows))
            eng.embed_tokens.weight_scale.data.copy_(local_scale)
        wkv = loader.get_tensor(ck + "engram.wkv.weight")
        if is_fp8:
            wkv = dequant_fp8_block(
                wkv,
                loader.get_tensor(ck + "engram.wkv.scale"),
            )
        loader.copy_in(pm + "engram.wkv.weight", wkv)
        loader.copy_in(pm + "engram.q_weight", loader.get_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.get_tensor(ck + "engram.k_weight"))

    def _load_weights_ascend(
        self,
        state_dicts: list[StateDict],
        *,
        layer_prefix: str = "layers",
        embedding_key: str = "embed.weight",
        norm_key: str = "norm.weight",
    ) -> None:
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        def _load_projection(ckpt_prefix: str, param_prefix: str, shard_dim: int | None = None) -> None:
            if isinstance(self.get_submodule(param_prefix), W8A8DynamicLinear):
                shard_dims = (
                    {suffix: shard_dim for suffix in ("weight", "weight_scale", "weight_offset")}
                    if shard_dim is not None
                    else None
                )
                load_w8a8_dynamic_projection(
                    loader,
                    ckpt_prefix,
                    param_prefix,
                    shard_dims,
                    require_weight_and_scale=True,
                    fill_missing_offset=True,
                    description=f"DeepSeek-V4.1 W8A8 {ckpt_prefix}",
                )
                return
            weight = loader.get_tensor(ckpt_prefix + ".weight")
            if not weight.is_floating_point():
                raise ValueError(f"Expected a floating-point weight for {ckpt_prefix}, got {weight.dtype}")
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            loader.copy_in(param_prefix + ".weight", weight)

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.get_tensor(embedding_key), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"{layer_prefix}.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            _load_projection(ck + "attn.wq_a", pm + "self_attn.q_a_proj")
            _load_projection(ck + "attn.wq_b", pm + "self_attn.q_b_proj", shard_dim=0)
            _load_projection(ck + "attn.wkv", pm + "self_attn.kv_proj")
            output_weight = loader.shard(loader.get_tensor(ck + "attn.wo_a.weight"), dim=0)
            if attn.o_a_proj.weight.ndim == 3:
                # A weight reload updates the storage already used by captured graphs.
                output_weight = attn._pack_o_a_weight(output_weight)
            loader.copy_in(pm + "self_attn.o_a_proj.weight", output_weight)
            loader.copy_in(
                pm + "self_attn.o_b_proj.weight",
                loader.shard(loader.get_tensor(ck + "attn.wo_b.weight"), dim=1),
            )
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.get_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.get_tensor(ck + "attn.kv_norm.weight"),
            )
            self._load_attention_extras(loader, ck, pm, attn, _load_projection)
            attn.process_weights_after_loading()
            mlp = self.model.layers[i].mlp
            if isinstance(mlp, DeepseekV4MoE):
                source_prefix = _resolve_dsv4_moe_prefix(loader, ck, f"DeepSeek-V4.1 layer {i}")
                DeepseekV4ForCausalLM._load_dsv4_moe(mlp, loader, ck, pm, i, source_prefix)
                mlp.process_weights_after_loading()
            elif isinstance(mlp, DeepseekV3MLP):
                DeepseekV4ForCausalLM._load_dsv4_dense_mlp(loader, ck, pm, mlp)
            eng = self.model.layers[i].engram
            if eng is not None:
                self._load_engram_ascend(loader, ck, pm, eng)
        loader.copy_in("model.norm.weight", loader.get_tensor(norm_key))
        self._load_lm_head(loader)

    def _load_attention_extras(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        attn: DeepseekV41Attention,
        load_indexer_projection: Callable[[str, str], None],
    ) -> None:
        cfg = self.cfg
        sink_key = ck + "attn.attn_sink"
        if not loader.has(sink_key):
            sink_key = ck + "attn.attn_sink.weight"
        if loader.has(sink_key):
            sink = loader.get_tensor(sink_key)
            if sink.dim() == 1 and sink.size(0) == cfg.n_heads and cfg.tp_size > 1:
                # Preserve already-local sinks; shard full per-head checkpoints.
                sink = loader.shard(sink, dim=0)
            loader.copy_in(pm + "self_attn.attn_sink", sink)
            attn.attn_sink_loaded = True
        self._load_common_layer_norms(loader, ck, pm)
        if attn.mode.is_full:
            loader.copy_in(
                pm + "self_attn.cmp_wkv.weight",
                loader.get_tensor(ck + "attn.compressor.wkv.weight"),
            )
            if attn.mode.compress_ratio > 1:
                loader.copy_in(
                    pm + "self_attn.cmp_wgate.weight",
                    loader.get_tensor(ck + "attn.compressor.wgate.weight"),
                )
            loader.copy_in(
                pm + "self_attn.cmp_norm.weight",
                loader.get_tensor(ck + "attn.compressor.norm.weight"),
            )
        if attn.indexer is not None:
            load_indexer_projection(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
            loader.copy_in(
                pm + "self_attn.indexer.weights_proj.weight",
                loader.get_tensor(ck + "attn.indexer.weights_proj.weight"),
            )
            if attn.indexer.owns_k:
                loader.copy_in(
                    pm + "self_attn.indexer.wk.weight",
                    loader.get_tensor(ck + "attn.indexer.wk.weight"),
                )
                loader.copy_in(
                    pm + "self_attn.indexer.k_norm.weight",
                    loader.get_tensor(ck + "attn.indexer.k_norm.weight"),
                )

    def _load_engram_ascend(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        eng: Engram,
    ) -> None:
        table = loader.get_tensor(ck + "engram.embed.weight")
        if table.dtype == torch.uint8:
            table = table.view(torch.int8)
        if table.dtype != torch.int8:
            raise NotImplementedError(
                f"engram table {ck} has dtype {table.dtype}; the ascend loader requires the int8_sym export"
            )
        scale = loader.get_tensor(ck + "engram.embed.scale")
        if scale.dtype not in (torch.float32, torch.uint8):
            raise ValueError(f"Expected FP32 or E8M0 group scales for {ck}engram.embed, got {scale.dtype}")
        start = eng.embed_tokens.vocab_start
        rows = eng.embed_tokens.part_num_embeddings
        if table.size(0) < eng.embed_tokens.num_embeddings:
            raise ValueError(
                f"engram table {ck} has {table.size(0)} rows, config declares {eng.embed_tokens.num_embeddings}"
            )
        if rows > 0:
            local_scales = scale.narrow(0, start, rows)
            groups = eng.embed_tokens.dim // eng.embed_tokens.quant_block
            if local_scales.shape != (rows, groups):
                raise ValueError(f"Expected {groups} Engram scales per row for {ck}, got {local_scales.shape}")
            local_codes = table.narrow(0, start, rows)
            if eng.embed_tokens.host_offload:
                from xllm.python.models.deepseek_v41_engram_host import EngramHostTable

                eng.embed_tokens.host_table = EngramHostTable(
                    local_codes,
                    local_scales,
                    eng.embed_tokens.weight.device,
                    self.cfg.dp_size,
                    eng.layer_hash_index,
                )
            else:
                if local_scales.dtype == torch.uint8:
                    # The native safetensors loader exposes E8M0 as exponent bytes.
                    _validate_ue8m0_scale_bytes(local_scales)
                # Copy from the mmap view directly into the device parameter.
                eng.embed_tokens.weight.data.copy_(local_codes)
                target_scales = eng.embed_tokens.weight_scale
                target_scales.data = local_scales.to(device=target_scales.device, copy=True)
        loader.copy_in(pm + "engram.wkv.weight", loader.get_tensor(ck + "engram.wkv.weight"))
        loader.copy_in(pm + "engram.q_weight", loader.get_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.get_tensor(ck + "engram.k_weight"))

    def _load_common_layer_norms(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
    ) -> None:
        loader.copy_in(
            pm + "input_layernorm.weight",
            loader.get_tensor(ck + "attn_norm.weight"),
        )
        loader.copy_in(
            pm + "post_attention_layernorm.weight",
            loader.get_tensor(ck + "ffn_norm.weight"),
        )
        for part in ("attn", "ffn"):
            for suffix in ("fn", "scale", "base"):
                name = f"hc_{part}_{suffix}"
                loader.copy_in(pm + "hc." + name, loader.get_tensor(ck + name))

    def _load_lm_head(self, loader: W8A8WeightLoader) -> None:
        lm_head_key = _require_checkpoint_key(
            loader,
            ("lm_head.weight", "model.lm_head.weight", "model.head.weight", "head.weight"),
            "checkpoint output-head weight",
        )
        loader.copy_in(
            "lm_head.weight",
            loader.shard(loader.get_tensor(lm_head_key), dim=0),
        )


_FP4_E2M1_MAX = 6.0
_FP8_AMAX_FLOOR = 1e-4
_FP4_E8M0_AMAX_FLOOR = 6.0 * (2.0**-126)
_FP4_E4M3_AMAX_FLOOR = 6.0 * (2.0**-9)

_FP4_NIBBLE_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Round-half-to-even index on the positive grid for each exact midpoint
# (0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0, ... : ties go to the even mantissa).
_E2M1_MIDPOINT_EVEN_INDEX = (0, 2, 2, 4, 4, 6, 6)


def dequant_fp8_block(
    weight_e4m3: torch.Tensor,
    scale_u8: torch.Tensor,
) -> torch.Tensor:
    if weight_e4m3.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got ndim={weight_e4m3.ndim}")
    block_size = 32
    out_dim, in_dim = weight_e4m3.shape
    expected_shape = (
        (out_dim + block_size - 1) // block_size,
        (in_dim + block_size - 1) // block_size,
    )
    if tuple(scale_u8.shape) != expected_shape:
        raise ValueError(
            f"scale shape {tuple(scale_u8.shape)} does not match expected "
            f"{expected_shape} for weight [{out_dim}, {in_dim}] with block {block_size}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(block_size, dim=0)
    scale = scale.repeat_interleave(block_size, dim=1)[:out_dim, :in_dim]
    return (_e4m3_to_float(weight_e4m3) * scale).to(torch.bfloat16)


def unpack_fp4(
    weight_i8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if weight_i8.dtype == torch.uint8:
        packed = weight_i8
    elif weight_i8.element_size() == 1:
        packed = weight_i8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte packed fp4 storage, got {weight_i8.dtype}")
    table = torch.tensor(_FP4_NIBBLE_VALUES, dtype=torch.float32, device=packed.device)
    low = table[(packed & 0x0F).to(torch.int64)]
    high = table[(packed >> 4).to(torch.int64)]
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    return unpacked.to(dtype)


def dequant_fp4_group(
    weight_i8: torch.Tensor,
    scale_u8: torch.Tensor,
    group: int = 32,
) -> torch.Tensor:
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    unpacked = unpack_fp4(weight_i8, dtype=torch.float32)
    in_dim = unpacked.size(-1)
    expected_groups = (in_dim + group - 1) // group
    if scale_u8.size(-1) != expected_groups:
        raise ValueError(
            f"scale has {scale_u8.size(-1)} groups, expected {expected_groups} for K={in_dim} with group={group}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(group, dim=-1)[..., :in_dim]
    return (unpacked * scale).to(torch.bfloat16)


def fp8_act_qdq(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    orig_dtype = x.dtype
    grouped = x.float().contiguous().unflatten(-1, (-1, group))
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP8_AMAX_FLOOR)
    scale = _pow2(_log2_ceil(amax * (1.0 / _E4M3_MAX)))
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_E4M3_MAX, _E4M3_MAX)
    y = _round_e4m3(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def fp4_act_qdq_e8m0(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    orig_dtype = x.dtype
    grouped = x.float().contiguous().unflatten(-1, (-1, group))
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP4_E8M0_AMAX_FLOOR)
    scale = _pow2(_log2_ceil(amax * (1.0 / _FP4_E2M1_MAX)))
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = _round_e2m1(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def fp4_act_qdq_e4m3_scale(x: torch.Tensor, group: int = 16) -> torch.Tensor:
    orig_dtype = x.dtype
    grouped = x.float().contiguous().unflatten(-1, (-1, group))
    amax = grouped.abs().amax(dim=-1).clamp_min(_FP4_E4M3_AMAX_FLOOR)
    scale = _round_e4m3(amax / _FP4_E2M1_MAX)
    normalized = torch.clamp(grouped / scale.unsqueeze(-1), -_FP4_E2M1_MAX, _FP4_E2M1_MAX)
    y = _round_e2m1(normalized) * scale.unsqueeze(-1)
    return y.flatten(-2).to(orig_dtype)


def _log2_ceil(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    has_mantissa = (bits & 0x7FFFFF) != 0
    return exponent - 127 + has_mantissa.to(torch.int32)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    bits = (exponent + 127) << 23
    return bits.contiguous().view(torch.float32)


_E2M1_TABLE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _e2m1_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cached = _E2M1_TABLE_CACHE.get(device)
    if cached is None:
        cached = (
            torch.tensor(_E2M1_POSITIVE_VALUES, dtype=torch.float32, device=device),
            torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=device),
            torch.tensor(_E2M1_MIDPOINT_EVEN_INDEX, dtype=torch.int64, device=device),
        )
        _E2M1_TABLE_CACHE[device] = cached
    return cached


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    positive, midpoints, even_index = _e2m1_tables(x.device)
    magnitude = x.abs().clamp(max=_FP4_E2M1_MAX)
    index = torch.searchsorted(midpoints, magnitude, right=False)
    clipped = index.clamp(max=len(_E2M1_MIDPOINTS) - 1)
    is_tie = magnitude == midpoints[clipped]
    rounded_index = torch.where(is_tie, even_index[clipped], index)
    rounded = positive[rounded_index]
    return torch.where(x < 0, -rounded, rounded)


_E4M3_MAX = 448.0
_E4M3_SUBNORMAL_THRESHOLD = 2.0**-6


def _round_e4m3(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.abs().clamp(max=_E4M3_MAX)
    # NPU has no frexp kernel; bit inspection avoids a synchronous CPU fallback during capture.
    bits = magnitude.view(torch.int32)
    exp_field = (bits >> 23) & 0xFF
    exponent = torch.where(exp_field > 0, exp_field - 126, 0).to(magnitude.dtype)
    step = torch.exp2(exponent - 4.0)
    q = torch.round(magnitude / step)
    normal_val = q * step
    q_sub = torch.round(magnitude / _E4M3_SUBNORMAL_STEP)
    sub_val = q_sub * _E4M3_SUBNORMAL_STEP
    val = torch.where(magnitude < _E4M3_SUBNORMAL_THRESHOLD, sub_val, normal_val)
    # NPU lacks copysign; signed zero differences disappear on the E4M3 grid.
    return torch.where(x < 0, -val, val)
