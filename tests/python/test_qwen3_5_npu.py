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

"""NPU-specific parallel and composition tests for Python Qwen3.5."""

from __future__ import annotations

import os
import sys
import types
from dataclasses import replace
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.python.qwen3_5_test_utils import (
    ConstantModule as _ConstantModule,
)
from tests.python.qwen3_5_test_utils import (
    StateDict as _StateDict,
)
from tests.python.qwen3_5_test_utils import (
    gemma_rms_norm as _gemma_rms_norm,
)
from tests.python.qwen3_5_test_utils import (
    make_config as _config,
)
from tests.python.qwen3_5_test_utils import (
    make_linear_config as _linear_config,
)
from tests.python.qwen3_5_test_utils import (
    make_moe_checkpoint as _moe_checkpoint,
)
from xllm.python import distributed, kernels
from xllm.python.attention.backend import (
    LayerCache,
    LinearVerifyGraphInputs,
    SharedKVCacheLayout,
    detect_shared_kv_cache_layout,
)
from xllm.python.distributed import collectives
from xllm.python.kernels_npu.causal_conv1d import (
    causal_conv1d_decode as npu_causal_conv1d_decode,
)
from xllm.python.layers.npu.layernorm import NpuGemmaRMSNorm
from xllm.python.layers.npu.mega_moe_metadata import MegaMoeMetadata
from xllm.python.layers.npu.mega_moe_metadata_builder import TokenOwnerMegaMoeMetadataBuilder
from xllm.python.layers.npu.qwen3_5.attention import NpuQwen3_5Attention
from xllm.python.layers.npu.qwen3_5.decoder_layer import (
    NpuQwen3_5DecoderLayer,
)
from xllm.python.layers.npu.qwen3_5.gated_delta_net import (
    NpuQwen3_5GatedDeltaNet,
)
from xllm.python.layers.npu.qwen3_5.gdn_metadata import (
    GdnDecodeMetadata,
    GdnMetadata,
    GdnPrefillMetadata,
    GdnVerifyMetadata,
)
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import (
    Qwen3_5GdnMetadataBuilder,
    _build_mega_prefill_indices,
    _compute_mega_prefill_num_matrices,
)
from xllm.python.layers.npu.qwen3_5.moe import (
    NpuQwen3_5SparseMoEBlock,
)
from xllm.python.layers.qwen3_5.decoder_layer import (
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
    get_execution_context,
)
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models.qwen3_5 import Qwen3_5Config

kernels.gemma_rms_norm = _gemma_rms_norm
kernels.moe_fused_topk = MagicMock()
kernels.grouped_moe_bf16 = MagicMock()
kernels.prepare_row_parallel_weight = MagicMock(side_effect=lambda weight: (weight, False))
distributed.all_gather_variable = MagicMock()
distributed.all_gather = MagicMock()
distributed.gather_dp_execution_tokens = MagicMock()
distributed.all_reduce_ = MagicMock()
distributed.tp_all_reduce = MagicMock()
distributed.moe_tp_all_reduce = MagicMock()
distributed.moe_ep_all_reduce = MagicMock()


@pytest.fixture
def grouped_linear_reference(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    def _project(hidden: torch.Tensor, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        return [hidden @ weight for weight in weights]

    grouped = MagicMock(side_effect=_project)
    monkeypatch.setattr(kernels, "grouped_linear", grouped)
    return grouped


def _mega_moe_config(*, tp_rank: int = 0, dp_rank: int = 0, max_tokens_per_rank: int = 4096) -> Qwen3_5Config:
    return _config(
        tp_size=2,
        tp_rank=tp_rank,
        dp_size=2,
        dp_rank=dp_rank,
        world_size=4,
        moe_tp_size=1,
        moe_tp_rank=0,
        ep_size=4,
        ep_rank=dp_rank * 2 + tp_rank,
        enable_mega_moe=True,
        mega_moe_context=torch.zeros(1, dtype=torch.int32),
        mega_moe_ccl_buffer_size=1024,
        mega_moe_num_max_tokens_per_rank=max_tokens_per_rank,
    )


def _input_batch(
    token_capacity: int,
    scheduled: tuple[int, ...],
    device: torch.device | str = "cpu",
) -> InputBatch:
    total = sum(scheduled)
    return InputBatch(
        input_ids=torch.zeros(token_capacity, dtype=torch.long, device=device),
        positions=torch.zeros(token_capacity, dtype=torch.long, device=device),
        num_reqs=len(scheduled),
        num_tokens=total,
        num_tokens_after_padding=token_capacity,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=(0,) * len(scheduled),
        query_start_loc=tuple(accumulate(scheduled, initial=0)),
        is_prefilling=(True,) * len(scheduled),
        is_padding=(torch.arange(token_capacity, device=device) >= total) if token_capacity != total else None,
        is_dummy=not scheduled,
    )


def _mega_moe_eager_metadata(
    cfg: Qwen3_5Config,
    local_token_count: int,
    execution_token_counts: tuple[int, ...],
) -> MegaMoeMetadata:
    return TokenOwnerMegaMoeMetadataBuilder(cfg).build(
        _input_batch(local_token_count, (local_token_count,) if local_token_count else ()),
        SimpleNamespace(
            slot_mapping=torch.zeros(local_token_count, dtype=torch.int32),
            dp_execution_token_counts=execution_token_counts,
            mega_moe_token_mask=None,
        ),
    )


def _mega_moe_graph_metadata(
    cfg: Qwen3_5Config,
    token_capacity: int,
    local_token_count: int,
) -> MegaMoeMetadata:
    mask = torch.zeros(token_capacity * cfg.dp_size, dtype=torch.int8)
    start = cfg.dp_rank * token_capacity
    mask[start : start + local_token_count].fill_(1)
    return TokenOwnerMegaMoeMetadataBuilder(cfg).build(
        _input_batch(token_capacity, (local_token_count,) if local_token_count else ()),
        SimpleNamespace(
            slot_mapping=torch.zeros(token_capacity, dtype=torch.int32),
            dp_execution_token_counts=(token_capacity,) * cfg.dp_size,
            mega_moe_token_mask=mask,
        ),
    )


def test_npu_decoder_factory_selects_privateuseone_backend() -> None:
    assert get_qwen3_5_decoder_layer_class("privateuseone") is NpuQwen3_5DecoderLayer
    assert NpuQwen3_5DecoderLayer.normalization_cls is NpuGemmaRMSNorm


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_npu_decoder_forward_does_not_require_tilelang_compiler(
    layer_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _linear_config(
        layer_types=[layer_type],
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5DecoderLayer(cfg, 0, torch.bfloat16, torch.device("cpu"), MagicMock())
    hidden = torch.ones(2, cfg.hidden_size, dtype=torch.bfloat16)
    monkeypatch.setattr(layer.input_layernorm, "forward", MagicMock(return_value=hidden))
    monkeypatch.setattr(layer.post_attention_layernorm, "forward", MagicMock(return_value=(hidden, hidden)))
    attention = layer.self_attn if layer_type == "full_attention" else layer.linear_attn
    monkeypatch.setattr(attention, "forward", MagicMock(return_value=hidden))
    monkeypatch.setattr(layer.mlp, "forward", MagicMock(return_value=hidden))
    monkeypatch.setitem(sys.modules, "tilelang", None)

    output, residual = layer(hidden, None, torch.arange(2))

    torch.testing.assert_close(output, hidden)
    torch.testing.assert_close(residual, hidden)


def test_npu_gemma_rms_norm_uses_fused_residual_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = torch.zeros(2, 4, dtype=torch.bfloat16)
    expected_residual = torch.ones(2, 4, dtype=torch.bfloat16)
    fused_norm = MagicMock(return_value=(expected_output, expected_residual))
    monkeypatch.setattr(
        kernels,
        "fused_add_gemma_rms_norm",
        fused_norm,
        raising=False,
    )
    layer = NpuGemmaRMSNorm(
        4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, dtype=torch.bfloat16)

    actual_output, actual_residual = layer(hidden, residual)

    fused_norm.assert_called_once_with(
        hidden,
        residual,
        layer.weight,
        layer.eps,
    )
    assert layer.weight.dtype == torch.bfloat16
    assert actual_output is expected_output
    assert actual_residual is expected_residual


def test_npu_fused_attention_preparation_matches_unfused_after_reload() -> None:
    cfg = _config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
        attention_bias=True,
    )

    def new_layer() -> NpuQwen3_5Attention:
        layer = NpuQwen3_5Attention(
            cfg,
            0,
            torch.float32,
            torch.device("cpu"),
            MagicMock(),
        )
        layer.use_fused_qkv = True
        return layer

    def checkpoint(offset: float) -> dict[str, torch.Tensor]:
        query = torch.arange(
            2 * cfg.n_heads * cfg.head_dim * cfg.hidden_size,
            dtype=torch.float32,
        ).view(2 * cfg.n_heads * cfg.head_dim, cfg.hidden_size)
        key = torch.arange(
            cfg.n_kv_heads * cfg.head_dim * cfg.hidden_size,
            dtype=torch.float32,
        ).view(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size)
        query_bias = torch.arange(
            2 * cfg.n_heads * cfg.head_dim,
            dtype=torch.float32,
        )
        key_bias = torch.arange(
            cfg.n_kv_heads * cfg.head_dim,
            dtype=torch.float32,
        )
        q_norm = torch.arange(cfg.head_dim, dtype=torch.float32)
        return {
            "q_proj.weight": query + offset,
            "k_proj.weight": key + offset,
            "v_proj.weight": key + offset + 100000,
            "q_proj.bias": query_bias + offset,
            "k_proj.bias": key_bias + offset,
            "v_proj.bias": key_bias + offset + 1000,
            "o_proj.weight": torch.full(
                (cfg.hidden_size, cfg.n_heads * cfg.head_dim),
                offset,
            ),
            "o_proj.bias": torch.full((cfg.hidden_size,), offset),
            "q_norm.weight": q_norm + offset,
            "k_norm.weight": q_norm + offset + 100,
        }

    def load(
        layer: NpuQwen3_5Attention,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        layer.load_weights(
            ScopedWeightLoader([_StateDict(tensors)]),
            context,
        )

    context = ParallelLoadContext(tp_rank=0, tp_size=1)
    layer = new_layer()
    initial_checkpoint = checkpoint(0.0)

    load(layer, initial_checkpoint)
    prepared_qkv = layer.qkv_proj.weight.detach().clone()
    assert layer.qkv_proj.bias is not None
    prepared_qkv_bias = layer.qkv_proj.bias.detach().clone()
    prepared_q_norm = layer.q_norm.weight.detach().clone()
    prepared_k_norm = layer.k_norm.weight.detach().clone()

    load(layer, initial_checkpoint)

    torch.testing.assert_close(layer.qkv_proj.weight, prepared_qkv)
    assert layer.qkv_proj.bias is not None
    torch.testing.assert_close(layer.qkv_proj.bias, prepared_qkv_bias)
    torch.testing.assert_close(layer.q_norm.weight, prepared_q_norm)
    torch.testing.assert_close(layer.k_norm.weight, prepared_k_norm)

    updated_checkpoint = checkpoint(10.0)
    incomplete_checkpoint = dict(updated_checkpoint)
    del incomplete_checkpoint["k_norm.weight"]
    with pytest.raises(KeyError, match="k_norm.weight"):
        load(layer, incomplete_checkpoint)

    load(layer, updated_checkpoint)
    fresh_layer = new_layer()
    load(fresh_layer, updated_checkpoint)

    torch.testing.assert_close(layer.qkv_proj.weight, fresh_layer.qkv_proj.weight)
    assert layer.qkv_proj.bias is not None
    assert fresh_layer.qkv_proj.bias is not None
    torch.testing.assert_close(layer.qkv_proj.bias, fresh_layer.qkv_proj.bias)
    torch.testing.assert_close(layer.q_norm.weight, fresh_layer.q_norm.weight)
    torch.testing.assert_close(layer.k_norm.weight, fresh_layer.k_norm.weight)
    torch.testing.assert_close(
        layer.q_norm.weight,
        updated_checkpoint["q_norm.weight"] + 1.0,
    )
    torch.testing.assert_close(
        layer.k_norm.weight,
        updated_checkpoint["k_norm.weight"] + 1.0,
    )

    hidden = torch.arange(
        2 * cfg.hidden_size,
        dtype=torch.float32,
    ).view(2, cfg.hidden_size)
    torch.testing.assert_close(
        layer.qkv_proj(hidden),
        fresh_layer.qkv_proj(hidden),
    )
    unfused_qg = torch.nn.functional.linear(
        hidden,
        updated_checkpoint["q_proj.weight"],
        updated_checkpoint["q_proj.bias"],
    )
    unfused_query, unfused_gate = unfused_qg.view(
        2,
        cfg.n_heads,
        2 * cfg.head_dim,
    ).chunk(2, dim=-1)
    fused_qg = torch.nn.functional.linear(
        hidden,
        layer.qkv_proj.weight[: 2 * layer.q_size],
        layer.qkv_proj.bias[: 2 * layer.q_size],
    )
    fused_query, fused_gate = fused_qg.split(layer.q_size, dim=-1)

    torch.testing.assert_close(
        fused_query,
        unfused_query.reshape(2, layer.q_size),
    )
    torch.testing.assert_close(
        fused_gate,
        unfused_gate.reshape(2, layer.q_size),
    )


def test_npu_gdn_rejects_unsupported_rms_norm_epsilon() -> None:
    cfg = _config(rms_norm_eps=1e-5)

    with pytest.raises(NotImplementedError, match="fixed epsilon"):
        NpuQwen3_5GatedDeltaNet(
            cfg,
            0,
            torch.bfloat16,
            torch.device("cpu"),
        )


def test_npu_gdn_accepts_float32_supported_rms_norm_epsilon() -> None:
    float32_epsilon = float(torch.tensor(1e-6, dtype=torch.float32).item())
    cfg = _config(rms_norm_eps=float32_epsilon)

    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )

    assert layer.cfg.rms_norm_eps == float32_epsilon


@pytest.mark.parametrize(
    ("num_key_heads", "num_value_heads", "error"),
    (
        (3, 3, "power-of-two"),
        (1, 6, "at most four"),
    ),
)
def test_npu_gdn_rejects_geometry_unsupported_by_decode(
    num_key_heads: int,
    num_value_heads: int,
    error: str,
) -> None:
    cfg = _config(
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
    )

    with pytest.raises(NotImplementedError, match=error):
        NpuQwen3_5GatedDeltaNet(
            cfg,
            0,
            torch.bfloat16,
            torch.device("cpu"),
        )


def test_npu_gdn_loads_native_conv_weight_layout() -> None:
    layer = NpuQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    checkpoint_conv = torch.arange(96, dtype=torch.float32).view(24, 1, 4)
    tensors = {
        "in_proj_qkv.weight": torch.zeros(24, 8),
        "in_proj_z.weight": torch.zeros(8, 8),
        "in_proj_b.weight": torch.zeros(2, 8),
        "in_proj_a.weight": torch.zeros(2, 8),
        "conv1d.weight": checkpoint_conv,
        "A_log": torch.zeros(2),
        "dt_bias": torch.zeros(2),
        "norm.weight": torch.zeros(4),
        "out_proj.weight": torch.zeros(8, 8),
    }
    layer.load_weights(ScopedWeightLoader([_StateDict(tensors)]), ParallelLoadContext(tp_rank=0, tp_size=1))
    assert layer.conv1d_weight.shape == (4, 24)
    torch.testing.assert_close(layer.conv1d_weight, checkpoint_conv.squeeze(1).transpose(0, 1))


@pytest.mark.parametrize("tp_size,tp_rank", [(size, rank) for size in (1, 2, 4) for rank in range(size)])
def test_npu_gdn_packs_rank_local_projection_weights(tp_size: int, tp_rank: int) -> None:
    cfg = _linear_config(
        linear_num_key_heads=4,
        linear_num_value_heads=12,
        tp_size=tp_size,
        tp_rank=tp_rank,
        world_size=tp_size,
        moe_tp_size=tp_size,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
    )
    global_key_dim = 16
    global_value_dim = 48
    global_conv_dim = 2 * global_key_dim + global_value_dim
    qkv = torch.arange(global_conv_dim * 8, dtype=torch.float32).view(
        global_conv_dim,
        8,
    )
    z = torch.arange(global_value_dim * 8, dtype=torch.float32).view(
        global_value_dim,
        8,
    )
    b = torch.arange(12 * 8, dtype=torch.float32).view(12, 8)
    a = b + 1000
    tensors = {
        "linear_attn.in_proj_qkv.weight": qkv,
        "linear_attn.in_proj_z.weight": z,
        "linear_attn.in_proj_b.weight": b,
        "linear_attn.in_proj_a.weight": a,
        "linear_attn.conv1d.weight": torch.zeros(global_conv_dim, 1, 4),
        "linear_attn.A_log": torch.zeros(12),
        "linear_attn.dt_bias": torch.zeros(12),
        "linear_attn.norm.weight": torch.zeros(4),
        "linear_attn.out_proj.weight": torch.zeros(8, global_value_dim),
    }

    layer.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "linear_attn."),
        ParallelLoadContext(tp_rank=tp_rank, tp_size=tp_size),
    )

    q, k, v = qkv.split((global_key_dim, global_key_dim, global_value_dim))
    expected_qkv = torch.cat(tuple(weight.chunk(tp_size)[tp_rank] for weight in (q, k, v)))
    qkv_weight, z_weight, b_weight, a_weight = layer._input_projection_weight_views()
    torch.testing.assert_close(qkv_weight, expected_qkv)
    torch.testing.assert_close(z_weight, z.chunk(tp_size)[tp_rank])
    torch.testing.assert_close(b_weight, b.chunk(tp_size)[tp_rank])
    torch.testing.assert_close(a_weight, a.chunk(tp_size)[tp_rank])
    assert qkv_weight.is_contiguous()
    assert z_weight.is_contiguous()
    assert qkv_weight.untyped_storage().data_ptr() == z_weight.untyped_storage().data_ptr()
    assert b_weight.untyped_storage().data_ptr() == a_weight.untyped_storage().data_ptr()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_gdn_grouped_prefill_matches_separate_projections(
    grouped_linear_reference: MagicMock,
    tp_size: int,
    num_tokens: int,
    dtype: torch.dtype,
) -> None:
    layer = NpuQwen3_5GatedDeltaNet(
        _linear_config(
            linear_num_key_heads=4,
            linear_num_value_heads=12,
            tp_size=tp_size,
            world_size=tp_size,
            moe_tp_size=tp_size,
        ),
        0,
        dtype,
        torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for weight in (layer.in_proj_qkvz.weight, layer.in_proj_ba.weight):
            weight.copy_(torch.randn(weight.shape, generator=generator, dtype=dtype))
    hidden = torch.randn(num_tokens, 8, generator=generator, dtype=dtype)
    qkv_weight, z_weight, b_weight, a_weight = layer._input_projection_weight_views()
    expected = (
        torch.nn.functional.linear(hidden, qkv_weight),
        torch.nn.functional.linear(hidden, a_weight),
        torch.nn.functional.linear(hidden, b_weight),
        torch.nn.functional.linear(hidden, z_weight).view(num_tokens, layer.num_v_heads, layer.value_head_dim),
    )
    prefill_outputs = layer._project_prefill_inputs(hidden)
    grouped_linear_reference.assert_called_once()
    weights = grouped_linear_reference.call_args.args[1]
    assert len(weights) == 4
    for weight, source in zip(weights, (qkv_weight, z_weight, b_weight, a_weight)):
        assert weight.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    for actual, reference in zip(prefill_outputs, expected):
        torch.testing.assert_close(actual, reference)
        assert actual.is_contiguous()


@pytest.mark.parametrize("update", ["in_place", "replace", "replace_module", "convert", "assign_state_dict"])
def test_npu_gdn_grouped_prefill_tracks_weight_updates(
    grouped_linear_reference: MagicMock,
    update: str,
) -> None:
    layer = NpuQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        layer.in_proj_qkvz.weight.fill_(0.25)
        layer.in_proj_ba.weight.fill_(0.5)
    hidden = torch.ones(3, 8)
    layer._project_prefill_inputs(hidden)
    if update == "convert":
        layer.double()
        hidden = hidden.double()
    elif update == "replace":
        layer.in_proj_qkvz.weight = torch.nn.Parameter(torch.full_like(layer.in_proj_qkvz.weight, 0.75))
        layer.in_proj_ba.weight = torch.nn.Parameter(torch.full_like(layer.in_proj_ba.weight, 1.0))
    elif update == "replace_module":
        layer.in_proj_qkvz = torch.nn.Linear(8, layer.in_proj_qkvz.weight.shape[0], bias=False)
        layer.in_proj_ba = torch.nn.Linear(8, layer.in_proj_ba.weight.shape[0], bias=False)
        with torch.no_grad():
            layer.in_proj_qkvz.weight.fill_(0.75)
            layer.in_proj_ba.weight.fill_(1.0)
    elif update == "assign_state_dict":
        state = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
        state["in_proj_qkvz.weight"].fill_(0.75)
        state["in_proj_ba.weight"].fill_(1.0)
        layer.load_state_dict(state, assign=True)
    else:
        with torch.no_grad():
            layer.in_proj_qkvz.weight.fill_(0.75)
            layer.in_proj_ba.weight.fill_(1.0)
    expected = layer._project_verify_inputs(hidden)
    actual = layer._project_prefill_inputs(hidden)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_npu_gdn_verify_preserves_separate_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = NpuQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        layer.in_proj_qkvz.weight.fill_(0.25)
        layer.in_proj_ba.weight.fill_(0.5)
    hidden = torch.ones(3, 8)
    linear = MagicMock(wraps=torch.nn.functional.linear)
    monkeypatch.setattr(torch.nn.functional, "linear", linear)

    mixed_qkv, a, b, z = layer._project_verify_inputs(hidden)

    assert linear.call_count == 4
    torch.testing.assert_close(mixed_qkv, torch.full((3, 24), 2.0))
    torch.testing.assert_close(z, torch.full((3, 2, 4), 2.0))
    torch.testing.assert_close(a, torch.full((3, 2), 4.0))
    torch.testing.assert_close(b, torch.full((3, 2), 4.0))


def test_npu_decode_uses_mega_fusion_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_size = 33
    calls = MagicMock()
    mega = MagicMock(side_effect=lambda _qkv, z, *_args: torch.zeros_like(z))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    calls.attach_mock(mega, "mega")
    calls.attach_mock(rms, "rms")
    monkeypatch.setattr(kernels, "mega_gdn_decode", mega, raising=False)
    monkeypatch.setattr(kernels, "causal_conv1d_decode", MagicMock(), raising=False)
    monkeypatch.setattr(kernels, "fused_sigmoid_gating_delta_rule_decode", MagicMock(), raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    with torch.no_grad():
        layer.in_proj_qkvz.weight.zero_()
        layer.in_proj_ba.weight.zero_()
    qkvz_forward = MagicMock(wraps=layer.in_proj_qkvz.forward)
    ba_forward = MagicMock(wraps=layer.in_proj_ba.forward)
    monkeypatch.setattr(layer.in_proj_qkvz, "forward", qkvz_forward)
    monkeypatch.setattr(layer.in_proj_ba, "forward", ba_forward)
    layer.out_proj = torch.nn.Identity()
    state_indices = torch.arange(1, batch_size + 1, dtype=torch.int32)
    metadata = SimpleNamespace(
        linear_state_indices=state_indices,
        has_initial_state=None,
        q_cu_seq_lens=None,
        q_seq_lens_host=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=False,
        is_spec_verify=False,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[
            LayerCache(
                key=None,
                value=None,
                conv=torch.zeros(
                    batch_size + 1,
                    3,
                    384,
                    dtype=torch.bfloat16,
                ),
                ssm=torch.zeros(
                    batch_size + 1,
                    1,
                    128,
                    128,
                    dtype=torch.float32,
                ),
            )
        ],
    )

    metadata.slot_mapping = torch.zeros(batch_size, dtype=torch.int32)
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    layer.execution_metadata_builders = (builder,)
    runner = EagerRunner(layer, None, torch.device("cpu"))
    runner.bind_layer_caches(context.layer_caches)
    context.layer_shared_cache.update(
        runner._build_execution_contexts(metadata, _input_batch(batch_size, (1,) * batch_size))
    )
    with forward_context(context):
        assert layer(torch.zeros(batch_size, 8, dtype=torch.bfloat16)).shape == (
            batch_size,
            128,
        )

    assert [call[0] for call in calls.mock_calls] == ["mega", "mega"]
    assert [call.args[0].shape[0] for call in mega.call_args_list] == [32, 1]
    torch.testing.assert_close(mega.call_args_list[0].args[9], state_indices[:32])
    torch.testing.assert_close(mega.call_args_list[0].args[10], state_indices[:32])
    torch.testing.assert_close(mega.call_args_list[1].args[9], state_indices[32:])
    torch.testing.assert_close(mega.call_args_list[1].args[10], state_indices[32:])
    qkvz_forward.assert_called_once()
    ba_forward.assert_called_once()
    kernels.causal_conv1d_decode.assert_not_called()
    kernels.fused_sigmoid_gating_delta_rule_decode.assert_not_called()
    rms.assert_not_called()


def test_npu_moe_loads_native_weight_order() -> None:
    cfg = _config(tp_rank=0, moe_tp_rank=1)
    layer = NpuQwen3_5DecoderLayer(cfg, 0, torch.bfloat16, torch.device("cpu"), MagicMock())
    tensors = _moe_checkpoint(cfg)
    context = ParallelLoadContext(tp_rank=0, tp_size=2, moe_tp_rank=1, moe_tp_size=2)
    layer.mlp.load_weights(ScopedWeightLoader([_StateDict(tensors)]), context)

    gate, up = tensors["experts.gate_up_proj"].chunk(2, dim=1)
    local_gate, local_up = gate.chunk(2, dim=1)[1], up.chunk(2, dim=1)[1]
    expected_w13 = torch.cat((local_gate, local_up), dim=1).transpose(1, 2)
    expected_w2 = tensors["experts.down_proj"].chunk(2, dim=2)[1].transpose(1, 2)
    assert isinstance(layer.mlp, NpuQwen3_5SparseMoEBlock)
    torch.testing.assert_close(layer.mlp.experts.w13, expected_w13.to(torch.bfloat16))
    torch.testing.assert_close(layer.mlp.experts.w2, expected_w2.to(torch.bfloat16))


@pytest.mark.parametrize("num_tokens", [1, 17])
def test_npu_gdn_uses_npu_prefill_fusion_boundary(
    monkeypatch: pytest.MonkeyPatch,
    grouped_linear_reference: MagicMock,
    num_tokens: int,
) -> None:
    calls = MagicMock()
    mega = MagicMock(return_value=torch.zeros(num_tokens, 1, 128, dtype=torch.bfloat16))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    for name, mock in (("mega", mega), ("rms", rms)):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(kernels, "mega_gdn_prefill", mega, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)
    for old_kernel in (
        "causal_conv1d_qkv_prefill",
        "fused_gdn_gating",
        "chunk_gated_delta_rule",
    ):
        monkeypatch.setattr(kernels, old_kernel, MagicMock(), raising=False)

    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
    )
    with torch.no_grad():
        layer.in_proj_qkvz.weight.zero_()
        layer.in_proj_ba.weight.zero_()
    layer.out_proj = torch.nn.Identity()
    metadata = SimpleNamespace(
        linear_state_indices=torch.tensor([2], dtype=torch.int32),
        linear_state_read_indices=torch.tensor([1], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([2], dtype=torch.int32),
        has_initial_state=torch.tensor([True], dtype=torch.bool),
        q_cu_seq_lens=torch.tensor([0, num_tokens], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([num_tokens], dtype=torch.int32),
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=False,
        is_spec_verify=False,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[
            LayerCache(
                key=None,
                value=None,
                conv=torch.zeros(3, 3, 384, dtype=torch.bfloat16),
                ssm=torch.zeros(3, 1, 128, 128, dtype=torch.float32),
            )
        ],
    )
    metadata.slot_mapping = torch.zeros(num_tokens, dtype=torch.int32)
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    layer.execution_metadata_builders = (builder,)
    runner = EagerRunner(layer, None, torch.device("cpu"))
    runner.bind_layer_caches(context.layer_caches)
    context.layer_shared_cache.update(
        runner._build_execution_contexts(metadata, _input_batch(num_tokens, (num_tokens,)))
    )
    with forward_context(context):
        output = layer(torch.zeros(num_tokens, 8, dtype=torch.bfloat16))

    assert output.shape == (num_tokens, 128)
    assert [call[0] for call in calls.mock_calls] == ["mega"]
    kernels.causal_conv1d_qkv_prefill.assert_not_called()
    kernels.fused_gdn_gating.assert_not_called()
    kernels.chunk_gated_delta_rule.assert_not_called()
    args = mega.call_args.args
    assert all(tensor.is_contiguous() for tensor in args[:4])
    torch.testing.assert_close(args[8], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(args[9], torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(args[10], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(args[11], torch.tensor([2], dtype=torch.int32))
    assert args[15] == 1


@pytest.mark.parametrize("draft_width", [1, 2, 3, 4, 16])
@pytest.mark.parametrize("legacy_chunked", [False, True])
def test_npu_mtp_gdn_keeps_logical_state_rows(
    draft_width: int, legacy_chunked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    width = draft_width + 1
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    cache = LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(3, width + 2, 384, dtype=torch.bfloat16),
        ssm=torch.zeros(3 * width, 1, 128, 128, dtype=torch.float32),
    )
    builder.bind_layer_caches([cache])
    accepted = torch.tensor([1, width], dtype=torch.int32)
    write = torch.tensor([1, 2], dtype=torch.int32)
    read = write
    metadata = SimpleNamespace(
        slot_mapping=torch.zeros(2 * width, dtype=torch.int32),
        q_seq_lens_host=torch.ones(2 * width, dtype=torch.int32),
        linear_state_read_indices=read,
        linear_state_write_indices=write,
        num_accepted_tokens=accepted,
        num_accepted_tokens_host_values=(),
        is_spec_verify=True,
        is_prefill=False,
        is_chunked_prefill=legacy_chunked,
        is_mixed=False,
        expanded_decode_metadata=SimpleNamespace(enabled=legacy_chunked),
    )

    def reject_readback(*args: object, **kwargs: object) -> None:
        raise AssertionError("MTP preparation must not read live accepted lengths on Host")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_readback)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_readback)
    result = builder.allocate_persistent(_input_batch(2 * width, (width, width)), metadata)
    assert isinstance(result, GdnVerifyMetadata)
    assert result.query_width == width
    assert result.state_indices is write
    assert result.packed_to_dense is None
    assert result.num_accepted_tokens is accepted
    assert result.state_caches[0] is cache
    accepted[0] = width
    assert result.num_accepted_tokens.data_ptr() == accepted.data_ptr()


@pytest.mark.parametrize("width", [1, 18])
def test_npu_mtp_gdn_rejects_unsupported_checkpoint_width(width: int) -> None:
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(
        [
            LayerCache(
                key=None,
                value=None,
                conv=torch.zeros(2, width + 2, 384, dtype=torch.bfloat16),
                ssm=torch.zeros(2 * width, 1, 128, 128, dtype=torch.float32),
            )
        ]
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.zeros(width, dtype=torch.int32),
        linear_state_indices=torch.ones(1, dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        is_spec_verify=True,
        is_prefill=False,
        is_chunked_prefill=False,
        is_mixed=False,
    )
    with pytest.raises(ValueError, match="2 to 17 checkpoints"):
        builder.build(_input_batch(width, (width,)), metadata)


def test_npu_mtp_gdn_projects_dense_groups_and_chunks_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("cpu"))
    layer.out_proj = torch.nn.Identity()
    batch_size, width = 33, 3
    indices = torch.arange(1, batch_size + 1, dtype=torch.int32)
    accepted = torch.arange(batch_size, dtype=torch.int32) % width + 1
    cache = LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(batch_size + 1, width + 2, 384, dtype=torch.bfloat16),
        ssm=torch.zeros((batch_size + 1) * width, 1, 128, 128, dtype=torch.float32),
    )
    metadata = GdnVerifyMetadata({0: cache}, indices, accepted, width, None)
    operation = MagicMock(side_effect=lambda _qkv, z, *_args: torch.zeros_like(z))
    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode", operation, raising=False)
    output = layer._verify(torch.zeros(batch_size * width, 8, dtype=torch.bfloat16), metadata)
    assert output.shape == (batch_size * width, 128)
    assert [call.args[0].shape for call in operation.call_args_list] == [(32, width, 384), (1, width, 384)]
    for call, start, end in zip(operation.call_args_list, (0, 32), (32, 33)):
        torch.testing.assert_close(call.args[9], indices[start:end])
        torch.testing.assert_close(call.args[10], indices[start:end])
        torch.testing.assert_close(call.args[11], accepted[start:end])


def test_npu_mega_prefill_builds_checkpoint_indices() -> None:
    conv_read, conv_write, ssm_read, ssm_write = _build_mega_prefill_indices(
        torch.tensor([1, 3], dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32),
        torch.tensor([False, True]),
        checkpoint_stride=2,
    )

    torch.testing.assert_close(conv_read, torch.tensor([-1, 3], dtype=torch.int32))
    torch.testing.assert_close(conv_write, torch.tensor([2, 4], dtype=torch.int32))
    torch.testing.assert_close(ssm_read, torch.tensor([-1, 6], dtype=torch.int32))
    torch.testing.assert_close(ssm_write, torch.tensor([4, 8], dtype=torch.int32))


def _shared_gdn_caches(
    width: int,
    device: torch.device | str = "cpu",
    *,
    num_key_heads: int = 1,
    num_value_heads: int = 1,
    full_head_dim: int = 128,
) -> list[LayerCache]:
    # Exact static-layout geometry for one Full layer and one Linear layer.
    full_row_bytes = 128 * full_head_dim * 2
    conv_dim = (2 * num_key_heads + num_value_heads) * 128
    ssm_row_bytes = num_value_heads * 128 * 128 * 4
    page_bytes = max(8 * full_row_bytes, (width + 1) * ssm_row_bytes)
    arena = torch.zeros(8 * page_bytes, dtype=torch.uint8, device=device)

    def _view(offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        elements = 1
        for dimension in shape:
            elements *= dimension
        element_bytes = torch.empty((), dtype=dtype).element_size()
        return arena.narrow(0, offset, elements * element_bytes).view(dtype).view(shape)

    full_rows = 7 * page_bytes // full_row_bytes + 1
    conv_row_bytes = (width + 2) * conv_dim * 2
    conv_rows = (7 * page_bytes + conv_row_bytes - 1) // conv_row_bytes + 1
    ssm_rows = 7 * page_bytes // ssm_row_bytes + width
    return [
        LayerCache(
            key=None,
            value=None,
            conv=_view(0, (conv_rows, width + 2, conv_dim), torch.bfloat16),
            ssm=_view(ssm_row_bytes, (ssm_rows, num_value_heads, 128, 128), torch.float32),
        ),
        LayerCache(
            key=_view(0, (full_rows, 128, 1, full_head_dim), torch.bfloat16),
            value=_view(full_row_bytes, (full_rows, 128, 1, full_head_dim), torch.bfloat16),
        ),
    ]


def _bounded_qwen38_gdn_builder(
    monkeypatch: pytest.MonkeyPatch, width: int = 5
) -> tuple[Qwen3_5GdnMetadataBuilder, SharedKVCacheLayout]:
    # Qwen3.8-27B TP2: Full=128*2*256*BF16, Conv=(W+2)*5120*BF16,
    # SSM=24*128*128*FP32. Keep the actual geometry while bounding metadata
    # test storage; the shared-arena detector has separate integration tests.
    page_bytes = (width + 1) * 1572864
    layout = SharedKVCacheLayout(
        page_size_tokens=128,
        full_row_stride=page_bytes // 131072,
        conv_row_stride=None,
        ssm_row_stride=width + 1,
        checkpoint_stride=width,
        logical_page_count=500001,
        page_bytes=page_bytes,
        conv_row_bytes=(width + 2) * 5120 * 2,
    )
    monkeypatch.setattr(
        "xllm.python.layers.npu.qwen3_5.gdn_metadata_builder.detect_shared_kv_cache_layout",
        lambda layer_caches: layout,
    )
    cfg = _linear_config(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        tp_size=2,
        world_size=2,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(
        [
            LayerCache(
                key=None,
                value=None,
                conv=torch.empty(1, width + 2, 5120, dtype=torch.bfloat16),
                ssm=torch.empty(width, 24, 128, 128, dtype=torch.float32),
            )
        ]
    )
    return builder, layout


@torch.inference_mode()
def test_npu_shared_gdn_cold_capture_snapshot_restores_selected_state() -> None:
    width = 3
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    caches = _shared_gdn_caches(width)
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(caches)
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    indices = torch.tensor([1, 3], dtype=torch.int32)
    conv_rows = layout.linear_state_indices(indices, role="conv").long()
    ssm_rows = (layout.linear_state_indices(indices, role="ssm")[:, None] + torch.arange(width)).reshape(-1).long()
    conv = torch.randn(2, width + 2, 384).to(torch.bfloat16)
    ssm = torch.randn(2 * width, 1, 128, 128)
    caches[0].conv.index_copy_(0, conv_rows, conv)
    caches[0].ssm.index_copy_(0, ssm_rows, ssm)
    full_rows = torch.tensor([5, 6]) * layout.full_row_stride
    caches[1].key.index_fill_(0, full_rows, 0.25)
    protected_full = caches[1].key.index_select(0, full_rows).clone()
    source = SimpleNamespace(
        is_spec_verify=True,
        linear_state_indices=indices,
        linear_state_read_indices=None,
        linear_state_write_indices=None,
    )
    snapshot = builder.snapshot_graph_state(source)
    caches[0].conv.index_fill_(0, conv_rows, -1)
    caches[0].ssm.index_fill_(0, ssm_rows, -2)
    snapshot.restore()
    torch.testing.assert_close(caches[0].conv.index_select(0, conv_rows), conv, rtol=0, atol=0)
    torch.testing.assert_close(caches[0].ssm.index_select(0, ssm_rows), ssm, rtol=0, atol=0)
    torch.testing.assert_close(caches[1].key.index_select(0, full_rows), protected_full, rtol=0, atol=0)


def test_npu_bounded_qwen38_gdn_prefill_maps_conv_rows_after_page_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    builder, layout = _bounded_qwen38_gdn_builder(monkeypatch)
    assert layout.conv_row_stride is None
    read = torch.tensor([1, 2, 34, 35, -1, -2], dtype=torch.int32)
    write = torch.tensor([2, 1, 35, 34, -2, -1], dtype=torch.int32)
    source = SimpleNamespace(
        linear_state_read_indices=read,
        linear_state_write_indices=write,
        has_initial_state=torch.tensor([True, False, True, True, False, False]),
        q_seq_lens_host=torch.ones(6, dtype=torch.int32),
        q_cu_seq_lens=torch.arange(7, dtype=torch.int32),
        q_cu_seq_lens_host_values=list(range(7)),
        is_spec_verify=False,
        is_prefill=False,
        is_chunked_prefill=True,
    )
    result = builder.build(_input_batch(6, (1,) * 6), source)
    assert isinstance(result, GdnPrefillMetadata)
    assert result.shared_checkpoint_stride == 5
    torch.testing.assert_close(result.conv_read_indices, torch.tensor([132, -1, 4477, 4608, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(
        result.conv_write_indices, torch.tensor([264, 132, 4608, 4477, -2, -1], dtype=torch.int32)
    )
    torch.testing.assert_close(result.ssm_read_indices, torch.tensor([6, -1, 204, 210, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(result.ssm_write_indices, torch.tensor([12, 6, 210, 204, -2, -1], dtype=torch.int32))


@pytest.mark.parametrize("mode", ["eager", "dense", "varlen"])
def test_npu_bounded_qwen38_gdn_verify_refreshes_persistent_physical_indices(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder, _ = _bounded_qwen38_gdn_builder(monkeypatch)
    # The large id also exercises int64 intermediates while the resulting
    # physical indices still use the kernel's int32 metadata ABI.
    logical_pages = torch.tensor([1, 2, 34, 35, -1, -2, 500000], dtype=torch.int32)
    indices = logical_pages.clone()
    accepted = torch.ones(7, dtype=torch.int32)
    source = SimpleNamespace(
        linear_state_indices=indices,
        num_accepted_tokens=accepted,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=True,
    )
    if mode != "eager":
        source.linear_verify_graph = LinearVerifyGraphInputs(
            indices,
            accepted,
            5,
            torch.arange(35, dtype=torch.int64) if mode == "dense" else None,
            torch.arange(35, dtype=torch.int64) if mode == "dense" else None,
            torch.arange(0, 36, 5, dtype=torch.int32) if mode == "varlen" else None,
        )
    batch = _input_batch(35, (5,) * 7)
    persistent = builder.allocate_persistent(batch, source)
    assert isinstance(persistent, GdnVerifyMetadata)
    conv_pointer, ssm_pointer = persistent.conv_state_indices.data_ptr(), persistent.ssm_state_indices.data_ptr()
    expected_conv = torch.tensor([132, 264, 4477, 4608, -1, -2, 65828572], dtype=torch.int32)
    expected_ssm = torch.tensor([6, 12, 204, 210, -1, -2, 3000000], dtype=torch.int32)
    torch.testing.assert_close(persistent.conv_state_indices, expected_conv)
    torch.testing.assert_close(persistent.ssm_state_indices, expected_ssm)
    for order in ([3, 2, 1, 0, 5, 4, 6], [4, 5, 0, 2, 1, 6, 3]):
        positions = torch.tensor(order)
        indices.copy_(logical_pages.index_select(0, positions))
        builder.update_persistent(persistent, batch, source)
        torch.testing.assert_close(persistent.conv_state_indices, expected_conv.index_select(0, positions))
        torch.testing.assert_close(persistent.ssm_state_indices, expected_ssm.index_select(0, positions))
        assert persistent.conv_state_indices.data_ptr() == conv_pointer
        assert persistent.ssm_state_indices.data_ptr() == ssm_pointer


def test_npu_bounded_qwen38_gdn_decode_maps_nonintegral_conv_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    builder, _ = _bounded_qwen38_gdn_builder(monkeypatch, width=1)
    indices = torch.tensor([1, 2, 34, 35, -1, -2], dtype=torch.int32)
    source = SimpleNamespace(
        linear_state_indices=indices,
        is_spec_verify=False,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    result = builder.build(_input_batch(6, (1,) * 6), source)
    assert isinstance(result, GdnDecodeMetadata)
    torch.testing.assert_close(
        result.conv_read_state_indices, torch.tensor([103, 205, 3482, 3584, -1, -2], dtype=torch.int32)
    )
    assert result.conv_read_state_indices is result.conv_write_state_indices
    torch.testing.assert_close(result.ssm_read_state_indices, torch.tensor([2, 4, 68, 70, -1, -2], dtype=torch.int32))
    assert result.ssm_read_state_indices is result.ssm_write_state_indices


@pytest.mark.parametrize("width", [1, 3, 5])
def test_npu_shared_gdn_stages_and_commits_only_selected_physical_rows(width: int) -> None:
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("cpu"))
    conv = torch.arange(10 * (width + 2) * 384, dtype=torch.float32).reshape(10, width + 2, 384)
    conv = conv.to(torch.bfloat16)
    ssm = torch.arange(25 * 128 * 128, dtype=torch.float32).reshape(25, 1, 128, 128)
    before_conv, before_ssm = conv.clone(), ssm.clone()
    physical_conv = torch.tensor([4, 7], dtype=torch.int32)
    physical_ssm = torch.tensor([10, 20], dtype=torch.int32)
    history_rows = 3 if width == 1 else width + 2
    staged = layer._stage_shared_state(
        conv, ssm, physical_conv, physical_ssm, checkpoint_stride=width, conv_rows=history_rows
    )
    torch.testing.assert_close(staged.indices, torch.tensor([1, 2], dtype=torch.int32))
    assert not staged.conv[0].count_nonzero()
    assert not staged.ssm[:width].count_nonzero()
    for dense_slot, conv_slot, ssm_slot in ((1, 4, 10), (2, 7, 20)):
        torch.testing.assert_close(staged.conv[dense_slot], conv[conv_slot, :history_rows])
        torch.testing.assert_close(
            staged.ssm[dense_slot * width : (dense_slot + 1) * width], ssm[ssm_slot : ssm_slot + width]
        )
        staged.conv[dense_slot].fill_(dense_slot + 10)
        staged.ssm[dense_slot * width : (dense_slot + 1) * width].fill_(dense_slot + 20)
        before_conv[conv_slot, :history_rows].fill_(dense_slot + 10)
        before_ssm[ssm_slot : ssm_slot + width].fill_(dense_slot + 20)
    staged.write_back(conv, ssm)
    torch.testing.assert_close(conv, before_conv)
    torch.testing.assert_close(ssm, before_ssm)


def test_npu_shared_gdn_staging_commits_only_valid_destination_slots() -> None:
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("cpu"))
    conv = torch.randn(10, 5, 384, dtype=torch.bfloat16)
    ssm = torch.randn(25, 1, 128, 128)
    expected_conv, expected_ssm = conv.clone(), ssm.clone()
    staged = layer._stage_shared_state(
        conv,
        ssm,
        torch.tensor([4, 7, 3, -1], dtype=torch.int32),
        torch.tensor([10, 20, 15, -1], dtype=torch.int32),
        checkpoint_stride=3,
        conv_rows=5,
    )
    staged.conv[3:].fill_(2)
    staged.ssm[9:].fill_(3)
    expected_conv[3].fill_(2)
    expected_ssm[15:18].fill_(3)
    staged.write_back(conv, ssm, slot_begin=3, valid_slots=torch.tensor([True, False]))
    torch.testing.assert_close(conv, expected_conv)
    torch.testing.assert_close(ssm, expected_ssm)


def test_npu_shared_gdn_graph_staging_reuses_buffers_and_refreshes_state() -> None:
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("cpu"))
    conv = torch.zeros(10, 5, 384, dtype=torch.bfloat16)
    ssm = torch.zeros(25, 1, 128, 128, dtype=torch.float32)
    conv[4].fill_(4)
    conv[7].fill_(7)
    ssm[10:13].fill_(10)
    ssm[20:23].fill_(20)
    state = AclGraphExecutionState({})
    context = ForwardContext(None, torch.device("cpu"), None, [], execution_state=state)
    with forward_context(context):
        first = layer._stage_shared_state(
            conv,
            ssm,
            torch.tensor([4, 7], dtype=torch.int32),
            torch.tensor([10, 20], dtype=torch.int32),
            checkpoint_stride=3,
            conv_rows=5,
        )
        second = layer._stage_shared_state(
            conv,
            ssm,
            torch.tensor([7, 4], dtype=torch.int32),
            torch.tensor([20, 10], dtype=torch.int32),
            checkpoint_stride=3,
            conv_rows=5,
        )
    assert first.conv.data_ptr() == second.conv.data_ptr()
    assert first.ssm.data_ptr() == second.ssm.data_ptr()
    assert first.indices.data_ptr() == second.indices.data_ptr()
    torch.testing.assert_close(second.conv[1], conv[7])
    torch.testing.assert_close(second.ssm[3:6], ssm[20:23])


@pytest.mark.parametrize("packed", [False, True])
def test_npu_shared_gdn_verify_updates_persistent_physical_indices(packed: bool) -> None:
    cfg = _linear_config(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    caches = _shared_gdn_caches(3)
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(caches)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    accepted = torch.tensor([1, 3], dtype=torch.int32)
    endpoints = torch.tensor([0, 1, 4], dtype=torch.int32) if packed else None
    graph = LinearVerifyGraphInputs(
        indices,
        accepted,
        3,
        None if packed else torch.arange(6, dtype=torch.int64),
        None if packed else torch.arange(6, dtype=torch.int64),
        endpoints,
    )
    source = SimpleNamespace(linear_verify_graph=graph, is_prefill=False, is_chunked_prefill=False, is_spec_verify=True)
    batch = _input_batch(4 if packed else 6, (1, 3) if packed else (3, 3))
    persistent = builder.allocate_persistent(batch, source)
    assert isinstance(persistent, GdnVerifyMetadata)
    pointers = (persistent.conv_state_indices.data_ptr(), persistent.ssm_state_indices.data_ptr())
    for ids in ([2, 4], [4, 1], [1, 3]):
        indices.copy_(torch.tensor(ids, dtype=torch.int32))
        builder.update_persistent(persistent, batch, source)
        torch.testing.assert_close(
            persistent.conv_state_indices,
            layout.linear_state_indices(torch.tensor(ids, dtype=torch.int32), role="conv"),
        )
        torch.testing.assert_close(
            persistent.ssm_state_indices, torch.tensor(ids, dtype=torch.int32) * layout.ssm_row_stride
        )
        assert pointers == (persistent.conv_state_indices.data_ptr(), persistent.ssm_state_indices.data_ptr())


@pytest.mark.parametrize("lengths", [(3, 3), (1, 3)], ids=["uniform", "ragged"])
def test_npu_shared_gdn_eager_verify_maps_logical_pages(lengths: tuple[int, int]) -> None:
    cfg = _linear_config(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    caches = _shared_gdn_caches(3)
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(caches)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    accepted = torch.tensor([1, 2], dtype=torch.int32)
    source = SimpleNamespace(
        linear_state_read_indices=indices,
        linear_state_write_indices=indices,
        num_accepted_tokens=accepted,
        num_accepted_tokens_host_values=(1, 2),
        q_seq_lens_host=torch.tensor(lengths, dtype=torch.int32),
        is_spec_verify=True,
        is_prefill=False,
        is_chunked_prefill=True,
    )
    result = builder.build(_input_batch(sum(lengths), lengths), source)
    assert isinstance(result, GdnVerifyMetadata)
    torch.testing.assert_close(result.conv_state_indices, layout.linear_state_indices(indices, role="conv"))
    torch.testing.assert_close(result.ssm_state_indices, indices * layout.ssm_row_stride)
    torch.testing.assert_close(result.state_indices, indices)
    torch.testing.assert_close(result.num_accepted_tokens, accepted)
    if lengths[0] != 3:
        torch.testing.assert_close(result.packed_to_dense, torch.tensor([0, 3, 4, 5]))


@pytest.mark.parametrize("width", [3, 5])
def test_npu_shared_gdn_ordinary_decode_rejects_verify_checkpoints(width: int) -> None:
    cfg = _linear_config(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(_shared_gdn_caches(width))
    source = SimpleNamespace(
        linear_state_indices=torch.tensor([1, 3], dtype=torch.int32),
        is_spec_verify=False,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    with pytest.raises(ValueError, match="requires one SSM row per Conv slot"):
        builder.build(_input_batch(2, (1, 1)), source)


@pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)
@pytest.mark.parametrize(
    "width,packed",
    [(1, False), (3, False), (5, False), (5, True)],
    ids=["decode", "verify3", "verify5", "ragged5"],
)
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_npu_shared_gdn_matches_dense_state_across_reordered_pages(width: int, packed: bool, graph_mode: bool) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(163)
    device = torch.device("npu:0")
    cfg = _linear_config(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    for parameter in layer.parameters():
        parameter.normal_(std=0.1)
    layer.norm_weight.fill_(1)
    shared = _shared_gdn_caches(width, device)
    layout = detect_shared_kv_cache_layout(shared)
    assert layout is not None
    cache = shared[0]
    reference = LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(8, width + 2, 384, device=device, dtype=torch.bfloat16),
        ssm=torch.zeros(8 * width, 1, 128, 128, device=device),
    )
    reference.conv[1:5].normal_(std=0.1)
    reference.ssm[width : 5 * width].normal_(std=0.1)
    initial_conv, initial_ssm = reference.conv.clone(), reference.ssm.clone()
    pages = torch.arange(1, 5, device=device, dtype=torch.int64)
    conv_rows = layout.linear_state_indices(pages, role="conv")
    ssm_rows = (pages[:, None] * layout.ssm_row_stride + torch.arange(width, device=device)).reshape(-1)
    dense_ssm_rows = (pages[:, None] * width + torch.arange(width, device=device)).reshape(-1)

    def restore_shared_state() -> None:
        cache.conv.index_copy_(0, conv_rows, initial_conv[1:5])
        cache.ssm.index_copy_(0, ssm_rows, initial_ssm.index_select(0, dense_ssm_rows))

    restore_shared_state()
    # Full pages occupy the same arena and must survive every Linear update.
    full_rows = torch.tensor([5, 6], device=device, dtype=torch.int64) * layout.full_row_stride
    shared[1].key.index_fill_(0, full_rows, 0.25)
    shared[1].value.index_fill_(0, full_rows, 0.5)
    full_key = shared[1].key.index_select(0, full_rows).clone()
    full_value = shared[1].value.index_select(0, full_rows).clone()
    indices = torch.tensor([1, 3], device=device, dtype=torch.int32)
    accepted = torch.ones(2, device=device, dtype=torch.int32)
    lengths = (1, width) if packed else (width, width)
    token_count = sum(lengths)
    hidden = torch.zeros(token_count, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    input_batch = _input_batch(token_count, lengths, device)
    source = SimpleNamespace(
        linear_state_read_indices=indices,
        linear_state_write_indices=indices,
        num_accepted_tokens=accepted,
        num_accepted_tokens_host_values=(1, 1),
        q_seq_lens_host=torch.tensor(lengths, dtype=torch.int32),
        is_spec_verify=width != 1,
        is_prefill=False,
        is_chunked_prefill=width != 1,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(shared)
    metadata = builder.allocate_persistent(input_batch, source)
    reference_builder = Qwen3_5GdnMetadataBuilder(cfg)
    reference_builder.bind_layer_caches([reference])
    reference_metadata = reference_builder.allocate_persistent(input_batch, source)
    if width != 1:
        assert isinstance(metadata, GdnVerifyMetadata)
        assert isinstance(reference_metadata, GdnVerifyMetadata)
        assert metadata.cu_seqlens is None
        assert reference_metadata.cu_seqlens is None
        if packed:
            assert metadata.packed_to_dense is not None
            torch.testing.assert_close(
                metadata.packed_to_dense,
                torch.tensor([0, *range(width, 2 * width)], device=device, dtype=torch.int64),
            )
        else:
            assert metadata.packed_to_dense is None
    persistent_addresses = [
        (current, name, tensor.data_ptr())
        for current in (metadata, reference_metadata)
        for name in (
            "state_indices",
            "num_accepted_tokens",
            "packed_to_dense",
            "conv_state_indices",
            "ssm_state_indices",
            "read_state_indices",
            "write_state_indices",
            "conv_read_state_indices",
            "conv_write_state_indices",
            "ssm_read_state_indices",
            "ssm_write_state_indices",
        )
        if (tensor := getattr(current, name, None)) is not None
    ]
    state = AclGraphExecutionState({})
    context = ForwardContext(None, device, None, shared, execution_state=state)

    def shared_forward() -> torch.Tensor:
        if isinstance(metadata, GdnVerifyMetadata):
            return layer._verify(hidden, metadata)
        assert isinstance(metadata, GdnDecodeMetadata)
        return layer._decode(
            hidden,
            cache.conv,
            cache.ssm,
            metadata.read_state_indices,
            metadata.write_state_indices,
            metadata.conv_read_state_indices,
            metadata.conv_write_state_indices,
            metadata.ssm_read_state_indices,
            metadata.ssm_write_state_indices,
        )

    if graph_mode:
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream), forward_context(context):
            for _ in range(3):
                shared_forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                actual = shared_forward()
        torch.npu.current_stream(device).wait_stream(stream)
        restore_shared_state()
        torch.npu.synchronize()

    for iteration, logical_pages in enumerate(([1, 3], [3, 1], [2, 4], [4, 2])):
        indices.copy_(torch.tensor(logical_pages, device=device, dtype=torch.int32))
        counts = (1, min(width, iteration + 1))
        accepted.copy_(torch.tensor(counts, device=device, dtype=torch.int32))
        source.num_accepted_tokens_host_values = counts
        builder.update_persistent(metadata, input_batch, source)
        reference_builder.update_persistent(reference_metadata, input_batch, source)
        for current, name, address in persistent_addresses:
            assert getattr(current, name).data_ptr() == address
        hidden.normal_(std=0.1)
        if graph_mode:
            graph.replay()
        else:
            actual = shared_forward()
        if isinstance(reference_metadata, GdnVerifyMetadata):
            expected = layer._verify(hidden, reference_metadata)
        else:
            assert isinstance(reference_metadata, GdnDecodeMetadata)
            expected = layer._decode(
                hidden,
                reference.conv,
                reference.ssm,
                reference_metadata.read_state_indices,
                reference_metadata.write_state_indices,
            )
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(cache.conv.index_select(0, conv_rows), reference.conv[1:5], atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            cache.ssm.index_select(0, ssm_rows),
            reference.ssm.index_select(0, dense_ssm_rows),
            atol=0.02,
            rtol=0.02,
        )
        torch.testing.assert_close(shared[1].key.index_select(0, full_rows), full_key, rtol=0, atol=0)
        torch.testing.assert_close(shared[1].value.index_select(0, full_rows), full_value, rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)
@pytest.mark.parametrize(
    "width,num_key_heads,num_value_heads,full_head_dim",
    [(1, 1, 1, 128), (3, 1, 1, 128), (5, 1, 1, 128), (5, 4, 12, 256)],
    ids=["stride1", "stride3", "stride5", "qwen38-tp4"],
)
@torch.inference_mode()
def test_npu_shared_gdn_prefill_matches_dense_state_across_chunks(
    width: int, num_key_heads: int, num_value_heads: int, full_head_dim: int
) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.manual_seed(167)
    device = torch.device("npu:0")
    cfg = _linear_config(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=num_key_heads,
        linear_num_value_heads=num_value_heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    for parameter in layer.parameters():
        parameter.normal_(std=0.1)
    layer.norm_weight.fill_(1)
    layer.out_proj = torch.nn.Identity()
    shared = _shared_gdn_caches(
        width,
        device,
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        full_head_dim=full_head_dim,
    )
    layout = detect_shared_kv_cache_layout(shared)
    assert layout is not None
    cache = shared[0]
    assert cache.ssm.shape[0] % cache.conv.shape[0] != 0
    reference = LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(8, width + 2, layer.conv_dim, device=device, dtype=torch.bfloat16),
        ssm=torch.zeros(8 * width, num_value_heads, 128, 128, device=device),
    )
    reference.conv[1:5].normal_(std=0.1)
    reference.ssm[width : 5 * width].normal_(std=0.1)
    pages = torch.arange(1, 5, device=device, dtype=torch.int64)
    conv_rows = layout.linear_state_indices(pages, role="conv")
    ssm_rows = (layout.linear_state_indices(pages, role="ssm")[:, None] + torch.arange(width, device=device)).reshape(
        -1
    )
    dense_ssm_rows = (pages[:, None] * width + torch.arange(width, device=device)).reshape(-1)
    cache.conv.index_copy_(0, conv_rows, reference.conv[1:5])
    cache.ssm.index_copy_(0, ssm_rows, reference.ssm.index_select(0, dense_ssm_rows))
    full_rows = torch.tensor([5, 6], device=device, dtype=torch.int64) * layout.full_row_stride
    shared[1].key.index_fill_(0, full_rows, 0.25)
    shared[1].value.index_fill_(0, full_rows, 0.5)
    full_key = shared[1].key.index_select(0, full_rows).clone()
    full_value = shared[1].value.index_select(0, full_rows).clone()
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(shared)
    reference_builder = Qwen3_5GdnMetadataBuilder(cfg)
    reference_builder.bind_layer_caches([reference])
    computed = {page: 0 for page in range(1, 5)}
    steps = (
        ((1, 3), (1, 3), (129, 3), (False, False)),
        ((3, 1), (2, 4), (5, 130), (True, True)),
        ((2, 4), (1, 3), (7, 2), (True, True)),
    )
    for step, (read_pages, write_pages, lengths, has_initial_state) in enumerate(steps):
        num_tokens = sum(lengths)
        batch = replace(
            _input_batch(num_tokens, lengths, device),
            num_computed_tokens=tuple(computed[page] for page in read_pages),
        )
        source = SimpleNamespace(
            linear_state_read_indices=torch.tensor(read_pages, device=device, dtype=torch.int32),
            linear_state_write_indices=torch.tensor(write_pages, device=device, dtype=torch.int32),
            has_initial_state=torch.tensor(has_initial_state, device=device, dtype=torch.bool),
            q_seq_lens_host=torch.tensor(lengths, dtype=torch.int32),
            q_cu_seq_lens=torch.tensor(batch.query_start_loc, device=device, dtype=torch.int32),
            q_cu_seq_lens_host_values=list(batch.query_start_loc),
            is_spec_verify=False,
            is_prefill=step == 0,
            is_chunked_prefill=step != 0,
        )
        metadata = builder.build(batch, source)
        reference_metadata = reference_builder.build(batch, source)
        assert isinstance(metadata, GdnPrefillMetadata)
        assert isinstance(reference_metadata, GdnPrefillMetadata)
        assert metadata.shared_checkpoint_stride == width
        assert reference_metadata.shared_checkpoint_stride is None
        hidden = torch.randn(num_tokens, cfg.hidden_size, device=device, dtype=torch.bfloat16) * 0.1
        with forward_context(ForwardContext(None, device, source, shared, layer_shared_cache={GdnMetadata: metadata})):
            actual = layer(hidden)
        with forward_context(
            ForwardContext(None, device, source, [reference], layer_shared_cache={GdnMetadata: reference_metadata})
        ):
            expected = layer(hidden)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(cache.conv.index_select(0, conv_rows), reference.conv[1:5], atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            cache.ssm.index_select(0, ssm_rows), reference.ssm.index_select(0, dense_ssm_rows), atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(shared[1].key.index_select(0, full_rows), full_key, rtol=0, atol=0)
        torch.testing.assert_close(shared[1].value.index_select(0, full_rows), full_value, rtol=0, atol=0)
        for write, previous, length in zip(write_pages, batch.num_computed_tokens, lengths):
            computed[write] = previous + length


def test_npu_mega_prefill_metadata_is_shared_within_one_forward() -> None:
    cfg = _linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    cache = LayerCache(
        key=None,
        value=None,
        conv=torch.zeros(5, 4, 640, dtype=torch.bfloat16),
        ssm=torch.zeros(10, 3, 128, 128, dtype=torch.float32),
    )
    model = torch.nn.Identity()
    model.execution_metadata_builders = (builder,)
    runner = EagerRunner(model, None, torch.device("cpu"))
    runner.bind_layer_caches([cache])
    metadata = SimpleNamespace(
        linear_state_read_indices=torch.tensor([1, 3], dtype=torch.int32),
        linear_state_write_indices=torch.tensor([2, 4], dtype=torch.int32),
        has_initial_state=torch.tensor([False, True]),
        q_seq_lens_host=torch.tensor([127, 129], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 127, 256], dtype=torch.int32),
        q_cu_seq_lens_host_values=[127, 256],
        slot_mapping=torch.zeros(256, dtype=torch.int32),
        is_prefill=True,
        is_chunked_prefill=False,
        is_dummy=False,
        is_spec_verify=False,
    )
    context = ForwardContext(
        None,
        torch.device("cpu"),
        metadata,
        [cache],
        layer_shared_cache=runner._build_execution_contexts(metadata, _input_batch(256, (127, 129))),
    )
    with forward_context(context):
        first = get_execution_context(GdnMetadata)
        second = get_execution_context(GdnMetadata)
    assert first is second
    assert isinstance(first, GdnPrefillMetadata)
    assert first.num_matrices == 9
    torch.testing.assert_close(first.cu_seqlens, torch.tensor([0, 127, 256], dtype=torch.int32))
    assert first.cu_seqlens is metadata.q_cu_seq_lens
    torch.testing.assert_close(first.ssm_read_indices, torch.tensor([-1, 6], dtype=torch.int32))
    assert first.state_caches[0] is cache


@pytest.mark.parametrize(
    ("query_lengths", "num_value_heads", "expected"),
    (
        ([1], 2, 2),
        ([127, 128, 129], 3, 12),
    ),
)
def test_npu_mega_prefill_num_matrices(
    query_lengths: list[int],
    num_value_heads: int,
    expected: int,
) -> None:
    assert _compute_mega_prefill_num_matrices(query_lengths, num_value_heads) == expected


def test_npu_decode_passes_native_weight_and_cache_to_tilelang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = torch.zeros(1, 24)
    kernel = MagicMock(return_value=output)
    tilelang_wrapper = types.ModuleType("xllm.python.kernels_npu.tilelang.causal_conv1d_decode")
    tilelang_wrapper.DIM_PER_CORE = 2048
    tilelang_wrapper._build_decode_kernel_jit = MagicMock(return_value=kernel)
    monkeypatch.setitem(sys.modules, "xllm.python.kernels_npu.tilelang.causal_conv1d_decode", tilelang_wrapper)
    value = torch.zeros(1, 24)
    weight = torch.zeros(4, 24)
    conv_state = torch.zeros(2, 3, 24)
    state_indices = torch.tensor([1], dtype=torch.int32)
    actual = npu_causal_conv1d_decode(value, weight, conv_state, state_indices)
    assert actual is output
    assert kernel.call_args.args[1] is weight
    assert kernel.call_args.args[2] is conv_state


def test_npu_moe_rejects_non_bf16_weights() -> None:
    with pytest.raises(NotImplementedError, match="support BF16 only"):
        NpuQwen3_5SparseMoEBlock(_config(), torch.float16, torch.device("cpu"))


def test_npu_moe_partitions_expert_axis_for_ep() -> None:
    cfg = _config(tp_size=4, tp_rank=2, world_size=4, moe_tp_size=1, moe_tp_rank=0, ep_size=4, ep_rank=2)
    layer = NpuQwen3_5DecoderLayer(cfg, 0, torch.bfloat16, torch.device("cpu"), MagicMock())
    tensors = _moe_checkpoint(cfg)
    context = ParallelLoadContext(tp_rank=2, tp_size=4, moe_tp_rank=0, moe_tp_size=1, ep_rank=2, ep_size=4)
    layer.mlp.load_weights(ScopedWeightLoader([_StateDict(tensors)]), context)
    expected_gate_up = tensors["experts.gate_up_proj"][4:6].transpose(1, 2)
    expected_down = tensors["experts.down_proj"][4:6].transpose(1, 2)
    torch.testing.assert_close(layer.mlp.experts.w13, expected_gate_up.to(torch.bfloat16))
    torch.testing.assert_close(layer.mlp.experts.w2, expected_down.to(torch.bfloat16))


@pytest.mark.parametrize(
    ("moe_tp_size", "ep_size", "dtype"),
    ((2, 1, torch.float32), (1, 4, torch.float32), (2, 2, torch.bfloat16)),
)
def test_npu_moe_uses_parallel_collectives(
    monkeypatch: pytest.MonkeyPatch,
    moe_tp_size: int,
    ep_size: int,
    dtype: torch.dtype,
) -> None:
    world_size = moe_tp_size * ep_size
    cfg = _config(
        tp_size=world_size,
        world_size=world_size,
        moe_tp_size=moe_tp_size,
        ep_size=ep_size,
        ep_rank=1 if ep_size > 1 else 0,
    )
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"))
    if ep_size == 1:
        block.experts.reduce_results = True
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=dtype))
    topk = (
        torch.ones(3, cfg.num_experts_per_tok, dtype=dtype),
        torch.zeros(3, cfg.num_experts_per_tok, dtype=torch.int32),
    )
    monkeypatch.setattr(kernels, "moe_fused_topk", MagicMock(return_value=topk))
    grouped_output = torch.ones(3, cfg.hidden_size, dtype=dtype)
    grouped = MagicMock(return_value=grouped_output)
    monkeypatch.setattr(kernels, "grouped_moe_bf16", grouped, raising=False)
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()
    assert block.experts(torch.zeros_like(grouped_output)) is grouped_output
    grouped.assert_called_once()
    reductions = (
        (moe_tp_size, distributed.moe_tp_all_reduce),
        (ep_size, distributed.moe_ep_all_reduce),
    )
    for size, collective in reductions:
        if size > 1:
            collective.assert_called_once_with(grouped_output)
        else:
            collective.assert_not_called()


@pytest.mark.parametrize(
    ("dp_rank", "token_counts", "dp_is_decode", "is_graph"),
    (
        pytest.param(0, (4, 4), (1, 1), True, id="graph-equal"),
        pytest.param(1, (3, 4), (1, 1), False, id="uneven-rank1"),
        pytest.param(0, (2, 3), (0, 1), False, id="mixed-prefill"),
        pytest.param(1, (3, 0), (1, 1), False, id="empty-rank1"),
        pytest.param(0, (0, 5), (1, 1), False, id="empty-rank0"),
    ),
)
def test_npu_moe_gathers_and_slices_execution_counts(
    monkeypatch: pytest.MonkeyPatch,
    dp_rank: int,
    token_counts: tuple[int, int],
    dp_is_decode: tuple[int, int],
    is_graph: bool,
) -> None:
    cfg = _config(tp_size=1, dp_size=2, dp_rank=dp_rank, moe_tp_size=2)
    block = NpuQwen3_5SparseMoEBlock(cfg, torch.bfloat16, torch.device("cpu"))
    counts = tuple(max(count, 1) for count in token_counts)
    rows, start, local_rows = sum(counts), sum(counts[:dp_rank]), counts[dp_rank]
    gathered = torch.arange(1, rows * cfg.hidden_size + 1, dtype=torch.float32)
    gathered = gathered.view(rows, cfg.hidden_size)
    grouped_output = gathered + 1000
    block.experts.gate = _ConstantModule(torch.zeros(rows, cfg.num_experts))
    topk = (
        torch.ones(rows, cfg.num_experts_per_tok),
        torch.zeros(rows, cfg.num_experts_per_tok, dtype=torch.int32),
    )
    monkeypatch.setattr(kernels, "moe_fused_topk", MagicMock(return_value=topk))
    grouped = MagicMock(return_value=grouped_output)
    monkeypatch.setattr(kernels, "grouped_moe_bf16", grouped, raising=False)
    # Restore the real branch owner; only its lower collectives are doubled.
    monkeypatch.setattr(distributed, "gather_dp_execution_tokens", collectives.gather_dp_execution_tokens)
    fixed, variable = MagicMock(return_value=gathered), MagicMock(return_value=gathered)
    monkeypatch.setattr(collectives, "all_gather", fixed)
    monkeypatch.setattr(collectives, "all_gather_variable", variable)
    metadata = SimpleNamespace(
        dp_execution_token_counts=counts,
        dp_is_decode=dp_is_decode,
        is_prefill=dp_is_decode[dp_rank] == 0,
        is_chunked_prefill=False,
    )
    context = ForwardContext(
        None,
        torch.device("cpu"),
        metadata,
        [],
        execution_state=AclGraphExecutionState({}) if is_graph else None,
    )
    local_input = gathered[start : start + local_rows].clone()
    with forward_context(context):
        output = block.experts(local_input)
    torch.testing.assert_close(output, grouped_output[start : start + local_rows])
    grouped.assert_called_once()
    assert grouped.call_args.args[0] is gathered
    if counts[0] == counts[1]:
        fixed.assert_called_once_with(local_input, dim=0, world_size=2, group_name="dp")
        variable.assert_not_called()
    else:
        variable.assert_called_once_with(local_input, list(counts), dp_rank, "dp")
        fixed.assert_not_called()


def test_npu_moe_token_owner_uses_shared_eager_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mega_moe_config()
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    hidden = (
        torch.arange(
            2 * cfg.hidden_size,
            dtype=torch.float32,
        )
        .view(2, cfg.hidden_size)
        .to(torch.bfloat16)
    )
    gate_forward = MagicMock(return_value=torch.zeros(2, cfg.num_experts, dtype=torch.bfloat16))
    block.experts.gate.forward = gate_forward
    topk_weights = torch.full(
        (2, cfg.num_experts_per_tok),
        0.5,
        dtype=torch.float32,
    )
    topk_ids = torch.zeros(
        2,
        cfg.num_experts_per_tok,
        dtype=torch.int32,
    )
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(return_value=(topk_weights, topk_ids)),
        raising=False,
    )
    owner_output = (
        torch.arange(
            3 * cfg.hidden_size,
            dtype=torch.float32,
        )
        .view(3, cfg.hidden_size)
        .to(torch.bfloat16)
    )
    mega_moe = MagicMock(return_value=owner_output)
    monkeypatch.setattr(kernels, "mega_moe", mega_moe, raising=False)
    broadcast = MagicMock()
    monkeypatch.setattr(distributed, "broadcast_", broadcast, raising=False)
    distributed.gather_dp_execution_tokens.reset_mock()
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(2, 3)),
        layer_caches=[],
        layer_shared_cache={
            MegaMoeMetadata: _mega_moe_eager_metadata(
                cfg,
                local_token_count=2,
                execution_token_counts=(2, 3),
            ),
        },
    )

    with forward_context(context):
        output = block.experts(hidden)

    torch.testing.assert_close(output, owner_output[:2])
    mega_moe_args = mega_moe.call_args.args
    assert mega_moe_args[0] is cfg.mega_moe_context
    assert mega_moe_args[10] == cfg.mega_moe_ccl_buffer_size
    assert mega_moe_args[11] == cfg.mega_moe_num_max_tokens_per_rank
    owner_input = mega_moe_args[1]
    assert owner_input.shape == (3, cfg.hidden_size)
    torch.testing.assert_close(owner_input[:2], hidden)
    torch.testing.assert_close(owner_input[2], torch.zeros_like(hidden[0]))
    assert mega_moe_args[2].shape == (3, cfg.num_experts_per_tok)
    assert mega_moe_args[3].shape == (3, cfg.num_experts_per_tok)
    torch.testing.assert_close(
        mega_moe_args[2][2],
        torch.zeros(cfg.num_experts_per_tok, dtype=torch.int32),
    )
    torch.testing.assert_close(
        mega_moe_args[3][2],
        torch.zeros(cfg.num_experts_per_tok, dtype=torch.float32),
    )
    torch.testing.assert_close(
        mega_moe_args[12],
        torch.tensor([1, 1, 0], dtype=torch.int8),
    )
    torch.testing.assert_close(gate_forward.call_args.args[0], hidden)
    broadcast.assert_called_once_with(output, 0, "tp")
    distributed.gather_dp_execution_tokens.assert_not_called()


def test_npu_moe_token_non_owner_uses_shared_eager_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mega_moe_config(tp_rank=1)
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    hidden = torch.ones(2, cfg.hidden_size, dtype=torch.bfloat16)
    gate_forward = MagicMock(side_effect=AssertionError("non-owner ran the router"))
    block.experts.gate.forward = gate_forward
    mega_moe = MagicMock(return_value=torch.zeros(3, cfg.hidden_size, dtype=torch.bfloat16))
    monkeypatch.setattr(kernels, "mega_moe", mega_moe, raising=False)
    expected = torch.full_like(hidden, 7)

    def _broadcast(output: torch.Tensor, src: int, group_name: str) -> None:
        assert src == 0
        assert group_name == "tp"
        output.copy_(expected)

    monkeypatch.setattr(distributed, "broadcast_", _broadcast, raising=False)
    mega_moe_metadata = _mega_moe_eager_metadata(
        cfg,
        local_token_count=2,
        execution_token_counts=(2, 3),
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(2, 3)),
        layer_caches=[],
        layer_shared_cache={
            MegaMoeMetadata: mega_moe_metadata,
        },
    )

    with forward_context(context):
        output = block.experts(hidden)

    mega_moe_args = mega_moe.call_args.args
    torch.testing.assert_close(output, expected)
    assert mega_moe_args[1] is mega_moe_metadata.dummy_input
    assert mega_moe_args[2] is mega_moe_metadata.dummy_topk_ids
    assert mega_moe_args[3] is mega_moe_metadata.dummy_topk_weights
    torch.testing.assert_close(
        mega_moe_args[1],
        torch.zeros(3, cfg.hidden_size, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        mega_moe_args[2],
        torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        mega_moe_args[3],
        torch.full((3, cfg.num_experts_per_tok), 0.5, dtype=torch.float32),
    )
    torch.testing.assert_close(
        mega_moe_args[12],
        torch.tensor([1, 0, 0], dtype=torch.int8),
    )
    gate_forward.assert_not_called()


def test_npu_moe_requires_execution_metadata() -> None:
    cfg = _mega_moe_config()
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(2, 3)),
        layer_caches=[],
    )

    with (
        forward_context(context),
        pytest.raises(
            RuntimeError,
            match="execution metadata is unavailable",
        ),
    ):
        block.experts(torch.zeros(2, cfg.hidden_size, dtype=torch.bfloat16))


def test_npu_moe_token_owner_reuses_graph_mask_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mega_moe_config(dp_rank=1)
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    hidden = (
        torch.arange(
            4 * cfg.hidden_size,
            dtype=torch.float32,
        )
        .view(4, cfg.hidden_size)
        .to(torch.bfloat16)
    )
    distributed.gather_dp_execution_tokens.reset_mock()
    block.experts.gate.forward = MagicMock(return_value=torch.zeros(4, cfg.num_experts, dtype=torch.bfloat16))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(
                    4,
                    cfg.num_experts_per_tok,
                    dtype=torch.bfloat16,
                ),
                torch.zeros(4, cfg.num_experts_per_tok, dtype=torch.int32),
            )
        ),
        raising=False,
    )
    owner_output = torch.full_like(hidden, 3)
    mega_moe = MagicMock(return_value=owner_output)
    monkeypatch.setattr(kernels, "mega_moe", mega_moe, raising=False)
    broadcast = MagicMock()
    monkeypatch.setattr(distributed, "broadcast_", broadcast, raising=False)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(4, 2)),
        layer_caches=[],
        execution_state=AclGraphExecutionState({}),
        layer_shared_cache={
            MegaMoeMetadata: _mega_moe_graph_metadata(
                cfg,
                token_capacity=4,
                local_token_count=2,
            ),
        },
    )

    with forward_context(context):
        output = block.experts(hidden)

    torch.testing.assert_close(output, owner_output)
    torch.testing.assert_close(mega_moe.call_args.args[1], hidden)
    assert mega_moe.call_args.args[3].dtype == torch.float32
    torch.testing.assert_close(
        mega_moe.call_args.args[12],
        torch.tensor([1, 1, 0, 0], dtype=torch.int8),
    )
    distributed.gather_dp_execution_tokens.assert_not_called()
    broadcast.assert_called_once_with(output, 0, "tp")


def test_npu_moe_token_non_owner_reuses_graph_dummy_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mega_moe_config(tp_rank=1)
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    hidden = torch.ones(4, cfg.hidden_size, dtype=torch.bfloat16)
    block.experts.gate.forward = MagicMock(side_effect=AssertionError("non-owner ran the router"))
    mega_moe = MagicMock(return_value=torch.zeros_like(hidden))
    monkeypatch.setattr(kernels, "mega_moe", mega_moe, raising=False)
    monkeypatch.setattr(distributed, "broadcast_", MagicMock(), raising=False)
    mega_moe_metadata = _mega_moe_graph_metadata(
        cfg,
        token_capacity=4,
        local_token_count=4,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(4, 4)),
        layer_caches=[],
        execution_state=AclGraphExecutionState({}),
        layer_shared_cache={
            MegaMoeMetadata: mega_moe_metadata,
        },
    )

    with forward_context(context):
        output = block.experts(hidden)

    first_args = mega_moe.call_args.args
    assert output.shape == hidden.shape
    assert first_args[1] is mega_moe_metadata.dummy_input
    assert first_args[2] is mega_moe_metadata.dummy_topk_ids
    assert first_args[3] is mega_moe_metadata.dummy_topk_weights
    torch.testing.assert_close(
        first_args[1],
        torch.zeros(4, cfg.hidden_size, dtype=torch.bfloat16),
    )
    torch.testing.assert_close(
        first_args[12],
        torch.tensor([1, 0, 0, 0], dtype=torch.int8),
    )
    block.experts.gate.forward.assert_not_called()


def test_npu_moe_falls_back_when_capacity_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _mega_moe_config(max_tokens_per_rank=2)
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    hidden = torch.ones(3, cfg.hidden_size, dtype=torch.bfloat16)
    expected = torch.full_like(hidden, 7)
    fallback = MagicMock(return_value=expected)
    monkeypatch.setattr(block.experts, "_forward_ep_level1", fallback)
    mega_moe = MagicMock()
    monkeypatch.setattr(kernels, "mega_moe", mega_moe, raising=False)
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=SimpleNamespace(dp_execution_token_counts=(3, 3)),
        layer_caches=[],
        layer_shared_cache={
            MegaMoeMetadata: _mega_moe_eager_metadata(
                cfg,
                local_token_count=3,
                execution_token_counts=(3, 3),
            ),
        },
    )

    with forward_context(context):
        output = block.experts(hidden)

    assert output is expected
    fallback.assert_called_once_with(hidden)
    mega_moe.assert_not_called()
