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

"""CUDA-specific parallel and composition tests for Python Qwen3.5."""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

if importlib.util.find_spec("xllm.python.kernels_cuda") is None:
    pytest.skip("CUDA kernels are unavailable in this build", allow_module_level=True)

from tests.python.qwen3_5_test_utils import (
    StateDict as _StateDict,
)
from tests.python.qwen3_5_test_utils import (
    gemma_rms_norm as _gemma_rms_norm,
)
from tests.python.qwen3_5_test_utils import (
    install_constant_gdn_projections as _install_gdn_projections,
)
from tests.python.qwen3_5_test_utils import (
    make_config as _config,
)
from tests.python.qwen3_5_test_utils import (
    make_gdn_forward_context as _gdn_forward_context,
)
from tests.python.qwen3_5_test_utils import (
    make_gdn_prefill_values as _gdn_prefill_values,
)
from tests.python.qwen3_5_test_utils import (
    make_linear_config as _linear_config,
)
from tests.python.qwen3_5_test_utils import (
    make_moe_checkpoint as _moe_checkpoint,
)
from xllm.python import distributed, kernels
from xllm.python.kernels_cuda.moe import supports_cutlass_moe
from xllm.python.layers.cuda.qwen3_5.attention import CudaQwen3_5Attention
from xllm.python.layers.cuda.qwen3_5.decoder_layer import CudaQwen3_5DecoderLayer
from xllm.python.layers.cuda.qwen3_5.gated_delta_net import CudaQwen3_5GatedDeltaNet
from xllm.python.layers.cuda.qwen3_5.moe import CudaQwen3_5SparseMoEBlock
from xllm.python.layers.fused_moe import FusedMoE
from xllm.python.layers.gated_mlp import GatedMLP
from xllm.python.layers.qwen3_5.decoder_layer import get_qwen3_5_decoder_layer_class
from xllm.python.model_executor.forward_context import forward_context
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader
from xllm.python.models import qwen3_5 as qwen3_5_model
from xllm.python.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5Model

kernels.supports_cutlass_moe = supports_cutlass_moe
kernels.gemma_rms_norm = _gemma_rms_norm
kernels.moe_fused_topk = MagicMock()
kernels.cutlass_fused_moe = MagicMock()
kernels.fused_moe = MagicMock()
kernels.prepare_row_parallel_weight = MagicMock(side_effect=lambda weight: (weight, False))
distributed.all_gather_variable = MagicMock()
distributed.all_gather = MagicMock()
distributed.all_reduce_ = MagicMock()
distributed.tp_all_reduce = MagicMock()
distributed.moe_tp_all_reduce = MagicMock()
distributed.moe_ep_all_reduce = MagicMock()


@pytest.fixture(autouse=True)
def _use_cuda_decoder_for_cpu_model_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qwen3_5_model,
        "get_qwen3_5_decoder_layer_class",
        lambda _device: CudaQwen3_5DecoderLayer,
    )


def _make_moe_layer(cfg: Qwen3_5Config, device: torch.device | None = None) -> FusedMoE:
    return FusedMoE(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.moe_intermediate_size,
        num_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        renormalize=True,
        moe_tp_size=cfg.moe_tp_size,
        moe_tp_rank=cfg.moe_tp_rank,
        ep_size=cfg.ep_size,
        ep_rank=cfg.ep_rank,
        dp_size=cfg.dp_size,
        dp_rank=cfg.dp_rank,
        dtype=torch.float32,
        device=device or torch.device("cpu"),
    )


def test_cuda_decoder_factory_selects_cuda_backend() -> None:
    assert get_qwen3_5_decoder_layer_class("cuda") is CudaQwen3_5DecoderLayer


def test_cuda_decode_uses_cuda_recurrent_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "resolve_gdn_prefill_backend", lambda: "triton", raising=False)
    layer = CudaQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    projections = _install_gdn_projections(layer)
    context = _gdn_forward_context(is_prefill=False)
    conv_output = projections["qkv"] + 600
    expected = torch.arange(1, 17, dtype=torch.float32).view(2, 8)
    conv = MagicMock(return_value=conv_output)
    recurrent = MagicMock(return_value=expected.view(1, 2, 2, 4))
    monkeypatch.setattr(kernels, "causal_conv1d_decode", conv, raising=False)
    monkeypatch.setattr(kernels, "fused_recurrent_gated_delta_rule_packed_decode", recurrent, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", lambda output, *_args: output, raising=False)
    with forward_context(context):
        torch.testing.assert_close(layer(torch.ones(2, 8)), expected)
    conv.assert_called_once()
    assert len(conv.call_args.args) == 4
    assert conv.call_args.args[1] is layer.conv1d_weight
    torch.testing.assert_close(conv.call_args.args[0], projections["qkv"])
    recurrent.assert_called_once()
    expected_args = (conv_output, projections["a"], projections["b"])
    for actual, reference in zip(recurrent.call_args.args[:3], expected_args):
        torch.testing.assert_close(actual, reference)
    assert recurrent.call_args.args[5] is context.layer_caches[0].ssm
    torch.testing.assert_close(recurrent.call_args.args[6], torch.tensor([2, 1], dtype=torch.int32))


def test_cuda_moe_loads_native_weight_order() -> None:
    cfg = _config(tp_rank=0, moe_tp_rank=1)
    layer = CudaQwen3_5DecoderLayer(cfg, 0, torch.bfloat16, torch.device("cpu"), MagicMock())
    tensors = _moe_checkpoint(cfg)
    context = ParallelLoadContext(tp_rank=0, tp_size=2, moe_tp_rank=1, moe_tp_size=2)
    layer.mlp.load_weights(ScopedWeightLoader([_StateDict(tensors)]), context)

    gate, up = tensors["experts.gate_up_proj"].chunk(2, dim=1)
    expected_w13 = torch.cat((up.chunk(2, dim=1)[1], gate.chunk(2, dim=1)[1]), dim=1)
    expected_w2 = tensors["experts.down_proj"].chunk(2, dim=2)[1]
    assert isinstance(layer.mlp, CudaQwen3_5SparseMoEBlock)
    torch.testing.assert_close(layer.mlp.experts.w13, expected_w13.to(torch.bfloat16))
    torch.testing.assert_close(layer.mlp.experts.w2, expected_w2.to(torch.bfloat16))


def test_moe_layer_schedule_matches_config() -> None:
    cfg = _config(
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        decoder_sparse_step=2,
        mlp_only_layers=[3],
    )

    assert not cfg.is_moe_layer(0)
    assert cfg.is_moe_layer(1)
    assert not cfg.is_moe_layer(2)
    assert not cfg.is_moe_layer(3)

    model = Qwen3_5Model(cfg, torch.float32, torch.device("cpu"))
    assert isinstance(model.layers[0].mlp, GatedMLP)
    assert isinstance(model.layers[1].mlp, CudaQwen3_5SparseMoEBlock)
    assert isinstance(model.layers[2].mlp, GatedMLP)
    assert isinstance(model.layers[3].mlp, GatedMLP)


@pytest.mark.parametrize(("tp_size", "tp_rank"), ((1, 0), (2, 1)))
def test_gdn_load_slices_asymmetric_key_and_value_channels(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    tp_rank: int,
) -> None:
    monkeypatch.setattr(kernels, "resolve_gdn_prefill_backend", lambda: "triton", raising=False)
    prepare = MagicMock(side_effect=lambda weight: (weight, False))
    monkeypatch.setattr(kernels, "prepare_row_parallel_weight", prepare)
    cfg = _linear_config(
        tp_size=tp_size,
        tp_rank=tp_rank,
        world_size=tp_size,
        moe_tp_size=tp_size,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
    )
    layer = CudaQwen3_5GatedDeltaNet(cfg, 0, torch.float32, torch.device("cpu"))
    qkv = torch.arange(1, 257, dtype=torch.float32).view(32, 8)
    conv = torch.arange(1, 129, dtype=torch.float32).view(32, 1, 4)
    out_proj = torch.arange(1, 129, dtype=torch.float32).view(8, 16)
    tensors = {
        "in_proj_qkv.weight": qkv,
        "in_proj_z.weight": torch.zeros(16, 8),
        "in_proj_b.weight": torch.zeros(4, 8),
        "in_proj_a.weight": torch.zeros(4, 8),
        "conv1d.weight": conv,
        "A_log": torch.zeros(4),
        "dt_bias": torch.zeros(4),
        "norm.weight": torch.zeros(4),
        "out_proj.weight": out_proj,
    }
    layer.load_weights(
        ScopedWeightLoader([_StateDict(tensors)]),
        ParallelLoadContext(tp_rank=tp_rank, tp_size=tp_size),
    )

    checkpoints = ((layer.in_proj_qkv.weight, qkv), (layer.conv1d_weight, conv.squeeze(1)))
    for actual, checkpoint in checkpoints:
        parts = checkpoint.split((8, 8, 16), dim=0)
        expected = torch.cat([part.chunk(tp_size, dim=0)[tp_rank] for part in parts])
        torch.testing.assert_close(actual, expected)
    assert layer.conv1d_weight.shape == (32 // tp_size, 4)
    prepare.assert_called_once()
    torch.testing.assert_close(prepare.call_args.args[0], out_proj.chunk(tp_size, dim=1)[tp_rank])


def test_cuda_gdn_keeps_historical_prefill_kernel_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernels, "resolve_gdn_prefill_backend", lambda: "triton", raising=False)
    layer = CudaQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    values = _gdn_prefill_values()
    conv_output = torch.arange(48, dtype=torch.float32).view(2, 24) + 600
    conv = MagicMock(return_value=conv_output)
    post_conv = MagicMock(return_value=values)
    monkeypatch.setattr(kernels, "causal_conv1d_prefill", conv, raising=False)
    monkeypatch.setattr(kernels, "fused_gdn_prefill_post_conv", post_conv, raising=False)
    projections = _install_gdn_projections(layer)
    context = _gdn_forward_context(is_prefill=True)
    expected_cache = context.layer_caches[0].ssm.clone()
    initial = torch.stack((expected_cache[2], torch.zeros_like(expected_cache[0])))
    final = torch.arange(64, dtype=torch.float32).view(2, 2, 4, 4) + 3000
    expected_cache[2].copy_(final[0])
    chunk = MagicMock(return_value=(values[2], final))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    trace = MagicMock()
    for name, mock in (("conv", conv), ("post_conv", post_conv), ("chunk", chunk), ("rms", rms)):
        trace.attach_mock(mock, name)
    monkeypatch.setattr(kernels, "chunk_gated_delta_rule", chunk, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)
    expected_output = values[2].clone()
    expected_output[1].zero_()
    with forward_context(context):
        torch.testing.assert_close(layer(torch.ones(2, 8)), expected_output.view(2, 8))
    assert [call[0] for call in trace.mock_calls] == ["conv", "post_conv", "chunk", "rms"]
    chunk.assert_called_once()
    for actual, reference in zip(chunk.call_args.args[:6], (*values, initial)):
        torch.testing.assert_close(actual, reference)
    assert chunk.call_args.args[-1] == "triton"
    torch.testing.assert_close(context.layer_caches[0].ssm, expected_cache)
    conv.assert_called_once()
    torch.testing.assert_close(conv.call_args.args[0], projections["qkv"])
    post_conv.assert_called_once()
    expected_args = (("mixed_qkv", conv_output), ("a", projections["a"]), ("b", projections["b"]))
    for name, reference in expected_args:
        torch.testing.assert_close(post_conv.call_args.kwargs[name], reference)


def test_backend_split_preserves_public_parameter_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 16,
        "intermediate_size": 16,
        "layer_types": ["full_attention", "linear_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "vocab_size": 8,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "tp_size": 1,
        "world_size": 1,
        "moe_tp_size": 1,
        "dtype": "float32",
        "device": "cpu",
    }

    parameter_names = set(dict(Qwen3_5ForCausalLM(config).named_parameters()))

    assert "model.layers.0.self_attn.qkv_proj.weight" in parameter_names
    assert "model.layers.0.mlp.gate_up_proj.weight" in parameter_names
    assert "model.layers.1.linear_attn.conv1d_weight" in parameter_names
    assert "model.layers.1.mlp.gate_up_proj.weight" in parameter_names


def test_attention_biases_are_loaded() -> None:
    cfg_values = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 16,
        "intermediate_size": 16,
        "layer_types": ["full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "vocab_size": 8,
        "attention_bias": True,
        "attn_output_gate": False,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "tp_size": 1,
        "world_size": 1,
        "moe_tp_size": 1,
        "dtype": "float32",
        "device": "cpu",
    }
    model = Qwen3_5ForCausalLM(cfg_values)
    attention_prepare = MagicMock()
    model.model.layers[0].self_attn.o_proj.process_weights_after_loading = attention_prepare
    tensors = {
        "model.embed_tokens.weight": torch.zeros(8, 8),
        "model.layers.0.input_layernorm.weight": torch.zeros(8),
        "model.layers.0.post_attention_layernorm.weight": torch.zeros(8),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros(4, 8),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros(4, 8),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros(8, 8),
        "model.layers.0.self_attn.q_proj.bias": torch.arange(8.0),
        "model.layers.0.self_attn.k_proj.bias": torch.arange(4.0) + 10,
        "model.layers.0.self_attn.v_proj.bias": torch.arange(4.0) + 20,
        "model.layers.0.self_attn.o_proj.bias": torch.arange(8.0) + 30,
        "model.layers.0.self_attn.q_norm.weight": torch.zeros(4),
        "model.layers.0.self_attn.k_norm.weight": torch.zeros(4),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros(16, 8),
        "model.layers.0.mlp.up_proj.weight": torch.zeros(16, 8),
        "model.layers.0.mlp.down_proj.weight": torch.zeros(8, 16),
        "model.norm.weight": torch.zeros(8),
        "lm_head.weight": torch.zeros(8, 8),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=0, tp_size=1)

    expected_qkv_bias = torch.cat(
        (
            tensors["model.layers.0.self_attn.q_proj.bias"],
            tensors["model.layers.0.self_attn.k_proj.bias"],
            tensors["model.layers.0.self_attn.v_proj.bias"],
        )
    )
    torch.testing.assert_close(model.model.layers[0].self_attn.qkv_proj.bias, expected_qkv_bias)
    torch.testing.assert_close(
        model.model.layers[0].self_attn.o_proj.bias,
        tensors["model.layers.0.self_attn.o_proj.bias"],
    )
    attention_prepare.assert_called_once_with()


def test_model_load_passes_the_complete_parallel_context() -> None:
    config = {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "rms_norm_eps": 1e-6,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 16,
        "vocab_size": 8,
        "layer_types": ["full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "moe_intermediate_size": 0,
        "shared_expert_intermediate_size": 0,
        "tp_size": 2,
        "tp_rank": 1,
        "dp_size": 2,
        "dp_rank": 1,
        "world_size": 4,
        "moe_tp_size": 2,
        "moe_tp_rank": 0,
        "ep_size": 2,
        "ep_rank": 1,
        "device": "cpu",
    }
    model = Qwen3_5ForCausalLM(config)
    layer_load = MagicMock()
    model.model.layers[0].load_weights = layer_load
    tensors = {
        "model.embed_tokens.weight": torch.zeros(8, 8),
        "model.norm.weight": torch.zeros(8),
        "lm_head.weight": torch.zeros(8, 8),
    }

    model.load_weights([_StateDict(tensors)], tp_rank=1, tp_size=2)

    context = layer_load.call_args.args[1]
    assert context == ParallelLoadContext(
        tp_rank=1,
        tp_size=2,
        dp_rank=1,
        dp_size=2,
        moe_tp_rank=0,
        moe_tp_size=2,
        ep_rank=1,
        ep_size=2,
    )


def test_full_attention_layers_share_rotary_table() -> None:
    cfg = _config(
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention"],
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        max_position_embeddings=16,
    )

    model = Qwen3_5Model(cfg, torch.float32, torch.device("cpu"))

    assert model.layers[0].self_attn.rotary is model.rotary
    assert model.layers[1].self_attn.rotary is model.rotary


@pytest.mark.parametrize("capability", ((8, 0), (9, 0)))
def test_moe_capability_selects_cuda_backend(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
) -> None:
    device = torch.device("cuda:1")
    query = MagicMock(return_value=capability)
    check = MagicMock(wraps=supports_cutlass_moe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", query)
    monkeypatch.setattr(kernels, "supports_cutlass_moe", check)
    cfg = _config(tp_size=1, world_size=1, moe_tp_size=1, num_experts=2, num_experts_per_tok=1)
    real_empty = torch.empty

    def _allocate_on_cpu(*size: Any, **kwargs: Any) -> torch.Tensor:
        assert kwargs["device"] == device
        return real_empty(*size, **dict(kwargs, device="cpu"))

    # Keep the constructor real; only tensor allocations use CPU storage.
    with monkeypatch.context() as allocations:
        allocations.setattr(torch, "empty", _allocate_on_cpu)
        layer = _make_moe_layer(cfg, device)
    check.assert_called_once_with(device)
    query.assert_called_once_with(1)
    assert layer._use_cutlass is (capability[0] == 9)
    layer.gate = torch.nn.Identity()
    topk = (torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int32))
    monkeypatch.setattr(kernels, "moe_fused_topk", MagicMock(return_value=topk))
    triton = MagicMock(return_value=torch.full((3, cfg.hidden_size), 17.0))
    cutlass = MagicMock(return_value=torch.full((3, cfg.hidden_size), 29.0))
    monkeypatch.setattr(kernels, "fused_moe", triton)
    monkeypatch.setattr(kernels, "cutlass_fused_moe", cutlass)
    selected, unused = (triton, cutlass) if capability[0] == 8 else (cutlass, triton)
    assert layer(torch.zeros(3, cfg.hidden_size)) is selected.return_value
    selected.assert_called_once()
    unused.assert_not_called()


@pytest.mark.parametrize(("tp_size", "dp_size"), ((2, 1), (1, 2)))
def test_attention_and_moe_parallel_topologies(tp_size: int, dp_size: int) -> None:
    cfg = _config(tp_size=tp_size, dp_size=dp_size)
    cfg.validate()
    layer = _make_moe_layer(cfg)
    assert layer.w13.shape == (8, 32, 64)
    assert layer.w2.shape == (8, 64, 16)


def test_full_world_ep_partitions_experts_without_inner_tp_sharding() -> None:
    cfg = _config(
        world_size=4,
        ep_size=4,
        ep_rank=3,
        dp_size=2,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    cfg.validate()

    layer = _make_moe_layer(cfg)

    assert layer.w13.shape == (2, 64, 64)
    assert layer.w2.shape == (2, 64, 32)


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "expected_kv_rank"),
    ((4, 2, 1), (8, 7, 1)),
)
def test_full_attention_load_replicates_kv_heads_by_rank(
    tp_size: int,
    tp_rank: int,
    expected_kv_rank: int,
) -> None:
    cfg = _config(
        tp_size=tp_size,
        tp_rank=tp_rank,
        world_size=tp_size,
        moe_tp_size=tp_size,
        num_attention_heads=8,
        num_key_value_heads=2,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        attn_output_gate=True,
    )
    layer = CudaQwen3_5Attention(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
        MagicMock(),
    )
    q = torch.arange(
        2 * cfg.n_heads * cfg.head_dim * cfg.hidden_size,
        dtype=torch.float32,
    ).view(2 * cfg.n_heads * cfg.head_dim, cfg.hidden_size)
    k = torch.arange(
        cfg.n_kv_heads * cfg.head_dim * cfg.hidden_size,
        dtype=torch.float32,
    ).view(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_size)
    v = k + 100000
    tensors = {
        "q_proj.weight": q,
        "k_proj.weight": k,
        "v_proj.weight": v,
        "o_proj.weight": torch.zeros(
            cfg.hidden_size,
            cfg.n_heads * cfg.head_dim,
        ),
        "q_norm.weight": torch.zeros(cfg.head_dim),
        "k_norm.weight": torch.zeros(cfg.head_dim),
    }
    context = ParallelLoadContext(tp_rank=tp_rank, tp_size=tp_size)

    layer.load_weights(ScopedWeightLoader([_StateDict(tensors)]), context)

    expected = torch.cat(
        (
            q.chunk(tp_size, dim=0)[tp_rank],
            k.chunk(2, dim=0)[expected_kv_rank],
            v.chunk(2, dim=0)[expected_kv_rank],
        )
    )
    torch.testing.assert_close(layer.qkv_proj.weight, expected)
