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

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Bind the CUDA kernel package without executing its __init__, which requires
# operators from the compiled runtime. conftest.py supplies the generic kernel
# and distributed stubs used by these CPU/mock tests.
_PYTHON_ROOT = Path(__file__).parents[2] / "xllm" / "python"
_KERNELS_CUDA = types.ModuleType("xllm.python.kernels_cuda")
_KERNELS_CUDA.__path__ = [str(_PYTHON_ROOT / "kernels_cuda")]
sys.modules.setdefault("xllm.python.kernels_cuda", _KERNELS_CUDA)

from tests.python.qwen3_5_test_utils import (  # noqa: E402
    StateDict as _StateDict,
)
from tests.python.qwen3_5_test_utils import (
    gemma_rms_norm as _gemma_rms_norm,
)
from tests.python.qwen3_5_test_utils import (
    install_constant_gdn_projections as _install_constant_gdn_projections,
)
from tests.python.qwen3_5_test_utils import (
    make_config as _config,
)
from tests.python.qwen3_5_test_utils import (
    make_gdn_forward_context as _gdn_forward_context,
)
from tests.python.qwen3_5_test_utils import (
    make_linear_config as _linear_config,
)
from xllm.python import distributed, kernels  # noqa: E402
from xllm.python.kernels_cuda.moe import supports_cutlass_moe  # noqa: E402
from xllm.python.layers.cuda.qwen3_5.attention import (  # noqa: E402
    CudaQwen3_5Attention,
)
from xllm.python.layers.cuda.qwen3_5.decoder_layer import (  # noqa: E402
    CudaQwen3_5DecoderLayer,
)
from xllm.python.layers.cuda.qwen3_5.gated_delta_net import (  # noqa: E402
    CudaQwen3_5GatedDeltaNet,
)
from xllm.python.layers.cuda.qwen3_5.moe import (  # noqa: E402
    CudaQwen3_5SparseMoEBlock,
)
from xllm.python.layers.fused_moe import FusedMoE  # noqa: E402
from xllm.python.layers.gated_mlp import GatedMLP  # noqa: E402
from xllm.python.layers.qwen3_5_decoder_layer import (  # noqa: E402
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.model_executor.forward_context import forward_context  # noqa: E402
from xllm.python.model_loader import (  # noqa: E402
    ParallelLoadContext,
    ScopedWeightLoader,
)
from xllm.python.models import qwen3_5 as qwen3_5_model  # noqa: E402
from xllm.python.models.qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5Model,
)

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
def _use_cuda_decoder_for_cpu_model_tests(monkeypatch):
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


def test_cuda_gdn_loads_native_conv_weight_layout(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    prepare_row_weight = MagicMock(side_effect=lambda weight: (weight, False))
    monkeypatch.setattr(
        kernels,
        "prepare_row_parallel_weight",
        prepare_row_weight,
        raising=False,
    )
    cfg = _linear_config()
    layer = CudaQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
    )
    conv_dim = 24
    checkpoint_conv = torch.arange(
        conv_dim * 4,
        dtype=torch.float32,
    ).view(conv_dim, 1, 4)
    tensors = {
        "linear_attn.in_proj_qkv.weight": torch.zeros(conv_dim, 8),
        "linear_attn.in_proj_z.weight": torch.zeros(8, 8),
        "linear_attn.in_proj_b.weight": torch.zeros(2, 8),
        "linear_attn.in_proj_a.weight": torch.zeros(2, 8),
        "linear_attn.conv1d.weight": checkpoint_conv,
        "linear_attn.A_log": torch.zeros(2),
        "linear_attn.dt_bias": torch.zeros(2),
        "linear_attn.norm.weight": torch.zeros(4),
        "linear_attn.out_proj.weight": torch.zeros(8, 8),
    }

    layer.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "linear_attn."),
        ParallelLoadContext(tp_rank=0, tp_size=1),
    )

    assert layer.conv1d_weight.shape == (conv_dim, 4)
    torch.testing.assert_close(layer.conv1d_weight, checkpoint_conv.squeeze(1))
    prepare_row_weight.assert_called_once()


def test_cuda_decode_uses_cuda_recurrent_kernel(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    conv = MagicMock(return_value=torch.zeros(1, 24))
    recurrent = MagicMock(return_value=torch.zeros(1, 1, 2, 4))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    monkeypatch.setattr(kernels, "causal_conv1d_decode", conv, raising=False)
    monkeypatch.setattr(
        kernels,
        "fused_recurrent_gated_delta_rule_packed_decode",
        recurrent,
        raising=False,
    )
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    layer = CudaQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    _install_constant_gdn_projections(layer)

    with forward_context(_gdn_forward_context(is_prefill=False)):
        assert layer(torch.zeros(1, 8)).shape == (1, 8)

    recurrent.assert_called_once()


def test_cuda_moe_loads_native_weight_order() -> None:
    cfg = _config(tp_rank=1, moe_tp_rank=1)
    layer = CudaQwen3_5DecoderLayer(
        cfg,
        0,
        torch.bfloat16,
        torch.device("cpu"),
        MagicMock(),
    )
    gate_up = torch.arange(
        cfg.num_experts * 2 * cfg.moe_intermediate_size * cfg.hidden_size,
        dtype=torch.float32,
    ).view(cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    down = torch.arange(
        cfg.num_experts * cfg.hidden_size * cfg.moe_intermediate_size,
        dtype=torch.float32,
    ).view(cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    tensors = {
        "mlp.gate.weight": torch.zeros(cfg.num_experts, cfg.hidden_size),
        "mlp.shared_expert_gate.weight": torch.zeros(1, cfg.hidden_size),
        "mlp.experts.gate_up_proj": gate_up,
        "mlp.experts.down_proj": down,
        "mlp.shared_expert.gate_proj.weight": torch.zeros(
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mlp.shared_expert.up_proj.weight": torch.zeros(
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
        ),
        "mlp.shared_expert.down_proj.weight": torch.zeros(
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
        ),
    }
    context = ParallelLoadContext(
        tp_rank=1,
        tp_size=2,
        moe_tp_rank=1,
        moe_tp_size=2,
    )

    layer.mlp.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "mlp."),
        context,
    )

    gate, up = gate_up.chunk(2, dim=1)
    local_gate = gate.chunk(2, dim=1)[1]
    local_up = up.chunk(2, dim=1)[1]
    expected_w13 = torch.cat((local_up, local_gate), dim=1)
    assert isinstance(layer.mlp, CudaQwen3_5SparseMoEBlock)
    torch.testing.assert_close(
        layer.mlp.experts.w13,
        expected_w13.to(torch.bfloat16),
    )


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


def test_gdn_load_slices_asymmetric_key_and_value_channels(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    cfg = _config(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        vocab_size=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=4,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
    )
    layer = CudaQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
    )
    global_key = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    global_value = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    qkv = torch.arange(
        (2 * global_key + global_value) * cfg.hidden_size,
        dtype=torch.float32,
    ).view(2 * global_key + global_value, cfg.hidden_size)
    conv = torch.arange(
        (2 * global_key + global_value) * cfg.linear_conv_kernel_dim,
        dtype=torch.float32,
    ).view(2 * global_key + global_value, 1, cfg.linear_conv_kernel_dim)
    tensors = {
        "in_proj_qkv.weight": qkv,
        "in_proj_z.weight": torch.zeros(global_value, cfg.hidden_size),
        "in_proj_b.weight": torch.zeros(
            cfg.linear_num_value_heads,
            cfg.hidden_size,
        ),
        "in_proj_a.weight": torch.zeros(
            cfg.linear_num_value_heads,
            cfg.hidden_size,
        ),
        "conv1d.weight": conv,
        "A_log": torch.zeros(cfg.linear_num_value_heads),
        "dt_bias": torch.zeros(cfg.linear_num_value_heads),
        "norm.weight": torch.zeros(cfg.linear_value_head_dim),
        "out_proj.weight": torch.zeros(cfg.hidden_size, global_value),
    }
    context = ParallelLoadContext(tp_rank=1, tp_size=2)

    layer.load_weights(ScopedWeightLoader([_StateDict(tensors)]), context)

    q, k, v = qkv.split((global_key, global_key, global_value), dim=0)
    expected_qkv = torch.cat(
        (
            q.chunk(2, dim=0)[1],
            k.chunk(2, dim=0)[1],
            v.chunk(2, dim=0)[1],
        )
    )
    cq, ck, cv = conv.squeeze(1).split(
        (global_key, global_key, global_value),
        dim=0,
    )
    expected_conv = torch.cat(
        (
            cq.chunk(2, dim=0)[1],
            ck.chunk(2, dim=0)[1],
            cv.chunk(2, dim=0)[1],
        )
    )
    torch.testing.assert_close(layer.in_proj_qkv.weight, expected_qkv)
    torch.testing.assert_close(layer.conv1d_weight, expected_conv)
    assert layer.conv_dim == 16


def test_cuda_gdn_keeps_historical_prefill_kernel_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        kernels,
        "resolve_gdn_prefill_backend",
        lambda: "triton",
        raising=False,
    )
    calls = MagicMock()
    conv = MagicMock(return_value=torch.zeros(1, 24))
    post_conv = MagicMock(
        return_value=(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )
    )
    chunk = MagicMock(
        return_value=(
            torch.zeros(1, 2, 4),
            torch.zeros(1, 2, 4, 4),
        )
    )
    rms = MagicMock(side_effect=lambda output, *_args: output)
    for name, mock in (
        ("conv", conv),
        ("post_conv", post_conv),
        ("chunk", chunk),
        ("rms", rms),
    ):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(kernels, "causal_conv1d_prefill", conv, raising=False)
    monkeypatch.setattr(
        kernels,
        "fused_gdn_prefill_post_conv",
        post_conv,
        raising=False,
    )
    monkeypatch.setattr(kernels, "chunk_gated_delta_rule", chunk, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    layer = CudaQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    _install_constant_gdn_projections(layer)
    with forward_context(_gdn_forward_context(is_prefill=True)):
        output = layer(torch.zeros(1, 8))

    assert output.shape == (1, 8)
    assert [call[0] for call in calls.mock_calls] == [
        "conv",
        "post_conv",
        "chunk",
        "rms",
    ]


def test_backend_split_preserves_public_parameter_paths(monkeypatch) -> None:
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


def test_pre_sm90_moe_uses_triton_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (8, 0))
    assert not kernels.supports_cutlass_moe(torch.device("cuda:0"))
    cfg = _config(
        tp_size=1,
        world_size=1,
        moe_tp_size=1,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
    )
    layer = _make_moe_layer(cfg)
    assert not layer._use_cutlass
    layer.gate = torch.nn.Identity()
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(3, 1, dtype=torch.float32),
                torch.zeros(3, 1, dtype=torch.int32),
            )
        ),
    )
    triton_output = torch.ones(3, cfg.hidden_size)
    triton_moe = MagicMock(return_value=triton_output)
    monkeypatch.setattr(kernels, "fused_moe", triton_moe)

    output = layer(torch.zeros(3, cfg.hidden_size))

    assert output is triton_output
    triton_moe.assert_called_once()


def test_attention_tp_and_moe_tp_use_distinct_valid_topology():
    cfg = _config()
    cfg.validate()

    layer = _make_moe_layer(cfg)

    assert cfg.tp_size == 2
    assert cfg.dp_size == 1
    assert cfg.moe_tp_size == 2
    assert cfg.ep_size == 1
    assert layer.w13.shape == (8, 32, 64)
    assert layer.w2.shape == (8, 64, 16)


def test_attention_dp_and_moe_tp_use_distinct_valid_topology():
    cfg = _config(tp_size=1, dp_size=2)
    cfg.validate()

    layer = _make_moe_layer(cfg)

    assert cfg.tp_size == 1
    assert cfg.dp_size == 2
    assert cfg.moe_tp_size == 2
    assert cfg.ep_size == 1
    assert layer.w13.shape == (8, 32, 64)
    assert layer.w2.shape == (8, 64, 16)


def test_full_world_ep_partitions_experts_without_inner_tp_sharding():
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
