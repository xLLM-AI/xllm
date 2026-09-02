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

import sys
import types
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
from xllm.python import distributed, kernels
from xllm.python.kernels_npu.causal_conv1d import (
    causal_conv1d_decode as npu_causal_conv1d_decode,
)
from xllm.python.layers.npu.qwen3_5.decoder_layer import (
    NpuQwen3_5DecoderLayer,
)
from xllm.python.layers.npu.qwen3_5.gated_delta_net import (
    NpuQwen3_5GatedDeltaNet,
)
from xllm.python.layers.npu.qwen3_5.moe import (
    NpuQwen3_5SparseMoEBlock,
)
from xllm.python.layers.qwen3_5_decoder_layer import (
    get_qwen3_5_decoder_layer_class,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader

kernels.gemma_rms_norm = _gemma_rms_norm
kernels.moe_fused_topk = MagicMock()
kernels.grouped_moe_bf16 = MagicMock()
kernels.prepare_row_parallel_weight = MagicMock(side_effect=lambda weight: (weight, False))
distributed.all_gather_variable = MagicMock()
distributed.all_gather = MagicMock()
distributed.all_reduce_ = MagicMock()
distributed.tp_all_reduce = MagicMock()
distributed.moe_tp_all_reduce = MagicMock()
distributed.moe_ep_all_reduce = MagicMock()


def test_npu_decoder_factory_selects_privateuseone_backend() -> None:
    assert get_qwen3_5_decoder_layer_class("privateuseone") is NpuQwen3_5DecoderLayer


def test_npu_gdn_loads_native_conv_weight_layout() -> None:
    cfg = _linear_config()
    layer = NpuQwen3_5GatedDeltaNet(
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

    assert layer.conv1d_weight.shape == (4, conv_dim)
    torch.testing.assert_close(
        layer.conv1d_weight,
        checkpoint_conv.squeeze(1).transpose(0, 1),
    )


def test_npu_decode_uses_npu_recurrent_kernel(monkeypatch) -> None:
    conv = MagicMock(return_value=torch.zeros(1, 24))
    recurrent = MagicMock(return_value=torch.zeros(1, 1, 2, 4))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    monkeypatch.setattr(kernels, "causal_conv1d_decode", conv, raising=False)
    monkeypatch.setattr(
        kernels,
        "fused_sigmoid_gating_delta_rule_decode",
        recurrent,
        raising=False,
    )
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    layer = NpuQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    _install_constant_gdn_projections(layer)

    with forward_context(_gdn_forward_context(is_prefill=False)):
        assert layer(torch.zeros(1, 8)).shape == (1, 8)

    recurrent.assert_called_once()


def test_npu_moe_loads_native_weight_order() -> None:
    cfg = _config(tp_rank=1, moe_tp_rank=1)
    layer = NpuQwen3_5DecoderLayer(
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
    expected_w13 = (
        torch.cat(
            (local_gate, local_up),
            dim=1,
        )
        .transpose(1, 2)
        .contiguous()
    )
    expected_w2 = down.chunk(2, dim=2)[1].transpose(1, 2).contiguous()
    assert isinstance(layer.mlp, NpuQwen3_5SparseMoEBlock)
    torch.testing.assert_close(
        layer.mlp.experts.w13,
        expected_w13.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        layer.mlp.experts.w2,
        expected_w2.to(torch.bfloat16),
    )


def test_npu_gdn_uses_npu_prefill_fusion_boundary(monkeypatch) -> None:
    calls = MagicMock()
    conv_qkv = MagicMock(
        return_value=(
            torch.zeros(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
        )
    )
    gating = MagicMock(
        return_value=(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
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
        ("conv_qkv", conv_qkv),
        ("gating", gating),
        ("chunk", chunk),
        ("rms", rms),
    ):
        calls.attach_mock(mock, name)
    monkeypatch.setattr(
        kernels,
        "causal_conv1d_qkv_prefill",
        conv_qkv,
        raising=False,
    )
    monkeypatch.setattr(kernels, "fused_gdn_gating", gating, raising=False)
    monkeypatch.setattr(kernels, "chunk_gated_delta_rule", chunk, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)

    layer = NpuQwen3_5GatedDeltaNet(
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
        "conv_qkv",
        "gating",
        "chunk",
        "rms",
    ]


def test_npu_decode_passes_native_weight_and_cache_to_tilelang(
    monkeypatch,
) -> None:
    output = torch.zeros(1, 24)
    kernel = MagicMock(return_value=output)
    tilelang_wrapper = types.ModuleType("xllm.python.kernels_npu.tilelang.causal_conv1d_decode")
    tilelang_wrapper.DIM_PER_CORE = 2048
    tilelang_wrapper._build_decode_kernel_jit = MagicMock(return_value=kernel)
    monkeypatch.setitem(
        sys.modules,
        "xllm.python.kernels_npu.tilelang.causal_conv1d_decode",
        tilelang_wrapper,
    )
    value = torch.zeros(1, 24)
    weight = torch.zeros(4, 24)
    conv_state = torch.zeros(2, 3, 24)
    state_indices = torch.tensor([1], dtype=torch.int32)

    actual = npu_causal_conv1d_decode(
        value,
        weight,
        conv_state,
        state_indices,
    )

    assert actual is output
    args = kernel.call_args.args
    assert args[1] is weight
    assert args[2] is conv_state


def test_npu_moe_rejects_non_bf16_weights() -> None:
    with pytest.raises(NotImplementedError, match="support BF16 only"):
        NpuQwen3_5SparseMoEBlock(
            _config(),
            torch.float16,
            torch.device("cpu"),
        )


def test_npu_moe_partitions_expert_axis_for_ep() -> None:
    cfg = _config(
        tp_size=4,
        tp_rank=2,
        world_size=4,
        moe_tp_size=1,
        moe_tp_rank=0,
        ep_size=4,
        ep_rank=2,
        num_attention_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    layer = NpuQwen3_5DecoderLayer(
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
        tp_rank=2,
        tp_size=4,
        moe_tp_rank=0,
        moe_tp_size=1,
        ep_rank=2,
        ep_size=4,
    )

    layer.mlp.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "mlp."),
        context,
    )

    expected_gate_up = gate_up.narrow(0, 4, 2).transpose(1, 2).contiguous()
    expected_down = down.narrow(0, 4, 2).transpose(1, 2).contiguous()
    torch.testing.assert_close(
        layer.mlp.experts.w13,
        expected_gate_up.to(torch.bfloat16),
    )
    torch.testing.assert_close(
        layer.mlp.experts.w2,
        expected_down.to(torch.bfloat16),
    )


def test_npu_moe_uses_moe_tp_collective(monkeypatch) -> None:
    cfg = _config()
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    block.experts.reduce_results = True
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=torch.float32))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(3, cfg.num_experts_per_tok),
                torch.zeros(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    grouped_output = torch.ones(3, cfg.hidden_size)
    grouped_moe = MagicMock(return_value=grouped_output)
    monkeypatch.setattr(kernels, "grouped_moe_bf16", grouped_moe, raising=False)
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()

    output = block.experts(torch.zeros(3, cfg.hidden_size))

    assert output is grouped_output
    distributed.moe_tp_all_reduce.assert_called_once_with(grouped_output)
    distributed.moe_ep_all_reduce.assert_not_called()
    grouped_moe.assert_called_once()


def test_npu_moe_uses_moe_ep_collective(monkeypatch) -> None:
    cfg = _config(
        tp_size=4,
        world_size=4,
        moe_tp_size=1,
        ep_size=4,
        ep_rank=1,
        num_attention_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=torch.float32))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(3, cfg.num_experts_per_tok),
                torch.zeros(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    grouped_output = torch.ones(3, cfg.hidden_size)
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()

    output = block.experts(torch.zeros(3, cfg.hidden_size))

    assert output is grouped_output
    distributed.moe_tp_all_reduce.assert_not_called()
    distributed.moe_ep_all_reduce.assert_called_once_with(grouped_output)


def test_npu_moe_uses_both_collectives_for_partial_ep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config(
        tp_size=4,
        world_size=4,
        moe_tp_size=2,
        ep_size=2,
        ep_rank=1,
        num_attention_heads=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    block.experts.gate = _ConstantModule(torch.zeros(3, cfg.num_experts, dtype=torch.bfloat16))
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.bfloat16,
                ),
                torch.zeros(
                    3,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    grouped_output = torch.ones(
        3,
        cfg.hidden_size,
        dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    distributed.moe_tp_all_reduce.reset_mock()
    distributed.moe_ep_all_reduce.reset_mock()

    output = block.experts(torch.zeros(3, cfg.hidden_size, dtype=torch.bfloat16))

    assert output is grouped_output
    distributed.moe_tp_all_reduce.assert_called_once_with(grouped_output)
    distributed.moe_ep_all_reduce.assert_called_once_with(grouped_output)


def test_npu_moe_graph_dp_gather_uses_fixed_shape(monkeypatch) -> None:
    cfg = _config(
        tp_size=1,
        dp_size=2,
        world_size=2,
        moe_tp_size=2,
        ep_size=1,
    )
    block = NpuQwen3_5SparseMoEBlock(
        cfg,
        torch.bfloat16,
        torch.device("cpu"),
    )
    gathered = torch.zeros(6, cfg.hidden_size)
    grouped_output = torch.arange(
        6 * cfg.hidden_size,
        dtype=torch.float32,
    ).view(6, cfg.hidden_size)
    block.experts.gate = _ConstantModule(torch.zeros(6, cfg.num_experts, dtype=torch.float32))
    distributed.all_gather.reset_mock()
    distributed.all_gather.return_value = gathered
    monkeypatch.setattr(
        kernels,
        "moe_fused_topk",
        MagicMock(
            return_value=(
                torch.ones(6, cfg.num_experts_per_tok),
                torch.zeros(
                    6,
                    cfg.num_experts_per_tok,
                    dtype=torch.int32,
                ),
            )
        ),
    )
    monkeypatch.setattr(
        kernels,
        "grouped_moe_bf16",
        MagicMock(return_value=grouped_output),
        raising=False,
    )
    metadata = SimpleNamespace(
        dp_token_counts=(2, 3),
        dp_is_decode=(1, 1),
        is_prefill=False,
        is_chunked_prefill=False,
    )
    context = ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=[],
        execution_state=AclGraphExecutionState({}),
    )

    with forward_context(context):
        output = block.experts(torch.zeros(2, cfg.hidden_size))

    assert output.shape == (2, cfg.hidden_size)
    torch.testing.assert_close(output, grouped_output[:2])
    gathered_input = distributed.all_gather.call_args.args[0]
    assert gathered_input.shape == (3, cfg.hidden_size)
    distributed.all_gather.assert_called_once_with(
        gathered_input,
        dim=0,
        world_size=2,
        group_name="dp",
    )
