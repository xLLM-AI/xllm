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
from xllm.python.distributed import collectives
from xllm.python.kernels_npu.causal_conv1d import causal_conv1d_decode as npu_causal_conv1d_decode
from xllm.python.layers.npu.qwen3_5.decoder_layer import NpuQwen3_5DecoderLayer
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.moe import NpuQwen3_5SparseMoEBlock
from xllm.python.layers.qwen3_5.decoder_layer import get_qwen3_5_decoder_layer_class
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
distributed.gather_dp_execution_tokens = MagicMock()
distributed.all_reduce_ = MagicMock()
distributed.tp_all_reduce = MagicMock()
distributed.moe_tp_all_reduce = MagicMock()
distributed.moe_ep_all_reduce = MagicMock()


def test_npu_decoder_factory_selects_privateuseone_backend() -> None:
    assert get_qwen3_5_decoder_layer_class("privateuseone") is NpuQwen3_5DecoderLayer


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


def test_npu_decode_uses_npu_recurrent_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = NpuQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    projections = _install_gdn_projections(layer)
    context = _gdn_forward_context(is_prefill=False)
    conv_output = projections["qkv"] + 600
    expected = torch.arange(1, 17, dtype=torch.float32).view(2, 8)
    conv = MagicMock(return_value=conv_output)
    recurrent = MagicMock(return_value=expected.view(1, 2, 2, 4))
    monkeypatch.setattr(kernels, "causal_conv1d_decode", conv, raising=False)
    monkeypatch.setattr(kernels, "fused_sigmoid_gating_delta_rule_decode", recurrent, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", lambda output, *_args: output, raising=False)
    with forward_context(context):
        torch.testing.assert_close(layer(torch.ones(2, 8)), expected)
    conv.assert_called_once()
    assert len(conv.call_args.args) == 5
    assert conv.call_args.args[1] is layer.conv1d_weight
    assert conv.call_args.args[4] is layer.conv1d_bias
    torch.testing.assert_close(conv.call_args.args[0], projections["qkv"])
    recurrent.assert_called_once()
    expected_args = (conv_output, projections["a"], projections["b"])
    for actual, reference in zip(recurrent.call_args.args[:3], expected_args):
        torch.testing.assert_close(actual, reference)
    assert recurrent.call_args.args[5] is context.layer_caches[0].ssm
    torch.testing.assert_close(recurrent.call_args.args[6], torch.tensor([2, 1], dtype=torch.int32))


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


def test_npu_gdn_uses_npu_prefill_fusion_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = NpuQwen3_5GatedDeltaNet(_linear_config(), 0, torch.float32, torch.device("cpu"))
    values = _gdn_prefill_values()
    conv_qkv = MagicMock(return_value=tuple(value.unsqueeze(0) for value in values[:3]))
    gating = MagicMock(return_value=tuple(value.unsqueeze(0) for value in values[3:]))
    monkeypatch.setattr(kernels, "causal_conv1d_qkv_prefill", conv_qkv, raising=False)
    monkeypatch.setattr(kernels, "fused_gdn_gating", gating, raising=False)
    projections = _install_gdn_projections(layer)
    context = _gdn_forward_context(is_prefill=True)
    expected_cache = context.layer_caches[0].ssm.clone()
    initial = torch.stack((expected_cache[2], torch.zeros_like(expected_cache[0])))
    final = torch.arange(64, dtype=torch.float32).view(2, 2, 4, 4) + 3000
    expected_cache[2].copy_(final[0])
    chunk = MagicMock(return_value=(values[2], final))
    rms = MagicMock(side_effect=lambda output, *_args: output)
    trace = MagicMock()
    for name, mock in (("conv_qkv", conv_qkv), ("gating", gating), ("chunk", chunk), ("rms", rms)):
        trace.attach_mock(mock, name)
    monkeypatch.setattr(kernels, "chunk_gated_delta_rule", chunk, raising=False)
    monkeypatch.setattr(kernels, "rms_norm_gated", rms, raising=False)
    expected_output = values[2].clone()
    expected_output[1].zero_()
    with forward_context(context):
        torch.testing.assert_close(layer(torch.ones(2, 8)), expected_output.view(2, 8))
    assert [call[0] for call in trace.mock_calls] == ["conv_qkv", "gating", "chunk", "rms"]
    chunk.assert_called_once()
    for actual, reference in zip(chunk.call_args.args[:6], (*values, initial)):
        torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(context.layer_caches[0].ssm, expected_cache)
    conv_qkv.assert_called_once()
    torch.testing.assert_close(conv_qkv.call_args.args[0], projections["qkv"])
    gating.assert_called_once()
    torch.testing.assert_close(gating.call_args.args[1], projections["a"])
    torch.testing.assert_close(gating.call_args.args[2], projections["b"])


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
