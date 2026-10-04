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
    make_config as _config,
)
from tests.python.qwen3_5_test_utils import (
    make_linear_config as _linear_config,
)
from tests.python.qwen3_5_test_utils import (
    make_moe_checkpoint as _moe_checkpoint,
)
from xllm.python import distributed, kernels
from xllm.python.attention.backend import LayerCache
from xllm.python.distributed import collectives
from xllm.python.kernels_npu.causal_conv1d import (
    causal_conv1d_decode as npu_causal_conv1d_decode,
)
from xllm.python.layers.npu.qwen3_5.decoder_layer import (
    NpuQwen3_5DecoderLayer,
)
from xllm.python.layers.npu.qwen3_5.gated_delta_net import (
    NpuQwen3_5GatedDeltaNet,
)
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnMetadata, GdnPrefillMetadata
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
from xllm.python.model_executor.runners.eager import EagerRunner
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


def test_npu_gdn_packs_rank_local_projection_weights() -> None:
    cfg = _linear_config(
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        tp_size=2,
        tp_rank=1,
        world_size=2,
        moe_tp_size=2,
    )
    layer = NpuQwen3_5GatedDeltaNet(
        cfg,
        0,
        torch.float32,
        torch.device("cpu"),
    )
    global_key_dim = 16
    global_value_dim = 16
    global_conv_dim = 2 * global_key_dim + global_value_dim
    qkv = torch.arange(global_conv_dim * 8, dtype=torch.float32).view(
        global_conv_dim,
        8,
    )
    z = torch.arange(global_value_dim * 8, dtype=torch.float32).view(
        global_value_dim,
        8,
    )
    b = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
    a = b + 1000
    tensors = {
        "linear_attn.in_proj_qkv.weight": qkv,
        "linear_attn.in_proj_z.weight": z,
        "linear_attn.in_proj_b.weight": b,
        "linear_attn.in_proj_a.weight": a,
        "linear_attn.conv1d.weight": torch.zeros(global_conv_dim, 1, 4),
        "linear_attn.A_log": torch.zeros(4),
        "linear_attn.dt_bias": torch.zeros(4),
        "linear_attn.norm.weight": torch.zeros(4),
        "linear_attn.out_proj.weight": torch.zeros(8, global_value_dim),
    }

    layer.load_weights(
        ScopedWeightLoader([_StateDict(tensors)], "linear_attn."),
        ParallelLoadContext(tp_rank=1, tp_size=2),
    )

    q, k, v = qkv.split((global_key_dim, global_key_dim, global_value_dim))
    expected_qkv = torch.cat((q.chunk(2)[1], k.chunk(2)[1], v.chunk(2)[1]))
    qkv_weight, z_weight, b_weight, a_weight = layer._input_projection_weight_views()
    torch.testing.assert_close(qkv_weight, expected_qkv)
    torch.testing.assert_close(z_weight, z.chunk(2)[1])
    torch.testing.assert_close(b_weight, b.chunk(2)[1])
    torch.testing.assert_close(a_weight, a.chunk(2)[1])
    assert qkv_weight.is_contiguous()
    assert z_weight.is_contiguous()
    assert qkv_weight.untyped_storage().data_ptr() == z_weight.untyped_storage().data_ptr()
    assert b_weight.untyped_storage().data_ptr() == a_weight.untyped_storage().data_ptr()


def test_npu_gdn_prefill_uses_packed_weight_views(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = NpuQwen3_5GatedDeltaNet(
        _linear_config(),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    with torch.no_grad():
        layer.in_proj_qkvz.weight.copy_(
            torch.arange(layer.in_proj_qkvz.weight.numel(), dtype=torch.float32).view_as(layer.in_proj_qkvz.weight)
        )
        layer.in_proj_ba.weight.copy_(
            torch.arange(layer.in_proj_ba.weight.numel(), dtype=torch.float32).view_as(layer.in_proj_ba.weight)
        )
    qkvz_forward = MagicMock(wraps=layer.in_proj_qkvz.forward)
    ba_forward = MagicMock(wraps=layer.in_proj_ba.forward)
    monkeypatch.setattr(layer.in_proj_qkvz, "forward", qkvz_forward)
    monkeypatch.setattr(layer.in_proj_ba, "forward", ba_forward)

    prefill_outputs = layer._project_prefill_inputs(torch.ones(2, 8))
    qkvz_forward.assert_not_called()
    ba_forward.assert_not_called()
    assert [tuple(output.shape) for output in prefill_outputs] == [
        (2, 24),
        (2, 2),
        (2, 2),
        (2, 2, 4),
    ]


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
    context.layer_shared_cache.update(runner._build_execution_contexts(metadata, metadata.slot_mapping))
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


def test_npu_gdn_uses_npu_prefill_fusion_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = MagicMock()
    mega = MagicMock(return_value=torch.zeros(1, 1, 128, dtype=torch.bfloat16))
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
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        q_seq_lens_host=torch.tensor([1], dtype=torch.int32),
        is_prefill=True,
        is_chunked_prefill=False,
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
    metadata.slot_mapping = torch.zeros(1, dtype=torch.int32)
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    layer.execution_metadata_builders = (builder,)
    runner = EagerRunner(layer, None, torch.device("cpu"))
    runner.bind_layer_caches(context.layer_caches)
    context.layer_shared_cache.update(runner._build_execution_contexts(metadata, metadata.slot_mapping))
    with forward_context(context):
        output = layer(torch.zeros(1, 8, dtype=torch.bfloat16))

    assert output.shape == (1, 128)
    assert [call[0] for call in calls.mock_calls] == ["mega"]
    kernels.causal_conv1d_qkv_prefill.assert_not_called()
    kernels.fused_gdn_gating.assert_not_called()
    kernels.chunk_gated_delta_rule.assert_not_called()
    args = mega.call_args.args
    torch.testing.assert_close(args[8], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(args[9], torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(args[10], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(args[11], torch.tensor([2], dtype=torch.int32))
    assert args[15] == 1


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
    )
    context = ForwardContext(
        None,
        torch.device("cpu"),
        metadata,
        [cache],
        layer_shared_cache=runner._build_execution_contexts(metadata, metadata.slot_mapping),
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
