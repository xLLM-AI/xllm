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

"""FlashComm1 admission, layout, and real TP collective equivalence.

Run device cases with two or four idle NPUs:
  torchrun --standalone --nproc_per_node=2 -m pytest -q tests/python/test_flash_comm1.py -k distributed
"""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from scripts.logger import logger
from tests.python.qwen3_5_test_utils import make_config
from xllm.python import distributed
from xllm.python.attention.backend import LayerCache
from xllm.python.layers.flash_comm1 import FlashComm1Context, FlashComm1MetadataBuilder, get_flash_comm1_context
from xllm.python.layers.linear import RowParallelLinear
from xllm.python.layers.npu.flash_comm1 import FlashComm1GatedMLP, FlashComm1RowParallelLinear
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnMetadata
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.input_batch import InputBatch


def _input_batch(num_tokens: int, scheduled: tuple[int, ...], device: torch.device) -> InputBatch:
    starts = [0]
    for length in scheduled:
        starts.append(starts[-1] + length)
    return InputBatch(
        input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
        positions=torch.zeros(num_tokens, dtype=torch.long, device=device),
        num_reqs=len(scheduled),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=(0,) * len(scheduled),
        query_start_loc=tuple(starts),
        is_prefilling=(True,) * len(scheduled),
        is_padding=None,
        is_dummy=False,
    )


def _flashcomm_inputs(num_tokens: int, **overrides: bool) -> tuple[InputBatch, SimpleNamespace]:
    scheduled = (num_tokens,) if num_tokens else ()
    metadata = _metadata(num_tokens, **overrides)
    return _input_batch(num_tokens, scheduled, torch.device("cpu")), metadata


def _gdn_build(builder: Qwen3_5GdnMetadataBuilder, metadata: SimpleNamespace) -> GdnMetadata:
    lengths_host = getattr(metadata, "q_seq_lens_host", None)
    if metadata.is_prefill and lengths_host is not None:
        lengths = tuple(int(value) for value in lengths_host.tolist())
    else:
        lengths = (1,) * metadata.slot_mapping.numel()
    return builder.build(_input_batch(sum(lengths), lengths, metadata.slot_mapping.device), metadata)


def _metadata(num_tokens: int, **overrides: bool) -> SimpleNamespace:
    fields = dict(is_prefill=True, is_chunked_prefill=False, is_mixed=False, is_spec_verify=False, is_dummy=False)
    fields.update(overrides)
    return SimpleNamespace(slot_mapping=torch.zeros(num_tokens, dtype=torch.int32), **fields)


def _capture_partials(layer: RowParallelLinear, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    partials: list[torch.Tensor] = []
    matmul = layer._forward_matmul

    def capture(x: torch.Tensor) -> torch.Tensor:
        output = matmul(x)
        # AllReduce mutates the projection output. Snapshot its actual input
        # before either collective, without substituting any computation.
        partials.append(output.detach().clone())
        return output

    monkeypatch.setattr(layer, "_forward_matmul", capture)
    return partials


def _sum_reference(
    partial: torch.Tensor, world_size: int, bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    values = distributed.tp_all_gather(partial, dim=0, world_size=world_size)
    values = values.reshape(world_size, *partial.shape).cpu().double()
    reference = values.sum(dim=0)
    magnitude = values.abs().sum(dim=0)
    additions = world_size - 1
    if bias is not None:
        bias = bias.detach().cpu().double()
        reference += bias
        magnitude += bias.abs()
        additions += 1
    # Bound any BF16 summation order by gamma_n * sum(abs(operands)),
    # including the replicated bias addition. A relative-only tolerance on
    # the final sum is invalid near cancellation. FP64 is the reference.
    unit_roundoff = torch.finfo(partial.dtype).eps / 2
    gamma = additions * unit_roundoff / (1 - additions * unit_roundoff)
    return reference, gamma * magnitude


def _assert_sum_roundoff(
    output: torch.Tensor, reference: torch.Tensor, bound: torch.Tensor, label: str, rank: int
) -> None:
    error = (output.detach().cpu().double() - reference).abs()
    logger.info(
        "rank=%s %s max_abs_error=%s max_roundoff_bound=%s", rank, label, error.max().item(), bound.max().item()
    )
    assert torch.all(error <= bound), f"{label} exceeds the BF16 summation error bound"


@pytest.mark.parametrize("num_tokens,active", [(0, False), (31, False), (32, True), (33, True)])
def test_prefill_threshold(num_tokens: int, active: bool) -> None:
    context = FlashComm1MetadataBuilder(2, 0, 32).build(*_flashcomm_inputs(num_tokens))
    assert (context is not None) == active
    if context is not None:
        assert context.local_num_tokens % 16 == 0
        assert context.local_num_tokens * context.tp_size >= num_tokens


@pytest.mark.parametrize("excluded", ["is_mixed", "is_spec_verify", "is_dummy"])
def test_excluded_batches_keep_replicated_rows(excluded: str) -> None:
    assert FlashComm1MetadataBuilder(2, 0, 0).build(*_flashcomm_inputs(33, **{excluded: True})) is None


def test_decode_stays_replicated_and_pure_chunked_prefill_can_shard() -> None:
    builder = FlashComm1MetadataBuilder(2, 0, 0)
    assert builder.build(*_flashcomm_inputs(33, is_prefill=False)) is None
    assert builder.build(*_flashcomm_inputs(33, is_prefill=False, is_chunked_prefill=True)) is not None


def test_configuration_defaults_and_negative_threshold() -> None:
    assert not make_config().enable_flashcomm1
    assert make_config().flashcomm1_min_prefill_tokens == 8192
    assert not make_config().enable_mmrs_fusion
    assert make_config().mmrs_comm_mode == "aiv"
    cfg = make_config(enable_flashcomm1=True, flashcomm1_min_prefill_tokens=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        cfg.validate()


@pytest.mark.parametrize("mode", ["aiv", "ai_cpu", "none", ""])
def test_mmrs_options_follow_prefill_admission(mode: str) -> None:
    cfg = make_config(enable_flashcomm1=True, enable_mmrs_fusion=True, mmrs_comm_mode=mode)
    cfg.validate()
    builder = FlashComm1MetadataBuilder(2, 0, 32, enable_mmrs_fusion=True, mmrs_comm_mode=mode)
    assert builder.build(*_flashcomm_inputs(31)) is None
    assert builder.build(*_flashcomm_inputs(32, is_prefill=False)) is None
    context = builder.build(*_flashcomm_inputs(32))
    assert context is not None and context.enable_mmrs_fusion
    assert context.mmrs_comm_mode == mode


def test_invalid_mmrs_mode_is_rejected_only_when_enabled() -> None:
    cfg = make_config(enable_flashcomm1=True, enable_mmrs_fusion=True, mmrs_comm_mode="invalid")
    with pytest.raises(ValueError, match="mmrs_comm_mode"):
        cfg.validate()
    cfg.enable_mmrs_fusion = False
    cfg.validate()
    cfg.enable_mmrs_fusion = True
    cfg.enable_flashcomm1 = False
    cfg.validate()


def _reject_unfused_matmul(x: torch.Tensor) -> torch.Tensor:
    raise AssertionError("MMRS must not execute a separate matmul")


@pytest.mark.parametrize("transposed", [False, True])
def test_mmrs_pads_inputs_reuses_weight_storage_and_adds_bias_once(
    monkeypatch: pytest.MonkeyPatch, transposed: bool
) -> None:
    layer = FlashComm1RowParallelLinear(256, 32, 2, bias=True, dtype=torch.bfloat16, device="cpu")
    if transposed:
        layer.weight = torch.nn.Parameter(layer.weight.t().contiguous())
        layer._weight_is_transposed = True
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.fill_(0.125)
    monkeypatch.setattr(layer, "_forward_matmul", _reject_unfused_matmul)
    seen: list[torch.Tensor] = []

    def mmrs(x: torch.Tensor, weight: torch.Tensor, world: int, mode: str) -> torch.Tensor:
        assert x.shape == (64, 256) and world == 2 and mode == "aiv"
        assert torch.count_nonzero(x[33:]) == 0
        assert weight.shape == (256, 32) and weight.data_ptr() == layer.weight.data_ptr()
        seen.append(weight)
        return torch.ones(32, 32, dtype=x.dtype)

    monkeypatch.setattr(distributed, "matmul_reduce_scatter", mmrs)
    plan = FlashComm1Context(33, 32, 2, 0, enable_mmrs_fusion=True)
    context = ForwardContext(None, torch.device("cpu"), None, [], layer_shared_cache={FlashComm1Context: plan})
    with forward_context(context), torch.no_grad():
        actual = layer(torch.ones(33, 256, dtype=torch.bfloat16))
        layer.weight.fill_(2)
        layer(torch.ones(33, 256, dtype=torch.bfloat16))
    torch.testing.assert_close(actual, torch.full_like(actual, 1.125), rtol=0, atol=0)
    assert len(seen) == 2 and torch.all(seen[0] == 2)


def test_mmrs_execution_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = FlashComm1RowParallelLinear(256, 32, 2, dtype=torch.bfloat16, device="cpu")
    monkeypatch.setattr(layer, "_forward_matmul", _reject_unfused_matmul)

    def fail(x: torch.Tensor, weight: torch.Tensor, world: int, mode: str) -> torch.Tensor:
        raise RuntimeError("MMRS execution failed")

    monkeypatch.setattr(distributed, "matmul_reduce_scatter", fail)
    plan = FlashComm1Context(32, 16, 2, 0, enable_mmrs_fusion=True)
    context = ForwardContext(None, torch.device("cpu"), None, [], layer_shared_cache={FlashComm1Context: plan})
    with forward_context(context), pytest.raises(RuntimeError, match="MMRS execution failed"):
        layer(torch.ones(32, 256, dtype=torch.bfloat16))


@pytest.mark.parametrize("width,dtype", [(128, torch.bfloat16), (256, torch.float32)])
def test_mmrs_unsupported_geometry_keeps_separate_reduction(
    monkeypatch: pytest.MonkeyPatch, width: int, dtype: torch.dtype
) -> None:
    layer = FlashComm1RowParallelLinear(width, 32, 2, dtype=dtype, device="cpu")
    with torch.no_grad():
        layer.weight.fill_(1)

    def reduce(x: torch.Tensor, world: int) -> torch.Tensor:
        return x.chunk(world)[0] * world

    def reject(x: torch.Tensor, weight: torch.Tensor, world: int, mode: str) -> torch.Tensor:
        raise AssertionError("unsupported MMRS must be excluded before execution")

    monkeypatch.setattr(distributed, "reduce_scatter", reduce)
    monkeypatch.setattr(distributed, "matmul_reduce_scatter", reject)
    plan = FlashComm1Context(32, 16, 2, 0, enable_mmrs_fusion=True)
    context = ForwardContext(None, torch.device("cpu"), None, [], layer_shared_cache={FlashComm1Context: plan})
    with forward_context(context), torch.no_grad():
        actual = layer(torch.ones(32, width, dtype=dtype))
    torch.testing.assert_close(actual, torch.full((16, 32), 2 * width, dtype=dtype), rtol=0, atol=0)


def test_context_is_scoped_to_one_forward() -> None:
    plan = FlashComm1Context(33, 32, 2, 0)
    context = ForwardContext(None, torch.device("cpu"), None, [], layer_shared_cache={FlashComm1Context: plan})
    with forward_context(context):
        assert get_flash_comm1_context() is plan
    assert get_flash_comm1_context() is None


@pytest.fixture(scope="module")
def npu_tp() -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (2, 4) or "LOCAL_RANK" not in os.environ:
        pytest.skip("requires two or four idle NPUs and torchrun")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("npu", rank)
    torch.npu.set_device(device)
    group = distributed.init_tp_group(
        os.environ["MASTER_ADDR"],
        int(os.environ["MASTER_PORT"]) + 7,
        rank,
        world_size,
        str(device),
        rank,
        world_size,
        0,
    )
    from xllm.python.distributed.npu.hccl import get_hccl_comm

    get_hccl_comm(group, device, initialize=True)
    return device, rank, world_size


@pytest.mark.parametrize("num_tokens", [1, 31, 32, 33, 97])
def test_distributed_projection_and_mlp_match_all_reduce(
    npu_tp: tuple[torch.device, int, int], num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    device, rank, world_size = npu_tp
    torch.manual_seed(1234)
    hidden = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device=device)
    plan = FlashComm1MetadataBuilder(world_size, rank, 0).build(*_flashcomm_inputs(num_tokens))
    assert plan is not None
    torch.testing.assert_close(plan.gather(plan.shard(hidden)), hidden, rtol=0, atol=0)

    torch.manual_seed(4321 + rank)
    local_hidden_size = 64 // world_size
    projection = FlashComm1RowParallelLinear(
        local_hidden_size, 64, world_size, bias=True, dtype=torch.bfloat16, device=device
    )
    mlp = FlashComm1GatedMLP(64, 128, world_size, torch.bfloat16, device)
    projection_partials = _capture_partials(projection, monkeypatch)
    mlp_partials = _capture_partials(mlp.down_proj, monkeypatch)
    with torch.no_grad():
        for parameter in (*projection.parameters(), *mlp.parameters()):
            parameter.normal_(std=0.02)
        projection.bias.fill_(0.125)
        local_input = hidden[:, rank * local_hidden_size : (rank + 1) * local_hidden_size].contiguous()
        expected_projection = projection(local_input)
        expected_mlp = mlp(hidden)
        context = ForwardContext(None, device, None, [], layer_shared_cache={FlashComm1Context: plan})
        with forward_context(context):
            actual_projection = plan.gather(projection(local_input))
            actual_mlp = plan.gather(mlp(plan.shard(hidden)))
    projection_ref, projection_bound = _sum_reference(projection_partials[0], world_size, projection.bias)
    mlp_ref, mlp_bound = _sum_reference(mlp_partials[0], world_size)
    torch.testing.assert_close(projection_partials[0], projection_partials[1], rtol=0, atol=0)
    torch.testing.assert_close(mlp_partials[0], mlp_partials[1], rtol=0, atol=0)
    for name, actual, expected, reference, bound in (
        ("projection", actual_projection, expected_projection, projection_ref, projection_bound),
        ("mlp", actual_mlp, expected_mlp, mlp_ref, mlp_bound),
    ):
        _assert_sum_roundoff(actual, reference, bound, f"{name}/reduce_scatter", rank)
        _assert_sum_roundoff(expected, reference, bound, f"{name}/all_reduce", rank)
        if world_size == 2:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_distributed_gdn_preserves_prefill_and_decode_state(
    npu_tp: tuple[torch.device, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    device, rank, world_size = npu_tp
    cfg = make_config(
        num_experts=0,
        tp_size=world_size,
        tp_rank=rank,
        world_size=world_size,
        moe_tp_size=world_size,
        linear_num_key_heads=world_size,
        linear_num_value_heads=world_size,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention"],
    )
    torch.manual_seed(5234 + rank)
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    partials = _capture_partials(layer.out_proj, monkeypatch)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.02)
        layer.norm_weight.fill_(1)
    torch.manual_seed(8234)
    hidden = torch.randn(37, 64, dtype=torch.bfloat16, device=device)
    decode_hidden = torch.randn(2, 64, dtype=torch.bfloat16, device=device)
    metadata = _metadata(37)
    metadata.slot_mapping = metadata.slot_mapping.to(device)
    metadata.linear_state_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    metadata.has_initial_state = torch.zeros(2, dtype=torch.bool, device=device)
    metadata.q_seq_lens_host = torch.tensor([17, 20], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 17, 37], dtype=torch.int32, device=device)
    metadata.q_cu_seq_lens_host_values = [0, 17, 37]
    plan = FlashComm1MetadataBuilder(world_size, rank, 0).build(_input_batch(37, (17, 20), device), metadata)
    outputs = []
    for enabled in (False, True):
        cache = LayerCache(
            key=None,
            value=None,
            conv=torch.zeros(3, 3, 384, dtype=torch.bfloat16, device=device),
            ssm=torch.zeros(3, 1, 128, 128, dtype=torch.float32, device=device),
        )
        builder = Qwen3_5GdnMetadataBuilder(cfg)
        builder.bind_layer_caches([cache])
        shared = {GdnMetadata: _gdn_build(builder, metadata), FlashComm1Context: plan if enabled else None}
        with (
            torch.no_grad(),
            forward_context(ForwardContext(None, device, metadata, [cache], layer_shared_cache=shared)),
        ):
            output = layer(plan.shard(hidden) if enabled else hidden)
            if enabled:
                output = plan.gather(output)
        prefill_conv, prefill_ssm = cache.conv.clone(), cache.ssm.clone()
        decode = _metadata(2, is_prefill=False)
        decode.slot_mapping = decode.slot_mapping.to(device)
        decode.linear_state_indices = metadata.linear_state_indices
        with (
            torch.no_grad(),
            forward_context(
                ForwardContext(
                    None, device, decode, [cache], layer_shared_cache={GdnMetadata: _gdn_build(builder, decode)}
                )
            ),
        ):
            decode_output = layer(decode_hidden)
        outputs.append((output, prefill_conv, prefill_ssm, decode_output, cache.conv, cache.ssm))
    prefill_ref, prefill_bound = _sum_reference(partials[0], world_size)
    decode_ref, decode_bound = _sum_reference(partials[1], world_size)
    torch.testing.assert_close(partials[0], partials[2], rtol=0, atol=0)
    torch.testing.assert_close(partials[1], partials[3], rtol=0, atol=0)
    # Recurrent state is independent of the output reduction and must remain
    # bitwise identical both immediately after prefill and after decode.
    for index in (1, 2, 4, 5):
        torch.testing.assert_close(outputs[1][index], outputs[0][index], rtol=0, atol=0)
    for index, name, reference, bound in (
        (0, "gdn_prefill", prefill_ref, prefill_bound),
        (3, "gdn_decode", decode_ref, decode_bound),
    ):
        _assert_sum_roundoff(outputs[0][index], reference, bound, f"{name}/off", rank)
        _assert_sum_roundoff(outputs[1][index], reference, bound, f"{name}/on", rank)
        if world_size == 2:
            torch.testing.assert_close(outputs[1][index], outputs[0][index], rtol=0, atol=0)


def _matmul_sum_reference(
    inputs: torch.Tensor, weight: torch.Tensor, world_size: int, bias: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 reference including matmul accumulation and low-precision reduction."""
    all_inputs = distributed.tp_all_gather(inputs.detach(), dim=0, world_size=world_size)
    all_weights = distributed.tp_all_gather(weight.detach(), dim=0, world_size=world_size)
    all_inputs = all_inputs.reshape(world_size, *inputs.shape).cpu().double()
    all_weights = all_weights.reshape(world_size, *weight.shape).cpu().double()
    partials = torch.bmm(all_inputs, all_weights.transpose(1, 2))
    reference = partials.sum(0)
    magnitude = partials.abs().sum(0)
    product_magnitude = torch.bmm(all_inputs.abs(), all_weights.transpose(1, 2).abs()).sum(0)
    fp32_roundoff = torch.finfo(torch.float32).eps / 2
    accumulation_error = inputs.shape[1] * fp32_roundoff / (1 - inputs.shape[1] * fp32_roundoff) * product_magnitude
    additions = world_size + 1  # Allow partial-result and final-output rounding.
    if bias is not None:
        bias = bias.detach().cpu().double()
        reference += bias
        magnitude += bias.abs()
        additions += 1
    roundoff = torch.finfo(inputs.dtype).eps / 2
    gamma = additions * roundoff / (1 - additions * roundoff)
    return reference, gamma * magnitude + (1 + gamma) * accumulation_error


@pytest.mark.parametrize(
    "tokens,transposed,bias",
    [(1, False, False), (31, False, False), (32, False, True), (33, False, True), (97, True, False)],
)
def test_distributed_mmrs_projection_matches_fp64(
    npu_tp: tuple[torch.device, int, int], monkeypatch: pytest.MonkeyPatch, tokens: int, transposed: bool, bias: bool
) -> None:
    device, rank, world_size = npu_tp
    torch.manual_seed(7213 + rank)
    layer = FlashComm1RowParallelLinear(256, 128, world_size, bias=bias, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        layer.weight.normal_(std=0.03)
        if bias:
            layer.bias.fill_(0.125)
        if transposed:
            layer.weight = torch.nn.Parameter(layer.weight.t().contiguous())
            layer._weight_is_transposed = True
    inputs = torch.randn(tokens, 256, dtype=torch.bfloat16, device=device) * 0.03
    plan = FlashComm1MetadataBuilder(world_size, rank, 0).build(*_flashcomm_inputs(tokens))
    assert plan is not None
    outputs = []
    for enabled in (False, True):
        context = ForwardContext(
            None, device, None, [], layer_shared_cache={FlashComm1Context: replace(plan, enable_mmrs_fusion=enabled)}
        )
        with forward_context(context), torch.no_grad(), monkeypatch.context() as guard:
            if enabled:
                guard.setattr(layer, "_forward_matmul", _reject_unfused_matmul)
            outputs.append(plan.gather(layer(inputs)))
    checkpoint_weight = layer.weight.t() if transposed else layer.weight
    reference, bound = _matmul_sum_reference(inputs, checkpoint_weight, world_size, layer.bias)
    _assert_sum_roundoff(outputs[0], reference, bound, "mmrs_projection/separate", rank)
    _assert_sum_roundoff(outputs[1], reference, bound, "mmrs_projection/fused", rank)


def test_distributed_mmrs_gdn_preserves_state_and_decode(
    npu_tp: tuple[torch.device, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    device, rank, world_size = npu_tp
    cfg = make_config(
        hidden_size=256,
        num_experts=0,
        tp_size=world_size,
        tp_rank=rank,
        world_size=world_size,
        moe_tp_size=world_size,
        linear_num_key_heads=world_size * 2,
        linear_num_value_heads=world_size * 2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention"],
    )
    torch.manual_seed(1297 + rank)
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.02)
        layer.norm_weight.fill_(1)
    projection_inputs: list[torch.Tensor] = []

    def capture_input(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        projection_inputs.append(args[0].detach().clone())

    layer.out_proj.register_forward_pre_hook(capture_input)
    mmrs_calls = []
    mmrs = distributed.matmul_reduce_scatter

    def record_mmrs(x: torch.Tensor, weight: torch.Tensor, world: int, mode: str) -> torch.Tensor:
        mmrs_calls.append(x.shape)
        return mmrs(x, weight, world, mode)

    monkeypatch.setattr(distributed, "matmul_reduce_scatter", record_mmrs)
    torch.manual_seed(1024)
    hidden = torch.randn(37, 256, dtype=torch.bfloat16, device=device)
    decode_hidden = torch.randn(2, 256, dtype=torch.bfloat16, device=device)
    metadata = _metadata(37)
    metadata.slot_mapping = metadata.slot_mapping.to(device)
    metadata.linear_state_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    metadata.has_initial_state = torch.zeros(2, dtype=torch.bool, device=device)
    metadata.q_seq_lens_host = torch.tensor([17, 20], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 17, 37], dtype=torch.int32, device=device)
    metadata.q_cu_seq_lens_host_values = [0, 17, 37]
    plan = FlashComm1MetadataBuilder(world_size, rank, 0).build(_input_batch(37, (17, 20), device), metadata)
    assert plan is not None
    results = []
    for enabled in (False, True):
        cache = LayerCache(
            key=None,
            value=None,
            conv=torch.zeros(3, 3, 768, dtype=torch.bfloat16, device=device),
            ssm=torch.zeros(3, 2, 128, 128, dtype=torch.float32, device=device),
        )
        builder = Qwen3_5GdnMetadataBuilder(cfg)
        builder.bind_layer_caches([cache])
        shared = {
            GdnMetadata: _gdn_build(builder, metadata),
            FlashComm1Context: replace(plan, enable_mmrs_fusion=enabled),
        }
        with (
            torch.no_grad(),
            forward_context(ForwardContext(None, device, metadata, [cache], layer_shared_cache=shared)),
        ):
            output = plan.gather(layer(plan.shard(hidden)))
        prefill_conv, prefill_ssm = cache.conv.clone(), cache.ssm.clone()
        decode = _metadata(2, is_prefill=False)
        decode.slot_mapping = decode.slot_mapping.to(device)
        decode.linear_state_indices = metadata.linear_state_indices
        with (
            torch.no_grad(),
            forward_context(
                ForwardContext(
                    None, device, decode, [cache], layer_shared_cache={GdnMetadata: _gdn_build(builder, decode)}
                )
            ),
        ):
            decode_output = layer(decode_hidden)
        results.append((output, prefill_conv, prefill_ssm, decode_output, cache.conv, cache.ssm))
    prefill_ref, prefill_bound = _matmul_sum_reference(projection_inputs[0], layer.out_proj.weight, world_size)
    decode_ref, decode_bound = _matmul_sum_reference(projection_inputs[1], layer.out_proj.weight, world_size)
    assert len(mmrs_calls) == 1, "only the opted-in prefill output projection should fuse"
    torch.testing.assert_close(projection_inputs[0], projection_inputs[2], rtol=0, atol=0)
    torch.testing.assert_close(projection_inputs[1], projection_inputs[3], rtol=0, atol=0)
    for index in (1, 2, 4, 5):
        torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
    for result in results:
        _assert_sum_roundoff(result[0], prefill_ref, prefill_bound, "mmrs_gdn/prefill", rank)
        _assert_sum_roundoff(result[3], decode_ref, decode_bound, "mmrs_gdn/decode", rank)
