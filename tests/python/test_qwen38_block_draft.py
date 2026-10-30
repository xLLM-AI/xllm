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

"""Qwen3.8 target capture and block-verification state contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tests.python.qwen3_5_test_utils import make_linear_config
from xllm.python import kernels
from xllm.python.attention.backend import LayerCache, LinearVerifyGraphInputs
from xllm.python.layers.flash_comm1 import FlashComm1Context
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnMetadata, GdnPrefillMetadata, GdnVerifyMetadata
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder
from xllm.python.layers.rotary_embedding import RotaryEmbedding
from xllm.python.model_executor.dspark_graph_input import DSparkAdaptiveGraphInputs
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner
from xllm.python.models import qwen3_5
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_5 import Qwen3_5Model
from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config
from xllm.python.models.qwen3_dspark import Qwen3DSparkConfig


class _ResidualLayer(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        residual = hidden if residual is None else hidden + residual
        return torch.full_like(hidden, self.delta), residual


class _FinalNorm(nn.Module):
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (hidden + residual) * 0.5, residual


@pytest.mark.parametrize("capture_layers", [(), (2, 0)])
def test_target_captures_unnormalized_post_layer_streams(capture_layers: tuple[int, ...]) -> None:
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model._inputs_embeds = None
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList(_ResidualLayer(delta) for delta in (1.0, 2.0, 3.0))
    model.norm = _FinalNorm()
    model.aux_hidden_capture = AuxHiddenCapture(capture_layers)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = model(hidden, torch.arange(2))

    if not capture_layers:
        torch.testing.assert_close(actual, (hidden + 6.0) * 0.5)
    else:
        output, captured = actual
        torch.testing.assert_close(output, (hidden + 6.0) * 0.5)
        torch.testing.assert_close(captured, torch.cat((hidden + 6.0, hidden + 1.0), dim=-1))


@pytest.mark.parametrize("capture_layers", [(), (2, 0)])
def test_target_capture_gathers_flashcomm1_rows(
    monkeypatch: pytest.MonkeyPatch, capture_layers: tuple[int, ...]
) -> None:
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model._inputs_embeds = None
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList(_ResidualLayer(delta) for delta in (1.0, 2.0, 3.0))
    model.norm = _FinalNorm()
    model.aux_hidden_capture = AuxHiddenCapture(capture_layers)
    context = FlashComm1Context(num_tokens=2, local_num_tokens=1, tp_size=2, tp_rank=0)
    monkeypatch.setattr(qwen3_5, "get_flash_comm1_context", lambda: context)
    monkeypatch.setattr(
        "xllm.python.distributed.tp_all_gather",
        lambda hidden, dim, world_size: torch.cat([hidden] * world_size, dim=dim),
    )
    hidden = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    actual = model(hidden, torch.arange(2))
    if capture_layers:
        output, captured = actual
        torch.testing.assert_close(output, (hidden + 6.0) * 0.5)
        torch.testing.assert_close(captured, torch.cat((hidden + 6.0, hidden + 1.0), dim=-1))
    else:
        torch.testing.assert_close(actual, (hidden + 6.0) * 0.5)


def _draft_config() -> dict:
    return {
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    }


def test_incoai_qwen38_dflash2_geometry() -> None:
    config = _draft_config()
    config.update(
        sliding_window=2048,
        dflash_config={
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 256,
            "selector_top_k": 16,
        },
    )
    cfg = DFlash2Qwen3Config.from_dict(config)
    cfg.validate()
    assert cfg.rope_theta == 10000000
    assert cfg.block_size == 8
    assert cfg.head_split() == (32, 8)


def test_reflected_legacy_rope_scaling_is_not_an_hf_dictionary() -> None:
    assert Qwen3Config.from_dict({"rope_scaling": -1}).rope_scaling == {}


def test_radixark_qwen38_dspark_preserves_yarn_and_null_window() -> None:
    config = _draft_config()
    config.update(
        markov_rank=256,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        sliding_window=None,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192},
    )
    cfg = Qwen3DSparkConfig.from_dict(config)
    cfg.validate()
    assert cfg.sliding_window == 0
    assert cfg.rope_scaling["factor"] == 32.0
    assert cfg.enable_confidence_head


def test_reflected_dspark_rope_matches_hf_rope() -> None:
    config = _draft_config()
    config.update(
        markov_rank=256,
        rope_scaling_rope_type="yarn",
        rope_scaling_factor=32.0,
        rope_scaling_original_max_position_embeddings=8192,
        rope_scaling_beta_fast=32,
        rope_scaling_beta_slow=1,
        rope_scaling_attn_factor=0.0,
    )
    cfg = Qwen3DSparkConfig.from_dict(config)
    cache = RotaryEmbedding(128, 32, cfg.rope_theta, rope_scaling=cfg.rope_scaling).cos_sin_cache
    expected = RotaryEmbedding(
        128,
        32,
        cfg.rope_theta,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192},
    ).cos_sin_cache
    torch.testing.assert_close(cache, expected)
    torch.testing.assert_close(cache[0, :64], torch.full((64,), 1.0 + 0.1 * math.log(32.0)))
    assert not torch.equal(cache[31], RotaryEmbedding(128, 32, cfg.rope_theta).cos_sin_cache[31])


def _verify_builder(checkpoint_stride: int = 8) -> Qwen3_5GdnMetadataBuilder:
    cfg = make_linear_config(
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
                conv=torch.zeros(3, checkpoint_stride + 2, 384, dtype=torch.bfloat16),
                ssm=torch.zeros(3 * checkpoint_stride, 1, 128, 128),
            )
        ]
    )
    return builder


def _gdn_input_batch(scheduled: tuple[int, ...]) -> InputBatch:
    total = sum(scheduled)
    boundaries = [0]
    for length in scheduled:
        boundaries.append(boundaries[-1] + length)
    return InputBatch(
        input_ids=torch.zeros(total, dtype=torch.long),
        positions=torch.zeros(total, dtype=torch.long),
        num_reqs=len(scheduled),
        num_tokens=total,
        num_tokens_after_padding=total,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=(0,) * len(scheduled),
        query_start_loc=tuple(boundaries),
        is_prefilling=(True,) * len(scheduled),
        is_padding=None,
        is_dummy=False,
    )


@pytest.mark.parametrize("end_only", [False, True])
def test_prefill_accepts_native_slot_cumulative_ends(end_only: bool) -> None:
    ends = [2, 5] if end_only else [0, 2, 5]
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(5, dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        is_dummy=False,
        is_spec_verify=False,
        is_prefill=True,
        is_chunked_prefill=False,
        q_seq_lens_host=torch.tensor([2, 3], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor(ends, dtype=torch.int32),
        q_cu_seq_lens_host_values=ends,
        has_initial_state=torch.tensor([False, True]),
    )
    actual = _verify_builder().build(_gdn_input_batch((2, 3)), metadata)
    assert isinstance(actual, GdnPrefillMetadata)
    torch.testing.assert_close(actual.cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32))
    if end_only:
        metadata.q_cu_seq_lens[0] = 1
        with pytest.raises(ValueError, match="do not match host"):
            _verify_builder().build(_gdn_input_batch((2, 3)), metadata)


@pytest.mark.parametrize("expanded", [False, True])
def test_verify_metadata_keeps_previous_accepted_checkpoint(expanded: bool) -> None:
    states = torch.tensor([1, 2], dtype=torch.int32)
    counts = torch.tensor([1, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(16, dtype=torch.int32),
        q_seq_lens_host=torch.tensor([8, 8], dtype=torch.int32),
        linear_state_indices=states.repeat_interleave(8) if expanded else states,
        num_accepted_tokens=counts,
        num_accepted_tokens_host_values=(1, 5),
        is_dummy=False,
        is_spec_verify=True,
    )
    actual = _verify_builder().build(_gdn_input_batch((8, 8)), metadata)
    assert isinstance(actual, GdnVerifyMetadata)
    assert actual.query_width == 8
    torch.testing.assert_close(actual.state_indices, states)
    torch.testing.assert_close(actual.num_accepted_tokens, counts)
    if not expanded:
        assert actual.num_accepted_tokens is counts
        assert actual.state_indices is states


def test_verify_rejects_query_width_exceeding_checkpoint_stride() -> None:
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(18, dtype=torch.int32),
        q_seq_lens_host=torch.tensor([9, 9], dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        num_accepted_tokens_host_values=(1, 1),
        is_dummy=False,
        is_spec_verify=True,
    )
    with pytest.raises(ValueError, match="checkpoint stride"):
        _verify_builder().build(_gdn_input_batch((9, 9)), metadata)


def test_verify_graph_uses_owned_device_maps_without_host_lengths() -> None:
    graph = LinearVerifyGraphInputs(
        torch.tensor([2, 1, 0], dtype=torch.int32),
        torch.tensor([3, 5, 1], dtype=torch.int32),
        8,
        torch.tensor([0, 1, 8, 9, 10, 24, 24, 24], dtype=torch.int64),
        torch.full((24,), 8, dtype=torch.int64),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(8, dtype=torch.int32),
        is_dummy=False,
        linear_verify_graph=graph,
    )
    tokens = torch.zeros(8, dtype=torch.int32)
    batch = _gdn_input_batch((2, 3)).bind_graph_inputs(tokens, tokens, torch.arange(8) >= 5)
    actual = _verify_builder().build(batch, metadata)
    assert isinstance(actual, GdnVerifyMetadata)
    assert actual.state_indices is graph.state_indices
    assert actual.num_accepted_tokens is graph.num_accepted_tokens
    assert actual.packed_to_dense is graph.packed_to_dense
    assert actual.dense_to_packed is graph.dense_to_packed
    assert actual.query_width == 8


@pytest.mark.parametrize("packed_kernel", [False, True], ids=["dense", "varlen"])
def test_verify_graph_requires_maps_only_for_dense_kernel(packed_kernel: bool) -> None:
    graph = LinearVerifyGraphInputs(
        torch.tensor([2, 1, 0], dtype=torch.int32),
        torch.tensor([3, 5, 1], dtype=torch.int32),
        8,
        cu_seqlens=torch.tensor([0, 2, 5, 5], dtype=torch.int32) if packed_kernel else None,
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(8, dtype=torch.int32),
        is_dummy=False,
        linear_verify_graph=graph,
    )
    tokens = torch.zeros(8, dtype=torch.int32)
    batch = _gdn_input_batch((2, 3)).bind_graph_inputs(tokens, tokens, torch.arange(8) >= 5)
    if not packed_kernel:
        with pytest.raises(ValueError, match="fixed request indices and token maps"):
            _verify_builder().build(batch, metadata)
        return
    actual = _verify_builder().build(batch, metadata)
    assert actual.state_indices is graph.state_indices
    assert actual.num_accepted_tokens is graph.num_accepted_tokens
    assert actual.cu_seqlens is graph.cu_seqlens
    assert actual.packed_to_dense is None and actual.dense_to_packed is None


@pytest.mark.parametrize("width", [2, 4, 6, 7, 8, 17])
def test_fixed_verify_routes_to_dense_kernel_without_host_lengths(monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    cfg = make_linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("cpu"))
    layer.out_proj = nn.Identity()
    builder = _verify_builder(width)
    states = torch.tensor([1, 2], dtype=torch.int32)
    metadata = builder.build(
        _gdn_input_batch((width, width)),
        SimpleNamespace(
            slot_mapping=torch.arange(2 * width, dtype=torch.int32),
            q_seq_lens_host=None,
            linear_state_indices=states,
            num_accepted_tokens=torch.tensor([1, width], dtype=torch.int32),
            num_accepted_tokens_host_values=(),
            is_dummy=False,
            is_spec_verify=True,
        ),
    )
    verify = MagicMock(return_value=torch.ones(2, width, 1, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode", verify, raising=False)
    varlen = MagicMock(side_effect=AssertionError("fixed verification must not invoke the varlen kernel"))
    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode_varlen", varlen, raising=False)
    context = ForwardContext(None, torch.device("cpu"), None, [])
    context.layer_shared_cache[GdnMetadata] = metadata
    with forward_context(context):
        output = layer(torch.zeros(2 * width, 8, dtype=torch.bfloat16))
    assert output.shape == (2 * width, 128)
    varlen.assert_not_called()
    args = verify.call_args.args
    assert args[0].shape == (2, width, 384)
    torch.testing.assert_close(args[9], states)
    torch.testing.assert_close(args[11], torch.tensor([1, width], dtype=torch.int32))


@pytest.mark.parametrize("padding_value", [77.0, float("nan")], ids=["finite", "nonfinite"])
def test_gdn_adaptive_packed_padding_does_not_enter_live_prefixes(
    monkeypatch: pytest.MonkeyPatch, padding_value: float
) -> None:
    layer = NpuQwen3_5GatedDeltaNet(
        make_linear_config(
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        ),
        0,
        torch.float32,
        torch.device("cpu"),
    )

    def project(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden, hidden[:, :1], hidden[:, :1], hidden.unsqueeze(1)

    dense_inputs: list[torch.Tensor] = []

    def verify(qkv: torch.Tensor, z: torch.Tensor, *args: object) -> torch.Tensor:
        dense_inputs.append(qkv.clone())
        return z.cumsum(dim=1)

    monkeypatch.setattr(layer, "_project_verify_inputs", project)
    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode", verify, raising=False)
    hidden = torch.arange(1, 9, dtype=torch.float32).view(8, 1).expand(8, 128).clone()
    hidden[5:].fill_(padding_value)
    metadata = GdnVerifyMetadata(
        state_caches={0: SimpleNamespace(conv=torch.empty(0), ssm=torch.empty(0))},
        state_indices=torch.tensor([1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        query_width=4,
        packed_to_dense=torch.tensor([0, 1, 4, 5, 6, 8, 8, 8]),
        dense_to_packed=torch.tensor([0, 1, 8, 8, 2, 3, 4, 8]),
    )

    actual = layer._verify(hidden, metadata)

    expected = torch.tensor([1, 3, 3, 7, 12, 0, 0, 0], dtype=torch.float32).view(8, 1).expand(8, 128)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        dense_inputs[0][:, :, 0], torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.float32)
    )


@pytest.mark.parametrize(
    ("entrypoint", "error_type", "error_message"),
    [
        ("python", TypeError, "write_state_indices"),
        ("native", RuntimeError, r"expected at most 15 argument\(s\) but received 16 argument\(s\)"),
    ],
)
def test_varlen_gdn_accepts_only_one_state_index_mapping(
    entrypoint: str, error_type: type[Exception], error_message: str
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from xllm.python.kernels_npu.gated_delta_net import mega_gdn_mtp_decode_varlen

    operator = mega_gdn_mtp_decode_varlen if entrypoint == "python" else torch.ops.xllm_ops.mega_gdn_mtp_decode_varlen
    with FakeTensorMode():
        arguments = dict(
            qkv=torch.empty(4, 384, dtype=torch.bfloat16),
            z=torch.empty(4, 1, 128, dtype=torch.bfloat16),
            b=torch.empty(4, 1, dtype=torch.bfloat16),
            a=torch.empty(4, 1, dtype=torch.bfloat16),
            conv_weight=torch.empty(4, 384, dtype=torch.bfloat16),
            conv_state=torch.empty(2, 10, 384, dtype=torch.bfloat16),
            a_log=torch.empty(1),
            dt_bias=torch.empty(1),
            ssm_state=torch.empty(16, 1, 128, 128),
            state_indices=torch.tensor([1, 0], dtype=torch.int32),
            num_accepted_tokens=torch.ones(2, dtype=torch.int32),
            norm_weight=torch.empty(128, dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
            out=torch.empty(4, 1, 128, dtype=torch.bfloat16),
            fla_ssm_state_layout=True,
        )
        assert operator(**arguments) is None
        with pytest.raises(error_type, match=error_message):
            operator(**arguments, write_state_indices=torch.tensor([0, 1], dtype=torch.int32))


@pytest.mark.parametrize("batch_size", [1, 2, 17, 31, 32, 33, 64, 65, 127, 128])
def test_varlen_gdn_verifies_packed_rows_without_dense_padding(
    monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    layer = NpuQwen3_5GatedDeltaNet(
        make_linear_config(
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        ),
        0,
        torch.float32,
        torch.device("cpu"),
    )
    token_count = batch_size * 2
    hidden = torch.arange(1, token_count + 4, dtype=torch.float32).view(-1, 1).expand(-1, 128).clone()
    hidden[token_count:].fill_(float("nan"))
    monkeypatch.setattr(layer, "_project_prefill_inputs", lambda x: (x, x[:, :1], x[:, :1], x.unsqueeze(1)))
    calls: list[tuple[int, int]] = []

    def verify(
        qkv: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        conv_weight: torch.Tensor,
        conv: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm: torch.Tensor,
        state_indices: torch.Tensor,
        accepted: torch.Tensor,
        norm: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
        layout: bool,
    ) -> None:
        assert qkv.shape == hidden.shape
        assert accepted.tolist() == [7] * state_indices.numel()
        calls.append((int(offsets[0]), int(offsets[-1])))
        for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            out[start:end].copy_(z[start:end].cumsum(dim=0))

    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode_varlen", verify, raising=False)
    metadata = GdnVerifyMetadata(
        {0: SimpleNamespace(conv=torch.empty(0), ssm=torch.empty(0))},
        torch.arange(1, batch_size + 1, dtype=torch.int32),
        torch.full((batch_size,), 7, dtype=torch.int32),
        8,
        None,
        None,
        torch.arange(0, token_count + 1, 2, dtype=torch.int32),
    )
    actual = layer._verify(hidden, metadata)
    expected = torch.zeros_like(hidden)
    expected[:token_count] = hidden[:token_count].reshape(batch_size, 2, 128).cumsum(dim=1).reshape(-1, 128)
    torch.testing.assert_close(actual, expected)
    assert calls == [(0, token_count)]


@pytest.mark.parametrize("key_heads,value_heads", [(1, 1), (4, 12)], ids=["minimal", "qwen38_tp4"])
def test_npu_verify_rejection_matches_sequential_decode(key_heads: int, value_heads: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from xllm.python.kernels_npu.gated_delta_net import mega_gdn_decode, mega_gdn_mtp_decode

    torch.manual_seed(23)
    device = torch.device("npu:0")
    batch, width, slots = 2, 8, 3
    conv_dim = (2 * key_heads + value_heads) * 128

    def random(shape: tuple[int, ...], scale: float = 0.1) -> torch.Tensor:
        return (torch.randn(shape) * scale).to(device=device, dtype=torch.bfloat16)

    conv_weight = random((4, conv_dim))
    a_log = torch.zeros(value_heads, device=device)
    dt_bias = torch.zeros(value_heads, device=device)
    norm_weight = torch.ones(128, device=device, dtype=torch.bfloat16)
    indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
    conv = torch.zeros(slots, width + 2, conv_dim, device=device, dtype=torch.bfloat16)
    ssm = torch.zeros(slots * width, value_heads, 128, 128, device=device)
    reference_conv = torch.zeros(slots, 3, conv_dim, device=device, dtype=torch.bfloat16)
    reference_ssm = torch.zeros(slots, value_heads, 128, 128, device=device)
    counts = torch.ones(batch, device=device, dtype=torch.int32)
    checkpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
    for iteration in range(2):
        if iteration:
            # Resume after different accepted prefixes of the preceding block.
            for row, accepted in enumerate((1, 5)):
                reference_conv[row + 1].copy_(checkpoints[accepted - 1][0][row + 1])
                reference_ssm[row + 1].copy_(checkpoints[accepted - 1][1][row + 1])
            counts.copy_(torch.tensor([1, 5], device=device, dtype=torch.int32))
        qkv = random((batch, width, conv_dim))
        z = random((batch, width, value_heads, 128))
        b = random((batch, width, value_heads))
        a = random((batch, width, value_heads))
        actual = mega_gdn_mtp_decode(
            qkv, z, b, a, conv_weight, conv, a_log, dt_bias, ssm, indices, indices, counts, norm_weight
        )
        checkpoints = []
        reference_outputs: list[torch.Tensor] = []
        for position in range(width):
            reference_outputs.append(
                mega_gdn_decode(
                    qkv[:, position].contiguous(),
                    z[:, position].contiguous(),
                    b[:, position].contiguous(),
                    a[:, position].contiguous(),
                    conv_weight,
                    reference_conv,
                    a_log,
                    dt_bias,
                    reference_ssm,
                    indices,
                    indices,
                    norm_weight,
                )
            )
            checkpoints.append((reference_conv.clone(), reference_ssm.clone()))
        expected = torch.stack(reference_outputs, dim=1)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("packed_kernel", [False, True], ids=["dense", "varlen"])
@torch.inference_mode()
def test_npu_packed_gdn_graph_replays_ragged_prefixes_after_rejection(packed_kernel: bool) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from xllm.python.kernels_npu.gated_delta_net import mega_gdn_decode

    torch.manual_seed(37)
    device = torch.device("npu:0")
    cfg = make_linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    for parameter in layer.parameters():
        parameter.normal_(std=0.1)
    layer.norm_weight.fill_(1)
    conv = torch.zeros(4, 10, 384, device=device, dtype=torch.bfloat16)
    ssm = torch.zeros(32, 1, 128, 128, device=device)
    reference_conv = torch.zeros(4, 3, 384, device=device, dtype=torch.bfloat16)
    reference_ssm = torch.zeros(4, 1, 128, 128, device=device)
    hidden = torch.zeros(8, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    graph_inputs = LinearVerifyGraphInputs(
        torch.zeros(3, device=device, dtype=torch.int32),
        torch.ones(3, device=device, dtype=torch.int32),
        8,
        None if packed_kernel else torch.full((8,), 24, device=device, dtype=torch.int64),
        None if packed_kernel else torch.full((24,), 8, device=device, dtype=torch.int64),
        torch.zeros(4, device=device, dtype=torch.int32) if packed_kernel else None,
    )
    inputs = DSparkAdaptiveGraphInputs(
        torch.zeros(264, device=device, dtype=torch.uint8),
        torch.tensor([2 | (1 << 32)], device=device, dtype=torch.int64),
        torch.ones(3, device=device, dtype=torch.int64),
        torch.zeros(3, device=device, dtype=torch.int32),
        torch.ones(3, device=device, dtype=torch.int32),
        torch.zeros(3, 1, device=device, dtype=torch.int32),
        torch.tensor([0, 2, 5, -99], device=device, dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 1, -1, -1, -1], device=device, dtype=torch.int32),
        torch.tensor([0, 1, 0, 1, 2, -1, -1, -1], device=device, dtype=torch.int32),
        128,
        linear_state_indices=torch.tensor([1, 2, 0], device=device, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, device=device, dtype=torch.int32),
    )
    metadata = GdnVerifyMetadata(
        {0: SimpleNamespace(conv=conv, ssm=ssm)},
        graph_inputs.state_indices,
        graph_inputs.num_accepted_tokens,
        8,
        graph_inputs.packed_to_dense,
        graph_inputs.dense_to_packed,
        graph_inputs.cu_seqlens,
    )

    def forward() -> torch.Tensor:
        PreparedAclGraphRunner._refresh_linear_verify_graph(inputs, graph_inputs, 8)
        return layer._verify(hidden, metadata)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            actual = forward()
        checkpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
        for iteration, widths in enumerate(((2, 3), (4, 1))):
            if iteration:
                for row, accepted in enumerate((1, 2)):
                    reference_conv[row + 1].copy_(checkpoints[accepted - 1][0][row + 1])
                    reference_ssm[row + 1].copy_(checkpoints[accepted - 1][1][row + 1])
                inputs.num_accepted_tokens.copy_(torch.tensor([1, 2, 1], device=device, dtype=torch.int32))
            total = sum(widths)
            packed = (torch.randn(8, cfg.hidden_size, device=device) * 0.1).to(torch.bfloat16)
            packed[total:].fill_(float("nan"))
            hidden.copy_(packed)
            offsets = (0, widths[0], total)
            inputs.offsets[:3].copy_(torch.tensor(offsets, device=device, dtype=torch.int32))
            inputs.owners.fill_(-1)
            inputs.columns.fill_(-1)
            for row, width in enumerate(widths):
                inputs.owners[offsets[row] : offsets[row + 1]].fill_(row)
                inputs.columns[offsets[row] : offsets[row + 1]].copy_(torch.arange(width, device=device))
            inputs.control.fill_(2)
            graph.replay()

            qkv, a, b, z = layer._project_prefill_inputs(hidden)
            checkpoints = []
            expected = torch.zeros(8, 128, device=device, dtype=torch.bfloat16)
            indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
            for step in range(max(widths)):
                selected = torch.tensor(
                    [offsets[row] + min(step, width - 1) for row, width in enumerate(widths)], device=device
                )
                output = mega_gdn_decode(
                    qkv.index_select(0, selected),
                    z.index_select(0, selected),
                    b.index_select(0, selected),
                    a.index_select(0, selected),
                    layer.conv1d_weight,
                    reference_conv,
                    layer.A_log,
                    layer.dt_bias,
                    reference_ssm,
                    indices,
                    indices,
                    layer.norm_weight,
                )
                for row, width in enumerate(widths):
                    if step < width:
                        expected[offsets[row] + step].copy_(output[row].reshape(-1))
                checkpoints.append((reference_conv.clone(), reference_ssm.clone()))
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize(
    "batch,key_heads,value_heads", [(2, 1, 1), (17, 4, 12), (33, 4, 12), (64, 4, 12), (65, 4, 12), (128, 4, 12)]
)
@torch.inference_mode()
def test_npu_varlen_rejection_preserves_state_across_reordered_batches(
    batch: int, key_heads: int, value_heads: int
) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from xllm.python.kernels_npu.gated_delta_net import mega_gdn_decode

    torch.manual_seed(149)
    device = torch.device("npu:0")
    cfg = make_linear_config(
        linear_num_key_heads=key_heads,
        linear_num_value_heads=value_heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    for parameter in layer.parameters():
        parameter.normal_(std=0.1)
    layer.norm_weight.fill_(1)
    width, rows = 8, batch * 8
    channels = (2 * key_heads + value_heads) * 128
    hidden = torch.zeros(rows, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    inputs = DSparkAdaptiveGraphInputs(
        torch.zeros(88 * batch, device=device, dtype=torch.uint8),
        torch.tensor([batch], device=device, dtype=torch.int64),
        torch.ones(batch, device=device, dtype=torch.int64),
        torch.zeros(batch, device=device, dtype=torch.int32),
        torch.ones(batch, device=device, dtype=torch.int32),
        torch.zeros(batch, 1, device=device, dtype=torch.int32),
        torch.arange(0, rows + 1, width, device=device, dtype=torch.int32),
        torch.arange(batch, device=device, dtype=torch.int32).repeat_interleave(width),
        torch.arange(width, device=device, dtype=torch.int32).repeat(batch),
        128,
        linear_state_indices=torch.arange(1, batch + 1, device=device, dtype=torch.int32),
        num_accepted_tokens=torch.ones(batch, device=device, dtype=torch.int32),
    )
    caches = [
        SimpleNamespace(
            conv=torch.zeros(batch + 1, width + 2, channels, device=device, dtype=torch.bfloat16),
            ssm=torch.zeros((batch + 1) * width, value_heads, 128, 128, device=device),
        )
        for _ in range(2)
    ]
    graphs, outputs = [], []
    for packed_kernel, cache in zip((False, True), caches):
        graph_inputs = LinearVerifyGraphInputs(
            torch.empty(batch, device=device, dtype=torch.int32),
            torch.empty(batch, device=device, dtype=torch.int32),
            width,
            None if packed_kernel else torch.empty(rows, device=device, dtype=torch.int64),
            None if packed_kernel else torch.empty(rows, device=device, dtype=torch.int64),
            torch.empty(batch + 1, device=device, dtype=torch.int32) if packed_kernel else None,
        )
        metadata = GdnVerifyMetadata(
            {0: cache},
            graph_inputs.state_indices,
            graph_inputs.num_accepted_tokens,
            width,
            graph_inputs.packed_to_dense,
            graph_inputs.dense_to_packed,
            graph_inputs.cu_seqlens,
        )
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            for _ in range(3):
                PreparedAclGraphRunner._refresh_linear_verify_graph(inputs, graph_inputs, rows)
                layer._verify(hidden, metadata)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                PreparedAclGraphRunner._refresh_linear_verify_graph(inputs, graph_inputs, rows)
                output = layer._verify(hidden, metadata)
        torch.npu.current_stream(device).wait_stream(stream)
        graphs.append(graph)
        outputs.append(output)
    torch.npu.synchronize()
    for cache in caches:
        cache.conv.zero_()
        cache.ssm.zero_()
    oracle_conv = torch.zeros(batch + 1, 3, channels, device=device, dtype=torch.bfloat16)
    oracle_ssm = torch.zeros(batch + 1, value_heads, 128, 128, device=device)
    committed_conv, committed_ssm = oracle_conv.clone(), oracle_ssm.clone()
    accepted_by_slot = [1] * (batch + 1)
    order = list(range(1, batch + 1))
    schedules = (order, order[::-1], order[::2], order[1:] + order[:1], [], order[::-1])
    for iteration, slots in enumerate(schedules):
        lengths = [8 if iteration in (0, 2) else 2 if iteration == 1 else 1 + (slot + iteration) % 8 for slot in slots]
        accepted = [min(length, (8, 7, 1, 5)[slot % 4]) for slot, length in zip(slots, lengths)]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        total = offsets[-1]
        inputs.control.fill_(len(slots))
        inputs.linear_state_indices.zero_()
        inputs.num_accepted_tokens.fill_(1)
        inputs.offsets.fill_(-99)
        inputs.offsets[: len(offsets)].copy_(torch.tensor(offsets, device=device, dtype=torch.int32))
        inputs.owners.fill_(-1)
        inputs.columns.fill_(-1)
        hidden.fill_(float("nan"))
        hidden[:total].normal_(std=0.1)
        if slots:
            inputs.linear_state_indices[: len(slots)].copy_(torch.tensor(slots, device=device, dtype=torch.int32))
            inputs.num_accepted_tokens[: len(slots)].copy_(
                torch.tensor([accepted_by_slot[slot] for slot in slots], device=device, dtype=torch.int32)
            )
        for row, length in enumerate(lengths):
            inputs.owners[offsets[row] : offsets[row + 1]].fill_(row)
            inputs.columns[offsets[row] : offsets[row + 1]].copy_(torch.arange(length, device=device))
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        assert outputs[1][total:].count_nonzero().item() == 0

        qkv, a, b, z = layer._project_prefill_inputs(hidden)
        expected = torch.zeros(rows, layer.value_dim, device=device, dtype=torch.bfloat16)
        for column in range(max(lengths, default=0)):
            active_rows = [row for row, length in enumerate(lengths) if column < length]
            selected = torch.tensor([offsets[row] + column for row in active_rows], device=device)
            state_ids = torch.tensor([slots[row] for row in active_rows], device=device, dtype=torch.int32)
            for start in range(0, len(active_rows), 32):
                selected_chunk = selected[start : start + 32]
                state_chunk = state_ids[start : start + 32]
                decoded = mega_gdn_decode(
                    qkv.index_select(0, selected_chunk),
                    z.index_select(0, selected_chunk),
                    b.index_select(0, selected_chunk),
                    a.index_select(0, selected_chunk),
                    layer.conv1d_weight,
                    oracle_conv,
                    layer.A_log,
                    layer.dt_bias,
                    oracle_ssm,
                    state_chunk,
                    state_chunk,
                    layer.norm_weight,
                )
                expected.index_copy_(0, selected_chunk, decoded.reshape(selected_chunk.numel(), -1))
            for row in active_rows:
                if column + 1 == accepted[row]:
                    slot = slots[row]
                    committed_conv[slot].copy_(oracle_conv[slot])
                    committed_ssm[slot].copy_(oracle_ssm[slot])
        torch.testing.assert_close(outputs[1], expected, atol=0.02, rtol=0.02)
        for slot, count in zip(slots, accepted):
            for cache in caches:
                torch.testing.assert_close(
                    cache.conv[slot, count - 1 : count + 2], committed_conv[slot], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    cache.ssm[slot * width + count - 1], committed_ssm[slot], atol=0.02, rtol=0.02
                )
            oracle_conv[slot].copy_(committed_conv[slot])
            oracle_ssm[slot].copy_(committed_ssm[slot])
            accepted_by_slot[slot] = count
            # Only the accepted checkpoint and its three Conv history rows
            # may be consumed next round, even after a much longer prefix.
            conv_history = caches[1].conv[slot, count - 1 : count + 2].clone()
            checkpoint = caches[1].ssm[slot * width + count - 1].clone()
            caches[1].conv[slot].fill_(float("nan"))
            caches[1].ssm[slot * width : (slot + 1) * width].fill_(float("nan"))
            caches[1].conv[slot, count - 1 : count + 2].copy_(conv_history)
            caches[1].ssm[slot * width + count - 1].copy_(checkpoint)
