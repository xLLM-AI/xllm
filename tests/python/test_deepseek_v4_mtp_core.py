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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_npu")

from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.runners.acl_graph import StaticGraphAttentionMetadata
from xllm.python.models import deepseek_v4, deepseek_v4_mtp
from xllm.python.models.deepseek_v4 import DeepseekV4Model
from xllm.python.models.deepseek_v4_mtp import DeepseekV4MtpLayer, DeepseekV4MtpModel


def _fusion_layer() -> DeepseekV4MtpLayer:
    layer = DeepseekV4MtpLayer.__new__(DeepseekV4MtpLayer)
    nn.Module.__init__(layer)
    layer.cfg = SimpleNamespace(hidden_size=2, hc_mult=2, rms_norm_eps=0.0, hc_eps=0.0)
    layer.enorm = nn.Identity()
    layer.hnorm = nn.Identity()
    layer.e_proj = nn.Identity()
    layer.h_proj = nn.Identity()
    return layer


@pytest.mark.parametrize(
    ("previous_hidden", "expected"),
    [
        (
            torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            torch.tensor([[[11.0, 22.0], [31.0, 42.0]]]),
        ),
        (
            torch.tensor([[10.0, 20.0]]),
            torch.tensor([[[11.0, 22.0], [11.0, 22.0]]]),
        ),
    ],
)
def test_mtp_fuses_target_hidden_layouts(
    previous_hidden: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    fused = _fusion_layer()._fuse_hidden_states(torch.tensor([[1.0, 2.0]]), previous_hidden)

    torch.testing.assert_close(fused, expected)


def test_mtp_rejects_unknown_hidden_width() -> None:
    with pytest.raises(ValueError, match=r"hc_mult \* hidden_size \(4\), but got 3"):
        _fusion_layer()._fuse_hidden_states(torch.ones(1, 2), torch.ones(1, 3))


def test_target_and_mtp_hc_merge_match_independent_reference() -> None:
    cfg = SimpleNamespace(rms_norm_eps=1e-6, hc_eps=1e-6)
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    head_fn = torch.arange(36, dtype=torch.float32).reshape(3, 12) / 32
    head_base = torch.tensor([-0.25, 0.0, 0.25], dtype=torch.float32)
    head_scale = torch.tensor([0.5], dtype=torch.float32)

    target = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(target)
    target.cfg = cfg
    target.hc_head_fn = nn.Parameter(head_fn.clone())
    target.hc_head_base = nn.Parameter(head_base.clone())
    target.hc_head_scale = nn.Parameter(head_scale.clone())

    draft = DeepseekV4MtpLayer.__new__(DeepseekV4MtpLayer)
    nn.Module.__init__(draft)
    draft.cfg = cfg
    draft.hc_head_fn = nn.Parameter(head_fn.clone())
    draft.hc_head_base = nn.Parameter(head_base.clone())
    draft.hc_head_scale = nn.Parameter(head_scale.clone())

    hidden_float = hidden.to(torch.float32)
    flattened = hidden_float.flatten(-2, -1)
    reciprocal_rms = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
    mixes = flattened @ head_fn.transpose(0, 1)
    weights = torch.sigmoid(mixes * reciprocal_rms * head_scale + head_base) + cfg.hc_eps
    expected = (weights.unsqueeze(-1) * hidden_float).sum(-2).to(hidden.dtype)

    torch.testing.assert_close(target._hc_head(hidden), expected)
    torch.testing.assert_close(draft._merge_hc_hidden(hidden), expected)


def test_mtp_model_reuses_target_rotary_table_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, torch.dtype, torch.device]] = []
    rotary_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    original_build_rotary_tables = deepseek_v4._build_rotary_tables

    def build_rotary_tables(
        cfg: object,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[object, object, object]:
        calls.append((cfg, dtype, device))
        return original_build_rotary_tables(cfg, dtype, device)

    def rotary_embedding(*args: object, **kwargs: object) -> SimpleNamespace:
        rotary_calls.append((args, kwargs))
        return SimpleNamespace(cos_sin_cache=torch.empty(0))

    assert deepseek_v4_mtp._build_rotary_tables is original_build_rotary_tables
    monkeypatch.setattr(deepseek_v4_mtp, "_build_rotary_tables", build_rotary_tables)
    monkeypatch.setattr(deepseek_v4, "DeepseekV4RotaryEmbedding", rotary_embedding)
    monkeypatch.setattr(deepseek_v4_mtp, "HiddenParallelEmbedding", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(deepseek_v4_mtp, "DeepseekV4MtpLayer", lambda *_args, **_kwargs: nn.Identity())
    monkeypatch.setattr(deepseek_v4_mtp, "RMSNorm", lambda *_args, **_kwargs: nn.Identity())
    cfg = SimpleNamespace(
        tp_size=1,
        vocab_size=8,
        hidden_size=4,
        n_layers=2,
        rms_norm_eps=1e-6,
        qk_rope_head_dim=4,
        max_position_embeddings=16,
        rope_scaling_factor=1.0,
        rope_theta=10000.0,
        rope_beta_fast=32,
        rope_beta_slow=1,
        compress_rope_theta=160000.0,
    )
    dtype = torch.float32
    device = torch.device("cpu")

    model = DeepseekV4MtpModel(cfg, dtype, device)

    assert calls == [(cfg, dtype, device)]
    assert len(rotary_calls) == 3
    assert not hasattr(model, "aux_hidden_capture")


def test_mtp_dummy_embedding_matches_target_hidden_width() -> None:
    model = DeepseekV4MtpModel.__new__(DeepseekV4MtpModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hc_mult=2, hidden_size=3)
    model.norm = SimpleNamespace(weight=torch.empty(3, dtype=torch.float32))

    embedding = model.make_dummy_input_embedding(torch.tensor([1, 2], dtype=torch.int32))

    assert tuple(embedding.shape) == (2, 6)
    assert embedding.dtype == torch.float32
    assert torch.count_nonzero(embedding) == 0


def test_mtp_forward_uses_dummy_flag_from_graph_metadata() -> None:
    class Embedding(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return torch.ones((input_ids.shape[0], 2))

    class Layer(nn.Module):
        def forward(
            self,
            inputs_embeds: torch.Tensor,
            previous_hidden_states: torch.Tensor,
            positions: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del positions, cos_sin_cache, input_ids
            assert torch.count_nonzero(previous_hidden_states) == 0
            return inputs_embeds, previous_hidden_states.reshape(-1, 2, 2)

    class Norm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(
            self,
            hidden: torch.Tensor,
            residual: torch.Tensor | None,
        ) -> torch.Tensor:
            del residual
            return hidden

    model = DeepseekV4MtpModel.__new__(DeepseekV4MtpModel)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        compress_ratios=(1,),
        cp_size=1,
        hc_mult=2,
        hidden_size=2,
    )
    model.embed_tokens = Embedding()
    model.layers = nn.ModuleList([Layer()])
    model.norm = Norm()
    model.rotary = SimpleNamespace(cos_sin_cache=torch.empty(0))
    model.compress_rotary_c4 = SimpleNamespace(cos_sin_cache=torch.empty(0))
    model.compress_rotary_c128 = SimpleNamespace(cos_sin_cache=torch.empty(0))
    metadata = StaticGraphAttentionMetadata(
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(2, dtype=torch.int32),
        paged_kv_indices=torch.zeros(1, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        is_dummy=True,
    )
    backend = SimpleNamespace(
        reset_forward=lambda _metadata: None,
        attach_rope_tables=lambda *_args, **_kwargs: None,
        prepare_dsa_metadata_for_forward=lambda _metadata: None,
        select_dsa_layer_rope=lambda *_args: None,
    )

    with forward_context(ForwardContext(backend, torch.device("cpu"), metadata, [])):
        hidden, aux_hidden = model(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            None,
        )

    assert hidden.shape == (1, 2)
    assert aux_hidden.shape == (1, 4)
    assert torch.count_nonzero(aux_hidden) == 0
