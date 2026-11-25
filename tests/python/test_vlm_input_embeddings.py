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

"""Per-step visual embeddings must survive interleaved request preparation."""

from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python import kernels
from xllm.python.attention.backend import AttentionBackend
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.runners.eager import EagerRunner
from xllm.python.models.deepseek_ocr import DeepseekOCRConfig, DeepseekOCRForCausalLM, DeepseekOCRModel
from xllm.python.models.glm5_next import Glm5NextConfig, Glm5NextForCausalLM, Glm5NextModel
from xllm.python.models.glm5_next_vl import Glm5NextVLModel
from xllm.python.models.joyai_asr import JoyaiASRForConditionalGeneration


@pytest.fixture(autouse=True)
def _cpu_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    native_rms_norm = kernels.rms_norm
    native_add_rms_norm = kernels.fused_add_rms_norm

    def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if value.device.type == "cpu":
            return F.rms_norm(value, (value.shape[-1],), weight, eps)
        return native_rms_norm(value, weight, eps)

    def add_rms_norm(
        value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.device.type == "cpu":
            combined = value + residual
            return F.rms_norm(combined, (combined.shape[-1],), weight, eps), combined
        return native_add_rms_norm(value, residual, weight, eps)

    monkeypatch.setattr(kernels, "rms_norm", rms_norm)
    monkeypatch.setattr(kernels, "fused_add_rms_norm", add_rms_norm)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("npu:0", marks=pytest.mark.skipif(not hasattr(torch, "npu"), reason="NPU runtime unavailable")),
    ]
)
def device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(request.param)


class _ZeroResidualUpdate(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del residual, rope_cos, rope_sin
        return torch.zeros_like(hidden), hidden


def _make_vlm(kind: str, device: torch.device) -> DeepseekOCRForCausalLM | Glm5NextVLModel:
    # Keep the real embedding, decoder forward, mHC and norm. Vision encoding
    # and attention weights are independent of the per-step input contract.
    if kind == "ocr":
        model = DeepseekOCRForCausalLM.__new__(DeepseekOCRForCausalLM)
        nn.Module.__init__(model)
        model.model = DeepseekOCRModel(
            DeepseekOCRConfig(
                hidden_size=4, n_layers=0, n_heads=1, head_dim=4, vocab_size=16, max_position_embeddings=16
            ),
            dtype=torch.float32,
            device=device,
        )
        # OCR's final fused norm requires the first layer's residual stream.
        model.model.layers = nn.ModuleList([_ZeroResidualUpdate()])
    else:
        model = Glm5NextVLModel.__new__(Glm5NextVLModel)
        nn.Module.__init__(model)
        model.model = Glm5NextModel(
            Glm5NextConfig(hidden_size=4, n_layers=0, n_heads=1, vocab_size=16),
            dtype=torch.float32,
            device=device,
        )
        model.language_model = model.model
        model._hidden_rot = None
    model.image_token_id = 3
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(torch.arange(64, dtype=torch.float32, device=device).view(16, 4) + 1)
    return model


def _make_runner(model: nn.Module, device: torch.device) -> EagerRunner:
    backend = create_autospec(AttentionBackend, instance=True)
    backend.is_mla = False
    return EagerRunner(model, backend, device)


@pytest.mark.parametrize("kind", ["ocr", "glm"])
def test_prepared_visual_inputs_are_independent_and_do_not_leak_into_text(kind: str, device: torch.device) -> None:
    model = _make_vlm(kind, device)
    runner = _make_runner(model.model, device)
    tokens = torch.tensor([1, 3, 2], device=device)
    positions = torch.arange(3, device=device)
    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False)
    text_output = runner.execute(tokens, positions, metadata)
    image_a = torch.tensor([[1.0, 3.0, 2.0, 7.0]], device=device)
    image_b = torch.tensor([[7.0, 2.0, 4.0, 1.0]], device=device)
    inputs_a = model.get_input_embeddings(tokens, image_a)
    assert inputs_a.deepstack_input_embeds is None
    embedding_a = inputs_a.input_embedding
    assert embedding_a.shape == (tokens.numel(), 4)
    embedding_a_saved = embedding_a.clone()
    embedding_b = model.get_input_embeddings(tokens, image_b).input_embedding
    torch.testing.assert_close(embedding_a, embedding_a_saved)
    torch.testing.assert_close(embedding_a.reshape(-1, 4)[1:2], image_a)
    torch.testing.assert_close(embedding_b.reshape(-1, 4)[1:2], image_b)

    # Preparing a later request must not replace a pending request's inputs,
    # and an unconsumed visual input must not affect an ordinary text step.
    torch.testing.assert_close(runner.execute(tokens, positions, metadata), text_output)
    for embedding in (embedding_a, embedding_b, embedding_a):
        output = runner.execute(tokens, positions, metadata, embedding)
        expected = F.rms_norm(embedding.reshape(-1, 4), (4,), model.model.norm.weight, model.model.norm.eps)
        torch.testing.assert_close(output, expected)
    torch.testing.assert_close(runner.execute(tokens, positions, metadata), text_output)


@pytest.mark.parametrize("kind", ["ocr", "glm"])
def test_explicit_embeddings_accept_chunk_local_rows_without_image_placeholders(
    kind: str, device: torch.device
) -> None:
    model = _make_vlm(kind, device)
    runner = _make_runner(model.model, device)
    tokens = torch.tensor([1, 2], device=device)
    positions = torch.arange(2, device=device)
    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=True)
    # A later prefill chunk has only text rows. It must keep token embeddings
    # even while another prepared image input is still owned by its caller.
    model.get_input_embeddings(torch.tensor([3], device=device), torch.tensor([[3.0, 1.0, 4.0, 2.0]], device=device))
    embedding = model.get_input_embeddings(tokens).input_embedding
    torch.testing.assert_close(
        runner.execute(tokens, positions, metadata, embedding),
        runner.execute(tokens, positions, metadata),
    )


def test_glm_accepts_flat_runner_embeddings_and_preserves_text_wrapper_mask(device: torch.device) -> None:
    model = _make_vlm("glm", device)
    tokens = torch.tensor([1, 3, 2], device=device)
    positions = torch.arange(3, device=device)
    metadata = SimpleNamespace(is_prefill=True, is_chunked_prefill=False)
    runner = _make_runner(model.model, device)
    embedding = model.get_input_embeddings(tokens, torch.tensor([[1.0, 4.0, 2.0, 3.0]], device=device)).input_embedding
    torch.testing.assert_close(
        runner.execute(tokens, positions, metadata, embedding),
        runner.execute(tokens, positions, metadata, embedding.unsqueeze(0)),
    )
    text_wrapper = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(text_wrapper)
    text_wrapper.model = model.model
    text_wrapper.lm_head = nn.Identity()
    with forward_context(ForwardContext(runner.attention_backend, runner.device, metadata, [])):
        masked_output = text_wrapper(tokens, positions, torch.tensor([[True, True, False]], device=device))
    torch.testing.assert_close(masked_output, runner.execute(tokens, positions, metadata))


@pytest.mark.parametrize("has_audio", [False, True])
@pytest.mark.parametrize("mask_rows", [0, 2])
def test_audio_embeddings_preserve_chunk_mask_and_pending_inputs(has_audio: bool, mask_rows: int) -> None:
    model = JoyaiASRForConditionalGeneration.__new__(JoyaiASRForConditionalGeneration)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding.from_pretrained(torch.arange(64).view(16, 4).float())
    tokens = torch.tensor([1, 3, 3, 2])
    mask = torch.tensor([False, True, True, False]) if mask_rows else torch.zeros(4, dtype=torch.bool)
    audio = torch.full((mask_rows, 4), 100.0) if has_audio else None

    inputs = model.get_input_embeddings(tokens, audio_embeds=audio, audio_mask=mask)
    expected = model.model.embed_tokens(tokens).clone()
    if has_audio:
        expected[mask] = 100.0
    torch.testing.assert_close(inputs.input_embedding, expected)
    assert inputs.deepstack_input_embeds is None

    model.get_input_embeddings(tokens, audio_embeds=torch.full((2, 4), 200.0), audio_mask=tokens == 3)
    torch.testing.assert_close(inputs.input_embedding, expected)
    torch.testing.assert_close(model.get_input_embeddings(tokens).input_embedding, model.model.embed_tokens(tokens))
