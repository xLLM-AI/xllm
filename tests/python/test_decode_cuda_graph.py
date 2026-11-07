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

"""CUDA graph output contracts and Qwen3 auxiliary-output regressions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import DecodeCudaGraphRunner
from xllm.python.models.qwen3 import Qwen3ForCausalLM
from xllm.python.platform import current_platform


@pytest.mark.parametrize("kind", ["tensor", "aux", "none_none", "aux_none", "none_topk", "aux_topk"])
def test_cuda_graph_output_snapshots_preserve_structure_and_ownership(kind: str) -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    aux_hidden = hidden + 100
    topk = torch.arange(8, dtype=torch.int64).reshape(4, 1, 2)
    outputs: dict[str, ModelExecutionOutput] = {
        "tensor": hidden,
        "aux": (hidden, aux_hidden),
        "none_none": (hidden, None, None),
        "aux_none": (hidden, aux_hidden, None),
        "none_topk": (hidden, None, topk),
        "aux_topk": (hidden, aux_hidden, topk),
    }
    output = outputs[kind]
    actual = DecodeCudaGraphRunner._slice_output(output, 3)
    assert isinstance(actual, tuple) == isinstance(output, tuple)
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = output if isinstance(output, tuple) else (output,)
    assert len(actual_items) == len(expected_items)
    for result, reference in zip(actual_items, expected_items):
        if reference is None:
            assert result is None
            continue
        assert result is not None
        assert result.shape == (3, *reference.shape[1:])
        assert result.dtype == reference.dtype
        assert result.device == reference.device
        expected = reference[:3].clone()
        reference.add_(1000)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layers_to_capture", [(), (1, 0)])
@torch.inference_mode()
def test_qwen3_cuda_graph_preserves_outputs_across_replays(layers_to_capture: tuple[int, ...]) -> None:
    if not current_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    pytest.importorskip("flashinfer")
    device = torch.device("cuda", torch.cuda.current_device())
    config = {
        "model_type": "qwen3",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "vocab_size": 256,
        "max_position_embeddings": 128,
        "layers_to_capture": layers_to_capture,
        "dtype": "bfloat16",
        "device": str(device),
    }
    model = Qwen3ForCausalLM(config).eval()
    generator = torch.Generator(device=device).manual_seed(20261020)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.02, generator=generator)
    executors = []
    for backend in ("off", "cudagraphs"):
        executor = ModelExecutor(model, {**config, "python_graph_backend": backend}, max_seqs_per_batch=4)
        executor.bind_kv_caches(
            [
                LayerCache(
                    torch.zeros(4, 128, 2, 64, dtype=torch.bfloat16, device=device),
                    torch.zeros(4, 128, 2, 64, dtype=torch.bfloat16, device=device),
                )
                for _ in range(2)
            ]
        )
        executors.append(executor)
    graph_runner = executors[1].decode_graph_runner
    assert isinstance(graph_runner, DecodeCudaGraphRunner)
    lengths = [0, 0, 0]
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step, requests in enumerate(((0, 1, 2), (0,), (2, 0), (1, 2, 0), (2, 1, 0))):
        batch_size = len(requests)
        tokens = torch.tensor([11 + request + step for request in requests], dtype=torch.int32, device=device)
        positions = torch.tensor([lengths[request] for request in requests], dtype=torch.int32, device=device)
        cumulative_lengths = [0]
        per_seq_lengths = []
        for request in requests:
            lengths[request] += 1
            cumulative_lengths.append(cumulative_lengths[-1] + lengths[request])
            per_seq_lengths.append(lengths[request])
        # Give each request a separate reversed physical page; rows may reorder.
        pages = torch.tensor([3 - request for request in requests], dtype=torch.int32, device=device)
        metadata = SimpleNamespace(
            is_prefill=False,
            is_chunked_prefill=False,
            slot_mapping=pages * 128 + positions,
            kv_cu_seq_lens=torch.tensor(cumulative_lengths, dtype=torch.int32, device=device),
            kv_seq_lens_host=torch.tensor(per_seq_lengths, dtype=torch.int32),
            paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
            paged_kv_indices=pages,
            paged_kv_last_page_len=positions + 1,
        )
        assert graph_runner.can_execute(tokens, metadata)
        expected = executors[0].execute(tokens, positions, metadata)
        actual = executors[1].execute(tokens, positions, metadata)
        if layers_to_capture:
            assert isinstance(expected, tuple) and isinstance(actual, tuple)
            assert len(actual) == 2
            assert actual[1].shape == (batch_size, config["hidden_size"] * len(layers_to_capture))
            pairs = zip(actual, expected)
        else:
            assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
            pairs = [(actual, expected)]
        for result, reference in pairs:
            assert result.shape[0] == batch_size
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
            retained.append((result, reference.clone()))
    # Reusing the same bucket must not overwrite either previously returned tensor.
    for result, reference in retained:
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
