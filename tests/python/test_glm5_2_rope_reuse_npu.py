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

"""Real NPU tests for shared GLM RoPE coefficients and their consumers."""

from __future__ import annotations

from itertools import product
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from xllm.python import kernels


@pytest.fixture(scope="module")
def runtime() -> tuple[Any, Any]:
    from xllm.python.models import deepseek_v32, glm5_2

    torch.npu.set_device(0)
    torch.set_grad_enabled(False)
    torch.manual_seed(2360)
    return glm5_2, deepseek_v32


@pytest.fixture(scope="module")
def rotary(runtime: tuple[Any, Any]) -> torch.nn.Module:
    glm, _ = runtime
    return glm.Glm52YarnRotaryEmbedding(
        64, 1048576, 1.0, 8000000.0, 32, 1, 1.0, 1.0, dtype=torch.bfloat16, device=torch.device("npu:0")
    )


@pytest.fixture(scope="module")
def raw_cache(rotary: torch.nn.Module) -> torch.Tensor:
    return rotary.cos_sin_cache.cpu().clone()


def _positions(rows: int, step: int = 0) -> torch.Tensor:
    # Noncontiguous input, repeated positions, zero/padding and the last valid row.
    values = torch.tensor([0, 32768, 32768, 1048575, 1, 65536, 17, 0], dtype=torch.int64)
    values = values.roll(step).repeat((rows + 7) // 8)[:rows]
    return torch.stack((values, values), dim=-1).npu()[:, 0]


def _reference_coefficients(
    raw_cache: torch.Tensor, positions: torch.Tensor, actual: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    # CPU indexing/splitting only; no production gather/expand or rotary forward.
    cos, sin = raw_cache.index_select(0, positions.cpu().to(torch.int64)).split(32, dim=-1)
    expected = (
        cos,
        sin,
        torch.cat((cos, cos), -1).reshape(-1, 1, 1, 64),
        torch.cat((sin, sin), -1).reshape(-1, 1, 1, 64),
    )
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == reference.shape and value.stride() == reference.stride()
        assert value.dtype == reference.dtype
        torch.testing.assert_close(value.cpu(), reference, rtol=0, atol=0)
    return expected


def _reference_indexer_rope(
    value: torch.Tensor,
    coefficients: tuple[torch.Tensor, ...],
    interleaved: bool,
) -> torch.Tensor:
    # Feed independently selected rows to the primitive so BF16 arithmetic stays
    # identical; the production coefficient selection and consumer helpers are
    # deliberately not part of this reference.
    value = value.clone()
    if interleaved:
        cos, sin = (coefficient.to(value.device).view(-1, 64) for coefficient in coefficients[2:])
        kernels.npu_inplace_partial_rotary_mul(value, cos, sin, 0, 64)
        return value
    cos, sin = (coefficient.to(value.device).unsqueeze(1) for coefficient in coefficients[:2])
    first, second = value[..., :32], value[..., 32:64]
    return torch.cat((first * cos - second * sin, second * cos + first * sin, value[..., 64:]), dim=-1)


@pytest.mark.parametrize(
    "rows,interleaved,quantized",
    [*product((1, 2, 4), (False, True), (False, True)), (8, False, False), (16, False, False)],
)
def test_indexer_query_and_cache_equal_raw_cache_reference(
    runtime: tuple[Any, Any],
    rotary: torch.nn.Module,
    raw_cache: torch.Tensor,
    interleaved: bool,
    quantized: bool,
    rows: int,
) -> None:
    glm, ds = runtime
    cfg = glm.Glm52Config(
        hidden_size=128,
        q_lora_rank=128,
        index_n_heads=4,
        index_head_dim=128,
        qk_rope_head_dim=64,
        index_topk=1,
        indexer_rope_interleave=interleaved,
    )
    indexer = glm.Glm52Indexer(cfg, torch.bfloat16, torch.device("npu:0"))
    indexer.wq_b = torch.nn.Linear(128, 4 * 128, bias=False, dtype=torch.bfloat16, device="npu:0")
    hidden = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu:0")
    positions = _positions(rows, 1)
    coefficients = rotary(positions.contiguous())
    reference_coefficients = _reference_coefficients(raw_cache, positions, coefficients)
    cos_sin = ds._select_indexer_query_cos_sin(interleaved, *coefficients, None)
    q = indexer.wq_b(hidden).view(rows, 4, 128)
    k = indexer.k_norm(indexer.wk(hidden))
    expected_q = _reference_indexer_rope(q, reference_coefficients, interleaved)
    expected_k = _reference_indexer_rope(k[:, None, :], reference_coefficients, interleaved).squeeze(1)
    expected_scale = None
    if quantized:
        expected_k = torch.matmul(expected_k, indexer.hadamard) * 128**-0.5
        expected_k, expected_scale = glm.kernels.dynamic_quant(expected_k)
        expected_scale = expected_scale.unsqueeze(-1).to(torch.float16)
    written: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    ctx = SimpleNamespace(
        cp_context=None,
        index_cache=torch.empty(rows, 128, dtype=torch.int8 if quantized else hidden.dtype, device=hidden.device),
        index_cache_scale=torch.empty(rows, 1, device=hidden.device) if quantized else None,
        update_index_cache=lambda key, scale: written.append((key.clone(), None if scale is None else scale.clone())),
    )
    before = tuple(coefficient.clone() for coefficient in coefficients)
    actual_q = indexer._project_query(hidden, cos_sin)
    indexer._update_index_cache(hidden, ctx, cos_sin)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_q[..., 64:], q[..., 64:], rtol=0, atol=0)
    assert len(written) == 1
    torch.testing.assert_close(written[0][0], expected_k, rtol=0, atol=0)
    if quantized:
        torch.testing.assert_close(written[0][1], expected_scale, rtol=0, atol=0)
    else:
        assert written[0][1] is None
        torch.testing.assert_close(written[0][0][..., 64:], k[..., 64:], rtol=0, atol=0)
    if interleaved:
        value = q.clone()
        output = indexer._apply_interleaved_rope(value, cos_sin)
        assert output.data_ptr() == value.data_ptr()
        torch.testing.assert_close(output, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(output[..., 64:], q[..., 64:], rtol=0, atol=0)
    for coefficient, unchanged in zip(coefficients, before, strict=True):
        torch.testing.assert_close(coefficient, unchanged, rtol=0, atol=0)


@pytest.mark.parametrize("interleaved", [False, True])
def test_aclgraph_changed_positions_and_alternating_target_draft_buckets(
    runtime: tuple[Any, Any], rotary: torch.nn.Module, raw_cache: torch.Tensor, interleaved: bool
) -> None:
    glm, ds = runtime
    # Separate graph allocations; target and each draft step own their coefficients.
    entries = []
    for rows, consumers in ((4, 3), (1, 1), (2, 3), (8, 1)):
        positions = _positions(rows).contiguous()
        query = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device="npu:0")
        key = torch.randn(rows, 128, dtype=torch.bfloat16, device="npu:0")
        cfg = glm.Glm52Config(
            hidden_size=128,
            q_lora_rank=128,
            index_n_heads=4,
            index_head_dim=128,
            qk_rope_head_dim=64,
            index_topk=1,
            indexer_rope_interleave=interleaved,
        )
        indexer = glm.Glm52Indexer(cfg, torch.bfloat16, torch.device("npu:0"))
        indexer.wk = torch.nn.Identity()
        cache = torch.zeros_like(key)
        ctx = SimpleNamespace(
            cp_context=None,
            index_cache=cache,
            index_cache_scale=None,
            update_index_cache=lambda value, _scale, cache=cache: cache.copy_(value),
        )

        def forward(
            positions: torch.Tensor = positions,
            query: torch.Tensor = query,
            key: torch.Tensor = key,
            indexer: torch.nn.Module = indexer,
            ctx: Any = ctx,
            consumers: int = consumers,
        ) -> tuple[torch.Tensor, ...]:
            coefficients = rotary(positions)
            cos_sin = ds._select_indexer_query_cos_sin(interleaved, *coefficients, None)
            value = query.clone()
            for _ in range(consumers):
                attention_q = ds._interleave_rope_with(value[..., :64], *coefficients[2:])
                if interleaved:
                    value = indexer._apply_interleaved_rope(value, cos_sin)
                else:
                    value = torch.cat(
                        (ds._apply_half_rope_with_cos_sin(value[..., :64], *cos_sin), value[..., 64:]), -1
                    )
                indexer._update_index_cache(key, ctx, cos_sin)
            return (*coefficients, attention_q, value, ctx.index_cache.clone())

        for _ in range(3):
            forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = forward()
        entries.append((positions, query, key, indexer, ctx, consumers, graph, outputs))

    # Update static positions in place, then revisit every captured graph. Retain
    # all graph outputs so allocator reuse cannot hide cross-entry corruption.
    snapshots = {}
    for step in range(1, 4):
        for entry_id in (0, 1, 2, 3, 1, 0):
            positions, query, key, indexer, ctx, consumers, graph, outputs = entries[entry_id]
            positions.copy_(_positions(positions.shape[0], step))
            graph.replay()
            torch.npu.synchronize()
            reference_coefficients = _reference_coefficients(raw_cache, positions, outputs[:4])
            reference_cos, reference_sin = (coefficient.to(query.device) for coefficient in reference_coefficients[2:])
            expected_value = query.clone()
            for _ in range(consumers):
                expected_attention = kernels.interleaved_rotary_embedding(
                    expected_value[..., :64], reference_cos, reference_sin
                )
                expected_value = _reference_indexer_rope(expected_value, reference_coefficients, interleaved)
            expected_cache = _reference_indexer_rope(
                indexer.k_norm(key)[:, None, :], reference_coefficients, interleaved
            ).squeeze(1)
            for actual, reference in zip(
                outputs[4:], (expected_attention, expected_value, expected_cache), strict=True
            ):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            torch.testing.assert_close(outputs[5][..., 64:], query[..., 64:], rtol=0, atol=0)
            for previous_id, previous in snapshots.items():
                if previous_id != entry_id:
                    for actual, reference in zip(entries[previous_id][-1], previous, strict=True):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            snapshots[entry_id] = tuple(value.clone() for value in outputs)
