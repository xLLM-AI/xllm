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

"""Real NPU tests: XLLM_RUN_NPU_TESTS=1 pytest --noconftest <this file>.

XLLM_NATIVE_LIBRARY must name the matching built xllm_export library. The
ordinary Python suite uses kernel stubs; this suite deliberately bypasses them.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("XLLM_RUN_NPU_TESTS") != "1", reason="explicit real NPU suite")


@pytest.fixture(scope="module")
def runtime() -> tuple[Any, Any]:
    import torch_npu

    torch.ops.load_library(os.environ["XLLM_NATIVE_LIBRARY"])
    import xllm.python

    xllm.python.initialize_runtime()
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


def _positions(rows: int, step: int = 0) -> torch.Tensor:
    # Noncontiguous input, repeated positions, zero/padding and the last valid row.
    values = torch.tensor([0, 32768, 32768, 1048575, 1, 65536, 17, 0], dtype=torch.int64)
    values = values.roll(step).repeat((rows + 7) // 8)[:rows]
    return torch.stack((values, values), dim=-1).npu()[:, 0]


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_coefficients_equal_legacy_gather(runtime: tuple[Any, Any], rotary: torch.nn.Module, rows: int) -> None:
    _, ds = runtime
    positions = _positions(rows)
    actual = rotary(positions.to(torch.int64).contiguous())
    expected = (
        *ds._gather_half_rope_cos_sin(rotary.cos_sin_cache, positions),
        *ds._gather_interleave_cos_sin(rotary.cos_sin_cache, positions),
    )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert actual[0].stride() == (64, 1)
    assert actual[2].shape == (rows, 1, 1, 64)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_indexer_query_and_cache_equal_legacy(
    runtime: tuple[Any, Any], rotary: torch.nn.Module, interleaved: bool, quantized: bool, rows: int
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
    cos_sin = glm._select_indexer_query_cos_sin(interleaved, *coefficients, None)
    q = indexer.wq_b(hidden).view(rows, 4, 128)
    k = indexer.k_norm(indexer.wk(hidden))

    def legacy(value: torch.Tensor) -> torch.Tensor:
        if interleaved:
            cos, sin = ds._gather_interleave_cos_sin(rotary.cos_sin_cache, positions)
            glm.kernels.npu_inplace_partial_rotary_mul(value, cos.view(rows, 64), sin.view(rows, 64), 0, 64)
            return value
        rotated = ds._apply_half_rope(rotary.cos_sin_cache, value[..., :64], positions)
        return torch.cat((rotated, value[..., 64:]), dim=-1)

    expected_q = legacy(q.clone())
    expected_k = legacy(k[:, None, :].clone()).squeeze(1)
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
    before = tuple(coefficient.clone() for coefficient in cos_sin)
    actual_q = indexer._project_query(hidden, cos_sin)
    indexer._update_index_cache(hidden, ctx, cos_sin)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(written[0][0], expected_k, rtol=0, atol=0)
    if quantized:
        torch.testing.assert_close(written[0][1], expected_scale, rtol=0, atol=0)
    for coefficient, unchanged in zip(cos_sin, before):
        torch.testing.assert_close(coefficient, unchanged, rtol=0, atol=0)
    if interleaved:
        value = q.clone()
        output = indexer._apply_interleaved_rope(value, cos_sin)
        assert output.data_ptr() == value.data_ptr()
        torch.testing.assert_close(output[..., 64:], q[..., 64:], rtol=0, atol=0)


@pytest.mark.parametrize("interleaved", [False, True])
def test_aclgraph_changed_positions_and_alternating_target_draft_buckets(
    runtime: tuple[Any, Any], rotary: torch.nn.Module, interleaved: bool
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
            cos_sin = glm._select_indexer_query_cos_sin(interleaved, *coefficients, None)
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
        entries.append((positions, forward, graph, outputs))

    # Update static positions in place, then revisit every captured graph. Retain
    # all graph outputs so allocator reuse cannot hide cross-entry corruption.
    snapshots = {}
    for step in range(1, 4):
        for entry_id in (0, 1, 2, 3, 1, 0):
            positions, forward, graph, outputs = entries[entry_id]
            positions.copy_(_positions(positions.shape[0], step))
            graph.replay()
            torch.npu.synchronize()
            expected = forward()
            torch.npu.synchronize()
            for actual, reference in zip(outputs, expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for previous_id, previous in snapshots.items():
                if previous_id != entry_id:
                    for actual, reference in zip(entries[previous_id][3], previous):
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            snapshots[entry_id] = tuple(value.clone() for value in outputs)
