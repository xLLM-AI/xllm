# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU regression test for the fused GLM-5.3-Flash kPool indexer."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import NamedTuple

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="requires Ascend NPU")
pytest.importorskip("cann_ops_transformer", reason="requires CANN PoolKeyIndexer")
if not torch.npu.is_available():
    pytest.skip("requires an available Ascend NPU", allow_module_level=True)

# Load only the wrapper under test. Importing the kernels_npu package would
# also load xllm_ops FakeTensor registrations, which require the xLLM worker
# extension even though this test exercises only the CANN operator.
_SPARSE_ATTENTION_PATH = (
    Path(__file__).parents[2] / "xllm/python/kernels_npu/sparse_attention.py"
)
_SPARSE_ATTENTION_SPEC = spec_from_file_location(
    "_xllm_npu_sparse_attention", _SPARSE_ATTENTION_PATH
)
if _SPARSE_ATTENTION_SPEC is None or _SPARSE_ATTENTION_SPEC.loader is None:
    raise ImportError(f"cannot load {_SPARSE_ATTENTION_PATH}")
_SPARSE_ATTENTION_MODULE = module_from_spec(_SPARSE_ATTENTION_SPEC)
_SPARSE_ATTENTION_SPEC.loader.exec_module(_SPARSE_ATTENTION_MODULE)
pool_key_indexer = _SPARSE_ATTENTION_MODULE.pool_key_indexer


class _PoolKeyIndexerCase(NamedTuple):
    """One layout/shape/dtype combination for the CANN operator."""

    batch_size: int
    query_length: int
    num_heads: int
    pool_count: int
    pool_size: int
    topk: int
    dtype: torch.dtype
    pool_tail_k: tuple[int, ...]
    return_value: bool = True
    layout_q: str = "BSND"
    layout_k: str = "BSND"
    mask_mode: int = 3
    compare_reference: bool = True


_CASES = {
    "baseline_fp16": _PoolKeyIndexerCase(
        1, 64, 8, 256, 16, 128, torch.float16, (0,),
    ),
    "bsnd_pool_size_1": _PoolKeyIndexerCase(
        1, 64, 8, 256, 1, 128, torch.float16, (0,),
    ),
    "pool_size_4_tail_3": _PoolKeyIndexerCase(
        1, 64, 8, 256, 4, 128, torch.float16, (3,),
    ),
    "bsnd_pool_size_16_mask_0": _PoolKeyIndexerCase(
        1, 64, 8, 256, 16, 64, torch.float16, (0,), True, "BSND", "BSND", 0, False
    ),
    "tnd_pool_size_1": _PoolKeyIndexerCase(
        2, 80, 8, 320, 1, 64, torch.float16, (0, 0), False, "TND", "TND", 0, False
    ),
    "tnd_pool_size_4": _PoolKeyIndexerCase(
        2, 80, 8, 320, 4, 64, torch.float16, (0, 0), False, "TND", "TND", 0, False
    ),
    "pa_bbnd_pool_size_16": _PoolKeyIndexerCase(
        2, 64, 8, 32, 16, 128, torch.float16, (0, 0), False, "BSND", "PA_BBND", 0, False
    ),
    "pa_bbnd_pool_size_4": _PoolKeyIndexerCase(
        2, 64, 8, 32, 4, 64, torch.float16, (0, 0), False, "BSND", "PA_BBND", 0, False
    ),
}


def _single_op_pool_key_indexer(
    query: torch.Tensor,
    pool_key: torch.Tensor,
    weights: torch.Tensor,
    pool_tail_k: torch.Tensor,
    topk: int,
    pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original small-op formula for dense BSND inputs."""
    query = query.cpu()
    pool_key = pool_key.cpu()
    weights = weights.cpu()
    pool_tail_k = pool_tail_k.cpu()

    output_width = topk + pool_size - 1
    sparse_count = topk // pool_size
    indices: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    scale = 1.0 / math.sqrt(query.size(-1))
    for batch in range(query.size(0)):
        q_batch = query[batch].float()
        w_batch = weights[batch].float()
        pool_key_batch = pool_key[batch, :, 0].float()
        scores = torch.einsum("snd,pd->snp", q_batch, pool_key_batch)
        scores = torch.einsum(
            "sn,snp->sp", w_batch, scores.clamp_min(0.0) * scale
        )
        length = pool_key_batch.size(0) * pool_size + int(pool_tail_k[batch])
        batch_indices = torch.full(
            (q_batch.size(0), output_width),
            -1,
            dtype=torch.int32,
        )
        batch_values = torch.full(
            (q_batch.size(0), sparse_count),
            float("-inf"),
            dtype=torch.float32,
        )
        for row in range(q_batch.size(0)):
            visible = max(
                0,
                min(
                    pool_key_batch.size(0),
                    (length - q_batch.size(0) + row + 1) // pool_size,
                ),
            )
            selected = min(visible, sparse_count)
            if selected:
                order = torch.argsort(
                    scores[row, :visible], descending=True, stable=True
                )[:selected]
                expanded = (
                    order[:, None] * pool_size
                    + torch.arange(pool_size)[None, :]
                ).reshape(-1)
                batch_indices[row, : expanded.numel()] = expanded.to(torch.int32)
                batch_values[row, :selected] = scores[row, order]
            tail = int(pool_tail_k[batch])
            visible_tail = max(
                0, min(tail, length - q_batch.size(0) + row - topk + 1)
            )
            if visible_tail:
                batch_indices[row, topk : topk + visible_tail] = torch.arange(
                    length - tail,
                    length - tail + visible_tail,
                    dtype=torch.int32,
                )
        indices.append(batch_indices)
        values.append(batch_values)
    return torch.stack(indices), torch.stack(values)


def _run_case(case: _PoolKeyIndexerCase) -> None:
    batch_size = case.batch_size
    query_length = case.query_length
    pool_size = case.pool_size
    pool_count = case.pool_count
    pool_tail_k = torch.tensor(
        case.pool_tail_k, dtype=torch.int32, device="npu"
    )
    head_dim = 128
    num_heads = case.num_heads

    torch.manual_seed(
        17 + batch_size + query_length + num_heads + pool_count + pool_size
    )
    actual_seq_q = None
    actual_seq_k = None
    block_table = None
    if case.layout_q == "TND":
        query_lengths = (32, 48)
        token_key_lengths = (128, 192)
        key_lengths = tuple(length // pool_size for length in token_key_lengths)
        query = torch.randn(
            sum(query_lengths), num_heads, head_dim,
            dtype=case.dtype, device="npu"
        )
        weights = torch.randn(
            sum(query_lengths), num_heads,
            dtype=case.dtype, device="npu"
        )
        pool_key = torch.randn(
            sum(key_lengths), 1, head_dim,
            dtype=case.dtype, device="npu"
        )
        actual_seq_q = torch.tensor(
            [0, query_lengths[0], sum(query_lengths)],
            dtype=torch.int32, device="npu"
        )
        actual_seq_k = torch.tensor(
            [0, key_lengths[0], sum(key_lengths)],
            dtype=torch.int32, device="npu"
        )
        output_rows = sum(query_lengths)
        expected_indices_shape = (
            output_rows, case.topk + pool_size - 1
        )
    elif case.layout_k == "PA_BBND":
        query = torch.randn(
            batch_size, query_length, num_heads, head_dim,
            dtype=case.dtype, device="npu"
        )
        weights = torch.randn(
            batch_size, query_length, num_heads,
            dtype=case.dtype, device="npu"
        )
        # CANN PA_BBND requires the physical block size to be a multiple of 16;
        # it is independent of the logical kPool pooling window.
        block_size = 16
        pool_counts = (pool_count // 2, pool_count // 2)
        block_count = pool_count // (2 * block_size)
        pool_key = torch.randn(
            block_count, block_size, 1, head_dim,
            dtype=case.dtype, device="npu"
        )
        actual_seq_k = torch.tensor(
            pool_counts, dtype=torch.int32, device="npu"
        )
        block_table = torch.tensor(
            [list(range(block_count))] * batch_size,
            dtype=torch.int32, device="npu"
        )
        expected_indices_shape = (
            batch_size, query_length, case.topk + pool_size - 1
        )
    else:
        query = torch.randn(
            batch_size, query_length, num_heads, head_dim,
            dtype=case.dtype, device="npu"
        )
        weights = torch.randn(
            batch_size, query_length, num_heads,
            dtype=case.dtype, device="npu"
        )
        pool_key = torch.randn(
            batch_size, pool_count, 1, head_dim,
            dtype=case.dtype, device="npu"
        )
        expected_indices_shape = (
            batch_size, query_length, case.topk + pool_size - 1
        )

    fused_indices, fused_values = pool_key_indexer(
        query,
        pool_key,
        weights,
        pool_tail_k,
        case.topk,
        pool_size,
        return_value=case.return_value,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
        layout_q=case.layout_q,
        layout_k=case.layout_k,
        mask_mode=case.mask_mode,
    )
    # Synchronize the fused result before running the CPU reference so device
    # kernel failures are reported at the operator boundary.
    fused_indices = fused_indices.cpu()
    fused_values = fused_values.cpu()
    assert fused_indices.shape == expected_indices_shape
    assert fused_indices.dtype == torch.int32
    expected_values_shape = (
        (expected_indices_shape[:-1] + (case.topk // pool_size,))
        if case.return_value
        else (0,)
    )
    assert fused_values.shape == expected_values_shape
    assert fused_values.dtype == torch.float32
    if case.compare_reference:
        ref_indices, ref_values = _single_op_pool_key_indexer(
            query,
            pool_key,
            weights,
            pool_tail_k,
            case.topk,
            pool_size,
        )
        torch.testing.assert_close(
            fused_indices.sort(dim=-1).values,
            ref_indices.sort(dim=-1).values,
            rtol=0,
            atol=0,
        )
        if case.return_value:
            torch.testing.assert_close(
                fused_values, ref_values, rtol=2e-2, atol=2e-2
            )


def test_fused_pool_key_indexer_matches_single_op() -> None:
    case_name = os.environ.get("KPOOL_CASE_NAME")
    if case_name is not None:
        if case_name not in _CASES:
            raise ValueError(f"unknown KPOOL_CASE_NAME: {case_name}")
        _run_case(_CASES[case_name])
        return

    failures = []
    results = []
    for name in _CASES:
        environment = os.environ.copy()
        environment["KPOOL_CASE_NAME"] = name
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"{Path(__file__).resolve()}::test_fused_pool_key_indexer_matches_single_op",
                    "-s",
                ],
                cwd=Path(__file__).parents[2],
                env=environment,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as error:
            failures.append(f"{name}: timed out after 180 seconds ({error})")
            results.append(f"{name}: TIMEOUT")
            continue
        if result.returncode != 0:
            output = (result.stdout + result.stderr)[-4000:]
            failures.append(f"{name}:\n{output}")
            results.append(f"{name}: FAIL ({_CASES[name]})")
        else:
            results.append(f"{name}: PASS ({_CASES[name]})")

    print("PoolKeyIndexer input matrix:")
    print("\n".join(results))
    assert not failures, "PoolKeyIndexer input matrix failures:\n" + "\n".join(
        failures
    )
