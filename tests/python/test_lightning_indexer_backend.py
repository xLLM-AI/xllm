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

"""Tests for the GLM-5 Lightning Indexer backend selection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from xllm.python.kernels_npu import sparse_attention


def _npu_query() -> SimpleNamespace:
    return SimpleNamespace(device=SimpleNamespace(type="npu"))


def _indexer_args() -> tuple[object, ...]:
    return (
        _npu_query(),
        object(),
        object(),
        object(),
        object(),
        object(),
        "TND",
        "PA_BSND",
        2048,
        3,
        sparse_attention.MAX_LIGHTNING_INDEXER_WINDOW,
        sparse_attention.MAX_LIGHTNING_INDEXER_WINDOW,
        False,
    )


def test_lightning_indexer_prefers_torch_npu_for_glm5_contract(monkeypatch) -> None:
    args = _indexer_args()
    expected_topk = object()
    native_op = Mock(return_value=(expected_topk, object()))
    monkeypatch.setattr(sparse_attention, "_TORCH_NPU_LIGHTNING_INDEXER", native_op)
    fallback = Mock()
    monkeypatch.setattr(sparse_attention, "_call_xllm_lightning_indexer", fallback)

    actual = sparse_attention.lightning_indexer(
        *args,
        prefer_torch_npu=True,
    )

    assert actual is expected_topk
    native_op.assert_called_once_with(
        query=args[0],
        key=args[1],
        weights=args[2],
        actual_seq_lengths_query=args[3],
        actual_seq_lengths_key=args[4],
        block_table=args[5],
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
    )
    fallback.assert_not_called()


def test_lightning_indexer_falls_back_for_unsupported_request(monkeypatch) -> None:
    args = list(_indexer_args())
    args[6] = "BSND"
    expected_topk = object()
    native_op = Mock(return_value=(object(), object()))
    fallback = Mock(return_value=expected_topk)
    monkeypatch.setattr(sparse_attention, "_TORCH_NPU_LIGHTNING_INDEXER", native_op)
    monkeypatch.setattr(sparse_attention, "_call_xllm_lightning_indexer", fallback)

    actual = sparse_attention.lightning_indexer(
        *args,
        prefer_torch_npu=True,
    )

    assert actual is expected_topk
    native_op.assert_not_called()
    fallback.assert_called_once_with(*args)


def test_lightning_indexer_falls_back_when_torch_npu_is_unavailable(monkeypatch) -> None:
    args = _indexer_args()
    expected_topk = object()
    fallback = Mock(return_value=expected_topk)
    monkeypatch.setattr(sparse_attention, "_TORCH_NPU_LIGHTNING_INDEXER", None)
    monkeypatch.setattr(sparse_attention, "_call_xllm_lightning_indexer", fallback)

    actual = sparse_attention.lightning_indexer(
        *args,
        prefer_torch_npu=True,
    )

    assert actual is expected_topk
    fallback.assert_called_once_with(*args)
