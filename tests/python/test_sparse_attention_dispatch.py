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

"""SFA interface selection and caller-owned output contracts."""

from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.kernels_npu import sparse_attention


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("entry", ["allocate", "out", "lse"])
def test_sparse_attention_preserves_backend_and_output_contract(
    monkeypatch: pytest.MonkeyPatch, rope: bool, entry: str
) -> None:
    query = torch.zeros(2, 4, 8)
    key = torch.zeros(1, 16, 1, 8)
    indices = torch.zeros(2, 1, 16, dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    query_rope = torch.zeros(2, 4, 4) if rope else None
    key_rope = torch.zeros(1, 16, 1, 4) if rope else None
    expected = (torch.ones_like(query), torch.ones(1, 2, 4), torch.full((1, 2, 4), 2.0))
    custom = MagicMock(return_value=expected)
    native = MagicMock(return_value=expected)

    def write_out(*args: object) -> torch.Tensor:
        output = args[-1]
        output.copy_(expected[0])
        return output

    custom_out = MagicMock(side_effect=write_out)
    monkeypatch.setattr(torch.ops.xllm_ops, "sparse_flash_attention_lse", custom, raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_sparse_flash_attention", native, raising=False)
    monkeypatch.setattr(torch.ops.xllm_ops, "sparse_flash_attention_lse_out", custom_out, raising=False)
    args = (
        query,
        key,
        key,
        indices,
        block_table,
        lengths,
        lengths,
        query_rope,
        key_rope,
        0.125,
        1,
        "TND",
        "PA_BSND",
        3,
    )
    if entry == "lse":
        result = sparse_attention.sparse_flash_attention_lse(
            *args, pre_tokens=31, next_tokens=7, attention_mode=2, return_softmax_lse=True
        )
        assert all(actual is wanted for actual, wanted in zip(result, expected))
        tail = (31, 7, 2, True)
    elif entry == "out":
        buffer = torch.zeros_like(query)
        result = sparse_attention.sparse_flash_attention_out(*args, output=buffer)
        assert result is buffer
        torch.testing.assert_close(result, expected[0])
        tail = (9223372036854775807, 9223372036854775807, 2, False)
    else:
        result = sparse_attention.sparse_flash_attention(*args)
        assert result is expected[0]
        tail = (9223372036854775807, 9223372036854775807, 2, False)

    if entry == "out" and rope:
        custom_out.assert_called_once_with(*args, *tail, buffer)
        custom.assert_not_called()
        native.assert_not_called()
    elif rope:
        custom_out.assert_not_called()
        native.assert_not_called()
        custom.assert_called_once_with(*args, *tail)
    else:
        custom_out.assert_not_called()
        custom.assert_not_called()
        native.assert_called_once_with(
            query,
            key,
            key,
            indices,
            0.125,
            block_table=block_table,
            actual_seq_lengths_query=lengths,
            actual_seq_lengths_kv=lengths,
            query_rope=None,
            key_rope=None,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=tail[0],
            next_tokens=tail[1],
            attention_mode=tail[2],
            return_softmax_lse=tail[3],
        )


@pytest.mark.parametrize("query_rope", [None, torch.zeros(1)])
@pytest.mark.parametrize("entry", ["allocate", "out", "lse"])
def test_partial_rope_pair_is_rejected_before_operator_execution(
    monkeypatch: pytest.MonkeyPatch, query_rope: torch.Tensor | None, entry: str
) -> None:
    custom = MagicMock()
    native = MagicMock()
    custom_out = MagicMock()
    monkeypatch.setattr(torch.ops.xllm_ops, "sparse_flash_attention_lse", custom, raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_sparse_flash_attention", native, raising=False)
    monkeypatch.setattr(torch.ops.xllm_ops, "sparse_flash_attention_lse_out", custom_out, raising=False)
    tensor = torch.zeros(1)
    key_rope = tensor if query_rope is None else None
    args = (
        tensor,
        tensor,
        tensor,
        tensor,
        None,
        None,
        None,
        query_rope,
        key_rope,
        0.125,
        1,
        "TND",
        "PA_BSND",
        3,
    )
    with pytest.raises(ValueError, match="both be present or absent"):
        if entry == "out":
            sparse_attention.sparse_flash_attention_out(*args, output=torch.empty_like(tensor))
        elif entry == "lse":
            sparse_attention.sparse_flash_attention_lse(*args)
        else:
            sparse_attention.sparse_flash_attention(*args)
    custom.assert_not_called()
    native.assert_not_called()
    custom_out.assert_not_called()
