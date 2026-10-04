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


def _resolve_linear_state_io_indices(
    metadata: SimpleNamespace,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from xllm.python.attention.backend import resolve_linear_state_io_indices

    return resolve_linear_state_io_indices(metadata)


def _metadata(
    write_indices: torch.Tensor | None,
    read_indices: torch.Tensor | None,
    *,
    is_prefill: bool,
    is_chunked_prefill: bool = False,
    is_spec_verify: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        linear_state_indices=write_indices,
        linear_state_read_indices=read_indices,
        linear_state_write_indices=write_indices,
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        is_spec_verify=is_spec_verify,
    )


def test_missing_read_indices_reuses_write_indices() -> None:
    write_indices = torch.tensor([3, 7], dtype=torch.int32)

    read_indices, resolved_write_indices = _resolve_linear_state_io_indices(
        _metadata(write_indices, None, is_prefill=True)
    )

    assert read_indices is write_indices
    assert resolved_write_indices is write_indices


def test_none_write_indices_reuses_legacy_indices() -> None:
    legacy_indices = torch.tensor([3, 7], dtype=torch.int32)
    metadata = _metadata(None, None, is_prefill=True)
    metadata.linear_state_indices = legacy_indices

    read_indices, write_indices = _resolve_linear_state_io_indices(metadata)

    assert read_indices is legacy_indices
    assert write_indices is legacy_indices


def test_prefill_accepts_distinct_read_and_write_indices() -> None:
    read_indices = torch.tensor([2, 5], dtype=torch.int32)
    write_indices = torch.tensor([3, 7], dtype=torch.int32)

    resolved_read_indices, resolved_write_indices = _resolve_linear_state_io_indices(
        _metadata(write_indices, read_indices, is_prefill=True)
    )

    assert torch.equal(resolved_read_indices, read_indices)
    assert torch.equal(resolved_write_indices, write_indices)


def test_chunked_prefill_accepts_distinct_read_and_write_indices() -> None:
    read_indices = torch.tensor([2], dtype=torch.int32)
    write_indices = torch.tensor([3], dtype=torch.int32)

    resolved_read_indices, resolved_write_indices = _resolve_linear_state_io_indices(
        _metadata(
            write_indices,
            read_indices,
            is_prefill=False,
            is_chunked_prefill=True,
        )
    )

    assert torch.equal(resolved_read_indices, read_indices)
    assert torch.equal(resolved_write_indices, write_indices)


def test_decode_rejects_distinct_read_and_write_indices() -> None:
    read_indices = torch.tensor([2], dtype=torch.int32)
    write_indices = torch.tensor([3], dtype=torch.int32)

    with pytest.raises(
        RuntimeError,
        match="read/write separation is only supported for non-speculative prefill",
    ):
        _resolve_linear_state_io_indices(_metadata(write_indices, read_indices, is_prefill=False))


def test_spec_verify_rejects_distinct_read_and_write_indices() -> None:
    read_indices = torch.tensor([2], dtype=torch.int32)
    write_indices = torch.tensor([3], dtype=torch.int32)

    with pytest.raises(
        RuntimeError,
        match="read/write separation is only supported for non-speculative prefill",
    ):
        _resolve_linear_state_io_indices(
            _metadata(
                write_indices,
                read_indices,
                is_prefill=True,
                is_spec_verify=True,
            )
        )


def test_read_and_write_indices_must_have_matching_shapes() -> None:
    read_indices = torch.tensor([2], dtype=torch.int32)
    write_indices = torch.tensor([3, 7], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="must have the same shape"):
        _resolve_linear_state_io_indices(_metadata(write_indices, read_indices, is_prefill=True))


def test_read_indices_require_write_indices() -> None:
    read_indices = torch.tensor([2], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="require write indices"):
        _resolve_linear_state_io_indices(_metadata(None, read_indices, is_prefill=True))
