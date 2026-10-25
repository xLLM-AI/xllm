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

"""Scratch workspace ownership across serialized ACL graph entries."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import torch

from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
    get_execution_buffer,
    get_execution_workspace,
)


@contextmanager
def _entry(pool: list[torch.Tensor] | None) -> Iterator[AclGraphExecutionState]:
    state = AclGraphExecutionState({}, serial_workspace_pool=pool)
    context = ForwardContext(Mock(), torch.device("cpu"), Mock(), [], execution_state=state)
    with forward_context(context):
        yield state


def test_serial_workspaces_reuse_capacity_and_keep_old_graph_addresses() -> None:
    pool: list[torch.Tensor] = []
    with _entry(pool):
        first = get_execution_workspace(("FIA", 32), lambda: torch.empty(128, dtype=torch.uint8))
        first.fill_(7)
        assert get_execution_workspace(("FIA", 32), Mock(side_effect=AssertionError)) is first
    with _entry(pool):
        smaller = get_execution_workspace(("FIA", 24), lambda: torch.empty(96, dtype=torch.uint8))
        assert smaller.data_ptr() == first.data_ptr()
        assert smaller.numel() == 96
        assert torch.all(smaller == 7)
    with _entry(pool):
        larger = get_execution_workspace(("FIA", 64), lambda: torch.empty(256, dtype=torch.uint8))
        assert larger.data_ptr() != first.data_ptr()
        assert torch.all(first == 7)
    assert [buffer.numel() for buffer in pool] == [128, 256]


def test_workspace_reuse_respects_dtype_and_private_buffers() -> None:
    pool: list[torch.Tensor] = []
    with _entry(pool):
        byte_workspace = get_execution_workspace(("FIA", 1), lambda: torch.empty(128, dtype=torch.uint8))
        private = get_execution_buffer(("metadata",), lambda: torch.zeros(128, dtype=torch.uint8))
    with _entry(pool):
        float_workspace = get_execution_workspace(("FIA", 2), lambda: torch.empty(64))
        other_private = get_execution_buffer(("metadata",), lambda: torch.ones(128, dtype=torch.uint8))
    assert float_workspace.dtype == torch.float32
    assert float_workspace.data_ptr() != byte_workspace.data_ptr()
    assert private.data_ptr() != other_private.data_ptr()
    assert len(pool) == 2


def test_graph_entries_without_serial_pool_keep_private_workspaces() -> None:
    with _entry(None):
        first = get_execution_workspace(("FIA", 24), lambda: torch.empty(128))
    with _entry(None):
        second = get_execution_workspace(("FIA", 24), lambda: torch.empty(128))
    assert first.data_ptr() != second.data_ptr()


def test_serial_workspace_rejects_multidimensional_storage() -> None:
    with _entry([]), pytest.raises(ValueError, match="one-dimensional"):
        get_execution_workspace(("FIA", 1), lambda: torch.empty(2, 4))
