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

"""Empty-DP coverage for the NPU ACL decode-graph runner."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.dsa_metadata import build_cache_specs
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def test_dsa_graph_empty_rank_uses_reserved_block_zero() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(
            page_size=128,
            is_mla=False,
            group_infos=group_infos,
        ),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(multi_block_tables=(), is_dummy=True)

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        batch_size=1,
    )

    for table in static_metadata.multi_block_tables:
        assert torch.all(table == 0)


def test_linear_verify_state_pads_accepted_counts_and_indices_for_bucket_tail() -> None:
    counts = (3, 1)
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.full((5,), 7, dtype=torch.int32),
            num_accepted_tokens=torch.full((5,), 5, dtype=torch.int32),
            num_accepted_tokens_host_values=(9, 9),
        ),
    )
    DecodeAclGraphRunner._fill_linear_verify_state(
        entry,
        SimpleNamespace(
            num_accepted_tokens=torch.tensor(counts, dtype=torch.int32),
            num_accepted_tokens_host_values=tuple(counts),
        ),
        (4, 4),
        torch.tensor([5, 9], dtype=torch.int32),
    )
    assert entry.static_metadata.linear_state_indices.tolist() == [5, 9, 0, 0, 0]
    assert entry.static_metadata.num_accepted_tokens.tolist() == [*counts, 1, 1, 1]
    assert entry.static_metadata.num_accepted_tokens_host_values == (*counts, 1, 1, 1)


@pytest.mark.parametrize("counts", [(0, 1), (1, 5)])
def test_linear_verify_state_rejects_counts_outside_verify_width(counts: tuple[int, ...]) -> None:
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.zeros(5, dtype=torch.int32),
            num_accepted_tokens=torch.ones(5, dtype=torch.int32),
            num_accepted_tokens_host_values=(),
        ),
    )
    with pytest.raises(ValueError, match=r"in \[1, verify width\]"):
        DecodeAclGraphRunner._fill_linear_verify_state(
            entry,
            SimpleNamespace(
                num_accepted_tokens=torch.tensor(counts, dtype=torch.int32),
                num_accepted_tokens_host_values=tuple(counts),
            ),
            (4, 4),
            torch.tensor([5, 9], dtype=torch.int32),
        )


@pytest.mark.parametrize("bad_accepted", [None, torch.ones(3, dtype=torch.int32)])
def test_linear_verify_state_requires_sequence_scoped_accepted_counts(bad_accepted: torch.Tensor | None) -> None:
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.zeros(5, dtype=torch.int32),
            num_accepted_tokens=torch.zeros(5, dtype=torch.int32),
            num_accepted_tokens_host_values=(),
        ),
    )
    with pytest.raises(ValueError, match="sequence-scoped"):
        DecodeAclGraphRunner._fill_linear_verify_state(
            entry,
            SimpleNamespace(num_accepted_tokens=bad_accepted, num_accepted_tokens_host_values=()),
            (4, 4),
            torch.tensor([5, 9], dtype=torch.int32),
        )


def test_linear_verify_state_accepts_row_expanded_state_indices() -> None:
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.full((4,), 7, dtype=torch.int32),
            num_accepted_tokens=torch.full((4,), 5, dtype=torch.int32),
            num_accepted_tokens_host_values=(9,),
        ),
    )
    DecodeAclGraphRunner._fill_linear_verify_state(
        entry,
        SimpleNamespace(num_accepted_tokens=torch.tensor([2], dtype=torch.int32)),
        (3,),
        torch.tensor([8, 8, 8], dtype=torch.int32),
    )
    assert entry.static_metadata.linear_state_indices.tolist() == [8, 0, 0, 0]
    assert entry.static_metadata.num_accepted_tokens.tolist() == [2, 1, 1, 1]
