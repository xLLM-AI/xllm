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
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.model_executor.forward_context import EplbRuntimeState
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _decode_runner(max_batch: int, dp_size: int) -> DecodeAclGraphRunner:
    return DecodeAclGraphRunner(
        torch.nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=max_batch,
        max_model_len=8,
        dp_size=dp_size,
    )


def _graph_entry(runner: DecodeAclGraphRunner, state: EplbRuntimeState) -> SimpleNamespace:
    return SimpleNamespace(batch_size=4, eplb=runner._allocate_graph_eplb_state(state, padded_batch_size=4))


def test_executor_forwards_worker_eplb_state_to_eager_runner() -> None:
    executor = object.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.layerwise_split_size = 1
    executor.decode_graph_runner = None
    executor.inductor_runner = None
    executor.eager_runner = MagicMock()
    executor.eager_runner.execute.return_value = torch.ones(1)

    metadata = SimpleNamespace(is_prefill=False, is_chunked_prefill=False)
    expert_load_data = torch.zeros((2, 4), dtype=torch.int64)
    decode_token_mask = torch.tensor([True, False])

    output = executor.execute(
        torch.zeros(2, dtype=torch.int64),
        torch.arange(2),
        metadata,
        expert_load_data=expert_load_data,
        eplb_decode_token_mask=decode_token_mask,
        is_graph_warmup=True,
    )

    state = executor.eager_runner.execute.call_args.kwargs["eplb"]
    assert isinstance(state, EplbRuntimeState)
    assert state.expert_load_data is expert_load_data
    assert state.decode_token_mask is decode_token_mask
    assert state.is_graph_warmup
    torch.testing.assert_close(output, torch.ones(1))


def test_acl_graph_eplb_decode_mask_updates_persistent_storage() -> None:
    runner = _decode_runner(max_batch=8, dp_size=2)
    expert_load_data = torch.zeros((2, 4), dtype=torch.int64)
    first = EplbRuntimeState(
        expert_load_data=expert_load_data,
        decode_token_mask=torch.tensor([True, False, True, True, False]),
        is_graph_warmup=True,
    )
    entry = _graph_entry(runner, first)
    metadata = SimpleNamespace(
        dp_execution_token_counts=(3, 2),
        raw_dp_execution_token_counts=(3, 2),
    )

    runner._fill_graph_eplb_decode_mask(entry, first, metadata)
    static_mask = entry.eplb.decode_token_mask
    assert static_mask is not None
    data_ptr = static_mask.data_ptr()
    assert static_mask.tolist() == [True, False, True, False, True, False, False, False]

    second = EplbRuntimeState(
        expert_load_data=expert_load_data,
        decode_token_mask=torch.tensor([False, True, True]),
        is_graph_warmup=False,
    )
    metadata.dp_execution_token_counts = (2, 1)
    metadata.raw_dp_execution_token_counts = (2, 1)
    runner._fill_graph_eplb_decode_mask(entry, second, metadata)

    assert static_mask.data_ptr() == data_ptr
    assert static_mask.tolist() == [False, True, False, False, True, False, False, False]


def test_acl_graph_eplb_decode_mask_ignores_empty_rank_dummy_rows() -> None:
    runner = _decode_runner(max_batch=8, dp_size=2)
    expert_load_data = torch.zeros((2, 4), dtype=torch.int64)
    state = EplbRuntimeState(
        expert_load_data=expert_load_data,
        decode_token_mask=torch.tensor([True, False, True]),
        is_graph_warmup=False,
    )
    entry = _graph_entry(runner, state)
    metadata = SimpleNamespace(
        dp_execution_token_counts=(3, 1),
        raw_dp_execution_token_counts=(3, 0),
    )

    runner._fill_graph_eplb_decode_mask(entry, state, metadata)
    static_mask = entry.eplb.decode_token_mask
    assert static_mask.tolist() == [True, False, True, False, False, False, False, False]

    state.decode_token_mask = torch.tensor([False, True, True])
    metadata.dp_execution_token_counts = (1, 3)
    metadata.raw_dp_execution_token_counts = (0, 3)
    runner._fill_graph_eplb_decode_mask(entry, state, metadata)

    assert static_mask.tolist() == [False, False, False, False, False, True, True, False]


def test_acl_graph_eplb_decode_mask_supports_non_dp_padding() -> None:
    runner = _decode_runner(max_batch=4, dp_size=1)
    state = EplbRuntimeState(
        expert_load_data=torch.zeros((1, 4), dtype=torch.int64),
        decode_token_mask=torch.tensor([True, False]),
        is_graph_warmup=False,
    )
    entry = _graph_entry(runner, state)
    metadata = SimpleNamespace(dp_execution_token_counts=(2,))

    runner._fill_graph_eplb_decode_mask(entry, state, metadata)
    static_mask = entry.eplb.decode_token_mask
    data_ptr = static_mask.data_ptr()
    assert static_mask.tolist() == [True, False, False, False]

    state.decode_token_mask = torch.tensor([False, True, True])
    runner._fill_graph_eplb_decode_mask(entry, state, metadata)

    assert static_mask.data_ptr() == data_ptr
    assert static_mask.tolist() == [False, True, True, False]


@pytest.mark.parametrize(
    ("source", "execution_counts", "raw_counts", "message"),
    [
        (torch.ones(2, dtype=torch.bool), (2,), (2,), "requires 2 token counts"),
        (torch.ones(2, dtype=torch.bool), (2, 1), (2, 1), "does not match DP token counts"),
        (torch.ones(5, dtype=torch.bool), (5, 1), (5, 0), "must fit the ACL graph batch bucket"),
    ],
)
def test_acl_graph_eplb_decode_mask_rejects_invalid_dp_layout(
    source: torch.Tensor,
    execution_counts: tuple[int, ...],
    raw_counts: tuple[int, ...],
    message: str,
) -> None:
    runner = _decode_runner(max_batch=8, dp_size=2)
    state = EplbRuntimeState(
        expert_load_data=torch.zeros((2, 4), dtype=torch.int64),
        decode_token_mask=source,
        is_graph_warmup=False,
    )
    entry = _graph_entry(runner, state)
    metadata = SimpleNamespace(
        dp_execution_token_counts=execution_counts,
        raw_dp_execution_token_counts=raw_counts,
    )

    with pytest.raises(RuntimeError, match=message):
        runner._fill_graph_eplb_decode_mask(entry, state, metadata)
