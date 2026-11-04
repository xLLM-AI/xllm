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

"""Native update batching preserves captured order and entry-private inputs."""

from unittest.mock import Mock, patch

import torch

from xllm.python.model_executor.forward_context import AclGraphNativeTask, AclGraphNativeTaskGroup, AclGraphTask
from xllm.python.model_executor.runners.acl_graph import AclGraphRunner, _group_graph_tasks


def test_native_groups_preserve_interleaved_updates_and_refresh_once() -> None:
    calls = []
    lengths = [7, 11]
    prepare = Mock()
    prepare.refresh.side_effect = lambda: lengths.__setitem__(slice(None), [13, 17])
    batch = Mock()
    batch.update.side_effect = lambda begin, end, values: calls.append((begin, end, values.copy()))
    legacy = AclGraphTask(Mock(), object(), lambda: calls.append("legacy"))
    plan = _group_graph_tasks(
        [
            AclGraphNativeTask(batch, 0, lengths, prepare),
            AclGraphNativeTask(batch, 1, lengths),
            legacy,
            AclGraphNativeTask(batch, 2, lengths),
            AclGraphNativeTask(batch, 3, lengths),
        ]
    )
    assert len(plan) == 3
    assert plan[1] is legacy
    with (
        patch.object(torch.npu, "graph_task_update_begin"),
        patch.object(torch.npu, "graph_task_update_end"),
    ):
        AclGraphRunner._update_graph_tasks(Mock(), plan)
    assert calls == [(0, 2, [13, 17]), "legacy", (2, 4, [13, 17])]
    prepare.refresh.assert_called_once_with()
    legacy.event.record.assert_called_once()


def test_native_groups_keep_different_entries_and_index_gaps_separate() -> None:
    first, second = Mock(), Mock()
    lengths = [5]
    other_lengths = [19]
    plan = _group_graph_tasks(
        [
            AclGraphNativeTask(first, 0, lengths),
            AclGraphNativeTask(first, 2, lengths),
            AclGraphNativeTask(second, 0, other_lengths),
            AclGraphNativeTask(second, 1, other_lengths),
        ]
    )
    assert len(plan) == 3
    assert all(isinstance(task, AclGraphNativeTaskGroup) for task in plan)
    AclGraphRunner._update_graph_tasks(Mock(), plan)
    assert first.update.call_args_list[0].args == (0, 1, lengths)
    assert first.update.call_args_list[1].args == (2, 3, lengths)
    second.update.assert_called_once_with(0, 2, other_lengths)
