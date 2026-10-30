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

from xllm.python.layers.moe_parallel import Eplv2CommPolicy, TokenParallelLayout
from xllm.python.models.glm5_next_metadata import Glm5NextEplv2MetadataBuilder


def test_tp_shards_reconstruct_uneven_dp_batches_with_common_dispatch_capacity() -> None:
    counts = (4, 1)
    capacities = set()
    for dp_rank, rows in enumerate(counts):
        source = torch.arange(rows * 2).reshape(rows, 2)
        shards = []
        for tp_rank in range(3):
            layout = TokenParallelLayout.from_dp(rows, 3, tp_rank, 2, dp_rank, counts)
            shard = layout.shard(source)
            shards.append(shard)
            dispatched = layout.pad_dispatch(shard)
            capacities.add(dispatched.shape[0])
            torch.testing.assert_close(dispatched[: layout.shard_tokens], shard)
            assert not dispatched[layout.shard_tokens :].any()
        torch.testing.assert_close(torch.cat(shards)[:rows], source)
    assert capacities == {2}


def test_persistent_metadata_reuses_storage_and_clears_previous_activity() -> None:
    builder = Glm5NextEplv2MetadataBuilder(2, 0)
    builder.bind_layer_caches([])
    batch = SimpleNamespace(num_tokens=4, num_tokens_after_padding=8, input_ids=torch.zeros(8, dtype=torch.long))
    persistent = builder.allocate_persistent(batch, None)
    address = persistent.local_token_mask.data_ptr()
    for valid in (4, 0, 2, 0):
        metadata = SimpleNamespace(raw_dp_execution_token_counts=(valid, 4))
        builder.update_persistent(persistent, batch, metadata)
        assert persistent.local_token_mask.data_ptr() == address
        torch.testing.assert_close(persistent.local_token_mask, torch.arange(8) < valid)
        torch.testing.assert_close(builder.build(batch, metadata).local_token_mask, torch.arange(4) < valid)


def test_comm_policy_admits_graph_only_within_mc2_capacity() -> None:
    policy = Eplv2CommPolicy.from_geometry(4096, 8, 4, 2, buffer_mb=64)
    assert policy.select(policy.mc2_capacity, graph=True) == "mc2"
    assert policy.select(policy.mc2_capacity + 1) == "alltoall"
    with pytest.raises(ValueError, match="requires eager execution"):
        policy.select(policy.mc2_capacity + 1, graph=True)
