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

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from xllm.python.attention.attn_dp_layout import AttnDpLayout


def _run_dp_tp_collectives(global_rank: int, rendezvous: str) -> None:
    """Exercise production packing with real TP2/DP2 Gloo collectives."""
    from datetime import timedelta

    import torch.distributed as dist

    from xllm.python.attention.attn_dp_collectives import (
        attention_latent_all_to_all_dp_tp,
        o_all_reduce_dp_tp,
        q_head_all_to_all_dp_tp,
        quantized_value_all_to_all_dp_tp,
    )
    from xllm.python.distributed import collectives

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=global_rank,
        world_size=4,
        timeout=timedelta(seconds=40),
    )
    try:
        groups = [dist.new_group(ranks=[tp, tp + 2], backend="gloo") for tp in range(2)]
        tp_rank, dp_rank = global_rank % 2, global_rank // 2
        weight_rank = tp_rank * 2 + dp_rank
        collectives._groups[("dp", "cpu")] = groups[tp_rank]
        collectives._groups[("attn_dp", "cpu")] = dist.group.WORLD
        topology = AttnDpLayout(4, 8, 3, 3, 6, owner_heads=4)
        counts, padded = (1, 3), 3
        rows = slice(dp_rank * padded, dp_rank * padded + counts[dp_rank])
        weight_heads = slice(weight_rank * 2, (weight_rank + 1) * 2)
        owner_heads = slice(tp_rank * 4, (tp_rank + 1) * 4)
        full_q = torch.arange(6 * 8 * 3, dtype=torch.float32).reshape(6, 8, 3)
        owner_q = q_head_all_to_all_dp_tp(
            full_q[:, weight_heads].contiguous(), topology, counts[dp_rank], padded, 2, 2, tp_rank, dp_rank
        )
        torch.testing.assert_close(owner_q, full_q[rows, owner_heads])
        latent = attention_latent_all_to_all_dp_tp(owner_q, topology, counts[dp_rank], padded, 2, 2, tp_rank, dp_rank)
        expected_latent = full_q[:, weight_heads].clone()
        expected_latent[1:3].zero_()
        torch.testing.assert_close(latent, expected_latent)

        full_value = (torch.arange(6 * 8 * 2).reshape(6, 8, 2) - 96).to(torch.int8)
        full_scale = torch.arange(6, dtype=torch.float32) + 0.125
        value, scale = quantized_value_all_to_all_dp_tp(
            full_value[rows, owner_heads].reshape(counts[dp_rank], 8),
            full_scale[rows],
            topology,
            2,
            counts[dp_rank],
            padded,
            2,
            2,
            tp_rank,
            dp_rank,
        )
        expected_value = full_value[:, weight_heads].clone()
        expected_value[1:3].zero_()
        expected_scale = full_scale.clone()
        expected_scale[1:3].fill_(1)
        assert torch.equal(value, expected_value)
        assert torch.equal(scale.view(torch.uint8), expected_scale.view(torch.uint8))
        with patch.object(collectives, "_all_reduce", dist.all_reduce):
            output = o_all_reduce_dp_tp(torch.full((6, 6), float(weight_rank + 1)), topology, padded, 2)
        torch.testing.assert_close(output, torch.full((6, 6), 10.0))

        if tp_rank == 0:
            input_splits, output_splits = ([1, 2], [1, 3]) if dp_rank == 0 else ([3, 0], [2, 0])
            payload = torch.arange(3, dtype=torch.float32) + 2 * dp_rank
            expected = torch.tensor([0.0, 2.0, 3.0, 4.0]) if dp_rank == 0 else torch.tensor([1.0, 2.0])
        else:
            input_splits, output_splits = [0, 0], [0, 0]
            payload = torch.empty(0, dtype=torch.float32)
            expected = payload
        received = torch.empty_like(expected)
        collectives.all_to_all_single_(
            received, payload, groups[tp_rank], output_split_sizes=output_splits, input_split_sizes=input_splits
        )
        torch.testing.assert_close(received, expected)
    finally:
        collectives._groups.clear()
        dist.destroy_process_group()


def test_dp_tp_routing_with_uneven_owner_rows(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(_run_dp_tp_collectives, args=(str(tmp_path / "gloo"),), nprocs=4, join=True)


@pytest.mark.parametrize("case", ["shape", "rows", "dtype", "contiguity"])
def test_all_to_all_rejects_invalid_buffers_before_communication(case: str) -> None:
    from types import SimpleNamespace

    from xllm.python.distributed import collectives

    input = torch.zeros(4, 3)
    output = torch.empty_like(input)
    if case == "shape":
        output = torch.empty(4, 2)
    elif case == "rows":
        input = torch.zeros(3, 3)
        output = torch.empty_like(input)
    elif case == "dtype":
        output = torch.empty(4, 3, dtype=torch.int32)
    else:
        input = torch.zeros(3, 4).transpose(0, 1)
    previous = collectives._groups.get(("dp", "cpu"))
    collectives._groups[("dp", "cpu")] = SimpleNamespace(size=lambda: 2)
    try:
        with (
            patch.object(collectives.dist, "all_to_all_single", side_effect=AssertionError("invalid HCCL call")),
            pytest.raises(ValueError),
        ):
            collectives.all_to_all_single(output, input, "dp")
    finally:
        if previous is None:
            collectives._groups.pop(("dp", "cpu"))
        else:
            collectives._groups[("dp", "cpu")] = previous
