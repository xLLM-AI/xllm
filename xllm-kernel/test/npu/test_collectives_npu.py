# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check the packaged collectives on two local NPUs.

export HCCL_OP_EXPANSION_MODE=AIV
torchrun --standalone --nproc-per-node=2 -m pytest xllm-kernel/test/npu/test_collectives_npu.py
"""

import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from xllm_kernel.ops.collectives import (
    all_gather,
    all_reduce,
    has_matmul_reduce_scatter,
    matmul_reduce_scatter,
    reduce_scatter,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("LOCAL_WORLD_SIZE") != "2" or os.environ.get("WORLD_SIZE") != "2",
    reason="requires two local NPU ranks launched with torchrun",
)


@pytest.fixture(scope="module")
def hccl_group() -> Iterator[dist.ProcessGroup]:
    device = torch.device("npu", torch.npu.current_device())
    dist.init_process_group("hccl", device_id=device)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("capture", [False, True], ids=["eager", "aclgraph"])
def test_current_stream_collectives(hccl_group: dist.ProcessGroup, capture: bool) -> None:
    if capture and os.environ.get("HCCL_OP_EXPANSION_MODE") != "AIV":
        pytest.skip("HCCL capture requires HCCL_OP_EXPANSION_MODE=AIV")
    device = torch.device("npu", torch.npu.current_device())
    comm = hccl_group._get_backend(device).get_hccl_comm(device.index)
    rank = hccl_group.rank()
    rows = torch.arange(32, dtype=torch.float32)
    local = rows.to(device)
    reduced = torch.empty_like(local)
    gathered = local.new_empty((2, local.numel()))
    scattered = torch.empty_like(local)

    def forward() -> None:
        reduced.copy_(local)
        all_reduce(reduced, comm)
        all_gather(local, gathered, comm)
        reduce_scatter(gathered + rank, scattered, comm)
        scattered.add_(reduced)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        forward()
    stream.synchronize()
    if capture:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            forward()

    for step in (3, 7):
        with torch.npu.stream(stream):
            local.copy_(rows + (rank + 1) * step)
            if capture:
                graph.replay()
            else:
                forward()
        stream.synchronize()
        expected = torch.stack((rows + step, rows + 2 * step))
        total = expected.sum(dim=0)
        torch.testing.assert_close(local.cpu(), expected[rank], rtol=0, atol=0)
        torch.testing.assert_close(reduced.cpu(), total, rtol=0, atol=0)
        torch.testing.assert_close(gathered.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scattered.cpu(), 2 * expected[rank] + 1 + total, rtol=0, atol=0)


def test_matmul_reduce_scatter_transposed_weight(hccl_group: dist.ProcessGroup) -> None:
    if not has_matmul_reduce_scatter():
        pytest.skip("TorchNPU does not provide fused matmul-reduce-scatter")
    device = torch.device("npu", torch.npu.current_device())
    hcom = hccl_group._get_backend(device).get_hccl_comm_name(device.index)
    x = torch.zeros(32, 256, dtype=torch.bfloat16, device=device)
    x[:, 0] = torch.arange(32, device=device) + hccl_group.rank()
    checkpoint_weight = torch.zeros(128, 256, dtype=x.dtype, device=device)
    checkpoint_weight[:, 0] = torch.arange(128, device=device) % 7 - 3
    weight = checkpoint_weight.t()

    reference = x.new_empty((16, 128))
    dist.reduce_scatter_tensor(reference, x @ weight, group=hccl_group)
    result = matmul_reduce_scatter(x, weight, hcom, hccl_group.size(), comm_mode="aiv")
    torch.testing.assert_close(result.cpu(), reference.cpu(), rtol=0, atol=0)
