# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run one rank of the HCCL or Gloo probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.logger import logger


def _exchange(
    local: torch.Tensor,
    gathered: torch.Tensor,
    send: torch.Tensor,
    received: torch.Tensor,
) -> None:
    work = dist.all_gather_into_tensor(gathered, local, async_op=True)
    work.wait()
    # Each sender contributes a distinct marker, so A2A source/destination
    # swaps cannot pass merely because every rank sends the gathered payload.
    torch.add(gathered, 100 * dist.get_rank(), out=send)
    dist.all_to_all_single(received, send)


def _check_payload(gathered: torch.Tensor, received: torch.Tensor, value: int) -> None:
    expected_gathered = torch.tensor([0, 1, 10, 11], dtype=gathered.dtype) + value
    expected_received = (
        torch.tensor(([0, 1, 100, 101], [10, 11, 110, 111])[dist.get_rank()], dtype=received.dtype) + value
    )
    torch.testing.assert_close(gathered.cpu(), expected_gathered, rtol=0, atol=0)
    torch.testing.assert_close(received.cpu(), expected_received, rtol=0, atol=0)


@torch.no_grad()
def _run_mixed_all_reduce_graph(device: torch.device, dtype: torch.dtype) -> None:
    from torch_npu import npu

    from xllm.python.distributed.npu.hccl import all_reduce_on_current_stream, get_hccl_comm

    group = cast(dist.ProcessGroup, dist.group.WORLD)
    get_hccl_comm(group, device, initialize=True)

    rank = group.rank()
    rows = (7, 448, 1024, 7)
    inputs = [torch.full((count, 5120), rank + 1, dtype=dtype, device=device) for count in rows]

    def operation() -> list[torch.Tensor]:
        outputs = []
        for value in inputs:
            reduced = value + 1
            all_reduce_on_current_stream(reduced, group)
            outputs.append(reduced * 2)
        return outputs

    stream = npu.Stream(device=device)
    stream.wait_stream(npu.current_stream())
    with npu.stream(stream):
        operation()
        operation()
    npu.synchronize()
    graph = npu.NPUGraph()
    with npu.graph(graph, stream=stream):
        outputs = operation()
    for replay in range(3):
        for value, output in zip(inputs, outputs, strict=True):
            value.add_(1)
            output.fill_(-99)
        graph.replay()
        npu.synchronize()
        expected = 14 + replay * 4
        for output in outputs:
            torch.testing.assert_close(output, torch.full_like(output, expected), atol=0, rtol=0)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gloo-eager", "hccl-graph", "hccl-mixed-all-reduce"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    use_graph = args.mode != "gloo-eager"
    mixed_all_reduce = args.mode == "hccl-mixed-all-reduce"
    if use_graph and (args.device is None or args.device < 0):
        parser.error("HCCL modes require an explicit nonnegative --device")
    if not use_graph and args.device is not None:
        parser.error("gloo-eager must not select an NPU device")

    torch.set_num_threads(1)
    npu_version = None
    device = torch.device("cpu")
    if use_graph:
        import torch_npu

        npu_version = torch_npu.__version__
        device = torch.device(f"npu:{args.device}")
        torch.npu.set_device(device)
    dtype = getattr(torch, args.dtype)
    result = {
        "rank": args.rank,
        "mode": args.mode,
        "dtype": args.dtype,
        "device": str(device),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_npu": npu_version,
        "environment": {
            name: os.environ.get(name)
            for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "HCCL_IF_IP")
        },
    }
    logger.info("Rank %d initialization: %s", args.rank, json.dumps(result))
    if mixed_all_reduce:
        from xllm import xllm_export  # noqa: F401  # Registers native HCCL ops.

    dist.init_process_group(
        backend="hccl" if use_graph else "gloo",
        init_method=(args.artifact_dir.resolve() / "rendezvous").as_uri(),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    if mixed_all_reduce:
        _run_mixed_all_reduce_graph(device, dtype)
    else:
        local = torch.tensor(([0, 1], [10, 11])[args.rank], dtype=dtype, device=device)
        gathered = torch.empty(4, dtype=dtype, device=device)
        send = torch.empty_like(gathered)
        received = torch.empty_like(gathered)
        logger.info("Rank %d eager warmup", args.rank)
        for _ in range(2):
            _exchange(local, gathered, send, received)
            _check_payload(gathered, received, 0)

        graph = None
        if use_graph:
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            capture_stream = torch.npu.Stream()
            logger.info("Rank %d capture begin", args.rank)
            with torch.npu.graph(graph, stream=capture_stream):
                _exchange(local, gathered, send, received)
            torch.npu.synchronize()
            logger.info("Rank %d capture end", args.rank)

        for value in [2, 5, 9] * 3:
            local.copy_(torch.tensor(([value, value + 1], [value + 10, value + 11])[args.rank], dtype=dtype))
            # Poison all outputs outside capture; stale buffers must never satisfy
            # the oracle even on the first replay or when an input value repeats.
            gathered.fill_(float("nan"))
            send.fill_(float("nan"))
            received.fill_(float("nan"))
            if graph is not None:
                torch.npu.synchronize()
                logger.info("Rank %d replay value=%d", args.rank, value)
                graph.replay()
                torch.npu.synchronize()
            else:
                _exchange(local, gathered, send, received)
            _check_payload(gathered, received, value)

    # On failure let the process boundary preserve the first traceback. Do not
    # enter a cleanup barrier that could hide it behind a second rank timeout.
    dist.destroy_process_group()
    logger.info("Rank %d checks passed", args.rank)


if __name__ == "__main__":
    try:
        _main()
    except Exception:
        logger.exception("Transport probe rank failed")
        raise
