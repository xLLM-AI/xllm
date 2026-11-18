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

r"""Profile ATB and transpose-BMM device kernels with the Ascend NPU profiler.

This is a manual profiling workload, not a pytest correctness test.
Run from the repository root in an NPU container with the CANN/ATB environment
loaded, matching torch/torch_npu packages, and the standalone xllm-kernel package
installed. The examples below use CANN 9.0 msprof's positional
application syntax; check the installed ``msprof --help`` for other versions.
Use a free device and a new output directory for each capture.

Capture Q split inputs separately for each implementation::

    ASCEND_RT_VISIBLE_DEVICES=0 msprof --output=./profile_atb_q_split \
        --task-time=on --ai-core=on --aicpu=on \
        python tests/python/profile_atb_matmul_ein_sum.py \
        --device 0 --implementation atb --projection q --input-layout q_split \
        --tokens 4 --heads 4 --warmup 30 --iterations 200

    ASCEND_RT_VISIBLE_DEVICES=0 msprof --output=./profile_tbmm_q_split \
        --task-time=on --ai-core=on --aicpu=on \
        python tests/python/profile_atb_matmul_ein_sum.py \
        --device 0 --implementation tbmm --projection q --input-layout q_split \
        --tokens 4 --heads 4 --warmup 30 --iterations 200

For contiguous Q or V, set ``--input-layout contiguous`` and ``--projection q``
or ``v`` for both captures, with distinct output directories. Operands are BF16:
Q uses [T,H,192] x [H,192,512]; V uses [T,H,512] x [H,512,256]. Q split keeps a
non-contiguous view of [T,H,256] storage rather than copying it before the call.

Compare device durations in the exported msprof kernel timeline, not Python
wall-clock time. Discard the first 30 projection calls (or the selected warmup
count), then measure the next 200. For Q split, include each Slice copy and its
following matmul kernel in both implementations; discard complete warmup pairs.
Report core and copy durations separately. Their sum excludes host/queue gaps
and is not model E2E latency. This example adds no host timer or graph replay.
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--implementation", choices=("atb", "tbmm"), required=True)
    parser.add_argument("--projection", choices=("q", "v"), default="v")
    parser.add_argument("--input-layout", choices=("contiguous", "q_split"), default="contiguous")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.warmup, args.iterations) <= 0:
        parser.error("tokens, heads, warmup and iterations must be positive")
    if args.input_layout == "q_split" and args.projection != "q":
        parser.error("q_split input layout requires --projection q")

    torch.npu.set_device(args.device)
    from xllm_kernel import initialize
    from xllm_kernel.ops import matmul_ein_sum

    if args.implementation == "atb":
        initialize(device="npu", operations=("matmul_ein_sum",))
    dim, out_dim = (192, 512) if args.projection == "q" else (512, 256)
    generator = torch.Generator().manual_seed(20260923)
    input_dim = dim + 64 if args.input_layout == "q_split" else dim
    backing = torch.randn((args.tokens, args.heads, input_dim), generator=generator, dtype=torch.bfloat16).npu()
    x = backing[..., :dim]
    weight = torch.randn((args.heads, dim, out_dim), generator=generator, dtype=torch.bfloat16).npu()
    kernel = matmul_ein_sum if args.implementation == "atb" else torch.ops.npu.npu_transpose_batchmatmul
    kwargs = {} if args.implementation == "atb" else {"perm_x1": (1, 0, 2), "perm_x2": (0, 1, 2), "perm_y": (1, 0, 2)}
    with torch.inference_mode():
        for _ in range(args.warmup):
            kernel(x, weight, **kwargs)
        torch.npu.synchronize()
        for _ in range(args.iterations):
            kernel(x, weight, **kwargs)
        torch.npu.synchronize()


if __name__ == "__main__":
    main()
