# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Equal-split all-to-all correctness under torchrun."""

from distributed_checks import check_collective
from xllm_kernel.numerics.all_to_all_single import SUITE


def test_all_to_all_single() -> None:
    check_collective(SUITE, "npu.hccl.all_to_all_single.process_group", "eager")
