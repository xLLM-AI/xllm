# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Opt-in multi-rank HCCL fixtures; never initialize on ordinary pytest collection."""

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def hccl_group() -> Iterator[None]:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("run with torchrun --nproc-per-node >= 2")
    pytest.importorskip("torch_npu")
    from xllm_kernel.numerics.distributed import session

    with session(True):
        yield
