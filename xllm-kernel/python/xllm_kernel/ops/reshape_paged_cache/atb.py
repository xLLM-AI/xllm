# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""reshape_paged_cache implementations supplied by atb; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load() -> Callable[..., Any]:
    import torch
    import torch_npu  # noqa: F401

    from xllm_kernel.backends.xllm import load_operator

    if not torch.npu.get_device_name().startswith("Ascend910"):
        raise ValueError("ATB reshape_paged_cache requires Ascend 910; the host Ascend 950 path uses ACLNN")
    return load_operator("reshape_paged_cache")
