# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""xlite RMSNorm through the package-owned, version-checked bridge."""

from collections.abc import Callable
from typing import Any

IMPLEMENTATION = "npu.xlite.rms_norm"


def load() -> Callable[..., Any]:
    from xllm_kernel.backends.xlite import load_operator

    return load_operator("rms_norm")
