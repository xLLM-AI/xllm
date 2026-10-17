# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""einsum implementations supplied by atb; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    return load_operator("atb_matmul_ein_sum")
