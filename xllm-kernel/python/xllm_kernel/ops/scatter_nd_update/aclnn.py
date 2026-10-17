# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""scatter_nd_update implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    return load_operator("scatter_nd_update")
