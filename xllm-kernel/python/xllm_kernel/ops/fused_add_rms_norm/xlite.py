# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""fused_add_rms_norm implementations supplied by xlite; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load() -> Callable[..., Any]:
    from xllm_kernel.backends.xlite import load_operator

    return load_operator("fused_add_rms_norm")
