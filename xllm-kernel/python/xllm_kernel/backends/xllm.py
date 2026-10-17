# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Load host Torch schemas; the schema namespace does not identify a solution."""

from collections.abc import Callable
from typing import Any


def load_operator(name: str) -> Callable[..., Any]:
    """Resolve an existing host binding without choosing its kernel backend."""
    import torch

    operator = getattr(torch.ops.xllm_ops, name, None)
    if operator is None:
        from xllm import xllm_export  # noqa: F401

        operator = getattr(torch.ops.xllm_ops, name, None)
    if operator is None:
        raise RuntimeError(f"load the matching xLLM native library before preparing xllm_ops::{name}")
    return operator
