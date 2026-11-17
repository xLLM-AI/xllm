# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Preparation-time Torch operator registration for the xllm_kernel namespace."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

_library: torch.library.Library | None = None


def register_custom_op(
    op_name: str,
    op_func: Callable[..., Any] | None = None,
    *,
    mutates_args: Iterable[str] | None = None,
    dispatch_key: str | None = None,
) -> Callable[..., Any]:
    """Register a real Python kernel or retrieve an existing C++ op.

    ``op_name`` is an unqualified name in ``xllm_kernel``. New Python kernels
    require explicit mutation metadata and a Torch dispatch key, such as
    ``CPU`` or ``PrivateUse1``. Without ``op_func``, the schema and device
    implementation must already be registered by the extension; mutation and
    dispatch metadata must be omitted. Call once during backend preparation.
    """
    import torch

    if op_func is None:
        if mutates_args is not None or dispatch_key is not None:
            raise ValueError("Existing operators retain their registered mutation and dispatch metadata")
        return getattr(torch.ops.xllm_kernel, op_name)
    if mutates_args is None or dispatch_key is None:
        raise ValueError("Python operators require explicit mutates_args and dispatch_key")

    global _library
    if _library is None:
        _library = torch.library.Library("xllm_kernel", "FRAGMENT")
    schema = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    _library.define(op_name + schema)
    _library.impl(op_name, op_func, dispatch_key)
    return getattr(torch.ops.xllm_kernel, op_name)
