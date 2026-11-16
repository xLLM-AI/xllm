# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Explicit package-owned native loading; importing this module is SDK-free."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from xllm_kernel.platform import current_platform


def _library_path() -> Path:
    return (Path(__file__).resolve().parent / "libxllm_kernel.so").resolve()


def resource_root() -> Path:
    """Return the package OPP root, never an xLLM host or source-tree path."""
    return _library_path().parent / "opp"


@cache
def load_native() -> Path:
    """Load package schemas once at startup, before backend preparation.

    Native-only programs link the same shared library directly. Resource
    registration precedes Torch/device initialization. PR02 supplies the first
    custom OPP component; an empty resource root is valid for PR01.
    """
    library = _library_path()
    if not library.is_file():
        raise RuntimeError("xllm-kernel native library is absent; rebuild with XLLM_KERNEL_BUILD_NATIVE=1")
    vendors = resource_root() / "vendors"
    if vendors.is_dir():
        roots = [str(path) for path in sorted(vendors.iterdir()) if path.is_dir()]
        existing = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(roots + [p for p in existing if p]))
    import torch

    # Hosts may disable Torch backend autoload before importing the runtime.
    if current_platform.is_npu():
        import torch_npu  # noqa: F401

    torch.ops.load_library(str(library))
    return library
