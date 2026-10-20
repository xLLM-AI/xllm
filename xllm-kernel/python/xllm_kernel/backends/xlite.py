# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Load the pinned package bridge once; fake registrations remain operation-owned."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any

_LOADED = False
_REGISTERED: set[str] = set()
_LOCK = RLock()
_REVISION = "4a6dc3102d5778928e4d21116bc4d4694d51e584"


def load_operator(operation: str) -> Callable[..., Any]:
    """Fail during preparation if the prebuilt bridge or matching launchers are absent."""
    import torch
    import torch_npu

    global _LOADED
    with _LOCK:
        if not _LOADED:
            extension = importlib.util.find_spec("xllm_kernel._xlite")
            dependency = importlib.util.find_spec("xlite")
            if extension is None or extension.origin is None or dependency is None or dependency.origin is None:
                raise RuntimeError("build xllm-kernel with XLLM_KERNEL_BUILD_XLITE=1 against the pinned GVirt first")
            manifest_path = Path(extension.origin).with_name("_xlite_build.json")
            if not manifest_path.is_file():
                raise RuntimeError("xlite bridge build manifest is missing")
            manifest = json.loads(manifest_path.read_text())
            package = Path(dependency.origin).parent
            if (
                manifest["schema_version"] != 1
                or manifest["gvirt_revision"] != _REVISION
                or (package / ".xlite_git_head").read_text().strip() != _REVISION
            ):
                raise RuntimeError("xlite launcher revision does not match the package bridge")
            if (manifest["torch"], manifest["torch_npu"]) != (torch.__version__, torch_npu.__version__):
                raise RuntimeError("xlite bridge Torch/torch_npu versions changed; rebuild it")
            for name, expected in manifest["libraries"].items():
                if hashlib.sha256((package / "lib" / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"xlite launcher binary changed: {name}")
            torch.ops.load_library(extension.origin)
            _LOADED = True
        function = getattr(torch.ops.xllm_kernel, f"xlite_{operation}", None)
        if function is None:
            raise RuntimeError(f"rebuild xllm-kernel: bridge does not contain xlite_{operation}")
        if operation not in _REGISTERED:
            importlib.import_module(f"xllm_kernel.fake.{operation}").register_fake()
            _REGISTERED.add(operation)
        return function
