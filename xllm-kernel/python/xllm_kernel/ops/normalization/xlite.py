# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Optional xlite solution, loaded and checked only during preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT
from xllm_kernel.registry import KernelRegistry, KernelSpec

_LOADED = False
IMPLEMENTATION = "npu.xlite.rms_norm"


def register(registry: KernelRegistry) -> None:
    """Require the tested launcher ABI, then register our own Torch operator."""
    import torch
    import torch_npu

    global _LOADED
    if not _LOADED:
        extension = importlib.util.find_spec("xllm_kernel._xlite")
        dependency = importlib.util.find_spec("xlite")
        if extension is None or extension.origin is None or dependency is None or dependency.origin is None:
            raise RuntimeError(
                "xlite RMSNorm requires its optional bridge and GVirt; build xllm-kernel with "
                "XLLM_KERNEL_BUILD_XLITE=1 in the matching NPU environment before worker startup"
            )
        manifest_path = Path(extension.origin).with_name("_xlite_build.json")
        if not manifest_path.is_file():
            raise RuntimeError("xlite bridge build manifest is missing; rebuild the extension")
        manifest = json.loads(manifest_path.read_text())
        xlite = Path(dependency.origin).parent
        revision = (xlite / ".xlite_git_head").read_text().strip()
        if manifest["schema_version"] != 1 or revision != manifest["gvirt_revision"]:
            raise RuntimeError("xlite launcher revision does not match the bridge build")
        if (torch.__version__, torch_npu.__version__) != (manifest["torch"], manifest["torch_npu"]):
            raise RuntimeError("xlite bridge PyTorch/torch_npu versions changed; rebuild the extension")
        for name, expected in manifest["libraries"].items():
            if hashlib.sha256((xlite / "lib" / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"xlite launcher binary changed: {name}; rebuild the bridge")
        torch.ops.load_library(extension.origin)
        from xllm_kernel.fake.normalization import register_fake

        register_fake()
        _LOADED = True
    registry.register(
        KernelSpec(
            name=IMPLEMENTATION,
            contract=RMS_NORM_CONTRACT,
            device="npu",
            execution_modes=("eager", "aclgraph"),
            input_domain="FP16/BF16, same NPU/dtype, ND, width divisible by 64 and <=8192, numel<=2**32-1",
            solution="xlite",
        ),
        torch.ops.xllm_kernel.xlite_rms_norm,
    )
