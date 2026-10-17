# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native RMSNorm from the package extension or an already loaded xLLM host."""

from __future__ import annotations

import importlib.util

from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT
from xllm_kernel.registry import KernelRegistry, KernelSpec

_LOADED = False


def has_package_extension() -> bool:
    """Check availability without importing Torch or loading native libraries."""
    return importlib.util.find_spec("xllm_kernel._npu") is not None


def register(registry: KernelRegistry) -> None:
    """Resolve the native dependency before publishing any callable."""
    import torch
    import torch_npu  # noqa: F401

    global _LOADED
    extension = importlib.util.find_spec("xllm_kernel._npu")
    if extension is not None:
        if extension.origin is None:
            raise RuntimeError("xllm_kernel._npu has no library path; rebuild the NPU extension")
        if not _LOADED:
            torch.ops.load_library(extension.origin)
            from xllm_kernel.fake.rms_norm import register_native_fake

            register_native_fake()
            _LOADED = True
        namespace = "xllm_kernel"
    else:
        namespace = "xllm_ops"
    ops = getattr(torch.ops, namespace)
    if not hasattr(ops, "rms_norm"):
        raise RuntimeError(
            "Native RMSNorm requires the package extension (build with XLLM_KERNEL_BUILD_NPU=1) "
            "or host operator xllm_ops::rms_norm; load xllm_export or initialize the xLLM C++ host"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(f"{namespace}::rms_norm", "PrivateUse1"):
        raise RuntimeError(f"{namespace}::rms_norm has no NPU (PrivateUse1) implementation")
    registry.register(
        KernelSpec(
            name="npu.xllm_native.rms_norm",
            contract=RMS_NORM_CONTRACT,
            device="npu",
            execution_modes=("eager", "aclgraph"),
            input_domain=f"native torch.ops.{namespace}.rms_norm tensor domain; no shape specialization",
        ),
        ops.rms_norm,
    )
