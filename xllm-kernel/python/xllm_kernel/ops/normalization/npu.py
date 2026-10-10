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

"""Adapter for a native Torch op already loaded by the xLLM NPU host."""

from __future__ import annotations

from xllm_kernel.ops.normalization import RMS_NORM_CONTRACT
from xllm_kernel.registry import KernelRegistry, KernelSpec


def register(registry: KernelRegistry) -> None:
    """Resolve the native dependency before publishing any callable."""
    import torch
    import torch_npu  # noqa: F401

    if not hasattr(torch.ops.xllm_ops, "rms_norm"):
        raise RuntimeError(
            "xllm_kernel requires host operator xllm_ops::rms_norm; "
            "load xllm_export or initialize the xLLM C++ host before kernel initialization"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key("xllm_ops::rms_norm", "PrivateUse1"):
        raise RuntimeError("xllm_ops::rms_norm has no NPU (PrivateUse1) implementation in this host")
    registry.register(
        KernelSpec(
            name="npu.xllm_native.rms_norm",
            contract=RMS_NORM_CONTRACT,
            device="npu",
            execution_modes=("eager", "aclgraph"),
            input_domain="native torch.ops.xllm_ops.rms_norm tensor domain; no shape specialization",
        ),
        torch.ops.xllm_ops.rms_norm,
    )
