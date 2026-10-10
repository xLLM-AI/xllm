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

"""Functional RMSNorm with a fixed native or explicitly selected xlite binding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.registry import KernelRegistry, OpContract, PreparedKernel

if TYPE_CHECKING:
    import torch

RMS_NORM_CONTRACT = OpContract(
    operation="rms_norm",
    version=1,
    mutates=(),
    aliases_inputs=False,
    numerical_mode="xllm_native",
)
_prepared: PreparedKernel | None = None
_configuration: tuple[str, str | None] | None = None


def initialize(*, device: str, implementation: str | None = None) -> PreparedKernel:
    """Bind an NPU operator once, before model import or graph capture.

    The default adapter requires the host's ``torch.ops.xllm_ops`` registration.
    The explicit xlite solution loads its package-owned, prebuilt extension
    here. Dependency checks and binding happen only during preparation, with
    no runtime compilation or execution-error fallback. Changing policy
    requires a fresh worker and fresh graphs.
    """
    global _prepared, _configuration
    configuration = (device, implementation)
    if _prepared is not None:
        if configuration != _configuration:
            raise RuntimeError("xllm_kernel is already initialized; restart the worker to change kernel policy")
        return _prepared
    if device != "npu":
        raise ValueError(f"xllm_kernel does not yet provide a {device!r} adapter")

    if implementation == "npu.xlite.rms_norm":
        from xllm_kernel.ops.normalization.xlite import register
    else:
        from xllm_kernel.ops.normalization.npu import register

    registry = KernelRegistry()
    register(registry)
    registry.freeze()
    prepared = registry.prepare(
        RMS_NORM_CONTRACT,
        device=device,
        execution_mode="aclgraph",
        implementation=implementation,
    )
    _configuration = configuration
    _prepared = prepared
    return prepared


def prepare_rms_norm() -> PreparedKernel:
    """Return the existing binding; this never searches for an implementation."""
    if _prepared is None:
        raise RuntimeError("call xllm_kernel.initialize(device='npu') after loading the host native Torch ops")
    return _prepared


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last dimension without modifying or aliasing the inputs.

    The default binding preserves the host operator's complete tensor domain.
    Explicit xlite selection restricts inputs to the domain in its KernelSpec;
    its bridge validates tensors before launch and never falls back. Both have
    no residual, gamma offset, quantization, or out-buffer semantics. Numerical
    equivalence between implementations is tolerance-based, not bitwise.
    """
    if _prepared is None:
        raise RuntimeError("xllm_kernel runtime is not initialized")
    return _prepared.function(input, weight, eps)
