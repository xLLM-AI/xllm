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

"""Functional RMSNorm with explicit ACLNN and xlite implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract, PreparedKernel

if TYPE_CHECKING:
    import torch

RMS_NORM_CONTRACT = OpContract(
    operation="rms_norm",
    version=1,
    mutates=(),
    aliases_inputs=False,
    numerical_mode="fp32_norm_cast",
)
OPERATOR = Operator(
    RMS_NORM_CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.rms_norm.aclnn:load_xllm",
            "ACLNN RMSNorm through the host xllm_ops schema; no shape specialization",
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.rms_norm.aclnn:load_torch_npu",
            "torch_npu RMSNorm tensor domain",
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.rms_norm.xlite:load",
            "FP16/BF16, same NPU/dtype, ND, width divisible by 64 and <=8192, numel<=2**32-1",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
_prepared: PreparedKernel | None = None
_configuration: tuple[str, str] | None = None


def initialize(*, device: str, implementation: str | None = None) -> PreparedKernel:
    """Bind an NPU operator once, before model import or graph capture.

    The default adapter requires the host's ``torch.ops.xllm_ops`` registration.
    The explicit xlite solution loads its package-owned, prebuilt extension
    here. Dependency checks and binding happen only during preparation, with
    no runtime compilation or execution-error fallback. Changing policy
    requires a fresh worker and fresh graphs.
    """
    global _prepared, _configuration
    if implementation is None:
        implementation = "npu.aclnn.rms_norm.xllm"
    configuration = (device, implementation)
    if _prepared is not None:
        if configuration != _configuration:
            raise RuntimeError("xllm_kernel is already initialized; restart the worker to change kernel policy")
        return _prepared
    if device != "npu":
        raise ValueError(f"xllm_kernel does not yet provide a {device!r} adapter")

    prepared = prepare(device, "aclgraph", implementation)
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
