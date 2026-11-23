# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind static decode graph metadata refresh."""

import torch

from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel import _npu_binding  # noqa: F401
    from xllm_kernel.ops.kv_cache import UPDATE_DECODE_GRAPH_METADATA_CONTRACT

    register_kernel(
        name="npu.torch.update_decode_graph_metadata",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Contiguous NPU decode inputs and preallocated static graph buffers.",
        solution="torch",
        binding="cpp",
        contract=UPDATE_DECODE_GRAPH_METADATA_CONTRACT,
    )(torch.ops.xllm_kernel.update_decode_graph_metadata.default)
