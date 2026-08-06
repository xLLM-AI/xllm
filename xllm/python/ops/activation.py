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

"""Graph interfaces for Python-authored activation operators."""

from __future__ import annotations

import torch


@torch.library.custom_op("xllm_triton::silu_and_mul", mutates_args=())
def silu_and_mul(value: torch.Tensor) -> torch.Tensor:
    """Run the CUDA Triton gated SiLU kernel as one graph node."""
    from xllm.python.kernels.triton.cuda.silu_and_mul import (
        silu_and_mul as triton_silu_and_mul,
    )

    return triton_silu_and_mul(value)


@silu_and_mul.register_fake
def _silu_and_mul_fake(value: torch.Tensor) -> torch.Tensor:
    shape = list(value.shape)
    shape[-1] //= 2
    return value.new_empty(shape)
