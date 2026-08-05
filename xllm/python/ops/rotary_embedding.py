# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public rotary-embedding graph operators."""

from __future__ import annotations

import torch

from xllm.python import platform


@torch.library.custom_op(
    "xllm_python::interleaved_rotary_embedding", mutates_args=()
)
def interleaved_rotary_embedding(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    if not platform.is_npu():
        raise NotImplementedError("interleaved rotary embedding requires NPU")
    from xllm.python.ops.npu.rotary_embedding import (
        interleaved_rotary_embedding as npu_interleaved_rotary_embedding,
    )

    return npu_interleaved_rotary_embedding(value, cosine, sine)


@interleaved_rotary_embedding.register_fake
def _interleaved_rotary_embedding_fake(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    del cosine, sine
    return torch.empty_like(value)


__all__ = ["interleaved_rotary_embedding"]
