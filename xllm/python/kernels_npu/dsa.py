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

"""DeepSeek-V4 compressed KV and shared sparse attention kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from xllm_kernel.ops.attention.dsv4 import (
    compressor as compressor,
)
from xllm_kernel.ops.attention.dsv4 import (
    sparse_attn_sharedkv as sparse_attn_sharedkv,
)
from xllm_kernel.ops.attention.dsv4 import (
    sparse_attn_sharedkv_metadata as sparse_attn_sharedkv_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable

supports_sparse_flash_mla_swa: Callable[[], bool] = torch.ops.xllm_ops.supports_sparse_flash_mla_swa
sparse_flash_mla_swa_metadata: Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = torch.ops.xllm_ops.sparse_flash_mla_swa_metadata
sparse_flash_mla_swa: Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ],
    torch.Tensor,
] = torch.ops.xllm_ops.sparse_flash_mla_swa
