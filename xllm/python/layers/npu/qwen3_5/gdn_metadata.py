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

"""Model-visible execution metadata for the Qwen3.5 NPU GDN path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch


class GdnStateCache(Protocol):
    """State tensors validated by the GDN metadata builder at cache binding."""

    conv: torch.Tensor
    ssm: torch.Tensor


@dataclass(frozen=True, slots=True)
class GdnMetadata:
    state_caches: Mapping[int, GdnStateCache]


@dataclass(frozen=True, slots=True)
class GdnPrefillMetadata(GdnMetadata):
    conv_read_indices: torch.Tensor
    conv_write_indices: torch.Tensor
    ssm_read_indices: torch.Tensor
    ssm_write_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    num_matrices: int


@dataclass(frozen=True, slots=True)
class GdnDecodeMetadata(GdnMetadata):
    read_state_indices: torch.Tensor
    write_state_indices: torch.Tensor
