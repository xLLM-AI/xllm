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
from dataclasses import dataclass, field
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
    # Physical rows per Linear group: a recurrent state owns one shared slot
    # per group. Ordinary storage has a single group.
    conv_read_indices: tuple[torch.Tensor, ...]
    conv_write_indices: tuple[torch.Tensor, ...]
    ssm_read_indices: tuple[torch.Tensor, ...]
    ssm_write_indices: tuple[torch.Tensor, ...]
    cu_seqlens: torch.Tensor
    num_matrices: int
    # Linear group of each state cache, empty for a single group.
    layer_groups: Mapping[int, int] = field(default_factory=dict)
    shared_checkpoint_stride: int | None = None


@dataclass(frozen=True, slots=True)
class GdnDecodeMetadata(GdnMetadata):
    read_state_indices: torch.Tensor
    write_state_indices: torch.Tensor
    layer_groups: Mapping[int, int] = field(default_factory=dict)
    # [layer group, batch, (Conv physical slot, SSM physical checkpoint row)].
    packed_read_state_indices: torch.Tensor | None = None
    packed_write_state_indices: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class GdnVerifyMetadata(GdnMetadata):
    state_indices: torch.Tensor
    num_accepted_tokens: torch.Tensor
    query_width: int
    # Packed token -> dense [request, verify position] row. None for uniform
    # batches, which can be reshaped without padding or copying.
    packed_to_dense: torch.Tensor | None = None
    # Device-built inverse mapping for adaptive graph rows. Missing dense
    # positions use the appended zero row at packed_token_count; padding in
    # packed_to_dense uses the appended zero row at batch_size * query_width.
    dense_to_packed: torch.Tensor | None = None
    # Device prefix endpoints for the fused packed verifier. Cache strides
    # and previous acceptance counts remain independent of current widths.
    cu_seqlens: torch.Tensor | None = None
    layer_groups: Mapping[int, int] = field(default_factory=dict)
    packed_state_indices: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class GdnDummyMetadata(GdnMetadata):
    pass
