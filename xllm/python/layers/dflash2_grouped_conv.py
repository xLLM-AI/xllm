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

"""DFlash2 token-block convolution used around attention and MLP blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


def _dflash2_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    if hidden_states.dim() != 2:
        raise ValueError("DFlash2 convolution hidden states must be two-dimensional")
    if delta.shape != (hidden_states.size(0), taps, num_groups):
        raise ValueError("DFlash2 convolution coefficient shape mismatch")
    if base.shape != (taps, hidden_states.size(1)):
        raise ValueError("DFlash2 convolution base-kernel shape mismatch")
    if hidden_states.size(1) != num_groups * group_size:
        raise ValueError("DFlash2 convolution hidden-size grouping mismatch")

    num_tokens = hidden_states.size(0)
    if num_tokens % block_size:
        raise ValueError("DFlash2 convolution rows must contain complete sequence blocks")

    blocks = hidden_states.view(num_tokens, num_groups, group_size)
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    positions = torch.arange(num_tokens, dtype=torch.long, device=hidden_states.device) % block_size

    for tap in range(1, min(taps, block_size, num_tokens)):
        valid = positions[tap:].ge(tap).view(num_tokens - tap, 1, 1).to(hidden_states.dtype)
        output[tap:].add_(coefficients[tap:, tap] * blocks[:-tap] * valid)
    return output.flatten(start_dim=1)


class DFlash2GroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        if min(hidden_size, taps, group_size, block_size) <= 0:
            raise ValueError("DFlash2 convolution dimensions must be positive")
        if hidden_size % group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size")

        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, taps, hidden_size, dtype=dtype, device=device))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).view(
            hidden_states.size(0),
            2,
            self.taps,
            self.num_groups,
        )
        return self._convolve(hidden_states, coefficients[:, 0], side=0), coefficients[:, 1]

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        return _dflash2_grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )
