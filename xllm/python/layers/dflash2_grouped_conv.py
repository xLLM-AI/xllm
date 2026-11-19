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


class DFlash2GroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        num_speculative_tokens: int,
    ) -> None:
        super().__init__()
        if min(hidden_size, taps, group_size) <= 0:
            raise ValueError("DFlash2 convolution dimensions must be positive")
        if hidden_size % group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size")

        self.num_speculative_tokens = num_speculative_tokens
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
        num_tokens = hidden_states.size(0)
        num_decoding_tokens = self.num_speculative_tokens + 1
        num_sequences = num_tokens // num_decoding_tokens
        blocks = hidden_states.view(num_sequences, num_decoding_tokens, self.num_groups, self.group_size)
        coefficients = (
            self.base_kernel[side].view(1, self.taps, self.num_groups, self.group_size) + delta.unsqueeze(-1)
        ).view(num_sequences, num_decoding_tokens, self.taps, self.num_groups, self.group_size)
        output = coefficients[:, :, 0] * blocks

        for tap in range(1, min(self.taps, num_decoding_tokens, num_tokens)):
            output[:, tap:].add_(coefficients[:, tap:, tap] * blocks[:, :-tap])
        return output.reshape_as(hidden_states)
