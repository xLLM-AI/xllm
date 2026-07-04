# Copyright 2025-2026 The xLLM Authors.
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

"""Reusable model layers (RMSNorm, rotary embedding, tensor-parallel linear /
embedding) for xllm_models.

At tp_size==1 the parallel layers below hold full-size weights and skip all
collectives, so they are numerically identical to plain ``nn.Linear`` /
``nn.Embedding`` and preserve the first-version single-card byte parity. At
tp_size>1 each rank holds a per-partition shard and the layer inserts the same
all-reduce / all-gather the native C++ parallel layers use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm_models import ops


class RMSNorm(nn.Module):
    """RMSNorm with optional fused residual-add, matching xLLM's apply_norm.

    - ``forward(x)`` -> normed x
    - ``forward(x, residual)`` -> (normed(x + residual), x + residual)
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return ops.rms_norm(x, self.weight, self.eps)
        return ops.fused_add_rms_norm(x, residual, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
    """Holds the NEOX-style RoPE cos/sin cache in the exact layout the fused
    ``xllm_ops.fused_qk_norm_rope`` kernel expects.

    ``cos_sin_cache`` is a single ``[max_position, head_dim]`` tensor whose first
    ``head_dim/2`` columns are ``cos(freqs)`` and last ``head_dim/2`` are
    ``sin(freqs)`` (``freqs = outer(positions, inv_freq)``). This matches C++
    ``MRotaryEmbedding::precomputed_cos_sin_cache()`` so both paths use identical
    rotary tables. Built on ``device`` in the model dtype to match C++ exactly.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        rope_theta: float,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_position, head_dim/2]
        # [cos(freqs) | sin(freqs)] -> [max_position, head_dim].
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        if dtype is not None:
            cos_sin_cache = cos_sin_cache.to(dtype)
        self.register_buffer(
            "cos_sin_cache", cos_sin_cache.contiguous(), persistent=False
        )


# --------------------------------------------------------------------------
# Tensor parallelism (WS1)
# --------------------------------------------------------------------------
def shard_tensor(
    tensor: torch.Tensor, dim: int, rank: int, world_size: int
) -> torch.Tensor:
    """Shard ``tensor`` along ``dim`` for ``rank``, mirroring the native
    ``StateDict::get_sharded_tensor`` exactly (state_dict.cpp): ``world_size==1``
    or a ``size(dim)`` smaller than ``world_size`` returns the whole tensor (the
    small-tensor escape hatch for norms / bias); otherwise the equal
    ``tensor.chunk(world_size, dim)[rank]`` — the same torch op as C++, so each
    rank's shard is byte-identical to the native path.
    """
    if world_size == 1:
        return tensor
    if tensor.size(dim) < world_size:
        return tensor
    assert tensor.size(dim) % world_size == 0, (
        f"cannot evenly shard dim {dim} of size {tensor.size(dim)} "
        f"across {world_size} ranks"
    )
    return tensor.chunk(world_size, dim=dim)[rank].contiguous()


class ColumnParallelLinear(nn.Module):
    """Linear sharded on the output dim (dim 0): each rank owns
    ``[out_per_partition, in]`` and computes its slice of the output. No
    communication unless ``gather_output`` (then an all-gather along the last
    dim reconstructs the full output — used by lm_head). Bias-free (Qwen3
    projections carry no bias). Mirrors native ColumnParallelLinear /
    QKVParallelLinear (which set gather_output=False so the following
    RowParallel all-reduce combines the partial outputs).
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_partition: int,
        tp_size: int,
        gather_output: bool = False,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.gather_output = gather_output
        self.weight = nn.Parameter(
            torch.empty(
                out_features_per_partition,
                in_features,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight)
        if self.gather_output and self.tp_size > 1:
            out = ops.all_gather(out, dim=-1, world_size=self.tp_size)
        return out


class RowParallelLinear(nn.Module):
    """Linear sharded on the input dim (dim 1): each rank owns
    ``[out, in_per_partition]`` and consumes its slice of an already-partitioned
    input, producing a partial output that is SUM all-reduced across the TP
    group. Bias-free (Qwen3). Mirrors native RowParallelLinear (o_proj /
    down_proj with enable_result_reduction=true).
    """

    def __init__(
        self,
        in_features_per_partition: int,
        out_features: int,
        tp_size: int,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features_per_partition,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            out = ops.all_reduce(out)
        return out


class HiddenParallelEmbedding(nn.Module):
    """Embedding sharded on the hidden/embedding dim (dim 1), NOT the vocab dim:
    each rank owns ``[vocab, hidden_per_partition]`` (the full vocab rows, a
    slice of hidden columns) and all-gathers along the last dim to rebuild the
    full hidden vector. Reproduces native WordEmbedding
    (word_embedding_impl.cpp: dim-1 shard + gather), required for byte parity.
    """

    def __init__(
        self,
        num_embeddings: int,
        hidden_per_partition: int,
        tp_size: int,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.tp_size = tp_size
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                hidden_per_partition,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(input_ids, self.weight)
        if self.tp_size > 1:
            out = ops.all_gather(out, dim=-1, world_size=self.tp_size)
        return out
