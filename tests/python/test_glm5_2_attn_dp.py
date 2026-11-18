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

"""Numerical attention-DP integration with CPU kernels and real Gloo groups."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = value.abs().amax(dim=-1).clamp_min(1e-6) / 127
    return (value / scale[:, None]).round().clamp(-127, 127).to(torch.int8), scale


class _OutputProjection(nn.Module):
    def __init__(self, weight: torch.Tensor, dynamic: bool, bias: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight
        self._dynamic_activation = dynamic
        self.bias = bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = (value / 0.1).round().clamp(-127, 127) * 0.1
        return F.linear(value, self.weight) + self.bias

    def forward_quantized(self, value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return F.linear(value.float() * scale[:, None], self.weight)


def _run_attention(global_rank: int, rendezvous: str, tp_size: int, dp_size: int) -> None:
    # Spawned interpreters register the real runtime; only the arithmetic
    # kernels below are replaced by explicit CPU references.
    from xllm import xllm_export  # noqa: F401
    from xllm.python import initialize_runtime

    initialize_runtime()
    from xllm.python.distributed import collectives
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context
    from xllm.python.models import glm5_2

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=global_rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        groups = [
            dist.new_group(ranks=[tp + dp * tp_size for dp in range(dp_size)], backend="gloo") for tp in range(tp_size)
        ]
        tp_rank, dp_rank = global_rank % tp_size, global_rank // tp_size
        weight_rank = tp_rank * dp_size + dp_rank
        owner_head_count = 4 // tp_size
        collectives._groups[("dp", "cpu")] = groups[tp_rank]
        collectives._groups[("attn_dp", "cpu")] = dist.group.WORLD
        cfg = glm5_2.Glm52Config.from_dict(
            dict(
                hidden_size=4,
                n_heads=4,
                tp_size=tp_size,
                tp_rank=tp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                world_size=4,
                enable_attn_dp_weight_sharding=True,
                q_lora_rank=4,
                kv_lora_rank=2,
                qk_nope_head_dim=2,
                qk_rope_head_dim=2,
                v_head_dim=2,
                index_head_dim=4,
                index_n_heads=1,
                index_topk=1,
                n_layers=1,
            )
        )
        counts, padded = (1, 3, 2, 1)[:dp_size], 3
        global_tokens = dp_size * padded
        hidden = torch.arange(global_tokens * 4).float().reshape(global_tokens, 4) / 20
        for owner, count in enumerate(counts):
            hidden[owner * padded + count : (owner + 1) * padded].zero_()
        q_weight = torch.arange(64).float().reshape(16, 4) / 100
        kv_weight = torch.arange(16).float().reshape(4, 4) / 50
        uk = torch.arange(16).float().reshape(4, 2, 2) / 30
        uv = torch.arange(16).float().reshape(4, 2, 2) / 20
        o_weight = (torch.arange(32).float().reshape(4, 8) - 15) / 50
        q = F.linear(hidden, q_weight).reshape(global_tokens, 4, 4)
        latent = torch.einsum("thd,hdk->thk", q[..., :2], uk)
        kv = F.linear(hidden, kv_weight)[..., :2]
        full_attention = latent + kv[:, None]
        bias = torch.arange(4).float() / 10
        head = slice(weight_rank, weight_rank + 1)
        rows = slice(dp_rank * padded, dp_rank * padded + counts[dp_rank])
        owner_heads = slice(tp_rank * owner_head_count, (tp_rank + 1) * owner_head_count)
        rope = (
            torch.ones(global_tokens, 1),
            torch.zeros(global_tokens, 1),
            torch.ones(global_tokens, 1, 1, 2),
            torch.zeros(global_tokens, 1, 1, 2),
        )

        def execute(
            q_latent: torch.Tensor,
            q_pe: torch.Tensor,
            k_latent: torch.Tensor,
            k_pe: torch.Tensor,
            attention: nn.Module,
            topk: torch.Tensor,
        ) -> torch.Tensor:
            assert q_latent.shape == (counts[dp_rank], owner_head_count, 2)
            assert k_latent.shape == (counts[dp_rank], 1, 2)
            torch.testing.assert_close(q_latent, latent[rows, owner_heads])
            torch.testing.assert_close(k_latent[:, 0], kv[rows])
            assert topk.shape[0] == counts[dp_rank]
            return q_latent + k_latent

        for dynamic in (False, True):
            attention = glm5_2.Glm52MLAAttention(cfg, 0, torch.float32, torch.device("cpu"))
            attention.q_a_proj = nn.Identity()
            attention.q_a_layernorm = nn.Identity()
            attention.q_b_proj = nn.Linear(4, 4, bias=False)
            attention.q_b_proj.weight.data.copy_(q_weight.reshape(4, 4, 4)[head].reshape(4, 4))
            attention.kv_a_proj_with_mqa = nn.Linear(4, 4, bias=False)
            attention.kv_a_proj_with_mqa.weight.data.copy_(kv_weight)
            attention.kv_a_layernorm = nn.Identity()
            attention.W_UK.copy_(uk[head])
            attention.W_UV.copy_(uv[head])
            attention.W_UV_owner = uv[owner_heads]
            attention.indexer = None
            attention.o_proj = _OutputProjection(
                o_weight[:, weight_rank * 2 : weight_rank * 2 + 2],
                dynamic,
                torch.zeros(4) if dynamic or weight_rank else bias,
            )
            backend = SimpleNamespace(execute_mla=execute)
            metadata = SimpleNamespace(dp_execution_token_counts=counts, is_prefill=False, is_chunked_prefill=False)
            context = ForwardContext(
                attention_backend=backend, device=torch.device("cpu"), metadata=metadata, layer_caches=[]
            )
            with (
                forward_context(context),
                patch.object(glm5_2, "matmul_ein_sum", side_effect=lambda a, b: torch.einsum("thd,hdk->thk", a, b)),
                patch.object(glm5_2.kernels, "dynamic_quant", side_effect=_quantize),
                patch.object(glm5_2, "_interleave_rope_with", side_effect=lambda q, cos, sin: q),
                patch.object(collectives, "_all_reduce", dist.all_reduce),
            ):
                result, _ = attention(hidden, *rope, (rope[2], rope[3]), torch.zeros(counts[dp_rank], 1, 1))
            expected = torch.zeros(global_tokens, 4)
            owner_features = owner_head_count * 2
            for owner_tp in range(tp_size):
                owner_head = slice(owner_tp * owner_head_count, (owner_tp + 1) * owner_head_count)
                value = torch.einsum("thd,hdk->thk", full_attention[:, owner_head], uv[owner_head]).reshape(
                    global_tokens, owner_features
                )
                if dynamic:
                    value_i8, scale = _quantize(value)
                    value = value_i8.float() * scale[:, None]
                else:
                    value = (value / 0.1).round().clamp(-127, 127) * 0.1
                expected += F.linear(value, o_weight[:, owner_tp * owner_features : (owner_tp + 1) * owner_features])
            if not dynamic:
                expected += bias
            torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    finally:
        collectives._groups.clear()
        dist.destroy_process_group()


@pytest.mark.parametrize(("tp_size", "dp_size"), [(2, 2), (1, 4)])
def test_attn_dp_attention_matches_tp_reference_with_uneven_dp_rows(tmp_path: Path, tp_size: int, dp_size: int) -> None:
    torch.multiprocessing.spawn(_run_attention, args=(str(tmp_path / "gloo"), tp_size, dp_size), nprocs=4, join=True)
