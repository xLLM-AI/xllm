# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""DeepSeek-V3.2 MTP model graph.

The speculative worker and MTP scheduling remain in C++.  This module only
describes the MTP model computation.  In particular, ``input_embedding`` is
the hidden state produced by the target model and supplied by the caller for
the next MTP step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.models.deepseek_v32 import (
    DeepseekV3Config,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekYarnRotaryEmbedding,
)
from xllm.python.models.weight_utils import W8A8WeightLoader

# The loader strips "model.", so a bare "shared_head.norm.weight" needs no entry.
_MTP_NORM_ALIASES: dict[str, tuple[str, ...]] = {
    "model.norm.weight": (
        "model.norm.weight",
        "model.final_norm.weight",
        "model.shared_head.norm.weight",
    ),
}


class DeepseekV32MtpModel(nn.Module):
    """MTP body matching ``MtpModelImplBase`` and ``DeepseekV32MtpModel``."""

    def __init__(self, cfg: DeepseekV3Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        tp = cfg.tp_size
        assert cfg.hidden_size % tp == 0

        self.cfg = cfg
        self.embed_tokens: nn.Module | None = None
        self.eh_proj = ColumnParallelLinear(
            2 * cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.rot = ColumnParallelLinear(
            cfg.hidden_size,
            cfg.hidden_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )
        self.enorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.hnorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.layers = nn.ModuleList([DeepseekV3DecoderLayer(cfg, i, dtype, device) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.rotary = DeepseekYarnRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.original_max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.rope_mscale,
            cfg.rope_mscale_all_dim,
            dtype=dtype,
            device=device,
        )
        self.enable_rot = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.embed_tokens is not None
        token_hidden = self.embed_tokens(input_ids)
        if input_embedding is None:
            input_embedding = token_hidden

        rotated_embedding = self.rot(input_embedding) if self.enable_rot else input_embedding
        enorm_out = self.enorm(token_hidden)
        hnorm_out = self.hnorm(rotated_embedding)

        h = self.eh_proj(torch.cat((enorm_out, hnorm_out), dim=-1))

        positions = positions.to(torch.int64).contiguous()
        half_rope_cos, half_rope_sin, rope_cos, rope_sin = self.rotary(positions)
        residual: torch.Tensor | None = None
        for layer in self.layers:
            h, residual = layer(
                h,
                residual,
                half_rope_cos,
                half_rope_sin,
                rope_cos,
                rope_sin,
            )
        h, _ = self.norm(h, residual)
        return h


class DeepseekV32MtpForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek-V3.2 MTP calculator; scheduling stays in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.model = DeepseekV32MtpModel(self.cfg, self.dtype, self.device)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        loader = W8A8WeightLoader(
            self,
            state_dicts,
            self.cfg.tp_size,
            self.cfg.tp_rank,
            src_prefixes=("", "model."),
            name_aliases=_MTP_NORM_ALIASES,
        )
        super().load_weights(
            state_dicts,
            tp_rank,
            tp_size,
            load_lm_head=False,
            load_embedding=False,
            loader=loader,
        )

        def copy_if_present(module_name: str, required: bool = False) -> bool:
            key = module_name + ".weight"
            if not loader.has(key):
                if required:
                    raise KeyError(f"missing required MTP weight: {key}")
                return False
            tensor = loader.load_tensor(key)
            parameter = self.get_parameter("model." + key)
            if tensor.shape != parameter.shape and tensor.dim() == 2:
                tensor = loader.shard(tensor, dim=0)
            loader.copy_in("model." + key, tensor)
            return True

        copy_if_present("eh_proj", required=True)
        copy_if_present("enorm", required=True)
        copy_if_present("hnorm", required=True)
        self.model.enable_rot = copy_if_present("rot")
