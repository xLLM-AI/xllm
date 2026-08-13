# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""DeepSeek-V3.2 MTP model graph.

The speculative worker and MTP scheduling remain in C++.  This module only
describes the MTP model computation.  In particular, ``input_embedding`` is
the hidden state produced by the target model and supplied by the caller for
the next MTP step.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from xllm.python.layers import ColumnParallelLinear, RMSNorm
from xllm.python.models.deepseek_v32 import (
    DeepseekV3Config,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekYarnRotaryEmbedding,
)


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
        h = self.eh_proj(torch.cat((self.enorm(token_hidden), self.hnorm(rotated_embedding)), dim=-1))
        positions = positions.to(torch.int64).contiguous()
        cos_sin_cache = self.rotary.cos_sin_cache
        residual: torch.Tensor | None = None
        for layer in self.layers:
            h, residual = layer(h, residual, positions, cos_sin_cache)
        h, _ = self.norm(h, residual)
        return h


class _MtpStateDictView:
    """Add aliases for the two MTP checkpoint naming conventions."""

    def __init__(self, state_dict: object) -> None:
        self._state_dict = state_dict

    @staticmethod
    def _aliases(name: str) -> Iterable[str]:
        if name == "model.norm.weight":
            yield "model.norm.weight"
            yield "model.final_norm.weight"
            yield "model.shared_head.norm.weight"
            yield "shared_head.norm.weight"
            return
        yield name
        if name.startswith("model."):
            yield name[len("model.") :]
        else:
            yield "model." + name

    def has(self, name: str) -> bool:
        return any(self._state_dict.has(alias) for alias in self._aliases(name))

    def get_tensor(self, name: str) -> torch.Tensor:
        for alias in self._aliases(name):
            if self._state_dict.has(alias):
                return self._state_dict.get_tensor(alias)
        raise KeyError(name)


class DeepseekV32MtpForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek-V3.2 MTP calculator; scheduling stays in the C++ worker."""

    def __init__(self, config: dict) -> None:
        super().__init__(config, build_model=False)
        self.model = DeepseekV32MtpModel(self.cfg, self.dtype, self.device)

    def load_weights(self, state_dicts: list, tp_rank: int, tp_size: int) -> None:
        views = [_MtpStateDictView(state_dict) for state_dict in state_dicts]
        super().load_weights(
            views,
            tp_rank,
            tp_size,
            load_lm_head=False,
            load_embedding=False,
        )

        def find(name: str):
            for state_dict in views:
                if state_dict.has(name):
                    return state_dict
            return None

        def copy_if_present(module_name: str, *aliases: str, required: bool = False) -> bool:
            state_dict = find(module_name + ".weight")
            if state_dict is None:
                for alias in aliases:
                    state_dict = find(alias + ".weight")
                    if state_dict is not None:
                        break
            if state_dict is None:
                if required:
                    raise KeyError(f"missing required MTP weight: {module_name}.weight")
                return False
            tensor = state_dict.get_tensor(module_name + ".weight")
            parameter = self.get_parameter("model." + module_name + ".weight")
            if tensor.shape != parameter.shape and tensor.dim() == 2:
                part = tensor.shape[0] // self.cfg.tp_size
                tensor = tensor.narrow(0, self.cfg.tp_rank * part, part)
            parameter.data.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))
            return True

        copy_if_present("eh_proj", required=True)
        copy_if_present("enorm", required=True)
        copy_if_present("hnorm", required=True)
        self.model.enable_rot = copy_if_present("rot")
