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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import lcm

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3RotaryPosEmbed,
)
from torch import nn
from torch.nn import functional as F

from xllm.python import _load_kernel_package
from xllm.python.dit.parallel import (
    ParallelLinear,
    SequenceParallel,
    StateDictShard,
    TensorParallel,
    load_component_weights,
)

_LaserAttention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]


@dataclass(frozen=True)
class PackedLayout:
    positions: torch.Tensor
    tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    timestep_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class PackedConditioning:
    text: torch.Tensor
    rotary: tuple[torch.Tensor, torch.Tensor]
    layout: PackedLayout
    local_start: int
    local_rows: int
    padded_rows: int


def _apply_rotary(states: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rotary
    rotary_dims = cos.size(-1)
    first, second = states[..., :rotary_dims].chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    position = states[..., :rotary_dims] * cos[None, :, None] + rotated * sin[None, :, None]
    return torch.cat((position, states[..., rotary_dims:]), dim=-1)


def _dense_attention(
    attn: MiniMaxH3Attention,
    states: torch.Tensor,
    rotary: tuple[torch.Tensor, torch.Tensor] | None = None,
    sp: SequenceParallel | None = None,
    live_rows: int | None = None,
    laser: _LaserAttention | None = None,
) -> torch.Tensor:
    query = attn.norm_q(attn.to_q(states).unflatten(-1, (-1, attn.head_dim)))
    key = attn.norm_k(attn.to_k(states).unflatten(-1, (-1, attn.head_dim)))
    value = attn.to_v(states).unflatten(-1, (-1, attn.head_dim))
    if rotary is not None:
        query = _apply_rotary(query, rotary)
        key = _apply_rotary(key, rotary)
    if sp is not None:
        query = sp.exchange(query, True)
        key = sp.exchange(key, True)
        value = sp.exchange(value, True)
    rows, heads, head_dim = query.shape[1:]
    if live_rows is None:
        live_rows = rows
    if laser is not None and live_rows >= 2048 and head_dim <= 128:
        # Exclude sequence-parallel padding from the softmax, then restore the
        # full sequence for the reverse all-to-all.
        attended = laser(
            query[:, :live_rows].transpose(1, 2),
            key[:, :live_rows].transpose(1, 2),
            value[:, :live_rows].transpose(1, 2),
            head_dim**-0.5,
        ).transpose(1, 2)
        if live_rows != rows:
            attended = F.pad(attended, (0, 0, 0, 0, 0, rows - live_rows))
        attended = attended.contiguous()
    else:
        # Padding is a separate document in the packed dense NPU attention.
        sequence_ends = [live_rows] if live_rows == rows else [live_rows, rows]
        if sp is not None and live_rows >= 24 * 1024 and head_dim == 128 and query.dtype == torch.bfloat16:
            attended = torch.ops.npu.npu_fusion_attention(
                query[0].contiguous(),
                key[0].contiguous(),
                value[0].contiguous(),
                heads,
                input_layout="TND",
                scale=head_dim**-0.5,
                actual_seq_qlen=sequence_ends,
                actual_seq_kvlen=sequence_ends,
            )[0]
        else:
            attended, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query[0].contiguous(),
                key[0].contiguous(),
                value[0].contiguous(),
                actual_seq_lengths=sequence_ends,
                actual_seq_lengths_kv=sequence_ends,
                num_heads=heads,
                num_key_value_heads=heads,
                scale=head_dim**-0.5,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                input_layout="TND",
            )
        attended = attended.unsqueeze(0)
    if sp is not None:
        attended = sp.exchange(attended, False)
    return attn.to_out[0](attended.flatten(-2))


class _RefinerAttention:
    def __init__(self, laser: _LaserAttention | None) -> None:
        self._laser = laser

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _dense_attention(attn, hidden_states, rotary_emb, laser=self._laser)


class MiniMaxH3Denoiser(nn.Module):
    def __init__(
        self,
        path: str,
        shards: Sequence[StateDictShard],
        device: torch.device,
        tp: TensorParallel,
        sp: SequenceParallel,
        *,
        laser_attention_enabled: bool = False,
    ) -> None:
        super().__init__()
        self._sp = sp
        self._laser: _LaserAttention | None = None
        if laser_attention_enabled:
            _load_kernel_package().initialize(
                device=device.type, execution_mode="eager", operations=("laser_attention",)
            )
            from xllm_kernel.ops.attention.laser import laser_attention

            self._laser = laser_attention
        config = MiniMaxH3Transformer3DModel.load_config(path)
        with torch.device("meta"):
            model = MiniMaxH3Transformer3DModel.from_config(config)
        cfg = model.config
        if (
            cfg.num_attention_heads % (tp.size * sp.size)
            or cfg.in_channels * cfg.patch_size[1] * cfg.patch_size[2] % tp.size
            or cfg.audio_in_channels % tp.size
            or tuple(cfg.patch_size[:1]) != (1,)
            or cfg.text_dim != 5120
        ):
            raise ValueError("MiniMax-H3 checkpoint dimensions are incompatible with the T2VA topology")

        for block in [*model.transformer_blocks, *model.token_refiner.refiner_blocks]:
            attn = block.attn
            attn.to_q = ParallelLinear(attn.to_q, tp)
            attn.to_k = ParallelLinear(attn.to_k, tp)
            attn.to_v = ParallelLinear(attn.to_v, tp)
            attn.to_out[0] = ParallelLinear(attn.to_out[0], tp, row=True)
            block.ff.net[0].proj = ParallelLinear(block.ff.net[0].proj, tp, packed=True)
            block.ff.net[2] = ParallelLinear(block.ff.net[2], tp, row=True)
        for block in model.token_refiner.refiner_blocks:
            block.attn.set_processor(_RefinerAttention(self._laser))
        for block in model.transformer_blocks:
            block.adaln_proj.linear = ParallelLinear(block.adaln_proj.linear, tp, gather=True)

        model.time_embedder.linear_1 = ParallelLinear(model.time_embedder.linear_1, tp)
        model.time_embedder.linear_2 = ParallelLinear(model.time_embedder.linear_2, tp, row=True)
        model.norm_out.linear = ParallelLinear(model.norm_out.linear, tp, gather=True)

        def weight_dtype(name: str) -> torch.dtype:
            fp32 = ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "time_embedder")
            return torch.float32 if name.split(".", 1)[0] in fp32 else torch.bfloat16

        load_component_weights(model, shards, device, weight_dtype)
        # The non-persistent frequency buffer is initialized on CPU, then moved.
        model.rope = MiniMaxH3RotaryPosEmbed(cfg.rope_freq_dim, cfg.rope_theta).to(device)
        self.model = model

    def prepare_conditioning(self, prompt_embed: torch.Tensor, layout: PackedLayout) -> PackedConditioning:
        text = self.model.token_refiner(self.model.context_embedder(prompt_embed.to(torch.bfloat16)))
        alignment = lcm(64, self._sp.size)
        rows = layout.tags.numel()
        padded_rows = rows + (-rows) % alignment
        local_rows = padded_rows // self._sp.size
        local_start = self._sp.rank * local_rows
        positions = F.pad(layout.positions, (0, 0, 0, padded_rows - rows))
        positions = positions.narrow(0, local_start, local_rows)
        cos, sin = self.model.rope(positions)
        return PackedConditioning(
            text,
            (cos.to(torch.bfloat16), sin.to(torch.bfloat16)),
            layout,
            local_start,
            local_rows,
            padded_rows,
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        conditioning: PackedConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        layout = conditioning.layout
        hidden = conditioning.text.new_zeros((1, conditioning.padded_rows, conditioning.text.size(-1)))
        hidden.index_copy_(1, layout.text_indices, conditioning.text)
        hidden.index_copy_(1, layout.video_indices, model.proj_in(video).to(hidden.dtype))
        hidden.index_copy_(1, layout.audio_indices, model.audio_proj_in(audio).to(hidden.dtype))
        hidden = hidden.narrow(1, conditioning.local_start, conditioning.local_rows)
        temb = model.time_embedder(model.time_proj(timestep))

        padding = conditioning.padded_rows - layout.tags.numel()
        modulation_indices = F.pad(timestep_indices * 3 + layout.tags, (0, padding))
        modulation_indices = modulation_indices.narrow(0, conditioning.local_start, conditioning.local_rows)
        for block in model.transformer_blocks:
            shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = block.adaln_proj(temb)
            attn_scale = scale_attn.index_select(0, modulation_indices)
            attn_shift = shift_attn.index_select(0, modulation_indices)
            attn_input = block.norm1(hidden) * (1 + attn_scale) + attn_shift
            attended = _dense_attention(
                block.attn, attn_input, conditioning.rotary, self._sp, layout.tags.numel(), self._laser
            )
            hidden = hidden + gate_attn.index_select(0, modulation_indices) * attended
            ff_scale = scale_ff.index_select(0, modulation_indices)
            ff_shift = shift_ff.index_select(0, modulation_indices)
            ff_input = block.norm2(hidden) * (1 + ff_scale) + ff_shift
            hidden = hidden + gate_ff.index_select(0, modulation_indices) * block.ff(ff_input)

        hidden = self._sp.gather(hidden.contiguous())[:, : layout.tags.numel()]
        hidden = model.norm_out(hidden, temb, timestep_indices).float()
        return (
            model.proj_out(hidden.index_select(1, layout.video_indices)),
            model.audio_proj_out(hidden.index_select(1, layout.audio_indices)),
        )
