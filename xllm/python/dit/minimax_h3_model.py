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
from typing import Protocol

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3RotaryPosEmbed,
)
from torch import nn
from torch.nn import functional as F

from xllm.python import _load_kernel_package

_WEIGHT_TRANSFER_BATCH_BYTES = 256 * 1024**2
_LaserAttention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]


class StateDictShard(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...
    def keys(self) -> list[str]: ...


@dataclass(frozen=True)
class TensorParallel:
    rank: int
    size: int
    reduce: Callable[[torch.Tensor], torch.Tensor]
    gather: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SequenceParallel:
    rank: int
    size: int
    gather: Callable[[torch.Tensor], torch.Tensor]
    exchange: Callable[[torch.Tensor, bool], torch.Tensor]


class _ParallelLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        group: TensorParallel,
        *,
        row: bool = False,
        gather: bool = False,
        packed: bool = False,
    ) -> None:
        super().__init__()
        self._group = group
        self._row = row
        self._gather = gather
        self._packed = packed
        axis = 1 if row else 0
        parts = 2 if packed else 1
        shape = list(base.weight.shape)
        if shape[axis] % (group.size * parts):
            raise ValueError(f"Cannot shard H3 projection {shape} across {group.size} ranks")
        shape[axis] //= group.size
        self.weight = nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)
        bias_size = base.out_features if row else shape[0]
        bias = nn.Parameter(torch.empty(bias_size, device="meta"), requires_grad=False)
        self.register_parameter("bias", bias if base.bias is not None else None)

    def shard(self, value: torch.Tensor, parameter: str) -> torch.Tensor:
        if self._row:
            return value if parameter == "bias" else value.chunk(self._group.size, dim=1)[self._group.rank]
        if self._packed:
            # The converted SwiGLU projection stores up and gate contiguously.
            return torch.cat([part.chunk(self._group.size)[self._group.rank] for part in value.chunk(2)])
        return value.chunk(self._group.size)[self._group.rank]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._row:
            result = self._group.reduce(F.linear(hidden_states, self.weight))
            return result if self.bias is None else result + self.bias
        result = F.linear(hidden_states, self.weight, self.bias)
        return self._group.gather(result) if self._gather else result


def load_component_weights(
    model: nn.Module,
    shards: Sequence[StateDictShard],
    device: torch.device,
    dtype_for: Callable[[str], torch.dtype],
) -> None:
    missing = set(model.state_dict())
    pending_transfers: list[torch.Tensor] = []
    pending_bytes = 0
    for shard in shards:
        for name in shard.keys():
            if name not in missing:
                continue
            parent, _, key = name.rpartition(".")
            module = model.get_submodule(parent)
            target = getattr(module, key)
            value = shard.get_tensor(name)
            if isinstance(module, _ParallelLinear):
                value = module.shard(value, key)
            if target.shape != value.shape:
                raise ValueError(f"H3 weight {name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
            if not value.is_floating_point() and target.is_floating_point():
                raise ValueError(f"H3 weight {name} must be a dense floating point tensor")
            dtype = dtype_for(name) if value.is_floating_point() else target.dtype
            if device.type == "npu" and value.device.type == "cpu":
                # Pack the local shard on the host and enqueue the transfer.
                # Keep the pinned staging buffer alive until its batch completes.
                staging = torch.empty(value.shape, dtype=dtype, device="cpu", pin_memory=True)
                staging.copy_(value)
                value = staging.to(device=device, non_blocking=True)
                pending_transfers.append(staging)
                pending_bytes += staging.nbytes
                if pending_bytes >= _WEIGHT_TRANSFER_BATCH_BYTES:
                    torch.npu.current_stream(device).synchronize()
                    pending_transfers.clear()
                    pending_bytes = 0
            else:
                value = value.to(device=device, dtype=dtype, memory_format=torch.contiguous_format)
            if isinstance(target, nn.Parameter):
                value = nn.Parameter(value, requires_grad=False)
            setattr(module, key, value)
            missing.remove(name)
    if pending_transfers:
        torch.npu.current_stream(device).synchronize()
    if missing:
        raise ValueError(f"Missing H3 weights in {type(model).__name__}: {', '.join(sorted(missing)[:8])}")
    model.eval()


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


class _FirstBlockCache:
    def __init__(
        self,
        warmup_steps: int,
        residual_diff_threshold: float,
        max_consecutive_reuse_steps: int,
        reduce: Callable[[torch.Tensor], torch.Tensor],
        world_size: int,
    ) -> None:
        self._warmup_steps = warmup_steps
        self._threshold = residual_diff_threshold
        self._max_reuse = max_consecutive_reuse_steps
        self._reduce = reduce
        self._world_size = world_size
        self._consecutive_hits = 0
        self._first_residual: torch.Tensor | None = None
        self._full_residual: torch.Tensor | None = None

    def reuse(self, step: int, first_residual: torch.Tensor) -> torch.Tensor | None:
        previous = self._first_residual
        if (
            previous is None
            or step < self._warmup_steps
            or (self._max_reuse and self._consecutive_hits >= self._max_reuse)
        ):
            return None

        # The residual covers only this SP shard. Every denoising rank must
        # make the same decision before entering the next block's collectives.
        stats = self._reduce(
            torch.stack(
                (
                    (previous - first_residual).abs().sum(dtype=torch.float32),
                    previous.abs().sum(dtype=torch.float32),
                )
            )
        )
        denominator = stats[1] + 1e-6 * previous.numel() * self._world_size
        if (stats[0] < self._threshold * denominator).item():
            self._consecutive_hits += 1
            return self._full_residual
        return None

    def save(self, first_residual: torch.Tensor, full_residual: torch.Tensor) -> None:
        self._first_residual = first_residual
        self._full_residual = full_residual
        self._consecutive_hits = 0


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
            attn.to_q = _ParallelLinear(attn.to_q, tp)
            attn.to_k = _ParallelLinear(attn.to_k, tp)
            attn.to_v = _ParallelLinear(attn.to_v, tp)
            attn.to_out[0] = _ParallelLinear(attn.to_out[0], tp, row=True)
            block.ff.net[0].proj = _ParallelLinear(block.ff.net[0].proj, tp, packed=True)
            block.ff.net[2] = _ParallelLinear(block.ff.net[2], tp, row=True)
        for block in model.token_refiner.refiner_blocks:
            block.attn.set_processor(_RefinerAttention(self._laser))
        for block in model.transformer_blocks:
            block.adaln_proj.linear = _ParallelLinear(block.adaln_proj.linear, tp, gather=True)

        model.time_embedder.linear_1 = _ParallelLinear(model.time_embedder.linear_1, tp)
        model.time_embedder.linear_2 = _ParallelLinear(model.time_embedder.linear_2, tp, row=True)
        model.norm_out.linear = _ParallelLinear(model.norm_out.linear, tp, gather=True)

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

    def _run_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        temb: torch.Tensor,
        modulation_indices: torch.Tensor,
        conditioning: PackedConditioning,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = block.adaln_proj(temb)
        attn_scale = scale_attn.index_select(0, modulation_indices)
        attn_shift = shift_attn.index_select(0, modulation_indices)
        attn_input = block.norm1(hidden) * (1 + attn_scale) + attn_shift
        attended = _dense_attention(
            block.attn, attn_input, conditioning.rotary, self._sp, conditioning.layout.tags.numel(), self._laser
        )
        hidden = hidden + gate_attn.index_select(0, modulation_indices) * attended
        ff_scale = scale_ff.index_select(0, modulation_indices)
        ff_shift = shift_ff.index_select(0, modulation_indices)
        ff_input = block.norm2(hidden) * (1 + ff_scale) + ff_shift
        return hidden + gate_ff.index_select(0, modulation_indices) * block.ff(ff_input)

    def _run_cached_blocks(
        self,
        hidden: torch.Tensor,
        temb: torch.Tensor,
        modulation_indices: torch.Tensor,
        conditioning: PackedConditioning,
        cache: _FirstBlockCache,
        step: int,
    ) -> torch.Tensor:
        original_hidden = hidden
        blocks = iter(self.model.transformer_blocks)
        hidden = self._run_block(next(blocks), hidden, temb, modulation_indices, conditioning)
        first_residual = hidden - original_hidden
        cached_residual = cache.reuse(step, first_residual)
        if cached_residual is not None:
            return original_hidden + cached_residual
        for block in blocks:
            hidden = self._run_block(block, hidden, temb, modulation_indices, conditioning)
        cache.save(first_residual, hidden - original_hidden)
        return hidden

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        conditioning: PackedConditioning,
        *,
        cache: _FirstBlockCache | None = None,
        step: int = 0,
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
        if cache is None:
            for block in model.transformer_blocks:
                hidden = self._run_block(block, hidden, temb, modulation_indices, conditioning)
        else:
            hidden = self._run_cached_blocks(hidden, temb, modulation_indices, conditioning, cache, step)

        hidden = self._sp.gather(hidden.contiguous())[:, : layout.tags.numel()]
        hidden = model.norm_out(hidden, temb, timestep_indices).float()
        return (
            model.proj_out(hidden.index_select(1, layout.video_indices)),
            model.audio_proj_out(hidden.index_select(1, layout.audio_indices)),
        )
