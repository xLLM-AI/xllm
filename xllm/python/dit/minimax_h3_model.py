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
from math import isfinite, lcm
from typing import TYPE_CHECKING, Protocol

import torch
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    MiniMaxH3RotaryPosEmbed,
)
from torch import nn
from torch.nn import functional as F

from xllm.python import _load_kernel_package

if TYPE_CHECKING:
    from xllm_kernel.ops.attention.rainfusion.backend import MinimaxH3RainfusionRequest

_WEIGHT_TRANSFER_BATCH_BYTES = 256 * 1024**2
_PACKED_SEQUENCE_ALIGNMENT = 64
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
    exchange: Callable[[torch.Tensor, bool, bool], Callable[[], torch.Tensor]]
    async_qkv: bool = False

    def sync(self, tensor: torch.Tensor, scatter_heads: bool) -> torch.Tensor:
        return self.exchange(tensor, scatter_heads, False)()

    def submit(self, tensor: torch.Tensor, scatter_heads: bool) -> Callable[[], torch.Tensor]:
        return self.exchange(tensor, scatter_heads, True)


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
    timestep_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    audio_start: int = 0
    video_start: int = 0
    num_condition_video_rows: int = 0
    num_condition_audio_rows: int = 0


@dataclass(frozen=True)
class PackedConditioning:
    text: torch.Tensor
    rotary: tuple[torch.Tensor, torch.Tensor]
    layout: PackedLayout
    local_start: int
    local_rows: int
    padded_rows: int


@dataclass(frozen=True)
class CacheDiTOptions:
    warmup_steps: int = 30
    residual_diff_threshold: float = 0.04
    max_cached_steps: int = 4
    max_consecutive_hits: int = 1
    front_blocks: int = 1
    back_blocks: int = 0
    linear_residual_prediction: bool = True


@dataclass(frozen=True)
class CacheDiTDecision:
    hit: bool
    relative_l1: float
    prediction_scale: float


@dataclass(frozen=True)
class CacheDiTRequest:
    state: "SimilarityResidualCacheState"
    target_rows: torch.Tensor
    used_rows: int
    reduce: Callable[[torch.Tensor], torch.Tensor] | None


class SimilarityResidualCacheState:
    def __init__(self, options: CacheDiTOptions) -> None:
        self.validate_options(options)
        self._options = options
        self._previous_previous_front_residual: torch.Tensor | None = None
        self._previous_front_residual: torch.Tensor | None = None
        self._previous_stack_residual: torch.Tensor | None = None
        self._stack_residual: torch.Tensor | None = None
        self._previous_stack_step = -1
        self._stack_step = -1
        self._prediction_scale = 0.0
        self._dense_forwards = 0
        self._cache_hits = 0
        self._similarity_checks = 0
        self._consecutive_hits = 0
        self._steps: list[tuple[int, bool, int]] = []

    @staticmethod
    def validate_options(options: CacheDiTOptions) -> None:
        if (
            options.warmup_steps < 0
            or not isfinite(options.residual_diff_threshold)
            or options.residual_diff_threshold <= 0.0
            or options.max_cached_steps < -1
            or options.max_consecutive_hits <= 0
            or options.front_blocks <= 0
            or options.back_blocks < 0
        ):
            raise ValueError("similarity residual cache options are invalid")

    def decide(
        self,
        step: int,
        front_residual: torch.Tensor,
        used_rows: int,
        reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
        row_indices: torch.Tensor | None = None,
    ) -> CacheDiTDecision:
        empty_collective_shard = (
            used_rows == 0 and reduce is not None and row_indices is not None and row_indices.numel() == 0
        )
        if (
            step < 0
            or front_residual.dim() < 1
            or used_rows < 0
            or (used_rows == 0 and not empty_collective_shard)
            or used_rows > front_residual.size(0)
        ):
            raise ValueError("similarity residual cache front residual is invalid")

        below_total_limit = self._options.max_cached_steps < 0 or self._cache_hits < self._options.max_cached_steps
        below_consecutive_limit = self._consecutive_hits < self._options.max_consecutive_hits
        decision = CacheDiTDecision(False, float("inf"), 0.0)
        if (
            step >= self._options.warmup_steps
            and self._previous_front_residual is not None
            and below_total_limit
            and below_consecutive_limit
        ):
            previous = self._previous_front_residual
            if (
                previous.shape != front_residual.shape
                or previous.device != front_residual.device
                or previous.dtype != front_residual.dtype
            ):
                raise ValueError("similarity residual cache tensor metadata changed")
            previous_previous = self._previous_previous_front_residual
            if previous_previous is not None and (
                previous_previous.shape != front_residual.shape
                or previous_previous.device != front_residual.device
                or previous_previous.dtype != front_residual.dtype
            ):
                raise ValueError("similarity residual cache tensor metadata changed")
            if row_indices is not None:
                if (
                    row_indices.dim() != 1
                    or row_indices.dtype != torch.int64
                    or row_indices.device != front_residual.device
                ):
                    raise ValueError("similarity residual cache row indices are invalid")
                if row_indices.numel() > 0 and (row_indices.min().item() < 0 or row_indices.max().item() >= used_rows):
                    raise ValueError("similarity residual cache row indices are invalid")

            statistics_options = {"dtype": torch.float32, "device": front_residual.device}
            if row_indices is not None and row_indices.numel() == 0:
                statistics = torch.zeros(5, **statistics_options)
            else:
                current = front_residual.narrow(0, 0, used_rows)
                previous_used = previous.narrow(0, 0, used_rows)
                if row_indices is not None:
                    current = current.index_select(0, row_indices)
                    previous_used = previous_used.index_select(0, row_indices)
                current_delta = (current - previous_used).to(torch.float32)
                prediction_numerator = torch.zeros((), **statistics_options)
                prediction_denominator = torch.zeros_like(prediction_numerator)
                if previous_previous is not None:
                    historical = previous_previous.narrow(0, 0, used_rows)
                    if row_indices is not None:
                        historical = historical.index_select(0, row_indices)
                    historical_delta = (previous_used - historical).to(torch.float32)
                    prediction_numerator = (current_delta * historical_delta).sum()
                    prediction_denominator = historical_delta.square().sum()
                statistics = torch.stack(
                    (
                        current_delta.abs().sum(),
                        previous_used.abs().to(torch.float32).sum(),
                        torch.full((), float(current.numel()), **statistics_options),
                        prediction_numerator,
                        prediction_denominator,
                    )
                )
            if reduce is not None:
                statistics = reduce(statistics)
            if statistics[2].item() <= 0.0:
                raise ValueError("similarity residual cache selected no tensor values")
            prediction_scale = 0.0
            if statistics[4].item() > 1e-6:
                prediction_scale = min(max((statistics[3] / statistics[4]).item(), 0.0), 2.0)
            decision = CacheDiTDecision(
                False,
                (statistics[0] / (statistics[1] + 1e-6)).item(),
                prediction_scale,
            )
            self._similarity_checks += 1
            decision = CacheDiTDecision(
                isfinite(decision.relative_l1)
                and decision.relative_l1 < self._options.residual_diff_threshold
                and self._stack_residual is not None,
                decision.relative_l1,
                decision.prediction_scale,
            )

        if decision.hit:
            self._cache_hits += 1
            self._consecutive_hits += 1
            self._prediction_scale = decision.prediction_scale
        else:
            self._previous_previous_front_residual = self._previous_front_residual
            self._previous_front_residual = front_residual.detach().clone()
            self._consecutive_hits = 0
            self._prediction_scale = 0.0
        return decision

    def record_dense(self, step: int, front_output: torch.Tensor, full_output: torch.Tensor) -> None:
        if (
            step < 0
            or (self._stack_step >= 0 and step <= self._stack_step)
            or front_output.shape != full_output.shape
            or front_output.device != full_output.device
            or front_output.dtype != full_output.dtype
        ):
            raise ValueError("similarity residual cache dense outputs do not match")
        self._previous_stack_residual = self._stack_residual
        self._previous_stack_step = self._stack_step
        self._stack_residual = (full_output - front_output).detach()
        self._stack_step = step
        self._dense_forwards += 1

    def apply(self, step: int, front_output: torch.Tensor) -> torch.Tensor:
        if (
            self._stack_residual is None
            or step <= self._stack_step
            or front_output.shape != self._stack_residual.shape
            or front_output.device != self._stack_residual.device
            or front_output.dtype != self._stack_residual.dtype
        ):
            raise ValueError("similarity residual cache has no compatible stack residual")
        residual = self._stack_residual
        if (
            self._options.linear_residual_prediction
            and self._previous_stack_residual is not None
            and self._previous_stack_step >= 0
            and self._stack_step > self._previous_stack_step
        ):
            residual = self._stack_residual + self._prediction_scale * (
                self._stack_residual - self._previous_stack_residual
            )
        return front_output + residual

    def record_step(self, step: int, hit: bool, executed_blocks: int) -> None:
        self._steps.append((step, hit, executed_blocks))

    def statistics(self) -> dict[str, object]:
        return {
            "cache_hits": self._cache_hits,
            "dense_forwards": self._dense_forwards,
            "similarity_checks": self._similarity_checks,
            "steps": list(self._steps),
        }

    @property
    def front_blocks(self) -> int:
        return self._options.front_blocks

    @property
    def back_blocks(self) -> int:
        return self._options.back_blocks

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def dense_forwards(self) -> int:
        return self._dense_forwards

    @property
    def similarity_checks(self) -> int:
        return self._similarity_checks

    @property
    def hit_steps(self) -> list[int]:
        return [step for step, hit, _ in self._steps if hit]


def make_cache_request(
    options: CacheDiTOptions,
    layout: PackedLayout,
    rank: int,
    size: int,
    reduce: Callable[[torch.Tensor], torch.Tensor] | None,
    device: torch.device,
) -> CacheDiTRequest:
    sequence_length = layout.tags.numel()
    alignment = lcm(_PACKED_SEQUENCE_ALIGNMENT, size)
    padding = (alignment - sequence_length % alignment) % alignment
    local_rows = (sequence_length + padding) // size
    start = rank * local_rows
    used_rows = min(max(sequence_length - start, 0), local_rows)
    targets: list[int] = []
    for begin, end in ((layout.video_start, sequence_length), (layout.audio_start, layout.video_start)):
        low = max(begin, start)
        high = min(end, start + used_rows)
        targets.extend(range(low - start, high - start))
    return CacheDiTRequest(
        SimilarityResidualCacheState(options),
        torch.tensor(targets, dtype=torch.int64, device=device),
        used_rows,
        reduce,
    )


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
    *,
    rainfusion: "MinimaxH3RainfusionRequest | None" = None,
) -> torch.Tensor:
    query = attn.norm_q(attn.to_q(states).unflatten(-1, (-1, attn.head_dim)))
    key = attn.norm_k(attn.to_k(states).unflatten(-1, (-1, attn.head_dim)))
    value = attn.to_v(states).unflatten(-1, (-1, attn.head_dim))
    if rotary is not None:
        query = _apply_rotary(query, rotary)
        key = _apply_rotary(key, rotary)
    local_heads = query.size(2)
    head_padding = 0
    if sp is not None:
        padded_heads = local_heads + (-local_heads) % sp.size
        head_padding = padded_heads - local_heads
        if head_padding:
            pad = (0, 0, 0, head_padding)
            query = F.pad(query, pad)
            key = F.pad(key, pad)
            value = F.pad(value, pad)
        if sp.async_qkv:
            query_work = sp.submit(query, True)
            key_work = sp.submit(key, True)
            value_work = sp.submit(value, True)
            query = query_work()
            key = key_work()
            value = value_work()
        else:
            query = sp.sync(query, True)
            key = sp.sync(key, True)
            value = sp.sync(value, True)
    rows, heads, head_dim = query.shape[1:]
    if live_rows is None:
        live_rows = rows
    if rainfusion is not None:
        # The compiled AOT pair owns geometry checks, capacity and both
        # launches. Q/K/V stay in the same post-Ulysses packed layout as the
        # dense NPU attention.
        attended = rainfusion.attention(query, key, value, live_rows)
    elif laser is not None and live_rows >= 2048 and head_dim <= 128:
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
        attended = sp.sync(attended, False)
        if head_padding:
            attended = attended.narrow(2, 0, local_heads).contiguous()
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
        rainfusion_enabled: bool = False,
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
        head_parallel_size = tp.size * sp.size
        padded_split_heads = (
            cfg.num_attention_heads == 56 and cfg.attention_head_dim == 128 and tp.size == 2 and sp.size == 8
        )
        if (
            (cfg.num_attention_heads % head_parallel_size and not padded_split_heads)
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
        if rainfusion_enabled:
            _load_kernel_package()
            from xllm_kernel.ops.attention.rainfusion.backend import MinimaxH3RainfusionBackend

            MinimaxH3RainfusionBackend().prepare_specialization(device)

    def prepare_conditioning(self, prompt_embed: torch.Tensor, layout: PackedLayout) -> PackedConditioning:
        text = self.model.token_refiner(self.model.context_embedder(prompt_embed.to(torch.bfloat16)))
        alignment = lcm(_PACKED_SEQUENCE_ALIGNMENT, self._sp.size)
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

    def prepare_rainfusion(
        self,
        layout: PackedLayout,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        steps: int,
    ) -> tuple["MinimaxH3RainfusionRequest | None", str]:
        from xllm_kernel.ops.attention.rainfusion.backend import (
            SPARSE_MIDDLE_STEPS,
            MinimaxH3RainfusionBackend,
        )

        valid_rows = int(layout.tags.numel())
        video_start = int(layout.video_start)
        if steps != 50:
            return None, "unsupported_native_step_count"
        if video_start + grid_t * grid_h * grid_w != valid_rows:
            return None, "unsupported_native_geometry"
        backend = MinimaxH3RainfusionBackend(enabled=True)
        reason = backend.unavailability_reason(valid_rows, video_start, grid_t, grid_h, grid_w)
        if reason:
            return None, reason
        request = backend.prepare(
            valid_rows,
            video_start,
            grid_t,
            grid_h,
            grid_w,
            SPARSE_MIDDLE_STEPS,
            self.model.proj_in.weight.device,
        )
        return request, "" if request is not None else "specialization_unavailable"

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        conditioning: PackedConditioning,
        *,
        step: int = 0,
        cache: CacheDiTRequest | None = None,
        rainfusion: "MinimaxH3RainfusionRequest | None" = None,
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

        def run_block(index: int, block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            active = (
                rainfusion
                if rainfusion is not None
                and rainfusion.policy.includes_step(step)
                and rainfusion.policy.includes_layer(index)
                else None
            )
            shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = block.adaln_proj(temb)
            attn_scale = scale_attn.index_select(0, modulation_indices)
            attn_shift = shift_attn.index_select(0, modulation_indices)
            attn_input = block.norm1(hidden_states) * (1 + attn_scale) + attn_shift
            attended = _dense_attention(
                block.attn,
                attn_input,
                conditioning.rotary,
                self._sp,
                layout.tags.numel(),
                laser=self._laser,
                rainfusion=active,
            )
            hidden_states = hidden_states + gate_attn.index_select(0, modulation_indices) * attended
            ff_scale = scale_ff.index_select(0, modulation_indices)
            ff_shift = shift_ff.index_select(0, modulation_indices)
            ff_input = block.norm2(hidden_states) * (1 + ff_scale) + ff_shift
            return hidden_states + gate_ff.index_select(0, modulation_indices) * block.ff(ff_input)

        blocks = model.transformer_blocks
        if cache is not None:
            middle_start = cache.state.front_blocks
            tail_start = len(blocks) - cache.state.back_blocks
            original_hidden = hidden
            for index in range(middle_start):
                hidden = run_block(index, blocks[index], hidden)
            front = hidden
            decision = cache.state.decide(
                step,
                (front - original_hidden).squeeze(0),
                cache.used_rows,
                cache.reduce,
                cache.target_rows,
            )
            executed_blocks = middle_start
            if decision.hit:
                hidden = cache.state.apply(step, front)
            else:
                for index in range(middle_start, tail_start):
                    hidden = run_block(index, blocks[index], hidden)
                cache.state.record_dense(step, front, hidden)
                executed_blocks += tail_start - middle_start
            for index in range(tail_start, len(blocks)):
                hidden = run_block(index, blocks[index], hidden)
            cache.state.record_step(step, decision.hit, executed_blocks + len(blocks) - tail_start)
        else:
            for index, block in enumerate(blocks):
                hidden = run_block(index, block, hidden)

        hidden = self._sp.gather(hidden.contiguous())[:, : layout.tags.numel()]
        hidden = model.norm_out(hidden, temb, timestep_indices).float()
        return (
            model.proj_out(hidden.index_select(1, layout.video_indices)),
            model.audio_proj_out(hidden.index_select(1, layout.audio_indices)),
        )
