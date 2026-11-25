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

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from math import floor, isfinite, sqrt
from typing import Any

import numpy as np
import torch
from diffusers import MiniMaxH3Scheduler
from PIL import Image

from scripts.logger import logger
from xllm.python.dit import ComponentLoad
from xllm.python.dit.minimax_h3_model import (
    CacheDiTOptions,
    CacheDiTRequest,
    MiniMaxH3Denoiser,
    PackedConditioning,
    PackedLayout,
    SequenceParallel,
    SimilarityResidualCacheState,
    StateDictShard,
    TensorParallel,
    make_cache_request,
)
from xllm.python.dit.minimax_h3_vae import TileParallel, load_audio_decoder, load_audio_geometry, load_video_decoder

_AUDIO_CHANNELS = 2
_ROPE_FRAME_SCALE = 5 / 3
_REFERENCE_IMAGE_SHORT_EDGE = 2048
_REFERENCE_CANVAS_SHORT_EDGE = 768
_REFERENCE_CANVAS_MAX_PIXELS = 768.0 * 1344.0
_KEYFRAME_NOISE_AUG = 0.999


@dataclass(frozen=True)
class _VideoGeometry:
    latent_frames: int
    latent_height: int
    latent_width: int
    output_frames: int
    spatial_ratio: int
    temporal_ratio: int
    latents_per_clip: int
    patch_height: int
    patch_width: int
    clip_length: int
    token_drop: int

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        channels = latent.shape[1]
        ph, pw = self.patch_height, self.patch_width
        shape = (1, channels, self.latent_frames, self.latent_height // ph, ph, self.latent_width // pw, pw)
        return latent.reshape(shape).permute(0, 2, 3, 5, 1, 4, 6).reshape(1, -1, channels * ph * pw)

    def unpatchify(self, rows: torch.Tensor) -> torch.Tensor:
        ph, pw = self.patch_height, self.patch_width
        shape = (1, self.latent_frames, self.latent_height // ph, self.latent_width // pw, -1, ph, pw)
        return (
            rows.reshape(shape)
            .permute(0, 4, 1, 2, 5, 3, 6)
            .reshape(1, -1, self.latent_frames, self.latent_height, self.latent_width)
        )


@dataclass(frozen=True)
class _ReferenceBlock:
    kind: str
    latent_frames: int = 0
    latent_height: int = 0
    latent_width: int = 0
    audio_latents: int = 0


@dataclass(frozen=True)
class _ReferenceData:
    blocks: tuple[_ReferenceBlock, ...]
    video_latents: tuple[torch.Tensor, ...]
    audio_rows: tuple[torch.Tensor, ...]


def _spatial_grid(size: int, patch: int, area_root: float, spatial_ratio: int) -> torch.Tensor:
    count = size // patch
    side = size / area_root
    return ((1 - side) / 2 + torch.arange(count, dtype=torch.float64) * (side / count)) * spatial_ratio * patch


def _build_layout(
    tags: tuple[int, ...], geometry: _VideoGeometry, audio_latents: int, device: torch.device
) -> PackedLayout:
    text_rows = len(tags)
    audio_rows = _AUDIO_CHANNELS * audio_latents
    area_root = sqrt(geometry.latent_height * geometry.latent_width)
    height = _spatial_grid(geometry.latent_height, geometry.patch_height, area_root, geometry.spatial_ratio)
    width = _spatial_grid(geometry.latent_width, geometry.patch_width, area_root, geometry.spatial_ratio)

    time_positions = []
    next_position = float(text_rows)
    for frame in range(geometry.latent_frames):
        time_positions.append(next_position)
        frame_span = 1 if frame % geometry.latents_per_clip == 0 else geometry.temporal_ratio
        next_position += _ROPE_FRAME_SCALE * frame_span
    video_pos = torch.cartesian_prod(torch.tensor(time_positions, dtype=torch.float64), height, width)

    text_pos = torch.zeros((text_rows, 3), dtype=torch.float64)
    text_pos[:, 0] = torch.arange(text_rows, dtype=torch.float64)
    audio_pos = torch.zeros((audio_rows, 3), dtype=torch.float64)
    audio_pos[:, 0] = (text_rows + torch.arange(audio_latents, dtype=torch.float64)).repeat(_AUDIO_CHANNELS)
    audio_pos[:, 2] = torch.stack((width[0], width[-1])).repeat_interleave(audio_latents)

    video_start = text_rows + audio_rows
    video_end = video_start + video_pos.shape[0]
    audio_mask = torch.zeros(video_end, dtype=torch.long)
    audio_mask[text_rows:video_start] = 1
    return PackedLayout(
        positions=torch.cat((text_pos, audio_pos, video_pos)).to(device=device, dtype=torch.float32),
        tags=torch.tensor(list(tags) + [2] * audio_rows + [0] * video_pos.shape[0], dtype=torch.long).to(device),
        video_indices=torch.arange(video_start, video_end, dtype=torch.long).to(device),
        audio_indices=torch.arange(text_rows, video_start, dtype=torch.long).to(device),
        text_indices=torch.arange(text_rows, dtype=torch.long).to(device),
        timestep_indices=(
            torch.zeros_like(audio_mask).to(device),
            audio_mask.to(device),
            (1 - audio_mask).to(device),
        ),
    )


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _round_half_up(value: float) -> int:
    return floor(value + 0.5)


def _reference_aspect_ratio(height: int, width: int, description: str) -> float:
    if height <= 0 or width <= 0:
        raise ValueError(f"{description} must have a positive size")
    ratio = width / height
    if ratio < 0.25 or ratio > 4.0:
        raise ValueError(f"{description} must be within 1:4 and 4:1")
    return ratio


def _reference_image_size(height: int, width: int, canvas_multiple: int) -> tuple[int, int]:
    _reference_aspect_ratio(height, width, "MiniMax-H3 reference images")
    scale = _REFERENCE_IMAGE_SHORT_EDGE / min(height, width)
    return _round_to_multiple(height * scale, canvas_multiple), _round_to_multiple(width * scale, canvas_multiple)


def _reference_canvas_size(height: int, width: int, canvas_multiple: int) -> tuple[int, int]:
    ratio = _reference_aspect_ratio(height, width, "MiniMax-H3 references")
    if ratio >= 1.0:
        canvas_width = _REFERENCE_CANVAS_SHORT_EDGE * ratio
        canvas_height = float(_REFERENCE_CANVAS_SHORT_EDGE)
    else:
        canvas_width = float(_REFERENCE_CANVAS_SHORT_EDGE)
        canvas_height = _REFERENCE_CANVAS_SHORT_EDGE / ratio
    area = canvas_width * canvas_height
    if area > _REFERENCE_CANVAS_MAX_PIXELS:
        scale = sqrt(_REFERENCE_CANVAS_MAX_PIXELS / area)
        canvas_width *= scale
        canvas_height *= scale
    return _round_to_multiple(canvas_height, canvas_multiple), _round_to_multiple(canvas_width, canvas_multiple)


def _resize_reference_pixels(pixels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    resized = Image.fromarray(pixels.permute(1, 2, 0).contiguous().numpy()).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    return torch.from_numpy(np.array(resized)).permute(2, 0, 1).contiguous()


def _reference_image_pixels(image: torch.Tensor, canvas_multiple: int) -> torch.Tensor:
    image = _unbatched_reference_image(image).detach().to(device="cpu", dtype=torch.uint8)
    height, width = (int(size) for size in image.shape[1:])
    target_height, target_width = _reference_image_size(height, width, canvas_multiple)
    if (height, width) == (target_height, target_width):
        return image.contiguous()
    return _resize_reference_pixels(image, target_height, target_width)


def _unbatched_reference_media(tensor: torch.Tensor, dim: int, description: str) -> torch.Tensor:
    if tensor.dim() == dim:
        if tensor.size(0) != 1:
            raise ValueError(f"MiniMax-H3 {description} batch size must be 1")
        return tensor[0]
    if tensor.dim() == dim - 1:
        return tensor
    raise ValueError(f"MiniMax-H3 {description} must have {dim} dimensions")


def _unbatched_reference_image(tensor: torch.Tensor) -> torch.Tensor:
    image = _unbatched_reference_media(tensor, 4, "reference image")
    if image.dim() != 3 or image.size(0) != 3:
        raise ValueError("MiniMax-H3 reference image must be [3, H, W]")
    return image


def _unbatched_reference_video(tensor: torch.Tensor) -> torch.Tensor:
    video = _unbatched_reference_media(tensor, 5, "reference video")
    if video.dim() != 4 or video.size(1) != 3 or video.size(0) <= 0:
        raise ValueError("MiniMax-H3 reference video must be [T, 3, H, W] with T > 0")
    return video


def _unbatched_reference_audio(tensor: torch.Tensor) -> torch.Tensor:
    audio = _unbatched_reference_media(tensor, 3, "reference audio")
    if audio.dim() != 2 or audio.size(0) != _AUDIO_CHANNELS:
        raise ValueError("MiniMax-H3 reference audio must be [2, S]")
    return audio


def _reference_video_fps(parameters: Mapping[str, Any]) -> float:
    value = parameters.get("prompt_video_fps")
    if value is None:
        raise ValueError("MiniMax-H3 reference video requires prompt_video_fps")
    fps = float(value)
    if not isfinite(fps) or fps <= 0.0:
        raise ValueError("MiniMax-H3 prompt_video_fps must be finite and positive")
    return fps


def _reference_normalized_frames(source_frames: int, source_fps: float, target_fps: float, target_frames: int) -> int:
    if source_frames <= 0 or source_fps <= 0.0 or target_fps <= 0.0 or target_frames <= 0:
        raise ValueError("MiniMax-H3 reference video frame metadata must be positive")
    frames = source_frames
    if source_fps != target_fps:
        frames = _round_half_up(source_frames * target_fps / source_fps)
    return min(frames, target_frames)


def _reference_video_encode_frames(normalized_frames: int, geometry: _VideoGeometry) -> int:
    if normalized_frames <= 0:
        raise ValueError("MiniMax-H3 reference video is empty after FPS resampling")
    snapped = (
        max(1, (normalized_frames - geometry.latents_per_clip) // geometry.clip_length) * geometry.clip_length
        + geometry.latents_per_clip
    )
    return min(normalized_frames, snapped)


def _reference_latent_frames(encode_frames: int, geometry: _VideoGeometry) -> int:
    if encode_frames <= 0:
        raise ValueError("MiniMax-H3 reference video is empty after FPS resampling")
    if encode_frames == 1:
        return 1
    whole_clips = max(0, encode_frames - geometry.latents_per_clip) // geometry.clip_length
    return whole_clips * geometry.latents_per_clip + geometry.latents_per_clip - geometry.token_drop


def _reference_audio_latents(sample_count: int, max_samples: int, hop_length: int) -> int:
    samples = min(sample_count, max_samples)
    return (samples + hop_length - 1) // hop_length


def _resample_reference_video(video: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    if source_fps == target_fps:
        return video
    scale = target_fps / source_fps
    final_slot = _round_half_up(video.size(0) * scale)
    indices = []
    for frame in range(video.size(0)):
        start = _round_half_up(frame * scale)
        end = _round_half_up((frame + 1) * scale) if frame + 1 < video.size(0) else final_slot
        indices.extend([frame] * (end - start))
    if not indices:
        raise ValueError("MiniMax-H3 reference video is empty after FPS resampling")
    return video.index_select(0, torch.tensor(indices, dtype=torch.long))


def _reference_video_pixels(
    video: torch.Tensor,
    source_fps: float,
    target_fps: float,
    target_frames: int,
    target_height: int,
    target_width: int,
    geometry: _VideoGeometry,
) -> torch.Tensor:
    video = video.detach().to(device="cpu", dtype=torch.uint8)
    video = _resample_reference_video(video, source_fps, target_fps)
    video = video[: min(video.size(0), target_frames)]
    video = video[: _reference_video_encode_frames(video.size(0), geometry)]
    if (video.size(2), video.size(3)) == (target_height, target_width):
        return video.contiguous()
    return torch.stack(
        [_resize_reference_pixels(frame, target_height, target_width) for frame in video],
    )


def _validate_reference_sources(media_sources: Sequence[tuple[str, str, torch.Tensor, Mapping[str, Any]]]) -> None:
    image_count = video_count = audio_count = video_audio_count = 0
    for index, (name, modality, _tensor, _parameters) in enumerate(media_sources):
        if name == "image":
            if modality != "image":
                raise ValueError("MiniMax-H3 image reference must use image modality")
            image_count += 1
        elif name == "prompt_video":
            if modality != "video":
                raise ValueError("MiniMax-H3 prompt_video reference must use video modality")
            video_count += 1
        elif name == "prompt_audio":
            if modality != "audio":
                raise ValueError("MiniMax-H3 prompt_audio reference must use audio modality")
            audio_count += 1
        elif name == "prompt_audio_in_video":
            if modality != "audio" or index == 0 or media_sources[index - 1][0] != "prompt_video":
                raise ValueError("MiniMax-H3 embedded audio must follow its prompt_video")
            video_audio_count += 1
        else:
            raise ValueError(f"MiniMax-H3 unsupported media source name: {name}")
    if image_count > 9 or video_count > 1 or audio_count > 1 or video_audio_count > video_count:
        raise ValueError("MiniMax-H3 Ref2VA accepts at most 9 images, 1 video and 1 separate audio")


def _reference_frame_grid(height: int, width: int, geometry: _VideoGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    area_root = sqrt(height * width)
    height_grid = _spatial_grid(height, geometry.patch_height, area_root, geometry.spatial_ratio)
    width_grid = _spatial_grid(width, geometry.patch_width, area_root, geometry.spatial_ratio)
    return torch.cartesian_prod(height_grid, width_grid), width_grid


def _reference_temporal_grid(num_frames: int, origin: float, geometry: _VideoGeometry) -> torch.Tensor:
    positions = []
    position = origin
    for frame in range(num_frames):
        positions.append(position)
        span = 1 if frame % geometry.latents_per_clip == 0 else geometry.temporal_ratio
        position += _ROPE_FRAME_SCALE * span
    return torch.tensor(positions, dtype=torch.float64)


def _reference_rope_frame_time(num_frames: int, geometry: _VideoGeometry) -> float:
    total = 0.0
    for frame in range(num_frames):
        total += _ROPE_FRAME_SCALE * (1 if frame % geometry.latents_per_clip == 0 else geometry.temporal_ratio)
    return total


def _fill_reference_audio_positions(
    positions: torch.Tensor,
    start: int,
    count: int,
    num_latents: int,
    origin: float,
    width_grid: torch.Tensor,
) -> None:
    if count == 0:
        return
    positions[start : start + count, 0] = (origin + torch.arange(num_latents, dtype=torch.float64)).repeat(
        _AUDIO_CHANNELS
    )
    positions[start : start + count, 2] = torch.cat(
        [
            torch.full((num_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )


def _require_reference_latent_shape(latent: torch.Tensor, block: _ReferenceBlock) -> None:
    expected = (1, latent.size(1), block.latent_frames, block.latent_height, block.latent_width)
    if tuple(latent.shape) != expected:
        raise ValueError(
            f"MiniMax-H3 {block.kind} reference latent {tuple(latent.shape)} does not match block {expected}"
        )


def _patchify_reference_latent(latent: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    channels = latent.size(1)
    frames = latent.size(2)
    height = latent.size(3)
    width = latent.size(4)
    return (
        latent.reshape(1, channels, frames, height // patch_h, patch_h, width // patch_w, patch_w)
        .permute(0, 2, 3, 5, 1, 4, 6)
        .reshape(1, -1, channels * patch_h * patch_w)
        .contiguous()
    )


def _reference_condition_rows(
    condition_latents: Sequence[torch.Tensor],
    generator: torch.Generator,
    scheduler: MiniMaxH3Scheduler,
    device: torch.device,
    patch_h: int,
    patch_w: int,
) -> torch.Tensor:
    packed = []
    for latent in condition_latents:
        noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(device)
        noised = scheduler.scale_noise(latent.to(device), _KEYFRAME_NOISE_AUG, noise)
        packed.append(_patchify_reference_latent(noised, patch_h, patch_w))
    return torch.cat(packed, dim=1).contiguous()


def _reference_timesteps(
    layout: PackedLayout, video_timestep: torch.Tensor, audio_timestep: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not layout.num_condition_video_rows and not layout.num_condition_audio_rows:
        if float(video_timestep) == float(audio_timestep):
            return video_timestep.reshape(1), torch.zeros(
                layout.tags.numel(), dtype=torch.long, device=layout.tags.device
            )
        video_first = float(video_timestep) < float(audio_timestep)
        timesteps = torch.stack((video_timestep, audio_timestep) if video_first else (audio_timestep, video_timestep))
        indices = torch.full(
            (layout.tags.numel(),), 0 if video_first else 1, dtype=torch.long, device=layout.tags.device
        )
        indices[layout.audio_start : layout.video_start] = 1 if video_first else 0
        return timesteps, indices

    values = torch.full((layout.tags.numel(),), float(video_timestep), dtype=torch.float32)
    if layout.num_condition_video_rows:
        values[layout.video_indices[: layout.num_condition_video_rows].cpu()] = max(
            float(video_timestep), _KEYFRAME_NOISE_AUG
        )
    if layout.num_condition_audio_rows:
        values[layout.audio_indices[: layout.num_condition_audio_rows].cpu()] = 1.0
    timesteps, indices = torch.unique(values, sorted=True, return_inverse=True)
    return timesteps.to(layout.tags.device), indices.to(layout.tags.device)


def _build_reference_layout(
    prompt_tags: Sequence[int],
    references: _ReferenceData,
    geometry: _VideoGeometry,
    audio_latents: int,
    device: torch.device,
) -> PackedLayout:
    text_rows = len(prompt_tags)
    target_frame_rows = (geometry.latent_height // geometry.patch_height) * (
        geometry.latent_width // geometry.patch_width
    )
    target_audio_rows = _AUDIO_CHANNELS * audio_latents
    reference_video_rows = sum(
        block.latent_frames
        * (block.latent_height // geometry.patch_height)
        * (block.latent_width // geometry.patch_width)
        for block in references.blocks
        if block.kind != "audio"
    )
    reference_audio_rows = sum(_AUDIO_CHANNELS * block.audio_latents for block in references.blocks)
    sequence_length = text_rows + reference_video_rows + reference_audio_rows + target_audio_rows
    sequence_length += geometry.latent_frames * target_frame_rows

    positions = torch.zeros((sequence_length, 3), dtype=torch.float64)
    positions[:text_rows, 0] = torch.arange(text_rows, dtype=torch.float64)
    target_grid, target_width_grid = _reference_frame_grid(geometry.latent_height, geometry.latent_width, geometry)
    video_blocks = []
    audio_blocks = []
    cursor = text_rows
    rotary_time = float(text_rows)
    for block in references.blocks:
        if block.kind == "image":
            frame_rows = (block.latent_height // geometry.patch_height) * (block.latent_width // geometry.patch_width)
            rows = block.latent_frames * frame_rows
            video_blocks.append(torch.arange(cursor, cursor + rows, dtype=torch.long))
            frame_grid, _ = _reference_frame_grid(block.latent_height, block.latent_width, geometry)
            time = _reference_temporal_grid(block.latent_frames, rotary_time, geometry)
            positions[cursor : cursor + rows, 0] = time.repeat_interleave(frame_rows)
            positions[cursor : cursor + rows, 1:] = frame_grid.repeat(block.latent_frames, 1)
            cursor += rows
            rotary_time += 1.0
        elif block.kind == "audio":
            rows = _AUDIO_CHANNELS * block.audio_latents
            audio_blocks.append(torch.arange(cursor, cursor + rows, dtype=torch.long))
            _fill_reference_audio_positions(
                positions, cursor, rows, block.audio_latents, rotary_time, target_width_grid
            )
            cursor += rows
            rotary_time += block.audio_latents
        else:
            frame_rows = (block.latent_height // geometry.patch_height) * (block.latent_width // geometry.patch_width)
            block_audio_rows = _AUDIO_CHANNELS * block.audio_latents
            video_rows = block.latent_frames * frame_rows
            if block_audio_rows:
                audio_blocks.append(torch.arange(cursor, cursor + block_audio_rows, dtype=torch.long))
            video_blocks.append(
                torch.arange(cursor + block_audio_rows, cursor + block_audio_rows + video_rows, dtype=torch.long)
            )
            frame_grid, width_grid = _reference_frame_grid(block.latent_height, block.latent_width, geometry)
            _fill_reference_audio_positions(
                positions, cursor, block_audio_rows, block.audio_latents, rotary_time, width_grid
            )
            video_start = cursor + block_audio_rows
            time = _reference_temporal_grid(block.latent_frames, rotary_time, geometry)
            positions[video_start : video_start + video_rows, 0] = time.repeat_interleave(frame_rows)
            positions[video_start : video_start + video_rows, 1:] = frame_grid.repeat(block.latent_frames, 1)
            cursor += block_audio_rows + video_rows
            rotary_time += max(float(block.audio_latents), _reference_rope_frame_time(block.latent_frames, geometry))

    audio_start = cursor
    video_start = audio_start + target_audio_rows
    _fill_reference_audio_positions(
        positions, audio_start, target_audio_rows, audio_latents, rotary_time, target_width_grid
    )
    target_time = _reference_temporal_grid(geometry.latent_frames, rotary_time, geometry)
    positions[video_start:, 0] = target_time.repeat_interleave(target_frame_rows)
    positions[video_start:, 1:] = target_grid.repeat(geometry.latent_frames, 1)
    video_blocks.append(torch.arange(video_start, sequence_length, dtype=torch.long))
    audio_blocks.append(torch.arange(audio_start, video_start, dtype=torch.long))
    text_indices = torch.arange(text_rows, dtype=torch.long)
    video_indices = torch.cat(video_blocks)
    audio_indices = torch.cat(audio_blocks)

    tags = torch.empty(sequence_length, dtype=torch.long)
    tags[text_indices] = torch.as_tensor(prompt_tags, dtype=torch.long)
    tags[audio_indices] = 2
    tags[video_indices] = 0
    return PackedLayout(
        positions=positions.to(device=device, dtype=torch.float32),
        tags=tags.to(device),
        video_indices=video_indices.to(device),
        audio_indices=audio_indices.to(device),
        text_indices=text_indices.to(device),
        timestep_indices=None,
        audio_start=audio_start,
        video_start=video_start,
        num_condition_video_rows=reference_video_rows,
        num_condition_audio_rows=reference_audio_rows,
    )


def _prepare_noise(
    geometry: _VideoGeometry, video_channels: int, audio_shape: tuple[int, int, int], seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Native H3 restarts the CPU generator for each modality with the same seed.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pin_memory = device.type == "npu"
    video_noise = torch.randn(
        (1, video_channels, geometry.latent_frames, geometry.latent_height, geometry.latent_width),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
        pin_memory=pin_memory,
    )
    video = geometry.patchify(video_noise.to(device, non_blocking=pin_memory))
    audio_noise = torch.randn(
        audio_shape,
        generator=generator.manual_seed(seed),
        dtype=torch.float32,
        device="cpu",
        pin_memory=pin_memory,
    )
    return video, audio_noise.to(device, non_blocking=pin_memory)


def _build_denoise_schedule(
    steps: int, video_shift: float, audio_shift: float, device: torch.device
) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler, tuple[tuple[torch.Tensor, int], ...]]:
    video_scheduler = MiniMaxH3Scheduler(shift=video_shift)
    audio_scheduler = MiniMaxH3Scheduler(shift=audio_shift)
    # Diffusers counts grid points, including the terminal sigma. Keep the
    # ordering decisions on CPU until the complete schedule is ready.
    for scheduler in (video_scheduler, audio_scheduler):
        scheduler.set_timesteps(steps + 1, device="cpu")
        scheduler.set_begin_index(0)
    video_times = video_scheduler.timesteps
    audio_times = audio_scheduler.timesteps
    modes = tuple(
        0 if video_time == audio_time else 1 if video_time < audio_time else 2
        for video_time, audio_time in zip(video_times.tolist(), audio_times.tolist(), strict=True)
    )
    joint_times = torch.stack((video_times, audio_times), dim=-1).sort(dim=-1).values
    schedule_tensors = (joint_times.flatten(), video_times, audio_times, video_scheduler.sigmas, audio_scheduler.sigmas)
    # The values share a dtype. Send the full schedule in one host-to-device copy.
    packed_schedule = torch.cat(schedule_tensors).to(device)
    (
        joint_times,
        video_scheduler.timesteps,
        audio_scheduler.timesteps,
        video_scheduler.sigmas,
        audio_scheduler.sigmas,
    ) = packed_schedule.split([tensor.numel() for tensor in schedule_tensors])
    step_layouts = tuple(
        (times if mode else times[:1], mode)
        for times, mode in zip(joint_times.view(-1, 2).unbind(), modes, strict=True)
    )
    return video_scheduler, audio_scheduler, step_layouts


class MiniMaxH3Pipeline:
    def __init__(
        self,
        device: str,
        is_driver: bool,
        tp_rank: int,
        tp_size: int,
        sp_rank: int,
        sp_size: int,
        vae_rank: int,
        vae_size: int,
        tp_reduce: Callable[[torch.Tensor], torch.Tensor],
        tp_gather: Callable[[torch.Tensor], torch.Tensor],
        sp_gather: Callable[[torch.Tensor], torch.Tensor],
        sp_exchange: Callable[[torch.Tensor, bool, bool], Callable[[], torch.Tensor]],
        vae_gather: Callable[[torch.Tensor], torch.Tensor],
        task_type: str = "fl2va",
        *,
        tp_broadcast: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        cache_options: Mapping[str, Any] | None = None,
        sp_reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
        sp_communication_overlap: bool = True,
        laser_attention_enabled: bool = False,
    ) -> None:
        if task_type not in ("fl2va", "ref2va"):
            raise ValueError(f"Unsupported MiniMax-H3 task_type: {task_type}")
        if cache_options is not None and task_type != "ref2va":
            raise ValueError("MiniMax-H3 CacheDiT requires Ref2VA")
        if cache_options is not None and sp_size > 1 and sp_reduce is None:
            raise ValueError("MiniMax-H3 CacheDiT requires a cross-SP reduction")
        self._device = torch.device(device)
        self._laser_attention_enabled = laser_attention_enabled
        self._is_driver = is_driver
        self._task_type = task_type
        self._tp = TensorParallel(tp_rank, tp_size, tp_reduce, tp_gather)
        self._sp = SequenceParallel(
            sp_rank,
            sp_size,
            sp_gather,
            sp_exchange,
            async_qkv=task_type == "ref2va" and sp_communication_overlap,
        )
        self._tp_broadcast = tp_broadcast
        self._tiles = TileParallel(vae_rank, vae_size, vae_gather)
        self._sp_reduce = sp_reduce
        self._cache_options = CacheDiTOptions(**cache_options) if cache_options is not None else None
        self.last_cache_stats: dict[str, object] | None = None
        self._scheduler_shifts = {"scheduler": 12.0, "audio_scheduler": 3.0}
        self._packed_layout = lru_cache(maxsize=1)(_build_layout)
        self._initial_noise = lru_cache(maxsize=1)(_prepare_noise)
        self._denoise_schedule = lru_cache(maxsize=1)(_build_denoise_schedule)

    def checkpoint_components(self) -> tuple[ComponentLoad, ...]:
        transformer = "transformer_ref" if self._task_type == "ref2va" else "transformer"
        audio_weights = self._tp.rank == 0 if self._task_type == "ref2va" else self._is_driver
        return (
            ComponentLoad(transformer),
            ComponentLoad("vae"),
            ComponentLoad("audio_vae", load_weights=audio_weights),
            ComponentLoad("scheduler", required=False, load_weights=False),
            ComponentLoad("audio_scheduler", required=False, load_weights=False),
        )

    def load_component(self, name: str, path: str, shards: Sequence[StateDictShard]) -> None:
        if name in ("transformer", "transformer_ref"):
            self._denoiser = MiniMaxH3Denoiser(
                path, shards, self._device, self._tp, self._sp, laser_attention_enabled=self._laser_attention_enabled
            )
        elif name == "vae":
            self._video_decoder = load_video_decoder(
                path, shards, self._device, self._tiles, with_encoder=self._task_type == "ref2va"
            )
        elif name == "audio_vae":
            self._audio_geometry = load_audio_geometry(path)
            if self._task_type == "ref2va" and self._tp.rank == 0:
                self._audio_codec = load_audio_decoder(
                    path, shards, self._device, with_encoder=True, with_decoder=self._is_driver
                )
                if self._is_driver:
                    self._audio_decoder = self._audio_codec
            elif self._task_type != "ref2va" and self._is_driver:
                self._audio_decoder = load_audio_decoder(path, shards, self._device)
            if self._is_driver:
                config = self._audio_decoder.config
                self._audio_mean = torch.tensor(config.latents_mean, dtype=torch.float32, device=self._device).view(
                    1, -1, 1
                )
                self._audio_std = torch.tensor(config.latents_std, dtype=torch.float32, device=self._device).view(
                    1, -1, 1
                )
        elif name in self._scheduler_shifts:
            config = MiniMaxH3Scheduler.load_config(path)
            self._scheduler_shifts[name] = float(config.get("shift", self._scheduler_shifts[name]))
        else:
            raise ValueError(f"Unknown MiniMax-H3 component: {name}")

    def forward(self, batch: Mapping[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
        if self._task_type == "fl2va" and (batch["batch_size"] != 1 or batch["media_sources"]):
            raise ValueError("MiniMax-H3 T2VA requires one prompt and no reference media")
        if self._task_type == "ref2va" and batch["batch_size"] != 1:
            raise ValueError("MiniMax-H3 Ref2VA requires one prompt")
        _, embedding, parameters = next(source for source in batch["tensor_sources"] if source[0] == "prompt_embed")
        generation = batch["generation_params"]
        if self._task_type == "ref2va":
            output = self._generate_reference(
                prompt_embed=embedding,
                prompt_tags=parameters["prompt_token_tags"],
                media_sources=batch["media_sources"],
                height=generation["height"],
                width=generation["width"],
                frames=generation["num_frames"],
                fps=generation["video_fps"],
                sampling_rate=generation["audio_sampling_rate"],
                steps=generation["num_inference_steps"],
                seed=generation["seed"] if generation["seed_is_set"] else 42,
            )
        else:
            output = self._generate(
                prompt_embed=embedding,
                prompt_tags=parameters["prompt_token_tags"],
                height=generation["height"],
                width=generation["width"],
                frames=generation["num_frames"],
                fps=generation["video_fps"],
                sampling_rate=generation["audio_sampling_rate"],
                steps=generation["num_inference_steps"],
                seed=generation["seed"] if generation["seed_is_set"] else 42,
            )
        if output is None:
            return [], [], []
        video, audio = output
        return [video], [audio], []

    def _video_geometry(self, height: int, width: int, frames: int) -> _VideoGeometry:
        config = self._video_decoder.config
        spatial = self._video_decoder.spatial_compression_ratio
        temporal = self._video_decoder.temporal_compression_ratio
        latents_per_clip = self._video_decoder.tokens_chunk_size
        _, ph, pw = self._denoiser.model.config.patch_size
        height_multiple, width_multiple = spatial * ph, spatial * pw
        height += (-height) % height_multiple
        width += (-width) % width_multiple
        frames += (latents_per_clip - frames) % config.clip_length
        latent_frames = (frames - latents_per_clip) // config.clip_length * latents_per_clip
        latent_frames += latents_per_clip - config.token_drop
        return _VideoGeometry(
            latent_frames,
            height // spatial,
            width // spatial,
            frames,
            spatial,
            temporal,
            latents_per_clip,
            ph,
            pw,
            config.clip_length,
            config.token_drop,
        )

    @torch.no_grad()
    def _generate(
        self,
        prompt_embed: torch.Tensor,
        prompt_tags: list[int],
        height: int,
        width: int,
        frames: int,
        fps: float,
        sampling_rate: int,
        steps: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        geometry = self._video_geometry(height, width, frames)
        duration = geometry.output_frames / fps
        if not 5 <= duration <= 15 or sampling_rate != self._audio_geometry.sampling_rate:
            raise ValueError("MiniMax-H3 requires 5–15 seconds after alignment and the checkpoint sampling rate")
        audio_latents = floor(duration * (sampling_rate // self._audio_geometry.hop_length) + 0.5)
        video_config = self._denoiser.model.config

        audio_shape = (1, _AUDIO_CHANNELS * audio_latents, video_config.audio_in_channels)
        # The denoiser reads the initial noise and the scheduler returns new tensors.
        video, audio = self._initial_noise(geometry, video_config.in_channels, audio_shape, seed, self._device)
        layout = self._packed_layout(tuple(prompt_tags), geometry, audio_latents, self._device)
        conditioning = self._denoiser.prepare_conditioning(prompt_embed, layout)
        video, audio = self._denoise(video, audio, conditioning, steps)

        frames_out = self._video_decoder.decode_pixels(geometry.unpatchify(video))
        if not self._is_driver:
            return None
        audio = audio.reshape(_AUDIO_CHANNELS, audio_latents, -1).transpose(1, 2)
        waveform = self._audio_decoder.decode(audio * self._audio_std + self._audio_mean, return_dict=False)[0]
        waveform = waveform.transpose(0, 1)[..., : floor(duration * sampling_rate + 0.5)].contiguous()
        # The VAE clamps pixels to [0, 1]. Match the MP4 encoder's RGB8 truncation.
        video_bytes = frames_out.mul_(255.0).to(torch.uint8)
        return video_bytes, waveform

    def _denoise(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        conditioning: PackedConditioning,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_template, audio_template, step_layouts = self._denoise_schedule(
            steps, self._scheduler_shifts["scheduler"], self._scheduler_shifts["audio_scheduler"], self._device
        )
        # Schedule tensors are read-only. Each request advances its own cursor.
        video_scheduler, audio_scheduler = copy(video_template), copy(audio_template)
        timestep_indices = conditioning.layout.timestep_indices
        for step, (times, mode) in enumerate(step_layouts):
            video_t = video_scheduler.timesteps[step]
            audio_t = audio_scheduler.timesteps[step]
            video_pred, audio_pred = self._denoiser(video, audio, times, timestep_indices[mode], conditioning)
            video = video_scheduler.step(video_pred, video_t, video, return_dict=False)[0]
            audio = audio_scheduler.step(audio_pred, audio_t, audio, return_dict=False)[0]
        return video, audio

    @torch.no_grad()
    def _prepare_reference_data(
        self,
        media_sources: Sequence[tuple[str, str, torch.Tensor, Mapping[str, Any]]],
        geometry: _VideoGeometry,
        num_frames: int,
        fps: float,
    ) -> _ReferenceData:
        _validate_reference_sources(media_sources)
        canvas_multiple = geometry.spatial_ratio * geometry.patch_height
        max_audio_samples = int(num_frames / fps * self._audio_geometry.sampling_rate)
        blocks: list[_ReferenceBlock] = []
        video_latents: list[torch.Tensor] = []
        audio_inputs: list[tuple[torch.Tensor, int]] = []
        for index, (name, _modality, tensor, parameters) in enumerate(media_sources):
            if name == "prompt_audio_in_video":
                continue
            if name == "image":
                image = _unbatched_reference_image(tensor)
                target_height, target_width = _reference_image_size(
                    int(image.size(1)), int(image.size(2)), canvas_multiple
                )
                pixels = _reference_image_pixels(image, canvas_multiple)
                block = _ReferenceBlock(
                    "image", 1, target_height // geometry.spatial_ratio, target_width // geometry.spatial_ratio
                )
                blocks.append(block)
                latent = self._video_decoder.encode_reference_condition(
                    pixels.to(self._device, torch.uint8).unsqueeze(0).unsqueeze(2)
                )
                _require_reference_latent_shape(latent, block)
                video_latents.append(latent)
                continue
            if name == "prompt_video":
                video = _unbatched_reference_video(tensor)
                source_fps = _reference_video_fps(parameters)
                target_height, target_width = _reference_canvas_size(
                    int(video.size(2)), int(video.size(3)), canvas_multiple
                )
                normalized_frames = _reference_normalized_frames(int(video.size(0)), source_fps, fps, num_frames)
                encode_frames = _reference_video_encode_frames(normalized_frames, geometry)
                latent_frames = _reference_latent_frames(encode_frames, geometry)
                has_audio = index + 1 < len(media_sources) and media_sources[index + 1][0] == "prompt_audio_in_video"
                audio_latents = 0
                soundtrack = None
                if has_audio:
                    soundtrack = _unbatched_reference_audio(media_sources[index + 1][2])
                    audio_latents = _reference_audio_latents(
                        int(soundtrack.size(1)), max_audio_samples, self._audio_geometry.hop_length
                    )
                block = _ReferenceBlock(
                    "video",
                    latent_frames,
                    target_height // geometry.spatial_ratio,
                    target_width // geometry.spatial_ratio,
                    audio_latents,
                )
                blocks.append(block)
                video_pixels = _reference_video_pixels(
                    video, source_fps, fps, num_frames, target_height, target_width, geometry
                )
                latent = self._video_decoder.encode_reference_condition(
                    video_pixels.permute(1, 0, 2, 3).to(self._device, torch.uint8).unsqueeze(0)
                )
                _require_reference_latent_shape(latent, block)
                video_latents.append(latent)
                if soundtrack is not None:
                    audio_inputs.append((soundtrack, _AUDIO_CHANNELS * audio_latents))
                continue
            audio = _unbatched_reference_audio(tensor)
            audio_latents = _reference_audio_latents(
                int(audio.size(1)), max_audio_samples, self._audio_geometry.hop_length
            )
            blocks.append(_ReferenceBlock("audio", audio_latents=audio_latents))
            audio_inputs.append((audio, _AUDIO_CHANNELS * audio_latents))
        audio_rows = tuple(
            self._reference_audio_rows(waveform, row_count, max_audio_samples) for waveform, row_count in audio_inputs
        )
        return _ReferenceData(tuple(blocks), tuple(video_latents), audio_rows)

    def _reference_audio_rows(self, waveform: torch.Tensor, row_count: int, max_audio_samples: int) -> torch.Tensor:
        if self._tp.rank == 0:
            row = self._encode_reference_audio(waveform, max_audio_samples)
        else:
            row = torch.empty(
                (1, row_count, self._audio_geometry.latent_channels),
                dtype=torch.float32,
                device=self._device,
            )
        if self._tp_broadcast is not None:
            return self._tp_broadcast(row, 0)
        if self._tp.size > 1:
            raise RuntimeError("MiniMax-H3 Ref2VA requires a TP broadcast callback")
        return row

    def _encode_reference_audio(self, waveform: torch.Tensor, max_audio_samples: int) -> torch.Tensor:
        samples = min(int(waveform.size(1)), max_audio_samples)
        trimmed = waveform[:, :samples].to(self._device, torch.float32).contiguous()
        return self._audio_codec.encode_reference_rows(trimmed)

    @torch.no_grad()
    def _generate_reference(
        self,
        prompt_embed: torch.Tensor,
        prompt_tags: list[int],
        media_sources: Sequence[tuple[str, str, torch.Tensor, Mapping[str, Any]]],
        height: int,
        width: int,
        frames: int,
        fps: float,
        sampling_rate: int,
        steps: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        geometry = self._video_geometry(height, width, frames)
        duration = geometry.output_frames / fps
        if not 5 <= duration <= 15 or sampling_rate != self._audio_geometry.sampling_rate:
            raise ValueError("MiniMax-H3 requires 5–15 seconds after alignment and the checkpoint sampling rate")
        audio_latents = floor(duration * (sampling_rate // self._audio_geometry.hop_length) + 0.5)
        video_config = self._denoiser.model.config

        references = self._prepare_reference_data(media_sources, geometry, geometry.output_frames, fps)
        layout = _build_reference_layout(prompt_tags, references, geometry, audio_latents, self._device)
        video_template, audio_template, _ = self._denoise_schedule(
            steps, self._scheduler_shifts["scheduler"], self._scheduler_shifts["audio_scheduler"], self._device
        )
        video_scheduler, audio_scheduler = copy(video_template), copy(audio_template)
        if references.video_latents:
            reference_generator = torch.Generator(device="cpu").manual_seed(seed)
            condition_rows = _reference_condition_rows(
                references.video_latents,
                reference_generator,
                video_scheduler,
                self._device,
                geometry.patch_height,
                geometry.patch_width,
            )
        else:
            condition_rows = torch.empty(
                (1, 0, video_config.in_channels * geometry.patch_height * geometry.patch_width),
                dtype=torch.float32,
                device=self._device,
            )
        if references.audio_rows:
            audio_condition_rows = torch.cat(references.audio_rows, dim=1).contiguous()
        else:
            audio_condition_rows = torch.empty(
                (1, 0, video_config.audio_in_channels), dtype=torch.float32, device=self._device
            )

        audio_shape = (1, _AUDIO_CHANNELS * audio_latents, video_config.audio_in_channels)
        video, audio = self._initial_noise(geometry, video_config.in_channels, audio_shape, seed, self._device)
        video = torch.cat([condition_rows, video], dim=1)
        audio = torch.cat([audio_condition_rows, audio], dim=1)
        conditioning = self._denoiser.prepare_conditioning(prompt_embed, layout)
        cache_request: CacheDiTRequest | None = None
        if self._cache_options is not None:
            block_count = len(self._denoiser.model.transformer_blocks)
            SimilarityResidualCacheState.validate_options(self._cache_options)
            if not 0 < self._cache_options.front_blocks < block_count:
                raise ValueError("MiniMax-H3 CacheDiT front_blocks must be within the block count")
            if not 0 <= self._cache_options.back_blocks < block_count - self._cache_options.front_blocks:
                raise ValueError("MiniMax-H3 CacheDiT back_blocks must fit after front_blocks")
            cache_request = make_cache_request(
                self._cache_options,
                layout,
                self._sp.rank,
                self._sp.size,
                self._sp_reduce,
                self._device,
            )
        video, audio = self._denoise_reference(
            video, audio, conditioning, video_scheduler, audio_scheduler, cache_request
        )
        if cache_request is not None:
            state = cache_request.state
            self.last_cache_stats = state.statistics()
            if self._is_driver:
                logger.info(
                    "MiniMax-H3 CacheDiT hits=%d dense_forwards=%d similarity_checks=%d hit_steps=%s",
                    state.cache_hits,
                    state.dense_forwards,
                    state.similarity_checks,
                    state.hit_steps,
                )
        video = video[:, layout.num_condition_video_rows :]
        audio = audio[:, layout.num_condition_audio_rows :]

        frames_out = self._video_decoder.decode_pixels(geometry.unpatchify(video))
        if not self._is_driver:
            return None
        audio = audio.reshape(_AUDIO_CHANNELS, audio_latents, -1).transpose(1, 2)
        waveform = self._audio_decoder.decode(audio * self._audio_std + self._audio_mean, return_dict=False)[0]
        waveform = waveform.transpose(0, 1)[..., : floor(duration * sampling_rate + 0.5)].contiguous()
        video_bytes = frames_out.mul_(255.0).to(torch.uint8)
        return video_bytes, waveform

    def _denoise_reference(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        conditioning: PackedConditioning,
        video_scheduler: MiniMaxH3Scheduler,
        audio_scheduler: MiniMaxH3Scheduler,
        cache: CacheDiTRequest | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layout = conditioning.layout
        video = video.clone()
        audio = audio.clone()
        for step in range(video_scheduler.timesteps.numel()):
            video_t = video_scheduler.timesteps[step]
            audio_t = audio_scheduler.timesteps[step]
            timesteps, timestep_indices = _reference_timesteps(layout, video_t, audio_t)
            video_pred, audio_pred = self._denoiser(
                video, audio, timesteps, timestep_indices, conditioning, step=step, cache=cache
            )
            if layout.num_condition_video_rows:
                video[:, layout.num_condition_video_rows :] = video_scheduler.step(
                    video_pred[:, layout.num_condition_video_rows :].float(),
                    video_t,
                    video[:, layout.num_condition_video_rows :],
                    return_dict=False,
                )[0]
            else:
                video = video_scheduler.step(video_pred.float(), video_t, video, return_dict=False)[0]
            if layout.num_condition_audio_rows:
                audio[:, layout.num_condition_audio_rows :] = audio_scheduler.step(
                    audio_pred[:, layout.num_condition_audio_rows :].float(),
                    audio_t,
                    audio[:, layout.num_condition_audio_rows :],
                    return_dict=False,
                )[0]
            else:
                audio = audio_scheduler.step(audio_pred.float(), audio_t, audio, return_dict=False)[0]
        return video, audio
