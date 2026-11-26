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
from math import floor, sqrt
from typing import Any

import torch
from diffusers import MiniMaxH3Scheduler

from xllm.python.dit import ComponentLoad
from xllm.python.dit.minimax_h3_model import (
    MiniMaxH3Denoiser,
    PackedConditioning,
    PackedLayout,
    SequenceParallel,
    StateDictShard,
    TensorParallel,
)
from xllm.python.dit.minimax_h3_vae import TileParallel, load_audio_decoder, load_audio_geometry, load_video_decoder

_AUDIO_CHANNELS = 2
_ROPE_FRAME_SCALE = 5 / 3


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
        sp_exchange: Callable[[torch.Tensor, bool], torch.Tensor],
        vae_gather: Callable[[torch.Tensor], torch.Tensor],
        *,
        laser_attention_enabled: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._laser_attention_enabled = laser_attention_enabled
        self._is_driver = is_driver
        self._tp = TensorParallel(tp_rank, tp_size, tp_reduce, tp_gather)
        self._sp = SequenceParallel(sp_rank, sp_size, sp_gather, sp_exchange)
        self._tiles = TileParallel(vae_rank, vae_size, vae_gather)
        self._scheduler_shifts = {"scheduler": 12.0, "audio_scheduler": 3.0}
        self._packed_layout = lru_cache(maxsize=1)(_build_layout)
        self._initial_noise = lru_cache(maxsize=1)(_prepare_noise)
        self._denoise_schedule = lru_cache(maxsize=1)(_build_denoise_schedule)

    def checkpoint_components(self) -> tuple[ComponentLoad, ...]:
        return (
            ComponentLoad("transformer"),
            ComponentLoad("vae"),
            ComponentLoad("audio_vae", load_weights=self._is_driver),
            ComponentLoad("scheduler", required=False, load_weights=False),
            ComponentLoad("audio_scheduler", required=False, load_weights=False),
        )

    def load_component(self, name: str, path: str, shards: Sequence[StateDictShard]) -> None:
        if name == "transformer":
            self._denoiser = MiniMaxH3Denoiser(
                path, shards, self._device, self._tp, self._sp, laser_attention_enabled=self._laser_attention_enabled
            )
        elif name == "vae":
            self._video_decoder = load_video_decoder(path, shards, self._device, self._tiles)
        elif name == "audio_vae":
            self._audio_geometry = load_audio_geometry(path)
            if self._is_driver:
                self._audio_decoder = load_audio_decoder(path, shards, self._device)
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
        if batch["batch_size"] != 1 or batch["media_sources"]:
            raise ValueError("MiniMax-H3 T2VA requires one prompt and no reference media")
        _, embedding, parameters = next(source for source in batch["tensor_sources"] if source[0] == "prompt_embed")
        generation = batch["generation_params"]
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
            latent_frames, height // spatial, width // spatial, frames, spatial, temporal, latents_per_clip, ph, pw
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
