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
from math import prod

import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
from torch.nn import functional as F

from xllm.python.dit.minimax_h3_model import StateDictShard, load_component_weights


@dataclass(frozen=True)
class TileParallel:
    rank: int
    size: int
    gather: Callable[[torch.Tensor], torch.Tensor]


def _fp32(_name: str) -> torch.dtype:
    return torch.float32


class MiniMaxH3VideoDecoder(AutoencoderKLMiniMaxH3):
    _tile_parallel: TileParallel

    def _decode_clip(self, latent: torch.Tensor) -> torch.Tensor:
        ratio = self.spatial_compression_ratio
        height, width = (size * ratio for size in latent.shape[-2:])
        y_starts, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_starts, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )
        cols = len(x_starts)
        count = len(y_starts) * cols

        def decode_tile(index: int) -> torch.Tensor:
            row, col = divmod(index, cols)
            y, x = y_starts[row] // ratio, x_starts[col] // ratio
            tile = latent[..., y : y + y_lengths[row] // ratio, x : x + x_lengths[col] // ratio]
            return self.decoder(self.post_quant_conv(tile.contiguous()))

        group = self._tile_parallel
        if count < group.size:
            decoded = [decode_tile(index) for index in range(count)]
        else:
            local = [decode_tile(index) for index in range(group.rank, count, group.size)]
            slots = (count + group.size - 1) // group.size
            local.extend(torch.zeros_like(local[0]) for _ in range(slots - len(local)))
            gathered = group.gather(torch.stack(local))
            decoded = [gathered[index % group.size, index // group.size] for index in range(count)]
        rows = [decoded[index : index + cols] for index in range(0, count, cols)]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        mean = latents.new_tensor(self.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.config.latents_std).view(1, -1, 1, 1, 1)
        decoded = self.decode(latents * std + mean, return_dict=False)[0]
        pixel_mean = decoded.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = decoded.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        return decoded.mul_(pixel_std).add_(pixel_mean).clamp_(0, 1).permute(0, 2, 1, 3, 4).contiguous()

    def encode_reference_condition(self, pixels: torch.Tensor, seed: int = 42) -> torch.Tensor:
        mean = pixels.new_tensor(self.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = pixels.new_tensor(self.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
        normalized = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        posterior = self.encode(normalized, return_dict=False)[0]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = posterior.sample(generator=generator).to(torch.float16).to(torch.float32)
        return (latent - mean) / std


def load_video_decoder(
    path: str,
    shards: Sequence[StateDictShard],
    device: torch.device,
    group: TileParallel,
    *,
    with_encoder: bool = False,
) -> MiniMaxH3VideoDecoder:
    config = MiniMaxH3VideoDecoder.load_config(path)
    with torch.device("meta"):
        model = MiniMaxH3VideoDecoder.from_config(config)
    if not with_encoder:
        del model.encoder, model.quant_conv
    model._tile_parallel = group
    load_component_weights(model, shards, device, _fp32)
    head_dim = int(model.config.decoder_attention_head_dim * model.config.decoder_rope_dim_ratio)
    model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(head_dim, theta=model.config.decoder_rope_theta).to(device)
    model.set_attention_backend("_native_npu")
    return model


class MiniMaxH3AudioCodec(AutoencoderKLMiniMaxH3Audio):
    def encode_reference_rows(self, waveform: torch.Tensor) -> torch.Tensor:
        hop_length = prod(self.config.encoder_rates)
        sample = waveform.to(torch.float32).unsqueeze(1)
        right_pad = (-sample.size(2)) % hop_length
        if right_pad:
            sample = F.pad(sample, (0, right_pad))
        posterior = self.encode(sample, return_dict=False)[0]
        latent = posterior.mode().to(torch.float32)
        mean = latent.new_tensor(self.config.latents_mean, dtype=torch.float32).view(1, -1, 1)
        std = latent.new_tensor(self.config.latents_std, dtype=torch.float32).view(1, -1, 1)
        return ((latent - mean) / std).reshape(1, -1, latent.size(1)).contiguous()


@dataclass(frozen=True)
class AudioGeometry:
    sampling_rate: int
    hop_length: int
    latent_channels: int


def load_audio_geometry(path: str) -> AudioGeometry:
    config = AutoencoderKLMiniMaxH3Audio.load_config(path)
    hop_length = prod(config["encoder_rates"])
    if config["sampling_rate"] != 32000 or hop_length <= 0 or 32000 % hop_length:
        raise ValueError("MiniMax-H3 T2VA requires a 32 kHz audio VAE with an integral latent rate")
    return AudioGeometry(config["sampling_rate"], hop_length, config["latent_channels"])


def load_audio_decoder(
    path: str,
    shards: Sequence[StateDictShard],
    device: torch.device,
    *,
    with_encoder: bool = False,
    with_decoder: bool = True,
) -> MiniMaxH3AudioCodec:
    config = MiniMaxH3AudioCodec.load_config(path)
    with torch.device("meta"):
        model = MiniMaxH3AudioCodec.from_config(config)
    if not with_encoder:
        del model.encoder, model.pre_block, model.mean_proj, model.logs_proj
    if not with_decoder:
        del model.decoder, model.dec_in_proj
    load_component_weights(model, shards, device, _fp32)
    return model
