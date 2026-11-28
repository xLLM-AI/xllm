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

from xllm.python.dit.parallel import StateDictShard, load_component_weights


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


def load_video_decoder(
    path: str, shards: Sequence[StateDictShard], device: torch.device, group: TileParallel
) -> MiniMaxH3VideoDecoder:
    config = MiniMaxH3VideoDecoder.load_config(path)
    with torch.device("meta"):
        model = MiniMaxH3VideoDecoder.from_config(config)
    del model.encoder, model.quant_conv
    model._tile_parallel = group
    load_component_weights(model, shards, device, _fp32)
    head_dim = int(model.config.decoder_attention_head_dim * model.config.decoder_rope_dim_ratio)
    model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(head_dim, theta=model.config.decoder_rope_theta).to(device)
    model.set_attention_backend("_native_npu")
    return model


@dataclass(frozen=True)
class AudioGeometry:
    sampling_rate: int
    hop_length: int


def load_audio_geometry(path: str) -> AudioGeometry:
    config = AutoencoderKLMiniMaxH3Audio.load_config(path)
    hop_length = prod(config["encoder_rates"])
    if config["sampling_rate"] != 32000 or hop_length <= 0 or 32000 % hop_length:
        raise ValueError("MiniMax-H3 T2VA requires a 32 kHz audio VAE with an integral latent rate")
    return AudioGeometry(config["sampling_rate"], hop_length)


def load_audio_decoder(
    path: str, shards: Sequence[StateDictShard], device: torch.device
) -> AutoencoderKLMiniMaxH3Audio:
    config = AutoencoderKLMiniMaxH3Audio.load_config(path)
    with torch.device("meta"):
        model = AutoencoderKLMiniMaxH3Audio.from_config(config)
    del model.encoder, model.pre_block, model.mean_proj, model.logs_proj
    load_component_weights(model, shards, device, _fp32)
    return model
