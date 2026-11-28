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

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from math import prod

import torch
from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoRotaryPosEmbed
from torch import nn
from torch.nn import functional as F

from scripts.logger import logger
from xllm.python.dit.minimax_h3_model import StateDictShard, TensorParallel, load_component_weights


@dataclass(frozen=True)
class TileParallel:
    rank: int
    size: int
    gather: Callable[[torch.Tensor], torch.Tensor]
    barrier: Callable[[], None] | None = None


def _fp32(_name: str) -> torch.dtype:
    return torch.float32


class _StageReady:
    def __init__(self, stages: "_PartitionedBlocks", operation: str) -> None:
        self._stages = stages
        self._operation = operation

    def __call__(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        if not self._stages.vote(True, hidden_states):
            raise RuntimeError(f"MiniMax-H3 video VAE peer failed during {self._operation}")
        return hidden_states


class _StageBlocks:
    def __init__(
        self,
        stages: "_PartitionedBlocks",
        blocks: tuple[nn.Module, ...],
        run_local: bool,
        operation: str,
    ) -> None:
        self._stages = stages
        self._blocks = blocks
        self._run_local = run_local
        self._operation = operation

    def __call__(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        error: BaseException | None = None
        try:
            if self._run_local:
                for block in self._blocks:
                    hidden_states = block(hidden_states, rotary_emb)
            if not hidden_states.is_contiguous():
                raise RuntimeError("MiniMax-H3 video VAE decoder activations are not contiguous")
        except BaseException as exc:
            error = exc
        if self._stages.vote(error is None, hidden_states):
            return hidden_states
        if error is not None:
            raise error
        raise RuntimeError(f"MiniMax-H3 video VAE peer failed during {self._operation}")


class _StageBroadcast:
    def __init__(self, broadcast: Callable[[torch.Tensor, int], torch.Tensor], root: int) -> None:
        self._broadcast = broadcast
        self._root = root

    def __call__(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self._broadcast(hidden_states, self._root)


class _PartitionedBlocks(nn.ModuleDict):
    """Decoder block halves staged with the native peer handoff order."""

    def __init__(self, blocks: dict[str, nn.Module], group: TensorParallel) -> None:
        super().__init__(blocks)
        if group.size != 2 or group.broadcast is None:
            raise ValueError("MiniMax-H3 video VAE decoder partition requires a TP2 group")
        self._group = group
        self._stage_rank = group.rank
        self._local_blocks = tuple(self._modules.values())
        self._ready: torch.Tensor | None = None
        self._failed = False
        self.layer_offset = min(int(key) for key in blocks)
        self.layer_count = len(blocks)

    def set_ready(self, ready: torch.Tensor) -> None:
        self._ready = ready

    @property
    def failed(self) -> bool:
        return self._failed

    def vote(self, succeeded: bool, hidden_states: torch.Tensor) -> bool:
        if self._ready is None:
            self._ready = hidden_states.new_empty((1,), dtype=torch.int32)
        self._ready.fill_(1 if succeeded else 0)
        self._ready = self._group.reduce(self._ready)
        all_succeeded = int(self._ready.item()) == self._group.size
        if not all_succeeded:
            self._failed = True
        return all_succeeded

    def run(self, operation: str, work: Callable[[], torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        error: BaseException | None = None
        result: torch.Tensor | None = None
        try:
            result = work()
        except BaseException as exc:
            error = exc
        if not self.vote(error is None, reference):
            if error is not None:
                raise error
            raise RuntimeError(f"MiniMax-H3 video VAE peer failed during {operation}")
        return result

    def __iter__(self) -> Iterator[Callable[..., torch.Tensor]]:
        yield _StageReady(self, "decoder input preparation")
        for rank in range(2):
            yield _StageBlocks(self, self._local_blocks, self._stage_rank == rank, "decoder block partition")
            yield _StageBroadcast(self._group.broadcast, rank)


class _StagedDecoder(nn.Module):
    """Vote the decoder output stage without copying the Diffusers forward."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    @property
    def _stages(self) -> "_PartitionedBlocks":
        return self.decoder.transformer_blocks

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        error: BaseException | None = None
        result: torch.Tensor | None = None
        try:
            result = self.decoder(hidden_states)
        except BaseException as exc:
            error = exc
        if not self._stages.failed:
            if not self._stages.vote(error is None, hidden_states):
                if error is not None:
                    raise error
                raise RuntimeError("MiniMax-H3 video VAE peer failed during decoder execution")
        if error is not None:
            raise error
        return result


class MiniMaxH3VideoDecoder(AutoencoderKLMiniMaxH3):
    _tile_parallel: TileParallel
    _decoder_partition: TensorParallel | None
    _decoder_layer_offset: int = 0
    _decoder_layer_count: int = 0
    _owned_tiles: tuple[int, ...] = ()
    _tile_count: int = 0

    @property
    def _decoder_stages(self) -> "_PartitionedBlocks | None":
        decoder = self.decoder.decoder if isinstance(self.decoder, _StagedDecoder) else self.decoder
        blocks = decoder.transformer_blocks
        return blocks if isinstance(blocks, _PartitionedBlocks) else None

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
            stages = self._decoder_stages
            if stages is None:
                post_quant = self.post_quant_conv(tile.contiguous())
            else:
                post_quant = stages.run("tile input preparation", lambda: self.post_quant_conv(tile.contiguous()), tile)
            return self.decoder(post_quant)

        group = self._tile_parallel
        partition = self._decoder_partition
        barrier = group.barrier
        owned_tiles: list[int] = []
        if barrier is not None:
            barrier()
        if count < group.size:
            decoded = [decode_tile(index) for index in range(count)]
            owned_tiles = list(range(count))
        else:
            if partition is None:
                local = [decode_tile(index) for index in range(group.rank, count, group.size)]
                owned_tiles = list(range(group.rank, count, group.size))
            else:
                # Both ranks of a TP2 pair decode the pair's tiles in the same
                # order and keep only their own VAE tile.
                pair_begin = group.rank - partition.rank
                local = []
                for start in range(pair_begin, count, group.size):
                    for index in range(start, min(start + 2, count)):
                        tile = decode_tile(index)
                        if index % group.size == group.rank:
                            local.append(tile)
                            owned_tiles.append(index)
            slots = (count + group.size - 1) // group.size
            local.extend(torch.zeros_like(local[0]) for _ in range(slots - len(local)))
            if barrier is not None:
                barrier()
            gathered = group.gather(torch.stack(local))
            decoded = [gathered[index % group.size, index // group.size] for index in range(count)]
        self._owned_tiles = tuple(owned_tiles)
        self._tile_count = count
        if barrier is not None:
            barrier()
        rows = [decoded[index : index + cols] for index in range(0, count, cols)]
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    def _log_decoder_assignment(self) -> None:
        partition = self._decoder_partition
        parameters = tuple(self.decoder.parameters())
        logger.info(
            "MiniMax-H3 video VAE rank=%d partitioned=%d stage_rank=%d "
            "layer_offset=%d layer_count=%d decoder_parameters=%d decoder_bytes=%d "
            "tiles=%d owned_tiles=%s",
            self._tile_parallel.rank,
            1 if partition is not None else 0,
            0 if partition is None else partition.rank,
            self._decoder_layer_offset,
            self._decoder_layer_count,
            sum(parameter.numel() for parameter in parameters),
            sum(parameter.numel() * parameter.element_size() for parameter in parameters),
            self._tile_count,
            ",".join(str(index) for index in self._owned_tiles) if self._owned_tiles else "none",
        )

    def decode_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        mean = latents.new_tensor(self.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.config.latents_std).view(1, -1, 1, 1, 1)
        decoded = self.decode(latents * std + mean, return_dict=False)[0]
        self._log_decoder_assignment()
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


def _partition_decoder_blocks(decoder: nn.Module, group: TensorParallel) -> tuple[int, int]:
    blocks = decoder.transformer_blocks
    layers = len(blocks)
    split = layers // 2
    begin = split if group.rank == 1 else 0
    end = split if group.rank == 0 else layers
    decoder.transformer_blocks = _PartitionedBlocks({str(index): blocks[index] for index in range(begin, end)}, group)
    return begin, end


def load_video_decoder(
    path: str,
    shards: Sequence[StateDictShard],
    device: torch.device,
    group: TileParallel,
    *,
    with_encoder: bool = False,
    partition: TensorParallel | None = None,
) -> MiniMaxH3VideoDecoder:
    config = MiniMaxH3VideoDecoder.load_config(path)
    with torch.device("meta"):
        model = MiniMaxH3VideoDecoder.from_config(config)
    if not with_encoder:
        del model.encoder, model.quant_conv
    if partition is not None and partition.rank != group.rank % 2:
        raise ValueError("MiniMax-H3 video VAE decoder partition requires paired TP2 and VAE ranks")
    model._tile_parallel = group
    model._decoder_partition = partition
    model._decoder_layer_count = len(model.decoder.transformer_blocks)
    stages: _PartitionedBlocks | None = None
    if partition is not None:
        begin, end = _partition_decoder_blocks(model.decoder, partition)
        model._decoder_layer_offset = begin
        model._decoder_layer_count = end - begin
        stages = model.decoder.transformer_blocks
    load_component_weights(model, shards, device, _fp32)
    if stages is not None:
        stages.set_ready(torch.ones((1,), dtype=torch.int32, device=device))
    head_dim = int(model.config.decoder_attention_head_dim * model.config.decoder_rope_dim_ratio)
    model.decoder.rope = MiniMaxH3VideoRotaryPosEmbed(head_dim, theta=model.config.decoder_rope_theta).to(device)
    model.set_attention_backend("_native_npu")
    if stages is not None:
        model.decoder = _StagedDecoder(model.decoder)
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
