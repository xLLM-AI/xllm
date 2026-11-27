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
# limitations under the License.

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    UMT5Config,
    UMT5EncoderModel,
)

from xllm.python.dit import ComponentLoad
from xllm.python.dit.parallel import SequenceParallel, StateDictShard, TensorParallel, load_component_weights
from xllm.python.dit.wan_i2v_model import load_transformer


class _WanCausalConv3d(WanCausalConv3d):
    def __init__(self, convolution: WanCausalConv3d) -> None:
        width, _, height, _, temporal, _ = convolution._padding
        super().__init__(
            convolution.in_channels,
            convolution.out_channels,
            convolution.kernel_size,
            convolution.stride,
            padding=(temporal // 2, height, width),
        )
        self.padding = (0, height, width)
        self._temporal_padding = temporal

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        temporal_padding = self._temporal_padding
        if cache_x is not None and temporal_padding:
            x = torch.cat((cache_x.to(x.device), x), dim=2)
            temporal_padding -= cache_x.size(2)
        # PyTorch copies the full tensor when F.pad is called with all zeros.
        if temporal_padding:
            x = F.pad(x, (0, 0, 0, 0, temporal_padding, 0))
        return torch.nn.Conv3d.forward(self, x)


class WanI2VPipeline:
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
        self._is_driver = is_driver
        self._tp = TensorParallel(tp_rank, tp_size, tp_reduce, tp_gather)
        self._sp = SequenceParallel(sp_rank, sp_size, sp_gather, sp_exchange)
        self._components: dict[str, Any] = {}

    def checkpoint_components(self) -> tuple[ComponentLoad, ...]:
        return (
            ComponentLoad("transformer"),
            ComponentLoad("text_encoder"),
            ComponentLoad("tokenizer", load_weights=False),
            ComponentLoad("image_encoder"),
            ComponentLoad("image_processor", load_weights=False),
            ComponentLoad("vae"),
            ComponentLoad("scheduler", load_weights=False),
        )

    def load_component(self, name: str, path: str, shards: Sequence[StateDictShard]) -> None:
        if name == "transformer":
            component = load_transformer(path, shards, self._device, self._tp, self._sp)
        elif name == "text_encoder":
            config = UMT5Config.from_pretrained(path, local_files_only=True)
            config._attn_implementation = "eager"
            with torch.device("meta"):
                component = UMT5EncoderModel(config)
            component.tie_weights()
            load_component_weights(component, shards, self._device, lambda _: torch.bfloat16)
        elif name == "tokenizer":
            # Match the SentencePiece tokenizer used to train Wan's UMT5 encoder.
            component = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
        elif name == "image_encoder":
            config = CLIPVisionConfig.from_pretrained(path, local_files_only=True)
            config._attn_implementation = "eager"
            with torch.device("meta"):
                component = CLIPVisionModelWithProjection(config)
            load_component_weights(component, shards, self._device, lambda _: torch.float32)
            embeddings = component.vision_model.embeddings
            embeddings.position_ids = torch.arange(embeddings.num_positions, device=self._device).expand(1, -1)
        elif name == "image_processor":
            component = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
        elif name == "vae":
            config = AutoencoderKLWan.load_config(path)
            with torch.device("meta"):
                component = AutoencoderKLWan.from_config(config)
                for layer_name, layer in component.named_modules():
                    if isinstance(layer, WanCausalConv3d):
                        component.set_submodule(layer_name, _WanCausalConv3d(layer))
            load_component_weights(component, shards, self._device, lambda _: torch.float32)
        elif name == "scheduler":
            config = UniPCMultistepScheduler.load_config(path)
            if config.get("_class_name") != "UniPCMultistepScheduler" or not config.get("use_flow_sigmas"):
                raise ValueError("Wan2.1 I2V requires a UniPC flow-matching scheduler")
            component = UniPCMultistepScheduler.from_config(config)
        else:
            raise ValueError(f"Unknown Wan2.1 I2V component: {name}")
        self._components[name] = component
        if name == "scheduler":
            self._pipeline = WanImageToVideoPipeline(**self._components)
            self._pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def forward(self, batch: Mapping[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
        generation = batch["generation_params"]
        tensors = {name: tensor for name, tensor, _ in batch["tensor_sources"]}
        # The worker batches decoded RGB8 media as BCHW. CLIP's processor expects
        # byte pixels, while the VAE processor expects floating point [0, 1].
        image = batch["media_sources"][0][2].cpu()
        pipeline = self._pipeline
        image_embed = pipeline.encode_image(image, device=self._device).to(pipeline.transformer.dtype)
        prompt_embed = tensors.get("prompt_embed")
        negative_embed = tensors.get("negative_prompt_embed")
        prompt_embed, negative_embed = pipeline.encode_prompt(
            prompt=None if prompt_embed is not None else batch["prompts"],
            negative_prompt=None if negative_embed is not None else batch["negative_prompts"] or None,
            do_classifier_free_guidance=generation["guidance_scale"] > 1,
            prompt_embeds=prompt_embed,
            negative_prompt_embeds=negative_embed,
            max_sequence_length=512,
            device=self._device,
            dtype=pipeline.transformer.dtype,
        )
        prompt_embed = prompt_embed.to(pipeline.transformer.dtype)
        if negative_embed is not None:
            negative_embed = negative_embed.to(pipeline.transformer.dtype)
        # Keep the checkpoint solver settings while applying xLLM's request shift.
        scheduler = UniPCMultistepScheduler.from_config(
            self._components["scheduler"].config, flow_shift=generation["flow_shift"]
        )
        scheduler.set_timesteps(generation["num_inference_steps"], device=self._device)
        frames = generation["num_frames"] // pipeline.vae_scale_factor_temporal
        frames = frames * pipeline.vae_scale_factor_temporal + 1
        height, width = generation["height"], generation["width"]
        pixels = pipeline.video_processor.preprocess(image.float() / 255, height=height, width=width).to(self._device)
        generator = torch.Generator(device="cpu").manual_seed(generation["seed"])
        latents, condition = pipeline.prepare_latents(
            image=pixels,
            batch_size=1,
            num_channels_latents=pipeline.vae.config.z_dim,
            height=height,
            width=width,
            num_frames=frames,
            dtype=torch.float32,
            device=self._device,
            generator=generator,
            latents=tensors.get("latent"),
        )
        rotary_emb = pipeline.transformer.prepare_rotary(latents)
        for timestep in scheduler.timesteps:
            model_input = torch.cat((latents, condition), dim=1).to(pipeline.transformer.dtype)
            timestep_input = timestep.expand(1)
            noise = pipeline.transformer(
                hidden_states=model_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embed,
                encoder_hidden_states_image=image_embed,
                return_dict=False,
                rotary_emb=rotary_emb,
            )[0]
            if generation["guidance_scale"] > 1:
                unconditional = pipeline.transformer(
                    hidden_states=model_input,
                    timestep=timestep_input,
                    encoder_hidden_states=negative_embed,
                    encoder_hidden_states_image=image_embed,
                    return_dict=False,
                    rotary_emb=rotary_emb,
                )[0]
                noise = unconditional + generation["guidance_scale"] * (noise - unconditional)
            latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]

        if not self._is_driver:
            return [], [], []
        mean = latents.new_tensor(pipeline.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(pipeline.vae.config.latents_std).view(1, -1, 1, 1, 1)
        video = pipeline.vae.decode(latents * std + mean, return_dict=False)[0]
        video = video.float().mul_(0.5).add_(0.5).clamp_(0, 1).mul_(255).to(torch.uint8)
        # The native video encoder consumes B,F,C,H,W RGB8 tensors.
        return [video.permute(0, 2, 1, 3, 4).contiguous()], [], []
