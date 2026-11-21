# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
# Architecture adapted from vLLM-Omni (Apache-2.0), commit
# 74b022ec52668b7914bd3da742649f7d44975b99, diffusion/models/gr00t/modeling/.
"""GR00T N1.7 eager action head using portable PyTorch/Diffusers operators.

Parameter names match the checkpoint below ``action_head.``. This module does
not load a second VLM or depend on vLLM, CUDA FlashAttention, or TensorRT.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from xllm.vla.gr00t_attention import _FlashKeyIndices, _Gr00tFlashProcessor


@dataclass(frozen=True)
class Gr00tActionConfig:
    backbone_embedding_dim: int = 2048
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    max_state_dim: int = 132
    max_action_dim: int = 132
    max_num_embodiments: int = 32
    state_history_length: int = 1
    action_horizon: int = 40
    max_seq_len: int = 1024
    num_inference_timesteps: int = 4
    num_timestep_buckets: int = 1000
    add_pos_embed: bool = True
    use_vlln: bool = True
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    diffusion_model_cfg: dict[str, Any] = field(default_factory=dict)
    vl_self_attention_cfg: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "Gr00tActionConfig":
        values = {key: config[key] for key in cls.__dataclass_fields__ if key in config}
        result = cls(**values)
        for key in (
            "backbone_embedding_dim",
            "hidden_size",
            "input_embedding_dim",
            "max_state_dim",
            "max_action_dim",
            "max_num_embodiments",
            "state_history_length",
            "action_horizon",
            "num_inference_timesteps",
            "num_timestep_buckets",
            "attend_text_every_n_blocks",
        ):
            if getattr(result, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not result.diffusion_model_cfg:
            raise ValueError("GR00T checkpoint must specify diffusion_model_cfg")
        inner_dim = result.diffusion_model_cfg["num_attention_heads"] * result.diffusion_model_cfg["attention_head_dim"]
        if inner_dim != result.input_embedding_dim:
            raise ValueError("DiT attention width must equal input_embedding_dim")
        if result.diffusion_model_cfg["output_dim"] != result.hidden_size:
            raise ValueError("DiT output_dim must equal action decoder hidden_size")
        if result.add_pos_embed and result.action_horizon > result.max_seq_len:
            raise ValueError("action_horizon exceeds the learned position embedding table")
        if result.use_alternate_vl_dit and not result.diffusion_model_cfg.get("interleave_self_attention"):
            raise ValueError("AlternateVLDiT requires interleave_self_attention")
        return result


class _CategoryLinear(nn.Module):
    def __init__(self, categories: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(categories, input_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(categories, output_dim))
        nn.init.normal_(self.W, std=0.02)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.W[ids]) + self.b[ids].unsqueeze(1)


class _CategoryMLP(nn.Module):
    def __init__(self, categories: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = _CategoryLinear(categories, input_dim, hidden_dim)
        self.layer2 = _CategoryLinear(categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, ids)), ids)


class _ActionEncoder(nn.Module):
    def __init__(self, config: Gr00tActionConfig) -> None:
        super().__init__()
        width = config.input_embedding_dim
        self.W1 = _CategoryLinear(config.max_num_embodiments, config.max_action_dim, width)
        self.W2 = _CategoryLinear(config.max_num_embodiments, 2 * width, width)
        self.W3 = _CategoryLinear(config.max_num_embodiments, width, width)
        self.width = width

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        half = self.width // 2
        frequencies = torch.exp(
            torch.arange(half, device=actions.device, dtype=torch.float32) * (-math.log(10000) / half)
        )
        angles = timesteps.float()[:, None, None] * frequencies[None, None, :]
        time_embeds = torch.cat((angles.sin(), angles.cos()), dim=-1)
        embeds = self.W1(actions, ids)
        time_embeds = time_embeds.expand(-1, actions.shape[1], -1).to(embeds.dtype)
        hidden = self.W2(torch.cat((embeds, time_embeds), dim=-1), ids)
        return self.W3(hidden * torch.sigmoid(hidden), ids)


class _AdaNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, 2 * width)
        self.norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(F.silu(time)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class _Block(nn.Module):
    def __init__(self, config: dict[str, Any], cross_dim: int | None, adaptive: bool) -> None:
        super().__init__()
        width = config["num_attention_heads"] * config["attention_head_dim"]
        if config.get("positional_embeddings") is not None:
            raise ValueError("This GR00T implementation requires positional_embeddings=null")
        self.norm1 = _AdaNorm(width) if adaptive else nn.LayerNorm(width, eps=1e-5)
        self.attn1 = Attention(
            query_dim=width,
            heads=config["num_attention_heads"],
            dim_head=config["attention_head_dim"],
            dropout=config.get("dropout", 0.1),
            bias=config.get("attention_bias", True),
            cross_attention_dim=cross_dim,
            upcast_attention=config.get("upcast_attention", False),
        )
        self.norm3 = nn.LayerNorm(
            width,
            eps=config.get("norm_eps", 1e-5),
            elementwise_affine=config.get("norm_elementwise_affine", False) if adaptive else True,
        )
        self.ff = FeedForward(
            width,
            dropout=config.get("dropout", 0.1),
            activation_fn=config.get("activation_fn", "gelu-approximate"),
            final_dropout=config.get("final_dropout", True),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.norm1(x, time) if isinstance(self.norm1, _AdaNorm) else self.norm1(x)
        x = x + self.attn1(normalized, encoder_hidden_states=context, attention_mask=mask)
        return x + self.ff(self.norm3(x))


class _TimestepEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=width)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(timestep).to(self.timestep_embedder.linear_1.weight.dtype))


class _DiT(nn.Module):
    def __init__(self, config: Gr00tActionConfig, flash_keys: _FlashKeyIndices | None = None) -> None:
        super().__init__()
        self.config = config
        args = config.diffusion_model_cfg
        if args.get("norm_type", "ada_norm") != "ada_norm":
            raise ValueError("GR00T action DiT requires ada_norm")
        width = config.input_embedding_dim
        self.timestep_encoder = _TimestepEncoder(width)
        self.transformer_blocks = nn.ModuleList(
            [
                _Block(
                    args,
                    None if i % 2 and args.get("interleave_self_attention") else config.backbone_embedding_dim,
                    True,
                )
                for i in range(args["num_layers"])
            ]
        )
        if flash_keys is not None:
            for i, block in enumerate(self.transformer_blocks):
                kind = 0 if i % (2 * config.attend_text_every_n_blocks) == 0 else 1
                block.attn1.set_processor(_Gr00tFlashProcessor(flash_keys, kind))
        self.norm_out = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(width, 2 * width)
        self.proj_out_2 = nn.Linear(width, config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        time = self.timestep_encoder(timestep)
        for i, block in enumerate(self.transformer_blocks):
            if i % 2 and self.config.diffusion_model_cfg.get("interleave_self_attention"):
                x = block(x, time)
                continue
            mask = None
            if self.config.use_alternate_vl_dit:
                selected = ~image_mask if i % (2 * self.config.attend_text_every_n_blocks) == 0 else image_mask
                mask = selected & attention_mask
            x = block(x, time, context, mask)
        shift, scale = self.proj_out_1(F.silu(time)).chunk(2, dim=1)
        return self.proj_out_2(self.norm_out(x) * (1 + scale[:, None]) + shift[:, None])


class _VLTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(config, None, False) for _ in range(config["num_layers"])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block(x)
        return x


class Gr00tActionHead(nn.Module):
    """Predict normalized actions from pre-final-RMSNorm VLM features.

    Call ``eval()`` for inference. ``initial_noise`` allows exact comparisons
    between backends whose random number generators differ. RTC is not yet
    supported; each invocation produces an independent complete action chunk.
    """

    def __init__(self, config: Gr00tActionConfig, *, attention_backend: str = "math") -> None:
        super().__init__()
        if attention_backend not in ("math", "flash"):
            raise ValueError("GR00T action attention_backend must be math or flash")
        if attention_backend == "flash" and (
            not config.use_alternate_vl_dit
            or config.attend_text_every_n_blocks != 2
            or not config.diffusion_model_cfg.get("interleave_self_attention")
        ):
            raise ValueError("GR00T Flash requires the alternating text/image checkpoint layout")
        self.attention_backend = attention_backend
        self._flash_keys = _FlashKeyIndices() if attention_backend == "flash" else None
        self.config = config
        self.model = _DiT(config, self._flash_keys)
        self.state_encoder = _CategoryMLP(
            config.max_num_embodiments,
            config.max_state_dim * config.state_history_length,
            config.hidden_size,
            config.input_embedding_dim,
        )
        self.action_encoder = _ActionEncoder(config)
        self.action_decoder = _CategoryMLP(
            config.max_num_embodiments, config.hidden_size, config.hidden_size, config.max_action_dim
        )
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        vl_config = config.vl_self_attention_cfg
        self.vl_self_attention = (
            _VLTransformer(vl_config) if vl_config and vl_config.get("num_layers", 0) > 0 else nn.Identity()
        )
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
        if self._flash_keys is not None:
            for module in self.vl_self_attention.modules():
                if isinstance(module, Attention):
                    module.set_processor(_Gr00tFlashProcessor(self._flash_keys))

    @torch.inference_mode()
    def forward(
        self,
        backbone_features: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self._validate_inputs(backbone_features, state, embodiment_id, image_mask, attention_mask)
        backend = SDPBackend.MATH if self.attention_backend == "math" else SDPBackend.FLASH_ATTENTION
        if self._flash_keys is not None:
            self._flash_keys.indices = self._prepare_flash_indices(image_mask, attention_mask)
        with sdpa_kernel(backend):
            return self._forward(
                backbone_features, state, embodiment_id, image_mask, attention_mask, initial_noise, generator
            )

    def _prepare_flash_indices(
        self, image_mask: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_mask.shape[0] != 1:
            raise ValueError("GR00T Flash requires batch size one")
        text = torch.nonzero(((~image_mask) & attention_mask)[0], as_tuple=False).flatten()
        image = torch.nonzero((image_mask & attention_mask)[0], as_tuple=False).flatten()
        if not text.numel() or not image.numel():
            raise ValueError("GR00T Flash requires visible text and image keys")
        return text, image

    def _validate_inputs(
        self,
        backbone_features: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        config = self.config
        batch, sequence, width = backbone_features.shape
        if self.training:
            raise ValueError("GR00T policy inference requires eval()")
        if width != config.backbone_embedding_dim or state.shape != (
            batch,
            config.state_history_length,
            config.max_state_dim,
        ):
            raise ValueError("Backbone or state shape does not match the GR00T checkpoint")
        if embodiment_id.shape != (batch,) or embodiment_id.dtype != torch.int64:
            raise ValueError("embodiment_id must be an int64 tensor of shape [B]")
        if torch.any((embodiment_id < 0) | (embodiment_id >= config.max_num_embodiments)):
            raise ValueError("embodiment_id is outside the checkpoint category table")
        if image_mask.shape != (batch, sequence) or attention_mask.shape != (batch, sequence):
            raise ValueError("Policy masks must have shape [B, sequence]")
        if image_mask.dtype != torch.bool or attention_mask.dtype != torch.bool:
            raise ValueError("Policy masks must be boolean, with True meaning visible")

    def _forward(
        self,
        backbone_features: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        image_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        initial_noise: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        config = self.config
        batch = backbone_features.shape[0]
        features = self.vl_self_attention(self.vlln(backbone_features))
        state_features = self.state_encoder(state.reshape(batch, 1, -1), embodiment_id)
        shape = (batch, config.action_horizon, config.max_action_dim)
        if initial_noise is None:
            actions = torch.randn(shape, device=features.device, dtype=features.dtype, generator=generator)
        else:
            if (
                initial_noise.shape != shape
                or initial_noise.device != features.device
                or initial_noise.dtype != features.dtype
            ):
                raise ValueError("initial_noise must match action shape, device and dtype")
            actions = initial_noise.clone()
        for step in range(config.num_inference_timesteps):
            timestep = torch.full(
                (batch,),
                step * config.num_timestep_buckets // config.num_inference_timesteps,
                device=features.device,
                dtype=torch.int64,
            )
            encoded = self.action_encoder(actions, timestep, embodiment_id)
            if config.add_pos_embed:
                positions = torch.arange(config.action_horizon, device=features.device)
                encoded = encoded + self.position_embedding(positions).unsqueeze(0)
            predicted = self.model(
                torch.cat((state_features, encoded), dim=1), features, timestep, image_mask, attention_mask
            )
            velocity = self.action_decoder(predicted, embodiment_id)[:, -config.action_horizon :]
            actions = actions + (1.0 / config.num_inference_timesteps) * velocity
        return actions
