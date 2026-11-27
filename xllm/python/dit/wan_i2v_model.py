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

from collections.abc import Sequence
from typing import Any

import torch
from diffusers import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanAttention, WanRotaryPosEmbed
from torch.nn import functional as F

from xllm.python.dit.parallel import (
    ParallelLinear,
    ParallelRMSNorm,
    SequenceParallel,
    StateDictShard,
    TensorParallel,
    load_component_weights,
)


def _apply_rotary(states: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = rotary
    rotated = torch.ops.npu.npu_rotary_mul(states.float(), cos, sin, rotary_mode="interleave")
    return rotated.to(states.dtype)


def _attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, live_rows: int | None = None
) -> torch.Tensor:
    query_rows, heads, head_dim = query.shape[1:]
    padding = 0
    if live_rows is not None and live_rows < query_rows:
        # Restore the padded rows after attention for the inverse SP exchange.
        padding = query_rows - live_rows
        query, key, value = (tensor[:, :live_rows] for tensor in (query, key, value))
    output = torch.ops.npu.npu_fusion_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        head_num=heads,
        input_layout="BNSD",
        scale=head_dim**-0.5,
        pre_tockens=65535,
        next_tockens=65535,
    )[0].transpose(1, 2)
    return F.pad(output, (0, 0, 0, 0, 0, padding)) if padding else output


class _WanAttentionProcessor:
    def __init__(self, sp: SequenceParallel) -> None:
        self._sp = sp
        self.live_rows = 0

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        image_states = None
        if encoder_hidden_states is None:
            context = hidden_states
        else:
            # Wan2.1 concatenates CLIP image tokens before 512 UMT5 tokens.
            image_states = encoder_hidden_states[:, :-512]
            context = encoder_hidden_states[:, -512:]
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(-1, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(context)).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(context).unflatten(-1, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary(query, rotary_emb)
            key = _apply_rotary(key, rotary_emb)
        if encoder_hidden_states is None:
            query = self._sp.exchange(query.contiguous(), True)
            key = self._sp.exchange(key.contiguous(), True)
            value = self._sp.exchange(value.contiguous(), True)
            output = _attention(query, key, value, self.live_rows)
            output = self._sp.exchange(output, False)
        else:
            output = _attention(query, key, value)
            image_key = attn.norm_added_k(attn.add_k_proj(image_states)).unflatten(-1, (attn.heads, -1))
            image_value = attn.add_v_proj(image_states).unflatten(-1, (attn.heads, -1))
            output = output + _attention(query, image_key, image_value)
        return attn.to_out[1](attn.to_out[0](output.flatten(-2)))


class WanI2VTransformer(WanTransformer3DModel):
    _sp: SequenceParallel

    def prepare_rotary(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The rotary frequencies depend only on the latent grid, which is fixed
        # across denoising steps and both CFG branches of a request.
        cos, sin = self.rope(latents)
        padding = (-cos.size(1)) % self._sp.size
        if padding:
            cos = F.pad(cos, (0, 0, 0, 0, 0, padding))
            sin = F.pad(sin, (0, 0, 0, 0, 0, padding))
        return (
            cos.chunk(self._sp.size, dim=1)[self._sp.rank].contiguous(),
            sin.chunk(self._sp.size, dim=1)[self._sp.rank].contiguous(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        *,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        batch, _, frames, height, width = hidden_states.shape
        pt, ph, pw = self.config.patch_size
        rotary = self.prepare_rotary(hidden_states) if rotary_emb is None else rotary_emb
        states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        live_rows = states.shape[1]
        padding = (-live_rows) % self._sp.size
        if padding:
            states = F.pad(states, (0, 0, 0, padding))
        states = states.chunk(self._sp.size, dim=1)[self._sp.rank].contiguous()
        temb, timestep_proj, text, image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        context = torch.cat((image, text), dim=1)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        for block in self.blocks:
            block.attn1.processor.live_rows = live_rows
            states = block(states, context, timestep_proj, rotary)

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        states = (self.norm_out(states.float()) * (1 + scale) + shift).to(states.dtype)
        states = self._sp.gather(self.proj_out(states).contiguous())[:, :live_rows]
        states = states.reshape(batch, frames // pt, height // ph, width // pw, pt, ph, pw, -1)
        output = states.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)


def load_transformer(
    path: str,
    shards: Sequence[StateDictShard],
    device: torch.device,
    tp: TensorParallel,
    sp: SequenceParallel,
) -> WanI2VTransformer:
    config = WanI2VTransformer.load_config(path)
    if (
        config.get("in_channels") != 36
        or config.get("out_channels") != 16
        or config.get("image_dim") != 1280
        or config.get("added_kv_proj_dim") != 5120
        or config.get("num_attention_heads") != 40
        or config.get("attention_head_dim") != 128
        or config.get("num_layers") != 40
        or config.get("text_dim") != 4096
        or config.get("pos_embed_seq_len") is not None
        or tuple(config.get("patch_size", ())) != (1, 2, 2)
    ):
        raise ValueError("Python Wan I2V requires the dense Wan2.1-I2V-14B checkpoint")
    if config["num_attention_heads"] % (tp.size * sp.size) or config["ffn_dim"] % tp.size:
        raise ValueError("Wan attention heads and feed-forward width must be divisible by the TP/SP topology")
    with torch.device("meta"):
        model = WanI2VTransformer.from_config(config)
    model._sp = sp
    for block in model.blocks:
        for attn in (block.attn1, block.attn2):
            for name in ("to_q", "to_k", "to_v"):
                setattr(attn, name, ParallelLinear(getattr(attn, name), tp))
            attn.to_out[0] = ParallelLinear(attn.to_out[0], tp, row=True)
            attn.norm_q = ParallelRMSNorm(attn.norm_q, tp)
            attn.norm_k = ParallelRMSNorm(attn.norm_k, tp)
            attn.heads //= tp.size
            attn.set_processor(_WanAttentionProcessor(sp))
        block.attn2.add_k_proj = ParallelLinear(block.attn2.add_k_proj, tp)
        block.attn2.add_v_proj = ParallelLinear(block.attn2.add_v_proj, tp)
        block.attn2.norm_added_k = ParallelRMSNorm(block.attn2.norm_added_k, tp)
        block.ffn.net[0].proj = ParallelLinear(block.ffn.net[0].proj, tp)
        block.ffn.net[2] = ParallelLinear(block.ffn.net[2], tp, row=True)

    def weight_dtype(name: str) -> torch.dtype:
        return torch.float32 if any(part in model._keep_in_fp32_modules for part in name.split(".")) else torch.bfloat16

    load_component_weights(model, shards, device, weight_dtype)
    # Non-persistent RoPE buffers are not checkpoint tensors; materialize them separately.
    model.rope = WanRotaryPosEmbed(
        config["attention_head_dim"], config["patch_size"], model.config.rope_max_seq_len
    ).to(device=device, dtype=torch.float32)
    return model
