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

"""JoyaiASR speech-to-text model with a Conformer encoder and Qwen2 decoder."""

from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from xllm.python.layers import ColumnParallelLinear
from xllm.python.model_loader import ParallelLoadContext, ScopedWeightLoader, load_causal_lm_weights
from xllm.python.models.base import PyModelBase
from xllm.python.models.qwen2 import Qwen2Config, Qwen2Model

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)

        left_context = right_context = 3  # both exclude the current frame
        self.context = left_context + 1 + right_context

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.pad(x, (0, 0, 0, self.context - 1), "constant", 0.0)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x_mask = (positions < input_lengths.unsqueeze(1)).to(torch.uint8).unsqueeze(1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        n, c, t, d = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(n, t, c * d))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask.eq(0)


class RelPositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).float()
        position = torch.cat([torch.flip(position, [0]), -position[1:]]).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, position.size(0), d_model, requires_grad=False)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # Non-persistent: never loaded from the checkpoint (vLLM skip list).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read-only view; no per-encode [1, 2T, d_model] clone.
        tmax, t = self.pe.size(1), x.size(1)
        return self.pe[:, tmax // 2 - t + 1 : tmax // 2 + t].detach()


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("0", nn.LayerNorm(d_model)),
                    ("1", nn.Linear(d_model, d_model * 4)),
                    ("2", nn.SiLU()),
                    ("4", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


class RelPosMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int) -> None:
        super().__init__()
        if n_head <= 0 or d_model % n_head != 0:
            raise ValueError("Conformer d_model must be divisible by a positive n_head")
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * self.d_k, d_model, bias=False)

        self.scale = 1.0 / (self.d_k**0.5)
        self.linear_pos = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty([n_head, self.d_k]))
        self.pos_bias_v = nn.Parameter(torch.empty([n_head, self.d_k]))

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        n, h, t1, t2 = x.size()
        zero_pad = torch.zeros((n, h, t1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(n, h, t2 + 1, t1)
        x = x_padded[:, :, 1:].view_as(x)
        return x[:, :, :, : x.size(-1) // 2 + 1]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = q
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn = matrix_ac + matrix_bd
        attn.mul_(self.scale)

        if mask is not None:
            mask = mask.unsqueeze(1)
            # In-place fills are safe (executor runs under no_grad) and
            # save two [batch, head, T, T] temporaries per layer.
            attn.masked_fill_(mask, -float("inf"))
            attn = torch.softmax(attn, dim=-1)
            attn.masked_fill_(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        return self.fc(output) + residual


class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Conformer convolution kernel_size must be odd")
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 4, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask, 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)
        out = self.pointwise_conv2(out)
        if mask is not None:
            out.masked_fill_(mask, 0.0)
        out = out.transpose(1, 2)
        return out + residual


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, kernel_size: int = 33) -> None:
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model)
        self.conv = ConformerConvolution(d_model, kernel_size)
        self.ffn2 = ConformerFeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=pad_mask)
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        return self.layer_norm(out)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers_enc: int,
        n_head: int,
        d_model: int,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ) -> None:
        super().__init__()

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model, max_len=pe_maxlen)

        self.layer_stack = nn.ModuleList(
            [RelPosEmbConformerBlock(d_model, n_head, kernel_size) for _ in range(n_layers_enc)]
        )

    def forward(
        self,
        padded_input: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc_output, input_lengths, pad_mask = self.input_preprocessor(padded_input, input_lengths)
        pos_emb = self.positional_encoding(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, pad_mask=pad_mask)

        return enc_output, input_lengths


class JoyaiASRAdapter(nn.Module):
    """2-frame-concat projector: Linear(dim*2 -> llm) + ReLU + Linear."""

    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 2) -> None:
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.ds, feat_dim * self.ds)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


def _batch_ctc_greedy_search(
    ctc_logits: torch.Tensor, speech_lengths: torch.Tensor, blank_id: int = 0
) -> list[list[int]]:
    """Batch CTC greedy decode -> token ids per item (vLLM-parity fold:
    collapse consecutive duplicates and drop blanks)."""
    lengths = speech_lengths.tolist()
    results: list[list[int]] = []
    for i in range(ctc_logits.size(0)):
        length = max(1, min(int(lengths[i]), ctc_logits.size(1)))
        hyp = ctc_logits[i, :length].max(dim=-1).indices.tolist()
        results.append([key for key, _ in itertools.groupby(hyp) if key != blank_id])
    return results


class JoyaiASRForConditionalGeneration(PyModelBase):
    model: Qwen2Model
    lm_head: ColumnParallelLinear

    def __init__(self, config: dict) -> None:
        super().__init__()

        encoder_cfg = {
            "idim": int(config.get("mm_audio_idim", 80)),
            "d_model": int(config.get("mm_audio_d_model", 1280)),
            "n_layers": int(config.get("mm_audio_n_layers", 16)),
            "n_head": int(config.get("mm_audio_n_head", 20)),
            "kernel_size": int(config.get("mm_audio_kernel_size", 33)),
            "pe_maxlen": int(config.get("mm_audio_pe_maxlen", 5000)),
            "downsample_rate": int(config.get("mm_audio_downsample_rate", 2)),
        }
        self.text_cfg = Qwen2Config.from_dict(config)

        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "cuda"))
        self.dtype = dtype
        self.device = device

        self.ctc_enable = bool(config.get("use_ctc", False))

        self.encoder = ConformerEncoder(
            encoder_cfg["idim"],
            encoder_cfg["n_layers"],
            encoder_cfg["n_head"],
            encoder_cfg["d_model"],
            encoder_cfg["kernel_size"],
            encoder_cfg["pe_maxlen"],
        ).to(dtype=dtype, device=device)
        self.encoder_projector = JoyaiASRAdapter(
            encoder_cfg["d_model"],
            self.text_cfg.hidden_size,
            encoder_cfg["downsample_rate"],
        ).to(dtype=dtype, device=device)

        self.ctc_lo: nn.Linear | None = None

        tp = self.text_cfg.tp_size
        if tp <= 0 or self.text_cfg.vocab_size % tp != 0:
            raise ValueError("Audio LLM vocab_size must be divisible by a positive tp_size")
        self.model = Qwen2Model(self.text_cfg, dtype, device)
        self.lm_head = ColumnParallelLinear(
            self.text_cfg.hidden_size,
            self.text_cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

        # Per-audio encode cache keyed by content hash, byte-budgeted with
        # FIFO eviction, not CTC-gated (single-threaded executor). Raw tower
        # output + CTC candidates only; blocks assemble per request.
        self._encode_cache: dict[int, tuple[torch.Tensor, list[int]]] = {}
        self._encode_cache_bytes = 0
        self._encode_cache_budget_bytes = 0

    def configure_runtime(self, config: Mapping[str, object]) -> None:
        cache_size_mb = config.get("max_encoder_cache_size", 0)
        if not isinstance(cache_size_mb, int) or cache_size_mb < 0:
            raise ValueError("max_encoder_cache_size must be a nonnegative integer")
        self._encode_cache_budget_bytes = cache_size_mb * 1024 * 1024

    def _cache_encode_entry(self, hash_key: int, embeds: torch.Tensor, hyp: list[int]) -> None:
        entry_bytes = embeds.element_size() * embeds.numel()
        if entry_bytes > self._encode_cache_budget_bytes:
            return
        # Retained entries must not pin the whole padded batch buffer.
        embeds = embeds.clone()
        while self._encode_cache_bytes + entry_bytes > self._encode_cache_budget_bytes and self._encode_cache:
            ev = self._encode_cache.pop(next(iter(self._encode_cache)))
            self._encode_cache_bytes -= ev[0].element_size() * ev[0].numel()
        self._encode_cache[hash_key] = (embeds, hyp)
        self._encode_cache_bytes += entry_bytes

    def _ctc_block(self, hyp: list[int], pad_count: int) -> torch.Tensor:
        """[pad_count, hidden] CTC rows: embed_tokens(hyp) truncated to the
        reserved count, zero past its end (the pad id is also an EOS whose
        embedding leaks early-termination pressure); empty hyp -> zeros."""
        block = torch.zeros(
            pad_count,
            self.text_cfg.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        if hyp:
            take = hyp[:pad_count]
            ids = torch.tensor(take, dtype=torch.long, device=self.device)
            block[: len(take)] = self.model.embed_tokens(ids).to(self.dtype)
        return block

    def _assemble_blocks(
        self,
        entries: list[tuple[torch.Tensor, list[int]]],
        pad_counts: list[int],
    ) -> list[torch.Tensor]:
        """One [audio; ctc] block per entry in item order; row count equals
        the entry's mm_token_num, matching the executor's mask-prefix-sum
        slicing."""
        parts: list[torch.Tensor] = []
        for i, (embeds, hyp) in enumerate(entries):
            pad_count = pad_counts[i]
            if pad_count > 0:
                parts.append(torch.cat([embeds, self._ctc_block(hyp, pad_count)], dim=0))
            else:
                parts.append(embeds)
        return parts

    def encode(
        self,
        input_features: torch.Tensor,
        speech_lengths: torch.Tensor,
        audio_meta: torch.Tensor,
    ) -> list[torch.Tensor]:
        """audio_meta carries [hash, ctc_pad_num] for each audio.

        Cached audio rows and CTC hypotheses assemble with this call's pad counts.
        """
        if audio_meta.dim() != 2 or audio_meta.size(0) != speech_lengths.numel() or audio_meta.size(1) != 2:
            raise ValueError(
                f"audio_meta must be [num_audios, 2] (hash, ctc_pad_num), got shape {tuple(audio_meta.shape)}"
            )
        hash_keys = [int(h) for h in audio_meta[:, 0].tolist()]
        pad_counts = [int(n) for n in audio_meta[:, 1].tolist()]
        cached: list[tuple[torch.Tensor, list[int]] | None] = [self._encode_cache.get(h) for h in hash_keys]
        if all(entry is not None for entry in cached):
            return self._assemble_blocks([entry for entry in cached if entry is not None], pad_counts)

        lengths = speech_lengths.long().tolist()
        items = list(torch.split(input_features, lengths, dim=0))
        slot_by_hash: dict[int, int] = {}
        encode_indices: list[int] = []
        for i, entry in enumerate(cached):
            if entry is None and hash_keys[i] not in slot_by_hash:
                slot_by_hash[hash_keys[i]] = len(encode_indices)
                encode_indices.append(i)

        miss_items = [items[i] for i in encode_indices]
        miss_lengths = [lengths[i] for i in encode_indices]
        miss_batched = nn.utils.rnn.pad_sequence(miss_items, batch_first=True).to(dtype=self.dtype, device=self.device)
        miss_lens = torch.tensor(miss_lengths, dtype=torch.long, device=self.device)

        enc_output, enc_lengths = self.encoder(miss_batched, miss_lens)

        if self.ctc_lo is not None:
            with torch.inference_mode():
                ctc_logits = self.ctc_lo(enc_output)
            miss_ctc = _batch_ctc_greedy_search(ctc_logits, enc_lengths)
        else:
            miss_ctc = [[] for _ in encode_indices]

        speech_features, speech_lens = self.encoder_projector(enc_output, enc_lengths)
        encoded_entries = [
            (speech_features[slot, : int(n), :].to(self.dtype), miss_ctc[slot])
            for slot, n in enumerate(speech_lens.tolist())
        ]
        for hash_key, slot in slot_by_hash.items():
            entry = encoded_entries[slot]
            if entry[0].size(0) > 0:
                self._cache_encode_entry(hash_key, *entry)

        entries = [
            entry if entry is not None else encoded_entries[slot_by_hash[hash_keys[i]]]
            for i, entry in enumerate(cached)
        ]
        return self._assemble_blocks(entries, pad_counts)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        image_embeds: torch.Tensor | None = None,
        video_embeds: torch.Tensor | None = None,
        audio_embeds: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use the chunk mask; audio pad IDs can also occur as ordinary text."""
        inputs_embeds: torch.Tensor = self.model.embed_tokens(input_ids)

        if audio_embeds is None:
            return inputs_embeds

        if audio_mask is None:
            raise ValueError("audio_embeds requires the executor's chunk scatter mask")
        num_mask_rows = int(audio_mask.sum().item())
        if num_mask_rows == 0:
            # A chunk may contain only the text gap before CTC pads.
            return inputs_embeds
        if num_mask_rows != audio_embeds.size(0):
            raise ValueError(f"audio scatter mask rows ({num_mask_rows}) != audio embeds rows ({audio_embeds.size(0)})")
        inputs_embeds[audio_mask] = audio_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def load_weights(
        self,
        state_dicts: list[StateDict],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        encoder_weights = ScopedWeightLoader(state_dicts)
        self._load_encoder_weights(encoder_weights)
        if self.ctc_enable:
            self._load_ctc_weights(encoder_weights)
        ctx = ParallelLoadContext(tp_rank, tp_size)
        llm_weights = ScopedWeightLoader(state_dicts, src_prefixes=("llm.model.", "llm.", "model.", ""))
        load_causal_lm_weights(
            self.model,
            self.lm_head.weight,
            llm_weights,
            ctx,
            tie_word_embeddings=self.text_cfg.tie_word_embeddings,
        )

    def _load_encoder_weights(self, weights: ScopedWeightLoader) -> None:
        self.encoder.load_state_dict(
            {name: weights.get_tensor("encoder." + name) for name in self.encoder.state_dict()}
        )
        self.encoder_projector.load_state_dict(
            {name: weights.get_tensor("encoder_projector." + name) for name in self.encoder_projector.state_dict()}
        )

    def _load_ctc_weights(self, weights: ScopedWeightLoader) -> None:
        """CTC vocabulary size comes from checkpoint weights, not the LLM config."""
        try:
            weight = weights.get_tensor("ctc_lo.weight")
        except KeyError as error:
            raise KeyError(
                "--use_ctc is on but the checkpoint has no ctc_lo.weight "
                "(drop --use_ctc, or convert ctc_lo.pt into the safetensors "
                "shards first)"
            ) from error
        if self.ctc_lo is None:
            if weight.size(0) > self.text_cfg.vocab_size:
                raise ValueError(
                    f"ctc vocab ({weight.size(0)}) exceeds the llm vocab "
                    f"({self.text_cfg.vocab_size}): ctc hypotheses would not be "
                    "embeddable by embed_tokens"
                )
            self.ctc_lo = nn.Linear(
                weight.size(1),  # input: encoder d_model
                weight.size(0),  # output: CTC vocab (checkpoint-defined)
                bias=True,
            ).to(dtype=self.dtype, device=self.device)
        self.ctc_lo.load_state_dict({name: weights.get_tensor("ctc_lo." + name) for name in self.ctc_lo.state_dict()})
