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

"""Internal real paged-cache and node capture helpers for OLMo2 validation."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch

_BLOCK = 128


class _Runner:
    def __init__(self, model: Any, blocks: int = 40) -> None:
        from xllm.python.attention.backend import LayerCache
        from xllm.python.attention.flashinfer import FlashInferBackend

        self.model = model
        self.device = model.device
        attn = model.model.layers[0].self_attn.attn
        self.backend = FlashInferBackend(
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.scale,
            0,
            self.device,
            model.dtype,
            use_tensor_core_decode=attn.use_tensor_core_decode,
        )
        shape = (blocks, _BLOCK, attn.num_kv_heads, attn.head_dim)
        self.caches = [
            LayerCache(
                key=torch.empty(shape, dtype=model.dtype, device=self.device),
                value=torch.empty(shape, dtype=model.dtype, device=self.device),
            )
            for _ in model.model.layers
        ]
        self.backend.bind_kv_caches(self.caches)
        self.blocks = blocks
        self.last_metadata: dict[str, Any] = {}

    def pages(self, capacities: list[int]) -> list[list[int]]:
        counts = [math.ceil(n / _BLOCK) for n in capacities]
        assert sum(counts) <= self.blocks
        # Nonmonotonic physical page IDs prevent accidental logical-index use.
        pool = list(reversed(range(self.blocks)))
        result, cursor = [], 0
        for count in counts:
            result.append(pool[cursor : cursor + count])
            cursor += count
        return result

    def run(
        self,
        queries: list[list[int]],
        past_lengths: list[int],
        pages: list[list[int]],
        mode: str,
        last_only: bool = False,
        positions: list[int] | None = None,
    ) -> torch.Tensor:
        from xllm.python.model_executor.forward_context import ForwardContext, forward_context

        assert mode in ("prefill", "chunked", "decode")
        assert len(queries) == len(past_lengths) == len(pages)
        lengths = [len(q) for q in queries]
        totals = [p + n for p, n in zip(past_lengths, lengths, strict=True)]
        assert all(0 < n <= self.model.cfg.max_position_embeddings for n in totals)
        if mode == "prefill":
            assert not any(past_lengths)
        if mode == "decode":
            assert all(n == 1 for n in lengths)
        slots, pos, indices, indptr = [], [], [], [0]
        qo = [0]
        for past, query, total, page_ids in zip(past_lengths, queries, totals, pages, strict=True):
            slots.extend(page_ids[i // _BLOCK] * _BLOCK + i % _BLOCK for i in range(past, total))
            pos.extend(range(past, total))
            indices.extend(page_ids[: math.ceil(total / _BLOCK)])
            indptr.append(len(indices))
            qo.append(qo[-1] + len(query))

        def _t(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        metadata = SimpleNamespace(
            is_prefill=mode == "prefill",
            is_chunked_prefill=mode == "chunked",
            slot_mapping=_t(slots),
            q_cu_seq_lens=_t(qo),
            kv_cu_seq_lens=_t(qo),
            qo_indptr=_t(qo),
            paged_kv_indptr=_t(indptr),
            paged_kv_indices=_t(indices),
            paged_kv_last_page_len=_t([(n - 1) % _BLOCK + 1 for n in totals]),
            prepared_attention_state=None,
        )
        self.backend.prepare(metadata)
        context = ForwardContext(self.backend, self.device, metadata, self.caches)
        tokens = torch.tensor([i for q in queries for i in q], dtype=torch.long, device=self.device)
        pos_tensor = torch.tensor(pos if positions is None else positions, dtype=torch.long, device=self.device)
        with torch.inference_mode(), forward_context(context):
            hidden = self.model.model(tokens, pos_tensor)
            selected = (
                torch.tensor([n - 1 for n in qo[1:]], dtype=torch.long, device=self.device) if last_only else None
            )
            logits = self.model.compute_logits(hidden, selected)
        self.last_metadata = {
            "mode": mode,
            "query_lengths": lengths,
            "past_lengths": past_lengths,
            "total_lengths": totals,
            "physical_pages": pages,
            "slot_mapping": slots,
            "positions": pos_tensor.cpu().tolist(),
            "paged_kv_indptr": indptr,
            "paged_kv_indices": indices,
            "block_size": _BLOCK,
        }
        return logits


class _Capture:
    def __init__(self, model: Any, kind: str, runner: _Runner | None = None) -> None:
        self.nodes: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._kind = kind
        self._layer = 0
        self._original_rotate: Any = None
        self._original_execute: Any = None
        self._runner = runner
        self._hook(model.model.embed_tokens, "embedding")
        self._hook(model.model.norm, "final_norm")
        for i, layer in enumerate(model.model.layers):
            p = f"layer{i:02d}."
            self._hook(layer, p + "input", before=True)
            self._hook(layer, p + "residual_mlp")
            self._hook(layer.mlp, p + "residual_attention", before=True)
            for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
                self._hook(getattr(layer, name), p + name)
            self._hook(layer.self_attn.o_proj, p + "attention", before=True)
            self._hook(layer.self_attn.o_proj, p + "o_proj")
            self._hook(layer.mlp.down_proj, p + "silu_mul", before=True)
            self._hook(layer.mlp.down_proj, p + "down_proj")
            self._hook(layer.self_attn.q_norm, p + "q_norm")
            self._hook(layer.self_attn.k_norm, p + "k_norm")
            if kind == "hf":
                for name in ("q", "k", "v"):
                    self._hook(getattr(layer.self_attn, name + "_proj"), p + name + "_proj")
                for name in ("gate", "up"):
                    self._hook(getattr(layer.mlp, name + "_proj"), p + name + "_proj")

                def _layer(module: Any, args: Any, index: int = i) -> None:
                    self._layer = index

                self._handles.append(layer.self_attn.register_forward_pre_hook(_layer))
            else:
                self._fused(layer.self_attn.qkv_proj, p, ("q", "k", "v"))
                self._fused(layer.mlp.gate_up_proj, p, ("gate", "up"))
        if kind == "hf":
            import transformers.models.olmo2.modeling_olmo2 as hf

            self._original_rotate = hf.apply_rotary_pos_emb

            def _rotate(*args: Any, **kwargs: Any) -> Any:
                q, k = self._original_rotate(*args, **kwargs)
                for name, value in (("q", q), ("k", k)):
                    self._save(f"layer{self._layer:02d}.{name}_rope", value.transpose(1, 2).flatten(2))
                return q, k

            hf.apply_rotary_pos_emb = _rotate
        else:
            assert runner is not None
            self._original_execute = runner.backend.execute

            def _execute(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Any) -> torch.Tensor:
                self._save(f"layer{layer.layer_id:02d}.q_rope", q)
                self._save(f"layer{layer.layer_id:02d}.k_rope", k)
                return self._original_execute(q, k, v, layer)

            runner.backend.execute = _execute

    def _save(self, name: str, value: Any) -> None:
        if isinstance(value, tuple):
            value = value[0]
        self.nodes[name] = value.detach().reshape(-1, value.shape[-1]).float().cpu()

    def _hook(self, module: Any, name: str, before: bool = False) -> None:
        if before:

            def _before(mod: Any, args: Any) -> None:
                self._save(name, args[0])

            handle = module.register_forward_pre_hook(_before)
        else:

            def _after(mod: Any, args: Any, output: Any) -> None:
                self._save(name, output)

            handle = module.register_forward_hook(_after)
        self._handles.append(handle)

    def _fused(self, module: Any, prefix: str, names: tuple[str, ...]) -> None:
        def _after(mod: Any, args: Any, output: torch.Tensor) -> None:
            for name, value in zip(names, output.chunk(len(names), dim=-1), strict=True):
                self._save(prefix + name + "_proj", value)

        self._handles.append(module.register_forward_hook(_after))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        if self._kind == "hf":
            import transformers.models.olmo2.modeling_olmo2 as hf

            hf.apply_rotary_pos_emb = self._original_rotate
        elif self._runner is not None:
            self._runner.backend.execute = self._original_execute
