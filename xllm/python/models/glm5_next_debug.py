# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Dev-only dump/trace hooks for GLM-5-Next, kept out of the serving model.

Imported lazily by ``glm5_next.py`` / ``glm5_next_vl.py`` only when
``GLM5_NEXT_DUMP_DIR`` is set, so the serving import path carries no debug
scaffolding. Activating the env var pulls this module in and registers the
capture hooks; without it, none of this code is imported.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.models.glm5_next import _capturing_acl_graph


def install_dump_hooks(model: nn.Module, lm_head: nn.Module, outdir: str) -> list:
    """Capture per-layer/submodule + final-norm tensors during the engine's
    forward and save them after every ``model.forward``: the first (prefill)
    call writes ``<outdir>/tensors.safetensors``, subsequent single-token
    decode calls write ``<outdir>/decode_<k>.safetensors``. Keys mirror
    ``dump_model.py`` so ``compare_dumps.py`` works across xllm-engine vs
    transformers-ref; decode dumps let multi-step state be compared
    layer-by-layer.

    The model forward returns the final hidden (post hc_head + norm) for ALL
    positions; that hidden (``final_norm``) plus per-layer outputs are the
    comparable tensors (logits are computed separately by compute_logits, only
    for the sampled position, so they are not dumped here).
    """
    from safetensors.torch import save_file

    store: dict = {}
    handles: list = []
    # Per-forward counter: forward 0 is prefill (written as tensors.safetensors
    # for the existing compare tooling); decode steps write decode_<k>.safetensors
    # so multi-step state can be compared layer-by-layer against the reference.
    step = [0]

    def _cap(name: str, idx: int = 0):
        def fn(_m, _i, o):
            if _capturing_acl_graph():
                return
            t = o[idx] if isinstance(o, (tuple, list)) else o
            if t is None:
                return
            if t.dtype in (torch.int32, torch.int64, torch.bool):
                store[name] = t.detach().cpu()
            else:
                store[name] = t.detach().to(torch.float32).cpu()

        return fn

    def _pre_cap(name: str):
        def fn(_m, inp):
            if _capturing_acl_graph():
                return
            t = inp[0] if isinstance(inp, (tuple, list)) else inp
            store[name] = t.detach().to(torch.float32).cpu()

        return fn

    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(_cap(f"layer{i}", 0)))
        p = f"L{i}."
        handles.append(layer.input_layernorm.register_forward_hook(_cap(p + "input_layernorm")))
        # mHC internals: layer input streams, the post-attn recombination (the
        # input to ffn_hc), and the learned post/comb weights — to localize the
        # first divergence between the attention site and the MLP site.
        handles.append(layer.attn_hc.register_forward_pre_hook(_pre_cap(p + "attn_hc.input")))
        handles.append(layer.ffn_hc.register_forward_pre_hook(_pre_cap(p + "ffn_hc.input")))
        handles.append(layer.attn_hc.register_forward_hook(_cap(p + "attn_hc.post", 0)))
        handles.append(layer.attn_hc.register_forward_hook(_cap(p + "attn_hc.comb", 1)))
        handles.append(layer.ffn_hc.register_forward_hook(_cap(p + "ffn_hc.post", 0)))
        handles.append(layer.ffn_hc.register_forward_hook(_cap(p + "ffn_hc.comb", 1)))
        handles.append(layer.attn_hc.register_forward_hook(_cap(p + "attn_hc.collapse", 2)))
        handles.append(layer.self_attn.register_forward_hook(_cap(p + "self_attn", 0)))
        sa = layer.self_attn
        if hasattr(sa, "o_norm"):
            # KDA internals: isolate where the first divergence enters the
            # attention (conv1d output, forget gate, gated-norm output).
            handles.append(sa.conv1d.register_forward_hook(_cap(p + "self_attn.conv1d")))
            handles.append(sa.forget_gate.register_forward_hook(_cap(p + "self_attn.forget_gate")))
            handles.append(sa.o_norm.register_forward_hook(_cap(p + "self_attn.o_norm")))
        if hasattr(layer.self_attn, "o_proj"):
            handles.append(layer.self_attn.o_proj.register_forward_hook(_cap(p + "self_attn.o_proj")))
        if getattr(layer.self_attn, "indexer", None) is not None:
            handles.append(layer.self_attn.indexer.register_forward_hook(_cap(p + "self_attn.indexer.topk")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(_cap(p + "post_attention_layernorm")))
        handles.append(layer.ffn_hc.register_forward_hook(_cap(p + "ffn_hc.collapse", 2)))
        handles.append(layer.mlp.register_forward_hook(_cap(p + "mlp")))
        if hasattr(layer.mlp, "shared_experts") and getattr(layer.mlp, "shared_experts", None) is not None:
            handles.append(layer.mlp.shared_experts.register_forward_hook(_cap(p + "mlp.shared_experts")))
        if hasattr(layer.mlp, "experts") and getattr(layer.mlp, "experts", None) is not None:
            handles.append(layer.mlp.experts.register_forward_hook(_cap(p + "mlp.experts")))

            def _moe_cur(_m, _i, _o, _l=layer, _p=p):
                cs = getattr(_l.mlp.experts, "_debug_current_sum", None)
                if cs is not None:
                    store[_p + "mlp.experts_current_sum"] = cs.to(torch.float32).cpu()

            handles.append(layer.mlp.experts.register_forward_hook(_moe_cur))

            # router topk weights/indices (stored by Glm5NextMoE.forward)
            def _moe_router(_m, _i, _o, _l=layer, _p=p):
                if hasattr(_l.mlp, "_last_topk_weights"):
                    store[_p + "mlp.topk_weights"] = _l.mlp._last_topk_weights.to(torch.float32).cpu()
                    store[_p + "mlp.topk_indices"] = _l.mlp._last_topk_indices.cpu()

            handles.append(layer.mlp.register_forward_hook(_moe_router))
    handles.append(model.norm.register_forward_hook(_cap("final_norm")))

    def _save(_m, _i, _o):
        if _capturing_acl_graph():
            # D2H dumps are illegal mid-capture; warmup forwards already
            # saved the same static-input values.
            return
        os.makedirs(outdir, exist_ok=True)
        _save_file = {k: v.contiguous().cpu() for k, v in store.items()}
        if step[0] == 0:
            fname = os.path.join(outdir, "tensors.safetensors")
        else:
            fname = os.path.join(outdir, f"decode_{step[0] - 1}.safetensors")
        save_file(_save_file, fname)
        logger.info(f"[glm5_next dump] saved {len(_save_file)} tensors to {fname}")
        store.clear()
        step[0] += 1

    handles.append(model.register_forward_hook(_save))
    return handles
