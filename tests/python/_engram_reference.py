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


from __future__ import annotations

import torch

from xllm.python.models.deepseek_v41_engram import EngramHashState, EngramLayout, compute_hash_multipliers


def _engram_test_token_map() -> list[int]:
    return (torch.arange(64) % 16).tolist()


def _engram_test_layout(layer_id: int = 1) -> EngramLayout:
    return EngramLayout(
        layer_ids=[layer_id],
        num_embeddings=[290],
        max_ngram_size=3,
        n_heads=2,
        head_dim=32,
        vocab_size=64,
        compressed_vocab_size=16,
        pad_token_id=0,
    )


def _tiny_engram_hash_state(token_map: list[int] | None = None) -> EngramHashState:
    layout = _engram_test_layout()
    multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, 16)
    return EngramHashState(
        layout,
        torch.tensor(_engram_test_token_map() if token_map is None else token_map, dtype=torch.int64),
        multipliers,
    )
