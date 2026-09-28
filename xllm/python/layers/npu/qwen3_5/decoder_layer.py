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

"""NPU-owned Qwen3.5 decoder layer."""

from __future__ import annotations

from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.moe import NpuQwen3_5SparseMoEBlock
from xllm.python.layers.qwen3_5.attention import Qwen3_5Attention
from xllm.python.layers.qwen3_5.decoder_layer import Qwen3_5DecoderLayer


class NpuQwen3_5DecoderLayer(Qwen3_5DecoderLayer):
    # NPU inherits the base ``Qwen3_5Attention``: row-parallel weights stay
    # unfinalized for TileLang/CANN coexistence via the base's no-op
    # ``_finish_loading``, unlike the CUDA backend which finalizes ``o_proj``.
    attention_cls = Qwen3_5Attention
    gated_delta_net_cls = NpuQwen3_5GatedDeltaNet
    sparse_moe_cls = NpuQwen3_5SparseMoEBlock

    def _prepare_forward(self) -> None:
        if self.layer_id == 0:
            # TODO: Remove this backend-local workaround once TileLang's dynamic
            # symbol cache can safely persist between service forwards.
            import tilelang

            tilelang.disable_cache()
            tilelang.cache.clear_cache()
