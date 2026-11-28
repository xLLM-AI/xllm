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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xllm.python.dit.pipelines.minimax_h3 import MiniMaxH3Pipeline
    from xllm.python.dit.pipelines.wan_i2v import WanI2VPipeline


def get_pipeline_class(model_type: str) -> type[MiniMaxH3Pipeline] | type[WanI2VPipeline]:
    if model_type == "MiniMaxH3ModularPipeline":
        from xllm.python.dit.pipelines.minimax_h3 import MiniMaxH3Pipeline

        return MiniMaxH3Pipeline
    if model_type == "WanImageToVideoPipeline":
        from xllm.python.dit.pipelines.wan_i2v import WanI2VPipeline

        return WanI2VPipeline
    raise ValueError(f"Unsupported Python DiT pipeline: {model_type}")
