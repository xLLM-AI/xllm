# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner


class CompileRunner(BaseRunner):
    def __init__(
        self,
        model,
        attention_backend,
        device,
        backend: str,
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        compile_backend = self._resolve_compile_backend(backend)
        self.compiled_model = torch.compile(
            model, backend=compile_backend, fullgraph=fullgraph, dynamic=dynamic,
        )

    @staticmethod
    def _resolve_compile_backend(backend: str):
        if backend == "torchair":
            import torchair
            config = torchair.CompilerConfig()
            return torchair.get_npu_backend(compiler_config=config)
        return backend

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> torch.Tensor:
        self.attention_backend.prepare(metadata)
        with forward_context(
            ForwardContext(
                self.attention_backend, self.device,
                layer_synchronizer=layer_synchronizer,
            )
        ):
            return self.compiled_model(input_ids, positions)
