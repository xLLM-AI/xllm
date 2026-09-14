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

"""ViT Executor: lightweight executor for vision transformer models.

Unlike ModelExecutor which is designed for LLM models with attention backends
and KV cache management, ViTExecutor is a simple executor specifically for
vision transformer models that:
- Do not require attention backend infrastructure
- Do not manage KV caches
- Support eager/torchair/inductor backends via torch.compile
- Have simple forward signatures: forward(pixel_values, grid_thw)

This executor is used by Qwen3VLForConditionalGeneration.encode() to support
graph compilation (torchair/inductor) for the vision tower.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class ViTExecutor:
    """Lightweight executor for vision transformer models.

    Supports three execution modes:
    - eager: Direct model invocation (no compilation)
    - torchair: torch.compile with torchair backend (NPU graph compilation)
    - inductor: torch.compile with inductor backend (general graph compilation)

    Args:
        model: The vision transformer model (e.g., Qwen3VLVisionTransformer)
        backend: Execution backend, one of "eager", "torchair", "inductor"
        compile_kwargs: Additional kwargs passed to torch.compile (e.g., fullgraph, dynamic)
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = "eager",
        compile_kwargs: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.backend = backend.lower()
        self._compiled_model = None

        if self.backend not in ("eager", "torchair", "inductor"):
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Must be one of: eager, torchair, inductor"
            )

        if self.backend != "eager":
            self._compile_model(compile_kwargs or {})

    def _compile_model(self, compile_kwargs: dict) -> None:
        """Compile the model with the specified backend."""
        # Clean up environment variables to ensure a fresh state
        os.environ.pop("AUTOFUSE_FLAGS", None)
        os.environ.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        os.environ.pop("MAX_RUNTIME_CORE_NUMBER", None)
        if self.backend == "torchair":
            # Set torchair-specific environment variables
            os.environ["AUTOFUSE_FLAGS"] = (
                "--enable_autofuse=true;--autofuse_enable_pass=reduce,concat,transpose,gather,split,slice"
            )
            os.environ["MAX_RUNTIME_CORE_NUMBER"] = "3"
            import torchair
            from torchair import get_npu_backend
            print("compile model with torchair backend")
            compiler_config = torchair.CompilerConfig()
          #  compiler_config.ge_config.multistream_parallel_mode="cv"
            backend = get_npu_backend(compiler_config=compiler_config)
        elif self.backend == "inductor":
            # Set inductor-specific environment variables
            os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "ascendc"
            backend = "inductor"
        else:
            backend = self.backend

        # Merge default compile kwargs with user-provided ones
        # dynamic=False (static graph) by default
        default_kwargs = {"backend": backend, "dynamic": False}
        default_kwargs.update(compile_kwargs)

        self._compiled_model = torch.compile(self.model, **default_kwargs)

    @torch.inference_mode()
    def execute(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the vision transformer model.

        Args:
            pixel_values: Flattened patches, shape ``(total_patches, C*t*p*p)``
            grid_thw: Temporal/height/width grid, shape ``(num_images, 3)``

        Returns:
            Image embeddings, shape ``(total_image_tokens, out_hidden_size * (1 + num_deepstacks))``
        """
        if self._compiled_model is not None:
            return self._compiled_model(pixel_values, grid_thw)
        return self.model(pixel_values, grid_thw)

    def __repr__(self) -> str:
        return (
            f"ViTExecutor(backend={self.backend}, "
            f"model={self.model.__class__.__name__})"
        )
