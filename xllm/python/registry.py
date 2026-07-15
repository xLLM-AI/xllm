# Copyright 2025-2026 The xLLM Authors.
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

"""Model registry: maps a HF architecture name to its Python model class.

Mirrors vLLM's ``_MODELS`` / ``EntryClass`` table. The C++ side resolves the
class by the model's architecture (or model_type) string.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

_REGISTRY: Dict[str, Callable[[], Type[nn.Module]]] = {}


def register_model(
    *names: str,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        for name in names:
            _REGISTRY[name] = cls
        return cls

    return deco


def get_model_class(name: str) -> Type[nn.Module]:
    if name not in _REGISTRY:
        raise KeyError(
            f"model '{name}' not registered; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def _register_builtin_models() -> None:
    # Imported lazily to avoid import cycles at module load.
    from xllm.python.models.qwen3 import Qwen3ForCausalLM

    register_model("Qwen3ForCausalLM", "qwen3")(Qwen3ForCausalLM)


_register_builtin_models()
