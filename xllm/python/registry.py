# Copyright 2025-2026 The xLLM Authors.
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

"""Model registry: maps a HF architecture name to its Python model class.

Mirrors vLLM's ``_MODELS`` / ``EntryClass`` table. The C++ side resolves the
class by the model's architecture (or model_type) string.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Type

import torch.nn as nn

_ModelPath = tuple[str, str]
_REGISTRY: Dict[str, _ModelPath] = {}


def register_model(
    *names: str,
) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Register a model class for callers that already imported its module."""

    def deco(cls: Type[nn.Module]) -> Type[nn.Module]:
        path = (cls.__module__, cls.__name__)
        for name in names:
            _REGISTRY[name] = path
        return cls

    return deco


def _register_model_path(module_name: str, class_name: str, *names: str) -> None:
    path = (module_name, class_name)
    for name in names:
        _REGISTRY[name] = path


def get_model_class(name: str) -> Type[nn.Module]:
    if name not in _REGISTRY:
        raise KeyError(
            f"model '{name}' not registered; available: {sorted(_REGISTRY)}"
        )
    module_name, class_name = _REGISTRY[name]
    model_cls = getattr(import_module(module_name), class_name)
    return model_cls


def _register_builtin_models() -> None:
    _register_model_path(
        "xllm.python.models.qwen3",
        "Qwen3ForCausalLM",
        "Qwen3ForCausalLM",
        "qwen3",
    )
    _register_model_path(
        "xllm.python.models.qwen3_5",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "qwen3_5",
        "qwen3_5_text",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    )
    _register_model_path(
        "xllm.python.models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
        "qwen3_vl",
    )
    _register_model_path(
        "xllm.python.models.deepseek_v32",
        "DeepseekV3ForCausalLM",
        "deepseek_v32",
    )
    _register_model_path(
        "xllm.python.models.glm5_2",
        "Glm52ForCausalLM",
        "glm_moe_dsa",
    )

    _register_model_path(
        "xllm.python.models.deepseek_v32_mtp",
        "DeepseekV32MtpForCausalLM",
        "deepseek_v32_mtp",
    )


_register_builtin_models()
