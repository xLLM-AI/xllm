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

import json
import logging
import os
import socket
from typing import Dict, Optional, Tuple, Union

import psutil
import xllm_export


def _resolve_kv_cache_memory_fraction(
    kv_cache_memory_fraction: float,
    max_memory_utilization: Optional[float],
) -> float:
    """Resolve the KV-cache memory budget fraction, honoring the deprecated
    ``max_memory_utilization`` alias.

    ``max_memory_utilization`` was renamed to ``kv_cache_memory_fraction`` and
    its meaning changed to "fraction of free GPU memory (after loading weights)
    used for the KV cache" (see ``kv_cache_config.cpp``). The old name is kept
    as a deprecated alias: when set it overrides ``kv_cache_memory_fraction`` and
    is reinterpreted under the new semantics, mirroring the C++ gflag alias
    (which the offline pybind path would otherwise bypass).
    """
    if max_memory_utilization is not None:
        logging.warning(
            "`max_memory_utilization` is deprecated; use "
            "`kv_cache_memory_fraction` instead. It now means the fraction of "
            "free GPU memory (after loading weights) used for the KV cache, and "
            "overrides `kv_cache_memory_fraction`."
        )
        return max_memory_utilization
    return kv_cache_memory_fraction


def terminate_process(pid: int, timeout: Union[int, float] = 30) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    procs = children + [parent]

    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        _, port = s.getsockname()
    return port


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_model_type(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(
            "config.json or model_index.json is required for backend detection"
        )
    data = _read_json(config_path)
    model_type = data.get("model_type") or data.get("model_name")
    if not model_type:
        raise ValueError("config.json must contain model_type or model_name")
    return str(model_type)


def _get_model_backend(model_type: str) -> str:
    get_backend = getattr(xllm_export, "get_model_backend", None)
    if not callable(get_backend):
        raise ValueError(
            "xllm_export.get_model_backend is not available. "
            "Please rebuild xllm_export or explicitly specify backend."
        )
    try:
        backend = get_backend(model_type)
    except Exception as exc:
        raise ValueError(
            f"Failed to resolve backend for model_type: {model_type}"
        ) from exc
    if not backend:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return backend


def _infer_model_type_and_backend(model_path: str) -> Tuple[Optional[str], str]:
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        data = _read_json(model_index_path)
        if "_diffusers_version" in data:
            return None, "dit"

    model_type = _infer_model_type(model_path)
    return model_type, _get_model_backend(model_type)


def _configure_cpp_chat_template(
    use_cpp_chat_template: bool,
    model_type: str,
) -> None:
    configure = getattr(xllm_export, "configure_cpp_chat_template", None)
    if not callable(configure):
        raise ValueError(
            "xllm_export.configure_cpp_chat_template is not available. "
            "Please rebuild xllm_export."
        )
    configure(use_cpp_chat_template, model_type)
