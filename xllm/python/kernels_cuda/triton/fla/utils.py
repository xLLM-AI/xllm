# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
import triton

FLA_CHUNK_SIZE = 64


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    cache_entries: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for index, entry in enumerate(cache_entries):
            previous_args, previous_kwargs, result = entry
            if (
                len(args) == len(previous_args)
                and len(kwargs) == len(previous_kwargs)
                and all(
                    value is previous
                    for value, previous in zip(args, previous_args)
                )
                and all(
                    key in previous_kwargs and value is previous_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:index]
                    + cache_entries[index + 1 :]
                    + [(args, kwargs, result)]
                )
                return result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        contiguous_args = tuple(
            value.contiguous() if isinstance(value, torch.Tensor) else value
            for value in args
        )
        contiguous_kwargs = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        tensor = next(
            (
                value
                for value in (*args, *kwargs.values())
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        context = (
            torch.cuda.device(tensor.device)
            if tensor is not None and tensor.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def _is_nvidia() -> bool:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except (RuntimeError, AttributeError):
        return False


is_nvidia_hopper = (
    _is_nvidia()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = False
is_gather_supported = hasattr(triton.language, "gather")
is_amd = False
is_tma_supported = False


class _SharedMemory(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def for_architecture(cls, architecture: str) -> int:
        try:
            return cls[architecture.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(architecture: str = "none", tensor_idx: int = 0) -> bool:
    try:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            tensor_idx
        )
        return properties["max_shared_mem"] >= _SharedMemory.for_architecture(
            architecture
        )
    except (RuntimeError, AttributeError, KeyError):
        return False
