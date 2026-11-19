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

"""NPU-only public entry. Does not compute, cache outputs or JIT-build on a call."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any

_EXTENSION = None


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        spec = importlib.machinery.PathFinder.find_spec("_t5_qkv", [str(Path(__file__).resolve().parent)])
        if spec is None or spec.loader is None:
            raise RuntimeError(
                "NPU extension is not built. Run bash scripts/build.sh in the specified CANN environment."
            )
        _EXTENSION = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_EXTENSION)
    return _EXTENSION


def run(inputs: Any, attributes: Any) -> Any:
    return _extension().execute(inputs, attributes, False)


def run_direct(inputs: Any, attributes: Any) -> Any:
    return _extension().execute(inputs, attributes, True)


class Prepared:
    """Metadata-only plan; each call supplies inputs and creates fresh outputs.

    Current stream is resolved on every call, not at prepare time.
    """

    __slots__ = ("_attributes", "_shape", "_dtype", "_device")

    def __init__(self, inputs: Any, attributes: Any, shape: Any) -> None:
        self._attributes = dict(attributes)
        self._shape = tuple(shape)
        self._dtype, self._device = inputs[0].dtype, inputs[0].device

    def _check(self, inputs: Any) -> Any:
        shape = _extension().validate(inputs, self._attributes)
        if tuple(shape) != self._shape or inputs[0].dtype != self._dtype or inputs[0].device != self._device:
            raise ValueError("prepared metadata does not match inputs")

    def __call__(self, inputs: Any) -> Any:
        self._check(inputs)
        return run(inputs, self._attributes)

    def run(self, inputs: Any) -> Any:
        return self(inputs)

    def run_direct(self, inputs: Any) -> Any:
        self._check(inputs)
        return run_direct(inputs, self._attributes)


def prepare(inputs: Any, attributes: Any) -> Any:
    shape = _extension().validate(inputs, attributes)
    return Prepared(inputs, attributes, shape)
