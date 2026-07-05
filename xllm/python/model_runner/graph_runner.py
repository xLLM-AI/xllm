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

"""Graph runner: encapsulates CUDA-graph / torch.compile capture of the pure-GPU
model graph, keeping all graph-capture concerns out of the model definitions.

Mirrors SGLang's ``model_executor`` graph runners (which own graph
capture/replay). In xLLM the capture itself is delegated to torch.compile's
``cudagraphs`` / ``reduce-overhead`` backends (and, on the eager path, to the C++
piecewise cudagraph — ``PiecewiseGraphs`` in cuda_graph_executor_impl.cpp); this
runner owns the backend selection and the capture prerequisites (register_fake +
attention ``disallow_in_graph``), exposing a single callable over the model
graph. A model just constructs a ``GraphRunner(self.model)`` and calls it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

# Env switch selecting the torch.compile backend for the pure-GPU model graph
# (embed + decoder layers + final norm). Default ('off'/unset) leaves the model
# eager so the parity baseline is byte-identical to C++.
TC_BACKEND_ENV = "XLLM_TC_BACKEND"


def maybe_compile(model: nn.Module) -> nn.Module:
    """Optionally wrap the pure-GPU model graph in ``torch.compile``.

    Selected by ``XLLM_TC_BACKEND``. All backends share the prerequisites
    (``register_fake`` for xllm_ops + attention ``disallow_in_graph``), imported
    lazily below only when a backend is on so the eager path never registers
    them. attention graph-breaks each layer; the segments between attentions go
    to the backend.

    - ``off`` / unset       : no compile (eager parity baseline)
    - ``eager``             : ``backend="eager"`` — Dynamo trace only, no
                              cudagraph; verifies tracing + graph-break count
    - ``cudagraphs``        : ``backend="cudagraphs"`` — pure CUDA graph replay,
                              removes decode per-step host overhead
    - ``inductor``          : ``backend="inductor"`` — fuse vector segments
    - ``reduce-overhead``   : ``mode="reduce-overhead"`` — inductor +
                              cudagraph_trees
    """
    backend = os.environ.get(TC_BACKEND_ENV, "off").strip().lower()
    if backend in ("", "off", "none", "0"):
        return model

    # register_fake + attention disallow_in_graph — needed only once a compile
    # backend is on; importing here keeps the default eager path clean.
    from ..ops import fake_impls  # noqa: F401

    if backend == "reduce-overhead":
        return torch.compile(model, mode="reduce-overhead")
    if backend in ("eager", "cudagraphs", "inductor"):
        return torch.compile(model, backend=backend)
    raise ValueError(
        f"{TC_BACKEND_ENV}={backend!r} not recognized; expected one of "
        "off|eager|cudagraphs|inductor|reduce-overhead"
    )


class GraphRunner:
    """Owns the (optionally graph-captured) callable over a model's pure-GPU
    graph. Not an ``nn.Module`` — it wraps one — so a model can hold a
    ``GraphRunner`` without re-registering the compiled ``OptimizedModule`` as a
    duplicate submodule; the wrapped callable shares weights with the model.
    """

    def __init__(self, model: nn.Module):
        self._model = model
        self._runnable = maybe_compile(model)

    def __call__(self, *args, **kwargs):
        return self._runnable(*args, **kwargs)
