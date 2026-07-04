/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/torch.h>

#include <vector>

#include "core/framework/kv_cache/kv_cache.h"

// Bridge between xLLM's C++ worker and a Python-interpreter-based model
// executor. It owns the embedded CPython lifecycle and the thread-local
// forward context that the ``torch.ops.xllm_ops.attention`` op reads to reach
// the C++ paged-KV attention (flashinfer) while the model graph is driven from
// Python. The Python side stays hardware-agnostic: it always calls the single
// ``xllm_ops.attention`` symbol with no ``#ifdef``.

namespace xllm {

namespace layer {
struct AttentionMetadata;
class Attention;
}  // namespace layer

// Tensor-parallel process group (not owned; from CollectiveCommunicator via
// ParallelArgs). Full definition lives in framework/parallel_state.
class ProcessGroup;

// Per-forward state consulted by the attention op. Set on the worker thread for
// the duration of a single PyCausalLM::forward (the embedded interpreter runs
// the model synchronously on that same thread), then cleared.
struct PyForwardContext {
  layer::AttentionMetadata* attn_metadata = nullptr;
  std::vector<KVCache>* kv_caches = nullptr;
  // Shared config-only attention module (num_heads/head_dim/scale/...); the
  // per-layer plan and KV cache are selected via layer_id at call time.
  layer::Attention* attn = nullptr;
  // TP group for the collective ops (all_reduce / all_gather). Null at
  // tp_size==1, in which case the collectives are identity.
  ProcessGroup* tp_group = nullptr;
};

// Returns the calling thread's forward context (thread-local).
PyForwardContext* get_py_forward_context();

// Scopes a forward context on the current thread; restores the previous value
// on destruction so nested/re-entrant forwards stay correct.
class PyForwardContextGuard {
 public:
  PyForwardContextGuard(layer::AttentionMetadata* attn_metadata,
                        std::vector<KVCache>* kv_caches,
                        layer::Attention* attn,
                        ProcessGroup* tp_group);
  ~PyForwardContextGuard();

  PyForwardContextGuard(const PyForwardContextGuard&) = delete;
  PyForwardContextGuard& operator=(const PyForwardContextGuard&) = delete;

 private:
  PyForwardContext prev_;
};

// Initializes the embedded CPython interpreter (idempotent, process-wide),
// makes the ``xllm_models`` package importable, and force-links the xllm_ops
// torch library. Safe to call from any worker thread; the GIL is released
// afterwards so subsequent calls use ``py::gil_scoped_acquire``.
void ensure_python_interpreter();

}  // namespace xllm
