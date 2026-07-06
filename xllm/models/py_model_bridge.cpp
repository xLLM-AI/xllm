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

#include "models/py_model_bridge.h"

#include <Python.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <tuple>

#include "core/framework/parallel_state/parallel_state.h"
#include "core/kernels/cuda/xllm_ops_library.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/cuda/attention.h"

DEFINE_string(python_model_path,
              "",
              "Filesystem directory that contains the 'python' model package "
              "(xLLM's Python model executor), prepended to sys.path for the "
              "embedded interpreter. Falls back to the XLLM_PYTHON_MODEL_PATH "
              "env var when empty.");

namespace py = pybind11;

namespace xllm {

namespace {

thread_local PyForwardContext t_forward_context;

// torch.ops.xllm_ops.attention(q, k, v, layer_id) -> Tensor
//
// Called from the Python model graph. Selects the current layer's flashinfer
// plan + KV cache from the thread-local forward context and delegates to the
// same C++ layer::Attention the native model uses, so attention numerics are
// identical to the C++ path.
torch::Tensor py_attention(const torch::Tensor& q,
                           const torch::Tensor& k,
                           const torch::Tensor& v,
                           int64_t layer_id) {
  PyForwardContext* ctx = get_py_forward_context();
  CHECK(ctx != nullptr && ctx->attn_metadata != nullptr &&
        ctx->kv_caches != nullptr && ctx->attn != nullptr)
      << "xllm_ops.attention called outside a PyCausalLM forward context";
  CHECK(layer_id >= 0 &&
        layer_id < static_cast<int64_t>(ctx->kv_caches->size()))
      << "xllm_ops.attention layer_id " << layer_id << " out of range [0, "
      << ctx->kv_caches->size() << ")";

  auto& meta = *ctx->attn_metadata;
#if defined(USE_CUDA) || defined(USE_MUSA)
  // Mirror the native model loop: the flashinfer plan is computed once at
  // layer 0 and reused, keyed by plan_info->layer_id.
  if (meta.plan_info != nullptr) {
    meta.plan_info->layer_id = static_cast<int32_t>(layer_id);
  }
#endif

  // layer::Attention::forward takes non-const Tensor& (it may view/reshape and
  // writes k/v into the paged cache); copy the cheap tensor handles.
  torch::Tensor query = q;
  torch::Tensor key = k;
  torch::Tensor value = v;
  auto& kv_cache = (*ctx->kv_caches)[layer_id];
  auto result = (*ctx->attn)->forward(meta, query, key, value, kv_cache);
  return std::get<0>(result);
}

// torch.ops.xllm_ops.all_reduce(x) -> Tensor
//
// SUM all-reduce across the tensor-parallel group (RowParallel output combine).
// Identity when there is no TP group (single card). Runs on the tp_group
// carried by the thread-local forward context.
//
// OUT-OF-PLACE (functional): clones the input and reduces the clone, leaving
// the input untouched. The op schema is declared functional (``-> Tensor`` with
// no
// ``(a!)`` alias), so the impl must not mutate its input — otherwise
// torch.compile (which trusts the schema when the op is traced via its
// register_fake) could miscompile under cudagraph static-buffer reuse. This
// mirrors vLLM's
// ``_all_reduce_out_place`` / SGLang's ``outplace_all_reduce`` (both allocate a
// fresh output so the collective is safe to trace + capture in a CUDA graph).
torch::Tensor py_all_reduce(const torch::Tensor& input) {
  PyForwardContext* ctx = get_py_forward_context();
  if (ctx == nullptr || ctx->tp_group == nullptr) {
    return input;
  }
  torch::Tensor out = input.clone();
  return parallel_state::reduce(out, ctx->tp_group);
}

// torch.ops.xllm_ops.all_gather(x, dim, world_size) -> Tensor
//
// All-gather across the tensor-parallel group and concatenate along `dim` in
// rank order (ColumnParallel / embedding output combine). Identity without a TP
// group. Out-of-place (``parallel_state::gather`` allocates the gathered
// tensor).
// ``world_size`` is passed by the caller (== tp_size) purely so the Python
// register_fake can compute the gathered shape (``size(dim) * world_size``) at
// trace time; the live tp_group is authoritative at runtime (mirrors vLLM's
// ``all_gather(tensor, dim, world_size, group_name)``).
torch::Tensor py_all_gather(const torch::Tensor& input,
                            int64_t dim,
                            int64_t world_size) {
  PyForwardContext* ctx = get_py_forward_context();
  if (ctx == nullptr || ctx->tp_group == nullptr) {
    return input;
  }
  CHECK(ctx->tp_group->world_size() == world_size)
      << "all_gather world_size arg (" << world_size
      << ") != tp group world_size (" << ctx->tp_group->world_size() << ")";
  return parallel_state::gather(
      input, ctx->tp_group, static_cast<int32_t>(dim));
}

void prepend_sys_path(const std::string& dir) {
  if (dir.empty()) {
    return;
  }
  py::module_ sys = py::module_::import("sys");
  py::list path = py::reinterpret_borrow<py::list>(sys.attr("path"));
  // Skip if already present.
  for (auto item : path) {
    if (py::isinstance<py::str>(item) && item.cast<std::string>() == dir) {
      return;
    }
  }
  path.attr("insert")(0, py::str(dir));
}

}  // namespace

// Register the attention op as a fragment of the xllm_ops library defined in
// core/kernels/cuda/xllm_ops_library.cpp. Placing it in this TU (alongside
// ensure_python_interpreter, which PyCausalLM references) guarantees the static
// initializer is linked into the binary.
TORCH_LIBRARY_FRAGMENT(xllm_ops, m) {
  m.def("attention(Tensor q, Tensor k, Tensor v, int layer_id) -> Tensor");
  m.def("all_reduce(Tensor x) -> Tensor");
  m.def("all_gather(Tensor x, int dim, int world_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(xllm_ops, CUDA, m) {
  m.impl("attention", TORCH_FN(py_attention));
  m.impl("all_reduce", TORCH_FN(py_all_reduce));
  m.impl("all_gather", TORCH_FN(py_all_gather));
}

PyForwardContext* get_py_forward_context() { return &t_forward_context; }

PyForwardContextGuard::PyForwardContextGuard(
    layer::AttentionMetadata* attn_metadata,
    std::vector<KVCache>* kv_caches,
    layer::Attention* attn,
    ProcessGroup* tp_group)
    : prev_(t_forward_context) {
  t_forward_context.attn_metadata = attn_metadata;
  t_forward_context.kv_caches = kv_caches;
  t_forward_context.attn = attn;
  t_forward_context.tp_group = tp_group;
}

PyForwardContextGuard::~PyForwardContextGuard() { t_forward_context = prev_; }

void ensure_python_interpreter() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    // Force-link the xllm_ops CUDA kernel registrations (rms_norm,
    // silu_and_mul,
    // ...) so the embedded interpreter sees them via torch.ops.xllm_ops.
    ensure_xllm_ops_registered();

    const bool we_initialized = !Py_IsInitialized();
    if (we_initialized) {
      // Do not install signal handlers: the C++ server owns them.
      py::initialize_interpreter(/*init_signal_handlers=*/false);
    }

    {
      py::gil_scoped_acquire gil;
      std::string model_path = FLAGS_python_model_path;
      if (model_path.empty()) {
        const char* env = std::getenv("XLLM_PYTHON_MODEL_PATH");
        if (env != nullptr) {
          model_path = env;
        }
      }
      prepend_sys_path(model_path);
      // Validate the package is importable now, failing loudly at startup
      // rather than deep inside the first forward.
      try {
        py::module_::import("python");
      } catch (const py::error_already_set& e) {
        LOG(FATAL) << "Failed to import the 'python' model package for the "
                      "Python model executor. Set --python_model_path (or "
                      "XLLM_PYTHON_MODEL_PATH) to the directory containing the "
                      "'python' package. Error: "
                   << e.what();
      }
    }

    // Release the GIL taken by initialize_interpreter so worker threads can
    // reacquire it via py::gil_scoped_acquire. Only needed when we owned the
    // initialization on this thread.
    if (we_initialized) {
      PyEval_SaveThread();
    }
  });
}

}  // namespace xllm
