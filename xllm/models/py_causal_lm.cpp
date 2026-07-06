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

#include "models/py_causal_lm.h"

#include <c10/util/Exception.h>
#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "models/py_model_bridge.h"

namespace py = pybind11;

namespace xllm {

namespace {

std::string dtype_to_string(const torch::TensorOptions& options) {
  switch (c10::typeMetaToScalarType(options.dtype())) {
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kFloat16:
      return "float16";
    case torch::kFloat32:
      return "float32";
    case torch::kFloat64:
      return "float64";
    default:
      return "bfloat16";
  }
}

// Writes reflected PROPERTY fields into a Python dict. pybind11 casts each
// value to its natural Python type (int/float/bool/str, list, set); a
// disengaged optional becomes None so the Python side can tell "unset" apart
// from a defaulted value.
class PyDictVisitor final : public PropertyVisitor {
 public:
  explicit PyDictVisitor(py::dict& dict) : dict_(dict) {}

  void visit(const std::string& name, bool value) override { set(name, value); }
  void visit(const std::string& name, int32_t value) override {
    set(name, value);
  }
  void visit(const std::string& name, int64_t value) override {
    set(name, value);
  }
  void visit(const std::string& name, float value) override {
    set(name, value);
  }
  void visit(const std::string& name, double value) override {
    set(name, value);
  }
  void visit(const std::string& name, const std::string& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int32_t>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<int64_t>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<float>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<double>& value) override {
    set(name, value);
  }
  void visit(const std::string& name, const std::vector<bool>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::vector<std::string>& value) override {
    set(name, value);
  }
  void visit(const std::string& name,
             const std::unordered_set<int32_t>& value) override {
    set(name, value);
  }
  void visit_absent(const std::string& name) override {
    dict_[py::str(name)] = py::none();
  }

 private:
  template <typename T>
  void set(const std::string& name, const T& value) {
    dict_[py::str(name)] = value;
  }

  py::dict& dict_;
};

}  // namespace

PyCausalLM::PyCausalLM(const ModelContext& context)
    : model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      device_(context.get_tensor_options().device()),
      enable_mla_(context.get_model_args().enable_mla()) {
  // Bring up the embedded interpreter + 'python' model package + xllm_ops lib.
  ensure_python_interpreter();

  // Tensor parallelism: read the TP group so per-rank head counts (and the
  // Python-side weight shards, driven by tp_rank/tp_size in the config dict)
  // match the native path. tp_group_ is null / world_size 1 on a single card,
  // degrading cleanly to the TP=1 layout.
  const ParallelArgs& parallel_args = context.get_parallel_args();
  tp_group_ = parallel_args.tp_group_;
  tp_size_ = (tp_group_ != nullptr) ? tp_group_->world_size() : 1;
  tp_rank_ = (tp_group_ != nullptr) ? tp_group_->rank() : 0;

  // Config-only attention module, shared across layers. Constructed here (after
  // the worker's flashinfer workspace init on this thread) with PER-RANK head
  // counts, replicating qwen2_attention.cpp:47-65 so the flashinfer plan
  // matches the q/k/v slices the Python graph produces on this rank.
  const int64_t total_num_heads = model_args_.n_heads();
  const int64_t total_num_kv_heads =
      model_args_.n_kv_heads().value_or(model_args_.n_heads());
  const int64_t head_dim = model_args_.head_dim();
  CHECK(total_num_heads % tp_size_ == 0)
      << "n_heads " << total_num_heads << " not divisible by tp_size "
      << tp_size_;
  const int64_t num_heads = total_num_heads / tp_size_;
  int64_t num_kv_heads;
  if (total_num_kv_heads >= tp_size_) {
    CHECK(total_num_kv_heads % tp_size_ == 0)
        << "n_kv_heads " << total_num_kv_heads << " not divisible by tp_size "
        << tp_size_;
    num_kv_heads = total_num_kv_heads / tp_size_;
  } else {
    CHECK(tp_size_ % total_num_kv_heads == 0)
        << "tp_size " << tp_size_ << " not divisible by n_kv_heads "
        << total_num_kv_heads;
    num_kv_heads = 1;
  }
  const float scale = std::sqrt(1.0f / static_cast<float>(head_dim));
  attn_ = layer::Attention(
      num_heads, head_dim, scale, num_kv_heads, model_args_.sliding_window());
  register_module("attn", attn_);

  py::gil_scoped_acquire gil;
  const std::string module_name = context.get_model_args().model_type().empty()
                                      ? std::string("Qwen3ForCausalLM")
                                      : context.get_model_args().model_type();

  // Resolve the Python model class via the registry, then instantiate it with
  // the config dict. Keep the object alive for the model's lifetime.
  py::module_ registry = py::module_::import("python.registry");
  py::object model_cls = registry.attr("get_model_class")(py::str(module_name));
  py_model_ = model_cls(build_config_dict(parallel_args));
  py_model_.attr("eval")();

  forward_batch_cls_ =
      py::module_::import("python.forward_batch").attr("ForwardBatch");
}

PyCausalLM::~PyCausalLM() {
  // Release Python references under the GIL to avoid refcount races.
  py::gil_scoped_acquire gil;
  py_model_ = py::object();
  forward_batch_cls_ = py::object();
}

py::dict PyCausalLM::build_config_dict(
    const ParallelArgs& parallel_args) const {
  py::dict d;
  PyDictVisitor visitor(d);

  // Full, already-parsed model config (renames, defaults and derived fields
  // are resolved in the native model-args loader -- the Python side is the
  // single source of truth's consumer, not a re-implementer).
  visit_properties(model_args_, visitor);
  // Parallel-dimension sizes (tp/dp/ep/cp/...). Non-plain-data fields are
  // skipped by the reflection layer.
  visit_properties(parallel_args, visitor);

  // Runtime overrides that are authoritative over the reflected config values:
  //  - dtype: the actual tensor dtype (may differ from model_args.dtype() when
  //    e.g. quantization forces bfloat16);
  //  - device: the worker's device string;
  //  - tp_size/tp_rank: the per-rank TP layout derived from the TP process
  //    group, which drives the Python-side weight sharding.
  d["dtype"] = dtype_to_string(options_);
  d["device"] = c10::str(device_);
  d["tp_size"] = tp_size_;
  d["tp_rank"] = tp_rank_;
  return d;
}

void PyCausalLM::load_model(std::unique_ptr<ModelLoader> loader) {
  py::gil_scoped_acquire gil;
  auto& state_dicts = loader->get_state_dicts();

  // Ensure the embedded module is imported so pybind11 type_info for
  // PyStateDict is registered before py::cast.
  py::module_::import("xllm_weight_loader");

  py::list py_state_dicts;
  for (const auto& sd : state_dicts) {
    py_state_dicts.append(py::cast(PyStateDict(sd.get()),
                                   py::return_value_policy::move));
  }

  py_model_.attr("load_weights")(
      py_state_dicts,
      static_cast<int32_t>(tp_rank_),
      static_cast<int32_t>(tp_size_));
}

ModelOutput PyCausalLM::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& parameters) {
  torch::NoGradGuard no_grad;

  // Paged-KV attention metadata for this forward (CUDA path mirrors the native
  // qwen3 model): reuse the executor-supplied metadata when present, otherwise
  // build our own. In CUDA graph mode the CudaGraphExecutorImpl pre-builds a
  // persistent, bucket-padded and flashinfer-planned AttentionMetadata into
  // parameters.attn_metadata whose buffers keep fixed device addresses across
  // capture and replay. It must be reused verbatim: rebuilding here would drop
  // the flashinfer plan_info and the graph-mode row padding, and the rebuild's
  // device work (flashinfer plan / q_cu_seq_lens.to(kCUDA)) is illegal while a
  // stream is capturing. On the eager path parameters.attn_metadata is null and
  // we build it as before.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      parameters.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(parameters, enable_mla_));
  }

  PyForwardContextGuard guard(
      attn_metadata.get(), &kv_caches, &attn_, tp_group_);

  py::gil_scoped_acquire gil;
  py::object batch = forward_batch_cls_(py::arg("positions") = positions,
                                        py::arg("native_attention") = false);
  py::object hidden_obj = py_model_.attr("forward")(tokens, positions, batch);
  auto hidden = hidden_obj.cast<torch::Tensor>();
  return ModelOutput(hidden);
}

torch::Tensor PyCausalLM::logits(const torch::Tensor& hidden_states,
                                 const torch::Tensor& seleted_idxes) {
  torch::NoGradGuard no_grad;

  // compute_logits runs lm_head, whose ColumnParallel(gather_output=True) does
  // an all_gather over the vocab dim. That collective reads the TP group from
  // the thread-local forward context via py_all_gather, so establish the
  // context here too (attention/KV state is unused while computing logits,
  // hence null). Without it py_all_gather degrades to identity and the logits
  // keep only this rank's vocab shard — the sampler then never sees the other
  // ranks' tokens (e.g. a tp=2 rank0 sees only vocab [0, vocab/2)).
  PyForwardContextGuard guard(/*attn_metadata=*/nullptr,
                              /*kv_caches=*/nullptr,
                              /*attn=*/nullptr,
                              tp_group_);
  py::gil_scoped_acquire gil;
  py::object selected = seleted_idxes.defined()
                            ? py::object(py::cast(seleted_idxes))
                            : py::object(py::none());
  py::object out = py_model_.attr("compute_logits")(hidden_states, selected);
  return out.cast<torch::Tensor>();
}

}  // namespace xllm
