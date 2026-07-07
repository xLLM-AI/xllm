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

#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
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
  ensure_python_interpreter();

  const ParallelArgs& parallel_args = context.get_parallel_args();
  tp_group_ = parallel_args.tp_group_;
  tp_size_ = (tp_group_ != nullptr) ? tp_group_->world_size() : 1;
  tp_rank_ = (tp_group_ != nullptr) ? tp_group_->rank() : 0;

  py::gil_scoped_acquire gil;
  const std::string module_name = context.get_model_args().model_type().empty()
                                      ? std::string("Qwen3ForCausalLM")
                                      : context.get_model_args().model_type();

  py::module_ registry = py::module_::import("python.registry");
  py::object model_cls = registry.attr("get_model_class")(py::str(module_name));
  py_model_ = model_cls(build_config_dict(parallel_args));
  py_model_.attr("eval")();
}

PyCausalLM::~PyCausalLM() {
  py::gil_scoped_acquire gil;
  py_model_ = py::object();
}

py::dict PyCausalLM::build_config_dict(
    const ParallelArgs& parallel_args) const {
  py::dict d;
  PyDictVisitor visitor(d);
  visit_properties(model_args_, visitor);
  visit_properties(parallel_args, visitor);
  d["dtype"] = dtype_to_string(options_);
  d["device"] = c10::str(device_);
  d["tp_size"] = tp_size_;
  d["tp_rank"] = tp_rank_;
  d["python_graph_backend"] =
      ExecutionConfig::get_instance().python_graph_backend();
  d["python_graph_max_batch"] =
      ExecutionConfig::get_instance().python_graph_max_batch();
  return d;
}

void PyCausalLM::load_model(std::unique_ptr<ModelLoader> loader) {
  py::gil_scoped_acquire gil;
  auto& state_dicts = loader->get_state_dicts();
  py::module_::import("xllm_weight_loader");

  py::list py_state_dicts;
  for (const auto& sd : state_dicts) {
    py_state_dicts.append(
        py::cast(PyStateDict(sd.get()), py::return_value_policy::move));
  }

  py_model_.attr("load_weights")(py_state_dicts,
                                 static_cast<int32_t>(tp_rank_),
                                 static_cast<int32_t>(tp_size_));
}

ModelOutput PyCausalLM::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& parameters) {
  torch::NoGradGuard no_grad;

  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      parameters.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(parameters, enable_mla_));
  }

  py::gil_scoped_acquire gil;

  // Pass raw metadata tensors — Python owns the full attention lifecycle.
  py::dict meta;
  meta["slot_mapping"] = attn_metadata->slot_mapping;
  meta["paged_kv_indptr"] = attn_metadata->paged_kv_indptr;
  meta["paged_kv_indices"] = attn_metadata->paged_kv_indices;
  meta["paged_kv_last_page_len"] = attn_metadata->paged_kv_last_page_len;
  meta["is_prefill"] = attn_metadata->is_prefill;
  meta["is_chunked_prefill"] = attn_metadata->is_chunked_prefill;
  meta["enable_cuda_graph"] = attn_metadata->enable_cuda_graph;
  meta["use_tensor_core"] = false;
  if (attn_metadata->q_cu_seq_lens.defined()) {
    meta["q_cu_seq_lens"] = attn_metadata->q_cu_seq_lens;
  }
  if (attn_metadata->kv_cu_seq_lens.defined()) {
    meta["kv_cu_seq_lens"] = attn_metadata->kv_cu_seq_lens;
  }
  if (attn_metadata->qo_indptr.has_value() &&
      attn_metadata->qo_indptr->defined()) {
    meta["qo_indptr"] = attn_metadata->qo_indptr.value();
  }

  py::list kv_caches_py;
  for (auto& kv : kv_caches) {
    kv_caches_py.append(py::make_tuple(kv.get_k_cache(), kv.get_v_cache()));
  }

  py::object hidden_obj =
      py_model_.attr("forward")(tokens, positions, meta, kv_caches_py);
  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

torch::Tensor PyCausalLM::logits(const torch::Tensor& hidden_states,
                                 const torch::Tensor& seleted_idxes) {
  torch::NoGradGuard no_grad;
  py::gil_scoped_acquire gil;
  py::object selected = seleted_idxes.defined()
                            ? py::object(py::cast(seleted_idxes))
                            : py::object(py::none());
  py::object out = py_model_.attr("compute_logits")(hidden_states, selected);
  return out.cast<torch::Tensor>();
}

}  // namespace xllm
