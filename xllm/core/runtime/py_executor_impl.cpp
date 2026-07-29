/* Copyright 2026 The xLLM Authors.

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

#include "py_executor_impl.h"

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <memory>
#include <vector>

#include "common/metrics.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "models/llm/py_causal_lm.h"

namespace py = pybind11;

namespace xllm {

namespace {

class AttentionMetadataView final {
 public:
  explicit AttentionMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata)
      : metadata_(std::move(metadata)),
        kv_seq_lens_host_(make_kv_seq_lens_host(metadata_)) {}

  const torch::Tensor& slot_mapping() const { return metadata_->slot_mapping; }
  const torch::Tensor& paged_kv_indptr() const {
    return metadata_->paged_kv_indptr;
  }
  const torch::Tensor& paged_kv_indices() const {
    return metadata_->paged_kv_indices;
  }
  const torch::Tensor& paged_kv_last_page_len() const {
    return metadata_->paged_kv_last_page_len;
  }
  py::object qo_indptr() const {
    if (!metadata_->qo_indptr.has_value() || !metadata_->qo_indptr->defined()) {
      return py::none();
    }
    return py::cast(*metadata_->qo_indptr);
  }
  py::object q_cu_seq_lens() const {
    return optional_tensor(metadata_->q_cu_seq_lens);
  }
  py::object kv_cu_seq_lens() const {
    return optional_tensor(metadata_->kv_cu_seq_lens);
  }
  py::object kv_seq_lens_host() const {
    return optional_tensor(kv_seq_lens_host_);
  }
  py::object block_table() const {
    return optional_tensor(metadata_->block_table);
  }
  py::object kv_seq_lens() const {
    return optional_tensor(metadata_->kv_seq_lens);
  }
  bool is_prefill() const { return metadata_->is_prefill; }
  bool is_chunked_prefill() const { return metadata_->is_chunked_prefill; }

 private:
  static torch::Tensor make_kv_seq_lens_host(
      const std::shared_ptr<layer::AttentionMetadata>& metadata) {
    if (metadata->kv_seq_lens_vec.empty()) {
      return torch::Tensor();
    }

    std::shared_ptr<layer::AttentionMetadata> owner = metadata;
    return torch::from_blob(
        metadata->kv_seq_lens_vec.data(),
        {static_cast<int64_t>(metadata->kv_seq_lens_vec.size())},
        [owner = std::move(owner)](void*) mutable { owner.reset(); },
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  }

  static py::object optional_tensor(const torch::Tensor& tensor) {
    return tensor.defined() ? py::cast(tensor) : py::none();
  }

  std::shared_ptr<layer::AttentionMetadata> metadata_;
  torch::Tensor kv_seq_lens_host_;
};

}  // namespace

PYBIND11_EMBEDDED_MODULE(xllm_runtime, m) {
  py::class_<AttentionMetadataView>(m, "AttentionMetadataView")
      .def_property_readonly("slot_mapping",
                             &AttentionMetadataView::slot_mapping)
      .def_property_readonly("paged_kv_indptr",
                             &AttentionMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &AttentionMetadataView::paged_kv_indices)
      .def_property_readonly("paged_kv_last_page_len",
                             &AttentionMetadataView::paged_kv_last_page_len)
      .def_property_readonly("qo_indptr", &AttentionMetadataView::qo_indptr)
      .def_property_readonly("q_cu_seq_lens",
                             &AttentionMetadataView::q_cu_seq_lens)
      .def_property_readonly("kv_cu_seq_lens",
                             &AttentionMetadataView::kv_cu_seq_lens)
      .def_property_readonly("kv_seq_lens_host",
                             &AttentionMetadataView::kv_seq_lens_host)
      .def_property_readonly("block_table", &AttentionMetadataView::block_table)
      .def_property_readonly("kv_seq_lens", &AttentionMetadataView::kv_seq_lens)
      .def_property_readonly("is_prefill", &AttentionMetadataView::is_prefill)
      .def_property_readonly("is_chunked_prefill",
                             &AttentionMetadataView::is_chunked_prefill);
}

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_causal_lm_(dynamic_cast<PyCausalLM*>(model)),
      args_(args),
      options_(options),
      enable_mla_(args.enable_mla()) {
  CHECK(py_causal_lm_ != nullptr) << "PyExecutorImpl requires PyCausalLM";

  py::gil_scoped_acquire gil;
  py::module_::import("xllm_runtime");
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  py_executor_ =
      executor_module.attr("ModelExecutor")(py_causal_lm_->python_model(),
                                            py_causal_lm_->config_dict(),
                                            options_.max_seqs_per_batch());
}

PyExecutorImpl::~PyExecutorImpl() {
  py::gil_scoped_acquire gil;
  py_executor_ = py::object();
}

ForwardInput PyExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(params, enable_mla_));
  }

  py::gil_scoped_acquire gil;

  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py::list kv_caches_py;
    for (auto& kv : kv_caches) {
      kv_caches_py.append(py::make_tuple(kv.get_k_cache(), kv.get_v_cache()));
    }
    py_executor_.attr("bind_kv_caches")(kv_caches_py);
    kv_bound_ = true;
    kv_layer_count_ = num_layers;
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
  }

  py::object py_metadata = py::cast(AttentionMetadataView(attn_metadata));

  // Execute: one C++ -> Python call per step.
  py::object hidden_obj =
      py_executor_.attr("execute")(tokens, positions, py_metadata);
  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

}  // namespace xllm
