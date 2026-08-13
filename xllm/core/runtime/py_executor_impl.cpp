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

#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "models/llm/py_causal_lm.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace py = pybind11;

namespace xllm {

namespace {

// Thread-local pointer to the active PyCausalLM, set in PyExecutorImpl::run
// before each forward call so the xllm_runtime.tp_all_reduce / tp_all_gather
// Python-callable C++ functions can access the C++ ProcessGroup.
thread_local PyCausalLM* _active_py_causal_lm = nullptr;

py::object optional_tensor(const torch::Tensor& tensor) {
  return tensor.defined() ? py::cast(tensor) : py::none();
}

py::object optional_tensor_or_none(const std::optional<torch::Tensor>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) {
    return py::none();
  }
  return py::cast(*tensor);
}

torch::Tensor dsv4_sparse_attn_sharedkv_metadata(
    int64_t num_heads_q,
    int64_t num_heads_kv,
    int64_t head_dim,
    const c10::optional<torch::Tensor>& cu_seqlens_q,
    const c10::optional<torch::Tensor>& cu_seqlens_ori_kv,
    const c10::optional<torch::Tensor>& cu_seqlens_cmp_kv,
    const c10::optional<torch::Tensor>& seqused_q,
    const c10::optional<torch::Tensor>& seqused_kv,
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_kv,
    int64_t ori_topk,
    int64_t cmp_topk,
    int64_t cmp_ratio,
    int64_t ori_mask_mode,
    int64_t cmp_mask_mode,
    int64_t ori_win_left,
    int64_t ori_win_right,
    const std::string& layout_q,
    const std::string& layout_kv,
    bool has_ori_kv,
    bool has_cmp_kv) {
  kernel::SparseAttnSharedkvMetadataParams params;
  params.num_heads_q = num_heads_q;
  params.num_heads_kv = num_heads_kv;
  params.head_dim = head_dim;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_ori_kv = cu_seqlens_ori_kv;
  params.cu_seqlens_cmp_kv = cu_seqlens_cmp_kv;
  params.seqused_q = seqused_q;
  params.seqused_kv = seqused_kv;
  params.batch_size = batch_size;
  params.max_seqlen_q = max_seqlen_q;
  params.max_seqlen_kv = max_seqlen_kv;
  params.ori_topk = ori_topk;
  params.cmp_topk = cmp_topk;
  params.cmp_ratio = cmp_ratio;
  params.ori_mask_mode = ori_mask_mode;
  params.cmp_mask_mode = cmp_mask_mode;
  params.ori_win_left = ori_win_left;
  params.ori_win_right = ori_win_right;
  params.layout_q = layout_q;
  params.layout_kv = layout_kv;
  params.has_ori_kv = has_ori_kv;
  params.has_cmp_kv = has_cmp_kv;
  return kernel::sparse_attn_sharedkv_metadata(params);
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>>
dsv4_pack_metadata_tensors(const std::vector<torch::Tensor>& tensors,
                           const torch::Tensor& device_reference) {
  CHECK(device_reference.defined() && !device_reference.device().is_cpu())
      << "DSV4 metadata packing requires an accelerator reference tensor";

  constexpr size_t kAlignment = 64;
  std::vector<size_t> offsets;
  offsets.reserve(tensors.size());
  size_t total_bytes = 0;
  for (const auto& tensor : tensors) {
    CHECK(tensor.defined() && tensor.device().is_cpu() && tensor.numel() > 0)
        << "DSV4 metadata packing accepts non-empty CPU tensors only";
    CHECK(tensor.is_contiguous())
        << "DSV4 metadata packing requires contiguous tensors";
    total_bytes =
        ((total_bytes + kAlignment - 1) / kAlignment) * kAlignment;
    offsets.push_back(total_bytes);
    total_bytes += static_cast<size_t>(tensor.numel() * tensor.element_size());
  }

  auto host_buffer = torch::empty(
      {static_cast<int64_t>(total_bytes)},
      torch::TensorOptions()
          .dtype(torch::kUInt8)
          .device(torch::kCPU)
          .pinned_memory(true));
  auto* host_ptr = static_cast<uint8_t*>(host_buffer.data_ptr());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const size_t nbytes =
        static_cast<size_t>(tensor.numel() * tensor.element_size());
    std::memcpy(host_ptr + offsets[i], tensor.data_ptr(), nbytes);
  }

  auto device_buffer = host_buffer.to(
      torch::TensorOptions()
          .dtype(torch::kUInt8)
          .device(device_reference.device()),
      /*non_blocking=*/false);
  const auto* device_ptr =
      static_cast<const uint8_t*>(device_buffer.data_ptr());
  std::vector<torch::Tensor> views;
  views.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    views.push_back(get_tensor_from_blob(tensors[i].sizes().vec(),
                                         tensors[i].scalar_type(),
                                         device_ptr + offsets[i]));
  }
  return {device_buffer, views};
}

class AttentionMetadataView final {
 public:
  explicit AttentionMetadataView(
      std::shared_ptr<layer::AttentionMetadata> metadata,
      const ModelInputParams& params)
      : metadata_(std::move(metadata)),
        kv_seq_lens_host_(make_kv_seq_lens_host(metadata_)),
        q_seq_lens_host_(make_int32_host_tensor(params.attention.host.q_seq_lens)),
        multi_block_tables_(params.multi_block_tables),
        max_query_len_(params.meta.q_max_seq_len),
        max_seq_len_(params.meta.kv_max_seq_len),
        linear_state_indices_(params.embedding.linear_state_indices),
        dp_token_counts_(params.parallel.raw_dp_global_token_nums.empty()
                             ? params.parallel.dp_global_token_nums
                             : params.parallel.raw_dp_global_token_nums) {}

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
  py::object linear_state_indices() const {
    return optional_tensor(linear_state_indices_);
  }
  py::object has_initial_state() const {
    return optional_tensor(metadata_->has_initial_states);
  }
  const std::vector<int32_t>& dp_token_counts() const {
    return dp_token_counts_;
  }
  bool is_prefill() const { return metadata_->is_prefill; }
  bool is_chunked_prefill() const { return metadata_->is_chunked_prefill; }
  int64_t max_query_len() const { return max_query_len_; }
  int64_t max_seq_len() const { return max_seq_len_; }
  py::object dsa_metadata() const { return dsa_metadata_; }
  void set_dsa_metadata(py::object metadata) {
    dsa_metadata_ = std::move(metadata);
  }
  py::object q_seq_lens_host() const {
    return optional_tensor(q_seq_lens_host_);
  }
  py::list multi_block_tables() const {
    py::list tables;
    for (const torch::Tensor& table : multi_block_tables_) {
      tables.append(optional_tensor(table));
    }
    return tables;
  }
  py::object dsa_positions() const { return optional_tensor(dsa_positions_); }
  void set_dsa_positions(py::object value) {
    dsa_positions_ = value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
  }
  py::object dsa_cos_sin() const { return optional_tensor(dsa_cos_sin_); }
  void set_dsa_cos_sin(py::object value) {
    dsa_cos_sin_ = value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
  }
  py::object dsa_c4_cos_sin() const { return optional_tensor(dsa_c4_cos_sin_); }
  void set_dsa_c4_cos_sin(py::object value) {
    dsa_c4_cos_sin_ = value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
  }
  py::object dsa_c128_cos_sin() const { return optional_tensor(dsa_c128_cos_sin_); }
  void set_dsa_c128_cos_sin(py::object value) {
    dsa_c128_cos_sin_ = value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
  }
  int64_t dsa_graph_block_table_cols() const { return dsa_graph_block_table_cols_; }
  void set_dsa_graph_block_table_cols(int64_t value) {
    dsa_graph_block_table_cols_ = value;
  }
  bool dsa_graph_mode() const { return dsa_graph_mode_; }
  void set_dsa_graph_mode(bool value) { dsa_graph_mode_ = value; }

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

  // Wrap a host-side int32 vector as a CPU tensor view. The view is only valid
  // for the lifetime of the AttentionMetadataView (one ``execute`` call), which
  // always outlives the source vector referenced through ``params``.
  static torch::Tensor make_int32_host_tensor(const std::vector<int32_t>& vec) {
    if (vec.empty()) {
      return torch::Tensor();
    }
    return torch::from_blob(
        const_cast<int32_t*>(vec.data()),
        {static_cast<int64_t>(vec.size())},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  }

  std::shared_ptr<layer::AttentionMetadata> metadata_;
  torch::Tensor kv_seq_lens_host_;
  torch::Tensor q_seq_lens_host_;
  const std::vector<torch::Tensor>& multi_block_tables_;
  int64_t max_query_len_;
  int64_t max_seq_len_;
  py::object dsa_metadata_ = py::none();
  torch::Tensor linear_state_indices_;
  std::vector<int32_t> dp_token_counts_;
  torch::Tensor dsa_positions_;
  torch::Tensor dsa_cos_sin_;
  torch::Tensor dsa_c4_cos_sin_;
  torch::Tensor dsa_c128_cos_sin_;
  int64_t dsa_graph_block_table_cols_ = 0;
  bool dsa_graph_mode_ = false;
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
      .def_property_readonly("q_seq_lens_host",
                             &AttentionMetadataView::q_seq_lens_host)
      .def_property_readonly("multi_block_tables",
                             &AttentionMetadataView::multi_block_tables)
      .def_property_readonly("block_table", &AttentionMetadataView::block_table)
      .def_property_readonly("kv_seq_lens", &AttentionMetadataView::kv_seq_lens)
      .def_property_readonly("linear_state_indices",
                             &AttentionMetadataView::linear_state_indices)
      .def_property_readonly("has_initial_state",
                             &AttentionMetadataView::has_initial_state)
      .def_property_readonly("dp_token_counts",
                             &AttentionMetadataView::dp_token_counts)
      .def_property_readonly("max_query_len",
                             &AttentionMetadataView::max_query_len)
      .def_property_readonly("max_seq_len",
                             &AttentionMetadataView::max_seq_len)
      .def_property("dsa_metadata",
                    &AttentionMetadataView::dsa_metadata,
                    &AttentionMetadataView::set_dsa_metadata)
      .def_property("dsa_positions", &AttentionMetadataView::dsa_positions,
                    &AttentionMetadataView::set_dsa_positions)
      .def_property("dsa_cos_sin", &AttentionMetadataView::dsa_cos_sin,
                    &AttentionMetadataView::set_dsa_cos_sin)
      .def_property("dsa_c4_cos_sin", &AttentionMetadataView::dsa_c4_cos_sin,
                    &AttentionMetadataView::set_dsa_c4_cos_sin)
      .def_property("dsa_c128_cos_sin", &AttentionMetadataView::dsa_c128_cos_sin,
                    &AttentionMetadataView::set_dsa_c128_cos_sin)
      .def_property("dsa_graph_block_table_cols",
                    &AttentionMetadataView::dsa_graph_block_table_cols,
                    &AttentionMetadataView::set_dsa_graph_block_table_cols)
      .def_property("dsa_graph_mode", &AttentionMetadataView::dsa_graph_mode,
                    &AttentionMetadataView::set_dsa_graph_mode)
      .def_property_readonly("is_prefill", &AttentionMetadataView::is_prefill)
      .def_property_readonly("is_chunked_prefill",
                             &AttentionMetadataView::is_chunked_prefill);

#if defined(USE_NPU)
  py::class_<NPULayerSynchronizerImpl,
             std::shared_ptr<NPULayerSynchronizerImpl>>(m, "LayerSynchronizer")
      .def("record_event",
           [](NPULayerSynchronizerImpl& self, int64_t layer_id) {
             int32_t device_id = static_cast<int32_t>(
                 c10_npu::getCurrentNPUStream().device_index());
             return self.record_event(layer_id, device_id);
           });
#endif
  // Register C++ HCCL collectives for the Python model path. These reuse the
  // C++ ProcessGroup (no second HCCL communicator), avoiding the resource
  // conflict that arises when Python calls dist.init_process_group("hccl").
  // The Python model calls xllm_runtime.tp_all_reduce(tensor) /
  // xllm_runtime.tp_all_gather(tensor, dim) instead of distributed.all_reduce_.
  m.def("tp_all_reduce", [](torch::Tensor tensor) {
    // The active PyCausalLM is accessed via a thread-local singleton set in
    // PyExecutorImpl::run before each forward call.
    auto* lm = _active_py_causal_lm;
    if (lm != nullptr) {
      lm->tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("tp_all_gather", [](torch::Tensor tensor, int64_t dim) {
    auto* lm = _active_py_causal_lm;
    if (lm != nullptr) {
      return lm->tp_all_gather(tensor, dim);
    }
    return tensor;
  });
  m.def("moe_tp_all_reduce", [](torch::Tensor tensor) {
    auto* lm = _active_py_causal_lm;
    if (lm != nullptr) {
      lm->moe_tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("moe_ep_all_reduce", [](torch::Tensor tensor) {
    auto* lm = _active_py_causal_lm;
    if (lm != nullptr) {
      lm->moe_ep_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("cp_gather",
        [](torch::Tensor tensor,
           const std::vector<int32_t>& tokens_per_rank) {
          auto* lm = _active_py_causal_lm;
          if (lm != nullptr) {
            return lm->cp_gather(tensor, tokens_per_rank);
          }
          return tensor;
        });
  // Keep DeepSeek-V4 metadata construction on the same direct C++ call path as
  // the native model. The Python backend can A/B this against torch.ops without
  // changing any tensor or scalar argument.
  m.def("dsv4_sparse_attn_sharedkv_metadata",
        &dsv4_sparse_attn_sharedkv_metadata);
  m.def("dsv4_pack_metadata_tensors", &dsv4_pack_metadata_tensors);
}

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_causal_lm_(dynamic_cast<PyCausalLM*>(model)),
      args_(args),
      device_(device),
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

void PyExecutorImpl::bind_kv_caches(std::vector<KVCache>& kv_caches) {
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (kv_bound_) {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
    return;
  }

  py::list kv_caches_py;
  for (auto& kv : kv_caches) {
    kv_caches_py.append(py::make_tuple(
        optional_tensor(kv.get_k_cache()),
        optional_tensor(kv.get_v_cache()),
        optional_tensor(kv.get_index_cache()),
        optional_tensor(kv.get_conv_cache()),
        optional_tensor(kv.get_ssm_cache()),
        optional_tensor(kv.get_swa_cache()),
        optional_tensor(kv.get_compress_kv_state()),
        optional_tensor(kv.get_compress_score_state()),
        optional_tensor(kv.get_compress_index_kv_state()),
        optional_tensor(kv.get_compress_index_score_state()),
        optional_tensor_or_none(kv.get_indexer_cache_scale())));
  }
  py_executor_.attr("bind_kv_caches")(kv_caches_py);
  kv_bound_ = true;
  kv_layer_count_ = num_layers;
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);
  // Set the thread-local PyCausalLM so xllm_runtime.tp_all_reduce /
  // tp_all_gather can access the C++ ProcessGroup during Python forward.
  _active_py_causal_lm = py_causal_lm_;

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }

  py::gil_scoped_acquire gil;

  // Slot order must match ``LayerCache`` on the Python side. The first 5
  // slots are generic and the trailing 6 are DeepSeek-V4 DSA caches.
  bind_kv_caches(kv_caches);

  py::object py_metadata =
      py::cast(AttentionMetadataView(attn_metadata, params));

  py::object py_sync = py::none();
#if defined(USE_NPU)
  if (params.parallel.layer_synchronizer) {
    py_sync = py::cast(params.parallel.layer_synchronizer);
  }
#endif

  // Execute: one C++ -> Python call per step.
  py::object hidden_obj =
      py_executor_.attr("execute")(tokens, positions, py_metadata, py_sync);

  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

void PyExecutorImpl::prepare_graph_input(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
  if (!ExecutionConfig::get_instance().enable_graph_double_buffer()) {
    return;
  }

  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }

  py::gil_scoped_acquire gil;
  bind_kv_caches(kv_caches);
  py::object py_metadata = py::cast(AttentionMetadataView(attn_metadata, params));
  py_executor_.attr("prepare_graph_input")(tokens, positions, py_metadata);
}

}  // namespace xllm
