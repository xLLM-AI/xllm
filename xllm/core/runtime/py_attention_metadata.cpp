/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/py_attention_metadata.h"

#include <pybind11/stl.h>
#include <torch/python.h>

#include <algorithm>
#include <utility>

#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"
#include "core/util/pybind_helper.h"
#include "core/util/tensor_helper.h"

namespace py = pybind11;

namespace xllm {
namespace {

struct PythonObjectHolder final {
  explicit PythonObjectHolder(py::object value) : value(std::move(value)) {}

  ~PythonObjectHolder() { clear_python_object(value); }

  py::object value;
};

}  // namespace

PythonAttentionMetadata::PythonAttentionMetadata(py::object value)
    : object_holder_(std::make_shared<PythonObjectHolder>(std::move(value))) {}

py::object PythonAttentionMetadata::value() const {
  return std::static_pointer_cast<PythonObjectHolder>(object_holder_)->value;
}

void register_attention_metadata_views(py::module_& module) {
  py::class_<PyExpandedDecodeMetadataView>(module, "ExpandedDecodeMetadataView")
      .def_property_readonly("enabled", &PyExpandedDecodeMetadataView::enabled)
      .def_property_readonly("kv_seq_lens",
                             &PyExpandedDecodeMetadataView::kv_seq_lens)
      .def_property_readonly("block_table",
                             &PyExpandedDecodeMetadataView::block_table)
      .def_property_readonly("paged_kv_indptr",
                             &PyExpandedDecodeMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &PyExpandedDecodeMetadataView::paged_kv_indices)
      .def_property_readonly(
          "paged_kv_last_page_len",
          &PyExpandedDecodeMetadataView::paged_kv_last_page_len)
      .def_property_readonly(
          "paged_attention_tiling_data",
          &PyExpandedDecodeMetadataView::paged_attention_tiling_data)
      .def_property_readonly("kv_seq_lens_host",
                             &PyExpandedDecodeMetadataView::kv_seq_lens_host)
      .def_property_readonly(
          "kv_seq_lens_host_values",
          &PyExpandedDecodeMetadataView::kv_seq_lens_host_values);

  py::class_<PyAttentionMetadataView>(module, "AttentionMetadataView")
      .def_property("prepared_attention_state",
                    &PyAttentionMetadataView::prepared_attention_state,
                    &PyAttentionMetadataView::set_prepared_attention_state)
      .def_property_readonly("slot_mapping",
                             &PyAttentionMetadataView::slot_mapping)
      .def_property_readonly("local_slot_mapping",
                             &PyAttentionMetadataView::local_slot_mapping)
      .def_property_readonly("kv_split_size",
                             &PyAttentionMetadataView::kv_split_size)
      .def_property_readonly("kv_split_rank",
                             &PyAttentionMetadataView::kv_split_rank)
      .def_property_readonly("has_kv_shard",
                             &PyAttentionMetadataView::has_kv_shard)
      .def_property_readonly("paged_kv_indptr",
                             &PyAttentionMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &PyAttentionMetadataView::paged_kv_indices)
      .def_property_readonly("paged_kv_last_page_len",
                             &PyAttentionMetadataView::paged_kv_last_page_len)
      .def_property_readonly("qo_indptr", &PyAttentionMetadataView::qo_indptr)
      .def_property_readonly("q_cu_seq_lens",
                             &PyAttentionMetadataView::q_cu_seq_lens)
      .def_property_readonly(
          "q_cu_seq_lens_host_values",
          &PyAttentionMetadataView::q_cu_seq_lens_host_values)
      .def_property_readonly("kv_cu_seq_lens",
                             &PyAttentionMetadataView::kv_cu_seq_lens)
      .def_property_readonly("kv_seq_lens_host",
                             &PyAttentionMetadataView::kv_seq_lens_host)
      .def_property_readonly("kv_seq_lens_host_values",
                             &PyAttentionMetadataView::kv_seq_lens_host_values)
      .def_property_readonly(
          "new_cache_slots_host_values",
          &PyAttentionMetadataView::new_cache_slots_host_values)
      .def_property_readonly("q_seq_lens_host",
                             &PyAttentionMetadataView::q_seq_lens_host)
      .def_property_readonly("multi_block_tables",
                             &PyAttentionMetadataView::multi_block_tables)
      .def_property_readonly("block_table",
                             &PyAttentionMetadataView::block_table)
      .def_property_readonly("kv_seq_lens",
                             &PyAttentionMetadataView::kv_seq_lens)
      .def_property_readonly("linear_state_indices",
                             &PyAttentionMetadataView::linear_state_indices)
      .def_property_readonly(
          "linear_state_read_indices",
          &PyAttentionMetadataView::linear_state_read_indices)
      .def_property_readonly(
          "linear_state_write_indices",
          &PyAttentionMetadataView::linear_state_write_indices)
      .def_property_readonly("has_initial_state",
                             &PyAttentionMetadataView::has_initial_state)
      .def_property_readonly(
          "dp_execution_token_counts",
          &PyAttentionMetadataView::dp_execution_token_counts)
      .def_property_readonly(
          "raw_dp_execution_token_counts",
          &PyAttentionMetadataView::raw_dp_execution_token_counts)
      .def_property_readonly("dp_global_sequence_nums",
                             &PyAttentionMetadataView::dp_global_sequence_nums)
      .def_property_readonly("dp_is_decode",
                             &PyAttentionMetadataView::dp_is_decode)
      .def_property_readonly("q_seq_lens", &PyAttentionMetadataView::q_seq_lens)
      .def_property_readonly("expanded_decode_metadata",
                             &PyAttentionMetadataView::expanded_decode_metadata)
      .def_property_readonly("max_query_len",
                             &PyAttentionMetadataView::max_query_len)
      .def_property_readonly("max_seq_len",
                             &PyAttentionMetadataView::max_seq_len)
      .def_property("dsa_metadata",
                    &PyAttentionMetadataView::dsa_metadata,
                    &PyAttentionMetadataView::set_dsa_metadata)
      .def_property("dsa_positions",
                    &PyAttentionMetadataView::dsa_positions,
                    &PyAttentionMetadataView::set_dsa_positions)
      .def_property("dsa_cos_sin",
                    &PyAttentionMetadataView::dsa_cos_sin,
                    &PyAttentionMetadataView::set_dsa_cos_sin)
      .def_property("dsa_c4_cos_sin",
                    &PyAttentionMetadataView::dsa_c4_cos_sin,
                    &PyAttentionMetadataView::set_dsa_c4_cos_sin)
      .def_property("dsa_c128_cos_sin",
                    &PyAttentionMetadataView::dsa_c128_cos_sin,
                    &PyAttentionMetadataView::set_dsa_c128_cos_sin)
      .def_property("dsa_graph_block_table_cols",
                    &PyAttentionMetadataView::dsa_graph_block_table_cols,
                    &PyAttentionMetadataView::set_dsa_graph_block_table_cols)
      .def_property("dsa_graph_mode",
                    &PyAttentionMetadataView::dsa_graph_mode,
                    &PyAttentionMetadataView::set_dsa_graph_mode)
      .def_property_readonly("is_prefill", &PyAttentionMetadataView::is_prefill)
      .def_property_readonly("is_chunked_prefill",
                             &PyAttentionMetadataView::is_chunked_prefill)
      .def_property_readonly("is_mixed", &PyAttentionMetadataView::is_mixed)
      .def_property_readonly("is_spec_verify",
                             &PyAttentionMetadataView::is_spec_verify)
      .def_property_readonly("is_dummy", &PyAttentionMetadataView::is_dummy);
}

PyExpandedDecodeMetadataView::PyExpandedDecodeMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata)
    : metadata_(std::move(metadata)) {}

bool PyExpandedDecodeMetadataView::enabled() const {
  return metadata().enabled;
}

py::object PyExpandedDecodeMetadataView::kv_seq_lens() const {
  return optional_tensor(metadata().kv_seq_lens);
}

py::object PyExpandedDecodeMetadataView::block_table() const {
  return optional_tensor(metadata().block_table);
}

py::object PyExpandedDecodeMetadataView::paged_kv_indptr() const {
  return optional_tensor(metadata().paged_kv_indptr);
}

py::object PyExpandedDecodeMetadataView::paged_kv_indices() const {
  return optional_tensor(metadata().paged_kv_indices);
}

py::object PyExpandedDecodeMetadataView::paged_kv_last_page_len() const {
  return optional_tensor(metadata().paged_kv_last_page_len);
}

py::object PyExpandedDecodeMetadataView::paged_attention_tiling_data() const {
  return optional_tensor(metadata().paged_attention_tiling_data);
}

py::object PyExpandedDecodeMetadataView::kv_seq_lens_host() const {
  return optional_tensor(metadata().kv_seq_lens_host);
}

const std::vector<int32_t>&
PyExpandedDecodeMetadataView::kv_seq_lens_host_values() const {
  return metadata().kv_seq_lens_host_vec;
}

const layer::ExpandedDecodeMetadata& PyExpandedDecodeMetadataView::metadata()
    const {
  return metadata_->expanded_decode;
}

PyAttentionMetadataView::PyAttentionMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata)
    : metadata_(std::move(metadata)),
      kv_seq_lens_host_(
          make_host_int32_view(metadata_, metadata_->kv_seq_lens_vec)),
      q_seq_lens_host_(
          make_host_int32_view(metadata_, metadata_->q_seq_lens_vec)) {}

PyAttentionMetadataView::PyAttentionMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata,
    const ModelInputParams& params)
    : PyAttentionMetadataView(std::move(metadata)) {
  if (!params.multi_block_tables.empty()) {
    new_cache_slots_host_values_ = params.attention.host.new_cache_slots;
  }
  multi_block_tables_ = params.multi_block_tables;
  linear_state_indices_ = params.embedding.linear_state_indices;
  const auto& cache_ops = params.linear_state_cache_ops;
  const auto is_direct_read = [](const LinearStateCacheOp& op) {
    return op.restore_src_slot_id >= 0 && !op.restore_requested &&
           !op.reset_requested;
  };
  const bool has_direct_read =
      std::any_of(cache_ops.begin(), cache_ops.end(), is_direct_read);
  if (has_direct_read) {
    CHECK((metadata_->is_prefill || metadata_->is_chunked_prefill) &&
          !params.is_spec_verify)
        << "linear-state direct read is only supported for non-speculative "
           "prefill";
    CHECK(linear_state_indices_.defined());
    CHECK_EQ(cache_ops.size(), params.embedding.linear_state_ids.size())
        << "direct-read cache ops must align with host linear-state ids";
    CHECK_EQ(cache_ops.size(),
             static_cast<size_t>(linear_state_indices_.numel()))
        << "direct-read cache ops must align with Python metadata rows";

    std::vector<int32_t> read_ids = params.embedding.linear_state_ids;
    for (size_t i = 0; i < cache_ops.size(); ++i) {
      const LinearStateCacheOp& cache_op = cache_ops[i];
      if (is_direct_read(cache_op)) {
        read_ids[i] = cache_op.restore_src_slot_id;
      }
    }
    linear_state_read_indices_ =
        make_cpu_tensor(read_ids).to(linear_state_indices_.device());
  }
  // Python model kernels consume materialized execution rows. Empty DP ranks
  // therefore contribute the worker-created dummy row instead of zero rows.
  dp_execution_token_counts_ = params.parallel.dp_global_token_nums;
  dp_global_sequence_nums_ = params.parallel.dp_global_sequence_nums;
  for (int32_t& count : dp_execution_token_counts_) {
    if (count == 0) {
      count = 1;
    }
  }
  dp_is_decode_ = params.parallel.dp_is_decode;
  if (params.enable_graph) {
    raw_dp_execution_token_counts_ =
        params.parallel.raw_dp_global_token_nums.empty()
            ? params.parallel.dp_global_token_nums
            : params.parallel.raw_dp_global_token_nums;
    // Prepared ordinary decode has one token per actual sequence, before
    // padding.
    if (raw_dp_execution_token_counts_.empty()) {
      raw_dp_execution_token_counts_.push_back(
          params.meta.actual_num_sequences);
    }
  }
}

const torch::Tensor& PyAttentionMetadataView::slot_mapping() const {
  return metadata_->slot_mapping;
}

py::object PyAttentionMetadataView::local_slot_mapping() const {
  if (metadata_->kv_shard_batch_metadata == nullptr) {
    return py::none();
  }
  return optional_tensor(
      metadata_->kv_shard_batch_metadata->local_slot_mapping);
}

int32_t PyAttentionMetadataView::kv_split_size() const {
  if (metadata_->kv_shard_batch_metadata == nullptr) {
    return 1;
  }
  return metadata_->kv_shard_batch_metadata->kv_split_size;
}

int32_t PyAttentionMetadataView::kv_split_rank() const {
  if (metadata_->kv_shard_batch_metadata == nullptr) {
    return 0;
  }
  return metadata_->kv_shard_batch_metadata->kv_split_rank;
}

bool PyAttentionMetadataView::has_kv_shard() const {
  return metadata_->kv_shard_batch_metadata != nullptr;
}

const torch::Tensor& PyAttentionMetadataView::paged_kv_indptr() const {
  return metadata_->paged_kv_indptr;
}

const torch::Tensor& PyAttentionMetadataView::paged_kv_indices() const {
  return metadata_->paged_kv_indices;
}

const torch::Tensor& PyAttentionMetadataView::paged_kv_last_page_len() const {
  return metadata_->paged_kv_last_page_len;
}

py::object PyAttentionMetadataView::qo_indptr() const {
  return optional_tensor(metadata_->qo_indptr);
}

py::object PyAttentionMetadataView::q_cu_seq_lens() const {
  return optional_tensor(metadata_->q_cu_seq_lens);
}

const std::vector<int64_t>& PyAttentionMetadataView::q_cu_seq_lens_host_values()
    const {
  return metadata_->q_cu_seq_lens_host_vec;
}

py::object PyAttentionMetadataView::kv_cu_seq_lens() const {
  return optional_tensor(metadata_->kv_cu_seq_lens);
}

py::object PyAttentionMetadataView::kv_seq_lens_host() const {
  return optional_tensor(kv_seq_lens_host_);
}

const std::vector<int32_t>& PyAttentionMetadataView::kv_seq_lens_host_values()
    const {
  return metadata_->kv_seq_lens_vec;
}

const std::vector<int32_t>&
PyAttentionMetadataView::new_cache_slots_host_values() const {
  return new_cache_slots_host_values_;
}

py::object PyAttentionMetadataView::block_table() const {
  return optional_tensor(metadata_->block_table);
}

py::object PyAttentionMetadataView::kv_seq_lens() const {
  return optional_tensor(metadata_->kv_seq_lens);
}

py::object PyAttentionMetadataView::linear_state_indices() const {
  return optional_tensor(linear_state_indices_);
}

py::object PyAttentionMetadataView::linear_state_read_indices() const {
  return optional_tensor(linear_state_read_indices_);
}

py::object PyAttentionMetadataView::linear_state_write_indices() const {
  return optional_tensor(linear_state_indices_);
}

py::object PyAttentionMetadataView::has_initial_state() const {
  return optional_tensor(metadata_->has_initial_states);
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_execution_token_counts()
    const {
  return dp_execution_token_counts_;
}

const std::vector<int32_t>&
PyAttentionMetadataView::raw_dp_execution_token_counts() const {
  return raw_dp_execution_token_counts_;
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_global_sequence_nums()
    const {
  return dp_global_sequence_nums_;
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_is_decode() const {
  return dp_is_decode_;
}

py::object PyAttentionMetadataView::q_seq_lens() const {
  return optional_tensor(metadata_->q_seq_lens);
}

py::object PyAttentionMetadataView::q_seq_lens_host() const {
  return optional_tensor(q_seq_lens_host_);
}

py::list PyAttentionMetadataView::multi_block_tables() const {
  py::list tables;
  for (const torch::Tensor& table : multi_block_tables_) {
    tables.append(optional_tensor(table));
  }
  return tables;
}

PyExpandedDecodeMetadataView PyAttentionMetadataView::expanded_decode_metadata()
    const {
  return PyExpandedDecodeMetadataView(metadata_);
}

int64_t PyAttentionMetadataView::max_query_len() const {
  return metadata_->max_query_len;
}

int64_t PyAttentionMetadataView::max_seq_len() const {
  return metadata_->max_seq_len;
}

py::object PyAttentionMetadataView::prepared_attention_state() const {
  if (!prepared_attention_holder_) {
    return py::none();
  }
  return std::static_pointer_cast<PythonObjectHolder>(
             prepared_attention_holder_)
      ->value;
}

void PyAttentionMetadataView::set_prepared_attention_state(py::object value) {
  if (value.is_none()) {
    prepared_attention_holder_.reset();
    return;
  }
  prepared_attention_holder_ =
      std::make_shared<PythonObjectHolder>(std::move(value));
}

py::object PyAttentionMetadataView::dsa_metadata() const {
  if (!dsa_metadata_holder_) {
    return py::none();
  }
  return std::static_pointer_cast<PythonObjectHolder>(dsa_metadata_holder_)
      ->value;
}

void PyAttentionMetadataView::set_dsa_metadata(py::object value) {
  if (value.is_none()) {
    dsa_metadata_holder_.reset();
    return;
  }
  dsa_metadata_holder_ = std::make_shared<PythonObjectHolder>(std::move(value));
}

py::object PyAttentionMetadataView::dsa_positions() const {
  return optional_tensor(dsa_positions_);
}

void PyAttentionMetadataView::set_dsa_positions(py::object value) {
  dsa_positions_ = tensor_from_python(value);
}

py::object PyAttentionMetadataView::dsa_cos_sin() const {
  return optional_tensor(dsa_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_cos_sin(py::object value) {
  dsa_cos_sin_ = tensor_from_python(value);
}

py::object PyAttentionMetadataView::dsa_c4_cos_sin() const {
  return optional_tensor(dsa_c4_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_c4_cos_sin(py::object value) {
  dsa_c4_cos_sin_ = tensor_from_python(value);
}

py::object PyAttentionMetadataView::dsa_c128_cos_sin() const {
  return optional_tensor(dsa_c128_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_c128_cos_sin(py::object value) {
  dsa_c128_cos_sin_ = tensor_from_python(value);
}

int64_t PyAttentionMetadataView::dsa_graph_block_table_cols() const {
  return dsa_graph_block_table_cols_;
}

void PyAttentionMetadataView::set_dsa_graph_block_table_cols(int64_t value) {
  dsa_graph_block_table_cols_ = value;
}

bool PyAttentionMetadataView::dsa_graph_mode() const { return dsa_graph_mode_; }

void PyAttentionMetadataView::set_dsa_graph_mode(bool value) {
  dsa_graph_mode_ = value;
}

bool PyAttentionMetadataView::is_prefill() const {
  return metadata_->is_prefill;
}

bool PyAttentionMetadataView::is_chunked_prefill() const {
  return metadata_->is_chunked_prefill;
}

bool PyAttentionMetadataView::is_mixed() const { return metadata_->is_mixed; }

bool PyAttentionMetadataView::is_spec_verify() const {
  return metadata_->is_spec_verify;
}

bool PyAttentionMetadataView::is_dummy() const { return metadata_->is_dummy; }

torch::Tensor PyAttentionMetadataView::make_host_int32_view(
    const std::shared_ptr<layer::AttentionMetadata>& metadata,
    std::vector<int32_t>& host_vec) {
  if (host_vec.empty()) {
    return torch::Tensor();
  }

  std::shared_ptr<layer::AttentionMetadata> owner = metadata;
  return torch::from_blob(
      host_vec.data(),
      {static_cast<int64_t>(host_vec.size())},
      [owner = std::move(owner)](void*) mutable { owner.reset(); },
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
}

}  // namespace xllm
