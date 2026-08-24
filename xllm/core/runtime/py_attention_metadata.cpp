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
#include <torch/extension.h>

#include <utility>

#include "core/framework/model/model_input_params.h"
#include "core/layers/common/attention_metadata.h"

namespace py = pybind11;

namespace xllm {
namespace {

struct PythonObjectHolder final {
  explicit PythonObjectHolder(py::object value) : value(std::move(value)) {}

  ~PythonObjectHolder() {
    if (!Py_IsInitialized()) {
      (void)value.release();
      return;
    }
    py::gil_scoped_acquire gil;
    value = py::object();
  }

  py::object value;
};

}  // namespace

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
      .def_property_readonly("slot_mapping",
                             &PyAttentionMetadataView::slot_mapping)
      .def_property_readonly("paged_kv_indptr",
                             &PyAttentionMetadataView::paged_kv_indptr)
      .def_property_readonly("paged_kv_indices",
                             &PyAttentionMetadataView::paged_kv_indices)
      .def_property_readonly("paged_kv_last_page_len",
                             &PyAttentionMetadataView::paged_kv_last_page_len)
      .def_property_readonly("qo_indptr", &PyAttentionMetadataView::qo_indptr)
      .def_property_readonly("q_cu_seq_lens",
                             &PyAttentionMetadataView::q_cu_seq_lens)
      .def_property_readonly("kv_cu_seq_lens",
                             &PyAttentionMetadataView::kv_cu_seq_lens)
      .def_property_readonly("kv_seq_lens_host",
                             &PyAttentionMetadataView::kv_seq_lens_host)
      .def_property_readonly("kv_seq_lens_host_values",
                             &PyAttentionMetadataView::kv_seq_lens_host_values)
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
      .def_property_readonly("has_initial_state",
                             &PyAttentionMetadataView::has_initial_state)
      .def_property_readonly("dp_token_counts",
                             &PyAttentionMetadataView::dp_token_counts)
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
                             &PyAttentionMetadataView::is_chunked_prefill);
}

PyExpandedDecodeMetadataView::PyExpandedDecodeMetadataView(
    std::shared_ptr<layer::AttentionMetadata> metadata)
    : metadata_(std::move(metadata)) {}

bool PyExpandedDecodeMetadataView::enabled() const {
  return metadata().enabled;
}

py::object PyExpandedDecodeMetadataView::kv_seq_lens() const {
  return metadata().kv_seq_lens.defined() ? py::cast(metadata().kv_seq_lens)
                                          : py::none();
}

py::object PyExpandedDecodeMetadataView::block_table() const {
  return metadata().block_table.defined() ? py::cast(metadata().block_table)
                                          : py::none();
}

py::object PyExpandedDecodeMetadataView::paged_kv_indptr() const {
  return metadata().paged_kv_indptr.defined()
             ? py::cast(metadata().paged_kv_indptr)
             : py::none();
}

py::object PyExpandedDecodeMetadataView::paged_kv_indices() const {
  return metadata().paged_kv_indices.defined()
             ? py::cast(metadata().paged_kv_indices)
             : py::none();
}

py::object PyExpandedDecodeMetadataView::paged_kv_last_page_len() const {
  return metadata().paged_kv_last_page_len.defined()
             ? py::cast(metadata().paged_kv_last_page_len)
             : py::none();
}

py::object PyExpandedDecodeMetadataView::paged_attention_tiling_data() const {
  return metadata().paged_attention_tiling_data.defined()
             ? py::cast(metadata().paged_attention_tiling_data)
             : py::none();
}

py::object PyExpandedDecodeMetadataView::kv_seq_lens_host() const {
  return metadata().kv_seq_lens_host.defined()
             ? py::cast(metadata().kv_seq_lens_host)
             : py::none();
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
  multi_block_tables_ = params.multi_block_tables;
  linear_state_indices_ = params.embedding.linear_state_indices;
  dp_token_counts_ = params.parallel.raw_dp_global_token_nums.empty()
                         ? params.parallel.dp_global_token_nums
                         : params.parallel.raw_dp_global_token_nums;
  dp_is_decode_ = params.parallel.dp_is_decode;
}

const torch::Tensor& PyAttentionMetadataView::slot_mapping() const {
  return metadata_->slot_mapping;
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
  if (!metadata_->qo_indptr.has_value() || !metadata_->qo_indptr->defined()) {
    return py::none();
  }
  return py::cast(*metadata_->qo_indptr);
}

py::object PyAttentionMetadataView::q_cu_seq_lens() const {
  return optional_tensor(metadata_->q_cu_seq_lens);
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

py::object PyAttentionMetadataView::block_table() const {
  return optional_tensor(metadata_->block_table);
}

py::object PyAttentionMetadataView::kv_seq_lens() const {
  return optional_tensor(metadata_->kv_seq_lens);
}

py::object PyAttentionMetadataView::linear_state_indices() const {
  return optional_tensor(linear_state_indices_);
}

py::object PyAttentionMetadataView::has_initial_state() const {
  return optional_tensor(metadata_->has_initial_states);
}

const std::vector<int32_t>& PyAttentionMetadataView::dp_token_counts() const {
  return dp_token_counts_;
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
  dsa_positions_ =
      value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
}

py::object PyAttentionMetadataView::dsa_cos_sin() const {
  return optional_tensor(dsa_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_cos_sin(py::object value) {
  dsa_cos_sin_ =
      value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
}

py::object PyAttentionMetadataView::dsa_c4_cos_sin() const {
  return optional_tensor(dsa_c4_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_c4_cos_sin(py::object value) {
  dsa_c4_cos_sin_ =
      value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
}

py::object PyAttentionMetadataView::dsa_c128_cos_sin() const {
  return optional_tensor(dsa_c128_cos_sin_);
}

void PyAttentionMetadataView::set_dsa_c128_cos_sin(py::object value) {
  dsa_c128_cos_sin_ =
      value.is_none() ? torch::Tensor() : value.cast<torch::Tensor>();
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

py::object PyAttentionMetadataView::optional_tensor(
    const torch::Tensor& tensor) {
  return tensor.defined() ? py::cast(tensor) : py::none();
}

}  // namespace xllm
