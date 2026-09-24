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

#include "core/runtime/slot_buffer.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "core/layers/common/attention_metadata.h"
#include "core/platform/platform.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/task_execution_pipeline_speculative.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

enum class ResultDomain : uint8_t { SEQUENCE, TOKEN, TOP_TOKEN };

struct ResultField {
  torch::Tensor TokenResultTensors::* member;
  torch::ScalarType dtype;
  ResultDomain domain;
  double padding;
};

constexpr double kLogprobPadding = -std::numeric_limits<double>::infinity();
constexpr std::array<ResultField, 5> kResultFields = {{
    {&TokenResultTensors::tokens, torch::kInt64, ResultDomain::TOKEN, -1},
    {&TokenResultTensors::lengths, torch::kInt32, ResultDomain::SEQUENCE, 0},
    {&TokenResultTensors::logprobs,
     torch::kFloat32,
     ResultDomain::TOKEN,
     kLogprobPadding},
    {&TokenResultTensors::top_tokens,
     torch::kInt64,
     ResultDomain::TOP_TOKEN,
     -1},
    {&TokenResultTensors::top_logprobs,
     torch::kFloat32,
     ResultDomain::TOP_TOKEN,
     kLogprobPadding},
}};

uint64_t result_elements(const ResultField& field,
                         const SlotBufferCapacity& capacity) {
  if (capacity.max_result_width == 0) {
    return 0;
  }
  const uint64_t rows = capacity.max_sample_rows;
  if (field.domain == ResultDomain::SEQUENCE) {
    return rows;
  }
  const uint64_t tokens = rows * capacity.max_result_width;
  return field.domain == ResultDomain::TOP_TOKEN
             ? tokens * capacity.max_top_logprobs
             : tokens;
}

TokenResultTensors bind_result_views(const TokenResultTensors& storage,
                                     uint32_t rows,
                                     uint32_t width,
                                     uint32_t top_logprobs,
                                     bool logprobs,
                                     bool matrix) {
  TokenResultTensors view;
  if (rows == 0) {
    return view;
  }
  const int64_t tokens = static_cast<int64_t>(rows) * width;
  view.tokens = storage.tokens.narrow(/*dim=*/0, /*start=*/0, tokens);
  if (logprobs) {
    view.logprobs = storage.logprobs.narrow(/*dim=*/0, /*start=*/0, tokens);
  }
  if (top_logprobs != 0) {
    view.top_tokens =
        storage.top_tokens.narrow(/*dim=*/0, /*start=*/0, tokens * top_logprobs)
            .view({tokens, top_logprobs});
    view.top_logprobs =
        storage.top_logprobs
            .narrow(/*dim=*/0, /*start=*/0, tokens * top_logprobs)
            .view({tokens, top_logprobs});
  }
  if (matrix) {
    view.tokens = view.tokens.view({rows, width});
    view.lengths = storage.lengths.narrow(/*dim=*/0, /*start=*/0, rows);
    if (view.logprobs.defined()) {
      view.logprobs = view.logprobs.view({rows, width});
    }
    if (view.top_tokens.defined()) {
      view.top_tokens = view.top_tokens.view({rows, width, top_logprobs});
      view.top_logprobs = view.top_logprobs.view({rows, width, top_logprobs});
    }
  }
  return view;
}

enum class RowDomain : uint8_t { SELECTED, SAMPLE, HISTORY };

struct InputField {
  torch::Tensor SamplingParameters::* member;
  torch::ScalarType dtype;
  RowDomain domain;
};

constexpr std::array<InputField, 12> kFields = {{
    {&SamplingParameters::selected_token_idxes,
     torch::kInt32,
     RowDomain::SELECTED},
    {&SamplingParameters::frequency_penalties,
     torch::kFloat32,
     RowDomain::SELECTED},
    {&SamplingParameters::presence_penalties,
     torch::kFloat32,
     RowDomain::SELECTED},
    {&SamplingParameters::repetition_penalties,
     torch::kFloat32,
     RowDomain::SELECTED},
    {&SamplingParameters::temperatures, torch::kFloat32, RowDomain::SELECTED},
    {&SamplingParameters::top_p, torch::kFloat32, RowDomain::SELECTED},
    {&SamplingParameters::top_k, torch::kInt64, RowDomain::SELECTED},
    {&SamplingParameters::unique_token_ids, torch::kInt64, RowDomain::HISTORY},
    {&SamplingParameters::unique_token_counts,
     torch::kInt32,
     RowDomain::HISTORY},
    {&SamplingParameters::unique_token_ids_lens,
     torch::kInt32,
     RowDomain::SELECTED},
    {&SamplingParameters::sample_idxes, torch::kInt32, RowDomain::SAMPLE},
    {&SamplingParameters::do_sample, torch::kBool, RowDomain::SAMPLE},
}};

bool parameter_type(torch::ScalarType dtype) {
  return dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
         dtype == torch::kBFloat16;
}

uint64_t capacity_elements(const InputField& field,
                           const SlotBufferCapacity& capacity) {
  switch (field.domain) {
    case RowDomain::SELECTED:
      return capacity.max_selected_rows;
    case RowDomain::SAMPLE:
      return capacity.max_sample_rows;
    case RowDomain::HISTORY:
      return static_cast<uint64_t>(capacity.max_selected_rows) *
             capacity.max_unique_tokens;
  }
  LOG(FATAL) << "Unknown sampling row domain.";
  return 0;
}

torch::ScalarType storage_dtype(const InputField& field,
                                torch::ScalarType parameter_dtype) {
  return field.dtype == torch::kFloat32 ? parameter_dtype : field.dtype;
}

template <typename Scalar, typename Predicate>
bool all_values(const torch::Tensor& tensor, Predicate predicate) {
  const Scalar* values = tensor.data_ptr<Scalar>();
  for (int64_t index = 0; index < tensor.numel(); ++index) {
    if (!predicate(values[index])) {
      return false;
    }
  }
  return true;
}

template <typename Predicate>
bool valid_parameters(const torch::Tensor& tensor,
                      torch::ScalarType destination,
                      Predicate predicate) {
  if (!tensor.defined()) {
    return true;
  }
  const auto valid = [destination, predicate](float value) {
    if (!std::isfinite(value) || !predicate(value)) {
      return false;
    }
    if (destination == torch::kFloat16) {
      return std::isfinite(static_cast<float>(static_cast<c10::Half>(value)));
    }
    if (destination == torch::kBFloat16) {
      return std::isfinite(
          static_cast<float>(static_cast<c10::BFloat16>(value)));
    }
    return true;
  };
  switch (tensor.scalar_type()) {
    case torch::kFloat32:
      return all_values<float>(tensor, valid);
    case torch::kFloat16:
      return all_values<c10::Half>(tensor, valid);
    case torch::kBFloat16:
      return all_values<c10::BFloat16>(tensor, valid);
    default:
      return false;
  }
}

Status invalid_input() {
  return Status(StatusCode::INVALID_ARGUMENT,
                "Invalid or unsupported sampling input.");
}

Status validate_sampling_capacity(SlotBufferCapacity& capacity,
                                  uint64_t& bytes) {
  if (capacity.max_selected_rows == 0) {
    capacity.max_selected_rows = capacity.model.max_sequences;
  }
  if (capacity.max_sample_rows == 0) {
    capacity.max_sample_rows = capacity.model.max_sequences;
  }
  constexpr uint64_t kMaxElements = std::numeric_limits<int32_t>::max();
  if (!parameter_type(capacity.parameter_dtype) ||
      capacity.max_unique_tokens == 0 ||
      capacity.max_unique_tokens > kMaxElements || capacity.vocab_size == 0 ||
      capacity.vocab_size > kMaxElements ||
      capacity.max_top_logprobs > capacity.vocab_size ||
      capacity.max_selected_rows > kMaxElements ||
      capacity.max_sample_rows > capacity.max_selected_rows ||
      capacity.max_result_width > kMaxElements) {
    return invalid_input();
  }
  bytes = 0;
  constexpr uint64_t kMaxBytes = std::numeric_limits<int64_t>::max();
  for (const auto& field : kFields) {
    const uint64_t elements = capacity_elements(field, capacity);
    const uint64_t element_bytes =
        torch::elementSize(storage_dtype(field, capacity.parameter_dtype));
    if (elements > (kMaxBytes - bytes) / element_bytes) {
      return invalid_input();
    }
    bytes += elements * element_bytes;
  }
  // Check the token product before multiplying by the top-logprob width.
  const uint64_t tokens = static_cast<uint64_t>(capacity.max_sample_rows) *
                          capacity.max_result_width;
  if (capacity.max_top_logprobs != 0 &&
      tokens > kMaxBytes / capacity.max_top_logprobs) {
    return invalid_input();
  }
  for (const auto& field : kResultFields) {
    const uint64_t elements = result_elements(field, capacity);
    const uint64_t element_bytes = torch::elementSize(field.dtype);
    if (elements > (kMaxBytes - bytes) / element_bytes) {
      return invalid_input();
    }
    bytes += elements * element_bytes;
  }
  return Status();
}

bool overlaps_host(std::span<const int32_t> source,
                   const torch::Tensor& destination) {
  if (source.empty() || !destination.defined()) {
    return false;
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(source.data());
  const uintptr_t base = reinterpret_cast<uintptr_t>(destination.data_ptr());
  return start >= base ? start - base < destination.nbytes()
                       : base - start < source.size_bytes();
}

BatchInputMeta batch_input_meta(const ForwardInput& input) {
  BatchInputMeta batch = input.input_params.meta;
  // The builder leaves actual_num_sequences unset before Worker prepare.
  if (batch.actual_num_sequences == 0) {
    batch.actual_num_sequences = static_cast<int32_t>(
        input.input_params.attention.host.q_seq_lens.size());
  }
  // ProfileManager uses this as an output marker. Graphs are captured before
  // admission; the pipeline preserves the marker in its detached response.
  batch.is_graph_warmup = false;
  return batch;
}

void assign_prefix(std::vector<int32_t>& output,
                   const torch::Tensor& staging,
                   uint32_t count) {
  const int32_t* values = staging.data_ptr<int32_t>();
  output.assign(values, values + count);
}

int32_t maximum(const std::vector<int32_t>& values) {
  return values.empty() ? 0 : *std::max_element(values.begin(), values.end());
}

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}
constexpr uint64_t kModelInputAlignment = 16;
constexpr uint64_t kMaxTensorBytes = std::numeric_limits<int64_t>::max();

BatchInputMeta batch_meta(const ModelInputBatch& batch) {
  BatchInputMeta meta;
  meta.batch_forward_type = batch.forward_type;
  meta.actual_num_sequences = static_cast<int32_t>(batch.num_actual_sequences);
  meta.batch_id = batch.batch_id;
  meta.is_graph_warmup = batch.is_graph_warmup;
  return meta;
}

bool cpu_indices(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.device().is_cpu() &&
         tensor.scalar_type() == torch::kInt32 && tensor.dim() == 1 &&
         tensor.is_contiguous();
}

void copy_to_device(const torch::Tensor& destination,
                    const torch::Tensor& source,
                    const Stream& stream) {
  if (source.numel() == 0) {
    return;
  }
  auto stream_guard = stream.set_stream_guard();
  destination.copy_(source, /*non_blocking=*/true);
}

}  // namespace

// Called after validate_model; sizes and capacity are already checked.
Status SlotBuffer::validate_batch(const ModelInputHostView& input,
                                  const BatchInputMeta& batch) {
  const uint64_t rows = input.q_seq_lens.size();
  if (batch.actual_num_sequences < 0 ||
      static_cast<uint64_t>(batch.actual_num_sequences) > rows) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid physical or actual model input rows.");
  }
  switch (batch.batch_forward_type.value()) {
    case BatchForwardType::EMPTY:
      if (rows != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "Empty model input batch contains rows or tokens.");
      }
      return Status();
    case BatchForwardType::PREFILL:
    case BatchForwardType::CHUNKED_PREFILL:
    case BatchForwardType::DECODE:
    case BatchForwardType::MIXED:
      break;
    default:
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unknown model input forward type.");
  }
  // An empty logical shard inherits the active peer's forward type. Prepare
  // materializes its padding row after validation, without an actual sequence.
  for (int32_t row = 0; row < batch.actual_num_sequences; ++row) {
    if (input.q_seq_lens[row] <= 0 ||
        (batch.batch_forward_type.is_prefill() &&
         input.kv_seq_lens[row] != input.q_seq_lens[row])) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Invalid query or cache lengths for actual model rows.");
    }
  }
  if (batch.batch_forward_type.is_decode() &&
      std::any_of(input.q_seq_lens.begin(),
                  input.q_seq_lens.end(),
                  [](int32_t length) { return length > 1; })) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Decode model input rows must contain at most one token.");
  }
  return Status();
}

bool SlotBuffer::append_region(uint64_t elements,
                               Region& region,
                               uint64_t& total_bytes) {
  if (elements > kMaxTensorBytes / sizeof(int32_t)) {
    return false;
  }
  const uint64_t bytes = elements * sizeof(int32_t);
  const uint64_t padding =
      (kModelInputAlignment - bytes % kModelInputAlignment) %
      kModelInputAlignment;
  // total_bytes is aligned and bounded after each successful append. Check
  // both additions before performing them, including the final region's pad.
  if (bytes > kMaxTensorBytes - total_bytes ||
      padding > kMaxTensorBytes - total_bytes - bytes) {
    return false;
  }
  region = {total_bytes, bytes};
  total_bytes += bytes + padding;
  return true;
}

Status SlotBuffer::make_layout(const ModelInputCapacity& capacity,
                               Layout& layout) {
  if (capacity.max_tokens == 0 || capacity.max_sequences == 0 ||
      capacity.max_blocks_per_sequence == 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Model input capacities must all be positive.");
  }

  Layout planned;
  planned.capacity = capacity;
  planned.block_table_row_stride_bytes =
      static_cast<uint64_t>(capacity.max_blocks_per_sequence) * sizeof(int32_t);
  // Each dimension is uint32_t, so their product fits after widening both.
  const uint64_t block_elements =
      static_cast<uint64_t>(capacity.max_sequences) *
      static_cast<uint64_t>(capacity.max_blocks_per_sequence);
  if (!append_region(
          capacity.max_tokens, planned.token_ids, planned.total_bytes) ||
      !append_region(
          capacity.max_tokens, planned.positions, planned.total_bytes) ||
      !append_region(
          capacity.max_tokens, planned.new_cache_slots, planned.total_bytes) ||
      !append_region(
          capacity.max_sequences, planned.q_seq_lens, planned.total_bytes) ||
      !append_region(
          capacity.max_sequences, planned.kv_seq_lens, planned.total_bytes) ||
      !append_region(
          capacity.max_sequences, planned.q_cu_seq_lens, planned.total_bytes) ||
      !append_region(
          block_elements, planned.block_tables, planned.total_bytes)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Model input layout exceeds signed 64-bit tensor capacity.");
  }
  layout = planned;
  return Status();
}

torch::Tensor SlotBuffer::field_view(const torch::Tensor& buffer,
                                     const Region& region) {
  return buffer.narrow(/*dim=*/0,
                       static_cast<int64_t>(region.offset / sizeof(int32_t)),
                       static_cast<int64_t>(region.bytes / sizeof(int32_t)));
}

SlotBuffer::ModelTensors SlotBuffer::bind_views(const torch::Tensor& buffer,
                                                const Layout& layout) {
  ModelTensors views;
  views.token_ids = field_view(buffer, layout.token_ids);
  views.positions = field_view(buffer, layout.positions);
  views.new_cache_slots = field_view(buffer, layout.new_cache_slots);
  views.q_seq_lens = field_view(buffer, layout.q_seq_lens);
  views.kv_seq_lens = field_view(buffer, layout.kv_seq_lens);
  views.q_cu_seq_lens = field_view(buffer, layout.q_cu_seq_lens);
  views.block_tables = field_view(buffer, layout.block_tables)
                           .view({layout.capacity.max_sequences,
                                  layout.capacity.max_blocks_per_sequence});
  return views;
}

Status SlotBuffer::create(const SlotBufferCapacity& capacity,
                          const torch::Device& device,
                          std::unique_ptr<SlotBuffer>& output) {
  Layout layout;
  Status status = make_layout(capacity.model, layout);
  if (!status.ok()) {
    return status;
  }
  if (device.type() != Platform::type_torch() || !device.has_index() ||
      capacity.model.max_sequences > std::numeric_limits<int32_t>::max() ||
      (capacity.max_unique_tokens == 0) != (capacity.vocab_size == 0)) {
    return invalid_input();
  }
  // Validate every allocation, including sampling history, before allocating
  // even the model region. Rejected capacities must not trigger huge staging
  // allocations before their overflow is reported.
  if (capacity.max_unique_tokens != 0) {
    SlotBufferCapacity checked = capacity;
    uint64_t bytes = 0;
    status = validate_sampling_capacity(checked, bytes);
    if (!status.ok()) {
      return status;
    }
  }
  c10::DeviceGuard guard(device);
  auto input =
      std::unique_ptr<SlotBuffer>(new SlotBuffer(capacity, device, layout, 0));
  if (capacity.max_unique_tokens != 0) {
    status = input->configure_sampling(capacity);
    if (!status.ok()) {
      return status;
    }
    // Ordinary overlap uses row references into the preceding task's result.
    const uint32_t rows = capacity.model.max_sequences;
    const auto options =
        torch::TensorOptions().device(device).dtype(torch::kInt64);
    input->host_indices_ = torch::empty(
        {2, rows}, options.device(torch::kCPU).pinned_memory(true));
    input->device_indices_ = torch::empty({2, rows}, options);
    input->gathered_int64_ = torch::empty({rows}, options);
    input->gathered_int32_ = torch::empty({rows}, options.dtype(torch::kInt32));
  }
  output = std::move(input);
  return Status();
}

Status SlotBuffer::configure_sampling(const SlotBufferCapacity& requested) {
  if (sampling_host_.selected_token_idxes.defined() || copy_submitted_) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Sampling storage is already configured.");
  }
  SlotBufferCapacity capacity = requested;
  capacity.model = capacity_.model;
  capacity.enable_mla = capacity_.enable_mla;
  uint64_t bytes = 0;
  Status status = validate_sampling_capacity(capacity, bytes);
  if (!status.ok()) {
    return status;
  }
  c10::DeviceGuard guard(device_);
  capacity_ = capacity;
  auxiliary_bytes_ = bytes;
  for (const auto& field : kFields) {
    const int64_t elements =
        static_cast<int64_t>(capacity_elements(field, capacity_));
    const auto options = torch::TensorOptions().dtype(
        storage_dtype(field, capacity_.parameter_dtype));
    sampling_host_.*field.member = torch::empty(
        {elements}, options.device(torch::kCPU).pinned_memory(true));
    sampling_device_.*field.member =
        torch::empty({elements}, options.device(device_));
  }
  for (const auto& field : kResultFields) {
    const int64_t elements =
        static_cast<int64_t>(result_elements(field, capacity_));
    if (elements == 0) {
      continue;
    }
    const auto options = torch::TensorOptions().dtype(field.dtype);
    result_host_storage_.*field.member = torch::empty(
        {elements}, options.device(torch::kCPU).pinned_memory(true));
    result_device_storage_.*field.member =
        torch::empty({elements}, options.device(device_));
  }
  if (capacity_.max_result_width != 0) {
    result_ready_ = std::make_unique<StreamEvent>(device_.type());
  }
  return Status();
}

SlotBuffer::SlotBuffer(SlotBufferCapacity capacity,
                       torch::Device device,
                       Layout layout,
                       uint64_t auxiliary_bytes)
    : capacity_(std::move(capacity)),
      device_(std::move(device)),
      layout_(std::move(layout)),
      auxiliary_bytes_(auxiliary_bytes) {
  if (layout_.total_bytes == 0) {
    return;
  }
  const int64_t elements =
      static_cast<int64_t>(layout_.total_bytes / sizeof(int32_t));
  host_buffer_ = torch::empty({elements},
                              torch::TensorOptions()
                                  .dtype(torch::kInt32)
                                  .device(torch::kCPU)
                                  .pinned_memory(true));
  device_buffer_ = torch::empty(
      {elements}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
  CHECK_EQ(reinterpret_cast<uintptr_t>(host_buffer_.data_ptr()) %
               kModelInputAlignment,
           0);
  CHECK_EQ(reinterpret_cast<uintptr_t>(device_buffer_.data_ptr()) %
               kModelInputAlignment,
           0);
  model_host_ = bind_views(host_buffer_, layout_);
  model_device_ = bind_views(device_buffer_, layout_);

  AttentionHostInput& host = model_params_.attention.host;
  for (std::vector<int32_t>* lengths : {&host.q_seq_lens,
                                        &host.q_cu_seq_lens,
                                        &host.kv_seq_lens,
                                        &host.kv_cache_tokens_nums}) {
    lengths->reserve(capacity_.model.max_sequences);
  }
  host.new_cache_slots.reserve(capacity_.model.max_tokens);
  model_params_.attn_metadata = std::make_shared<layer::AttentionMetadata>();
  auto& metadata = *model_params_.attn_metadata;
  metadata.q_seq_lens_vec.reserve(capacity_.model.max_sequences);
  metadata.kv_seq_lens_vec.reserve(capacity_.model.max_sequences);
  metadata.q_cu_seq_lens_host_vec.reserve(capacity_.model.max_sequences);
}

Status SlotBuffer::validate_model(const ModelInputHostView& input) const {
  const ModelInputCapacity& capacity = capacity_.model;
  const uint64_t tokens = input.token_ids.size();
  const uint64_t rows = input.q_seq_lens.size();
  if (tokens > capacity.max_tokens || rows > capacity.max_sequences ||
      tokens > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      input.block_table_width > capacity.max_blocks_per_sequence ||
      input.positions.size() != tokens ||
      input.new_cache_slots.size() != tokens ||
      input.kv_seq_lens.size() != rows || input.q_cu_seq_lens.size() != rows ||
      input.block_tables.size() != rows * input.block_table_width ||
      (rows == 0 && (tokens != 0 || input.block_table_width != 0)) ||
      (rows != 0 && (tokens == 0 || input.block_table_width == 0))) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid model input sizes or capacity.");
  }
  int64_t cumulative = 0;
  for (uint64_t row = 0; row < rows; ++row) {
    cumulative += input.q_seq_lens[row];
    if (input.q_seq_lens[row] < 0 ||
        input.kv_seq_lens[row] < input.q_seq_lens[row] ||
        cumulative != input.q_cu_seq_lens[row]) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Inconsistent model input sequence lengths.");
    }
  }
  if (static_cast<uint64_t>(cumulative) != tokens) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Query lengths do not cover input tokens.");
  }
  for (const auto source : {input.token_ids,
                            input.positions,
                            input.new_cache_slots,
                            input.q_seq_lens,
                            input.kv_seq_lens,
                            input.q_cu_seq_lens,
                            input.block_tables}) {
    if (overlaps_host(source, host_buffer_)) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Input aliases destination staging storage.");
    }
  }
  return Status();
}

Status SlotBuffer::validate_sampling(const SamplingParameters& input,
                                     uint32_t model_tokens) const {
  if (input.filter_mask.defined() || input.filter_bitmask.defined() ||
      input.acc_logprob.defined() || input.is_embeddings ||
      input.use_beam_search || input.num_return_sequences != 0 ||
      input.max_top_logprobs < 0 ||
      model_tokens >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      static_cast<uint64_t>(input.max_top_logprobs) >
          capacity_.max_top_logprobs) {
    return invalid_input();
  }
  const int64_t rows = input.selected_token_idxes.defined()
                           ? input.selected_token_idxes.numel()
                           : 0;
  const int64_t samples =
      input.sample_idxes.defined() ? input.sample_idxes.numel() : 0;
  if (!sampling_host_.selected_token_idxes.defined() ||
      rows > capacity_.max_selected_rows ||
      samples > capacity_.max_sample_rows || samples > rows ||
      (rows > 0 && (samples == 0 || model_tokens == 0))) {
    return invalid_input();
  }
  int64_t width = 0;
  for (const auto& field : kFields) {
    const torch::Tensor& tensor = input.*field.member;
    if (!tensor.defined()) {
      continue;
    }
    const bool dtype_matches = field.dtype == torch::kFloat32
                                   ? parameter_type(tensor.scalar_type())
                                   : tensor.scalar_type() == field.dtype;
    const int64_t expected_rows =
        field.domain == RowDomain::SAMPLE ? samples : rows;
    const bool history = field.domain == RowDomain::HISTORY;
    if (!tensor.device().is_cpu() || !tensor.is_contiguous() ||
        !dtype_matches || tensor.dim() != (history ? 2 : 1) ||
        tensor.size(0) != expected_rows) {
      return invalid_input();
    }
    if (history) {
      if (tensor.size(1) > capacity_.max_unique_tokens ||
          (width != 0 && width != tensor.size(1)) ||
          (rows > 0 && tensor.size(1) == 0)) {
        return invalid_input();
      }
      width = tensor.size(1);
    }
  }
  const bool history = input.unique_token_ids.defined();
  if (history != input.unique_token_counts.defined() ||
      history != input.unique_token_ids_lens.defined() ||
      input.frequency_penalties.defined() !=
          input.presence_penalties.defined() ||
      ((input.frequency_penalties.defined() ||
        input.repetition_penalties.defined()) &&
       !history) ||
      (samples > 0 && !input.do_sample.defined())) {
    return invalid_input();
  }
  if (rows > 0) {
    if (!all_values<int32_t>(
            input.selected_token_idxes, [model_tokens](int32_t value) {
              return value >= 0 && static_cast<uint32_t>(value) < model_tokens;
            })) {
      return invalid_input();
    }
    int32_t previous = -1;
    if (!all_values<int32_t>(
            input.sample_idxes, [rows, &previous](int32_t value) {
              const bool valid = value > previous && value < rows;
              previous = value;
              return valid;
            })) {
      return invalid_input();
    }
  }
  const auto finite = [](float /*value*/) { return true; };
  if (!valid_parameters(
          input.frequency_penalties, capacity_.parameter_dtype, finite) ||
      !valid_parameters(
          input.presence_penalties, capacity_.parameter_dtype, finite) ||
      !valid_parameters(input.repetition_penalties,
                        capacity_.parameter_dtype,
                        [](float value) { return value > 0; }) ||
      !valid_parameters(input.temperatures,
                        capacity_.parameter_dtype,
                        [](float value) { return value >= 0; }) ||
      !valid_parameters(input.top_p,
                        capacity_.parameter_dtype,
                        [](float value) { return value >= 0 && value <= 1; })) {
    return invalid_input();
  }
  if (input.top_k.defined() &&
      !all_values<int64_t>(input.top_k, [](int64_t value) {
        return value <= std::numeric_limits<int32_t>::max();
      })) {
    return invalid_input();
  }
  if (history &&
      (!all_values<int64_t>(input.unique_token_ids,
                            [this](int64_t value) {
                              return value >= 0 &&
                                     static_cast<uint64_t>(value) <
                                         capacity_.vocab_size;
                            }) ||
       !all_values<int32_t>(input.unique_token_counts,
                            [](int32_t value) { return value >= 0; }) ||
       !all_values<int32_t>(
           input.unique_token_ids_lens,
           [width](int32_t value) { return value >= 0 && value <= width; }))) {
    return invalid_input();
  }
  return Status();
}
Status SlotBuffer::validate_previous_tokens(const ModelInputHostView& model,
                                            uint32_t previous_rows) const {
  if (previous_rows > capacity_.model.max_sequences) {
    return invalid("Previous token rows exceed Slot capacity.");
  }
  uint32_t begin = 0;
  for (const int32_t end : model.q_cu_seq_lens) {
    for (uint32_t offset = begin; offset < static_cast<uint32_t>(end);
         ++offset) {
      const int32_t token = model.token_ids[offset];
      if (token >= 0) {
        continue;
      }
      const int64_t source = -static_cast<int64_t>(token) - 1;
      if (offset + 1 != static_cast<uint32_t>(end) || source >= previous_rows) {
        return invalid(
            "Unknown last query token must name a previous output row.");
      }
    }
    begin = static_cast<uint32_t>(end);
  }
  return Status();
}

ModelInputHostView SlotBuffer::model_input_view(const ForwardInput& input) {
  const auto& host = input.input_params.attention.host;
  const auto tokens = int_span(input.host_token_ids());
  return {tokens,
          int_span(input.host_positions()),
          host.new_cache_slots.empty()
              ? int_span(input.input_params.attention.device.new_cache_slots)
              : std::span<const int32_t>(host.new_cache_slots),
          host.q_seq_lens,
          host.kv_seq_lens,
          host.q_cu_seq_lens,
          int_span(host.block_tables),
          tokens.empty()
              ? 0
              : static_cast<uint32_t>(host.block_tables.size(/*dim=*/1))};
}

Status SlotBuffer::validate(const ForwardInput& input,
                            uint32_t previous_rows,
                            const Stream& stream) const {
  if (copy_submitted_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Take or discard the pending result before Slot reuse.");
  }
  if (stream.get_stream()->device_index() != device_.index()) {
    return invalid("Input stream and Slot device differ.");
  }
  const ModelInputHostView model = model_input_view(input);
  Status status = validate_model(model);
  if (!status.ok()) {
    return status;
  }
  status = validate_batch(model, batch_input_meta(input));
  if (!status.ok()) {
    return status;
  }
  status = validate_sampling(input.sampling_params, model.token_ids.size());
  if (!status.ok()) {
    return status;
  }
  return validate_previous_tokens(model, previous_rows);
}

void SlotBuffer::prepare(const ForwardInput& input,
                         const Stream& stream,
                         uint32_t padded_batch_size) {
  CHECK(!copy_submitted_) << "Pending result prevents Slot reuse.";
  const ModelInputHostView model = model_input_view(input);
  BatchInputMeta batch = batch_input_meta(input);
  if (padded_batch_size != 0) {
    CHECK_LE(padded_batch_size, capacity_.model.max_sequences);
    CHECK_LE(padded_batch_size, capacity_.model.max_tokens);
    CHECK_GE(padded_batch_size, model.token_ids.size());
    CHECK_EQ(model.token_ids.size(), model.q_seq_lens.size());
    CHECK(model.token_ids.empty() || batch.batch_forward_type.is_decode());
    batch.batch_forward_type = BatchForwardType::DECODE;
  }
  // The caller validated all inputs before this first staging write.
  auto guard = stream.set_stream_guard();
  prepare_previous_tokens(model);
  if (padded_batch_size != 0) {
    prepare_model(model, batch, padded_batch_size);
  } else if (model.token_ids.empty() && !batch.batch_forward_type.is_empty()) {
    const std::array<int32_t, 1> zero{0};
    const std::array<int32_t, 1> one{1};
    // Block zero is reserved for padding by the block manager. The inherited
    // forward type joins peer collectives while actual_num_sequences stays
    // zero.
    prepare_model({one, zero, zero, one, one, one, zero, 1}, batch);
  } else {
    prepare_model(model, batch);
  }
  model_params_.enable_graph = padded_batch_size != 0;
  // Graph collectives use common physical rows; keep logical counts separate.
  auto& target = model_params_.parallel;
  const auto& parallel = input.input_params.parallel;
  target.dp_global_token_nums = parallel.dp_global_token_nums;
  target.dp_is_decode = parallel.dp_is_decode;
  target.raw_dp_global_token_nums.clear();
  if (padded_batch_size != 0 && !target.dp_global_token_nums.empty()) {
    target.raw_dp_global_token_nums = target.dp_global_token_nums;
    std::fill(target.dp_global_token_nums.begin(),
              target.dp_global_token_nums.end(),
              padded_batch_size);
    std::fill(target.dp_is_decode.begin(), target.dp_is_decode.end(), 1);
  }
  prepare_sampling(input.sampling_params);
  prepare_result();
}

void SlotBuffer::prepare_model(const ModelInputHostView& input,
                               const BatchInputMeta& batch,
                               uint32_t padded_batch_size) {
  const Layout& layout = layout_;
  const uint32_t actual_rows = static_cast<uint32_t>(input.q_seq_lens.size());
  const uint32_t token_count =
      padded_batch_size != 0 ? padded_batch_size
                             : static_cast<uint32_t>(input.token_ids.size());
  const uint32_t rows =
      padded_batch_size != 0 ? padded_batch_size : actual_rows;
  int32_t* host_data = host_buffer_.data_ptr<int32_t>();
  const std::array<std::span<const int32_t>, 6> sources = {
      input.token_ids,
      input.positions,
      input.new_cache_slots,
      input.q_seq_lens,
      input.kv_seq_lens,
      input.q_cu_seq_lens};
  const std::array<Region, 6> regions = {layout.token_ids,
                                         layout.positions,
                                         layout.new_cache_slots,
                                         layout.q_seq_lens,
                                         layout.kv_seq_lens,
                                         layout.q_cu_seq_lens};
  for (uint32_t i = 0; i < sources.size(); ++i) {
    const uint32_t count = i < 3 ? token_count : rows;
    if (count == 0) {
      continue;
    }
    const uint64_t offset = regions[i].offset / sizeof(int32_t);
    if (padded_batch_size != 0) {
      // Padding uses reserved block zero and never writes live KV slots.
      const int32_t padding = i == 2 ? -1 : (i == 1 ? 0 : 1);
      std::fill_n(host_data + offset, count, padding);
      if (i == 5) {
        std::iota(host_data + offset, host_data + offset + count, 1);
      }
    }
    if (!sources[i].empty()) {
      std::memcpy(
          host_data + offset, sources[i].data(), sources[i].size_bytes());
    }
    device_buffer_.narrow(/*dim=*/0, offset, count)
        .copy_(host_buffer_.narrow(/*dim=*/0, offset, count),
               /*non_blocking=*/true);
  }
  if (rows != 0) {
    const uint64_t offset = layout.block_tables.offset / sizeof(int32_t);
    const uint64_t stride = layout.block_table_row_stride_bytes;
    const uint64_t width =
        static_cast<uint64_t>(input.block_table_width) * sizeof(int32_t);
    const uint64_t bytes = stride * rows;
    if (width == stride && actual_rows == rows) {
      std::memcpy(host_data + offset, input.block_tables.data(), bytes);
    } else {
      std::memset(host_data + offset, 0, bytes);
      for (uint32_t row = 0; row < actual_rows; ++row) {
        std::memcpy(host_data + offset +
                        static_cast<uint64_t>(row) *
                            layout.capacity.max_blocks_per_sequence,
                    input.block_tables.data() +
                        static_cast<uint64_t>(row) * input.block_table_width,
                    width);
      }
    }
    // Host staging already includes zero padding. Copy complete contiguous
    // rows so the device keeps the fixed stride without a separate memset.
    model_device_.block_tables.narrow(/*dim=*/0, /*start=*/0, rows)
        .copy_(model_host_.block_tables.narrow(/*dim=*/0, /*start=*/0, rows),
               /*non_blocking=*/true);
  }
  model_params_.python_attention_metadata.reset();
  const ModelTensors& staging = model_host_;
  AttentionHostInput& host = model_params_.attention.host;
  // Read our staging only: input spans may borrow the previous Host metadata.
  assign_prefix(host.q_seq_lens, staging.q_seq_lens, rows);
  assign_prefix(host.q_cu_seq_lens, staging.q_cu_seq_lens, rows);
  assign_prefix(host.kv_seq_lens, staging.kv_seq_lens, rows);
  assign_prefix(host.new_cache_slots, staging.new_cache_slots, token_count);
  host.kv_cache_tokens_nums.resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    host.kv_cache_tokens_nums[row] =
        host.kv_seq_lens[row] - host.q_seq_lens[row];
  }
  host.block_tables = staging.block_tables.narrow(/*dim=*/0, /*start=*/0, rows);
  host.graph_q_seq_lens_data = staging.q_seq_lens.data_ptr<int32_t>();
  host.graph_kv_seq_lens_data = staging.kv_seq_lens.data_ptr<int32_t>();

  const ModelTensors& device = model_device_;
  tokens_ = device.token_ids.narrow(/*dim=*/0, /*start=*/0, token_count);
  positions_ = device.positions.narrow(/*dim=*/0, /*start=*/0, token_count);
  AttentionDeviceInput& attention = model_params_.attention.device;
  attention.q_seq_lens = device.q_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  attention.kv_seq_lens =
      device.kv_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  attention.q_cu_seq_lens =
      device.q_cu_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  attention.new_cache_slots =
      device.new_cache_slots.narrow(/*dim=*/0, /*start=*/0, token_count);
  attention.block_tables =
      device.block_tables.narrow(/*dim=*/0, /*start=*/0, rows);

  model_params_.meta.batch_forward_type = batch.batch_forward_type;
  model_params_.meta.num_sequences = static_cast<int32_t>(rows);
  model_params_.meta.actual_num_sequences = batch.actual_num_sequences;
  model_params_.meta.batch_id = batch.batch_id;
  model_params_.meta.is_graph_warmup = batch.is_graph_warmup;
  model_params_.meta.q_max_seq_len = maximum(host.q_seq_lens);
  model_params_.meta.kv_max_seq_len = maximum(host.kv_seq_lens);
  // These are the original Python paged-attention inputs. The Slot owns the
  // tensor views and Host metadata until its last reader retires. No model
  // invocation, rotary computation or attention kernel is created here.
  auto& metadata = *model_params_.attn_metadata;
  metadata.q_seq_lens = attention.q_seq_lens;
  metadata.kv_seq_lens = attention.kv_seq_lens;
  metadata.q_cu_seq_lens = attention.q_cu_seq_lens;
  metadata.qo_indptr = attention.q_cu_seq_lens;
  metadata.slot_mapping = attention.new_cache_slots;
  metadata.block_table =
      batch.batch_forward_type.is_prefill() && !capacity_.enable_mla
          ? torch::Tensor()
          : attention.block_tables;
  metadata.q_seq_lens_vec.assign(host.q_seq_lens.begin(),
                                 host.q_seq_lens.end());
  metadata.kv_seq_lens_vec.assign(host.kv_seq_lens.begin(),
                                  host.kv_seq_lens.end());
  metadata.q_cu_seq_lens_host_vec.assign(host.q_cu_seq_lens.begin(),
                                         host.q_cu_seq_lens.end());
  metadata.q_seq_lens_host =
      staging.q_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  metadata.kv_seq_lens_host =
      staging.kv_seq_lens.narrow(/*dim=*/0, /*start=*/0, rows);
  metadata.max_query_len = model_params_.meta.q_max_seq_len;
  metadata.max_seq_len = model_params_.meta.kv_max_seq_len;
  metadata.total_kv_len = std::accumulate(
      host.kv_seq_lens.begin(), host.kv_seq_lens.end(), int64_t{0});
  metadata.is_prefill = batch.batch_forward_type.is_prefill();
  metadata.is_chunked_prefill = batch.batch_forward_type.is_chunked_prefill() ||
                                batch.batch_forward_type.is_mixed();
  metadata.is_mixed = batch.batch_forward_type.is_mixed();
  metadata.is_dummy = rows == 0;
}

Status SlotBuffer::prepare_sampling(const SamplingParameters& input,
                                    uint32_t model_tokens,
                                    const Stream& stream) {
  Status status = validate_sampling(input, model_tokens);
  if (!status.ok()) {
    return status;
  }
  if (stream.get_stream()->device_index() != device_.index()) {
    return invalid_input();
  }
  auto guard = stream.set_stream_guard();
  prepare_sampling(input);
  return Status();
}

void SlotBuffer::prepare_sampling(const SamplingParameters& input) {
  SamplingParameters prepared;
  const bool empty = !input.selected_token_idxes.defined() ||
                     input.selected_token_idxes.numel() == 0;
  if (!empty) {
    for (const auto& field : kFields) {
      const torch::Tensor& source = input.*field.member;
      if (!source.defined()) {
        continue;
      }
      torch::Tensor host = (sampling_host_.*field.member)
                               .narrow(/*dim=*/0, /*start=*/0, source.numel());
      torch::Tensor device =
          (sampling_device_.*field.member)
              .narrow(/*dim=*/0, /*start=*/0, source.numel());
      host.copy_(source.view({-1}));
      device.copy_(host, /*non_blocking=*/true);
      prepared.*field.member = device.view(source.sizes());
    }
    prepared.all_random_sample =
        all_values<bool>(input.do_sample, [](bool value) { return value; });
    prepared.all_greedy_sample =
        all_values<bool>(input.do_sample, [](bool value) { return !value; });
    prepared.logprobs = input.logprobs;
    prepared.return_probs = input.return_probs;
    prepared.max_top_logprobs = input.max_top_logprobs;
  }
  cpu_do_sample_ = input.do_sample.defined()
                       ? sampling_host_.do_sample.narrow(
                             /*dim=*/0, /*start=*/0, input.do_sample.numel())
                       : torch::Tensor();
  sampling_params_ = std::move(prepared);
}

void SlotBuffer::prepare_previous_tokens(const ModelInputHostView& model) {
  const uint32_t capacity = capacity_.model.max_sequences;
  gather_count_ = 0;
  int64_t* host = host_indices_.data_ptr<int64_t>();
  for (uint32_t offset = 0; offset < model.token_ids.size(); ++offset) {
    const int32_t token = model.token_ids[offset];
    if (token >= 0) {
      continue;
    }
    host[gather_count_] = -static_cast<int64_t>(token) - 1;
    host[capacity + gather_count_] = offset;
    ++gather_count_;
  }
  if (gather_count_ != 0) {
    for (uint32_t region = 0; region < 2; ++region) {
      device_indices_.select(/*dim=*/0, region)
          .narrow(/*dim=*/0, /*start=*/0, gather_count_)
          .copy_(host_indices_.select(/*dim=*/0, region)
                     .narrow(/*dim=*/0, /*start=*/0, gather_count_),
                 /*non_blocking=*/true);
    }
  }
  gather_rows_ = device_indices_.select(/*dim=*/0, /*index=*/0)
                     .narrow(/*dim=*/0, /*start=*/0, gather_count_);
  gather_offsets_ = device_indices_.select(/*dim=*/0, /*index=*/1)
                        .narrow(/*dim=*/0, /*start=*/0, gather_count_);
  gather_output_int64_ =
      gathered_int64_.narrow(/*dim=*/0, /*start=*/0, gather_count_);
  gather_output_int32_ =
      gathered_int32_.narrow(/*dim=*/0, /*start=*/0, gather_count_);
}

void SlotBuffer::patch_previous_tokens(const torch::Tensor& previous_tokens) {
  const torch::Tensor& model_tokens = tokens_;

  if (!has_previous_tokens()) {
    return;
  }
  CHECK_EQ(previous_tokens.device(), device_indices_.device());
  CHECK_EQ(previous_tokens.scalar_type(), torch::kInt64);
  CHECK_EQ(previous_tokens.dim(), 1);
  CHECK_EQ(previous_tokens.numel(), capacity_.model.max_sequences);
  CHECK_EQ(model_tokens.device(), previous_tokens.device());
  CHECK_EQ(model_tokens.scalar_type(), torch::kInt32);
  CHECK_EQ(model_tokens.dim(), 1);
  CHECK(model_tokens.is_contiguous());
  torch::index_select_out(
      gather_output_int64_, previous_tokens, /*dim=*/0, gather_rows_);
  gather_output_int32_.copy_(gather_output_int64_);
  model_tokens.index_copy_(/*dim=*/0, gather_offsets_, gather_output_int32_);
}

torch::Tensor SlotBuffer::copy_cpu_do_sample() const {
  if (!cpu_do_sample_.defined()) {
    return {};
  }
  auto output = torch::empty(cpu_do_sample_.sizes(),
                             torch::TensorOptions()
                                 .dtype(torch::kBool)
                                 .device(torch::kCPU)
                                 .pinned_memory(/*pinned_memory=*/false));
  output.copy_(cpu_do_sample_);
  return output;
}

SlotBuffer::~SlotBuffer() {
  c10::DeviceGuard guard(device_);
  discard_result();
  result_ready_.reset();
}

void SlotBuffer::prepare_result() {
  const uint32_t rows =
      sampling_params_.sample_idxes.defined()
          ? static_cast<uint32_t>(sampling_params_.sample_idxes.numel())
          : 0;
  const uint32_t top_logprobs =
      sampling_params_.logprobs ? sampling_params_.max_top_logprobs : 0;
  result_host_ = bind_result_views(result_host_storage_,
                                   rows,
                                   1,
                                   top_logprobs,
                                   sampling_params_.logprobs,
                                   false);
  result_device_ = bind_result_views(result_device_storage_,
                                     rows,
                                     1,
                                     top_logprobs,
                                     sampling_params_.logprobs,
                                     false);
  result_sequences_ = rows;
  result_width_ = 1;
  result_top_logprobs_ = top_logprobs;
}

Status SlotBuffer::bind_result(uint32_t sequences,
                               uint32_t tokens_per_sequence,
                               uint32_t top_logprobs,
                               bool logprobs) {
  if (!result_ready_ || sequences > capacity_.max_sample_rows ||
      tokens_per_sequence > capacity_.max_result_width ||
      top_logprobs > capacity_.max_top_logprobs ||
      (!logprobs && top_logprobs != 0) ||
      (sequences == 0 &&
       (tokens_per_sequence != 0 || top_logprobs != 0 || logprobs)) ||
      (sequences != 0 && tokens_per_sequence == 0)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid token result shape or optional outputs.");
  }
  if (copy_submitted_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Take or discard the pending token result before reuse.");
  }
  result_host_ = bind_result_views(result_host_storage_,
                                   sequences,
                                   tokens_per_sequence,
                                   top_logprobs,
                                   logprobs,
                                   true);
  result_device_ = bind_result_views(result_device_storage_,
                                     sequences,
                                     tokens_per_sequence,
                                     top_logprobs,
                                     logprobs,
                                     true);
  result_sequences_ = sequences;
  result_width_ = tokens_per_sequence;
  result_top_logprobs_ = top_logprobs;
  return Status();
}

Status SlotBuffer::copy_result_to_host(const Stream& stream,
                                       const StreamEventPtr& producer_ready) {
  if (!result_ready_ || !producer_ready ||
      stream.get_stream()->device_index() != device_.index()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Token result copy requires a matching stream and producer event.");
  }
  if (copy_submitted_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "A result copy is already pending for this Slot.");
  }
  auto guard = stream.set_stream_guard();
  CHECK(stream.wait_event(producer_ready))
      << "Failed to wait for token result producer.";
  for (const auto& field : kResultFields) {
    const torch::Tensor& source = result_device_.*field.member;
    if (!source.defined()) {
      continue;
    }
    const torch::Tensor& destination = result_host_.*field.member;
    destination.copy_(source, /*non_blocking=*/true);
  }
  stream.record_event(*result_ready_);
  copy_submitted_ = true;
  return Status();
}

TokenResultTensors SlotBuffer::take_result() {
  CHECK(copy_submitted_) << "No token result copy has been submitted.";
  c10::DeviceGuard guard(device_);
  CHECK(result_ready_->synchronize()) << "Failed to wait for token result D2H.";
  TokenResultTensors result;
  const int32_t* lengths = result_host_.lengths.defined()
                               ? result_host_.lengths.const_data_ptr<int32_t>()
                               : nullptr;
  if (lengths != nullptr) {
    for (uint32_t row = 0; row < result_sequences_; ++row) {
      CHECK_GE(lengths[row], 0);
      CHECK_LE(static_cast<uint32_t>(lengths[row]), result_width_);
    }
  }
  for (const auto& field : kResultFields) {
    const torch::Tensor& source = result_host_.*field.member;
    if (!source.defined()) {
      continue;
    }
    torch::Tensor destination =
        torch::full(source.sizes(),
                    field.padding,
                    source.options().pinned_memory(/*pinned_memory=*/false));
    if (lengths == nullptr || field.domain == ResultDomain::SEQUENCE) {
      std::memcpy(destination.data_ptr(), source.data_ptr(), source.nbytes());
    } else {
      const uint64_t values_per_token =
          field.domain == ResultDomain::TOP_TOKEN ? result_top_logprobs_ : 1;
      const uint64_t token_bytes = values_per_token * source.element_size();
      const uint64_t row_bytes = token_bytes * result_width_;
      const char* from = static_cast<const char*>(source.data_ptr());
      char* to = static_cast<char*>(destination.data_ptr());
      for (uint32_t row = 0; row < result_sequences_; ++row) {
        std::memcpy(to + row * row_bytes,
                    from + row * row_bytes,
                    lengths[row] * token_bytes);
      }
    }
    result.*field.member = std::move(destination);
  }
  copy_submitted_ = false;
  return result;
}

void SlotBuffer::discard_result() {
  if (!copy_submitted_) {
    return;
  }
  c10::DeviceGuard guard(device_);
  CHECK(result_ready_->synchronize()) << "Failed to retire token result D2H.";
  copy_submitted_ = false;
}

uint64_t SlotBuffer::pinned_bytes() const {
  uint64_t bytes = (host_buffer_.defined() ? host_buffer_.nbytes() : 0) +
                   auxiliary_bytes_ +
                   (host_indices_.defined() ? host_indices_.nbytes() : 0);
  bytes += input_scratch_ ? input_scratch_->pinned_bytes_ : 0;
  return bytes;
}
uint64_t SlotBuffer::device_bytes() const {
  uint64_t bytes = (device_buffer_.defined() ? device_buffer_.nbytes() : 0) +
                   auxiliary_bytes_;
  for (const auto* tensor :
       {&device_indices_, &gathered_int64_, &gathered_int32_}) {
    bytes += tensor->defined() ? tensor->nbytes() : 0;
  }
  bytes += input_scratch_ ? input_scratch_->device_bytes_ : 0;
  return bytes;
}

Status SlotBuffer::create(const ModelInputCapacity& capacity,
                          bool enable_mla,
                          const torch::Device& device,
                          std::unique_ptr<SlotBuffer>& output) {
  return create(
      {capacity, 0, 0, 0, torch::kFloat32, enable_mla}, device, output);
}
Status SlotBuffer::validate(const ModelInputHostView& input,
                            const ModelInputBatch& batch) const {
  Status status = validate_model(input);
  if (!status.ok()) {
    return status;
  }
  return validate_batch(input, batch_meta(batch));
}
Status SlotBuffer::prepare(const ModelInputHostView& input,
                           const ModelInputBatch& batch,
                           const Stream& stream) {
  Status status = validate(input, batch);
  if (!status.ok()) {
    return status;
  }
  if (stream.get_stream()->device_index() != device_.index()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Speculative input stream/device mismatch.");
  }
  auto guard = stream.set_stream_guard();
  prepare_model(input, batch_meta(batch));
  model_params().enable_graph = false;
  return Status();
}
Status SlotBuffer::prepare_empty_shard(bool decode,
                                       uint64_t batch_id,
                                       const Stream& stream) {
  const std::array<int32_t, 1> zero{0};
  const std::array<int32_t, 1> one{1};
  return prepare(
      {one, zero, zero, one, one, one, zero, 1},
      {decode ? BatchForwardType::DECODE : BatchForwardType::CHUNKED_PREFILL,
       0,
       batch_id,
       false},
      stream);
}
Status SlotBuffer::prepare_decode_padded(const ModelInputHostView& input,
                                         const ModelInputBatch& batch,
                                         uint32_t padded_batch_size,
                                         const Stream& stream) {
  Status status = validate(input, batch);
  if (!status.ok()) {
    return status;
  }
  if (padded_batch_size == 0 || padded_batch_size < input.token_ids.size() ||
      padded_batch_size > capacity_.model.max_tokens ||
      padded_batch_size > capacity_.model.max_sequences ||
      (!batch.forward_type.is_decode() &&
       batch.forward_type.value() != BatchForwardType::EMPTY) ||
      stream.get_stream()->device_index() != device_.index()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid speculative graph batch or stream.");
  }
  auto guard = stream.set_stream_guard();
  auto meta = batch_meta(batch);
  meta.batch_forward_type = BatchForwardType::DECODE;
  prepare_model(input, meta, padded_batch_size);
  model_params().enable_graph = true;
  return Status();
}

Status SlotBuffer::create_mtp_input(const MtpInputSpec& spec,
                                    torch::ScalarType hidden_dtype,
                                    const torch::Device& device,
                                    std::unique_ptr<SlotBuffer>& output) {
  const bool prefill = spec.kind == MtpInvocationKind::PREFILL;
  if (spec.model.max_tokens == 0 || spec.model.max_sequences == 0 ||
      spec.model.max_blocks_per_sequence == 0 ||
      (spec.context_only && !prefill) ||
      (prefill &&
       (spec.hidden_size == 0 ||
        spec.hidden_size > std::numeric_limits<int32_t>::max() ||
        (hidden_dtype != torch::kFloat16 && hidden_dtype != torch::kBFloat16 &&
         hidden_dtype != torch::kFloat32) ||
        static_cast<uint64_t>(spec.model.max_sequences) * spec.hidden_size >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                torch::elementSize(hidden_dtype))) ||
      (!prefill &&
       (spec.block_size == 0 || spec.num_speculative_tokens == 0 ||
        spec.num_speculative_tokens >= std::numeric_limits<int32_t>::max() ||
        spec.block_size > std::numeric_limits<int32_t>::max() ||
        (spec.kind != MtpInvocationKind::DRAFT &&
         spec.kind != MtpInvocationKind::VALIDATE &&
         spec.kind != MtpInvocationKind::BLOCK_DRAFT) ||
        spec.mask_token_id < 0 ||
        spec.draft_step >= spec.num_speculative_tokens ||
        (spec.kind == MtpInvocationKind::VALIDATE && spec.draft_step != 0)))) {
    return invalid("Invalid fixed MTP model input specification.");
  }
  auto scratch = std::make_unique<SlotBuffer::InputScratch>();
  auto& view = *scratch;
  view.spec_ = spec;
  ModelInputCapacity capacity = spec.model;
  if (!prefill) {
    view.rows_per_sequence_ = spec.kind != MtpInvocationKind::DRAFT
                                  ? spec.num_speculative_tokens + 1
                                  : (spec.draft_step == 0 ? 2U : 1U);
    view.first_offset_ =
        spec.kind != MtpInvocationKind::DRAFT
            ? 0
            : (spec.draft_step == 0 ? -1
                                    : static_cast<int32_t>(spec.draft_step));
    const uint64_t rows = static_cast<uint64_t>(spec.model.max_sequences) *
                          view.rows_per_sequence_;
    if (rows > std::numeric_limits<int32_t>::max()) {
      return invalid("Expanded MTP rows exceed the model index range.");
    }
    capacity.max_tokens = static_cast<uint32_t>(rows);
    capacity.max_sequences = static_cast<uint32_t>(rows);
  }
  std::unique_ptr<SlotBuffer> input;
  if (spec.context_only) {
    if (device.type() != Platform::type_torch() || !device.has_index() ||
        capacity.max_sequences > std::numeric_limits<int32_t>::max()) {
      return invalid("Invalid context-only input device or row capacity.");
    }
    input.reset(new SlotBuffer(
        {capacity, 0, 0, 0, torch::kFloat32, spec.enable_mla}, device, {}, 0));
  } else {
    Status status =
        SlotBuffer::create(capacity, spec.enable_mla, device, input);
    if (!status.ok()) {
      return status;
    }
  }
  c10::DeviceGuard guard(device);
  const auto options = torch::TensorOptions().device(device);
  if (!spec.context_only) {
    view.tokens_.reserve(capacity.max_tokens);
  }
  if (prefill) {
    view.token_to_sequence_.reserve(capacity.max_tokens);
    view.published_mask_.resize(capacity.max_sequences);
    view.planned_rows_.reserve(capacity.max_sequences);
    view.published_rows_.reserve(capacity.max_sequences);
    view.sampled_token_rows_.reserve(capacity.max_sequences);
    const auto host = options.device(torch::kCPU).pinned_memory(true);
    view.host_indices_ =
        torch::empty({capacity.max_sequences}, host.dtype(torch::kInt64));
    view.device_indices_ =
        torch::empty({capacity.max_sequences}, options.dtype(torch::kInt64));
    view.host_lengths_ =
        torch::empty({2, capacity.max_sequences}, host.dtype(torch::kInt32));
    view.device_lengths_ =
        torch::empty({2, capacity.max_sequences}, options.dtype(torch::kInt32));
    view.bootstrap_hidden_storage_ =
        torch::empty({capacity.max_sequences, spec.hidden_size},
                     options.dtype(hidden_dtype));
    view.token_storage_ =
        torch::empty({capacity.max_sequences}, options.dtype(torch::kInt32));
  } else {
    for (auto* values : {&view.positions_,
                         &view.slots_,
                         &view.q_lengths_,
                         &view.kv_lengths_,
                         &view.query_ends_}) {
      values->reserve(capacity.max_tokens);
    }
    view.block_tables_.reserve(static_cast<uint64_t>(capacity.max_tokens) *
                               capacity.max_blocks_per_sequence);
    view.offsets_ =
        torch::arange(
            view.first_offset_,
            view.first_offset_ + static_cast<int32_t>(view.rows_per_sequence_),
            torch::TensorOptions().dtype(torch::kInt32))
            .to(device);
    view.cache_positions_storage_ =
        torch::empty({capacity.max_tokens}, options.dtype(torch::kInt64));
    view.block_indices_storage_ =
        torch::empty({capacity.max_tokens}, options.dtype(torch::kInt64));
    view.block_ids_storage_ =
        torch::empty({capacity.max_tokens}, options.dtype(torch::kInt32));
    view.cache_offsets_storage_ =
        torch::empty({capacity.max_tokens}, options.dtype(torch::kInt32));
    view.repair_scratch_storage_ =
        torch::empty({spec.model.max_sequences}, options.dtype(torch::kInt64));
  }
  for (const auto* tensor : {&view.host_indices_, &view.host_lengths_}) {
    view.pinned_bytes_ += tensor->defined() ? tensor->nbytes() : 0;
  }
  for (const auto* tensor : {&view.device_indices_,
                             &view.device_lengths_,
                             &view.bootstrap_hidden_storage_,
                             &view.token_storage_,
                             &view.offsets_,
                             &view.cache_positions_storage_,
                             &view.block_indices_storage_,
                             &view.block_ids_storage_,
                             &view.cache_offsets_storage_,
                             &view.repair_scratch_storage_}) {
    view.device_bytes_ += tensor->defined() ? tensor->nbytes() : 0;
  }
  input->input_scratch_ = std::move(scratch);
  output = std::move(input);
  return Status();
}

Status SlotBuffer::plan_mtp_decode(SlotBuffer& input,
                                   const ModelInputHostView& base,
                                   const ModelInputBatch& batch) {
  if (!input.input_scratch_ ||
      input.input_scratch_->spec_.kind == MtpInvocationKind::PREFILL) {
    return invalid("MTP decode requires an expanded model input.");
  }
  auto& view = *input.input_scratch_;
  Status status = input.validate(base, batch);
  if (!status.ok()) {
    return status;
  }
  const uint64_t count = base.q_seq_lens.size();
  if (!batch.forward_type.is_decode() || count == 0 ||
      count > view.spec_.model.max_sequences ||
      batch.num_actual_sequences != count || base.token_ids.size() != count ||
      std::any_of(base.q_seq_lens.begin(),
                  base.q_seq_lens.end(),
                  [](int32_t length) { return length != 1; })) {
    return invalid("MTP decode requires one unpadded base row per Sequence.");
  }
  const int64_t max_offset =
      static_cast<int64_t>(view.first_offset_) + view.rows_per_sequence_ - 1;
  const int64_t capacity_positions =
      static_cast<int64_t>(base.block_table_width) * view.spec_.block_size;
  for (uint64_t row = 0; row < count; ++row) {
    const int64_t first_position =
        static_cast<int64_t>(base.positions[row]) + view.first_offset_;
    const int64_t last_position =
        static_cast<int64_t>(base.positions[row]) + max_offset;
    const int64_t first_kv_length =
        static_cast<int64_t>(base.kv_seq_lens[row]) + view.first_offset_;
    const int64_t last_kv_length =
        static_cast<int64_t>(base.kv_seq_lens[row]) + max_offset;
    const int64_t repair_position =
        static_cast<int64_t>(base.positions[row]) + 1;
    if (first_position < 0 || first_kv_length <= 0 ||
        last_position >= capacity_positions ||
        last_kv_length > capacity_positions ||
        last_position > std::numeric_limits<int32_t>::max() ||
        last_kv_length > std::numeric_limits<int32_t>::max() ||
        (view.first_offset_ == -1 &&
         (repair_position >= capacity_positions ||
          repair_position > std::numeric_limits<int32_t>::max()))) {
      return invalid(
          "MTP invocation exceeds its reserved position or KV capacity.");
    }
  }
  if (std::any_of(base.block_tables.begin(),
                  base.block_tables.end(),
                  [&view](int32_t block) {
                    return block < 0 ||
                           static_cast<int64_t>(block) * view.spec_.block_size +
                                   view.spec_.block_size - 1 >
                               std::numeric_limits<int32_t>::max();
                  })) {
    return invalid("MTP cache block cannot be represented by int32 slots.");
  }
  for (auto* values : {&view.tokens_,
                       &view.positions_,
                       &view.slots_,
                       &view.q_lengths_,
                       &view.kv_lengths_,
                       &view.query_ends_,
                       &view.block_tables_}) {
    values->clear();
  }
  for (uint64_t row = 0; row < count; ++row) {
    const auto blocks = base.block_tables.subspan(row * base.block_table_width,
                                                  base.block_table_width);
    for (uint32_t item = 0; item < view.rows_per_sequence_; ++item) {
      const int32_t offset = view.first_offset_ + static_cast<int32_t>(item);
      view.tokens_.emplace_back(view.spec_.kind ==
                                        MtpInvocationKind::BLOCK_DRAFT
                                    ? view.spec_.mask_token_id
                                    : 0);
      view.positions_.emplace_back(base.positions[row] + offset);
      view.slots_.emplace_back(0);
      if (view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT) {
        continue;
      }
      view.q_lengths_.emplace_back(1);
      view.kv_lengths_.emplace_back(base.kv_seq_lens[row] + offset);
      view.query_ends_.emplace_back(static_cast<int32_t>(view.tokens_.size()));
      view.block_tables_.insert(
          view.block_tables_.end(), blocks.begin(), blocks.end());
    }
    if (view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT) {
      view.q_lengths_.emplace_back(view.rows_per_sequence_);
      view.kv_lengths_.emplace_back(base.kv_seq_lens[row] +
                                    view.rows_per_sequence_ - 1);
      view.query_ends_.emplace_back(static_cast<int32_t>(view.tokens_.size()));
      view.block_tables_.insert(
          view.block_tables_.end(), blocks.begin(), blocks.end());
    }
  }
  view.planned_ = {view.tokens_,
                   view.positions_,
                   view.slots_,
                   view.q_lengths_,
                   view.kv_lengths_,
                   view.query_ends_,
                   view.block_tables_,
                   base.block_table_width};
  return Status();
}

Status SlotBuffer::prepare_mtp_decode(SlotBuffer& input,
                                      const ModelInputHostView& base,
                                      const ModelInputBatch& batch,
                                      const Stream& stream) {
  Status status = plan_mtp_decode(input, base, batch);
  if (!status.ok()) {
    return status;
  }
  return prepare_planned_mtp_decode(input, base, batch, stream);
}

Status SlotBuffer::prepare_planned_mtp_decode(SlotBuffer& input,
                                              const ModelInputHostView& base,
                                              const ModelInputBatch& batch,
                                              const Stream& stream,
                                              uint32_t physical_rows) {
  auto& view = *input.input_scratch_;
  ModelInputBatch expanded = batch;
  if (view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT) {
    expanded.forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  expanded.num_actual_sequences =
      static_cast<uint32_t>(view.planned_.q_seq_lens.size());
  Status status = physical_rows == 0
                      ? input.prepare(view.planned_, expanded, stream)
                      : input.prepare_decode_padded(
                            view.planned_, expanded, physical_rows, stream);
  if (!status.ok()) {
    return status;
  }
  view.sequences_ = static_cast<uint32_t>(base.q_seq_lens.size());
  const int64_t count = view.sequences_;
  const int64_t rows = count * view.rows_per_sequence_;
  view.positions_view_ = input.positions()
                             .narrow(/*dim=*/0, /*start=*/0, rows)
                             .view({count, view.rows_per_sequence_});
  view.kv_lengths_view_ =
      input.model_params()
          .attention.device.kv_seq_lens
          .narrow(
              /*dim=*/0,
              /*start=*/0,
              view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT ? count : rows)
          .view({count,
                 view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT
                     ? 1
                     : view.rows_per_sequence_});
  view.tokens_view_ = input.tokens()
                          .narrow(/*dim=*/0, /*start=*/0, rows)
                          .view({count, view.rows_per_sequence_});
  view.cache_positions_ =
      view.cache_positions_storage_.narrow(/*dim=*/0, /*start=*/0, rows)
          .view({count, view.rows_per_sequence_});
  view.cache_first_column_ =
      view.cache_positions_.select(/*dim=*/1, /*index=*/0);
  view.model_first_column_ =
      view.positions_view_.select(/*dim=*/1, /*index=*/0);
  view.block_indices_ =
      view.block_indices_storage_.narrow(/*dim=*/0, /*start=*/0, rows)
          .view({rows, 1});
  view.block_ids_ = view.block_ids_storage_.narrow(/*dim=*/0, /*start=*/0, rows)
                        .view({rows, 1});
  view.cache_offsets_ =
      view.cache_offsets_storage_.narrow(/*dim=*/0, /*start=*/0, rows)
          .view({count, view.rows_per_sequence_});
  view.cache_offsets_flat_ = view.cache_offsets_.view({rows});
  view.future_positions_ =
      view.repair_scratch_storage_.narrow(/*dim=*/0, /*start=*/0, count);
  return Status();
}

void SlotBuffer::patch_mtp_decode(SlotBuffer& input,
                                  const MtpContextView& binding) {
  auto& view = *input.input_scratch_;
  CHECK_GT(view.sequences_, 0U);
  CHECK(binding.prepared_);
  CHECK_EQ(binding.tokens_.numel(), view.sequences_);
  CHECK_EQ(binding.tokens_.device(), input.tokens().device());
  const auto& state = binding.state_;
  torch::add_out(view.positions_view_,
                 state.positions.unsqueeze(/*dim=*/1),
                 view.offsets_);
  if (view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT) {
    view.kv_lengths_view_.copy_(state.kv_seq_lens.unsqueeze(/*dim=*/1));
    view.kv_lengths_view_.add_(view.rows_per_sequence_ - 1);
  } else {
    torch::add_out(view.kv_lengths_view_,
                   state.kv_seq_lens.unsqueeze(/*dim=*/1),
                   view.offsets_);
  }
  view.cache_positions_.copy_(view.positions_view_);
  if (view.first_offset_ == -1) {
    view.future_positions_.copy_(state.positions).add_(/*other=*/1);
    torch::where_out(view.cache_first_column_,
                     state.repair_required,
                     view.model_first_column_,
                     view.future_positions_);
    view.tokens_view_.select(/*dim=*/1, /*index=*/0)
        .copy_(state.previous_tokens);
    view.tokens_view_.select(/*dim=*/1, /*index=*/1).copy_(binding.tokens_);
  } else if (view.first_offset_ == 0) {
    view.tokens_view_.select(/*dim=*/1, /*index=*/0).copy_(binding.tokens_);
  }
  // Each expanded row owns its duplicated block table. Indices and outputs
  // are fixed views; no accepted length or token is read by the Host.
  auto indices_matrix =
      view.block_indices_.view({view.sequences_, view.rows_per_sequence_});
  torch::floor_divide_out(
      indices_matrix, view.cache_positions_, view.spec_.block_size);
  if (view.spec_.kind == MtpInvocationKind::BLOCK_DRAFT) {
    auto ids_matrix =
        view.block_ids_.view({view.sequences_, view.rows_per_sequence_});
    torch::gather_out(ids_matrix,
                      input.model_params().attention.device.block_tables,
                      /*dim=*/1,
                      indices_matrix);
  } else {
    torch::gather_out(
        view.block_ids_,
        input.model_params().attention.device.block_tables.narrow(
            /*dim=*/0, /*start=*/0, view.sequences_ * view.rows_per_sequence_),
        /*dim=*/1,
        view.block_indices_);
  }
  torch::remainder_out(
      view.cache_offsets_, view.cache_positions_, view.spec_.block_size);
  const auto slots =
      input.model_params().attention.device.new_cache_slots.narrow(
          /*dim=*/0, /*start=*/0, view.sequences_ * view.rows_per_sequence_);
  slots.copy_(view.block_ids_.view({-1}));
  slots.mul_(view.spec_.block_size).add_(view.cache_offsets_flat_);
}

Status SlotBuffer::plan_mtp_prefill(SlotBuffer& input,
                                    const ModelInputHostView& base,
                                    const ModelInputBatch& batch,
                                    std::span<const int32_t> extra_token_ids,
                                    const SamplingParameters& sampling) {
  if (!input.input_scratch_ ||
      input.input_scratch_->spec_.kind != MtpInvocationKind::PREFILL) {
    return invalid("MTP prefill requires a prompt model input.");
  }
  auto& view = *input.input_scratch_;
  Status status = input.validate(base, batch);
  if (!status.ok()) {
    return status;
  }
  const uint64_t rows = base.q_seq_lens.size();
  if ((!batch.forward_type.is_prefill() &&
       !batch.forward_type.is_chunked_prefill() &&
       !batch.forward_type.is_mixed() && !batch.forward_type.is_decode()) ||
      batch.num_actual_sequences != rows || extra_token_ids.size() != rows ||
      std::any_of(base.token_ids.begin(),
                  base.token_ids.end(),
                  [](int32_t token) { return token < 0; }) ||
      std::any_of(extra_token_ids.begin(),
                  extra_token_ids.end(),
                  [](int32_t token) { return token < -1; })) {
    return invalid(
        "MTP Prefill requires known query tokens and one extra token per row.");
  }
  const bool selected = sampling.selected_token_idxes.defined();
  const bool samples = sampling.sample_idxes.defined();
  if (selected != samples ||
      (selected && (!cpu_indices(sampling.selected_token_idxes) ||
                    !cpu_indices(sampling.sample_idxes)))) {
    return invalid(
        "MTP Prefill sampling indices must be contiguous CPU int32.");
  }
  const int64_t sample_count = samples ? sampling.sample_idxes.numel() : 0;
  const int64_t selected_count =
      selected ? sampling.selected_token_idxes.numel() : 0;
  if (sample_count > static_cast<int64_t>(rows)) {
    return invalid("MTP Prefill samples exceed model rows.");
  }
  view.token_to_sequence_.assign(base.token_ids.size(), -1);
  int64_t end = 0;
  uint32_t complete = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    end += base.q_seq_lens[row];
    if (base.positions[end - 1] == std::numeric_limits<int32_t>::max() ||
        base.kv_seq_lens[row] == std::numeric_limits<int32_t>::max()) {
      return invalid("MTP Prefill bootstrap position exceeds int32.");
    }
    view.token_to_sequence_[end - 1] = static_cast<int32_t>(row);
    complete += extra_token_ids[row] == -1 ? 1U : 0U;
  }
  if (sample_count != complete) {
    return invalid("MTP Prefill needs one sample for each completed chunk.");
  }
  std::fill(
      view.published_mask_.begin(), view.published_mask_.end(), uint8_t{0});
  view.planned_rows_.clear();
  view.sampled_token_rows_.clear();
  const int32_t* sample_data =
      samples ? sampling.sample_idxes.const_data_ptr<int32_t>() : nullptr;
  const int32_t* selected_data =
      selected ? sampling.selected_token_idxes.const_data_ptr<int32_t>()
               : nullptr;
  for (int64_t index = 0; index < sample_count; ++index) {
    const int32_t sample = sample_data[index];
    if (sample < 0 || sample >= selected_count) {
      return invalid("MTP Prefill sample index is outside selected rows.");
    }
    const int32_t token = selected_data[sample];
    if (token < 0 || static_cast<uint64_t>(token) >= base.token_ids.size()) {
      return invalid("MTP Prefill selected token is outside the query.");
    }
    const int32_t row = view.token_to_sequence_[token];
    if (row < 0 || extra_token_ids[row] != -1 ||
        view.published_mask_[row] != 0) {
      return invalid(
          "MTP Prefill samples must uniquely select completed row tails.");
    }
    view.published_mask_[row] = 1;
    view.planned_rows_.push_back(static_cast<uint32_t>(row));
    view.sampled_token_rows_.push_back(token);
  }
  if (view.spec_.context_only) {
    return Status();
  }
  view.tokens_.clear();
  int64_t start = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    const int32_t q = base.q_seq_lens[row];
    view.tokens_.insert(view.tokens_.end(),
                        base.token_ids.begin() + start + 1,
                        base.token_ids.begin() + start + q);
    view.tokens_.push_back(std::max(extra_token_ids[row], 0));
    start += q;
  }
  return Status();
}

Status SlotBuffer::prepare_mtp_prefill(SlotBuffer& input,
                                       const ModelInputHostView& base,
                                       const ModelInputBatch& batch,
                                       std::span<const int32_t> extra_token_ids,
                                       const SamplingParameters& sampling,
                                       const Stream& stream) {
  Status status =
      plan_mtp_prefill(input, base, batch, extra_token_ids, sampling);
  if (!status.ok()) {
    return status;
  }
  return prepare_planned_mtp_prefill(input, base, batch, stream);
}

Status SlotBuffer::prepare_planned_mtp_prefill(SlotBuffer& input,
                                               const ModelInputHostView& base,
                                               const ModelInputBatch& batch,
                                               const Stream& stream) {
  auto& view = *input.input_scratch_;
  if (stream.get_stream()->device_index() !=
      view.device_indices_.device().index()) {
    return invalid("MTP Prefill stream uses a different device.");
  }
  c10::DeviceGuard guard(view.device_indices_.device());
  if (!view.spec_.context_only) {
    ModelInputHostView shifted = base;
    shifted.token_ids = view.tokens_;
    Status status = input.prepare(shifted, batch, stream);
    if (!status.ok()) {
      return status;
    }
  }
  view.published_rows_ = view.planned_rows_;
  const int64_t samples = view.published_rows_.size();
  int64_t* indices = view.host_indices_.data_ptr<int64_t>();
  int32_t* positions =
      view.host_lengths_.select(/*dim=*/0, /*index=*/0).data_ptr<int32_t>();
  int32_t* lengths =
      view.host_lengths_.select(/*dim=*/0, /*index=*/1).data_ptr<int32_t>();
  for (int64_t index = 0; index < samples; ++index) {
    indices[index] = view.sampled_token_rows_[index];
    positions[index] = base.positions[indices[index]] + 1;
    lengths[index] = base.kv_seq_lens[view.published_rows_[index]] + 1;
  }
  view.indices_ = view.device_indices_.narrow(/*dim=*/0, /*start=*/0, samples);
  view.bootstrap_positions_ =
      view.device_lengths_.select(/*dim=*/0, /*index=*/0)
          .narrow(/*dim=*/0, /*start=*/0, samples);
  view.bootstrap_kv_lengths_ =
      view.device_lengths_.select(/*dim=*/0, /*index=*/1)
          .narrow(/*dim=*/0, /*start=*/0, samples);
  view.bootstrap_hidden_ =
      view.bootstrap_hidden_storage_.narrow(/*dim=*/0, /*start=*/0, samples);
  view.sampled_token_values_ =
      view.token_storage_.narrow(/*dim=*/0, /*start=*/0, samples);
  input.model_params().embedding.input_embedding = torch::Tensor();
  copy_to_device(view.indices_,
                 view.host_indices_.narrow(/*dim=*/0, /*start=*/0, samples),
                 stream);
  copy_to_device(view.bootstrap_positions_,
                 view.host_lengths_.select(/*dim=*/0, /*index=*/0)
                     .narrow(/*dim=*/0, /*start=*/0, samples),
                 stream);
  copy_to_device(view.bootstrap_kv_lengths_,
                 view.host_lengths_.select(/*dim=*/0, /*index=*/1)
                     .narrow(/*dim=*/0, /*start=*/0, samples),
                 stream);
  return Status();
}

void SlotBuffer::patch_mtp_prefill(SlotBuffer& input,
                                   const torch::Tensor& target_hidden,
                                   const torch::Tensor& sampled_tokens,
                                   MtpContextView& binding) {
  auto& view = *input.input_scratch_;
  CHECK_EQ(target_hidden.size(/*dim=*/0), input.tokens().numel());
  // Prefill is eager. The owning task retains its target output through
  // retirement, so the draft can borrow it without a second prompt-sized copy.
  input.model_params().embedding.input_embedding = target_hidden;
  initialize_mtp_context(input, target_hidden, sampled_tokens, binding);
  if (!view.published_rows_.empty()) {
    input.tokens().index_copy_(
        /*dim=*/0, view.indices_, view.sampled_token_values_);
  }
}

void SlotBuffer::initialize_mtp_context(SlotBuffer& input,
                                        const torch::Tensor& target_hidden,
                                        const torch::Tensor& sampled_tokens,
                                        MtpContextView& binding) {
  auto& view = *input.input_scratch_;
  CHECK_EQ(target_hidden.dim(), 2);
  CHECK_EQ(target_hidden.size(/*dim=*/1), view.spec_.hidden_size);
  CHECK_EQ(target_hidden.device(), view.bootstrap_hidden_storage_.device());
  CHECK_EQ(target_hidden.scalar_type(),
           view.bootstrap_hidden_storage_.scalar_type());
  const int64_t samples = view.published_rows_.size();
  CHECK_EQ(binding.tokens_.numel(), samples);
  if (samples == 0) {
    return;
  }
  CHECK(binding.prepared_);
  CHECK_EQ(sampled_tokens.device(), target_hidden.device());
  CHECK_EQ(sampled_tokens.scalar_type(), torch::kInt64);
  CHECK_EQ(sampled_tokens.dim(), 2);
  CHECK_EQ(sampled_tokens.size(/*dim=*/0), samples);
  CHECK_EQ(sampled_tokens.size(/*dim=*/1), 1);
  view.sampled_token_values_.copy_(sampled_tokens.squeeze(/*dim=*/1));
  torch::index_select_out(
      view.bootstrap_hidden_, target_hidden, /*dim=*/0, view.indices_);
  binding.tokens_.copy_(sampled_tokens.squeeze(/*dim=*/1));
  const auto& state = binding.state_;
  if (state.hidden.defined()) {
    state.previous_tokens.copy_(binding.tokens_);
    state.hidden.select(/*dim=*/1, /*index=*/0).zero_();
    state.hidden.select(/*dim=*/1, /*index=*/1).copy_(view.bootstrap_hidden_);
    state.repair_required.zero_();
  }
  state.positions.copy_(view.bootstrap_positions_);
  state.kv_seq_lens.copy_(view.bootstrap_kv_lengths_);
}

}  // namespace xllm
