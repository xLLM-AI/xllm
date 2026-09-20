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

namespace xllm {
namespace {

struct ResultField {
  torch::Tensor TokenResultTensors::* member;
  torch::ScalarType dtype;
  bool is_top_token;
};

constexpr std::array<ResultField, 4> kResultFields = {{
    {&TokenResultTensors::tokens, torch::kInt64, false},
    {&TokenResultTensors::logprobs, torch::kFloat32, false},
    {&TokenResultTensors::top_tokens, torch::kInt64, true},
    {&TokenResultTensors::top_logprobs, torch::kFloat32, true},
}};

TokenResultTensors bind_result_views(const TokenResultTensors& storage,
                                     const SamplingParameters& sampling) {
  TokenResultTensors view;
  const int64_t rows =
      sampling.sample_idxes.defined() ? sampling.sample_idxes.numel() : 0;
  if (rows == 0) {
    return view;
  }
  view.tokens = storage.tokens.narrow(/*dim=*/0, /*start=*/0, rows);
  if (sampling.logprobs) {
    view.logprobs = storage.logprobs.narrow(/*dim=*/0, /*start=*/0, rows);
  }
  if (sampling.logprobs && sampling.max_top_logprobs != 0) {
    view.top_tokens =
        storage.top_tokens
            .narrow(/*dim=*/0, /*start=*/0, rows * sampling.max_top_logprobs)
            .view({rows, sampling.max_top_logprobs});
    view.top_logprobs =
        storage.top_logprobs
            .narrow(/*dim=*/0, /*start=*/0, rows * sampling.max_top_logprobs)
            .view({rows, sampling.max_top_logprobs});
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
      return capacity.model.max_sequences;
    case RowDomain::SAMPLE:
      return capacity.model.max_sequences;
    case RowDomain::HISTORY:
      return static_cast<uint64_t>(capacity.model.max_sequences) *
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

bool overlaps_host(std::span<const int32_t> source,
                   const torch::Tensor& destination) {
  if (source.empty()) {
    return false;
  }
  const uintptr_t start = reinterpret_cast<uintptr_t>(source.data());
  const uintptr_t base = reinterpret_cast<uintptr_t>(destination.data_ptr());
  return start >= base ? start - base < destination.nbytes()
                       : base - start < source.size_bytes();
}

// Called after validate_model; sizes and capacity are already checked.
Status validate_batch(const ModelInputHostView& input,
                      const ModelInputBatch& batch) {
  const uint64_t rows = input.q_seq_lens.size();
  if (batch.num_actual_sequences > rows) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Invalid physical or actual model input rows.");
  }
  switch (batch.forward_type.value()) {
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
  if (rows == 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Nonempty model input batch requires physical rows.");
  }
  for (uint32_t row = 0; row < batch.num_actual_sequences; ++row) {
    if (input.q_seq_lens[row] <= 0 ||
        (batch.forward_type.is_prefill() &&
         input.kv_seq_lens[row] != input.q_seq_lens[row])) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Invalid query or cache lengths for actual model rows.");
    }
  }
  if (batch.forward_type.is_decode() &&
      std::any_of(input.q_seq_lens.begin(),
                  input.q_seq_lens.end(),
                  [](int32_t length) { return length > 1; })) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Decode model input rows must contain at most one token.");
  }
  return Status();
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

}  // namespace

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
  constexpr uint64_t kMaxElements = std::numeric_limits<int32_t>::max();
  if (device.type() != Platform::type_torch() || !device.has_index() ||
      !parameter_type(capacity.parameter_dtype) ||
      capacity.model.max_sequences > kMaxElements ||
      capacity.max_unique_tokens == 0 ||
      capacity.max_unique_tokens > kMaxElements || capacity.vocab_size == 0 ||
      capacity.vocab_size > kMaxElements ||
      capacity.max_top_logprobs > capacity.vocab_size) {
    return invalid_input();
  }
  uint64_t bytes = 0;
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
  for (const auto& field : kResultFields) {
    const uint64_t elements =
        static_cast<uint64_t>(capacity.model.max_sequences) *
        (field.is_top_token ? capacity.max_top_logprobs : 1);
    const uint64_t element_bytes = torch::elementSize(field.dtype);
    if (elements > (kMaxBytes - bytes) / element_bytes) {
      return invalid_input();
    }
    bytes += elements * element_bytes;
  }
  c10::DeviceGuard guard(device);
  output = std::unique_ptr<SlotBuffer>(
      new SlotBuffer(capacity, device, layout, bytes));
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
  const uint32_t rows = capacity_.model.max_sequences;
  const auto options =
      torch::TensorOptions().device(device_).dtype(torch::kInt64);
  host_indices_ = torch::empty(
      {2, rows},
      options.device(torch::kCPU).pinned_memory(/*pinned_memory=*/true));
  device_indices_ = torch::empty({2, rows}, options);
  gathered_int64_ = torch::empty({rows}, options);
  gathered_int32_ = torch::empty({rows}, options.dtype(torch::kInt32));
  for (const auto& field : kResultFields) {
    const int64_t count = static_cast<int64_t>(rows) *
                          (field.is_top_token ? capacity_.max_top_logprobs : 1);
    if (count == 0) {
      continue;
    }
    const auto result_options = torch::TensorOptions().dtype(field.dtype);
    result_host_storage_.*field.member = torch::empty(
        {count}, result_options.device(torch::kCPU).pinned_memory(true));
    result_device_storage_.*field.member =
        torch::empty({count}, result_options.device(device_));
  }
  result_ready_ = std::make_unique<StreamEvent>(device_.type());
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
      input.num_return_sequences != 0 || input.max_top_logprobs < 0 ||
      static_cast<uint64_t>(input.max_top_logprobs) >
          capacity_.max_top_logprobs) {
    return invalid_input();
  }
  const int64_t rows = input.selected_token_idxes.defined()
                           ? input.selected_token_idxes.numel()
                           : 0;
  const int64_t samples =
      input.sample_idxes.defined() ? input.sample_idxes.numel() : 0;
  if (rows > capacity_.model.max_sequences || samples > rows ||
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

Status SlotBuffer::validate(const ModelInputHostView& model,
                            const ModelInputBatch& batch,
                            const SamplingParameters& sampling,
                            uint32_t previous_rows,
                            const Stream& stream) const {
  if (copy_submitted_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "Take or discard the pending result before Slot reuse.");
  }
  if (stream.get_stream()->device_index() != device_.index()) {
    return invalid("Input stream and Slot device differ.");
  }
  Status status = validate_model(model);
  if (!status.ok()) {
    return status;
  }
  status = validate_batch(model, batch);
  if (!status.ok()) {
    return status;
  }
  status = validate_sampling(sampling, model.token_ids.size());
  if (!status.ok()) {
    return status;
  }
  return validate_previous_tokens(model, previous_rows);
}

void SlotBuffer::prepare(const ModelInputHostView& model,
                         const ModelInputBatch& batch,
                         const SamplingParameters& sampling,
                         const Stream& stream) {
  CHECK(!copy_submitted_) << "Pending result prevents Slot reuse.";
  // The caller validated all inputs before this first staging write. Mapping
  // indices must be copied before metadata updates can invalidate borrowed
  // spans.
  auto guard = stream.set_stream_guard();
  prepare_previous_tokens(model);
  prepare_model(model, batch);
  prepare_sampling(sampling);
  prepare_result();
}

void SlotBuffer::prepare_model(const ModelInputHostView& input,
                               const ModelInputBatch& batch) {
  const Layout& layout = layout_;
  const uint32_t token_count = static_cast<uint32_t>(input.token_ids.size());
  const uint32_t rows = static_cast<uint32_t>(input.q_seq_lens.size());
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
    if (sources[i].empty()) {
      continue;
    }
    const uint64_t offset = regions[i].offset / sizeof(int32_t);
    std::memcpy(host_data + offset, sources[i].data(), sources[i].size_bytes());
    device_buffer_.narrow(/*dim=*/0, offset, sources[i].size())
        .copy_(host_buffer_.narrow(/*dim=*/0, offset, sources[i].size()),
               /*non_blocking=*/true);
  }
  if (rows != 0) {
    const uint64_t offset = layout.block_tables.offset / sizeof(int32_t);
    const uint64_t stride = layout.block_table_row_stride_bytes;
    const uint64_t width =
        static_cast<uint64_t>(input.block_table_width) * sizeof(int32_t);
    const uint64_t bytes = stride * rows;
    if (width == stride) {
      std::memcpy(host_data + offset, input.block_tables.data(), bytes);
    } else {
      std::memset(host_data + offset, 0, bytes);
      for (uint32_t row = 0; row < rows; ++row) {
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

  model_params_.meta.batch_forward_type = batch.forward_type;
  model_params_.meta.num_sequences = static_cast<int32_t>(rows);
  model_params_.meta.actual_num_sequences =
      static_cast<int32_t>(batch.num_actual_sequences);
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
      batch.forward_type.is_prefill() && !capacity_.enable_mla
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
  metadata.is_prefill = batch.forward_type.is_prefill();
  metadata.is_chunked_prefill =
      batch.forward_type.is_chunked_prefill() || batch.forward_type.is_mixed();
  metadata.is_mixed = batch.forward_type.is_mixed();
  metadata.is_dummy = rows == 0;
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
  result_host_ = bind_result_views(result_host_storage_, sampling_params_);
  result_device_ = bind_result_views(result_device_storage_, sampling_params_);
}

Status SlotBuffer::copy_result_to_host(const Stream& stream,
                                       const StreamEventPtr& producer_ready) {
  if (!producer_ready ||
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
  for (const auto& field : kResultFields) {
    const torch::Tensor& source = result_host_.*field.member;
    if (!source.defined()) {
      continue;
    }
    torch::Tensor destination =
        torch::empty(source.sizes(),
                     source.options().pinned_memory(/*pinned_memory=*/false));
    std::memcpy(destination.data_ptr(), source.data_ptr(), source.nbytes());
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
  return host_buffer_.nbytes() + auxiliary_bytes_ + host_indices_.nbytes();
}
uint64_t SlotBuffer::device_bytes() const {
  return device_buffer_.nbytes() + auxiliary_bytes_ + device_indices_.nbytes() +
         gathered_int64_.nbytes() + gathered_int32_.nbytes();
}

}  // namespace xllm
