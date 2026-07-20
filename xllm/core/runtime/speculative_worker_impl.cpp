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

#include "speculative_worker_impl.h"

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/speculative_config.h"
#include "spec_input_builder.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
#define TENSOR_REPEAT(tensor_, repeats)                                       \
  do {                                                                        \
    tensor_ = tensor_.defined()                                               \
                  ? tensor_.repeat_interleave(/*repeats=*/repeats, /*dim=*/0) \
                  : tensor_;                                                  \
  } while (0)

Slice<int32_t> tensor_slice(const torch::Tensor& tensor) {
  return {tensor.data_ptr<int32_t>(), static_cast<size_t>(tensor.numel())};
}

}  // namespace

SpeculativeWorkerImpl::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options)
    : WorkerImpl(parallel_args, device, options) {
  impl_ =
      std::make_unique<LLMWorkerImpl>(parallel_args, device, target_options);
}

bool SpeculativeWorkerImpl::init_model(const std::string& model_weights_path,
                                       int32_t random_seed,
                                       MasterStatus master_status) {
  // Base class only loads the target model.
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
    if (result) {
      dtype_ = impl_->dtype();
      embedding_size_ = impl_->hidden_size();
    }
  }
  enable_fused_kernel_ =
      impl_->get_optimization_config().enable_fused_spec_kernel;
  return result;
}

bool SpeculativeWorkerImpl::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

#if defined(USE_NPU)
bool SpeculativeWorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer(kv_cache_shape);
}
#endif

std::optional<ForwardOutput> SpeculativeWorkerImpl::step(
    const ForwardInput& input) {
  if (input.input_params.meta.num_sequences == 0 ||
      input.token_ids.numel() == 0) {
    return step_empty(input);
  }

  if (input.input_params.meta.batch_forward_type.is_decode()) {
    return step_decode(input);
  }
  return step_prefill(input);
}

ForwardInput SpeculativeWorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
  // only process decode batch, so prepare draft input here.
  ForwardInput& new_inputs = inputs;

  auto& input_params = new_inputs.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t block_size = options_.block_size();

  Slice<int32_t> token_ids = tensor_slice(inputs.token_ids_host);
  torch::Tensor last_token_ids = safe_to(
      last_step_output_.sample_output.next_tokens.flatten(), torch::kCPU);
  Slice<int64_t> last_tokens_ids_slice = {
      last_token_ids.data_ptr<int64_t>(),
      static_cast<size_t>(last_token_ids.numel())};

  // Determine how many tokens were decoded in the last step
  // If the output is 2D, it means multiple tokens were generated per sequence
  int32_t last_step_decode_num = 1;
  if (last_step_output_.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num = last_step_output_.sample_output.next_tokens.size(1);
  }

  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(num_sequences);
  buf.out_positions.reserve(num_sequences);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_new_cache_slots.reserve(num_sequences);
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(inputs);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    specBuilder::append_decode_row_from_last_step(row_ctx,
                                                  seq_id,
                                                  token_ids[seq_id],
                                                  last_tokens_ids_slice,
                                                  last_step_decode_num,
                                                  block_size,
                                                  buf);
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "step-update kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "step-update positions/tokens mismatch";

  specBuilder::set_token_position_tensors(new_inputs,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          inputs.token_ids.options(),
                                          inputs.positions.options());
  // update the input_params
  input_params.meta.kv_max_seq_len = buf.meta.kv_max_seq_len;
  input_params.attention.host.kv_seq_lens = std::move(buf.out_kv_seq_lens);
  input_params.attention.host.new_cache_slots =
      std::move(buf.out_new_cache_slots);
  input_params.attention.rebuild_device_buffer(device_);
  new_inputs.device_tensors_ready = true;

  return new_inputs;
}

void SpeculativeWorkerImpl::update_sampling_params(
    SamplingParameters& sampling_params,
    const int32_t num_val_tokens,
    const int32_t total_num_val_tokens) {
  std::vector<int32_t> selected_token_idxes_vec;
  selected_token_idxes_vec.reserve(total_num_val_tokens);
  for (int32_t i = 0; i < total_num_val_tokens; i++) {
    selected_token_idxes_vec.emplace_back(i);
  }
  torch::Tensor selected_token_idxes = torch::tensor(selected_token_idxes_vec);

  // sample_idxes equals to selected_token_idxes since only process decode batch
  sampling_params.selected_token_idxes = selected_token_idxes.to(device_);
  sampling_params.sample_idxes = selected_token_idxes.to(device_);

  TENSOR_REPEAT(sampling_params.frequency_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.presence_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.repetition_penalties, num_val_tokens);
  TENSOR_REPEAT(sampling_params.temperatures, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_p, num_val_tokens);
  TENSOR_REPEAT(sampling_params.top_k, num_val_tokens);
  TENSOR_REPEAT(sampling_params.unique_token_ids, num_val_tokens);
  TENSOR_REPEAT(sampling_params.unique_token_counts, num_val_tokens);
  TENSOR_REPEAT(sampling_params.unique_token_ids_lens, num_val_tokens);
  TENSOR_REPEAT(sampling_params.do_sample, num_val_tokens);
}

void SpeculativeWorkerImpl::prepare_validate_inputs(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  validate_input = input.to(device_, dtype_);
  validate_input.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  torch::TensorOptions token_options = validate_input.token_ids.options();
  torch::TensorOptions position_options = validate_input.positions.options();

  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = options_.block_size();
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);

  Slice<int32_t> token_ids = tensor_slice(input.token_ids_host);
  Slice<int32_t> positions = tensor_slice(input.positions_host);
  Slice<int32_t> kv_seq_lens = input.input_params.attention.host.kv_seq_lens;
  specBuilder::DecodeBuildBuffers buf;
  buf.out_token_ids.reserve(total_num_val_tokens);
  buf.out_positions.reserve(total_num_val_tokens);
  buf.out_new_cache_slots.reserve(total_num_val_tokens);
  if (!::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    buf.out_kv_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_seq_lens.reserve(total_num_val_tokens);
    buf.out_q_cu_seq_lens.reserve(total_num_val_tokens);
    buf.out_block_tables.reserve(static_cast<size_t>(total_num_val_tokens) *
                                 row_ctx.block_table_stride);
  }

  std::vector<int32_t> atb_kv_seq_lens_vec = {};
  std::vector<int32_t> atb_q_seq_lens_vec = {};
  std::vector<int32_t> atb_q_cu_seq_lens_vec = {};
  int32_t atb_kv_max_seq_len = 0;
  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    int32_t start_position = positions[seq_id];
    int32_t kv_len =
        specBuilder::calc_kv_len(kv_seq_lens, seq_id, /*offset=*/0);
    CHECK_EQ(start_position + 1, kv_len)
        << "validate position/kv_len mismatch, seq_id=" << seq_id
        << ", start_position=" << start_position << ", kv_len=" << kv_len;

    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.token_id = token_ids[seq_id];
      } else {
        row.token_id = -val_idx;
      }
      row.position_offset = val_idx;
      row.append_kv_len =
          !::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel();
      row.append_q_len_one =
          !::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel();
      row.append_block_table =
          !::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel();
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
    }

    if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
      const int32_t kv_len_after_validation = kv_len + num_speculative_tokens;
      specBuilder::update_kv_seq_lens_and_max(
          atb_kv_seq_lens_vec, kv_len_after_validation, atb_kv_max_seq_len);
      specBuilder::append_q_seq_len(
          atb_q_seq_lens_vec, atb_q_cu_seq_lens_vec, num_val_tokens);
    }
  }

  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_token_ids.size())
      << "validate kv slots/tokens mismatch";
  CHECK_EQ(buf.out_positions.size(), buf.out_token_ids.size())
      << "validate positions/tokens mismatch";

  specBuilder::set_token_position_tensors(validate_input,
                                          buf.out_token_ids,
                                          buf.out_positions,
                                          token_options,
                                          position_options);
  // update the input_params
  if (!::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    input_params.meta.num_sequences = total_num_val_tokens;
    input_params.meta.q_max_seq_len = 1;
    input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  } else {
    input_params.meta.q_max_seq_len = num_val_tokens;
    input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  }
  if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     num_val_tokens,
                                     std::move(atb_q_seq_lens_vec),
                                     std::move(atb_q_cu_seq_lens_vec),
                                     atb_kv_max_seq_len,
                                     std::move(atb_kv_seq_lens_vec));
  } else {
    specBuilder::update_input_params(input_params,
                                     buf,
                                     1,
                                     std::move(buf.out_q_seq_lens),
                                     std::move(buf.out_q_cu_seq_lens),
                                     buf.meta.kv_max_seq_len,
                                     std::move(buf.out_kv_seq_lens),
                                     /*update_block_tables=*/true);
  }
  input_params.attention.rebuild_device_buffer(device_);

  // update the sampling_params
  update_sampling_params(
      validate_input.sampling_params, num_val_tokens, total_num_val_tokens);

  // update dp_global_token_nums for dp/ep parallel
  for (auto& it : input_params.parallel.dp_global_token_nums) {
    it *= num_val_tokens;
  }
  validate_input.device_tensors_ready = true;
}

void SpeculativeWorkerImpl::prepare_work_before_execute(
    const ForwardInput& input,
    ForwardInput& processed_input) {
  WorkerImpl::prepare_work_before_execute(input, processed_input);
}
}  // namespace xllm
