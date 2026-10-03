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

#include "core/runtime/task_execution_pipeline_speculative.h"

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <array>
#include <limits>

#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/dflash2_sampling.h"
#include "core/framework/sampling/gumbel_sampling.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/decode_graph_bucket.h"
#include "core/runtime/task_execution_pipeline.h"

namespace xllm {
bool TaskExecutionPipeline::block_draft() const {
  return speculative_capacity_->kind != SpeculativeTaskKind::MTP;
}

uint32_t TaskExecutionPipeline::context_hidden_size() const {
  return block_draft() ? speculative_capacity_->context_hidden_size
                       : speculative_capacity_->common.hidden_size;
}

namespace {

ModelInputBatch model_batch(const ForwardInput& input) {
  const auto& meta = input.input_params.meta;
  const uint32_t rows = input.input_params.attention.host.q_seq_lens.size();
  return {rows == 0 ? BatchForwardType::EMPTY : meta.batch_forward_type,
          rows,
          meta.batch_id,
          false};
}

ParallelInput expanded_parallel(const ParallelInput& base,
                                uint32_t scale,
                                bool pad_empty) {
  ParallelInput result = base;
  for (int32_t& count : result.dp_global_token_nums) {
    count = count == 0 && pad_empty ? scale : count * scale;
  }
  for (int32_t& count : result.raw_dp_global_token_nums) {
    count *= scale;
  }
  return result;
}

uint32_t task_graph_batch_size(const ParallelInput& input,
                               std::span<const int64_t> buckets) {
  int32_t maximum = 0;
  for (uint32_t rank = 0; rank < input.dp_global_token_nums.size(); ++rank) {
    const int32_t count = input.dp_global_token_nums[rank];
    if (count != 0 && input.dp_is_decode[rank] != 1) {
      return 0;
    }
    maximum = std::max(maximum, count);
  }
  const auto bucket = std::lower_bound(buckets.begin(), buckets.end(), maximum);
  return maximum == 0 || bucket == buckets.end() ? 0 : *bucket;
}

void pad_task_parallel(ParallelInput& input, uint32_t rows) {
  if (rows != 0) {
    std::fill(input.dp_global_token_nums.begin(),
              input.dp_global_token_nums.end(),
              rows);
    std::fill(input.dp_is_decode.begin(), input.dp_is_decode.end(), 1);
  }
}

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

StreamEventPtr reusable_event(const torch::Device& device) {
  return std::make_shared<StreamEvent>(device.type());
}

void record(const Stream& stream, const StreamEventPtr& event) {
  stream.record_event(*event);
}

torch::Tensor prefix(const torch::Tensor& storage, int64_t count) {
  return storage.narrow(/*dim=*/0, /*start=*/0, count);
}

std::vector<int64_t> decode_graph_batch_sizes(int64_t maximum) {
  std::vector<int64_t> sizes;
  sizes.reserve(maximum / 16 + 5);
  for (int64_t size = 1; size <= maximum;) {
    const int64_t bucket = std::min(runtime::get_decode_graph_token_bucket(
                                        size, /*enable_no_padding=*/false),
                                    maximum);
    sizes.emplace_back(bucket);
    size = bucket + 1;
  }
  return sizes;
}

void copy_input(const torch::Tensor& device,
                const torch::Tensor& host,
                const Stream& stream) {
  if (host.numel() == 0) {
    return;
  }
  CHECK_EQ(device.nbytes(), host.nbytes());
  auto stream_guard = stream.set_stream_guard();
  device.copy_(host, /*non_blocking=*/true);
}

SampleOutput sample_views(const TokenResultTensors& result) {
  SampleOutput output;
  if (!result.tokens.defined()) {
    return output;
  }
  output.next_tokens = result.tokens.view({-1});
  if (result.logprobs.defined()) {
    output.logprobs = result.logprobs.view({-1});
  }
  if (result.top_tokens.defined()) {
    const int64_t top = result.top_tokens.size(/*dim=*/2);
    output.top_tokens = result.top_tokens.view({-1, top});
    output.top_logprobs = result.top_logprobs.view({-1, top});
  }
  return output;
}

constexpr std::array<torch::Tensor SamplingParameters::*, 10> kRepeatedRows = {
    &SamplingParameters::frequency_penalties,
    &SamplingParameters::presence_penalties,
    &SamplingParameters::repetition_penalties,
    &SamplingParameters::temperatures,
    &SamplingParameters::top_p,
    &SamplingParameters::top_k,
    &SamplingParameters::unique_token_ids,
    &SamplingParameters::unique_token_counts,
    &SamplingParameters::unique_token_ids_lens,
    &SamplingParameters::do_sample};

uint64_t state_bytes(const MtpContextTensors& state) {
  uint64_t bytes = 0;
  for (const auto* tensor : {&state.previous_tokens,
                             &state.hidden,
                             &state.positions,
                             &state.kv_seq_lens,
                             &state.repair_required}) {
    if (tensor->defined()) {
      bytes += tensor->nbytes();
    }
  }
  return bytes;
}

}  // namespace

Status TaskExecutionPipeline::create(
    ThreadPool& state_executor,
    TaskModel target,
    TaskModel draft,
    const SpeculativeTaskCapacity& capacity,
    std::unique_ptr<TaskExecutionPipeline>& output) {
  const auto& common = capacity.common;
  const uint64_t width =
      static_cast<uint64_t>(capacity.num_speculative_tokens) + 1;
  const uint64_t rows =
      static_cast<uint64_t>(common.model.max_sequences) * width;
  if (state_executor.size() != 1 || common.slot_count == 0 ||
      common.slot_count > 2 || common.model.max_sequences == 0 ||
      common.model.max_tokens == 0 || common.max_positions == 0 ||
      common.max_kv_seq_len == 0 ||
      common.max_kv_seq_len > common.max_positions ||
      common.max_positions > std::numeric_limits<int32_t>::max() ||
      common.hidden_size == 0 || common.logical_block_size == 0 ||
      common.logical_block_size > std::numeric_limits<int32_t>::max() ||
      common.vocab_size == 0 ||
      (capacity.kind == SpeculativeTaskKind::MTP && !common.enable_mla) ||
      (capacity.kind != SpeculativeTaskKind::MTP &&
       (capacity.context_hidden_size == 0 || capacity.mask_token_id < 0 ||
        static_cast<uint32_t>(capacity.mask_token_id) >= common.vocab_size)) ||
      (capacity.kind == SpeculativeTaskKind::DFLASH &&
       capacity.draft_sampling_mode != DraftSamplingMode::GREEDY) ||
      (capacity.kind == SpeculativeTaskKind::DFLASH2 &&
       capacity.selector_top_k == 0) ||
      capacity.num_speculative_tokens == 0 ||
      (capacity.draft_sampling_mode != DraftSamplingMode::GREEDY &&
       capacity.draft_sampling_mode != DraftSamplingMode::PROBABILISTIC) ||
      capacity.fused_rejection || common.dp_size == 0 ||
      common.dp_rank >= common.dp_size ||
      rows > std::numeric_limits<int32_t>::max() ||
      (capacity.reuse_topk && capacity.index_topk == 0) ||
      target.model.device().type() != Platform::type_torch() ||
      !target.model.device().has_index() ||
      target.model.device() != draft.model.device() ||
      target.model.options().dtype() != draft.model.options().dtype() ||
      !target.executor.supports_prepared_attention_metadata() ||
      !draft.executor.supports_prepared_attention_metadata()) {
    return invalid("Invalid MTP capacity or prepared MLA model resources.");
  }
  c10::DeviceGuard guard(target.model.device());
  auto program = std::unique_ptr<TaskExecutionPipeline>(
      new TaskExecutionPipeline(state_executor,
                                target.model,
                                target.executor,
                                target.kv_caches,
                                common));
  program->speculative_capacity_ =
      std::make_unique<SpeculativeTaskCapacity>(capacity);
  program->draft_ = std::make_unique<TaskModel>(draft);
  program->hidden_dtype_ = target.model.options().dtype().toScalarType();
  Status status = program->initialize_speculative();
  if (!status.ok()) {
    return status;
  }
  CHECK_EQ(Device(target.model.device()).current_stream()->synchronize(), 0);
  program->start();
  output = std::move(program);
  return Status();
}

Status TaskExecutionPipeline::initialize_speculative() {
  const auto& c = capacity_;
  const uint32_t count = c.model.max_sequences;
  const uint32_t width = speculative_capacity_->num_speculative_tokens + 1;
  const uint32_t expanded = count * width;
  Status status = create_mtp_context(0,
                                     block_draft() ? 0 : c.hidden_size,
                                     hidden_dtype_,
                                     device_.unwrap(),
                                     context_);
  if (!status.ok()) {
    return status;
  }
  const uint32_t draft_samples = block_draft() ? count * (width - 1) : count;
  const uint32_t max_batch_size = std::min(c.max_graph_batch_size, expanded);
  graph_batch_sizes_ = decode_graph_batch_sizes(max_batch_size);
  captured_graph_batch_sizes_.reserve(graph_batch_sizes_.size());
  const auto options = model_.options();
  const auto pinned = options.device(torch::kCPU).pinned_memory(true);
  SlotBufferCapacity base_capacity{c.model,
                                   c.max_unique_tokens,
                                   c.vocab_size,
                                   c.max_top_logprobs,
                                   c.parameter_dtype,
                                   c.enable_mla};
  base_capacity.max_result_width = width;
  SlotBufferCapacity target_capacity = base_capacity;
  target_capacity.max_selected_rows = expanded;
  target_capacity.max_sample_rows = expanded;
  target_capacity.max_result_width = 0;
  slots_.reserve(c.slot_count);
  for (uint32_t slot_id = 0; slot_id < c.slot_count; ++slot_id) {
    auto task_slot = std::make_unique<Slot>();
    auto slot = std::make_unique<SpeculativeSlot>();
    if (!block_draft()) {
      slot->dummy_hidden = torch::zeros({1, c.hidden_size}, options);
    }
    status = SlotBuffer::create(
        base_capacity, device_.unwrap(), slot->target_prefill);
    if (!status.ok()) {
      return status;
    }
    MtpInputSpec prefill_spec{c.model, context_hidden_size()};
    prefill_spec.context_only = block_draft();
    status = SlotBuffer::create_mtp_input(
        prefill_spec, hidden_dtype_, device_.unwrap(), slot->draft_prefill);
    if (!status.ok()) {
      return status;
    }
    const MtpInputSpec validate_spec{c.model,
                                     0,
                                     c.logical_block_size,
                                     width - 1,
                                     MtpInvocationKind::VALIDATE,
                                     0,
                                     c.enable_mla};
    status = SlotBuffer::create_mtp_input(
        validate_spec, hidden_dtype_, device_.unwrap(), slot->validate_input);
    if (!status.ok()) {
      return status;
    }
    status = slot->validate_input->configure_sampling(target_capacity);
    if (!status.ok()) {
      return status;
    }
    status = context_create(*context_, count, slot->state);
    if (!status.ok()) {
      return status;
    }
    const auto indices = options.dtype(torch::kInt32);
    slot->draft_sample_indices = torch::arange(draft_samples, indices);
    if (block_draft()) {
      slot->draft_block_indices = torch::arange(expanded, indices)
                                      .view({count, width})
                                      .narrow(1, 1, width - 1)
                                      .reshape({-1});
    } else {
      slot->draft_first_indices = torch::arange(count, indices).mul(2).add(1);
    }
    slot->greedy_do_sample =
        torch::zeros({draft_samples}, options.dtype(torch::kBool));
    slot->target_token_storage =
        torch::empty({expanded}, options.dtype(torch::kInt64));
    slot->proposal_storage =
        torch::empty({count * (width - 1)}, options.dtype(torch::kInt64));
    slot->accepted_mask_storage =
        torch::empty({expanded}, options.dtype(torch::kBool));
    slot->host_hidden_storage =
        torch::empty({count, context_hidden_size()}, pinned);
    if (block_draft()) {
      slot->context_slots_storage =
          torch::empty({expanded}, options.dtype(torch::kInt32));
    }
    slot->upper_positions.reserve(count);
    slot->upper_kv_lengths.reserve(count);
    auto& bootstrap = slot->bootstrap;
    bootstrap.host_rows = torch::empty({count}, pinned.dtype(torch::kInt64));
    bootstrap.row_storage = torch::empty({count}, options.dtype(torch::kInt64));
    bootstrap.host_tokens = torch::empty({count}, pinned.dtype(torch::kInt64));
    bootstrap.token_storage =
        torch::empty({count}, options.dtype(torch::kInt64));
    for (auto* state : {&bootstrap.host_state, &bootstrap.storage}) {
      const auto state_options =
          state == &bootstrap.host_state ? pinned : options;
      state->positions =
          torch::empty({count}, state_options.dtype(torch::kInt32));
      state->kv_seq_lens =
          torch::empty({count}, state_options.dtype(torch::kInt32));
      if (!block_draft()) {
        state->hidden = torch::empty({count, 2, c.hidden_size}, state_options);
        state->repair_required =
            torch::zeros({count}, state_options.dtype(torch::kBool));
      }
    }
    const uint32_t draft_calls = block_draft() ? 1 : width - 1;
    slot->drafts.reserve(draft_calls);
    for (uint32_t step = 0; step < draft_calls; ++step) {
      auto invocation = std::make_unique<SpeculativeSlot::Draft>();
      auto spec = validate_spec;
      spec.kind = block_draft() ? MtpInvocationKind::BLOCK_DRAFT
                                : MtpInvocationKind::DRAFT;
      spec.draft_step = step;
      spec.enable_mla = !block_draft();
      spec.mask_token_id =
          block_draft() ? speculative_capacity_->mask_token_id : 0;
      status = SlotBuffer::create_mtp_input(
          spec, hidden_dtype_, device_.unwrap(), invocation->input);
      if (!status.ok()) {
        return status;
      }
      invocation->token_storage =
          torch::empty({draft_samples}, options.dtype(torch::kInt64));
      const uint32_t rows_per_sequence =
          invocation->input->input_scratch_->rows_per_sequence_;
      if (!block_draft()) {
        invocation->graph_batch_sizes = decode_graph_batch_sizes(
            max_batch_size / width * rows_per_sequence);
        invocation->captured_graph_batch_sizes.reserve(
            invocation->graph_batch_sizes.size());
      }
      if (step != 0 || (!block_draft() && max_batch_size != 0)) {
        invocation->hidden_storage =
            torch::empty({count * rows_per_sequence, c.hidden_size}, options);
        if (step != 0 && speculative_capacity_->reuse_topk) {
          invocation->topk_storage =
              torch::empty({count, 1, speculative_capacity_->index_topk},
                           options.dtype(torch::kInt32));
        }
      }
      slot->drafts.emplace_back(std::move(invocation));
    }
    task_slot->input_ready = reusable_event(device_.unwrap());
    task_slot->output_ready = reusable_event(device_.unwrap());
    task_slot->speculative = std::move(slot);
    slots_.emplace_back(std::move(task_slot));
  }
  return Status();
}

Status TaskExecutionPipeline::plan_parallel(const ParallelInput& input,
                                            uint32_t dp_size,
                                            uint32_t dp_rank,
                                            uint32_t local_tokens,
                                            bool local_decode,
                                            ParallelInput& output,
                                            bool& run_models,
                                            bool& decode) {
  if (dp_size == 0 || dp_rank >= dp_size ||
      local_tokens > std::numeric_limits<int32_t>::max()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Invalid Task DP topology.");
  }
  ParallelInput plan;
  if (dp_size == 1 && input.dp_global_token_nums.empty()) {
    if (!input.raw_dp_global_token_nums.empty() ||
        !input.dp_is_decode.empty() ||
        !input.dp_global_kv_max_seq_lens.empty() ||
        !input.dp_global_batch_generations.empty() ||
        !input.dp_global_sequence_nums.empty()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Task DP summaries require token counts.");
    }
    plan.dp_global_token_nums = {static_cast<int32_t>(local_tokens)};
    plan.raw_dp_global_token_nums = plan.dp_global_token_nums;
    plan.dp_is_decode = {local_decode ? 1 : 0};
  } else {
    if (input.dp_global_token_nums.size() != dp_size ||
        input.raw_dp_global_token_nums.size() != dp_size ||
        input.dp_is_decode.size() != dp_size ||
        (!input.dp_global_kv_max_seq_lens.empty() &&
         input.dp_global_kv_max_seq_lens.size() != dp_size) ||
        (!input.dp_global_batch_generations.empty() &&
         input.dp_global_batch_generations.size() != dp_size) ||
        (!input.dp_global_sequence_nums.empty() &&
         input.dp_global_sequence_nums.size() != dp_size) ||
        std::any_of(input.dp_global_sequence_nums.begin(),
                    input.dp_global_sequence_nums.end(),
                    [](int32_t count) { return count < 0; }) ||
        input.dp_global_token_nums[dp_rank] !=
            static_cast<int32_t>(local_tokens) ||
        !std::equal(input.dp_global_token_nums.begin(),
                    input.dp_global_token_nums.end(),
                    input.raw_dp_global_token_nums.begin()) ||
        std::any_of(input.dp_global_token_nums.begin(),
                    input.dp_global_token_nums.end(),
                    [](int32_t count) { return count < 0; }) ||
        std::any_of(input.dp_is_decode.begin(),
                    input.dp_is_decode.end(),
                    [](int32_t phase) { return phase != 0 && phase != 1; }) ||
        std::any_of(input.dp_global_kv_max_seq_lens.begin(),
                    input.dp_global_kv_max_seq_lens.end(),
                    [](int32_t length) { return length < 0; }) ||
        (local_tokens != 0 &&
         input.dp_is_decode[dp_rank] != static_cast<int32_t>(local_decode))) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Task requires aligned unpadded DP counts and phases.");
    }
    plan.dp_global_token_nums.assign(input.dp_global_token_nums.begin(),
                                     input.dp_global_token_nums.end());
    plan.raw_dp_global_token_nums = plan.dp_global_token_nums;
    plan.dp_is_decode.assign(input.dp_is_decode.begin(),
                             input.dp_is_decode.end());
    plan.dp_global_kv_max_seq_lens.assign(
        input.dp_global_kv_max_seq_lens.begin(),
        input.dp_global_kv_max_seq_lens.end());
    plan.dp_global_batch_generations.assign(
        input.dp_global_batch_generations.begin(),
        input.dp_global_batch_generations.end());
    plan.dp_global_sequence_nums.assign(input.dp_global_sequence_nums.begin(),
                                        input.dp_global_sequence_nums.end());
  }
  bool all_decode = true;
  bool active = false;
  for (uint32_t rank = 0; rank < dp_size; ++rank) {
    if (plan.dp_global_token_nums[rank] == 0) {
      continue;
    }
    active = true;
    all_decode = all_decode && plan.dp_is_decode[rank] == 1;
  }
  run_models = active;
  decode = active && all_decode;
  output = std::move(plan);
  return Status();
}

Status TaskExecutionPipeline::validate_bootstrap(
    const ForwardInput& input) const {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  const auto& rows = input.input_params.embedding.mtp_bootstrap_row_idxes;
  const auto& hidden = input.input_params.embedding.mtp_bootstrap_embeddings;
  if (rows.empty()) {
    if (hidden.defined() && hidden.numel() != 0) {
      return invalid("MTP bootstrap hidden requires explicit sequence rows.");
    }
    return Status();
  }
  if (!batch.forward_type.is_decode() || !hidden.defined() ||
      !hidden.device().is_cpu() || hidden.dim() != 2 ||
      hidden.size(/*dim=*/0) != static_cast<int64_t>(rows.size()) ||
      hidden.size(/*dim=*/1) != context_hidden_size() ||
      !hidden.is_floating_point()) {
    return invalid("MTP bootstrap requires CPU hidden rows for Decode.");
  }
  std::vector<uint8_t> seen(host.q_seq_lens.size(), /*value=*/0);
  for (const int32_t row : rows) {
    if (row < 0 || static_cast<uint64_t>(row) >= seen.size() || seen[row] ||
        host.token_ids[row] < 0) {
      return invalid(
          "MTP bootstrap rows must be unique and have known tokens.");
    }
    seen[row] = 1;
  }
  return Status();
}

Status TaskExecutionPipeline::validate_input(SpeculativeSlot& slot,
                                             const ForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  Status status = slot.target_prefill->validate(host, batch);
  if (!status.ok()) {
    return status;
  }

  const auto& c = speculative_capacity_->common;
  const uint64_t count = host.q_seq_lens.size();
  if (batch.num_actual_sequences != count ||
      input.input_params.embedding.embedding_ids.size() > count ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size() ||
      batch.is_graph_warmup || input.sampling_params.return_probs ||
      (!input.sampling_params.logprobs &&
       input.sampling_params.max_top_logprobs != 0) ||
      (!batch.forward_type.is_decode() && !batch.forward_type.is_prefill() &&
       !batch.forward_type.is_chunked_prefill() &&
       !batch.forward_type.is_mixed() &&
       batch.forward_type.value() != BatchForwardType::EMPTY)) {
    return invalid(
        "MTP requires unpadded ordinary sampling and sequence keys.");
  }
  auto& base_parallel = slot.target_prefill->model_params().parallel;
  status = plan_parallel(input.input_params.parallel,
                         c.dp_size,
                         c.dp_rank,
                         static_cast<uint32_t>(host.token_ids.size()),
                         batch.forward_type.is_decode(),
                         base_parallel,
                         slot.run_models,
                         slot.decode);
  if (!status.ok()) {
    return status;
  }
  const uint32_t width = speculative_capacity_->num_speculative_tokens + 1;
  for (const int32_t tokens : base_parallel.dp_global_token_nums) {
    if (static_cast<uint64_t>(tokens) * width >
        std::numeric_limits<int32_t>::max()) {
      return invalid("MTP DP token expansion exceeds int32 capacity.");
    }
  }
  slot.validate_input->model_params().parallel =
      expanded_parallel(base_parallel, slot.decode ? width : 1, false);
  for (uint32_t step = 0; step < slot.drafts.size(); ++step) {
    const uint32_t scale = !slot.decode    ? 1
                           : block_draft() ? width
                           : step == 0     ? 2
                                           : 1;
    slot.drafts[step]->input->model_params().parallel =
        expanded_parallel(base_parallel, scale, slot.decode && block_draft());
  }
  if (slot.decode &&
      input.input_params.embedding.embedding_ids.size() != count) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP decode requires one embedding id per model row.");
  }
  slot.target_padded_batch_size =
      task_graph_batch_size(slot.validate_input->model_params().parallel,
                            captured_graph_batch_sizes_);
  for (uint32_t step = 0; step < slot.drafts.size(); ++step) {
    auto& draft = *slot.drafts[step];
    draft.padded_batch_size = task_graph_batch_size(
        draft.input->model_params().parallel, draft.captured_graph_batch_sizes);
  }
  slot.rows = static_cast<uint32_t>(count);
  slot.model_tokens = static_cast<uint32_t>(host.token_ids.size());
  status = slot.target_prefill->validate_sampling(input.sampling_params,
                                                  slot.model_tokens);
  if (!status.ok()) {
    return status;
  }
  if (slot.decode &&
      (host.token_ids.size() != count ||
       std::any_of(host.q_seq_lens.begin(),
                   host.q_seq_lens.end(),
                   [](int32_t length) { return length != 1; }))) {
    return invalid("MTP Decode requires one base token per sequence.");
  }
  status = validate_bootstrap(input);
  if (!status.ok()) {
    return status;
  }
  if (!slot.run_models) {
    return Status();
  }
  if (kv_caches_.empty() || kv_caches_.front().empty() ||
      draft_->kv_caches.empty() || draft_->kv_caches.front().empty()) {
    return Status(StatusCode::UNAVAILABLE, "MTP requires Target and Draft KV.");
  }
  if (count == 0) {
    return Status();
  }
  const int64_t blocks =
      std::min(kv_caches_.front().get_k_cache().size(/*dim=*/0),
               draft_->kv_caches.front().get_k_cache().size(/*dim=*/0));
  if (std::any_of(
          host.block_tables.begin(),
          host.block_tables.end(),
          [blocks](int32_t block) { return block < 0 || block >= blocks; })) {
    return invalid("MTP page table is outside allocated Target/Draft cache.");
  }
  slot.upper_positions.clear();
  slot.upper_kv_lengths.clear();
  uint64_t offset = 0;
  for (uint64_t row = 0; row < count; ++row) {
    const int64_t kv = host.kv_seq_lens[row];
    const int64_t q = host.q_seq_lens[row];
    // Cache hits may have q < kv even when scheduler chunking is disabled.
    if (kv > c.max_kv_seq_len ||
        (kv + c.logical_block_size - 1) / c.logical_block_size >
            host.block_table_width) {
      return invalid("MTP KV length or Prefill mode exceeds capacity.");
    }
    for (int64_t item = 0; item < q; ++item, ++offset) {
      const int64_t position = host.positions[offset];
      const int32_t token = host.token_ids[offset];
      if (position != kv - q + item || position < 0 ||
          position >= c.max_positions ||
          (token >= 0 && static_cast<uint32_t>(token) >= c.vocab_size) ||
          (!slot.decode && token < 0)) {
        return invalid("Invalid MTP base token or rotary position.");
      }
      const int32_t block = host.block_tables[row * host.block_table_width +
                                              position / c.logical_block_size];
      if (host.new_cache_slots[offset] !=
          static_cast<int64_t>(block) * c.logical_block_size +
              position % c.logical_block_size) {
        return invalid("MTP KV write slot does not match the page table.");
      }
    }
    if (slot.decode) {
      // Scheduler's one pending placeholder can represent K+1 accepted tokens.
      const int64_t pending =
          host.token_ids[row] < 0
              ? speculative_capacity_->num_speculative_tokens
              : 0;
      const int64_t upper_position = host.positions[row] + pending;
      const int64_t upper_kv = kv + pending;
      if (upper_position + speculative_capacity_->num_speculative_tokens >=
              c.max_positions ||
          upper_kv + speculative_capacity_->num_speculative_tokens >
              c.max_kv_seq_len ||
          host.positions[row] < 1) {
        return invalid(
            "MTP speculative position upper bound exceeds capacity.");
      }
      slot.upper_positions.push_back(static_cast<int32_t>(upper_position));
      slot.upper_kv_lengths.push_back(static_cast<int32_t>(upper_kv));
    } else if (kv >= c.max_positions || kv >= c.max_kv_seq_len) {
      return invalid("MTP Prefill must leave capacity for its next token.");
    }
  }
  for (const int32_t token : input.input_params.embedding.extra_token_ids) {
    if (token < -1 ||
        (token >= 0 && static_cast<uint32_t>(token) >= c.vocab_size)) {
      return invalid("Invalid MTP Prefill extra token.");
    }
  }
  if (!slot.decode) {
    return SlotBuffer::plan_mtp_prefill(
        *slot.draft_prefill,
        host,
        batch,
        input.input_params.embedding.extra_token_ids,
        input.sampling_params);
  }
  auto conservative = host;
  conservative.positions = slot.upper_positions;
  conservative.kv_seq_lens = slot.upper_kv_lengths;
  status =
      SlotBuffer::plan_mtp_decode(*slot.validate_input, conservative, batch);
  if (!status.ok()) {
    return status;
  }
  for (auto& invocation : slot.drafts) {
    status =
        SlotBuffer::plan_mtp_decode(*invocation->input, conservative, batch);
    if (!status.ok()) {
      return status;
    }
  }
  return Status();
}

Status TaskExecutionPipeline::plan_sampling(SpeculativeSlot& slot,
                                            const ForwardInput& input) {
  const auto& base = input.sampling_params;
  slot.samples = base.sample_idxes.defined() ? base.sample_idxes.numel() : 0;
  if (slot.rows == 0 || !slot.decode) {
    return Status();
  }
  if (slot.samples != slot.rows || !base.selected_token_idxes.defined() ||
      base.selected_token_idxes.numel() != slot.rows) {
    return invalid("MTP Decode must sample every base row.");
  }
  const auto* selected = base.selected_token_idxes.const_data_ptr<int32_t>();
  const auto* samples = base.sample_idxes.const_data_ptr<int32_t>();
  for (uint32_t row = 0; row < slot.rows; ++row) {
    if (selected[row] != static_cast<int32_t>(row) ||
        samples[row] != static_cast<int32_t>(row)) {
      return invalid("MTP Decode sampling rows must be in sequence order.");
    }
  }
  const int64_t width = speculative_capacity_->num_speculative_tokens + 1;
  slot.host_validate_sampling = base;
  for (const auto member : kRepeatedRows) {
    if ((base.*member).defined()) {
      slot.host_validate_sampling.*member =
          (base.*member).repeat_interleave(width, 0);
    }
  }
  const auto indices =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  slot.host_validate_sampling.selected_token_idxes =
      torch::arange(slot.rows * width, indices);
  slot.host_validate_sampling.sample_idxes =
      slot.host_validate_sampling.selected_token_idxes;
  // Rejection sampling produces the final log probabilities once.
  slot.host_validate_sampling.logprobs = false;
  slot.host_validate_sampling.max_top_logprobs = 0;
  return slot.validate_input->validate_sampling(slot.host_validate_sampling,
                                                slot.rows * width);
}

void TaskExecutionPipeline::prepare_bootstrap(SpeculativeSlot& slot,
                                              const ForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  auto& b = slot.bootstrap;
  const int64_t count =
      input.input_params.embedding.mtp_bootstrap_row_idxes.size();
  b.rows = prefix(b.row_storage, count);
  b.tokens = prefix(b.token_storage, count);
  b.state.hidden =
      block_draft() ? torch::Tensor() : prefix(b.storage.hidden, count);
  b.state.positions = prefix(b.storage.positions, count);
  b.state.kv_seq_lens = prefix(b.storage.kv_seq_lens, count);
  b.state.repair_required = block_draft()
                                ? torch::Tensor()
                                : prefix(b.storage.repair_required, count);
  if (count == 0) {
    return;
  }
  auto host_rows = prefix(b.host_rows, count);
  auto host_tokens = prefix(b.host_tokens, count);
  auto host_positions = prefix(b.host_state.positions, count);
  auto host_kv = prefix(b.host_state.kv_seq_lens, count);
  for (int64_t index = 0; index < count; ++index) {
    const int32_t row =
        input.input_params.embedding.mtp_bootstrap_row_idxes[index];
    host_rows.data_ptr<int64_t>()[index] = row;
    host_tokens.data_ptr<int64_t>()[index] = host.token_ids[row];
    host_positions.data_ptr<int32_t>()[index] = host.positions[row];
    host_kv.data_ptr<int32_t>()[index] = host.kv_seq_lens[row];
  }
  if (!block_draft()) {
    auto host_hidden = prefix(b.host_state.hidden, count);
    auto host_repair = prefix(b.host_state.repair_required, count);
    host_hidden.select(/*dim=*/1, /*index=*/0).zero_();
    host_hidden.select(/*dim=*/1, /*index=*/1)
        .copy_(input.input_params.embedding.mtp_bootstrap_embeddings);
    copy_input(b.state.hidden, host_hidden, prepare_stream_);
    copy_input(b.state.repair_required, host_repair, prepare_stream_);
  }
  copy_input(b.rows, host_rows, prepare_stream_);
  copy_input(b.tokens, host_tokens, prepare_stream_);
  copy_input(b.state.positions, host_positions, prepare_stream_);
  copy_input(b.state.kv_seq_lens, host_kv, prepare_stream_);
}

void TaskExecutionPipeline::apply_bootstrap(SpeculativeSlot& slot) {
  const auto& b = slot.bootstrap;
  if (b.rows.numel() == 0) {
    return;
  }
  const auto& state = slot.state->state_;
  slot.state->tokens_.index_copy_(/*dim=*/0, b.rows, b.tokens);
  if (!block_draft()) {
    state.previous_tokens.index_copy_(/*dim=*/0, b.rows, b.tokens);
    state.hidden.index_copy_(/*dim=*/0, b.rows, b.state.hidden);
    state.repair_required.index_copy_(
        /*dim=*/0, b.rows, b.state.repair_required);
  }
  state.positions.index_copy_(/*dim=*/0, b.rows, b.state.positions);
  state.kv_seq_lens.index_copy_(/*dim=*/0, b.rows, b.state.kv_seq_lens);
}

Status TaskExecutionPipeline::warmup_speculative_graphs(
    const ForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  CHECK(input.input_params.meta.is_graph_warmup);
  if (graph_batch_sizes_.empty()) {
    return Status();
  }
  // Reuse scheduler-owned warmup KV and each SpeculativeSlot's final model
  // input views.
  for (auto& task_slot : slots_) {
    auto* slot = task_slot->speculative.get();
    Status status = validate_input(*slot, input);
    if (!status.ok()) {
      return status;
    }
  }
  const uint32_t padded_batch_size = task_graph_batch_size(
      slots_.front()->speculative->validate_input->model_params().parallel,
      graph_batch_sizes_);
  const auto bucket = std::lower_bound(captured_graph_batch_sizes_.begin(),
                                       captured_graph_batch_sizes_.end(),
                                       padded_batch_size);
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  if (padded_batch_size != 0 && (bucket == captured_graph_batch_sizes_.end() ||
                                 *bucket != padded_batch_size)) {
    for (auto& task_slot : slots_) {
      auto* slot = task_slot->speculative.get();
      auto& binding = *slot->validate_input;
      Status status;
      if (slot->rows == 0) {
        status = binding.prepare_decode_padded(
            {},
            {BatchForwardType::EMPTY, 0, batch.batch_id},
            padded_batch_size,
            prepare_stream_);
      } else {
        auto conservative = host;
        conservative.positions = slot->upper_positions;
        conservative.kv_seq_lens = slot->upper_kv_lengths;
        status = SlotBuffer::prepare_planned_mtp_decode(*slot->validate_input,
                                                        conservative,
                                                        batch,
                                                        prepare_stream_,
                                                        padded_batch_size);
      }
      CHECK(status.ok()) << status.message();
      CHECK_EQ(prepare_stream_.synchronize(), 0);

      pad_task_parallel(binding.model_params().parallel, padded_batch_size);
      executor_.prepare_attention_metadata(kv_caches_, binding.model_params());
      executor_.warmup_prepared_graph(binding.tokens(),
                                      binding.positions(),
                                      kv_caches_,
                                      binding.model_params());
    }
    CHECK_EQ(task_stream_.synchronize(), 0);
    captured_graph_batch_sizes_.insert(bucket, padded_batch_size);
  }
  // Different sequence counts can share a target bucket while requiring
  // different draft buckets. Always visit the draft bindings independently.
  if (!block_draft()) {
    warmup_draft_graphs(input);
  }
  return Status();
}

void TaskExecutionPipeline::warmup_draft_graphs(const ForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  for (auto& task_slot : slots_) {
    auto* slot = task_slot->speculative.get();
    for (uint32_t step = 0; step < slot->drafts.size(); ++step) {
      auto& invocation = *slot->drafts[step];
      const auto& parallel = invocation.input->model_params().parallel;
      const uint32_t padded_batch_size =
          task_graph_batch_size(parallel, invocation.graph_batch_sizes);
      auto& captured = invocation.captured_graph_batch_sizes;
      const auto bucket =
          std::lower_bound(captured.begin(), captured.end(), padded_batch_size);
      if (padded_batch_size == 0 ||
          (bucket != captured.end() && *bucket == padded_batch_size)) {
        continue;
      }
      auto& binding = *invocation.input;
      Status status;
      if (slot->rows == 0) {
        status = binding.prepare_decode_padded(
            {},
            {BatchForwardType::EMPTY, 0, batch.batch_id},
            padded_batch_size,
            prepare_stream_);
      } else {
        auto conservative = host;
        conservative.positions = slot->upper_positions;
        conservative.kv_seq_lens = slot->upper_kv_lengths;
        status = SlotBuffer::prepare_planned_mtp_decode(*invocation.input,
                                                        conservative,
                                                        batch,
                                                        prepare_stream_,
                                                        padded_batch_size);
      }
      CHECK(status.ok()) << status.message();
      prepare_draft_state(*slot, step, /*warmup=*/true);
      CHECK_EQ(prepare_stream_.synchronize(), 0);
      pad_task_parallel(binding.model_params().parallel, padded_batch_size);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
      draft_->executor.warmup_prepared_graph(binding.tokens(),
                                             binding.positions(),
                                             draft_->kv_caches,
                                             binding.model_params());
      CHECK_EQ(task_stream_.synchronize(), 0);
      captured.insert(bucket, padded_batch_size);
    }
  }
}

void TaskExecutionPipeline::prepare_draft_state(SpeculativeSlot& slot,
                                                uint32_t step,
                                                bool warmup) {
  auto& invocation = *slot.drafts[step];
  auto& binding = *invocation.input;
  auto& params = binding.model_params();
  params.mtp_topk_state.reset();
  if (step == 0 && !params.enable_graph) {
    params.embedding.input_embedding =
        slot.rows == 0
            ? slot.dummy_hidden
            : slot.state->state_.hidden.view(
                  {2 * slot.rows, speculative_capacity_->common.hidden_size});
    return;
  }
  const int64_t logical_rows =
      slot.rows * invocation.input->input_scratch_->rows_per_sequence_;
  const int64_t physical_rows = binding.tokens().numel();
  invocation.next_hidden = prefix(invocation.hidden_storage, logical_rows);
  params.embedding.input_embedding =
      prefix(invocation.hidden_storage, physical_rows);
  auto guard = prepare_stream_.set_stream_guard();
  // Launch supplies logical rows after their producer completes. Padding has
  // no producer, so initialize it here; capture warmup initializes every row.
  const int64_t clear_from = warmup ? 0 : logical_rows;
  params.embedding.input_embedding
      .narrow(/*dim=*/0, clear_from, physical_rows - clear_from)
      .zero_();
  if (step != 0 && speculative_capacity_->reuse_topk) {
    invocation.next_topk = prefix(invocation.topk_storage, slot.rows);
    auto topk = prefix(invocation.topk_storage, physical_rows);
    topk.narrow(/*dim=*/0, clear_from, physical_rows - clear_from).zero_();
    params.mtp_topk_state = MtpTopkState::from_tensor(std::move(topk));
  }
}

Status TaskExecutionPipeline::prepare_speculative(uint32_t slot_id,
                                                  const ForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  if (slot_id >= slots_.size()) {
    return invalid("Invalid MTP SpeculativeSlot index.");
  }
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  Status status = validate_input(slot, input);
  if (!status.ok()) {
    return status;
  }
  status = plan_sampling(slot, input);
  if (!status.ok()) {
    return status;
  }
  c10::DeviceGuard guard(device_.unwrap());
  if (kv_caches_.empty() || kv_caches_.front().empty()) {
    return Status(StatusCode::UNAVAILABLE,
                  "MTP embedding resources are not allocated.");
  }
  const int64_t embedding_rows = kv_caches_.front().get_k_cache().size(0);
  if (embedding_rows <= 0 ||
      embedding_rows > std::numeric_limits<int32_t>::max()) {
    return invalid("Invalid allocated MTP embedding capacity.");
  }
  if (context_->request_ids_.empty()) {
    std::unique_ptr<MtpContextStorage> context;
    status = create_mtp_context(static_cast<uint32_t>(embedding_rows),
                                block_draft() ? 0 : capacity_.hidden_size,
                                hidden_dtype_,
                                device_.unwrap(),
                                context);
    if (!status.ok()) {
      return status;
    }
    // Views retain the storage object's address, not its allocation contents.
    *context_ = std::move(*context);
    LOG(INFO) << "Task pipeline embedding context: rows=" << embedding_rows
              << ", device_bytes=" << mtp_context_bytes(*context_);
  }
  CHECK_EQ(context_->request_ids_.size(), embedding_rows);
  if (slot.decode && slot.rows != 0) {
    status = context_prepare_decode(
        *slot.state,
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        input.input_params.embedding.mtp_bootstrap_row_idxes,
        prepare_stream_);
  } else {
    status = context_prepare_prefill(
        *slot.state,
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        input.input_params.embedding.extra_token_ids,
        slot.rows == 0 ? std::span<const uint32_t>()
                       : slot.draft_prefill->input_scratch_->planned_rows_,
        prepare_stream_);
  }
  if (!status.ok()) {
    return status;
  }
  status = slot.target_prefill->prepare_sampling(
      input.sampling_params, slot.model_tokens, prepare_stream_);
  CHECK(status.ok()) << status.message();
  const uint32_t width =
      slot.decode ? speculative_capacity_->num_speculative_tokens + 1 : 1;
  status = slot.samples == 0 ? slot.target_prefill->bind_result(0, 0, 0, false)
                             : slot.target_prefill->bind_result(
                                   slot.samples,
                                   width,
                                   input.sampling_params.logprobs
                                       ? input.sampling_params.max_top_logprobs
                                       : 0,
                                   input.sampling_params.logprobs);
  CHECK(status.ok()) << status.message();
  slot.is_warmup = input.input_params.meta.is_graph_warmup;
  if (speculative_capacity_->common.dp_size > 1) {
    VLOG(1)
        << "MTP DP task batch=" << batch.batch_id
        << " dp_rank=" << speculative_capacity_->common.dp_rank
        << " local_tokens=" << slot.model_tokens << " local_rows=" << slot.rows
        << " samples=" << slot.samples << " run_models=" << slot.run_models
        << " speculative_decode=" << slot.decode
        << " padded_batch_size=" << slot.target_padded_batch_size
        << " local_decode=" << batch.forward_type.is_decode() << " base_counts="
        << slot.target_prefill->model_params().parallel.dp_global_token_nums
        << " first_draft_counts="
        << slot.drafts.front()
               ->input->model_params()
               .parallel.dp_global_token_nums
        << " validate_counts="
        << slot.validate_input->model_params().parallel.dp_global_token_nums;
  }
  slot.host_hidden =
      prefix(slot.host_hidden_storage, slot.decode ? 0 : slot.samples);
  if (slot.rows == 0) {
    if (slot.run_models) {
      prepare_empty_shard(slot, batch.batch_id);
    }
    record(prepare_stream_, slots_[slot_id]->input_ready);
    return Status();
  }
  if (slot.decode) {
    prepare_bootstrap(slot, input);
    auto conservative = host;
    conservative.positions = slot.upper_positions;
    conservative.kv_seq_lens = slot.upper_kv_lengths;
    status =
        SlotBuffer::prepare_planned_mtp_decode(*slot.validate_input,
                                               conservative,
                                               batch,
                                               prepare_stream_,
                                               slot.target_padded_batch_size);
    CHECK(status.ok()) << status.message();
    status = slot.validate_input->prepare_sampling(
        slot.host_validate_sampling, slot.rows * width, prepare_stream_);
    CHECK(status.ok()) << status.message();
    slot.target_tokens = prefix(slot.target_token_storage, slot.rows * width);
    slot.proposal = prefix(slot.proposal_storage, slot.rows * (width - 1))
                        .view({slot.rows, width - 1});
    slot.accepted_mask = prefix(slot.accepted_mask_storage, slot.rows * width)
                             .view({slot.rows, width});
    if (block_draft()) {
      slot.context_slots =
          prefix(slot.context_slots_storage, slot.rows * width);
    }
    SampleOutput target_output;
    target_output.next_tokens = slot.target_tokens;
    slot.target_sampling =
        Sampling{slot.validate_input->sampling_params(), target_output, {}};
    const auto& base = slot.target_prefill->sampling_params();
    slot.rejection = std::make_unique<RejectionSampler>(
        base.do_sample,
        base.all_random_sample,
        base.all_greedy_sample,
        base.logprobs,
        base.max_top_logprobs,
        speculative_capacity_->fused_rejection);
    for (uint32_t step = 0; step < slot.drafts.size(); ++step) {
      auto& invocation = *slot.drafts[step];
      status =
          SlotBuffer::prepare_planned_mtp_decode(*invocation.input,
                                                 conservative,
                                                 batch,
                                                 prepare_stream_,
                                                 invocation.padded_batch_size);
      CHECK(status.ok()) << status.message();
      auto& binding = *invocation.input;
      if (block_draft()) {
        binding.model_params().embedding.input_embedding = torch::Tensor();
      } else {
        prepare_draft_state(slot, step, /*warmup=*/false);
      }
      SamplingParameters params = base;
      params.logprobs = false;
      params.max_top_logprobs = 0;
      const uint32_t draft_rows = slot.rows * (block_draft() ? width - 1 : 1);
      if (block_draft()) {
        auto stream_guard = prepare_stream_.set_stream_guard();
        for (const auto member : kRepeatedRows) {
          if ((base.*member).defined()) {
            params.*member = (base.*member).repeat_interleave(width - 1, 0);
          }
        }
        params.selected_token_idxes =
            prefix(slot.draft_block_indices, draft_rows);
        params.sample_idxes = prefix(slot.draft_sample_indices, draft_rows);
      } else if (step == 0) {
        params.selected_token_idxes =
            prefix(slot.draft_first_indices, slot.rows);
      }
      if (speculative_capacity_->draft_sampling_mode ==
          DraftSamplingMode::GREEDY) {
        params.do_sample = prefix(slot.greedy_do_sample, draft_rows);
        params.all_greedy_sample = true;
        params.all_random_sample = false;
      }
      params.return_probs = !params.all_greedy_sample;
      SampleOutput draft_output;
      draft_output.next_tokens = prefix(invocation.token_storage, draft_rows);
      invocation.sampling = Sampling{std::move(params), draft_output, {}};

      pad_task_parallel(binding.model_params().parallel,
                        invocation.padded_batch_size);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
    }
    auto& binding = *slot.validate_input;

    pad_task_parallel(binding.model_params().parallel,
                      slot.target_padded_batch_size);
    executor_.prepare_attention_metadata(kv_caches_, binding.model_params());
  } else {
    status = slot.target_prefill->prepare(host, batch, prepare_stream_);
    CHECK(status.ok()) << status.message();
    status = SlotBuffer::prepare_planned_mtp_prefill(
        *slot.draft_prefill, host, batch, prepare_stream_);
    CHECK(status.ok()) << status.message();
    slot.target_sampling =
        Sampling{slot.target_prefill->sampling_params(),
                 sample_views(slot.target_prefill->device_result()),
                 {}};
    auto& target_input = *slot.target_prefill;

    executor_.prepare_attention_metadata(kv_caches_,
                                         target_input.model_params());
    if (!block_draft()) {
      auto& draft_input = *slot.draft_prefill;
      draft_input.model_params().parallel =
          slot.target_prefill->model_params().parallel;
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  draft_input.model_params());
    }
  }
  record(prepare_stream_, slots_[slot_id]->input_ready);
  return Status();
}

void TaskExecutionPipeline::prepare_empty_shard(SpeculativeSlot& slot,
                                                uint64_t batch_id) {
  // Physical rows participate in MoE collectives without a Sequence lease,
  // sampling or publication. Only reserved padding block zero is written.
  // Each zero-count peer executes one row,
  // including first Draft and Validate; only active peers expand by 2 or K+1.
  const auto prepare = [&](SlotBuffer& binding,
                           const ParallelInput& parallel,
                           TaskModel resource,
                           bool draft) {
    const uint32_t padded_batch_size =
        draft ? 0 : slot.target_padded_batch_size;
    const Status status = padded_batch_size == 0
                              ? binding.prepare_empty_shard(
                                    slot.decode, batch_id, prepare_stream_)
                              : binding.prepare_decode_padded(
                                    {},
                                    {BatchForwardType::EMPTY, 0, batch_id},
                                    padded_batch_size,
                                    prepare_stream_);
    CHECK(status.ok()) << status.message();
    binding.model_params().parallel = parallel;
    pad_task_parallel(binding.model_params().parallel, padded_batch_size);
    binding.model_params().mtp_topk_state.reset();
    binding.model_params().embedding.input_embedding =
        draft ? slot.dummy_hidden : torch::Tensor();
    resource.executor.prepare_attention_metadata(resource.kv_caches,
                                                 binding.model_params());
  };
  if (slot.decode) {
    if (block_draft()) {
      const std::array<int32_t, 1> zero{0};
      const std::array<int32_t, 1> one{1};
      const ModelInputHostView host{zero, zero, zero, one, one, one, zero, 1};
      const ModelInputBatch batch{BatchForwardType::DECODE, 1, batch_id};
      auto& invocation = *slot.drafts.front();
      Status status = SlotBuffer::prepare_mtp_decode(
          *invocation.input, host, batch, prepare_stream_);
      CHECK(status.ok()) << status.message();
      auto& binding = *invocation.input;

      binding.model_params().meta.actual_num_sequences = 0;
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
    } else {
      for (uint32_t step = 0; step < slot.drafts.size(); ++step) {
        auto& invocation = *slot.drafts[step];
        auto& binding = *invocation.input;
        const Status status =
            invocation.padded_batch_size == 0
                ? binding.prepare_empty_shard(
                      /*decode=*/true, batch_id, prepare_stream_)
                : binding.prepare_decode_padded(
                      {},
                      {BatchForwardType::EMPTY, 0, batch_id},
                      invocation.padded_batch_size,
                      prepare_stream_);
        CHECK(status.ok()) << status.message();
        prepare_draft_state(slot, step, /*warmup=*/false);

        pad_task_parallel(binding.model_params().parallel,
                          invocation.padded_batch_size);
        draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                    binding.model_params());
      }
    }
    prepare(*slot.validate_input,
            slot.validate_input->model_params().parallel,
            TaskModel{model_, executor_, kv_caches_},
            /*draft=*/false);
  } else {
    prepare(*slot.target_prefill,
            slot.target_prefill->model_params().parallel,
            TaskModel{model_, executor_, kv_caches_},
            /*draft=*/false);
    if (!block_draft()) {
      prepare(*slot.draft_prefill,
              slot.target_prefill->model_params().parallel,
              *draft_,
              /*draft=*/true);
    }
  }
}

void TaskExecutionPipeline::launch_empty_shard(SpeculativeSlot& slot) {
  const auto run = [](SlotBuffer& binding, TaskModel resource) {
    return resource.executor.forward(binding.tokens(),
                                     binding.positions(),
                                     resource.kv_caches,
                                     binding.model_params());
  };
  if (slot.decode) {
    for (auto& invocation : slot.drafts) {
      invocation->output = run(*invocation->input, *draft_);
    }
    slot.target_output =
        run(*slot.validate_input, TaskModel{model_, executor_, kv_caches_});
  } else {
    slot.target_output =
        run(*slot.target_prefill, TaskModel{model_, executor_, kv_caches_});
    if (block_draft()) {
      write_block_context(slot, /*prefill=*/true);
      return;
    }
    slot.drafts.front()->output = run(*slot.draft_prefill, *draft_);
  }
}

void TaskExecutionPipeline::launch_prefill(SpeculativeSlot& slot) {
  auto& target_input = *slot.target_prefill;
  slot.target_output = executor_.forward(target_input.tokens(),
                                         target_input.positions(),
                                         kv_caches_,
                                         target_input.model_params());
  if (slot.samples != 0) {
    slot.target_logits = model_.logits(
        slot.target_output.hidden_states,
        slot.target_prefill->sampling_params().selected_token_idxes);
    sample(*slot.target_sampling, slot.target_logits);
    slot.target_prefill->device_result().lengths.fill_(1);
    if (block_draft() || capacity_.cp_sampling_group != nullptr) {
      synchronize_samples(slot.target_prefill->device_result());
    } else {
      auto tokens = slot.target_prefill->device_result().tokens;
      synchronize_tokens(
          tokens, slot.target_prefill->sampling_params().all_greedy_sample);
    }
  }
  if (block_draft()) {
    SlotBuffer::initialize_mtp_context(
        *slot.draft_prefill,
        slot.target_output.aux_hidden_states,
        slot.target_prefill->device_result().tokens,
        *slot.state);
    write_block_context(slot, /*prefill=*/true);
    context_publish(*slot.state);
    return;
  }
  SlotBuffer::patch_mtp_prefill(*slot.draft_prefill,
                                slot.target_output.hidden_states,
                                slot.target_prefill->device_result().tokens,
                                *slot.state);
  auto& invocation = *slot.drafts.front();
  auto& draft_input = *slot.draft_prefill;
  invocation.output = draft_->executor.forward(draft_input.tokens(),
                                               draft_input.positions(),
                                               draft_->kv_caches,
                                               draft_input.model_params());
  context_publish(*slot.state);
}

void TaskExecutionPipeline::launch_decode(SpeculativeSlot& slot) {
  context_gather(*slot.state);
  apply_bootstrap(slot);
  const uint32_t width = speculative_capacity_->num_speculative_tokens + 1;
  if (block_draft()) {
    launch_block_draft(slot);
  } else {
    for (uint32_t step = 0; step < slot.drafts.size(); ++step) {
      auto& invocation = *slot.drafts[step];
      SlotBuffer::patch_mtp_decode(*invocation.input, *slot.state);
      auto& binding = *invocation.input;
      if (step == 0 && binding.model_params().enable_graph) {
        invocation.next_hidden.copy_(slot.state->state_.hidden.view(
            {2 * slot.rows, speculative_capacity_->common.hidden_size}));
      }
      if (step != 0) {
        const auto& previous = *slot.drafts[step - 1];
        prefix(binding.tokens(), slot.rows).copy_(previous.sampled.next_tokens);
        const int64_t previous_rows = step == 1 ? 2 : 1;
        invocation.next_hidden.copy_(
            previous.output.hidden_states
                .narrow(/*dim=*/0, /*start=*/0, slot.rows * previous_rows)
                .view({slot.rows,
                       previous_rows,
                       speculative_capacity_->common.hidden_size})
                .select(/*dim=*/1, previous_rows - 1));
        if (speculative_capacity_->reuse_topk) {
          CHECK(previous.output.mtp_topk_state != nullptr)
              << "GLM Draft did not publish its configured TopK state.";
          const auto topk = previous.output.mtp_topk_state->as_tensor();
          CHECK(topk.has_value())
              << "Prepared draft requires tensor-backed TopK state.";
          invocation.next_topk.copy_(
              topk->narrow(/*dim=*/0, /*start=*/0, slot.rows * previous_rows)
                  .view({slot.rows,
                         previous_rows,
                         1,
                         speculative_capacity_->index_topk})
                  .select(/*dim=*/1, previous_rows - 1));
        }
      }
      invocation.output = draft_->executor.forward(binding.tokens(),
                                                   binding.positions(),
                                                   draft_->kv_caches,
                                                   binding.model_params());
      invocation.logits = draft_->model.logits(
          invocation.output.hidden_states,
          invocation.sampling->params.selected_token_idxes);
      invocation.sampled = sample(*invocation.sampling, invocation.logits);
      synchronize_tokens(invocation.sampled.next_tokens,
                         invocation.sampling->params.all_greedy_sample);
      slot.proposal.select(/*dim=*/1, step)
          .copy_(invocation.sampled.next_tokens);
    }
  }
  SlotBuffer::patch_mtp_decode(*slot.validate_input, *slot.state);
  auto& validate = *slot.validate_input;
  validate.tokens()
      .narrow(/*dim=*/0, /*start=*/0, slot.rows * width)
      .view({slot.rows, width})
      .narrow(/*dim=*/1, /*start=*/1, width - 1)
      .copy_(slot.proposal);
  slot.target_output = executor_.forward(validate.tokens(),
                                         validate.positions(),
                                         kv_caches_,
                                         validate.model_params());
  slot.target_output.hidden_states = slot.target_output.hidden_states.narrow(
      /*dim=*/0, /*start=*/0, slot.rows * width);
  if (block_draft()) {
    slot.target_output.aux_hidden_states =
        slot.target_output.aux_hidden_states.narrow(
            /*dim=*/0, /*start=*/0, slot.rows * width);
  }
  slot.target_logits = model_.logits(
      slot.target_output.hidden_states,
      slot.validate_input->sampling_params().selected_token_idxes);
  sample(*slot.target_sampling, slot.target_logits);
  const auto target_tokens = slot.target_tokens.view({slot.rows, width});
  const auto bonus = target_tokens.narrow(/*dim=*/1, width - 1, /*length=*/1);
  const auto& base = slot.target_prefill->sampling_params();
  SampleOutput accepted;
  if (base.all_greedy_sample && !base.logprobs) {
    accepted.next_tokens =
        std::get<1>(RejectionSampler::greedy_sample_from_token_ids(
            slot.proposal,
            target_tokens.narrow(/*dim=*/1, /*start=*/0, width - 1),
            bonus,
            /*mask_out_rejected_tokens=*/true));
  } else {
    std::optional<torch::Tensor> draft_probs;
    if (slot.drafts.front()->sampled.probs.defined()) {
      std::vector<torch::Tensor> probabilities;
      probabilities.reserve(slot.drafts.size());
      for (const auto& invocation : slot.drafts) {
        probabilities.push_back(invocation->sampled.probs);
      }
      draft_probs = block_draft() ? slot.drafts.front()->sampled.probs
                                  : torch::stack(probabilities, /*dim=*/1);
    }
    accepted = slot.rejection->forward(
        DraftProposal(slot.proposal, draft_probs),
        slot.target_logits.view(
            {slot.rows, width, speculative_capacity_->common.vocab_size}),
        bonus,
        /*mask_out_rejected_tokens=*/true);
  }
  const auto& result = slot.target_prefill->device_result();
  result.tokens.copy_(accepted.next_tokens);
  if (result.logprobs.defined()) {
    result.logprobs.copy_(accepted.logprobs);
  }
  if (result.top_tokens.defined()) {
    result.top_tokens.copy_(accepted.top_tokens);
    result.top_logprobs.copy_(accepted.top_logprobs);
  }
  if (block_draft() || capacity_.cp_sampling_group != nullptr) {
    synchronize_samples(result);
  } else {
    auto tokens = result.tokens;
    synchronize_tokens(tokens, base.all_greedy_sample);
  }
  torch::ne_out(slot.accepted_mask, result.tokens, /*other=*/-1);
  auto lengths = result.lengths;
  torch::sum_out(
      lengths, slot.accepted_mask, {1}, /*keepdim=*/false, torch::kInt32);
  if (block_draft()) {
    write_block_context(slot, /*prefill=*/false);
  }
  const auto& context = block_draft() ? slot.target_output.aux_hidden_states
                                      : slot.target_output.hidden_states;
  context_advance(*slot.state,
                  result.tokens,
                  result.lengths,
                  context.view({slot.rows, width, context_hidden_size()}));
  context_publish(*slot.state);
}

void TaskExecutionPipeline::launch_block_draft(SpeculativeSlot& slot) {
  auto& invocation = *slot.drafts.front();
  SlotBuffer::patch_mtp_decode(*invocation.input, *slot.state);
  auto& input = *invocation.input;
  const int64_t steps = speculative_capacity_->num_speculative_tokens;
  const auto& selected = invocation.sampling->params.selected_token_idxes;
  SamplingParameters selector_sampling = slot.target_prefill->sampling_params();
  if (speculative_capacity_->draft_sampling_mode == DraftSamplingMode::GREEDY) {
    selector_sampling.all_greedy_sample = true;
    selector_sampling.all_random_sample = false;
    selector_sampling.do_sample = prefix(slot.greedy_do_sample, slot.rows);
  }
  if (speculative_capacity_->kind == SpeculativeTaskKind::DFLASH2) {
    slot.block_gumbel =
        sample_gumbel_noise(slot.rows,
                            steps,
                            speculative_capacity_->selector_top_k,
                            selector_sampling,
                            device_.unwrap());
    if (capacity_.sampling_group != nullptr &&
        capacity_.sampling_group->world_size() > 1) {
      capacity_.sampling_group->broadcast(slot.block_gumbel, /*root_rank=*/0);
    }
  }
  invocation.output = draft_->executor.forward(input.tokens(),
                                               input.positions(),
                                               draft_->kv_caches,
                                               input.model_params());
  invocation.logits =
      draft_->model.logits(invocation.output.hidden_states, selected);
  if (speculative_capacity_->kind == SpeculativeTaskKind::DFLASH2) {
    auto hidden = invocation.output.hidden_states.index_select(
        /*dim=*/0, selected.to(torch::kInt64));
    const auto candidates = draft_->model.dflash2_candidates(
        hidden.view(
            {slot.rows, steps, speculative_capacity_->common.hidden_size}),
        invocation.logits.view(
            {slot.rows, steps, speculative_capacity_->common.vocab_size}),
        slot.state->tokens_);
    auto sampled =
        sample_dflash2_path(candidates,
                            selector_sampling,
                            slot.block_gumbel,
                            speculative_capacity_->common.vocab_size,
                            speculative_capacity_->draft_sampling_mode ==
                                    DraftSamplingMode::PROBABILISTIC &&
                                !selector_sampling.all_greedy_sample);
    slot.proposal.copy_(sampled.token_ids);
    invocation.sampled.probs = std::move(sampled.dense_probs);
  } else {
    invocation.sampled = sample(*invocation.sampling, invocation.logits);
    slot.proposal.copy_(
        invocation.sampled.next_tokens.view({slot.rows, steps}));
  }
}

void TaskExecutionPipeline::synchronize_tokens(torch::Tensor& tokens,
                                               bool all_greedy) {
  if (all_greedy && capacity_.cp_sampling_group == nullptr) {
    return;
  }
  for (ProcessGroup* group :
       {capacity_.sampling_group, capacity_.cp_sampling_group}) {
    if (group == nullptr || group->world_size() <= 1) {
      continue;
    }
    CHECK(tokens.is_contiguous());
    group->broadcast(tokens, /*root_rank=*/0);
  }
}

void TaskExecutionPipeline::synchronize_samples(
    const TokenResultTensors& result) {
  for (const auto* tensor : {&result.tokens,
                             &result.logprobs,
                             &result.top_tokens,
                             &result.top_logprobs}) {
    if (tensor->defined()) {
      auto view = *tensor;
      synchronize_tokens(view, /*all_greedy=*/false);
    }
  }
}

void TaskExecutionPipeline::write_block_context(SpeculativeSlot& slot,
                                                bool prefill) {
  auto& input = prefill ? *slot.target_prefill : *slot.validate_input;
  const auto& hidden = slot.target_output.aux_hidden_states;
  CHECK(hidden.defined())
      << "Block draft requires captured target auxiliary hidden states.";
  CHECK_EQ(hidden.size(/*dim=*/1), context_hidden_size());
  torch::Tensor cache_slots =
      input.model_params().attention.device.new_cache_slots;
  torch::Tensor positions = input.positions();
  if (!prefill) {
    const int64_t rows =
        slot.rows * (speculative_capacity_->num_speculative_tokens + 1);
    slot.context_slots.copy_(cache_slots.narrow(/*dim=*/0, /*start=*/0, rows));
    slot.context_slots.view(slot.accepted_mask.sizes())
        .masked_fill_(slot.accepted_mask.logical_not(), /*value=*/-1);
    cache_slots = slot.context_slots;
    positions = positions.narrow(/*dim=*/0, /*start=*/0, rows);
  }
  // The serial task stream orders accepted-context publication before any
  // subsequent block reads the draft cache. Rejected rows disable the scatter;
  // no accepted length or token is copied to Host to build this operation.
  slot.drafts.front()->output = draft_->model.write_context_kv(
      hidden, positions, cache_slots, draft_->kv_caches, input.model_params());
  CHECK(slot.drafts.front()->output.hidden_states.defined());
}

void TaskExecutionPipeline::launch_speculative(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  CHECK(task_stream_.wait_event(slots_[slot_id]->input_ready));
  if (slot.rows == 0 && slot.run_models) {
    launch_empty_shard(slot);
  } else if (slot.rows != 0) {
    if (slot.decode) {
      launch_decode(slot);
    } else {
      launch_prefill(slot);
    }
  }
  record(task_stream_, slots_[slot_id]->output_ready);
  if (!slot.decode && slot.samples != 0) {
    CHECK(result_stream_.wait_event(slots_[slot_id]->output_ready));
    auto result_guard = result_stream_.set_stream_guard();
    const auto& hidden = slot.draft_prefill->input_scratch_->bootstrap_hidden_;
    CHECK_EQ(hidden.nbytes(), slot.host_hidden.nbytes());
    slot.host_hidden.copy_(hidden, /*non_blocking=*/true);
  }
  const Status status = slot.target_prefill->copy_result_to_host(
      result_stream_, slots_[slot_id]->output_ready);
  CHECK(status.ok()) << status.message();
}

ForwardOutput TaskExecutionPipeline::consume_speculative(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  c10::DeviceGuard guard(device_.unwrap());
  auto tokens = slot.target_prefill->take_result();
  ForwardOutput output;
  output.do_sample = slot.target_prefill->copy_cpu_do_sample();
  output.is_graph_warmup = slot.is_warmup;
  const bool prefill = !slot.decode;
  if (tokens.tokens.defined()) {
    output.sample_output.next_tokens =
        prefill ? tokens.tokens.squeeze(1) : tokens.tokens;
    if (tokens.logprobs.defined()) {
      output.logprobs = true;
      output.sample_output.logprobs =
          prefill ? tokens.logprobs.squeeze(1) : tokens.logprobs;
    }
    if (tokens.top_tokens.defined()) {
      output.max_top_logprobs = tokens.top_tokens.size(-1);
      output.sample_output.top_tokens =
          prefill ? tokens.top_tokens.squeeze(1) : tokens.top_tokens;
      output.sample_output.top_logprobs =
          prefill ? tokens.top_logprobs.squeeze(1) : tokens.top_logprobs;
    }
  }
  if (prefill && slot.samples != 0) {
    output.sample_output.embeddings =
        torch::empty(slot.host_hidden.sizes(),
                     slot.host_hidden.options().pinned_memory(false));
    output.sample_output.embeddings.copy_(slot.host_hidden);
  }
  release_outputs(slot);
  return output;
}

void TaskExecutionPipeline::discard_speculative(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  c10::DeviceGuard guard(device_.unwrap());
  slot.target_prefill->discard_result();
  release_outputs(slot);
}

void TaskExecutionPipeline::release_outputs(SpeculativeSlot& slot) {
  context_release(*slot.state);
  slot.draft_prefill->model_params().embedding.input_embedding =
      torch::Tensor();
  slot.target_output = ModelOutput();
  slot.target_logits = torch::Tensor();
  slot.target_sampling.reset();
  slot.rejection.reset();
  for (auto& invocation : slot.drafts) {
    invocation->output = ModelOutput();
    invocation->logits = torch::Tensor();
    invocation->sampled = SampleOutput();
    invocation->sampling.reset();
  }
  slot.host_validate_sampling = SamplingParameters();
  slot.block_gumbel = torch::Tensor();
}

uint64_t TaskExecutionPipeline::speculative_pinned_bytes() const {
  uint64_t bytes = 0;
  for (const auto& task_slot : slots_) {
    const auto& slot = *task_slot->speculative;
    bytes +=
        slot.target_prefill->pinned_bytes() +
        slot.draft_prefill->pinned_bytes() +
        slot.validate_input->pinned_bytes() + slot.state->host_rows_.nbytes() +
        slot.host_hidden_storage.nbytes() + slot.bootstrap.host_rows.nbytes() +
        slot.bootstrap.host_tokens.nbytes() +
        state_bytes(slot.bootstrap.host_state);
    for (const auto& invocation : slot.drafts) {
      bytes += invocation->input->pinned_bytes();
    }
  }
  return bytes;
}

uint64_t TaskExecutionPipeline::speculative_device_bytes() const {
  uint64_t bytes = mtp_context_bytes(*context_);
  for (const auto& task_slot : slots_) {
    const auto& slot = *task_slot->speculative;
    bytes +=
        slot.target_prefill->device_bytes() +
        slot.draft_prefill->device_bytes() +
        slot.validate_input->device_bytes() +
        slot.draft_sample_indices.nbytes() + slot.greedy_do_sample.nbytes() +
        slot.target_token_storage.nbytes() + slot.proposal_storage.nbytes() +
        slot.accepted_mask_storage.nbytes() +
        context_device_bytes(*slot.state) +
        slot.bootstrap.row_storage.nbytes() +
        slot.bootstrap.token_storage.nbytes() +
        state_bytes(slot.bootstrap.storage);
    for (const auto& invocation : slot.drafts) {
      bytes += invocation->input->device_bytes();
      bytes += invocation->token_storage.nbytes();
      for (const auto* tensor :
           {&invocation->hidden_storage, &invocation->topk_storage}) {
        bytes += tensor->defined() ? tensor->nbytes() : 0;
      }
    }
    for (const auto* tensor : {&slot.context_slots_storage,
                               &slot.draft_block_indices,
                               &slot.draft_first_indices,
                               &slot.dummy_hidden}) {
      bytes += tensor->defined() ? tensor->nbytes() : 0;
    }
  }
  return bytes;
}

const SampleOutput& TaskExecutionPipeline::sample(Sampling& sampling,
                                                  torch::Tensor& logits) {
  if (!sampling.params.sample_idxes.defined() ||
      sampling.params.sample_idxes.numel() == 0) {
    return sampling.output;
  }
  CHECK_EQ(logits.dim(), 2);
  CHECK_EQ(logits.size(/*dim=*/1), speculative_capacity_->common.vocab_size);
  CHECK_EQ(logits.size(/*dim=*/0),
           sampling.params.selected_token_idxes.numel());
  if (sampling.params.top_k.defined() && sampling.params.top_p.defined()) {
    sampling.params.top_p = sampling.params.top_p.to(logits.scalar_type());
  }
  sampling.sampled = sampler_.forward(logits, sampling.params);
  sampling.output.next_tokens.copy_(sampling.sampled.next_tokens);
  if (sampling.output.logprobs.defined()) {
    sampling.output.logprobs.copy_(sampling.sampled.logprobs);
  }
  if (sampling.output.top_tokens.defined()) {
    sampling.output.top_tokens.copy_(sampling.sampled.top_tokens);
    sampling.output.top_logprobs.copy_(sampling.sampled.top_logprobs);
  }
  sampling.output.probs = sampling.sampled.probs;
  return sampling.output;
}

namespace {
constexpr std::array<torch::Tensor MtpContextTensors::*, 5> kContextFields = {
    &MtpContextTensors::previous_tokens,
    &MtpContextTensors::hidden,
    &MtpContextTensors::positions,
    &MtpContextTensors::kv_seq_lens,
    &MtpContextTensors::repair_required};
}  // namespace

Status TaskExecutionPipeline::create_mtp_context(
    uint32_t capacity,
    uint32_t hidden_size,
    torch::ScalarType dtype,
    const torch::Device& device,
    std::unique_ptr<MtpContextStorage>& output) {
  const uint64_t element_bytes =
      dtype == torch::kFloat32 ? sizeof(float) : sizeof(uint16_t);
  if (capacity > std::numeric_limits<int32_t>::max() ||
      device.type() != Platform::type_torch() || !device.has_index() ||
      hidden_size > std::numeric_limits<int32_t>::max() ||
      (dtype != torch::kFloat16 && dtype != torch::kBFloat16 &&
       dtype != torch::kFloat32) ||
      static_cast<uint64_t>(capacity) * hidden_size >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
              (2 * element_bytes)) {
    return invalid("Invalid MTP embedding context capacity or dtype.");
  }
  auto context = std::make_unique<MtpContextStorage>();
  context->request_ids_.resize(capacity);
  context->published_.resize(capacity);
  c10::DeviceGuard guard(device);
  const auto options = torch::TensorOptions().device(device);
  context->tokens_ = torch::empty({capacity}, options.dtype(torch::kInt64));
  if (hidden_size != 0) {
    context->mtp_state_.previous_tokens =
        torch::empty({capacity}, options.dtype(torch::kInt64));
    context->mtp_state_.hidden =
        torch::empty({capacity, 2, hidden_size}, options.dtype(dtype));
    context->mtp_state_.repair_required =
        torch::empty({capacity}, options.dtype(torch::kBool));
  }
  context->mtp_state_.positions =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  context->mtp_state_.kv_seq_lens =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  output = std::move(context);
  return Status();
}

uint64_t TaskExecutionPipeline::mtp_context_bytes(
    const MtpContextStorage& cache) {
  return cache.tokens_.nbytes() + state_bytes(cache.mtp_state_);
}

// Block drafts keep their hidden context in KV. Their cross-round state only
// needs the next token and its position; MTP additionally keeps two hidden
// rows.
Status TaskExecutionPipeline::context_create(
    MtpContextStorage& context,
    uint32_t capacity,
    std::unique_ptr<MtpContextView>& output) {
  if (!context.tokens_.defined() || !context.mtp_state_.positions.defined() ||
      !context.mtp_state_.kv_seq_lens.defined() || capacity == 0 ||
      capacity > std::numeric_limits<int32_t>::max()) {
    return invalid("Speculative context requires a fixed Slot capacity.");
  }
  auto storage = std::make_unique<MtpContextView>();
  auto& view = *storage;
  view.context_ = &context;
  view.capacity_ = capacity;
  view.sorted_ids_.reserve(capacity);

  c10::DeviceGuard guard(context.tokens_.device());
  const auto options = context.tokens_.options();
  view.host_rows_ =
      torch::empty({capacity}, options.device(torch::kCPU).pinned_memory(true));
  view.device_rows_ = torch::empty({capacity}, options);
  view.token_storage_ = torch::empty({capacity}, options);
  view.storage_.positions =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  view.storage_.kv_seq_lens =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  const bool history = context.mtp_state_.hidden.defined();
  // Both algorithms gather the last accepted token. Only MTP also gathers
  // its predecessor and tracks full-acceptance repair.
  view.index_storage_ = torch::empty({history ? 2 : 1, capacity}, options);
  if (history) {
    view.storage_.previous_tokens = torch::empty({capacity}, options);
    view.storage_.hidden =
        torch::empty({capacity, 2, context.mtp_state_.hidden.size(/*dim=*/2)},
                     context.mtp_state_.hidden.options());
    view.storage_.repair_required =
        torch::empty({capacity}, options.dtype(torch::kBool));
    view.mask_storage_ = torch::empty({capacity}, options.dtype(torch::kBool));
  }
  view.identity_rows_.reserve(capacity);
  view.model_to_embedding_.resize(capacity);
  view.publication_mask_.resize(capacity);
  view.bootstrap_mask_.resize(capacity);
  output = std::move(storage);
  return Status();
}

Status TaskExecutionPipeline::context_prepare(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    bool read_published_state,
    const Stream& stream) {
  if (embedding_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP binding exceeds fixed Slot capacity.");
  }
  view.identity_rows_.clear();
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    view.identity_rows_.push_back(row);
  }
  return context_prepare_impl(view,
                              embedding_ids,
                              request_ids,
                              view.identity_rows_,
                              read_published_state,
                              stream,
                              {});
}

Status TaskExecutionPipeline::context_prepare_prefill(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    std::span<const int32_t> extra_token_ids,
    std::span<const uint32_t> published_sequence_rows,
    const Stream& stream) {
  if (extra_token_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP prefill exceeds fixed Slot capacity.");
  }
  // ForwardInputBuilder emits embedding ids only for completed chunks, in
  // model order. Publication follows sampling order, which may differ.
  uint32_t completed = 0;
  for (uint32_t row = 0; row < extra_token_ids.size(); ++row) {
    view.model_to_embedding_[row] =
        extra_token_ids[row] == -1 ? static_cast<int32_t>(completed++) : -1;
  }
  if (embedding_ids.size() != completed ||
      published_sequence_rows.size() != completed) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP prefill requires one embedding id per completed row.");
  }
  view.identity_rows_.clear();
  for (const uint32_t row : published_sequence_rows) {
    if (row >= extra_token_ids.size() || view.model_to_embedding_[row] < 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP prefill publication requires completed model rows.");
    }
    view.identity_rows_.push_back(view.model_to_embedding_[row]);
  }
  return context_prepare_impl(view,
                              embedding_ids,
                              request_ids,
                              view.identity_rows_,
                              /*read_published_state=*/false,
                              stream,
                              {});
}

Status TaskExecutionPipeline::context_prepare_decode(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    std::span<const int32_t> bootstrap_rows,
    const Stream& stream) {
  if (embedding_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP decode exceeds fixed Slot capacity.");
  }
  view.identity_rows_.clear();
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    view.identity_rows_.push_back(row);
  }
  return context_prepare_impl(view,
                              embedding_ids,
                              request_ids,
                              view.identity_rows_,
                              /*read_published_state=*/true,
                              stream,
                              bootstrap_rows);
}

Status TaskExecutionPipeline::context_prepare_impl(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    std::span<const uint32_t> published_sequence_rows,
    bool read_published_state,
    const Stream& stream,
    std::span<const int32_t> bootstrap_rows) {
  if (view.prepared_ || embedding_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP binding is occupied or exceeds fixed Slot capacity.");
  }
  if (embedding_ids.size() != request_ids.size()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP embedding and request ids must be row aligned.");
  }
  view.sorted_ids_.assign(embedding_ids.begin(), embedding_ids.end());
  std::sort(view.sorted_ids_.begin(), view.sorted_ids_.end());
  if (!view.sorted_ids_.empty() &&
      (view.sorted_ids_.front() <= 0 ||
       static_cast<uint64_t>(view.sorted_ids_.back()) >=
           view.context_->request_ids_.size() ||
       std::adjacent_find(view.sorted_ids_.begin(), view.sorted_ids_.end()) !=
           view.sorted_ids_.end())) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "MTP requires unique allocated embedding rows; zero is padding.");
  }
  if (stream.get_stream()->device_index() !=
      view.context_->tokens_.device().index()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP binding stream uses a different device.");
  }
  std::fill(
      view.publication_mask_.begin(), view.publication_mask_.end(), uint8_t{0});
  for (const uint32_t row : published_sequence_rows) {
    if (row >= embedding_ids.size() || view.publication_mask_[row] != 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP Prefill publication rows must be unique model rows.");
    }
    view.publication_mask_[row] = 1;
  }
  std::fill(
      view.bootstrap_mask_.begin(), view.bootstrap_mask_.end(), uint8_t{0});
  for (const int32_t row : bootstrap_rows) {
    if (row < 0 || static_cast<uint64_t>(row) >= embedding_ids.size() ||
        view.bootstrap_mask_[row] != 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP bootstrap rows must be unique model rows.");
    }
    view.bootstrap_mask_[row] = 1;
  }
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    const int32_t id = embedding_ids[row];
    if (read_published_state && view.bootstrap_mask_[row] == 0 &&
        (!view.context_->published_[id] ||
         view.context_->request_ids_[id] != request_ids[row])) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP decode has no published context for this request.");
    }
  }
  // Validation is complete. Host metadata only promises FIFO publication;
  // Prepare never overwrites the device row while an earlier Task uses it.
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    const int32_t id = embedding_ids[row];
    view.context_->request_ids_[id] = request_ids[row];
    view.context_->published_[id] = view.publication_mask_[row];
  }
  c10::DeviceGuard guard(view.context_->tokens_.device());
  view.prepared_ = true;
  view.read_published_state_ = read_published_state;
  const int64_t count = static_cast<int64_t>(published_sequence_rows.size());
  int64_t* host = view.host_rows_.data_ptr<int64_t>();
  for (int64_t index = 0; index < count; ++index) {
    host[index] = embedding_ids[published_sequence_rows[index]];
  }
  view.rows_ = view.device_rows_.narrow(/*dim=*/0, /*start=*/0, count);
  view.tokens_ = view.token_storage_.narrow(/*dim=*/0, /*start=*/0, count);
  for (const auto member : kContextFields) {
    const torch::Tensor& storage = view.storage_.*member;
    view.state_.*member = storage.defined()
                              ? storage.narrow(/*dim=*/0, /*start=*/0, count)
                              : torch::Tensor();
  }
  view.last_indices_ = view.index_storage_.select(/*dim=*/0, /*index=*/0)
                           .narrow(/*dim=*/0, /*start=*/0, count)
                           .view({count, 1});
  view.current_token_output_ = view.tokens_.unsqueeze(/*dim=*/1);
  if (view.state_.hidden.defined()) {
    view.previous_indices_ = view.index_storage_.select(/*dim=*/0, /*index=*/1)
                                 .narrow(/*dim=*/0, /*start=*/0, count)
                                 .view({count, 1});
    const int64_t hidden_size = view.state_.hidden.size(/*dim=*/2);
    view.last_hidden_indices_ =
        view.last_indices_.unsqueeze(/*dim=*/2).expand({count, 1, hidden_size});
    view.previous_hidden_indices_ =
        view.previous_indices_.unsqueeze(/*dim=*/2).expand(
            {count, 1, hidden_size});
    view.previous_token_output_ =
        view.state_.previous_tokens.unsqueeze(/*dim=*/1);
    view.current_hidden_output_ =
        view.state_.hidden.narrow(/*dim=*/1, /*start=*/1, /*length=*/1);
    view.previous_hidden_output_ =
        view.state_.hidden.narrow(/*dim=*/1, /*start=*/0, /*length=*/1);
    view.no_previous_ =
        view.mask_storage_.narrow(/*dim=*/0, /*start=*/0, count);
    view.no_previous_mask_ = view.no_previous_.view({count, 1, 1});
  }
  if (count != 0) {
    auto stream_guard = stream.set_stream_guard();
    view.rows_.copy_(view.host_rows_.narrow(/*dim=*/0, /*start=*/0, count),
                     /*non_blocking=*/true);
  }
  return Status();
}

void TaskExecutionPipeline::context_gather(MtpContextView& view) {
  if (!view.prepared_ || view.tokens_.numel() == 0) {
    return;
  }
  CHECK(view.read_published_state_);
  torch::index_select_out(
      view.tokens_, view.context_->tokens_, /*dim=*/0, view.rows_);
  for (const auto member : kContextFields) {
    const torch::Tensor& source = view.context_->mtp_state_.*member;
    if (!source.defined()) {
      continue;
    }
    torch::index_select_out(view.state_.*member, source, /*dim=*/0, view.rows_);
  }
}

void TaskExecutionPipeline::context_publish(MtpContextView& view) {
  if (!view.prepared_ || view.tokens_.numel() == 0) {
    return;
  }
  view.context_->tokens_.index_copy_(/*dim=*/0, view.rows_, view.tokens_);
  for (const auto member : kContextFields) {
    const torch::Tensor& destination = view.context_->mtp_state_.*member;
    if (!destination.defined()) {
      continue;
    }
    destination.index_copy_(/*dim=*/0, view.rows_, view.state_.*member);
  }
}

void TaskExecutionPipeline::context_advance(
    MtpContextView& view,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_lengths,
    const torch::Tensor& target_hidden) {
  if (!view.prepared_ || view.tokens_.numel() == 0) {
    return;
  }
  const int64_t count = view.tokens_.numel();
  CHECK_EQ(accepted_tokens.device(), view.tokens_.device());
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kInt64);
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.size(/*dim=*/0), count);
  CHECK_GE(accepted_tokens.size(/*dim=*/1), 2);
  CHECK_EQ(accepted_lengths.device(), view.tokens_.device());
  CHECK_EQ(accepted_lengths.scalar_type(), torch::kInt32);
  CHECK_EQ(accepted_lengths.dim(), 1);
  CHECK_EQ(accepted_lengths.numel(), count);
  if (view.state_.hidden.defined()) {
    CHECK_EQ(target_hidden.device(), view.tokens_.device());
    CHECK_EQ(target_hidden.scalar_type(), view.state_.hidden.scalar_type());
    CHECK_EQ(target_hidden.dim(), 3);
    CHECK_EQ(target_hidden.size(/*dim=*/0), count);
    CHECK_EQ(target_hidden.size(/*dim=*/1), accepted_tokens.size(/*dim=*/1));
    CHECK_EQ(target_hidden.size(/*dim=*/2), view.state_.hidden.size(/*dim=*/2));
  }
  // The rejection sampler guarantees one valid prefix of length [1, width].
  // Keep lengths on Device; only the statically known width is read on Host.
  view.last_indices_.copy_(accepted_lengths.unsqueeze(/*dim=*/1));
  view.last_indices_.sub_(/*other=*/1);
  torch::gather_out(view.current_token_output_,
                    accepted_tokens,
                    /*dim=*/1,
                    view.last_indices_);
  if (view.state_.hidden.defined()) {
    view.previous_indices_.copy_(view.last_indices_);
    view.previous_indices_.sub_(/*other=*/1).clamp_min_(/*min=*/0);
    torch::gather_out(view.previous_token_output_,
                      accepted_tokens,
                      /*dim=*/1,
                      view.previous_indices_);
    torch::gather_out(view.current_hidden_output_,
                      target_hidden,
                      /*dim=*/1,
                      view.last_hidden_indices_);
    torch::gather_out(view.previous_hidden_output_,
                      target_hidden,
                      /*dim=*/1,
                      view.previous_hidden_indices_);
    torch::le_out(view.no_previous_, accepted_lengths, /*other=*/1);
    view.previous_hidden_output_.masked_fill_(view.no_previous_mask_,
                                              /*value=*/0);
    torch::eq_out(view.state_.repair_required,
                  accepted_lengths,
                  accepted_tokens.size(/*dim=*/1));
  }
  view.state_.positions.add_(accepted_lengths);
  view.state_.kv_seq_lens.add_(accepted_lengths);
}

void TaskExecutionPipeline::context_release(MtpContextView& view) {
  view.prepared_ = false;
  view.read_published_state_ = false;
}

uint64_t TaskExecutionPipeline::context_device_bytes(
    const MtpContextView& view) {
  return view.device_rows_.nbytes() + view.token_storage_.nbytes() +
         state_bytes(view.storage_) + view.index_storage_.nbytes() +
         (view.mask_storage_.defined() ? view.mask_storage_.nbytes() : 0);
}

}  // namespace xllm
