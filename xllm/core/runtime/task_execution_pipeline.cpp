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

#include "core/runtime/task_execution_pipeline.h"

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <limits>

#include "core/platform/platform.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

}  // namespace

TaskExecutionPipeline::TaskExecutionPipeline(ThreadPool& state_executor,
                                             CausalLM& model,
                                             Executor& executor,
                                             std::vector<KVCache>& kv_caches,
                                             LlmTaskCapacity capacity)
    : model_(model),
      executor_(executor),
      kv_caches_(kv_caches),
      capacity_(std::move(capacity)),
      device_(model.device()),
      prepare_stream_(device_.unwrap()),
      task_stream_(device_.unwrap()),
      result_stream_(device_.unwrap()),
      state_executor_(state_executor),
      execution_(capacity_.slot_count),
      completed_(capacity_.slot_count) {}

Status TaskExecutionPipeline::create(
    ThreadPool& state_executor,
    CausalLM& model,
    Executor& executor,
    std::vector<KVCache>& kv_caches,
    const LlmTaskCapacity& capacity,
    std::unique_ptr<TaskExecutionPipeline>& output) {
  if (state_executor.size() != 1) {
    return invalid("Task pipeline requires one state thread.");
  }
  if (!executor.supports_prepared_attention_metadata()) {
    return invalid(
        "LLM task pipeline requires supported Python prepared metadata.");
  }
  if (capacity.slot_count == 0 || capacity.slot_count > 2 ||
      capacity.model.max_sequences == 0 ||
      capacity.model.max_sequences > std::numeric_limits<int32_t>::max() ||
      model.device().type() != Platform::type_torch() ||
      !model.device().has_index() || capacity.max_kv_seq_len == 0 ||
      capacity.max_positions == 0 || capacity.block_size == 0 ||
      capacity.max_positions > std::numeric_limits<int32_t>::max() ||
      capacity.block_size > std::numeric_limits<int32_t>::max() ||
      capacity.max_kv_seq_len > capacity.max_positions) {
    return invalid("Invalid ordinary LLM capacity or indexed device.");
  }
  c10::DeviceGuard guard(model.device());
  auto pipeline =
      std::unique_ptr<TaskExecutionPipeline>(new TaskExecutionPipeline(
          state_executor, model, executor, kv_caches, capacity));
  const uint32_t rows = capacity.model.max_sequences;
  if (capacity.slot_count == 2) {
    pipeline->previous_tokens_ = torch::empty(
        {rows},
        torch::TensorOptions().device(model.device()).dtype(torch::kInt64));
  }
  pipeline->slots_.reserve(capacity.slot_count);
  for (uint32_t slot_id = 0; slot_id < capacity.slot_count; ++slot_id) {
    auto slot = std::make_unique<Slot>();
    Status status = SlotBuffer::create({capacity.model,
                                        capacity.max_unique_tokens,
                                        capacity.vocab_size,
                                        capacity.max_top_logprobs,
                                        capacity.parameter_dtype,
                                        capacity.enable_mla},
                                       model.device(),
                                       slot->buffer);
    if (!status.ok()) {
      return status;
    }
    slot->input_ready = std::make_shared<StreamEvent>(model.device().type());
    slot->output_ready = std::make_shared<StreamEvent>(model.device().type());
    pipeline->slots_.emplace_back(std::move(slot));
  }
  // Initialization is outside serving. Complete allocation-stream writes
  // before handing storage to either execution thread.
  CHECK_EQ(pipeline->device_.current_stream()->synchronize(), 0);
  // Start execution only after every Slot and initialization write is ready.
  folly::Promise<std::thread::id> promise;
  auto future = promise.getFuture();
  state_executor.schedule([promise = std::move(promise)]() mutable {
    promise.setValue(std::this_thread::get_id());
  });
  pipeline->state_thread_id_ = std::move(future).get();
  pipeline->launch_thread_ =
      std::thread([runtime = pipeline.get()] { runtime->launch_loop(); });
  output = std::move(pipeline);
  return Status();
}

void TaskExecutionPipeline::check_external_thread() const {
  CHECK(std::this_thread::get_id() != state_thread_id_);
  CHECK(std::this_thread::get_id() != launch_thread_.get_id());
}

Status TaskExecutionPipeline::validate_input(const ForwardInput& source) {
  const auto& params = source.input_params;
  const auto& host = params.attention.host;
  const auto& meta = params.meta;
  const auto& embedding = params.embedding;
  const auto& copy = params.block_copy;
  if (source.device_tensors_ready || source.metadata_ready_event != nullptr ||
      !source.retained_device_tensors.empty() ||
      source.step_decode.has_value() || source.skip_sampling_for_logits_only ||
      source.return_selected_hidden || !source.transfer_kv_infos.empty() ||
      !source.json_object_states.empty() ||
      !source.json_object_state_snapshots.empty() ||
      !source.json_object_invalid_draft.empty() ||
      !source.json_object_errors.empty() || params.is_spec_verify ||
      params.prefill_without_cache || !params.linear_state_cache_ops.empty() ||
      !params.linear_state_validity_mask.empty() ||
      !params.multi_block_tables.empty() || params.mtp_topk_state != nullptr ||
      params.mtp_shifted_token_ids.defined() ||
      params.num_accepted_tokens.defined() ||
      !params.num_accepted_tokens_host.empty() ||
      !std::holds_alternative<std::monostate>(params.rec_params) ||
      embedding.input_embedding.defined() ||
      embedding.mtp_shifted_token_ids.defined() ||
      !embedding.mtp_bootstrap_row_idxes.empty() ||
      embedding.mtp_bootstrap_embeddings.defined() ||
      !copy.swap_blocks.empty() || copy.src_block_indices.defined() ||
      copy.dst_block_indices.defined() || copy.cum_sum.defined() ||
      params.multimodal.mm_data.valid() ||
      !params.multimodal.deep_stacks.empty() ||
      params.parallel.cp_plan.enabled() ||
      params.parallel.layer_wise_load_synchronizer != nullptr ||
      params.parallel.dp_global_token_nums.size() > 1 ||
      params.expert.expert_load_data.defined() ||
      params.expert.expert_array.defined() ||
      params.expert.eplb_decode_token_mask.defined() ||
      !host.ring_cur_seqlen.empty() || !host.ring_cache_seqlen.empty()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Task pipeline requires ordinary CPU LLM input without "
                  "speculative, structured, transfer or recurrent state.");
  }
  // Full-attention rows use -1 to keep this transport field row-aligned.
  // Only those inactive sentinels may be omitted from the prepared program.
  const auto inactive = [](int32_t id) { return id == -1; };
  const auto& linear_ids = embedding.linear_state_ids;
  const auto& linear_indices = embedding.linear_state_indices;
  if ((!linear_ids.empty() &&
       (linear_ids.size() != host.q_seq_lens.size() ||
        !std::all_of(linear_ids.begin(), linear_ids.end(), inactive))) ||
      (linear_indices.defined() &&
       (!is_cpu_int_tensor(linear_indices, /*dimensions=*/1) ||
        linear_indices.numel() !=
            static_cast<int64_t>(host.q_seq_lens.size()) ||
        !std::all_of(int_span(linear_indices).begin(),
                     int_span(linear_indices).end(),
                     inactive)))) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Task pipeline does not support active linear-attention state.");
  }
  const auto& tokens = source.host_token_ids();
  const auto& slots = params.attention.device.new_cache_slots;
  if (source.input_host_buffer_has_layout ||
      (tokens.defined() && !is_cpu_int_tensor(tokens, /*dimensions=*/1)) ||
      (source.host_positions().defined() &&
       !is_cpu_int_tensor(source.host_positions(), /*dimensions=*/1)) ||
      (host.block_tables.defined() &&
       !is_cpu_int_tensor(host.block_tables, /*dimensions=*/2)) ||
      (host.new_cache_slots.empty() && slots.defined() &&
       !is_cpu_int_tensor(slots, /*dimensions=*/1)) ||
      params.graph.tiling_data.defined() ||
      params.graph.input_tokens_override.defined() ||
      params.graph.use_expanded_decode_for_spec_verify_attention ||
      params.parallel.layer_synchronizer != nullptr) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "Task pipeline requires unpacked CPU input and no graph overrides.");
  }
  const auto& positions = source.host_positions();
  const bool empty = !tokens.defined() || tokens.numel() == 0;
  const int64_t rows = static_cast<int64_t>(host.q_seq_lens.size());
  if (meta.num_sequences < 0 || meta.actual_num_sequences < 0 ||
      meta.num_sequences != rows ||
      (meta.actual_num_sequences != 0 && meta.actual_num_sequences != rows) ||
      (empty && rows != 0) ||
      (!empty && (!is_cpu_int_tensor(tokens, /*dimensions=*/1) ||
                  !is_cpu_int_tensor(positions, /*dimensions=*/1) ||
                  !is_cpu_int_tensor(host.block_tables, /*dimensions=*/2) ||
                  host.block_tables.size(/*dim=*/0) != rows ||
                  host.block_tables.size(/*dim=*/1) >
                      std::numeric_limits<int32_t>::max()))) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Task pipeline requires unpadded CPU int32 model input.");
  }
  return Status();
}

TaskSubmission TaskExecutionPipeline::submit(const ForwardInput& input) {
  check_external_thread();
  ForwardInput unpacked;
  const ForwardInput* source = &input;
  if (input.input_host_buffer_has_layout) {
    CHECK(detail::unpack_from_input_host_buffer(
        input, device_.unwrap(), unpacked));
    source = &unpacked;
  }
  const Status status = validate_input(*source);
  if (!status.ok()) {
    return TaskSubmission{status, 0};
  }
  folly::Promise<TaskSubmission> promise;
  auto future = promise.getFuture();
  state_executor_.schedule(
      [this, source, promise = std::move(promise)]() mutable {
        if (accepted_.size() == capacity_.slot_count) {
          promise.setValue(TaskSubmission{
              Status(StatusCode::RESOURCE_EXHAUSTED,
                     "All task Slots still hold unconsumed results."),
              0});
          return;
        }
        // With at most two Slots, one remaining Task identifies the other
        // free Slot. The accepted FIFO is the only Host ownership record.
        const uint32_t slot_id =
            accepted_.empty() ? 0U : 1U - accepted_.front().slot_id;
        CHECK_LT(next_task_id_, std::numeric_limits<uint64_t>::max());
        Status status = prepare(slot_id, *source);
        if (!status.ok()) {
          promise.setValue(TaskSubmission{std::move(status), 0});
          return;
        }
        const SlotTicket ticket{slot_id, next_task_id_++};
        accepted_.emplace_back(ticket);
        execution_.push(ticket);
        VLOG(1) << "Task pipeline accepted task_id=" << ticket.task_id
                << " slot_id=" << ticket.slot_id
                << " pending=" << accepted_.size();
        promise.setValue(TaskSubmission{Status(), ticket.task_id});
      });
  return std::move(future).get();
}

folly::Future<TaskResult> TaskExecutionPipeline::take_result_async(
    uint64_t task_id) {
  return take_result_impl(task_id);
}

folly::Future<TaskResult> TaskExecutionPipeline::take_result_async() {
  return take_result_impl(std::nullopt);
}

folly::Future<TaskResult> TaskExecutionPipeline::take_result_impl(
    std::optional<uint64_t> expected_task_id) {
  check_external_thread();
  folly::Promise<TaskResult> promise;
  auto future = promise.getFuture();
  state_executor_.schedule([this,
                            expected_task_id,
                            promise = std::move(promise)]() mutable {
    if (accepted_.empty() ||
        (expected_task_id.has_value() &&
         (*expected_task_id == 0 ||
          *expected_task_id != accepted_.front().task_id))) {
      promise.setValue(TaskResult{Status(StatusCode::INVALID_ARGUMENT,
                                         "No matching oldest unconsumed Task."),
                                  {}});
      return;
    }
    const SlotTicket ticket = wait_completed_front();
    auto output = consume(ticket.slot_id);
    accepted_.pop_front();
    VLOG(1) << "Task pipeline consumed task_id=" << ticket.task_id
            << " slot_id=" << ticket.slot_id << " pending=" << accepted_.size();
    promise.setValue(TaskResult{Status(), std::move(output), ticket.task_id});
  });
  return future;
}

TaskExecutionPipeline::SlotTicket
TaskExecutionPipeline::wait_completed_front() {
  CHECK(!accepted_.empty());
  const SlotTicket ticket = completed_.pop();
  CHECK_EQ(ticket.task_id, accepted_.front().task_id);
  CHECK_EQ(ticket.slot_id, accepted_.front().slot_id);
  return ticket;
}

void TaskExecutionPipeline::launch_loop() {
  while (true) {
    const SlotTicket ticket = execution_.pop();
    if (ticket.task_id == 0) {
      return;
    }
    launch(ticket.slot_id);
    // The completion queue covers every Slot, including unrequested results.
    completed_.push(ticket);
  }
}

TaskExecutionPipeline::~TaskExecutionPipeline() {
  // Creation can fail before the Launch thread starts. No work exists yet.
  if (!launch_thread_.joinable()) {
    return;
  }
  check_external_thread();
  folly::Promise<folly::Unit> promise;
  auto future = promise.getFuture();
  state_executor_.schedule([promise = std::move(promise)]() mutable {
    promise.setValue(folly::unit);
  });
  std::move(future).get();
  execution_.push(SlotTicket{});
  launch_thread_.join();
  while (!accepted_.empty()) {
    const SlotTicket ticket = wait_completed_front();
    discard(ticket.slot_id);
    accepted_.pop_front();
  }
  CHECK(completed_.empty());
}

Status TaskExecutionPipeline::validate(
    const Slot& slot,
    const ModelInputHostView& model,
    const ModelInputBatch& batch,
    const SamplingParameters& sampling) const {
  Status status = slot.buffer->validate(
      model,
      batch,
      sampling,
      capacity_.slot_count == 2 ? accepted_tail_.sample_rows : 0,
      prepare_stream_);
  if (!status.ok()) {
    return status;
  }
  const auto& host = model;
  if (batch.num_actual_sequences != host.q_seq_lens.size() ||
      batch.is_graph_warmup || sampling.return_probs ||
      (!sampling.logprobs && sampling.max_top_logprobs != 0)) {
    return invalid(
        "Ordinary LLM requires actual eager rows and token results.");
  }
  if (host.token_ids.empty()) {
    return Status();
  }
  if (kv_caches_.empty() || kv_caches_.front().empty()) {
    return Status(StatusCode::UNAVAILABLE, "KV cache is not allocated.");
  }
  const int64_t blocks = kv_caches_.front().get_k_cache().size(/*dim=*/0);
  const int64_t block_size = capacity_.block_size;
  int64_t offset = 0;
  for (uint32_t row = 0; row < host.q_seq_lens.size(); ++row) {
    const int32_t q = host.q_seq_lens[row];
    const int32_t kv = host.kv_seq_lens[row];
    // Prefix-cache hits also have q < kv when scheduler chunking is disabled.
    // SlotBuffer validates the row lengths against the batch forward type.
    if (static_cast<uint32_t>(kv) > capacity_.max_kv_seq_len ||
        (kv + block_size - 1) / block_size > host.block_table_width) {
      return invalid("KV length exceeds the LLM contract.");
    }
    for (int32_t index = 0; index < q; ++index, ++offset) {
      const int32_t position = host.positions[offset];
      const int32_t token = host.token_ids[offset];
      if (position != kv - q + index || position < 0 ||
          static_cast<uint32_t>(position) >= capacity_.max_positions ||
          (token >= 0 &&
           static_cast<uint32_t>(token) >= capacity_.vocab_size)) {
        return invalid("Invalid ordinary token or rotary position.");
      }
      const int32_t block = host.block_tables[static_cast<uint64_t>(row) *
                                                  host.block_table_width +
                                              position / block_size];
      if (host.new_cache_slots[offset] !=
          block * block_size + position % block_size) {
        return invalid("KV write slot does not match the row's page table.");
      }
    }
  }
  if (std::any_of(
          host.block_tables.begin(),
          host.block_tables.end(),
          [blocks](int32_t block) { return block < 0 || block >= blocks; })) {
    return invalid("KV page index is outside allocated cache.");
  }
  return Status();
}

Status TaskExecutionPipeline::prepare(uint32_t slot_id,
                                      const ForwardInput& input) {
  if (slot_id >= slots_.size()) {
    return invalid("Invalid LLM Slot index.");
  }
  Slot& slot = *slots_[slot_id];
  const auto& input_params = input.input_params;
  const auto& host = input_params.attention.host;
  const auto& meta = input_params.meta;
  const auto& tokens = input.host_token_ids();
  const auto& positions = input.host_positions();
  const auto& cache_slots = input_params.attention.device.new_cache_slots;
  const bool empty = !tokens.defined() || tokens.numel() == 0;
  const uint32_t rows = static_cast<uint32_t>(host.q_seq_lens.size());
  // BatchInputBuilder leaves actual_num_sequences unset before Worker prepare.
  // ProfileManager also marks ordinary eager warmup as is_graph_warmup.
  // It is an output-metrics marker here, not permission for graph execution.
  // The Slot preserves it in the detached response; execution remains eager.
  const ModelInputBatch batch{
      meta.batch_forward_type, rows, meta.batch_id, false};
  const ModelInputHostView model{
      int_span(tokens),
      int_span(positions),
      host.new_cache_slots.empty()
          ? int_span(cache_slots)
          : std::span<const int32_t>(host.new_cache_slots),
      host.q_seq_lens,
      host.kv_seq_lens,
      host.q_cu_seq_lens,
      int_span(host.block_tables),
      empty ? 0 : static_cast<uint32_t>(host.block_tables.size(/*dim=*/1))};
  Status status = validate(slot, model, batch, input.sampling_params);
  if (!status.ok()) {
    return status;
  }
  c10::DeviceGuard guard(device_.unwrap());
  slot.buffer->prepare(model, batch, input.sampling_params, prepare_stream_);
  if (!model.token_ids.empty()) {
    executor_.prepare_attention_metadata(kv_caches_,
                                         slot.buffer->model_params());
  }
  const auto& params = slot.buffer->sampling_params();
  const uint32_t samples =
      params.sample_idxes.defined() ? params.sample_idxes.numel() : 0;
  slot.sampling = params;
  slot.is_warmup = meta.is_graph_warmup;
  prepare_stream_.record_event(*slot.input_ready);
  CHECK_LT(accepted_tail_.id, std::numeric_limits<uint64_t>::max());
  slot.expected_producer = accepted_tail_;
  slot.step = {accepted_tail_.id + 1, samples};
  accepted_tail_ = slot.step;
  return Status();
}

void TaskExecutionPipeline::launch(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  Slot& slot = *slots_[slot_id];
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  CHECK(task_stream_.wait_event(slot.input_ready))
      << "Failed to wait for LLM task input.";
  CHECK_EQ(slot.step.id, published_tail_.id + 1);
  if (slot.buffer->has_previous_tokens()) {
    CHECK_EQ(slot.expected_producer.id, published_tail_.id);
    CHECK_EQ(slot.expected_producer.sample_rows, published_tail_.sample_rows);
    CHECK_GT(published_tail_.sample_rows, 0);
    slot.buffer->patch_previous_tokens(previous_tokens_);
  }
  if (slot.buffer->tokens().numel() != 0) {
    slot.model_output = executor_.forward(slot.buffer->tokens(),
                                          slot.buffer->positions(),
                                          kv_caches_,
                                          slot.buffer->model_params());
  }
  auto& params = slot.sampling;
  if (params.selected_token_idxes.defined() &&
      params.selected_token_idxes.numel() != 0) {
    slot.logits = model_.logits(slot.model_output.hidden_states,
                                params.selected_token_idxes);
    // Native top-k/top-p consumes top-p in the logits dtype. Retain the
    // conversion through Consume so asynchronous sampler reads stay valid.
    if (params.top_k.defined() && params.top_p.defined()) {
      params.top_p = params.top_p.to(slot.logits.scalar_type());
    }
    slot.sample_output = sampler_.forward(slot.logits, params);
    const auto& result = slot.buffer->device_result();
    result.tokens.copy_(slot.sample_output.next_tokens);
    if (result.logprobs.defined()) {
      result.logprobs.copy_(slot.sample_output.logprobs);
    }
    if (result.top_tokens.defined()) {
      result.top_tokens.copy_(slot.sample_output.top_tokens);
      result.top_logprobs.copy_(slot.sample_output.top_logprobs);
    }
  }
  // The independent buffer survives Consume and immediate Slot reuse. Copy
  // before output_ready so the existing D2H/Consume fence covers this reader.
  if (previous_tokens_.defined() && slot.step.sample_rows != 0) {
    previous_tokens_.narrow(/*dim=*/0, /*start=*/0, slot.step.sample_rows)
        .copy_(slot.buffer->device_result().tokens);
  }
  published_tail_ = slot.step;  // Zero rows invalidate all previous tokens.
  task_stream_.record_event(*slot.output_ready);
  const Status status =
      slot.buffer->copy_result_to_host(result_stream_, slot.output_ready);
  CHECK(status.ok()) << status.message();
}

ForwardOutput TaskExecutionPipeline::consume(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  Slot& slot = *slots_[slot_id];
  c10::DeviceGuard guard(device_.unwrap());
  TokenResultTensors tokens = slot.buffer->take_result();
  ForwardOutput result;
  result.do_sample = slot.buffer->copy_cpu_do_sample();
  result.is_graph_warmup = slot.is_warmup;
  result.logprobs = tokens.logprobs.defined();
  result.max_top_logprobs =
      tokens.top_tokens.defined() ? tokens.top_tokens.size(/*dim=*/1) : 0;
  result.sample_output.next_tokens = std::move(tokens.tokens);
  result.sample_output.logprobs = std::move(tokens.logprobs);
  result.sample_output.top_tokens = std::move(tokens.top_tokens);
  result.sample_output.top_logprobs = std::move(tokens.top_logprobs);
  release_outputs(slot);
  return result;
}

void TaskExecutionPipeline::discard(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  Slot& slot = *slots_[slot_id];
  c10::DeviceGuard guard(device_.unwrap());
  slot.buffer->discard_result();
  release_outputs(slot);
}

void TaskExecutionPipeline::release_outputs(Slot& slot) {
  slot.logits = torch::Tensor();
  slot.model_output = ModelOutput();
  slot.sample_output = SampleOutput();
  slot.sampling = SamplingParameters();
}

uint64_t TaskExecutionPipeline::pinned_bytes() const {
  uint64_t bytes = 0;
  for (const auto& slot : slots_) {
    bytes += slot->buffer->pinned_bytes();
  }
  return bytes;
}

uint64_t TaskExecutionPipeline::device_bytes() const {
  uint64_t bytes = previous_tokens_.defined() ? previous_tokens_.nbytes() : 0;
  for (const auto& slot : slots_) {
    bytes += slot->buffer->device_bytes();
  }
  return bytes;
}

}  // namespace xllm
