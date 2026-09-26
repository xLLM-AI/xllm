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

#include "core/runtime/unified_mtp_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>

#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/speculative/speculative_profile_registry.h"
#include "core/framework/speculative/verify_layout.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_runtime_helpers.h"
#include "core/runtime/py_executor_impl.h"
#include "core/runtime/task_execution_pipeline.h"
#include "core/runtime/unified_mtp_executor.h"
#include "core/util/tensor_helper.h"

namespace xllm {
namespace {
runtime::Options unified_leaf_options(const runtime::Options& options,
                                      bool draft) {
  runtime::Options leaf_options = options;
  leaf_options.enable_task_pipeline(false)
      .enable_schedule_overlap(false)
      .is_draft_engine(draft)
      .enable_graph_aux_hidden_states(true);
  if (draft) {
    leaf_options.num_decoding_tokens(1).num_speculative_tokens(0);
  }
  return leaf_options;
}
}  // namespace

using mtp_detail::check_mtp_decode_states;

bool supports_unified_mtp_request(const LlmForwardInput& input, bool adaptive) {
  const SamplingParameters& sampling = input.sampling_params;
  return !adaptive && input.json_object_states.empty() &&
         input.input_params.multi_block_tables.empty() &&
         sampling.all_greedy_sample && !sampling.use_beam_search &&
         !sampling.temperatures.defined() && !sampling.top_k.defined() &&
         !sampling.top_p.defined() && !sampling.frequency_penalties.defined() &&
         !sampling.presence_penalties.defined() &&
         !sampling.repetition_penalties.defined() &&
         !sampling.filter_mask.defined() && !sampling.filter_bitmask.defined();
}

UnifiedMtpWorkerImpl::UnifiedMtpWorkerImpl(const ParallelArgs& parallel_args,
                                           const torch::Device& device,
                                           const runtime::Options& options,
                                           WorkerType worker_type)
    : MTPWorkerImpl<LlmForwardInput>(
          parallel_args,
          device,
          options,
          unified_leaf_options(options, false),
          unified_leaf_options(options, true),
          worker_type,
          /*enable_adaptive_speculative_decode=*/true) {}

UnifiedMtpWorkerImpl::~UnifiedMtpWorkerImpl() {
  // Derived graph owners must survive all queued work, before base teardown.
  drain_pending_execution();
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  unified_executor_.reset();
}

std::optional<ForwardOutput> UnifiedMtpWorkerImpl::step(
    const LlmForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode() ||
      input.input_params.meta.num_sequences == 0 ||
      input.token_ids.numel() == 0 ||
      !should_run_speculative_decode(input.input_params)) {
    clear_unified_device_state();
  }
  // Empty DP shards must let the compatibility executor consume a matching
  // prelaunch, keeping collective counts aligned with active peers. Prefill
  // and a real Unified transition retire stale prelaunches in their entries.
  return SpeculativeWorkerImpl<LlmForwardInput>::step(input);
}

std::optional<ForwardOutput> UnifiedMtpWorkerImpl::step_decode(
    const LlmForwardInput& input) {
  if (supports_unified_python_mtp_graph(input)) {
    return step_unified(input);
  }
  // Capability routing is decided before any input/KV mutation. Never retry
  // a failed capture/replay with the compatibility executor.
  clear_unified_device_state();
  if (unified_executor_ != nullptr) {
    // A legacy request can enter Python ACL graph capture immediately after a
    // Unified replay.  Keep the two graph owners from sharing NPU capture
    // state: the executor destructor drains its stream before dropping the
    // registry, and queued outputs own snapshots. The next Unified request
    // recreates its variants.
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    unified_executor_.reset();
  }
  return MTPWorkerImpl<LlmForwardInput>::step_decode(input);
}

bool UnifiedMtpWorkerImpl::init_model(const std::string& model_weights_path,
                                      int32_t random_seed,
                                      MasterStatus master_status) {
  const bool result = MTPWorkerImpl<LlmForwardInput>::init_model(
      model_weights_path, random_seed, master_status);
  // Model/backend/topology capabilities are immutable during serving. Resolve
  // once after the pair loads; each step only examines request controls.
  unified_graph_capable_ = result && supports_unified_configuration();
  return result;
}

::xllm::Status UnifiedMtpWorkerImpl::create_task_pipeline(
    std::unique_ptr<TaskExecutionPipeline>& output) {
  if (impl_ == nullptr || draft_impl_ == nullptr ||
      options_.num_speculative_tokens() <= 0 ||
      options_.enable_adaptive_speculative_decode()) {
    return ::xllm::Status(
        StatusCode::INVALID_ARGUMENT,
        "Unified MTP task pipeline requires fixed greedy MTP models.");
  }
  LlmTaskCapacity capacity;
  ::xllm::Status status = impl_->task_capacity(options_, capacity);
  if (!status.ok()) {
    return status;
  }
  status = TaskExecutionPipeline::create(
      threadpool_, impl_->task_model(), *this, capacity, output);
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << "Unified MTP task execution pipeline: slots="
            << capacity.slot_count
            << ", speculative_tokens=" << options_.num_speculative_tokens();
  return status;
}

void UnifiedMtpWorkerImpl::prepare_task_pipeline_input(
    const LlmForwardInput& input,
    LlmForwardInput& prepared) {
  // The Unified task queue owns two independent prepared inputs.  Its launch
  // thread consumes them in order, so preparation may run while the previous
  // graph is on compute_stream_.  Keep the metadata event dependency, but do
  // not insert a whole prepare_stream -> compute_stream wait here.
  prepare_input_on_stream(input,
                          prepared,
                          *prepare_stream_,
                          /*record_ready_event=*/true,
                          /*restore_linear_state=*/true,
                          /*wait_for_compute_stream=*/false);
}

std::optional<ForwardOutput> UnifiedMtpWorkerImpl::execute_task_pipeline(
    const LlmForwardInput& prepared) {
  // The queue owns execution and results. JSON needs the preceding accepted
  // tokens as well, but must not use the driver's legacy last-step slot.
  if (!enable_schedule_overlap() || prepared.json_object_states.empty()) {
    task_json_output_ = ForwardOutput();
    task_json_sample_sequence_ids_.clear();
    return step(prepared);
  }

  LlmForwardInput input = prepared.clone();
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(input.runtime.metadata_ready_event));
  if (input.token_ids.numel() > 0 &&
      input.input_params.meta.batch_forward_type.has_decode()) {
    update_json_object_states_by_output(
        input, task_json_output_, task_json_sample_sequence_ids_);
    sanitize_json_object_error_inputs(input);
  }
  auto output = step(input);
  task_json_output_ = ForwardOutput();
  task_json_sample_sequence_ids_.clear();
  if (output.has_value()) {
    output->json_object_errors.insert(output->json_object_errors.end(),
                                      input.json_object_errors.begin(),
                                      input.json_object_errors.end());
    if (output->sample_output.next_tokens.defined()) {
      if (output->ready_event != nullptr) {
        CHECK(output->ready_event->synchronize());
      }
      // Output tensors may borrow a pinned-result lease. Own the small token
      // copy so consuming a task cannot invalidate the next grammar update.
      task_json_output_.sample_output.next_tokens =
          safe_to(
              output->sample_output.next_tokens,
              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
              /*non_blocking=*/false)
              .clone();
      task_json_sample_sequence_ids_ = input.sample_sequence_ids;
    }
  }
  return output;
}

bool UnifiedMtpWorkerImpl::supports_unified_python_mtp_graph(
    const LlmForwardInput& input) const {
  return unified_graph_capable_ &&
         draft_sampling_mode_ == DraftSamplingMode::GREEDY &&
         supports_unified_mtp_request(
             input,
             adaptive_enabled() && SpeculativeProfileRegistry::get_instance()
                                       .has_validate_time_predictor());
}

bool UnifiedMtpWorkerImpl::supports_unified_configuration() const {
#if defined(USE_NPU)
  if (!::xllm::ExecutionConfig::get_instance().enable_unified_mtp_graph()) {
    VLOG(1) << "MTP unified Python graph disabled by startup configuration";
    return false;
  }
  // For LLM decode, tp_size is the effective attention TP width published by
  // the communicator (world_size / dp_size for the v1 cp=1 path).  It is not
  // an independent launch knob.  EP is a separate MoE topology and may be
  // equal to or smaller than world_size.  CP is a prefill-only concern and is
  // outside the v1 decode graph scope.
  const int32_t world_size = std::max<int32_t>(parallel_args_.world_size(), 1);
  const int32_t dp_size = std::max<int32_t>(parallel_args_.dp_size(), 1);
  const int32_t tp_size = parallel_args_.tp_size();
  const int32_t ep_size = parallel_args_.ep_size();
  const bool attention_topology_valid =
      tp_size > 0 && tp_size * dp_size == world_size;
  const bool moe_topology_valid = ep_size > 0 && world_size % ep_size == 0;
  const int32_t moe_tp_size = moe_topology_valid ? world_size / ep_size : 0;
  const std::string& backend =
      ExecutionConfig::get_instance().python_graph_backend();
  const bool acl_backend = backend.empty() || backend == "off" ||
                           backend == "none" || backend == "0" ||
                           backend == "aclgraph";
  const bool unsupported_shape =
      !acl_backend || !::xllm::ExecutionConfig::get_instance().enable_graph() ||
      options_.num_speculative_tokens() <= 0 || impl_ == nullptr ||
      draft_impl_ == nullptr || !attention_topology_valid ||
      !moe_topology_valid || parallel_args_.dp_size() != 1 ||
      parallel_args_.cp_size() != 1 ||
      parallel_args_.kv_split_size_effective() != 1 ||
      context_.get_model_args().model_type() != "glm_moe_dsa" ||
      combined_draft_execution_path_ !=
          mtp_async::CombinedDraftExecutionPath::GLM_MOE_DSA_SPARSE_ATTENTION ||
      !ModelConfig::is_python_model_impl(context_.get_model_impl()) ||
      !ModelConfig::is_python_model_impl(
          draft_impl_->context_.get_model_impl());
  if (unsupported_shape) {
    VLOG(1) << "MTP unified Python graph rejected: k="
            << options_.num_speculative_tokens() << ", enable_graph="
            << ::xllm::ExecutionConfig::get_instance().enable_graph()
            << ", tp=" << parallel_args_.tp_size()
            << ", dp=" << parallel_args_.dp_size()
            << ", cp=" << parallel_args_.cp_size()
            << ", ep=" << parallel_args_.ep_size()
            << ", world=" << parallel_args_.world_size()
            << ", attention_topology_valid=" << attention_topology_valid
            << ", moe_topology_valid=" << moe_topology_valid
            << ", moe_tp=" << moe_tp_size
            << ", overlap=" << enable_schedule_overlap()
            << ", model_impl=" << context_.get_model_impl()
            << ", draft_model_impl="
            << (draft_impl_ == nullptr
                    ? std::string("<null>")
                    : draft_impl_->context_.get_model_impl());
    return false;
  }
  if (SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    VLOG(1) << "MTP unified Python graph rejected: ATB speculative kernel";
    return false;
  }
  const bool target_python_executor_ready =
      impl_->model_executor() != nullptr &&
      impl_->model_executor()->python_impl() != nullptr;
  const bool draft_python_executor_ready =
      draft_impl_->model_executor() != nullptr &&
      draft_impl_->model_executor()->python_impl() != nullptr;
  const bool supported =
      target_python_executor_ready && draft_python_executor_ready;
  if (supported) {
    VLOG(1) << "MTP unified Python graph capable: speculative_tokens="
            << options_.num_speculative_tokens();
  } else {
    VLOG(1) << "MTP unified Python graph rejected: Python executor is not "
               "ready, target_python_executor="
            << target_python_executor_ready
            << ", draft_python_executor=" << draft_python_executor_ready;
  }
  return supported;
#else
  return false;
#endif
}

std::optional<ForwardOutput> UnifiedMtpWorkerImpl::run_unified_python_mtp_graph(
    const LlmForwardInput& input,
    const LlmForwardInput& current_draft_input,
    int32_t num_speculative_tokens) {
#if defined(USE_NPU)
  PyExecutorImpl* target_executor = impl_->model_executor()->python_impl();
  PyExecutorImpl* draft_executor = draft_impl_->model_executor()->python_impl();
  CHECK(target_executor != nullptr && draft_executor != nullptr);
  UnifiedMtpExecutionResult execution =
      unified_executor().execute(*target_executor,
                                 *draft_executor,
                                 input,
                                 current_draft_input,
                                 num_speculative_tokens,
                                 context_.get_model_args().vocab_size(),
                                 logical_block_size(),
                                 uses_step_major_validate_layout());
  const int32_t batch_size = input.input_params.meta.num_sequences;
  detail::MtpPyGraphOutput& graph_output = execution.output;
  torch::Tensor base_positions = std::move(execution.base_positions);
  torch::Tensor kv_seq_lens = std::move(execution.base_kv_seq_lens);
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

  ForwardOutput target_output;
  target_output.do_sample = input.sampling_params.do_sample;
  target_output.logprobs = input.sampling_params.logprobs;
  target_output.max_top_logprobs = input.sampling_params.max_top_logprobs;
  target_output.sample_output.next_tokens = graph_output.committed_tokens;
  target_output.sample_output.embeddings = graph_output.target_embeddings;
  // Preserve the rejection sampler's [batch, K+1, ...] output layout.
  // Worker serialization indexes the request first, then the committed row.
  target_output.sample_output.probs = graph_output.target_probs;
  target_output.sample_output.logprobs = graph_output.committed_log_probs;
  target_output.sample_output.top_logprobs = graph_output.target_top_log_probs;
  target_output.sample_output.top_tokens = graph_output.target_top_tokens;

  // Persist the pre-acceptance graph base together with the committed-token
  // matrix. The fused next-draft preparation adds the accepted prefix to this
  // base; persisting graph_output.next_* here would advance the same prefix a
  // second time on the next invocation. This matches the eager MTP context
  // contract, where base_* are the target verify input base values.
  // These bases already own contiguous snapshots, including for batch=1
  // where contiguous() alone would retain the scheduler's input storage.
  // Reuse those snapshots instead of cloning them a second time.

  // The graph writes token IDs and counts into one arena. One owned Device
  // snapshot and one pinned D2H lease serve both consumers, with no pack op.
  CHECK(graph_output.token_state.defined());
  CHECK_EQ(graph_output.token_state.scalar_type(), torch::kUInt8);
  const int64_t token_bytes =
      graph_output.committed_tokens.numel() * sizeof(int64_t);
  CHECK_EQ(graph_output.token_state.numel(),
           token_bytes + batch_size * static_cast<int64_t>(sizeof(int32_t)));
  torch::Tensor state_host =
      acquire_accepted_tokens_host_buffer(graph_output.token_state);
  torch::Tensor accepted_tokens_host =
      state_host.narrow(0, 0, token_bytes)
          .view(torch::kLong)
          .view({batch_size, num_speculative_tokens + 1});
  torch::Tensor accepted_count_host =
      state_host.narrow(0, token_bytes, batch_size * sizeof(int32_t))
          .view(torch::kInt);
  state_host.copy_(graph_output.token_state, /*non_blocking=*/true);
  StreamEventPtr ready_event = compute_stream_->record_event();
  if (ready_event == nullptr) {
    const int32_t ret = compute_stream_->synchronize();
    CHECK_EQ(ret, 0) << "failed to synchronize unified MTP graph output";
  }
  if (enable_schedule_overlap()) {
    target_output.next_tokens_host = accepted_tokens_host;
  }
  stage_target_context_write(input,
                             target_output.sample_output,
                             base_positions,
                             kv_seq_lens,
                             ready_event,
                             std::move(accepted_tokens_host),
                             {},
                             std::move(accepted_count_host),
                             /*allow_immutable_compact_view=*/true);
  target_output.ready_event = ready_event;

  if (!enable_schedule_overlap()) {
    torch::Tensor accepted_tokens_cpu_result = snapshot_pending_target_tokens();
    target_output.ready_event.reset();
    target_output.sample_output.next_tokens =
        // The pinned destination is reused on the next decode. Publish an
        // owned CPU snapshot instead of retaining its view.
        std::move(accepted_tokens_cpu_result);
  }
  target_output.sample_output.embeddings = torch::Tensor();
  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  // The committed-token matrix is a verify result; mark it with the shared row
  // layout so the consumer records speculative metrics for this path.
  target_output.spec_verify_layouts =
      make_verify_layouts(target_output.sample_output.next_tokens,
                          /*per_seq_draft_counts=*/nullptr,
                          input.json_object_states);
  return target_output;
#else
  (void)input;
  (void)current_draft_input;
  (void)num_speculative_tokens;
  return std::nullopt;
#endif
}

std::optional<ForwardOutput> UnifiedMtpWorkerImpl::step_unified(
    const LlmForwardInput& input) {
  CHECK(embedding_cache_ != nullptr);
  retire_legacy_prelaunch();
  const bool bootstrap =
      input.input_params.embedding.mtp_bootstrap_embeddings.defined();
  if (bootstrap) {
    clear_unified_device_state();
    flush_pending_target_context();
    prepare_draft_input(input);
  }
  UnifiedMtpExecutor& executor = unified_executor();
  const UnifiedMtpContinuation* continuation = executor.continuation_for(input);
  if (continuation == nullptr && !bootstrap &&
      pending_target_context_matches(input)) {
    // Import an accepted legacy generation only on a route transition. Once
    // Unified executes, its executor owns the authoritative Device state;
    // output publication/Host-cache queue retirement does not choose inputs.
    const auto& pending = latest_pending_target_context();
    continuation = &executor.remember({pending.accepted_tokens,
                                       pending.accepted_embeddings,
                                       pending.base_positions,
                                       pending.base_kv_seq_lens,
                                       pending.embedding_ids,
                                       pending.request_ids});
  }
  if (continuation != nullptr) {
    LlmForwardInput prepared =
        executor.prepare_next(input,
                              continuation->accepted_tokens,
                              continuation->accepted_embeddings,
                              continuation->base_positions,
                              continuation->base_kv_seq_lens,
                              embedding_cache_->embedding_placeholder(),
                              logical_block_size());
    return run_unified_python_mtp_graph(
        input, prepared, options_.num_speculative_tokens());
  }

  // Bootstrap or changed request order: flush completed generations once,
  // then materialize a corrected input from per-request cache state.
  clear_unified_device_state();
  flush_pending_target_context();
  LlmForwardInput corrected = input.clone();
  auto states = embedding_cache_->read_decode_states(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids);
  if (!input.input_params.meta.is_graph_warmup) {
    check_mtp_decode_states(states,
                            input.input_params.embedding.request_ids,
                            input.token_ids_host,
                            enable_schedule_overlap());
  }
  update_decode_step_input(corrected, states);
  LlmForwardInput prepared;
  prepare_draft_extend_inputs(
      corrected,
      states,
      prepared,
      /*force_two_rows=*/true,
      /*wait_for_compute_stream=*/!options_.enable_task_pipeline());
  return run_unified_python_mtp_graph(
      corrected, prepared, options_.num_speculative_tokens());
}

UnifiedMtpExecutor& UnifiedMtpWorkerImpl::unified_executor() {
  if (unified_executor_ == nullptr) {
    unified_executor_ = std::make_unique<UnifiedMtpExecutor>(*compute_stream_);
  }
  return *unified_executor_;
}

void UnifiedMtpWorkerImpl::clear_unified_device_state() {
  if (unified_executor_ != nullptr) {
    unified_executor_->clear_continuation();
  }
}

}  // namespace xllm
