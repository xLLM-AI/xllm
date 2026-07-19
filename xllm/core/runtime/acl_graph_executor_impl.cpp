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

#include "acl_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>

#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/metrics.h"
#include "core/kernels/ops_api.h"
#include "core/platform/device.h"
#include "core/platform/npu/acl_graph_task_update_context.h"
#include "core/util/utils.h"
#include "platform/npu/device_capture_lock.h"

namespace xllm::npu {

namespace {
constexpr uint64_t kSpecVerifyGraphKeyMask = 1ull << 63;
constexpr uint64_t kSpecVerifyQMaxSeqLenShift = 32;
constexpr uint64_t kSpecVerifyBucketMask = (1ull << 16) - 1;
constexpr uint64_t kSpecVerifyFieldMask = (1ull << 16) - 1;
constexpr uint64_t kSpecVerifyExpandedBlockMask = (1ull << 15) - 1;
constexpr uint64_t kStaticGraphTaskHashSeed = 0x6a09e667f3bcc909ull;

bool uses_static_mtp_graph_task_variant(const ModelInputParams& params,
                                        uint32_t bucket_num_tokens) {
  return params.is_spec_verify &&
         params.meta.batch_forward_type.is_chunked_prefill() &&
         params.graph.use_expanded_decode_for_spec_verify_attention &&
         params.graph.spec_verify_source_addresses_stable &&
         bucket_num_tokens >= 4 && bucket_num_tokens <= 6 &&
         params.meta.num_sequences == 1 &&
         params.embedding.linear_state_ids.size() == 1 &&
         params.num_accepted_tokens_host.size() == 1;
}

uint64_t mix_graph_key(uint64_t hash, uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  value ^= value >> 31;
  return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}

uint64_t static_mtp_graph_task_key(uint64_t base_key,
                                   const ModelInputParams& params) {
  uint64_t hash = mix_graph_key(kStaticGraphTaskHashSeed, base_key);
  hash = mix_graph_key(
      hash, static_cast<uint64_t>(params.embedding.linear_state_ids.front()));
  hash = mix_graph_key(
      hash, static_cast<uint64_t>(params.num_accepted_tokens_host.front()));
  for (const int64_t value : params.parallel.query_start_loc) {
    hash = mix_graph_key(hash, static_cast<uint64_t>(value));
  }
  // Retain the spec-verify namespace bit. A signature comparison on replay
  // guards correctness even in the astronomically unlikely event of a hash
  // collision.
  return hash | kSpecVerifyGraphKeyMask;
}

uint64_t static_mtp_graph_task_key(uint64_t base_key,
                                   int64_t linear_state_id,
                                   int64_t num_accepted_tokens,
                                   int64_t spec_width) {
  uint64_t hash = mix_graph_key(kStaticGraphTaskHashSeed, base_key);
  hash = mix_graph_key(hash, static_cast<uint64_t>(linear_state_id));
  hash = mix_graph_key(hash, static_cast<uint64_t>(num_accepted_tokens));
  hash = mix_graph_key(hash, 0);
  hash = mix_graph_key(hash, static_cast<uint64_t>(spec_width));
  return hash | kSpecVerifyGraphKeyMask;
}

std::pair<torch::Tensor, torch::Tensor> find_attention_plan_kv_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return {std::move(k_cache), std::move(v_cache)};
    }
  }
  return {torch::Tensor(), torch::Tensor()};
}

std::vector<torch::Tensor> spec_verify_input_sources(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    bool fused_metadata_update) {
  const torch::Tensor& graph_tokens =
      params.graph.input_tokens_override.defined()
          ? params.graph.input_tokens_override
          : tokens;
  if (fused_metadata_update) {
    return {positions,
            params.attention.device.kv_seq_lens,
            params.attention.device.new_cache_slots,
            params.attention.device.block_tables,
            params.embedding.linear_state_indices,
            params.num_accepted_tokens};
  }
  return {graph_tokens,
          positions,
          params.attention.device.q_seq_lens,
          params.attention.device.kv_seq_lens,
          params.attention.device.new_cache_slots,
          params.attention.device.block_tables,
          params.embedding.linear_state_indices,
          params.num_accepted_tokens,
          params.attention.device.q_cu_seq_lens,
          params.graph.expanded_kv_seq_lens,
          params.graph.expanded_block_tables};
}

bool same_tensor_sources(const std::vector<torch::Tensor>& captured,
                         const std::vector<torch::Tensor>& current) {
  if (captured.size() != current.size()) {
    LOG(ERROR) << "speculative verify input source count changed: captured="
               << captured.size() << ", current=" << current.size();
    return false;
  }
  for (size_t i = 0; i < captured.size(); ++i) {
    if (!captured[i].defined() || !current[i].defined() ||
        captured[i].data_ptr() != current[i].data_ptr() ||
        captured[i].sizes() != current[i].sizes() ||
        captured[i].strides() != current[i].strides()) {
      LOG(ERROR) << "speculative verify input update source changed: index="
                 << i << ", captured_ptr=" << captured[i].data_ptr()
                 << ", current_ptr=" << current[i].data_ptr()
                 << ", captured_sizes=" << captured[i].sizes()
                 << ", current_sizes=" << current[i].sizes()
                 << ", captured_strides=" << captured[i].strides()
                 << ", current_strides=" << current[i].strides();
      return false;
    }
  }
  return true;
}

}  // namespace

bool AclGraph::capture(CausalLM* model,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens) {
  // Save bucket num_tokens for this graph instance
  num_tokens_ = bucket_num_tokens;

  // Get actual num_tokens from tokens tensor
  // const uint32_t actual_num_tokens = tokens.size(0);

  auto& tensor_options = model->options();

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // For hybrid models (e.g., qwen3_next with mixed GDN/full_attention layers),
  // we need to find the first Full Attention layer to get the correct kv_cache.
  // GDN layers have empty key_cache_/value_cache_ while Full Attention layers
  // have valid kv caches. Using layer 0's cache directly would be incorrect
  // if layer 0 is a GDN layer.
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));
  CHECK_GE(num_tokens_, actual_num_tokens)
      << "num_tokens_ >= actual_num_tokens";
  const bool fused_spec_verify_token_update =
      persistent_param_.supports_fused_spec_verify_token_update(params);
  auto graph_params = persistent_param_.update(tokens,
                                               k_cache,
                                               v_cache,
                                               positions,
                                               params,
                                               num_tokens_,
                                               /*return_capture_params=*/true,
                                               /*skip_token_update=*/
                                               fused_spec_verify_token_update);
  if (fused_spec_verify_token_update) {
    persistent_param_.run_fused_spec_verify_token_update(params);
  }

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";
  prepare_model_graph_metadata(
      model,
      persistent_param_.persistent_positions(num_tokens_),
      graph_params.value());

  if (model->is_hybrid_linear_attention()) {
    graph_task_context_ = std::make_shared<AclGraphTaskUpdateContext>();
    graph_task_context_->begin_capture();
    graph_params->graph.acl_graph_task_update_context = graph_task_context_;
  }
  const bool capture_static_graph_tasks =
      uses_static_mtp_graph_task_variant(graph_params.value(), num_tokens_);
  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Acquire device-level lock to prevent prepare_work_before_execute from
  // executing simultaneously, which would trigger synchronous operations
  // that conflict with capture mode
  auto device_idx = tensor_options.device().index();
  Device::empty_cache(device_idx);

  bool need_restore_stream = false;
  graph_stream_ = stream;

  // capture lock scope
  {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_idx);
    std::lock_guard<std::mutex> lock_guard(capture_lock);

    if (c10_npu::getCurrentNPUStream(device_idx) ==
        c10_npu::getDefaultNPUStream(device_idx)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      aclrtSynchronizeStream(capture_stream_.value().stream());
      graph_stream_ = capture_stream_.value().stream();
      need_restore_stream = true;
    }
    LOG(INFO) << "capture begin, bucket_num_tokens: " << bucket_num_tokens
              << ", actual_num_tokens: " << actual_num_tokens;

    // no mempool id, will create a new one; capture mode is thread local, allow
    // other threads to execute synchronous operations
    graph_.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    has_internal_spec_verify_input_update_ =
        model->is_hybrid_linear_attention() &&
        params.graph.spec_verify_source_addresses_stable &&
        params.is_spec_verify &&
        params.meta.batch_forward_type.is_chunked_prefill() &&
        actual_num_tokens == bucket_num_tokens &&
        params.meta.q_max_seq_len == actual_num_tokens;
    if (has_internal_spec_verify_input_update_) {
      internal_spec_verify_input_sources_ =
          persistent_param_.capture_spec_verify_input_update(
              tokens, positions, params, bucket_num_tokens);
    }
    // Execute forward pass - NPUGraph mempool manages temporary tensors
    auto forward_result =
        model->forward({persistent_param_.persistent_tokens(num_tokens_)},
                       {persistent_param_.persistent_positions(num_tokens_)},
                       kv_cache,
                       {graph_params.value()});

    // Store result in persistent buffer owned by NPUGraph mempool
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    graph_.capture_end();
    if (graph_task_context_ != nullptr) {
      graph_task_context_->end_capture();
    }
    // Lock is automatically released here when lock goes out of scope
    if (need_restore_stream) {
      c10_npu::setCurrentNPUStream(
          c10_npu::getDefaultNPUStream(tensor_options.device().index()));
    }
  }
  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(graph_stream_);
  aclrtSynchronizeStream(stream);
  graph_.replay();
  update_graph_tasks(graph_params.value());
  if (capture_static_graph_tasks) {
    capture_static_graph_task_signature(graph_params.value());
  }
  make_current_stream_wait_for_graph(stream);
  return true;
}

bool AclGraph::update_graph_tasks(const ModelInputParams& params) {
  if (graph_task_context_ == nullptr ||
      graph_task_context_->causal_conv1d_tasks.empty()) {
    return false;
  }

  const std::vector<int64_t> empty_host_args;
  CHECK(!params.parallel.query_start_loc.empty())
      << "causal_conv1d graph update requires padded query_start_loc";
  CHECK(!params.embedding.linear_state_ids.empty())
      << "causal_conv1d graph update requires padded cache indices";

  std::vector<int64_t> linear_state_indices_host(
      params.embedding.linear_state_ids.begin(),
      params.embedding.linear_state_ids.end());

  c10_npu::NPUStream update_stream = update_stream_.value();
  c10_npu::NPUStreamGuard stream_guard(update_stream);

  for (auto& task : graph_task_context_->causal_conv1d_tasks) {
    CHECK_EQ(params.parallel.query_start_loc.back(), task.x.size(0))
        << "causal_conv1d graph update host args must be padded to the "
           "capture x.shape[0]";
    CHECK_EQ(linear_state_indices_host.size() + 1,
             params.parallel.query_start_loc.size())
        << "cache_indices must be sequence-scoped";

    const std::vector<int64_t>& num_accepted_tokens =
        task.branch == CausalConv1dGraphBranch::kSpecVerify
            ? params.num_accepted_tokens_host
            : empty_host_args;
    if (task.branch == CausalConv1dGraphBranch::kSpecVerify) {
      CHECK_EQ(num_accepted_tokens.size(), linear_state_indices_host.size())
          << "spec causal_conv1d graph update requires accepted-token counts";
    }

    c10_npu::graph_task_update_begin(update_stream, task.handle);
    xllm::kernel::causal_conv1d_out(
        task.output,
        task.x,
        task.weight,
        task.conv_state,
        task.bias,
        torch::IntArrayRef(params.parallel.query_start_loc),
        torch::IntArrayRef(linear_state_indices_host),
        torch::IntArrayRef(empty_host_args),
        torch::IntArrayRef(num_accepted_tokens),
        task.activation_mode,
        task.pad_slot_id,
        task.run_mode);
    c10_npu::graph_task_update_end(update_stream);
    if (task.event != nullptr) {
      task.event->record(update_stream);
    }
  }
  return true;
}

void AclGraph::signal_static_graph_tasks(
    const c10_npu::NPUStream& signal_stream) {
  CHECK(graph_task_context_ != nullptr);
  for (auto& task : graph_task_context_->causal_conv1d_tasks) {
    CHECK(task.event != nullptr)
        << "static graph-task replay requires a captured ready event";
    task.event->record(signal_stream);
  }
}

bool AclGraph::static_graph_task_signature_matches(
    const ModelInputParams& params) const {
  if (!has_static_graph_task_signature_) {
    return false;
  }
  return params.parallel.query_start_loc == static_query_start_loc_ &&
         params.embedding.linear_state_ids == static_linear_state_ids_ &&
         params.num_accepted_tokens_host == static_num_accepted_tokens_;
}

void AclGraph::capture_static_graph_task_signature(
    const ModelInputParams& params) {
  static_query_start_loc_ = params.parallel.query_start_loc;
  static_linear_state_ids_ = params.embedding.linear_state_ids;
  static_num_accepted_tokens_ = params.num_accepted_tokens_host;
  has_static_graph_task_signature_ = true;
  LOG(INFO) << "Captured static MTP graph-task signature: linear_state_id="
            << static_linear_state_ids_.front()
            << ", accepted_tokens=" << static_num_accepted_tokens_.front();
}

AclGraph::~AclGraph() {
  if (graph_stream_ != nullptr) {
    aclrtSynchronizeStream(graph_stream_);
  } else if (capture_stream_.has_value()) {
    aclrtSynchronizeStream(capture_stream_.value().stream());
  }
  if (replay_done_event_ != nullptr) {
    aclrtDestroyEvent(replay_done_event_);
    replay_done_event_ = nullptr;
  }
  if (replay_input_ready_event_ != nullptr) {
    aclrtDestroyEvent(replay_input_ready_event_);
    replay_input_ready_event_ = nullptr;
  }
}

void AclGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  // Get a secondary stream from high-priority pool for graph capture.
  // This is required because NPUGraph::capture_begin() enforces that capture
  // must be performed on a non-default stream (see
  // torch_npu/csrc/core/npu/NPUGraph.cpp:159).
  capture_stream_ = c10_npu::getStreamFromPool(true, device_index);
  update_stream_ = c10_npu::getStreamFromPool(true, device_index);
  device_index_ = device_index;
  CHECK_EQ(aclrtCreateEventWithFlag(&replay_input_ready_event_, ACL_EVENT_SYNC),
           ACL_SUCCESS)
      << "Failed to create ACL graph replay input-ready event";
  CHECK_EQ(aclrtCreateEventWithFlag(&replay_done_event_, ACL_EVENT_SYNC),
           ACL_SUCCESS)
      << "Failed to create ACL graph replay completion event";
  LOG(INFO) << "Initialized capture_stream"
            << ", id: " << capture_stream_.value().id()
            << ", device_index: " << device_index;
}

void AclGraph::make_graph_wait_for_current_stream(aclrtStream current_stream) {
  CHECK_NE(graph_stream_, nullptr) << "graph_stream is not initialized";
  CHECK_NE(replay_input_ready_event_, nullptr)
      << "replay_input_ready_event is not initialized";
  if (current_stream == graph_stream_) {
    return;
  }
  CHECK_EQ(aclrtRecordEvent(replay_input_ready_event_, current_stream),
           ACL_SUCCESS)
      << "aclrtRecordEvent(replay_input_ready_event) failed";
  CHECK_EQ(aclrtStreamWaitEvent(graph_stream_, replay_input_ready_event_),
           ACL_SUCCESS)
      << "aclrtStreamWaitEvent(graph_stream, replay_input_ready_event) failed";
}

void AclGraph::make_current_stream_wait_for_graph(aclrtStream current_stream) {
  CHECK_NE(graph_stream_, nullptr) << "graph_stream is not initialized";
  CHECK_NE(replay_done_event_, nullptr)
      << "replay_done_event is not initialized";
  CHECK_EQ(aclrtRecordEvent(replay_done_event_, graph_stream_), ACL_SUCCESS)
      << "aclrtRecordEvent(replay_done_event) failed";
  if (current_stream != graph_stream_) {
    CHECK_EQ(aclrtStreamWaitEvent(current_stream, replay_done_event_),
             ACL_SUCCESS)
        << "aclrtStreamWaitEvent(current_stream, replay_done_event) failed";
  }
}

void AclGraph::prepare_model_graph_metadata(CausalLM* model,
                                            const torch::Tensor& positions,
                                            ModelInputParams& params) {
  CHECK(model != nullptr) << "ACL graph model must not be null";
  if (!model->requires_graph_forward_metadata()) {
    return;
  }
  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
    CHECK(model_graph_metadata_state_)
        << "ACL graph metadata state must be initialized during capture";
  }
  model->prepare_graph_forward_metadata(
      model_graph_metadata_state_.get(), positions, params);
  CHECK(params.attn_metadata)
      << "model graph metadata preparation did not populate attn_metadata";
}

ModelOutput AclGraph::replay(CausalLM* model,
                             const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params) {
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  // Note: tiling_data is updated in update() if needed - for hybrid models
  // (e.g., qwen3_next with mixed GDN/attention layers), tiling should only
  // be updated when Full Attention layers are involved, which is determined
  // by k_cache being valid and non-empty
  const bool needs_graph_metadata = model->requires_graph_forward_metadata() ||
                                    model->is_hybrid_linear_attention();
  const bool replay_inputs_prepared =
      replay_inputs_prepared_.exchange(false, std::memory_order_acq_rel);
  const bool can_use_prepared_inputs =
      replay_inputs_prepared && params.graph.input_tokens_override.defined() &&
      !needs_graph_metadata;
  std::optional<ModelInputParams> graph_params;
  if (has_internal_spec_verify_input_update_) {
    const auto current_sources = spec_verify_input_sources(
        tokens,
        positions,
        params,
        persistent_param_.supports_fused_spec_verify_metadata_update(params));
    CHECK(same_tensor_sources(internal_spec_verify_input_sources_,
                              current_sources))
        << "speculative verify graph input update source address changed "
           "across replay";
    if (persistent_param_.supports_fused_spec_verify_token_update(params)) {
      persistent_param_.run_fused_spec_verify_token_update(params);
    }
    // The leading captured graph nodes copy all dynamic device fields into
    // persistent graph buffers. Host-only task parameters remain current and
    // are consumed by update_graph_tasks() below.
    graph_params = params;
  } else if (can_use_prepared_inputs) {
    persistent_param_.update_tokens(
        tokens, params, actual_num_tokens, num_tokens_);
  } else {
    auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
    graph_params = persistent_param_.update(tokens,
                                            k_cache,
                                            v_cache,
                                            positions,
                                            params,
                                            num_tokens_,
                                            needs_graph_metadata);
    if (needs_graph_metadata) {
      CHECK(graph_params.has_value())
          << "ACL graph replay requires persistent params for graph metadata";
      prepare_model_graph_metadata(
          model,
          persistent_param_.persistent_positions(num_tokens_),
          graph_params.value());
    }
  }

  // Replay captured graph - NPUGraph mempool reuses temporary tensors
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  if (has_internal_spec_verify_input_update_) {
    // The graph's leading input update reads sources finalized by the current
    // compute stream (including the last draft token). Express that dependency
    // on device; a host synchronize here would recreate the bubble we remove.
    make_graph_wait_for_current_stream(stream);
  }
  const bool use_static_graph_tasks =
      graph_params.has_value() &&
      static_graph_task_signature_matches(graph_params.value());
  const bool static_graph_tasks_prepared =
      params.graph.spec_verify_static_graph_tasks_prepared;
  CHECK(!static_graph_tasks_prepared || use_static_graph_tasks)
      << "prepared static graph tasks do not match the replay signature";
  if (use_static_graph_tasks && !static_graph_tasks_prepared) {
    // Cold/fallback path: the final-draft pre-submit could not find this graph
    // variant. Ring its ready event immediately before replay; steady MTP5
    // cycles use the compute-stream pre-submit path instead.
    CHECK(update_stream_.has_value());
    signal_static_graph_tasks(update_stream_.value());
  }
  graph_.replay();
  if (model->is_hybrid_linear_attention()) {
    CHECK(graph_params.has_value())
        << "update() should return ModelInputParams for graph task update";
    if (use_static_graph_tasks) {
      // This graph variant's task-ready event was recorded before replay.
    } else {
      update_graph_tasks(graph_params.value());
    }
  }
  make_current_stream_wait_for_graph(stream);

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in AclGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(get_hidden_states(actual_num_tokens));
}

void AclGraph::prepare_replay_inputs(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_cache,
                                     const ModelInputParams& params) {
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  persistent_param_.update(tokens,
                           k_cache,
                           v_cache,
                           positions,
                           params,
                           num_tokens_,
                           /*return_capture_params=*/false,
                           /*skip_token_update=*/true);
  replay_inputs_prepared_.store(true, std::memory_order_release);
}

bool AclGraph::prepare_static_graph_tasks(
    const ModelInputParams& params,
    const c10_npu::NPUStream& signal_stream) {
  if (!static_graph_task_signature_matches(params)) {
    return false;
  }
  // The final draft graph has already been submitted to signal_stream. Queue
  // each captured task-ready event behind it, so the target graph observes a
  // fresh generation without a cross-stream credit/wait or host synchronize.
  signal_static_graph_tasks(signal_stream);
  return true;
}

bool AclGraph::prepare_static_mtp_graph_tasks(
    int64_t linear_state_id,
    int64_t num_accepted_tokens,
    int64_t spec_width,
    const c10_npu::NPUStream& signal_stream) {
  if (!has_static_graph_task_signature_ ||
      static_query_start_loc_ != std::vector<int64_t>({0, spec_width}) ||
      static_linear_state_ids_ !=
          std::vector<int32_t>({static_cast<int32_t>(linear_state_id)}) ||
      static_num_accepted_tokens_ !=
          std::vector<int64_t>({num_accepted_tokens})) {
    return false;
  }
  signal_static_graph_tasks(signal_stream);
  return true;
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  const bool need_update_attn_mask = model->is_hybrid_linear_attention();
  const bool is_hybrid_linear_attn = model->is_hybrid_linear_attention();
  graph_slot_count_ =
      ::xllm::ExecutionConfig::get_instance().enable_graph_double_buffer() ? 2
                                                                           : 1;
  for (int32_t slot_idx = 0; slot_idx < graph_slot_count_; ++slot_idx) {
    graph_slots_[slot_idx].persistent_param =
        std::make_unique<GraphPersistentParam>(args_,
                                               device_,
                                               options_,
                                               need_update_attn_mask,
                                               is_hybrid_linear_attn);
  }
}

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
ModelOutput AclGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  const bool in_decoding_phase =
      params_single.meta.batch_forward_type.is_decode();
  const bool in_spec_verify_phase =
      params_single.is_spec_verify &&
      params_single.meta.batch_forward_type.is_chunked_prefill();
  VLOG(50) << "in_decoding_phase: " << in_decoding_phase
           << " in_spec_verify_phase: " << in_spec_verify_phase
           << " q_max_seq_len: " << params_single.meta.q_max_seq_len
           << " n_layers: " << args_.n_layers();
  if ((!in_decoding_phase && !in_spec_verify_phase) || args_.n_layers() == 1) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  if (in_spec_verify_phase && !model_->is_hybrid_linear_attention()) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for spec verify because the "
           "chunked-prefill validate graph path is currently only adapted for "
           "hybrid linear attention models.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  if (in_decoding_phase &&
      params_single.parallel.dp_global_token_nums.size() > 1) {
    if (params_single.parallel.dp_is_decode.size() !=
        params_single.parallel.dp_global_token_nums.size()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because dp_is_decode size ("
          << params_single.parallel.dp_is_decode.size()
          << ") does not match dp_global_token_nums size ("
          << params_single.parallel.dp_global_token_nums.size()
          << "); ACL graph decode requires valid DP forward metadata. "
          << "dp_global_token_nums="
          << params_single.parallel.dp_global_token_nums
          << ", dp_is_decode=" << params_single.parallel.dp_is_decode;
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    if (std::find(params_single.parallel.dp_is_decode.begin(),
                  params_single.parallel.dp_is_decode.end(),
                  0) != params_single.parallel.dp_is_decode.end()) {
      LOG_FIRST_N(WARNING, 1)
          << "Falling back to eager mode because not all DP ranks are in "
             "decode phase; ACL graph decode requires all DP ranks to be "
             "decode to avoid using prefill or chunked-prefill token counts "
             "as graph bucket size. dp_global_token_nums="
          << params_single.parallel.dp_global_token_nums
          << ", dp_is_decode=" << params_single.parallel.dp_is_decode;
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
  }

  // Only use acl graph in decode phase for performance optimization
  // For DP, decode graph bucket should be based on global max tokens across dp
  // groups; local shard can be empty on some ranks.
  uint32_t graph_num_tokens = tokens_tensor.size(/*dim=*/0);
  if (params_single.parallel.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params_single.parallel.dp_global_token_nums);
  }
  // Keep actual n_tokens for replay output slicing.
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t local_batch_size = n_tokens / options_.num_decoding_tokens();
  const uint32_t global_batch_size =
      graph_num_tokens / options_.num_decoding_tokens();

  // Large decode batches create too many/too large ACL graphs and may OOM.
  // Fall back to eager mode when batch size exceeds the safety threshold.
  // Use global_batch_size so all DP ranks make the same decision and stay in
  // sync on HCCL collectives.
  const uint32_t decode_batch_size_limit = static_cast<uint32_t>(
      std::max<int32_t>(1,
                        ::xllm::ExecutionConfig::get_instance()
                            .acl_graph_decode_batch_size_limit()));
  if (global_batch_size > decode_batch_size_limit) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because decode batch_size (global="
        << global_batch_size << ", local=" << local_batch_size << ") > "
        << decode_batch_size_limit
        << "; ACL graph is disabled for this request size to avoid OOM. "
        << "This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = args_.max_position_embeddings();
  const bool seq_len_supported =
      params_single.meta.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because kv_max_seq_len ("
        << params_single.meta.kv_max_seq_len << ") > max_seq_len ("
        << max_seq_len << "). This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const uint64_t graph_key = get_graph_key(bucket_num_tokens, params_single);

  // Check if captured graph exists for this bucket num_tokens
  int32_t slot_idx = 0;
  AclGraph* replay_graph = nullptr;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    slot_idx = next_replay_slot_;
    next_replay_slot_ = (next_replay_slot_ + 1) % graph_slot_count_;
    last_started_replay_slot_ = slot_idx;
    auto& slot = graph_slots_[slot_idx];
    slot.is_prepared = false;
    auto it = slot.graphs.find(graph_key);
    if (it != slot.graphs.end()) {
      replay_graph = it->second.get();
    }
  }
  auto& active_slot = graph_slots_[slot_idx];
  auto& active_persistent_param = *active_slot.persistent_param;

  if (replay_graph != nullptr) {
    // Replay the existing graph
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in replay mode";
    ModelOutput result = replay_graph->replay(
        model_, tokens_tensor, positions_tensor, kv_caches, params_single);
    // Handle aux_hidden_states based on options
    if (options_.enable_graph_aux_hidden_states()) {
      torch::Tensor aux_hidden_states =
          active_persistent_param.aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(
            result.hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return result;
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  auto graph =
      std::make_unique<AclGraph>(active_persistent_param, device_.index());
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in capture mode";
  bool capture_success = false;
  try {
    capture_success = graph->capture(model_,
                                     options_,
                                     tokens_tensor,
                                     positions_tensor,
                                     params_single,
                                     kv_caches,
                                     bucket_num_tokens);
  } catch (const std::exception& e) {
    LOG(ERROR) << "ACL graph capture threw exception for bucket num_tokens="
               << bucket_num_tokens << ": " << e.what()
               << ". Falling back to eager mode.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    // Save the graph for future reuse
    active_slot.graphs[graph_key] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    torch::Tensor hidden_states =
        active_slot.graphs[graph_key]->get_hidden_states(n_tokens);
    if (options_.enable_graph_aux_hidden_states()) {
      torch::Tensor aux_hidden_states =
          active_persistent_param.aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return ModelOutput(hidden_states);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture ACL graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

void AclGraphExecutorImpl::prepare_graph_input(const torch::Tensor& tokens,
                                               const torch::Tensor& positions,
                                               std::vector<KVCache>& kv_caches,
                                               const ModelInputParams& params) {
  const bool in_decoding_phase = params.meta.batch_forward_type.is_decode();
  const bool in_spec_verify_phase =
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  if ((!in_decoding_phase && !in_spec_verify_phase) || args_.n_layers() == 1) {
    return;
  }
  if (model_->requires_graph_forward_metadata()) {
    return;
  }
  if (in_spec_verify_phase && !model_->is_hybrid_linear_attention()) {
    return;
  }
  if (in_decoding_phase && params.parallel.dp_global_token_nums.size() > 1) {
    if (params.parallel.dp_is_decode.size() !=
        params.parallel.dp_global_token_nums.size()) {
      return;
    }
    if (std::find(params.parallel.dp_is_decode.begin(),
                  params.parallel.dp_is_decode.end(),
                  0) != params.parallel.dp_is_decode.end()) {
      return;
    }
  }
  if (params.meta.kv_max_seq_len > args_.max_position_embeddings()) {
    return;
  }
  if (graph_slot_count_ <= 1) {
    return;
  }

  uint32_t graph_num_tokens = tokens.size(/*dim=*/0);
  if (params.parallel.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params.parallel.dp_global_token_nums);
  }
  if (graph_num_tokens == 0) {
    return;
  }
  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);
  const uint64_t graph_key = get_graph_key(bucket_num_tokens, params);

  AclGraph* graph = nullptr;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    if (last_started_replay_slot_ < 0) {
      return;
    }
    const int32_t prepare_slot =
        (last_started_replay_slot_ + 1) % graph_slot_count_;
    auto& slot = graph_slots_[prepare_slot];
    if (slot.is_prepared) {
      return;
    }
    auto it = slot.graphs.find(graph_key);
    if (it == slot.graphs.end()) {
      return;
    }
    graph = it->second.get();
    slot.is_prepared = true;
  }
  graph->prepare_replay_inputs(tokens, positions, kv_caches, params);
}

bool AclGraphExecutorImpl::prepare_static_graph_tasks(
    const torch::Tensor& tokens,
    const ModelInputParams& params,
    const Stream& signal_stream) {
  if (!model_->is_hybrid_linear_attention() || graph_slot_count_ != 1 ||
      !params.is_spec_verify ||
      !params.meta.batch_forward_type.is_chunked_prefill() ||
      !params.graph.use_expanded_decode_for_spec_verify_attention) {
    return false;
  }
  uint32_t graph_num_tokens = tokens.size(0);
  if (params.parallel.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params.parallel.dp_global_token_nums);
  }
  if (graph_num_tokens == 0) {
    return false;
  }
  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);
  const uint64_t graph_key = get_graph_key(bucket_num_tokens, params);
  AclGraph* graph = nullptr;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto& graphs = graph_slots_[0].graphs;
    auto it = graphs.find(graph_key);
    if (it == graphs.end()) {
      return false;
    }
    graph = it->second.get();
  }
  return graph->prepare_static_graph_tasks(params, *signal_stream.get_stream());
}

bool AclGraphExecutorImpl::prepare_static_mtp_graph_tasks(
    int64_t linear_state_id,
    int64_t num_accepted_tokens,
    int64_t spec_width,
    int64_t block_table_width,
    const Stream& signal_stream) {
  if (!model_->is_hybrid_linear_attention() || graph_slot_count_ != 1 ||
      spec_width < 4 || spec_width > 6 || block_table_width < 1 ||
      block_table_width > kSpecVerifyExpandedBlockMask) {
    return false;
  }
  const uint64_t bucket_num_tokens = static_cast<uint64_t>(spec_width);
  const uint64_t q_max_seq_len = static_cast<uint64_t>(spec_width);
  const uint64_t width = static_cast<uint64_t>(block_table_width);
  const uint64_t base_key = kSpecVerifyGraphKeyMask | (width << 48) |
                            (width << 32) | (q_max_seq_len << 16) |
                            bucket_num_tokens;
  const uint64_t graph_key = static_mtp_graph_task_key(
      base_key, linear_state_id, num_accepted_tokens, spec_width);
  AclGraph* graph = nullptr;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto& graphs = graph_slots_[0].graphs;
    auto it = graphs.find(graph_key);
    if (it == graphs.end()) {
      return false;
    }
    graph = it->second.get();
  }
  return graph->prepare_static_mtp_graph_tasks(linear_state_id,
                                               num_accepted_tokens,
                                               spec_width,
                                               *signal_stream.get_stream());
}

void AclGraph::print_graph_tensors() const {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_tokens_: " << persistent_param_.persistent_tokens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_positions_: "
      << persistent_param_.persistent_positions();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_new_cache_slots_: "
      << persistent_param_.persistent_new_cache_slots();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph q_seq_lens_: " << persistent_param_.q_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph kv_seq_lens_: " << persistent_param_.kv_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_block_tables_: "
      << persistent_param_.persistent_block_tables();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph hidden_states_: " << persistent_param_.hidden_states();
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t AclGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 16, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

uint64_t AclGraphExecutorImpl::get_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params) const {
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill()) {
    const uint64_t q_max_seq_len =
        static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1));
    if (params.graph.spec_verify_source_addresses_stable) {
      CHECK(params.attention.device.block_tables.defined());
      CHECK(params.graph.expanded_block_tables.defined());
      const uint64_t block_table_width =
          static_cast<uint64_t>(params.attention.device.block_tables.size(1));
      const uint64_t expanded_block_table_width =
          static_cast<uint64_t>(params.graph.expanded_block_tables.size(1));
      CHECK_LE(bucket_num_tokens, kSpecVerifyBucketMask);
      CHECK_LE(q_max_seq_len, kSpecVerifyFieldMask);
      CHECK_LE(block_table_width, kSpecVerifyFieldMask);
      CHECK_LE(expanded_block_table_width, kSpecVerifyExpandedBlockMask);
      // Captured D2D copies encode tensor shapes. Specialize the MTP target
      // graph by both block-table widths so a synthetic warmup graph cannot be
      // replayed with a real request's wider table view. Source storage may be
      // shared; its view shape and stride are still part of graph identity.
      const uint64_t base_key =
          kSpecVerifyGraphKeyMask | (expanded_block_table_width << 48) |
          (block_table_width << 32) | (q_max_seq_len << 16) |
          static_cast<uint64_t>(bucket_num_tokens);
      if (uses_static_mtp_graph_task_variant(params, bucket_num_tokens)) {
        return static_mtp_graph_task_key(base_key, params);
      }
      return base_key;
    }
    return static_cast<uint64_t>(bucket_num_tokens) | kSpecVerifyGraphKeyMask |
           (q_max_seq_len << kSpecVerifyQMaxSeqLenShift);
  }
  return static_cast<uint64_t>(bucket_num_tokens);
}

}  // namespace xllm::npu
