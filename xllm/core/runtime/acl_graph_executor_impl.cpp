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
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "core/kernels/ops_api.h"
#include "core/platform/device.h"
#include "core/platform/npu/acl_graph_task_update_context.h"
#include "core/runtime/mtp_async_state.h"
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
constexpr size_t kMaxStaticMtpGraphVariantsPerSlot = 16;
constexpr uint64_t kMlaGraphKeyMask = 1ull << 62;
constexpr uint64_t kMlaGraphKeyPayloadMask = (1ull << 62) - 1;
bool uses_static_mtp_graph_task_variant(const ModelInputParams& params,
                                        uint32_t bucket_num_tokens,
                                        int64_t block_size) {
  const int64_t batch_size = params.meta.num_sequences;
  const int64_t spec_width = params.meta.q_max_seq_len;
  return params.is_spec_verify &&
         params.meta.batch_forward_type.is_chunked_prefill() &&
         params.graph.use_expanded_decode_for_spec_verify_attention &&
         params.graph.spec_verify_source_addresses_stable &&
         kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
             spec_width, block_size) &&
         batch_size == 1 && spec_width > 0 &&
         bucket_num_tokens == batch_size * spec_width &&
         params.parallel.query_start_loc.size() ==
             static_cast<size_t>(batch_size + 1) &&
         params.embedding.linear_state_ids.size() ==
             static_cast<size_t>(batch_size) &&
         params.num_accepted_tokens_host.size() ==
             static_cast<size_t>(batch_size);
}

uint64_t mix_graph_key(uint64_t hash, uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  value ^= value >> 31;
  return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}

constexpr uint64_t paged_attention_plan_bucket_unchecked(int64_t max_kv,
                                                         int64_t block_size) {
  const uint64_t block =
      static_cast<uint64_t>((max_kv + block_size - 1) / block_size);
  // Keep the exact block endpoint separate. This preserves a conservative
  // boundary for vendor plans that may switch strategy on aligned lengths,
  // while reducing cold classifications from every token to at most two per
  // block.
  const uint64_t is_block_endpoint = max_kv % block_size == 0 ? 1 : 0;
  return (block << 1) | is_block_endpoint;
}

static_assert(paged_attention_plan_bucket_unchecked(127, 128) == 2);
static_assert(paged_attention_plan_bucket_unchecked(128, 128) == 3);
static_assert(paged_attention_plan_bucket_unchecked(129, 128) == 4);
static_assert(paged_attention_plan_bucket_unchecked(255, 128) == 4);
static_assert(paged_attention_plan_bucket_unchecked(256, 128) == 5);
static_assert(paged_attention_plan_bucket_unchecked(257, 128) == 6);

uint64_t paged_attention_plan_bucket(int64_t max_kv, int64_t block_size) {
  CHECK_GT(max_kv, 0);
  CHECK_GT(block_size, 0);
  return paged_attention_plan_bucket_unchecked(max_kv, block_size);
}

uint64_t spec_verify_packed_graph_key(uint32_t bucket_num_tokens,
                                      uint64_t q_max_seq_len,
                                      uint64_t block_table_width,
                                      uint64_t expanded_block_table_width) {
  CHECK_LE(bucket_num_tokens, kSpecVerifyBucketMask);
  CHECK_LE(q_max_seq_len, kSpecVerifyFieldMask);
  CHECK_LE(block_table_width, kSpecVerifyFieldMask);
  CHECK_LE(expanded_block_table_width, kSpecVerifyExpandedBlockMask);
  return kSpecVerifyGraphKeyMask | (expanded_block_table_width << 48) |
         (block_table_width << 32) | (q_max_seq_len << 16) |
         static_cast<uint64_t>(bucket_num_tokens);
}

uint64_t spec_verify_attention_plan_lookup_key(
    uint64_t packed_graph_key,
    const std::vector<int32_t>& expanded_kv_seq_lens,
    int64_t block_size) {
  CHECK(!expanded_kv_seq_lens.empty());
  const int64_t max_kv = *std::max_element(expanded_kv_seq_lens.begin(),
                                           expanded_kv_seq_lens.end());
  return mix_graph_key(packed_graph_key,
                       paged_attention_plan_bucket(max_kv, block_size));
}

uint64_t spec_verify_attention_plan_lookup_key(uint32_t bucket_num_tokens,
                                               const ModelInputParams& params,
                                               int64_t block_size) {
  CHECK(params.attention.device.block_tables.defined());
  CHECK(params.graph.expanded_block_tables.defined());
  const uint64_t packed_key = spec_verify_packed_graph_key(
      bucket_num_tokens,
      static_cast<uint64_t>(std::max<int32_t>(params.meta.q_max_seq_len, 1)),
      static_cast<uint64_t>(params.attention.device.block_tables.size(1)),
      static_cast<uint64_t>(params.graph.expanded_block_tables.size(1)));
  return spec_verify_attention_plan_lookup_key(
      packed_key, params.graph.expanded_kv_seq_lens_vec, block_size);
}

uint64_t static_mtp_graph_task_key(uint64_t base_key,
                                   const StaticGraphTaskSignature& signature) {
  uint64_t hash = mix_graph_key(kStaticGraphTaskHashSeed, base_key);
  hash = mix_graph_key(hash, static_cast<uint64_t>(signature.linear_state_id));
  hash =
      mix_graph_key(hash, static_cast<uint64_t>(signature.num_accepted_tokens));
  hash = mix_graph_key(hash,
                       static_cast<uint64_t>(signature.query_start_loc_begin));
  hash =
      mix_graph_key(hash, static_cast<uint64_t>(signature.query_start_loc_end));
  // Retain the spec-verify namespace bit. A signature comparison on replay
  // guards correctness even in the astronomically unlikely event of a hash
  // collision.
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

std::optional<std::array<const void*, 11>> spec_verify_input_addresses(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params) {
  const torch::Tensor& graph_tokens =
      params.graph.input_tokens_override.defined()
          ? params.graph.input_tokens_override
          : tokens;
  // The graph key fixes tensor view shapes; this address list protects their
  // backing storage. Fixed-capacity packed buffers keep the corresponding
  // strides stable across replay generations.
  const std::array<const torch::Tensor*, 11> sources = {
      &graph_tokens,
      &positions,
      &params.attention.device.q_seq_lens,
      &params.attention.device.kv_seq_lens,
      &params.attention.device.new_cache_slots,
      &params.attention.device.block_tables,
      &params.embedding.linear_state_indices,
      &params.num_accepted_tokens,
      &params.attention.device.q_cu_seq_lens,
      &params.graph.expanded_kv_seq_lens,
      &params.graph.expanded_block_tables};
  std::array<const void*, 11> addresses;
  for (size_t i = 0; i < sources.size(); ++i) {
    if (!sources[i]->defined()) {
      return std::nullopt;
    }
    addresses[i] = sources[i]->data_ptr();
  }
  return addresses;
}

ModelOutput forward_eager(CausalLM* model,
                          const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          std::vector<KVCache>& kv_cache,
                          const ModelInputParams& params) {
  const torch::Tensor& verify_tokens =
      params.graph.input_tokens_override.defined()
          ? params.graph.input_tokens_override
          : tokens;
  torch::Tensor materialized_tokens =
      mtp_async::materialize_speculative_verify_tokens(
          verify_tokens, params.graph.spec_verify_draft_token_sources);
  return model->forward(materialized_tokens, positions, kv_cache, params);
}

void hash_graph_key_value(uint64_t& hash, uint64_t value) {
  constexpr uint64_t kFnvPrime = 1099511628211ull;
  for (int32_t i = 0; i < 8; ++i) {
    hash ^= (value >> (i * 8)) & 0xffull;
    hash *= kFnvPrime;
  }
}

uint64_t get_mla_graph_key(uint32_t bucket_num_tokens,
                           int32_t capture_kv_seq_len_bucket) {
  constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  uint64_t hash = kFnvOffsetBasis;
  hash_graph_key_value(hash, bucket_num_tokens);
  hash_graph_key_value(hash, static_cast<uint64_t>(capture_kv_seq_len_bucket));

  return kMlaGraphKeyMask | (hash & kMlaGraphKeyPayloadMask);
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
  const bool update_spec_verify_tokens =
      params.graph.spec_verify_source_addresses_stable &&
      params.graph.input_tokens_override.defined() && params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill();
  auto graph_params = persistent_param_.update(tokens,
                                               k_cache,
                                               v_cache,
                                               positions,
                                               params,
                                               num_tokens_,
                                               /*return_capture_params=*/true,
                                               /*skip_token_update=*/
                                               update_spec_verify_tokens,
                                               /*for_capture=*/true);
  if (update_spec_verify_tokens) {
    persistent_param_.update_spec_verify_inputs(
        tokens,
        positions,
        params,
        num_tokens_,
        SpecVerifyInputUpdateScope::TOKENS_ONLY);
  }

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";
  const auto spec_verify_attention_plan =
      persistent_param_.paged_attention_plan_descriptor(
          actual_num_tokens, params.meta.q_max_seq_len);
  int64_t spec_verify_kv_split_core_count = 0;
  if (spec_verify_attention_plan.has_value()) {
    spec_verify_kv_split_core_count = static_cast<int64_t>(
        spec_verify_attention_plan->normalized_tiling.at(static_cast<size_t>(
            spec_verify_attention_plan->layout.kv_split_core_count_offset)));
  }
  const bool can_use_explicit_spec_verify_replay_update =
      model->is_hybrid_linear_attention() &&
      kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          params.meta.q_max_seq_len, options.block_size()) &&
      params.graph.spec_verify_source_addresses_stable &&
      params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill() &&
      spec_verify_attention_plan.has_value() &&
      spec_verify_kv_split_core_count > 0 &&
      actual_num_tokens == bucket_num_tokens && params.meta.num_sequences > 0 &&
      params.meta.q_max_seq_len > 0 &&
      actual_num_tokens == static_cast<uint32_t>(params.meta.num_sequences) *
                               params.meta.q_max_seq_len;
  if (can_use_explicit_spec_verify_replay_update) {
    spec_verify_block_size_ = options.block_size();
    spec_verify_kv_split_core_count_ = spec_verify_kv_split_core_count;
    spec_verify_paged_attention_tiling_layout_ =
        spec_verify_attention_plan->layout;
    const int64_t paged_attention_tiling_words = static_cast<int64_t>(
        spec_verify_attention_plan->normalized_tiling.size());
    CHECK_GT(paged_attention_tiling_words, 0);
    CHECK_LE(paged_attention_tiling_words,
             persistent_param_.tiling_data().numel());
    graph_paged_attention_tiling_data_ =
        persistent_param_.tiling_data().clone();
    // The paged-attention launch and the TileLang dynamic update must target
    // storage owned by this graph. A slot's persistent tiling tensor is shared
    // by every graph variant and can be overwritten while another variant is
    // still replaying.
    graph_params->graph.tiling_data = graph_paged_attention_tiling_data_;
    graph_params->graph.expanded_tiling_data =
        graph_paged_attention_tiling_data_;
  }
  prepare_model_graph_metadata(
      model,
      persistent_param_.persistent_positions(num_tokens_),
      graph_params.value());
  if (graph_paged_attention_tiling_data_.defined()) {
    spec_verify_input_addresses_at_capture_ =
        spec_verify_input_addresses(tokens, positions, params);
    // Raw TileLang launches are not replayed by the captured ACL graph. Run
    // the initial tiling update on the producer stream and let the existing
    // stream synchronization complete it before capture begins.
    update_spec_verify_attention_tiling(graph_params.value());
  }

  if (model->is_hybrid_linear_attention()) {
    graph_task_context_ = std::make_shared<AclGraphTaskUpdateContext>();
    graph_task_context_->begin_capture();
    graph_params->graph.acl_graph_task_update_context = graph_task_context_;
  }
  const bool capture_static_graph_tasks = uses_static_mtp_graph_task_variant(
      graph_params.value(), num_tokens_, options.block_size());
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
    VLOG(kGraphExecutorLogVerboseLevel)
        << "ACL graph capture begin, bucket_num_tokens=" << bucket_num_tokens
        << ", actual_num_tokens=" << actual_num_tokens;

    // no mempool id, will create a new one; capture mode is thread local, allow
    // other threads to execute synchronous operations
    bool capture_started = false;
    try {
      graph_.capture_begin(
          {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
      capture_started = true;
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
        persistent_param_.set_aux_hidden_states(
            forward_result.aux_hidden_states);
      }
      graph_.capture_end();
      capture_started = false;
    } catch (...) {
      if (capture_started) {
        try {
          graph_.capture_end();
        } catch (const std::exception& cleanup_error) {
          LOG(ERROR) << "ACL graph capture_end during cleanup failed: "
                     << cleanup_error.what();
        } catch (...) {
          LOG(ERROR) << "ACL graph capture_end during cleanup failed.";
        }
        graph_.reset();
      }
      if (need_restore_stream) {
        c10_npu::setCurrentNPUStream(
            c10_npu::getDefaultNPUStream(tensor_options.device().index()));
      }
      throw;
    }
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
  const auto current_signature = make_static_graph_task_signature(params);
  return current_signature.has_value() &&
         static_graph_task_signature_ == current_signature;
}

void AclGraph::capture_static_graph_task_signature(
    const ModelInputParams& params) {
  static_graph_task_signature_ = make_static_graph_task_signature(params);
  CHECK(static_graph_task_signature_.has_value());
  LOG(INFO) << "Captured static MTP graph-task signature: linear_state_id="
            << static_graph_task_signature_->linear_state_id
            << ", accepted_tokens="
            << static_graph_task_signature_->num_accepted_tokens;
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
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Initialized capture_stream"
      << ", id: " << capture_stream_.value().id()
      << ", device_index: " << static_cast<int32_t>(device_index);
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

void AclGraph::update_spec_verify_attention_tiling(
    const ModelInputParams& params) {
  CHECK(graph_paged_attention_tiling_data_.defined());
  CHECK(spec_verify_paged_attention_tiling_layout_.has_value());
  kernel::npu::tilelang::spec_verify_attention_tiling_update(
      params.graph.expanded_kv_seq_lens,
      graph_paged_attention_tiling_data_,
      spec_verify_paged_attention_tiling_layout_.value(),
      params.meta.q_max_seq_len,
      spec_verify_block_size_,
      params.meta.kv_max_seq_len,
      spec_verify_kv_split_core_count_);
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
  if (graph_paged_attention_tiling_data_.defined()) {
    const auto current_addresses =
        spec_verify_input_addresses(tokens, positions, params);
    if (!spec_verify_input_addresses_at_capture_.has_value() ||
        current_addresses != spec_verify_input_addresses_at_capture_) {
      LOG_FIRST_N(ERROR, 1)
          << "Falling back to eager speculative verification because graph "
             "input source storage moved after capture.";
      COUNTER_INC(num_model_execution_total_eager);
      return forward_eager(model, tokens, positions, kv_cache, params);
    }
    // Raw TileLang launches are not replayed as part of the captured ACL
    // graph on this runtime. Refresh the persistent metadata and the
    // graph-owned paged-attention tiling explicitly on the producer stream,
    // then let make_graph_wait_for_current_stream() carry the dependency.
    persistent_param_.update_spec_verify_inputs(
        tokens,
        positions,
        params,
        num_tokens_,
        SpecVerifyInputUpdateScope::ALL_INPUTS);
    update_spec_verify_attention_tiling(params);
    // Explicit producer-stream updates have populated the persistent graph
    // inputs. Host-only task parameters remain current and are consumed by
    // update_graph_tasks() below.
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

  if (graph_paged_attention_tiling_data_.defined()) {
    // The producer stream has refreshed inputs that include the final draft
    // token. Make graph replay wait for those updates on device; a host
    // synchronize here would recreate the bubble we remove.
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
    // variant. Signal its task-ready events immediately before replay; steady
    // supported-width cycles use the compute-stream pre-submit path instead.
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
  if (graph_paged_attention_tiling_data_.defined()) {
    return;
  }
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

bool AclGraph::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const c10_npu::NPUStream& signal_stream) {
  if (static_graph_task_signature_ !=
      make_static_graph_task_signature(signal)) {
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
        std::make_unique<GraphPersistentParam>(
            args_,
            device_,
            options_,
            need_update_attn_mask,
            is_hybrid_linear_attn,
            model_->supports_mla_graph_kv_bucketing());
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
    return forward_eager(model_, tokens, positions, kv_caches, params);
  }
  if (in_spec_verify_phase && !model_->is_hybrid_linear_attention()) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for spec verify because the "
           "chunked-prefill validate graph path is currently only adapted for "
           "hybrid linear attention models.";
    COUNTER_INC(num_model_execution_total_eager);
    return forward_eager(model_, tokens, positions, kv_caches, params);
  }
  // CP shards the query rows of a prefill batch and gathers them per layer, so
  // token counts and collectives differ from the captured decode shape. Decode
  // itself runs with CP inactive (both CP paths return early on decode), which
  // is why graph mode and CP can coexist -- but spec-verify chunked prefill is
  // a non-decode batch that reaches capture, so it must stay eager under CP.
  if (in_spec_verify_phase && options_.cp_size() > 1) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode for spec verify because context "
           "parallel (cp_size="
        << options_.cp_size()
        << ") shards prefill rows, which the captured graph shape does not "
           "describe.";
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
      return forward_eager(model_, tokens, positions, kv_caches, params);
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
      return forward_eager(model_, tokens, positions, kv_caches, params);
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
    return forward_eager(model_, tokens, positions, kv_caches, params);
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
    return forward_eager(model_, tokens, positions, kv_caches, params);
  }

  int32_t slot_idx = 0;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    slot_idx = next_replay_slot_;
    next_replay_slot_ = (next_replay_slot_ + 1) % graph_slot_count_;
    last_started_replay_slot_ = slot_idx;
    auto& slot = graph_slots_[slot_idx];
    slot.is_prepared = false;
  }
  auto& active_slot = graph_slots_[slot_idx];
  auto& active_persistent_param = *active_slot.persistent_param;

  uint64_t attention_plan_class = 0;
  const bool needs_attention_plan_class =
      params_single.is_spec_verify &&
      params_single.meta.batch_forward_type.is_chunked_prefill() &&
      params_single.graph.spec_verify_source_addresses_stable;
  if (needs_attention_plan_class) {
    const uint64_t lookup_key = spec_verify_attention_plan_lookup_key(
        bucket_num_tokens, params_single, options_.block_size());
    if (auto plan_class = find_spec_verify_attention_plan_class(lookup_key)) {
      attention_plan_class = plan_class.value();
    }
    if (attention_plan_class == 0) {
      // Cold path for a previously unseen KV block bucket. Run ATB Setup to
      // classify its immutable tiling plan, then reuse any graph already
      // captured for that plan class.
      auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_caches);
      auto descriptor =
          active_persistent_param.classify_spec_verify_paged_attention_plan(
              tokens_tensor, k_cache, v_cache, params_single);
      if (!descriptor.has_value()) {
        LOG_FIRST_N(ERROR, 1)
            << "Falling back to eager speculative verification because the "
               "paged-attention tiling layout cannot be classified safely.";
        COUNTER_INC(num_model_execution_total_eager);
        return forward_eager(
            model_, tokens, positions, kv_caches, params_single);
      }
      std::lock_guard<std::mutex> lock(graph_slots_mutex_);
      auto descriptor_it =
          std::find(spec_verify_attention_plan_descriptors_.begin(),
                    spec_verify_attention_plan_descriptors_.end(),
                    descriptor.value());
      if (descriptor_it == spec_verify_attention_plan_descriptors_.end()) {
        spec_verify_attention_plan_descriptors_.push_back(
            std::move(descriptor.value()));
        attention_plan_class = spec_verify_attention_plan_descriptors_.size();
      } else {
        attention_plan_class =
            static_cast<uint64_t>(
                std::distance(spec_verify_attention_plan_descriptors_.begin(),
                              descriptor_it)) +
            1;
      }
      auto [it, inserted] = spec_verify_attention_plan_classes_.emplace(
          lookup_key, attention_plan_class);
      CHECK(inserted || it->second == attention_plan_class)
          << "paged-attention plan class changed for one KV block bucket";
      attention_plan_class = it->second;
    }
  }

  const uint64_t graph_key =
      get_graph_key(bucket_num_tokens, params_single, attention_plan_class);
  std::shared_ptr<AclGraph> replay_graph;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto it = active_slot.graphs.find(graph_key);
    if (it != active_slot.graphs.end()) {
      replay_graph = it->second;
    }
  }

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
      std::make_shared<AclGraph>(active_persistent_param, device_.index());
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
               << bucket_num_tokens << ": " << e.what();
    if (model_->supports_mla_graph_kv_bucketing()) {
      throw;
    }
    LOG(ERROR) << "Falling back to eager mode.";
    COUNTER_INC(num_model_execution_total_eager);
    return forward_eager(model_, tokens, positions, kv_caches, params);
  }

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    const bool static_mtp_variant = uses_static_mtp_graph_task_variant(
        params_single, bucket_num_tokens, options_.block_size());
    {
      std::lock_guard<std::mutex> lock(graph_slots_mutex_);
      if (static_mtp_variant) {
        while (active_slot.static_mtp_graph_keys.size() >=
               kMaxStaticMtpGraphVariantsPerSlot) {
          const uint64_t evicted_key =
              active_slot.static_mtp_graph_keys.front();
          active_slot.static_mtp_graph_keys.pop_front();
          active_slot.graphs.erase(evicted_key);
        }
        active_slot.static_mtp_graph_keys.push_back(graph_key);
      }
      // shared_ptr keeps a replay/prepare that already left the map alive if a
      // later capture evicts this static variant.
      active_slot.graphs[graph_key] = graph;
    }

    // Return the output from capture (no need to replay since capture
    // already executed)
    torch::Tensor hidden_states = graph->get_hidden_states(n_tokens);
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
  return forward_eager(model_, tokens, positions, kv_caches, params);
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
  uint64_t attention_plan_class = 0;
  if (params.is_spec_verify &&
      params.meta.batch_forward_type.is_chunked_prefill() &&
      params.graph.spec_verify_source_addresses_stable) {
    const uint64_t lookup_key = spec_verify_attention_plan_lookup_key(
        bucket_num_tokens, params, options_.block_size());
    auto plan_class = find_spec_verify_attention_plan_class(lookup_key);
    if (!plan_class.has_value()) {
      return;
    }
    attention_plan_class = plan_class.value();
  }
  const uint64_t graph_key =
      get_graph_key(bucket_num_tokens, params, attention_plan_class);

  std::shared_ptr<AclGraph> graph;
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
    graph = it->second;
    slot.is_prepared = true;
  }
  graph->prepare_replay_inputs(tokens, positions, kv_caches, params);
}

bool AclGraphExecutorImpl::prepare_static_mtp_graph_tasks(
    const SpecVerifyGraphTaskSignal& signal,
    const Stream& signal_stream) {
  if (!model_->is_hybrid_linear_attention() || graph_slot_count_ != 1 ||
      !kernel::npu::tilelang::has_spec_verify_graph_update_specialization(
          signal.spec_width, options_.block_size()) ||
      signal.block_table_width < 1 ||
      signal.block_table_width > kSpecVerifyExpandedBlockMask ||
      signal.max_kv_seq_len < 1) {
    return false;
  }
  const uint64_t bucket_num_tokens = static_cast<uint64_t>(signal.spec_width);
  const uint64_t q_max_seq_len = static_cast<uint64_t>(signal.spec_width);
  const uint64_t width = static_cast<uint64_t>(signal.block_table_width);
  const uint64_t packed_key = spec_verify_packed_graph_key(
      static_cast<uint32_t>(bucket_num_tokens), q_max_seq_len, width, width);
  const uint64_t lookup_key =
      mix_graph_key(packed_key,
                    paged_attention_plan_bucket(signal.max_kv_seq_len,
                                                options_.block_size()));
  auto attention_plan_class = find_spec_verify_attention_plan_class(lookup_key);
  if (!attention_plan_class.has_value()) {
    return false;
  }
  const uint64_t base_key =
      mix_graph_key(packed_key, attention_plan_class.value());
  const uint64_t graph_key = static_mtp_graph_task_key(
      base_key, make_static_graph_task_signature(signal));
  std::shared_ptr<AclGraph> graph;
  {
    std::lock_guard<std::mutex> lock(graph_slots_mutex_);
    auto& graphs = graph_slots_[0].graphs;
    auto it = graphs.find(graph_key);
    if (it == graphs.end()) {
      return false;
    }
    graph = it->second;
  }
  return graph->prepare_static_mtp_graph_tasks(signal,
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
    // For num_tokens > 8, use multiples of 16.
    return ((num_tokens + 15) / 16) * 16;
  }
}

std::optional<uint64_t>
AclGraphExecutorImpl::find_spec_verify_attention_plan_class(
    uint64_t lookup_key) {
  std::lock_guard<std::mutex> lock(graph_slots_mutex_);
  auto it = spec_verify_attention_plan_classes_.find(lookup_key);
  if (it == spec_verify_attention_plan_classes_.end()) {
    return std::nullopt;
  }
  return it->second;
}

uint64_t AclGraphExecutorImpl::get_graph_key(
    uint32_t bucket_num_tokens,
    const ModelInputParams& params,
    uint64_t attention_plan_class) const {
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
      // Persistent graph inputs encode tensor view shapes. Specialize the MTP
      // target graph by both block-table widths so a synthetic warmup graph
      // cannot be replayed with a real request's wider table view.
      const uint64_t packed_key =
          spec_verify_packed_graph_key(bucket_num_tokens,
                                       q_max_seq_len,
                                       block_table_width,
                                       expanded_block_table_width);
      CHECK_NE(attention_plan_class, 0)
          << "stable speculative-verify graph requires an attention plan "
             "class";
      const uint64_t base_key = mix_graph_key(packed_key, attention_plan_class);
      if (uses_static_mtp_graph_task_variant(
              params, bucket_num_tokens, options_.block_size())) {
        const auto signature = make_static_graph_task_signature(params);
        CHECK(signature.has_value());
        return static_mtp_graph_task_key(base_key, signature.value());
      }
      return base_key;
    }
    return static_cast<uint64_t>(bucket_num_tokens) | kSpecVerifyGraphKeyMask |
           (q_max_seq_len << kSpecVerifyQMaxSeqLenShift);
  }
  if (model_->supports_mla_graph_kv_bucketing()) {
    const int32_t capture_kv_seq_len_bucket =
        get_mla_capture_kv_seq_len_bucket(params, options_);
    return get_mla_graph_key(bucket_num_tokens, capture_kv_seq_len_bucket);
  }
  return static_cast<uint64_t>(bucket_num_tokens);
}

}  // namespace xllm::npu
