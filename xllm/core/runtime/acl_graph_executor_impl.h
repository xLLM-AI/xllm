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

#pragma once

#include <absl/container/flat_hash_map.h>
#include <acl/acl.h>
#include <torch/torch.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "core/common/macros.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/runtime/acl_graph_persistent_param.h"
#include "executor_impl.h"
#include "executor_impl_factory.h"
#include "options.h"

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "torch_npu/csrc/core/npu/NPUGraph.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace xllm::npu {

struct AclGraphTaskUpdateContext;

struct StaticGraphTaskSignature {
  int64_t linear_state_id = 0;
  int64_t num_accepted_tokens = 0;
  int64_t query_start_loc_begin = 0;
  int64_t query_start_loc_end = 0;

  bool operator==(const StaticGraphTaskSignature&) const = default;
};

inline std::optional<StaticGraphTaskSignature> make_static_graph_task_signature(
    const ModelInputParams& params) {
  if (params.parallel.query_start_loc.size() != 2 ||
      params.embedding.linear_state_ids.size() != 1 ||
      params.num_accepted_tokens_host.size() != 1) {
    return std::nullopt;
  }
  return StaticGraphTaskSignature{
      .linear_state_id = params.embedding.linear_state_ids.front(),
      .num_accepted_tokens = params.num_accepted_tokens_host.front(),
      .query_start_loc_begin = params.parallel.query_start_loc.front(),
      .query_start_loc_end = params.parallel.query_start_loc.back(),
  };
}

inline StaticGraphTaskSignature make_static_graph_task_signature(
    const SpecVerifyGraphTaskSignal& signal) {
  return StaticGraphTaskSignature{
      .linear_state_id = signal.linear_state_id,
      .num_accepted_tokens = signal.num_accepted_tokens,
      .query_start_loc_begin = 0,
      .query_start_loc_end = signal.spec_width,
  };
}

// ACL graph executor using libtorch NPUGraph for memory management
// NPUGraph provides mempool to manage temporary tensors during forward pass
class AclGraph {
 public:
  explicit AclGraph(GraphPersistentParam& persistent_param,
                    c10::DeviceIndex device_index)
      : persistent_param_(persistent_param), device_index_(device_index) {
    // Initialize capture stream in constructor
    initialize_capture_stream(device_index);
  }

  ~AclGraph();

  // Capture computation graph for given bucket num_tokens
  bool capture(CausalLM* model,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_num_tokens);

  // Replay captured graph with new input data
  ModelOutput replay(CausalLM* model,
                     const torch::Tensor& tokens,
                     const torch::Tensor& positions,
                     std::vector<KVCache>& kv_cache,
                     const ModelInputParams& params);

  void prepare_replay_inputs(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params);

  bool prepare_static_mtp_graph_tasks(const SpecVerifyGraphTaskSignal& signal,
                                      const c10_npu::NPUStream& signal_stream);

  // Get the hidden states from the last capture
  torch::Tensor get_hidden_states(uint32_t actual_num_tokens = 0) const {
    return persistent_param_.hidden_states(actual_num_tokens);
  }

 private:
  // Print graph held tensors for debugging
  void print_graph_tensors() const;

  // Initialize capture stream if not already initialized
  void initialize_capture_stream(c10::DeviceIndex device_index);
  void make_graph_wait_for_current_stream(aclrtStream current_stream);
  void make_current_stream_wait_for_graph(aclrtStream current_stream);
  void prepare_model_graph_metadata(CausalLM* model,
                                    const torch::Tensor& positions,
                                    ModelInputParams& params);
  void update_spec_verify_attention_tiling(const ModelInputParams& params);

  bool update_graph_tasks(const ModelInputParams& params);
  void signal_static_graph_tasks(const c10_npu::NPUStream& signal_stream);
  bool static_graph_task_signature_matches(
      const ModelInputParams& params) const;
  void capture_static_graph_task_signature(const ModelInputParams& params);

  // NPUGraph with mempool for managing temporary tensors during forward pass
  c10_npu::NPUGraph graph_;
  uint32_t num_tokens_;

  // Reference to persistent parameters (shared across multiple AclGraph
  // instances)
  GraphPersistentParam& persistent_param_;
  std::unique_ptr<ModelGraphMetadataState> model_graph_metadata_state_;

  // Fallback non-default stream for capture when callers are on default stream.
  std::optional<c10_npu::NPUStream> capture_stream_;
  aclrtStream graph_stream_ = nullptr;
  aclrtEvent replay_input_ready_event_ = nullptr;
  aclrtEvent replay_done_event_ = nullptr;
  c10::DeviceIndex device_index_;
  std::shared_ptr<AclGraphTaskUpdateContext> graph_task_context_;
  std::optional<c10_npu::NPUStream> update_stream_;
  std::atomic<bool> replay_inputs_prepared_{false};
  std::optional<StaticGraphTaskSignature> static_graph_task_signature_;
  std::optional<std::array<const void*, 11>>
      spec_verify_input_addresses_at_capture_;
  torch::Tensor graph_paged_attention_tiling_data_;
  std::optional<kernel::npu::PagedAttentionTilingLayout>
      spec_verify_paged_attention_tiling_layout_;
  int64_t spec_verify_block_size_ = 0;
  int64_t spec_verify_kv_split_core_count_ = 0;
};

// Executor implementation using ACL graph optimization
// Uses NPUGraph mempool to reduce memory allocation overhead during inference
class AclGraphExecutorImpl : public ExecutorImpl {
 public:
  AclGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~AclGraphExecutorImpl() override = default;

  ForwardInput prepare_inputs(Batch& batch) override;

  // Execute model with graph optimization for decode phase
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

  void prepare_graph_input(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           std::vector<KVCache>& kv_caches,
                           const ModelInputParams& params) override;

  bool prepare_static_mtp_graph_tasks(const SpecVerifyGraphTaskSignal& signal,
                                      const Stream& signal_stream) override;

  [[nodiscard]] int32_t graph_slot_count_for_test() const {
    return graph_slot_count_;
  }

 private:
  // not own
  CausalLM* model_;

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  struct GraphSlot {
    std::unique_ptr<GraphPersistentParam> persistent_param;
    absl::flat_hash_map<uint64_t, std::shared_ptr<AclGraph>> graphs;
    std::deque<uint64_t> static_mtp_graph_keys;
    bool is_prepared = false;
  };
  std::array<GraphSlot, 2> graph_slots_;
  absl::flat_hash_map<uint64_t, uint64_t> spec_verify_attention_plan_classes_;
  std::vector<PagedAttentionPlanDescriptor>
      spec_verify_attention_plan_descriptors_;
  std::mutex graph_slots_mutex_;
  int32_t graph_slot_count_ = 2;
  int32_t next_replay_slot_ = 0;
  int32_t last_started_replay_slot_ = -1;

  // Get bucket num_tokens for given num_tokens
  // For num_tokens <= 8: use 1, 2, 4, 8
  // For num_tokens > 8: use multiples of 16
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;

  uint64_t get_graph_key(uint32_t bucket_num_tokens,
                         const ModelInputParams& params,
                         uint64_t attention_plan_class = 0) const;
  std::optional<uint64_t> find_spec_verify_attention_plan_class(
      uint64_t lookup_key);
};
REGISTER_EXECUTOR("npu", AclGraphExecutorImpl);
}  // namespace xllm::npu
