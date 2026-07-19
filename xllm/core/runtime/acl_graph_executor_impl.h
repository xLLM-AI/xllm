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

  bool prepare_static_graph_tasks(const ModelInputParams& params,
                                  const c10_npu::NPUStream& signal_stream);
  bool prepare_static_mtp_graph_tasks(int64_t linear_state_id,
                                      int64_t num_accepted_tokens,
                                      int64_t spec_width,
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
  bool has_static_graph_task_signature_ = false;
  std::vector<int64_t> static_query_start_loc_;
  std::vector<int32_t> static_linear_state_ids_;
  std::vector<int64_t> static_num_accepted_tokens_;
  bool has_internal_spec_verify_input_update_ = false;
  std::vector<torch::Tensor> internal_spec_verify_input_sources_;
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

  bool prepare_static_graph_tasks(const torch::Tensor& tokens,
                                  const ModelInputParams& params,
                                  const Stream& signal_stream) override;
  bool prepare_static_mtp_graph_tasks(int64_t linear_state_id,
                                      int64_t num_accepted_tokens,
                                      int64_t spec_width,
                                      int64_t block_table_width,
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
    absl::flat_hash_map<uint64_t, std::unique_ptr<AclGraph>> graphs;
    bool is_prepared = false;
  };
  std::array<GraphSlot, 2> graph_slots_;
  std::mutex graph_slots_mutex_;
  int32_t graph_slot_count_ = 2;
  int32_t next_replay_slot_ = 0;
  int32_t last_started_replay_slot_ = -1;

  // Get bucket num_tokens for given num_tokens
  // For num_tokens < 8: use 1, 2, 4, 8
  // For num_tokens >= 8: use multiples of 8
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;

  uint64_t get_graph_key(uint32_t bucket_num_tokens,
                         const ModelInputParams& params) const;
};
REGISTER_EXECUTOR("npu", AclGraphExecutorImpl);
}  // namespace xllm::npu
