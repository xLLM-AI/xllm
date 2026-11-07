/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <framework/graphs/MLUGraph.h>
#include <torch/types.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_params_view.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/runtime/executor_impl.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/mlu_graph_planner.h"
#include "core/runtime/options.h"
#include "core/runtime/rec_executor_backend.h"
#include "core/runtime/rec_executor_impl_factory.h"

namespace xllm::mlu {
// Shared persistent buffers for graph execution. Every graph slices the same
// fixed-capacity tensors, so a capture keeps the addresses it recorded as long
// as each replay rewrites the whole slice it captured.
class GraphPersistentParam final {
 public:
  GraphPersistentParam(const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~GraphPersistentParam() = default;

  template <typename Params>
  void bind_buffers(Params& graph_params, const GraphPlan& plan);

  template <typename Params>
  void update_input_buffer(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const Params& params,
                           Params& graph_params,
                           const GraphPlan& plan);

  std::size_t get_persistent_tensor_bytes() const;

  // input tensors
  torch::Tensor tokens_;
  torch::Tensor positions_;

  // mrope
  bool use_mrope_ = false;
  // output
  torch::Tensor output_;
  torch::Tensor aux_hidden_states_;

 private:
  // attn_metadata
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor new_cache_slots_;
  torch::Tensor block_table_;
  torch::Tensor linear_state_indices_;

  // for vl
  torch::Tensor input_embeds_;
};

// Buffers that are owned per graph because their row count follows the layout.
// Request-row widths differ per layout, so sharing one persistent slice would
// let a later capture invalidate the pointers an earlier graph recorded.
struct GraphOwnedBuffers final {
  torch::Tensor num_accepted_tokens;
  torch::Tensor linear_state_validity_mask;
  std::vector<torch::Tensor> multi_block_tables;
};

// graph executor using libtorch MLUGraph for memory management
// MLUGraph provides mempool to manage temporary tensors during forward pass
class MluGraph final {
 public:
  MluGraph(GraphPersistentParam* persistent_param, GraphPlan plan);

  // Capture computation graph for given bucket num_tokens.
  // All buckets must capture on the same MLU stream so the caching allocator
  // can reuse scratch freed by earlier captures within the shared mempool;
  // capturing on different streams defeats its per-stream block reuse.
  template <typename ModelType>
  ModelOutput capture(ModelType* model,
                      std::vector<KVCache>& kv_cache,
                      const torch_mlu::MempoolId_t& pool,
                      const torch_mlu::MLUStream& capture_stream,
                      const runtime::Options& options);

  // Replay captured graph with new input data
  ModelOutput replay();

  template <typename ModelType, typename Params>
  void update_input_buffer(ModelType* model,
                           const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const Params& params,
                           bool is_init = false);

  void prepare_model_graph_metadata(CausalLM* model,
                                    const ModelParamsView& params);

  bool matches_graph_input(const ModelParamsView&) const { return true; }
  bool matches_graph_input(const RecModelParams& params) const;

 private:
  template <typename Params>
  Params& capture_params() {
    if constexpr (std::is_same_v<Params, ModelParamsView>) {
      return params_.value();
    } else {
      static_assert(std::is_same_v<Params, RecModelParams>);
      return rec_params_.value();
    }
  }

  // Each graph owns the parameter snapshot it captured with, so capturing a
  // different layout can never rebind this graph's tensors.
  std::optional<ModelParamsView> params_;
  std::optional<RecModelParams> rec_params_;
  GraphPlan plan_;
  bool enable_aux_hidden_states_ = false;
  // MLUGraph with mempool for managing temporary tensors during forward pass
  torch_mlu::MLUGraph graph_;

  // Reference to persistent parameters (shared across multiple MluGraph
  // instances)
  GraphPersistentParam* persistent_param_;  // not owned

  // Per-graph metadata state for models that require graph-forward
  // metadata preparation (e.g., DeepSeek V4 DSA metadata).
  std::unique_ptr<ModelGraphMetadataState> model_graph_metadata_state_;

  GraphOwnedBuffers owned_buffers_;
};

// Executor implementation using MLU graph optimization
// Uses MLUGraph mempool to reduce memory allocation overhead during inference
class MluGraphExecutorImpl final : public ExecutorImpl,
                                   public RecExecutorBackend {
 public:
  MluGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  MluGraphExecutorImpl(RecModel* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~MluGraphExecutorImpl() override = default;

  void set_dp_process_group(ProcessGroup* group) override;

  // Execute model with graph optimization for decode phase
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const RecModelParams& params) override;

 private:
  template <typename ModelType, typename Params>
  ModelOutput run_impl(ModelType* model,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       std::vector<KVCache>& kv_caches,
                       const Params& params);

  template <typename ModelType, typename Params>
  ModelOutput run_eager(ModelType* model,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const Params& params);
  void init_param_once();
  void log_memory_after_capture();

  CausalLM* model_;                // not owned
  RecModel* rec_model_ = nullptr;  // not owned
  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  torch_mlu::MempoolId_t graph_pool_;
  // Fixed capture stream shared by every bucket capture. Lazily initialized on
  // the first capture so the allocator can reuse pool scratch across buckets.
  std::optional<torch_mlu::MLUStream> graph_capture_stream_;
  std::size_t last_pool_reserved_bytes_ = 0;
  std::size_t peak_pool_reserved_bytes_ = 0;

  std::unique_ptr<MluGraphPlanner> planner_;
  ProcessGroup* dp_group_ = nullptr;  // owned by the worker context
  MtpModelCapabilities mtp_capabilities_;

  std::unordered_map<GraphKey, std::unique_ptr<MluGraph>, GraphKeyHash> graphs_;
  std::unordered_map<uint32_t, std::unique_ptr<MluGraph>> rec_graphs_;
  std::unique_ptr<GraphPersistentParam> persistent_param_;
};
REGISTER_EXECUTOR("mlu", MluGraphExecutorImpl);
REGISTER_REC_EXECUTOR("mlu", MluGraphExecutorImpl);
}  // namespace xllm::mlu
