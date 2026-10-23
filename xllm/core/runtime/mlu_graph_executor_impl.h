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
#include <optional>
#include <type_traits>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_family_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/runtime/executor_impl.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/options.h"
#include "core/runtime/rec_executor_backend.h"
#include "core/runtime/rec_executor_impl_factory.h"

namespace xllm::mlu {
// Helper class to hold persistent parameters for graph execution
// Multiple MluGraph instances can share the same GraphPersistentParam object
class GraphPersistentParam final {
 public:
  GraphPersistentParam(const TransformerModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~GraphPersistentParam() = default;

  template <typename Params>
  void init_params(const Params& params,
                   uint32_t padding_num_tokens,
                   uint32_t padding_needed);

  // Update persistent tensors with new input data
  template <typename Params>
  void update_input_buffer(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const Params& params,
                           uint32_t padding_needed);

  std::size_t get_persistent_tensor_bytes() const;

  // input tensors
  torch::Tensor tokens_;
  torch::Tensor positions_;
  std::optional<ModelParamsView> params_;
  std::optional<RecModelParams> rec_params_;

  template <typename Params>
  Params& capture_params() {
    if constexpr (std::is_same_v<Params, ModelParamsView>) {
      return params_.value();
    } else {
      static_assert(std::is_same_v<Params, RecModelParams>);
      return rec_params_.value();
    }
  }
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
  uint32_t num_decoding_tokens_;
  torch::Tensor linear_state_indices_;

  // for vl
  torch::Tensor input_embeds_;

  // for mtp model
  torch::Tensor embedding_;

  // linear state indices for GDN models
  torch::Tensor linear_state_indices(uint32_t actual_batch_size = 0) const {
    if (linear_state_indices_.numel() == 0) {
      return linear_state_indices_;
    }
    if (actual_batch_size > 0) {
      return linear_state_indices_.slice(
          /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size);
    }
    return linear_state_indices_;
  }
};

// graph executor using libtorch MLUGraph for memory management
// MLUGraph provides mempool to manage temporary tensors during forward pass
class MluGraph final {
 public:
  MluGraph(GraphPersistentParam* persistent_param, uint32_t padding_num_tokens);

  // Capture computation graph for given bucket num_tokens.
  // All buckets must capture on the same MLU stream so the caching allocator
  // can reuse scratch freed by earlier captures within the shared mempool;
  // capturing on different streams defeats its per-stream block reuse.
  template <typename ModelType>
  void capture(ModelType* model,
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

  // Accessor for graph metadata state (used by executor to prepare
  // metadata before replay).
  ModelGraphMetadataState* model_graph_metadata_state() {
    return model_graph_metadata_state_.get();
  }

  void prepare_model_graph_metadata(CausalLM* model,
                                    const ModelParamsView& params);

  bool matches_graph_input(const ModelParamsView&) const { return true; }
  bool matches_graph_input(const RecModelParams& params) const;

 private:
  std::optional<RecModelParams> rec_graph_input_;
  // MLUGraph with mempool for managing temporary tensors during forward pass
  torch_mlu::MLUGraph graph_;

  // Reference to persistent parameters (shared across multiple MluGraph
  // instances)
  GraphPersistentParam* persistent_param_;  // not owned
  uint32_t padding_num_tokens_;

  // Per-graph metadata state for models that require graph-forward
  // metadata preparation (e.g., DeepSeek V4 DSA metadata).
  std::unique_ptr<ModelGraphMetadataState> model_graph_metadata_state_;
};

// Executor implementation using MLU graph optimization
// Uses MLUGraph mempool to reduce memory allocation overhead during inference
class MluGraphExecutorImpl final : public ExecutorImpl,
                                   public RecExecutorBackend {
 public:
  MluGraphExecutorImpl(CausalLM* model,
                       const TransformerModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  MluGraphExecutorImpl(RecModel* model,
                       const TransformerModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~MluGraphExecutorImpl() override = default;

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
  TransformerModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  torch_mlu::MempoolId_t graph_pool_;
  // Fixed capture stream shared by every bucket capture. Lazily initialized on
  // the first capture so the allocator can reuse pool scratch across buckets.
  std::optional<torch_mlu::MLUStream> graph_capture_stream_;
  int64_t max_tokens_for_graph_mode_ = 0;
  std::size_t last_pool_reserved_bytes_ = 0;
  std::size_t peak_pool_reserved_bytes_ = 0;

  std::unordered_map<uint32_t, std::unique_ptr<MluGraph>> graphs_;
  std::unique_ptr<GraphPersistentParam> persistent_param_;
};
REGISTER_EXECUTOR("mlu", MluGraphExecutorImpl);
REGISTER_REC_EXECUTOR("mlu", MluGraphExecutorImpl);
}  // namespace xllm::mlu
