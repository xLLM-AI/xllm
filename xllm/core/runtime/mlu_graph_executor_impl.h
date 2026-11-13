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
#include <unordered_map>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_params_view.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/runtime/executor_impl.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/mlu_graph_planner.h"
#include "core/runtime/options.h"

namespace xllm::mlu {
class GraphPersistentParam final {
 public:
  GraphPersistentParam(const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       const MtpModelCapabilities& capabilities);

  ~GraphPersistentParam() = default;

  void bind_buffers(ModelParamsView& graph_params, const GraphPlan& plan);

  void update_input_buffer(const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const ModelParamsView& params,
                           ModelParamsView& graph_params,
                           const GraphPlan& plan);

  std::size_t get_persistent_tensor_bytes() const;

  torch::Tensor tokens_;
  torch::Tensor positions_;

  bool use_mrope_ = false;
  torch::Tensor output_;
  torch::Tensor aux_hidden_states_;

 private:
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor new_cache_slots_;
  torch::Tensor block_table_;
  torch::Tensor linear_state_indices_;

  torch::Tensor input_embeds_;
};

struct GraphOwnedBuffers final {
  torch::Tensor num_accepted_tokens;
  std::vector<torch::Tensor> multi_block_tables;
};

struct GraphActionWorkspace final {
  torch::Tensor input_host;
  torch::Tensor input_device;
  torch::Tensor output_device;
  torch::Tensor output_host;
};

class MluGraph final {
 public:
  MluGraph(GraphPersistentParam* persistent_param, GraphPlan plan);

  // All buckets must capture on the same MLU stream so the caching allocator
  // can reuse scratch freed by earlier captures within the shared mempool;
  // capturing on different streams defeats its per-stream block reuse.
  ModelOutput capture(CausalLM* model,
                      std::vector<KVCache>& kv_cache,
                      const torch_mlu::MempoolId_t& pool,
                      const torch_mlu::MLUStream& capture_stream,
                      const runtime::Options& options);

  ModelOutput replay();

  void update_input_buffer(CausalLM* model,
                           const torch::Tensor& tokens,
                           const torch::Tensor& positions,
                           const ModelParamsView& params,
                           bool is_init = false);

  void prepare_model_graph_metadata(CausalLM* model,
                                    const ModelParamsView& params);

 private:
  ModelOutput persistent_output() const;

  std::optional<ModelParamsView> params_;
  GraphPlan plan_;
  bool enable_aux_hidden_states_ = false;
  torch_mlu::MLUGraph graph_;

  GraphPersistentParam* persistent_param_;  // not owned

  std::unique_ptr<ModelGraphMetadataState> model_graph_metadata_state_;

  GraphOwnedBuffers owned_buffers_;
};

class MluGraphExecutorImpl final : public ExecutorImpl {
 public:
  MluGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~MluGraphExecutorImpl() override = default;

  void set_dp_process_group(ProcessGroup* group) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params,
                  const torch::Tensor& logits_indices) override;

 private:
  ModelOutput run_eager(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelParamsView& params);
  void init_param_once();
  void log_memory_after_capture();

  CausalLM* model_;  // not owned
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
  ProcessGroup* dp_group_ = nullptr;
  GraphActionWorkspace graph_action_workspace_;
  MtpModelCapabilities mtp_capabilities_;

  std::unordered_map<GraphKey, std::unique_ptr<MluGraph>, GraphKeyHash> graphs_;
  std::unique_ptr<GraphPersistentParam> persistent_param_;
};
REGISTER_EXECUTOR("mlu", MluGraphExecutorImpl);

}  // namespace xllm::mlu
