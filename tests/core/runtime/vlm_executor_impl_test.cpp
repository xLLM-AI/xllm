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

#include "core/runtime/vlm_executor_impl.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/config/execution_config.h"
#include "core/platform/platform.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

class RoutingTestModel final : public CausalVLM {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelParamsView& /*params*/) override {
    return ModelOutput((tokens + positions).unsqueeze(/*dim=*/1));
  }

  MMDict encode(const ModelParamsView& /*params*/) override {
    ++encoder_calls;
    return {};
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& tokens,
      const ModelParamsView& /*params*/) override {
    ++embedding_calls;
    return tokens.unsqueeze(/*dim=*/1);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& /*selected_indices*/) override {
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}
  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  const torch::TensorOptions& options() const override { return options_; }

  int32_t encoder_calls = 0;
  int32_t embedding_calls = 0;
  int32_t executor_runs = 0;
  int32_t graph_prepares = 0;
  torch::Tensor prepared_positions;

 private:
  torch::TensorOptions options_ = torch::TensorOptions().device(torch::kCPU);
};

// Observe VLM delegation without requiring a device graph runtime.
class RoutingTestExecutor final : public ExecutorImpl {
 public:
  explicit RoutingTestExecutor(RoutingTestModel* model) : model_(model) {}

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelParamsView& params) override {
    ++model_->executor_runs;
    return model_->forward(tokens, positions, kv_caches, params);
  }

  void prepare_graph_input(const torch::Tensor& /*tokens*/,
                           const torch::Tensor& positions,
                           std::vector<KVCache>& /*kv_caches*/,
                           const ModelParamsView& /*params*/) override {
    ++model_->graph_prepares;
    model_->prepared_positions = positions;
  }

 private:
  RoutingTestModel* model_;
};

TEST(VlmExecutorImplTest, RoutingUsesIndependentInstanceGraphModes) {
  static const bool registered =
      ExecutorImplFactory::get_instance().register_creator(
          Platform::type_str(),
          "vlm_decode_options_test",
          [](CausalLM* model,
             const ModelArgs& /*args*/,
             const torch::Device& /*device*/,
             const runtime::Options& options) {
            EXPECT_TRUE(options.enable_graph() ||
                        options.enable_prefill_piecewise_graph());
            return std::make_unique<RoutingTestExecutor>(
                dynamic_cast<RoutingTestModel*>(model));
          });
  ASSERT_TRUE(registered);
  ExecutionConfig& config = ExecutionConfig::get_instance();
  const ExecutionConfig previous_config = config;
  ScopeGuard restore_config([&] { config = previous_config; });
  const torch::Tensor tokens = torch::tensor({1, 2, 3}, torch::kInt32);
  const torch::Tensor positions =
      torch::tensor({{10, 20, 30}, {40, 50, 60}, {70, 80, 90}}, torch::kInt32);
  const torch::Tensor expected =
      torch::tensor({11, 22, 33}, torch::kInt32).unsqueeze(/*dim=*/1);
  VlmModelParams params;
  params.meta.batch_forward_type = BatchForwardType::DECODE;
  params.meta.q_max_seq_len = 1;
  params.meta.num_sequences = 3;
  ModelParamsView view(params);
  ModelArgs args;
  args.rope_scaling_rope_type("mrope");

  struct GraphModeCase {
    bool decode;
    bool prefill;
  };
  const std::vector<GraphModeCase> cases{
      {false, false}, {true, false}, {false, true}, {true, true}};
  for (const auto& test_case : cases) {
    const bool enable_graph = test_case.decode;
    SCOPED_TRACE(enable_graph);
    SCOPED_TRACE(test_case.prefill);
    config.enable_graph(!enable_graph);
    runtime::Options options;
    options.backend("vlm")
        .max_encoder_cache_size(0)
        .extension_impl("vlm_decode_options_test")
        .enable_graph(enable_graph)
        .enable_prefill_piecewise_graph(test_case.prefill);
    RoutingTestModel model;
    VlmExecutorImpl executor(&model, args, model.device(), options);
    options.enable_graph(!enable_graph)
        .enable_prefill_piecewise_graph(!test_case.prefill);
    std::vector<KVCache> caches;

    executor.prepare_graph_input(tokens, positions, caches, view);
    EXPECT_EQ(model.graph_prepares, enable_graph ? 1 : 0);
    if (enable_graph) {
      EXPECT_TRUE(torch::equal(model.prepared_positions, positions[0]));
    }
    for (int32_t step = 0; step < 2; ++step) {
      config.enable_graph(step == 0 ? !enable_graph : enable_graph);
      const ModelOutput output = executor.run(tokens, positions, caches, view);
      EXPECT_TRUE(torch::equal(output.hidden_states, expected));
      EXPECT_EQ(model.executor_runs, enable_graph ? step + 1 : 0);
    }
    EXPECT_EQ(model.encoder_calls, 0);
    EXPECT_EQ(model.embedding_calls, 0);

    VlmModelParams prefill_params;
    prefill_params.meta.batch_forward_type = BatchForwardType::PREFILL;
    prefill_params.meta.num_sequences = 1;
    prefill_params.meta.q_max_seq_len = 3;
    prefill_params.attention.host.q_seq_lens = {3};
    prefill_params.attention.host.kv_seq_lens = {3};
    ModelParamsView prefill_view(prefill_params);
    const ModelOutput output =
        executor.run(tokens, positions[0], caches, prefill_view);
    EXPECT_TRUE(torch::equal(output.hidden_states, expected));
    EXPECT_EQ(model.encoder_calls, 1);
    EXPECT_EQ(model.embedding_calls, 1);
    EXPECT_EQ(
        model.executor_runs,
        (enable_graph ? 2 : 0) + (enable_graph || test_case.prefill ? 1 : 0));
  }
}

}  // namespace
}  // namespace xllm
