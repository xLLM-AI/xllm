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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/options.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

class EplbLoadTestModel final : public CausalLM {
 public:
  explicit EplbLoadTestModel(const ModelContext& context)
      : options_(context.get_tensor_options()) {}

  ModelOutput forward(const torch::Tensor& /*tokens*/,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelParamsView& /*parameters*/) override {
    return {};
  }

  torch::Tensor logits(const torch::Tensor& /*hidden_states*/,
                       const torch::Tensor& /*selected_idxes*/) override {
    return {};
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}

  torch::Device device() const override { return options_.device(); }

  const torch::TensorOptions& options() const override { return options_; }

  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}

  void update_expert_weight(int32_t /*layer_id*/) override {}

 private:
  torch::TensorOptions options_;
};

class EplbInitWorker final : public LLMWorkerImpl {
 public:
  using LLMWorkerImpl::LLMWorkerImpl;

  bool initialize_model(int32_t num_speculative_tokens) {
    constexpr const char* kModelType = "eplb_worker_load_test";
    static const bool registered = ModelRegistry::register_causallm_factory(
        kModelType, "native", [](const ModelContext& context) {
          return std::make_unique<EplbLoadTestModel>(context);
        });
    if (!registered) {
      return false;
    }
    context_ = make_context(num_speculative_tokens);
    return init_model(context_);
  }

  ModelContext make_context(int32_t num_speculative_tokens) {
    ModelArgs args;
    args.model_type("eplb_worker_load_test")
        .dtype("float32")
        .n_layers(3)
        .first_k_dense_replace(1)
        .n_routed_experts(4);
    return load_model_config(std::move(args),
                             QuantArgs{},
                             num_speculative_tokens,
                             /*tokenizer=*/nullptr);
  }

  const torch::Tensor& expert_load_data() const { return expert_load_data_; }

  bool has_eplb_executor() const { return eplb_executor_ != nullptr; }
};

class EplbWorkerOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EPLBConfig& eplb = EPLBConfig::get_instance();
    previous_eplb_.enabled = eplb.enable_eplb();
    previous_eplb_.redundant_experts_num = eplb.redundant_experts_num();
    previous_eplb_.use_decode_only_load = eplb.eplb_use_decode_only_load();
    previous_model_impl_ = ModelConfig::get_instance().model_impl();
    previous_kernel_backend_ =
        KernelConfig::get_instance().npu_kernel_backend();
    previous_enable_graph_ = ExecutionConfig::get_instance().enable_graph();
    ModelConfig::get_instance().model_impl("native");
    KernelConfig::get_instance().npu_kernel_backend("TORCH");
    ExecutionConfig::get_instance().enable_graph(false);
  }

  void TearDown() override {
    EPLBConfig::get_instance()
        .enable_eplb(previous_eplb_.enabled)
        .redundant_experts_num(previous_eplb_.redundant_experts_num)
        .eplb_use_decode_only_load(previous_eplb_.use_decode_only_load);
    ModelConfig::get_instance().model_impl(previous_model_impl_);
    KernelConfig::get_instance().npu_kernel_backend(previous_kernel_backend_);
    ExecutionConfig::get_instance().enable_graph(previous_enable_graph_);
  }

 private:
  EplbWorkerConfig previous_eplb_;
  std::string previous_model_impl_;
  std::string previous_kernel_backend_;
  bool previous_enable_graph_ = false;
};

TEST_F(EplbWorkerOptionsTest,
       RetainsWorkerSnapshotAcrossTargetAndDraftContexts) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator is required for worker construction.";
  }
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    EPLBConfig::get_instance()
        .enable_eplb(!enabled)
        .redundant_experts_num(99)
        .eplb_use_decode_only_load(false);
    EplbWorkerConfig config;
    config.enabled = enabled;
    config.redundant_experts_num = 3;
    config.use_decode_only_load = true;
    runtime::Options options;
    options.backend("llm")
        .enable_graph(false)
        .enable_speculative_decode(true)
        .num_speculative_tokens(4)
        .eplb_worker_config(config);
    runtime::Options draft_options = options;
    draft_options.is_draft_engine(true);
    ParallelArgs parallel_args(/*rank=*/0,
                               /*world_size=*/1,
                               /*process_group=*/nullptr);
    EplbWorkerConfig conflicting_config;
    conflicting_config.enabled = !enabled;
    parallel_args.eplb_worker_config(conflicting_config);
    EplbInitWorker target(parallel_args, device, options);
    EplbInitWorker draft(parallel_args, device, draft_options);
    options.eplb_worker_config(EplbWorkerConfig{});
    draft_options.eplb_worker_config(EplbWorkerConfig{});
    parallel_args.eplb_worker_config(EplbWorkerConfig{});

    ModelContext target_context =
        target.make_context(/*num_speculative_tokens=*/4);
    ModelContext draft_context =
        draft.make_context(/*num_speculative_tokens=*/0);
    const ModelContext copied_context = target_context;
    const ModelContext quantized_context =
        draft_context.with_quant_args(QuantArgs{});
    const EplbWorkerConfig* snapshots[] = {
        &target.options().eplb_worker_config(),
        &draft.options().eplb_worker_config(),
        &target_context.get_parallel_args().eplb_worker_config(),
        &draft_context.get_parallel_args().eplb_worker_config(),
        &copied_context.get_parallel_args().eplb_worker_config(),
        &quantized_context.get_parallel_args().eplb_worker_config()};
    for (const EplbWorkerConfig* snapshot : snapshots) {
      EXPECT_EQ(snapshot->enabled, enabled);
      EXPECT_EQ(snapshot->redundant_experts_num, 3);
      EXPECT_TRUE(snapshot->use_decode_only_load);
    }
    EXPECT_FALSE(target.options().is_draft_engine());
    EXPECT_TRUE(draft.options().is_draft_engine());
    EXPECT_EQ(target.options().num_speculative_tokens(), 4);
    EXPECT_EQ(draft.options().num_speculative_tokens(), 4);
    EXPECT_EQ(target_context.num_speculative_tokens(), 4);
    EXPECT_EQ(draft_context.num_speculative_tokens(), 0);
    EXPECT_EQ(copied_context.num_speculative_tokens(), 4);
    EXPECT_EQ(quantized_context.num_speculative_tokens(), 0);
  }
}

TEST_F(EplbWorkerOptionsTest,
       LoadsFollowInjectedConfigAndExecutorBelongsToTarget) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator is required for worker initialization.";
  }
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  const ParallelArgs parallel_args(/*rank=*/0,
                                   /*world_size=*/1,
                                   /*process_group=*/nullptr);
  for (const bool enabled : {false, true}) {
    SCOPED_TRACE(enabled);
    EPLBConfig::get_instance()
        .enable_eplb(!enabled)
        .redundant_experts_num(99)
        .eplb_use_decode_only_load(false);
    EplbWorkerConfig config;
    config.enabled = enabled;
    config.redundant_experts_num = 3;
    config.use_decode_only_load = true;
    runtime::Options target_options;
    target_options.backend("llm")
        .enable_graph(false)
        .enable_speculative_decode(true)
        .num_speculative_tokens(4)
        .eplb_worker_config(config);
    runtime::Options draft_options = target_options;
    draft_options.is_draft_engine(true);
    EplbInitWorker target(parallel_args, device, target_options);
    EplbInitWorker draft(parallel_args, device, draft_options);
    target_options.eplb_worker_config(EplbWorkerConfig{});
    draft_options.eplb_worker_config(EplbWorkerConfig{});

    ASSERT_TRUE(target.initialize_model(/*num_speculative_tokens=*/4));
    ASSERT_TRUE(draft.initialize_model(/*num_speculative_tokens=*/0));
    EXPECT_EQ(target.has_eplb_executor(), enabled);
    EXPECT_FALSE(draft.has_eplb_executor());
    EXPECT_EQ(target.expert_load_data().defined(), enabled);
    EXPECT_EQ(draft.expert_load_data().defined(), enabled);
    if (!enabled) {
      continue;
    }
    const std::vector<int64_t> expected_shape{2, 7};
    ASSERT_TRUE(target.expert_load_data().defined());
    ASSERT_TRUE(draft.expert_load_data().defined());
    EXPECT_EQ(target.expert_load_data().sizes().vec(), expected_shape);
    EXPECT_EQ(draft.expert_load_data().sizes().vec(), expected_shape);
    EXPECT_EQ(target.expert_load_data().scalar_type(), torch::kInt64);
    EXPECT_EQ(draft.expert_load_data().scalar_type(), torch::kInt64);
    EXPECT_EQ(target.expert_load_data().device(), device);
    EXPECT_EQ(draft.expert_load_data().device(), device);
    EXPECT_NE(target.expert_load_data().data_ptr(),
              draft.expert_load_data().data_ptr());
    EXPECT_EQ(target.expert_load_data().count_nonzero().item<int64_t>(), 0);
    EXPECT_EQ(draft.expert_load_data().count_nonzero().item<int64_t>(), 0);
  }
}

}  // namespace
}  // namespace xllm
