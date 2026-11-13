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
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/llm_worker_impl.h"
#include "core/runtime/mtp_worker_impl.h"
#include "core/runtime/options.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

class VlmMtpOptionsWorker final : public MTPWorkerImpl<VlmForwardInput> {
 public:
  VlmMtpOptionsWorker(const torch::Device& device,
                      const runtime::Options& options)
      : MTPWorkerImpl<VlmForwardInput>(ParallelArgs(/*rank=*/0,
                                                    /*world_size=*/1,
                                                    /*dp_size=*/1,
                                                    /*process_group=*/nullptr),
                                       device,
                                       options,
                                       WorkerType::VLM) {}

  const runtime::Options& target_options() const {
    return impl_.get()->*&VlmMtpOptionsWorker::options_;
  }

  const runtime::Options& draft_options() const {
    return draft_impl_.get()->*&VlmMtpOptionsWorker::options_;
  }

  const ParallelArgs& target_parallel_args() const {
    return impl_.get()->*&VlmMtpOptionsWorker::parallel_args_;
  }

  const ParallelArgs& draft_parallel_args() const {
    return draft_impl_.get()->*&VlmMtpOptionsWorker::parallel_args_;
  }
};

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

  bool initialize_model() {
    constexpr const char* kModelType = "mtp_eplb_load_test";
    static const bool registered = ModelRegistry::register_causallm_factory(
        kModelType, "native", [](const ModelContext& context) {
          return std::make_unique<EplbLoadTestModel>(context);
        });
    if (!registered) {
      return false;
    }
    ModelArgs args;
    args.model_type(kModelType)
        .n_layers(3)
        .first_k_dense_replace(1)
        .n_routed_experts(4);
    ModelContext context(parallel_args_,
                         args,
                         QuantArgs{},
                         torch::TensorOptions().device(device_.unwrap()));
    context.set_model_impl("native");
    return init_model(context);
  }

  const torch::Tensor& expert_load_data() const { return expert_load_data_; }

  bool has_eplb_executor() const { return eplb_executor_ != nullptr; }
};

class MtpEplbOptionsTest : public ::testing::Test {
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

TEST(MtpWorkerOptionsTest, VlmTargetKeepsVisionWhileDraftUsesText) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator is required for MTP worker construction.";
  }
  const torch::Device device(Platform::type_torch(), /*index=*/0);
  runtime::Options options;
  options.backend("vlm").enable_graph(true).num_speculative_tokens(4);
  VlmMtpOptionsWorker worker(device, options);

  EXPECT_EQ(worker.target_options().backend(), "vlm");
  EXPECT_EQ(worker.draft_options().backend(), "llm");
  EXPECT_TRUE(worker.target_options().enable_graph());
  EXPECT_TRUE(worker.draft_options().enable_graph());
}

TEST_F(MtpEplbOptionsTest, TargetAndDraftRetainEplbSnapshotAndDistinctRoles) {
  if (Platform::device_count() < 1) {
    GTEST_SKIP() << "An accelerator is required for MTP worker construction.";
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
    options.backend("vlm")
        .enable_graph(true)
        .enable_speculative_decode(true)
        .num_decoding_tokens(5)
        .num_speculative_tokens(4)
        .eplb_worker_config(config);
    VlmMtpOptionsWorker worker(device, options);
    options.eplb_worker_config(EplbWorkerConfig{});

    const EplbWorkerConfig* snapshots[] = {
        &worker.target_options().eplb_worker_config(),
        &worker.draft_options().eplb_worker_config(),
        &worker.target_parallel_args().eplb_worker_config(),
        &worker.draft_parallel_args().eplb_worker_config()};
    for (const EplbWorkerConfig* snapshot : snapshots) {
      EXPECT_EQ(snapshot->enabled, enabled);
      EXPECT_EQ(snapshot->redundant_experts_num, 3);
      EXPECT_TRUE(snapshot->use_decode_only_load);
    }
    EXPECT_EQ(worker.target_options().backend(), "vlm");
    EXPECT_EQ(worker.draft_options().backend(), "llm");
    EXPECT_FALSE(worker.target_options().is_draft_engine());
    EXPECT_TRUE(worker.draft_options().is_draft_engine());
    EXPECT_EQ(worker.target_options().num_decoding_tokens(), 5);
    EXPECT_EQ(worker.draft_options().num_decoding_tokens(), 1);
    EXPECT_EQ(worker.target_options().num_speculative_tokens(), 4);
    EXPECT_EQ(worker.draft_options().num_speculative_tokens(), 0);
    EXPECT_TRUE(worker.target_options().enable_graph());
    EXPECT_TRUE(worker.draft_options().enable_graph());
  }
}

TEST_F(MtpEplbOptionsTest,
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
        .num_decoding_tokens(5)
        .num_speculative_tokens(4)
        .eplb_worker_config(config);
    runtime::Options draft_options = target_options;
    draft_options.is_draft_engine(true)
        .num_decoding_tokens(1)
        .num_speculative_tokens(0);
    EplbInitWorker target(parallel_args, device, target_options);
    EplbInitWorker draft(parallel_args, device, draft_options);
    target_options.eplb_worker_config(EplbWorkerConfig{});
    draft_options.eplb_worker_config(EplbWorkerConfig{});

    ASSERT_TRUE(target.initialize_model());
    ASSERT_TRUE(draft.initialize_model());
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
