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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/platform/device.h"
#include "core/runtime/dcu_graph_executor_impl.h"
#include "core/runtime/options.h"
#include "core/util/scope_guard.h"

namespace xllm {
namespace {

bool IsDcuAvailable() {
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  return err == hipSuccess && count > 0;
}

void DcuSynchronize() {
  hipError_t err = hipDeviceSynchronize();
  CHECK_EQ(err, hipSuccess)
      << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
}

class ScopedExecutionConfigSnapshot final {
 public:
  ScopedExecutionConfigSnapshot() = default;
  ~ScopedExecutionConfigSnapshot() {
    ExecutionConfig::get_instance() = previous_;
  }

 private:
  ExecutionConfig previous_ = ExecutionConfig::get_instance();
};

class DcuGraphExecutorTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("dcu_graph_executor_test");
    google::SetStderrLogging(google::INFO);

    if (IsDcuAvailable()) {
      xllm::Device xllm_device(0);
      xllm_device.set_device();
    }

    FLAGS_block_size = 1;
    FLAGS_max_tokens_per_batch = 128;

    torch::manual_seed(0);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new DcuGraphExecutorTestEnvironment);

torch::Device InitXllmDcuDeviceForTest(int32_t device_index = 0) {
  xllm::Device xllm_device(device_index);
  xllm_device.set_device();
  return xllm_device.unwrap();
}

// Minimal fake model for DCU graph tests.
// It intentionally avoids CUDA Attention / FlashInfer / PlanInfo.
class FakeSimpleCausalLM final : public CausalLM {
 public:
  FakeSimpleCausalLM(const ModelArgs& args, const torch::Device& device)
      : args_(args),
        device_(device),
        options_(
            torch::TensorOptions().dtype(torch::kBFloat16).device(device)) {
    const int64_t vocab_size = std::max<int64_t>(args_.vocab_size(), 1024);

    embedding_ =
        register_module("embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(
                            vocab_size, args_.hidden_size())));

    proj_ = register_module("proj",
                            torch::nn::Linear(torch::nn::LinearOptions(
                                args_.hidden_size(), args_.hidden_size())));

    this->to(device_);

    proj_->to(torch::kBFloat16);
    embedding_->weight.data().normal_();
    proj_->weight.data().normal_();

    if (proj_->bias.defined()) {
      proj_->bias.data().normal_();
    }
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& params) override {
    (void)kv_caches;
    if (params.enable_graph) {
      captured_token_rows_ = tokens.size(0);
    }

    auto token_ids = tokens.to(torch::kInt64);
    auto x = embedding_->forward(token_ids).to(torch::kBFloat16);
    auto y = proj_->forward(x);

    auto pos_bias =
        positions.to(y.scalar_type()).reshape({positions.size(0), 1}) *
        static_cast<double>(0.001);

    return ModelOutput(y + pos_bias);
  }

  int64_t captured_token_rows() const { return captured_token_rows_; }

  const torch::TensorOptions& options() const override { return options_; }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    (void)selected_idxes;

    const int64_t vocab_size = std::max<int64_t>(args_.vocab_size(), 1024);
    return torch::zeros({hidden_states.size(0), vocab_size},
                        torch::dtype(torch::kFloat32).device(device_));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    (void)loader;
  }

  torch::Device device() const override { return device_; }

  void prepare_expert_weight(int32_t expert_id,
                             const std::vector<int32_t>& expert_ids) override {
    (void)expert_id;
    (void)expert_ids;
  }

  void update_expert_weight(int32_t expert_id) override { (void)expert_id; }

 private:
  int64_t captured_token_rows_ = 0;
  ModelArgs args_;
  torch::Device device_;
  torch::TensorOptions options_;

  torch::nn::Embedding embedding_{nullptr};
  torch::nn::Linear proj_{nullptr};
};

ModelArgs MakeFakeModelArgs(int64_t hidden_size = 64) {
  ModelArgs args;

  args.model_type("fake_simple");
  args.dtype("bfloat16");
  args.hidden_size(hidden_size);
  args.max_position_embeddings(128);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(1);
  args.head_dim(hidden_size);
  args.n_kv_heads(1);

  return args;
}

runtime::Options MakeRuntimeOptions(int32_t max_seqs_per_batch = 8) {
  runtime::Options options;

  options.block_size(1);
  options.max_tokens_per_batch(128);
  options.max_seqs_per_batch(max_seqs_per_batch);

  return options;
}

template <typename Params = ModelParamsView>
Params MakeDecodeParams(const torch::Device& device,
                        int32_t num_tokens,
                        int32_t kv_len = 4) {
  CHECK_GT(num_tokens, 0);
  CHECK_GT(kv_len, 0);

  Params p = [] { return ModelInputSnapshot(LlmModelParams()).view(); }();

  p.meta.batch_forward_type = BatchForwardType::DECODE;
  p.meta.num_sequences = num_tokens;
  p.meta.kv_max_seq_len = kv_len;
  p.meta.q_max_seq_len = 1;
  p.enable_graph = false;

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  std::vector<int32_t> q_seq_lens(num_tokens, /*value=*/1);
  std::vector<int32_t> kv_seq_lens(num_tokens, kv_len);
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;

  q_cu_seq_lens.reserve(num_tokens + 1);
  kv_cu_seq_lens.reserve(num_tokens + 1);

  for (int32_t i = 0; i <= num_tokens; ++i) {
    q_cu_seq_lens.push_back(i);
    kv_cu_seq_lens.push_back(i * kv_len);
  }

  p.attention.host.q_seq_lens = q_seq_lens;
  p.attention.host.kv_seq_lens = kv_seq_lens;
  p.attention.host.q_cu_seq_lens = q_cu_seq_lens;

  p.attention.device.q_seq_lens = torch::tensor(q_seq_lens, iopt);
  p.attention.device.kv_seq_lens = torch::tensor(kv_seq_lens, iopt);
  p.attention.device.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, iopt);

  std::vector<int32_t> slots;
  slots.reserve(num_tokens);

  for (int32_t i = 0; i < num_tokens; ++i) {
    slots.push_back(i * kv_len + kv_len - 1);
  }

  p.attention.host.new_cache_slots = slots;
  p.attention.device.new_cache_slots = torch::tensor(slots, iopt);

  p.attention.device.block_tables =
      torch::arange(0, num_tokens * kv_len, iopt).reshape({num_tokens, kv_len});

  p.attention.device.paged_kv_indptr = torch::tensor(kv_cu_seq_lens, iopt);
  p.attention.device.paged_kv_indices =
      torch::arange(0, num_tokens * kv_len, iopt);
  p.attention.device.paged_kv_last_page_len = torch::ones({num_tokens}, iopt);

  return p;
}

ModelParamsView MakePrefillParams(const torch::Device& device,
                                  int32_t num_tokens) {
  CHECK_GT(num_tokens, 0);

  ModelParamsView p = ModelInputSnapshot(LlmModelParams()).view();

  p.meta.batch_forward_type = BatchForwardType::PREFILL;
  p.meta.num_sequences = 1;
  p.meta.kv_max_seq_len = num_tokens;
  p.meta.q_max_seq_len = num_tokens;
  p.enable_graph = false;

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  p.attention.host.q_seq_lens = {num_tokens};
  p.attention.host.kv_seq_lens = {num_tokens};
  p.attention.host.q_cu_seq_lens = {0, num_tokens};

  p.attention.device.q_seq_lens = torch::tensor({num_tokens}, iopt);
  p.attention.device.kv_seq_lens = torch::tensor({num_tokens}, iopt);
  p.attention.device.q_cu_seq_lens = torch::tensor({0, num_tokens}, iopt);

  p.attention.host.new_cache_slots = std::vector<int32_t>(num_tokens);
  std::iota(p.attention.host.new_cache_slots.begin(),
            p.attention.host.new_cache_slots.end(),
            0);
  p.attention.device.new_cache_slots = torch::arange(0, num_tokens, iopt);
  p.attention.device.block_tables =
      torch::arange(0, num_tokens, iopt).unsqueeze(0);

  p.attention.device.paged_kv_indptr = torch::tensor({0, num_tokens}, iopt);
  p.attention.device.paged_kv_indices = torch::arange(0, num_tokens, iopt);
  p.attention.device.paged_kv_last_page_len = torch::tensor({1}, iopt);

  return p;
}

std::vector<KVCache> MakeKvCaches(const torch::Device& device,
                                  int64_t num_pages,
                                  int64_t page_size,
                                  int64_t num_kv_heads,
                                  int64_t head_dim) {
  auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  auto k_cache =
      torch::randn({num_pages, page_size, num_kv_heads, head_dim}, opt);
  auto v_cache =
      torch::randn({num_pages, page_size, num_kv_heads, head_dim}, opt);

  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(KVCacheTensors{k_cache, v_cache});
  return kv_caches;
}

bool AllCloseBf16(const torch::Tensor& lhs,
                  const torch::Tensor& rhs,
                  double rtol = 1e-2,
                  double atol = 1e-2) {
  return torch::allclose(
      lhs.to(torch::kFloat32), rhs.to(torch::kFloat32), rtol, atol);
}

TEST(DcuGraphExecutorTest, PersistentBuffersUseRuntimeTokenCapacity) {
  if (!IsDcuAvailable()) {
    GTEST_SKIP() << "DCU/HIP is not available at runtime.";
  }
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const int32_t previous_tokens = scheduler_config.max_tokens_per_batch();
  ScopeGuard restore_config(
      [&] { scheduler_config.max_tokens_per_batch(previous_tokens); });
  scheduler_config.max_tokens_per_batch(1);

  const torch::Device device = InitXllmDcuDeviceForTest(/*device_index=*/0);
  runtime::Options options = MakeRuntimeOptions(/*max_seqs_per_batch=*/3);
  options.max_tokens_per_batch(7);
  runtime::dcu::DcuGraphPersistentParam buffers(
      MakeFakeModelArgs(), device, options);
  options.max_tokens_per_batch(99);
  scheduler_config.max_tokens_per_batch(2);
  EXPECT_EQ(buffers.persistent_tokens(/*n_tokens=*/0).size(0), 7);
  EXPECT_EQ(buffers.persistent_positions(/*n_tokens=*/0).size(0), 7);

  const torch::Tensor hidden = torch::ones(
      {7, 4}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
  buffers.set_aux_hidden_states(hidden);
  EXPECT_EQ(buffers.aux_hidden_states(/*n_tokens=*/0).size(0), 7);
  EXPECT_TRUE(torch::equal(buffers.aux_hidden_states(/*n_tokens=*/7), hidden));
}

TEST(DcuGraphExecutorTest, DecodeGraphUsesInstanceSwitchAndDpModes) {
  if (!IsDcuAvailable()) {
    GTEST_SKIP() << "DCU/HIP is not available at runtime.";
  }

  ScopedExecutionConfigSnapshot config_snapshot;
  const torch::Device device = InitXllmDcuDeviceForTest(/*device_index=*/0);
  ModelArgs args = MakeFakeModelArgs(/*hidden_size=*/64);
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::Tensor tokens = torch::tensor({1, 2, 3}, int_options);
  const torch::Tensor positions = torch::tensor({0, 1, 2}, int_options);

  struct DecodeCase {
    const char* name;
    bool has_dp;
    bool all_decode;
  };
  const std::vector<DecodeCase> cases{{"single_rank", false, true},
                                      {"dp_all_decode", true, true},
                                      {"dp_mixed", true, false}};
  for (const bool enable_graph : {false, true}) {
    SCOPED_TRACE(enable_graph);
    for (const auto& test_case : cases) {
      SCOPED_TRACE(test_case.name);
      ExecutionConfig::get_instance()
          .enable_graph(!enable_graph)
          .enable_graph_mode_decode_no_padding(false);
      runtime::Options options = MakeRuntimeOptions(/*max_seqs_per_batch=*/8);
      options.enable_graph(enable_graph)
          .enable_graph_mode_decode_no_padding(true);
      auto model = std::make_unique<FakeSimpleCausalLM>(args, device);
      runtime::dcu::DcuGraphExecutorImpl executor(
          model.get(), args, device, options);
      options.enable_graph(!enable_graph);

      auto params = MakeDecodeParams(device, /*num_tokens=*/3);
      if (test_case.has_dp) {
        params.parallel.dp_global_token_nums = {3, 5};
        params.parallel.dp_is_decode = {1, test_case.all_decode ? 1 : 0};
      }
      auto caches = MakeKvCaches(device,
                                 /*num_pages=*/16,
                                 /*page_size=*/1,
                                 /*num_kv_heads=*/1,
                                 /*head_dim=*/64);
      const torch::Tensor eager_output =
          model->forward(tokens, positions, caches, params)
              .hidden_states.clone();
      DcuSynchronize();
      const bool captures = enable_graph && test_case.all_decode;
      const int64_t captured_rows = captures ? (test_case.has_dp ? 5 : 3) : 0;
      for (int32_t step = 0; step < 2; ++step) {
        const torch::Tensor output =
            executor.run(tokens, positions, caches, params)
                .hidden_states.clone();
        DcuSynchronize();
        EXPECT_EQ(model->captured_token_rows(), captured_rows);
        EXPECT_TRUE(AllCloseBf16(output, eager_output));
      }
    }
  }
}

TEST(DcuGraphExecutorTest, PrefillGraphUsesInstanceSwitchAndTokenLimit) {
  if (!IsDcuAvailable()) {
    GTEST_SKIP() << "DCU/HIP is not available at runtime.";
  }

  ScopedExecutionConfigSnapshot config_snapshot;
  const torch::Device device = InitXllmDcuDeviceForTest(/*device_index=*/0);
  ModelArgs args = MakeFakeModelArgs(/*hidden_size=*/64);
  constexpr int32_t kNumTokens = 7;
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const torch::Tensor tokens = torch::arange(1, kNumTokens + 1, int_options);
  const torch::Tensor positions = torch::arange(0, kNumTokens, int_options);
  const auto params = MakePrefillParams(device, kNumTokens);

  struct PrefillCase {
    const char* name;
    bool piecewise_enabled;
    int32_t token_limit;
    int32_t max_sequences;
    bool captures;
  };
  const std::vector<PrefillCase> cases{
      {"disabled", false, 128, 1, false},
      {"over_limit", true, 6, 1, false},
      {"at_limit", true, 7, 7, true},
      {"sequence_capacity_floor", true, 1, 7, true}};
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ExecutionConfig::get_instance()
        .enable_graph(true)
        .enable_prefill_piecewise_graph(!test_case.piecewise_enabled)
        .max_tokens_for_graph_mode(test_case.captures ? 1 : 128);
    runtime::Options options = MakeRuntimeOptions(test_case.max_sequences);
    options.enable_prefill_piecewise_graph(test_case.piecewise_enabled)
        .max_tokens_for_graph_mode(test_case.token_limit);
    auto model = std::make_unique<FakeSimpleCausalLM>(args, device);
    runtime::dcu::DcuGraphExecutorImpl executor(
        model.get(), args, device, options);
    options.enable_prefill_piecewise_graph(!test_case.piecewise_enabled)
        .max_tokens_for_graph_mode(128);
    auto caches = MakeKvCaches(device,
                               /*num_pages=*/16,
                               /*page_size=*/1,
                               /*num_kv_heads=*/1,
                               /*head_dim=*/64);
    const torch::Tensor eager_output =
        model->forward(tokens, positions, caches, params).hidden_states.clone();
    DcuSynchronize();
    for (int32_t step = 0; step < 2; ++step) {
      const torch::Tensor output =
          executor.run(tokens, positions, caches, params).hidden_states.clone();
      DcuSynchronize();
      EXPECT_EQ(model->captured_token_rows(),
                test_case.captures ? kNumTokens : 0);
      EXPECT_TRUE(AllCloseBf16(output, eager_output));
    }
  }
}

}  // namespace
}  // namespace xllm
