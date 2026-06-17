/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/platform/device.h"
#include "core/runtime/dcu_graph_executor_impl.h"
#include "core/runtime/options.h"

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
    FLAGS_enable_graph_mode_decode_no_padding = true;

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
                      const ModelInputParams& params) override {
    (void)kv_caches;
    (void)params;

    auto token_ids = tokens.to(torch::kInt64);
    auto x = embedding_->forward(token_ids).to(torch::kBFloat16);
    auto y = proj_->forward(x);

    auto pos_bias =
        positions.to(y.scalar_type()).reshape({positions.size(0), 1}) *
        static_cast<double>(0.001);

    return ModelOutput(y + pos_bias);
  }

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
  options.max_seqs_per_batch(max_seqs_per_batch);

  return options;
}

ModelInputParams MakeDecodeParams(const torch::Device& device,
                                  int32_t num_tokens,
                                  int32_t kv_len = 4) {
  CHECK_GT(num_tokens, 0);
  CHECK_GT(kv_len, 0);

  ModelInputParams p;

  p.meta.batch_forward_type = BatchForwardType::DECODE;
  p.meta.num_sequences = num_tokens;
  p.meta.kv_max_seq_len = kv_len;
  p.meta.q_max_seq_len = 1;
  p.enable_graph = false;

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_cu_seq_lens;

  q_cu_seq_lens.reserve(num_tokens + 1);
  kv_cu_seq_lens.reserve(num_tokens + 1);

  for (int32_t i = 0; i <= num_tokens; ++i) {
    q_cu_seq_lens.push_back(i);
    kv_cu_seq_lens.push_back(i * kv_len);
  }

  p.attention.host.q_seq_lens = q_cu_seq_lens;
  p.attention.host.kv_seq_lens = kv_cu_seq_lens;
  p.attention.host.q_cu_seq_lens = q_cu_seq_lens;

  p.attention.device.q_seq_lens = torch::tensor(q_cu_seq_lens, iopt);
  p.attention.device.kv_seq_lens = torch::tensor(kv_cu_seq_lens, iopt);
  p.attention.device.q_cu_seq_lens = p.attention.device.q_seq_lens;

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

ModelInputParams MakePrefillParams(const torch::Device& device,
                                   int32_t num_tokens) {
  CHECK_GT(num_tokens, 0);

  ModelInputParams p;

  p.meta.batch_forward_type = BatchForwardType::PREFILL;
  p.meta.num_sequences = 1;
  p.meta.kv_max_seq_len = num_tokens;
  p.meta.q_max_seq_len = num_tokens;
  p.enable_graph = false;

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  p.attention.host.q_seq_lens = {0, num_tokens};
  p.attention.host.kv_seq_lens = {0, num_tokens};
  p.attention.host.q_cu_seq_lens = {0, num_tokens};

  p.attention.device.q_seq_lens = torch::tensor({0, num_tokens}, iopt);
  p.attention.device.kv_seq_lens = torch::tensor({0, num_tokens}, iopt);
  p.attention.device.q_cu_seq_lens = p.attention.device.q_seq_lens;

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

TEST(DcuGraphExecutorTest, DecodeCaptureAndReplay) {
  if (!IsDcuAvailable()) {
    GTEST_SKIP() << "DCU/HIP is not available at runtime.";
  }

  const bool old_enable_graph = FLAGS_enable_graph;
  const bool old_decode_no_padding = FLAGS_enable_graph_mode_decode_no_padding;

  FLAGS_enable_graph = true;
  FLAGS_enable_graph_mode_decode_no_padding = true;

  const torch::Device device = InitXllmDcuDeviceForTest(0);

  ModelArgs args = MakeFakeModelArgs(64);
  runtime::Options options = MakeRuntimeOptions(8);

  auto model = std::make_unique<FakeSimpleCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::dcu::DcuGraphExecutorImpl>(
      model.get(), args, device, options);

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  auto tokens = torch::tensor({1, 2, 3}, iopt);
  auto positions = torch::tensor({0, 1, 2}, iopt);
  auto params = MakeDecodeParams(device, 3);

  auto kv = MakeKvCaches(device,
                         /*num_pages=*/16,
                         /*page_size=*/1,
                         /*num_kv_heads=*/1,
                         /*head_dim=*/64);

  auto eager_out =
      model->forward(tokens, positions, kv, params).hidden_states.clone();
  DcuSynchronize();

  auto out1 =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  DcuSynchronize();

  auto out2 =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  DcuSynchronize();

  EXPECT_TRUE(AllCloseBf16(out1, eager_out))
      << "DCU graph capture output should match eager output";

  EXPECT_TRUE(AllCloseBf16(out2, eager_out))
      << "DCU graph replay output should match eager output";

  EXPECT_TRUE(AllCloseBf16(out1, out2))
      << "DCU graph capture and replay outputs should match";

  FLAGS_enable_graph = old_enable_graph;
  FLAGS_enable_graph_mode_decode_no_padding = old_decode_no_padding;
}

TEST(DcuGraphExecutorTest, PrefillFallsBackToEager) {
  if (!IsDcuAvailable()) {
    GTEST_SKIP() << "DCU/HIP is not available at runtime.";
  }

  const bool old_enable_graph = FLAGS_enable_graph;

  FLAGS_enable_graph = true;

  const torch::Device device = InitXllmDcuDeviceForTest(0);

  ModelArgs args = MakeFakeModelArgs(64);
  runtime::Options options = MakeRuntimeOptions(8);

  auto model = std::make_unique<FakeSimpleCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::dcu::DcuGraphExecutorImpl>(
      model.get(), args, device, options);

  constexpr int32_t kNumTokens = 7;

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  auto tokens = torch::arange(1, kNumTokens + 1, iopt);
  auto positions = torch::arange(0, kNumTokens, iopt);
  auto params = MakePrefillParams(device, kNumTokens);

  auto kv = MakeKvCaches(device,
                         /*num_pages=*/16,
                         /*page_size=*/1,
                         /*num_kv_heads=*/1,
                         /*head_dim=*/64);

  auto eager_out =
      model->forward(tokens, positions, kv, params).hidden_states.clone();
  DcuSynchronize();

  auto out =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  DcuSynchronize();

  EXPECT_EQ(out.size(0), kNumTokens);
  EXPECT_EQ(out.size(1), args.hidden_size());

  EXPECT_TRUE(torch::isfinite(out.to(torch::kFloat32)).all().item<bool>());
  EXPECT_TRUE(AllCloseBf16(out, eager_out))
      << "DCU prefill path should fallback to eager and match eager output";

  FLAGS_enable_graph = old_enable_graph;
}

}  // namespace
}  // namespace xllm
