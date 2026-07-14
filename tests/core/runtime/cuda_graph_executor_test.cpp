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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/cuda/attention.h"
#include "core/layers/cuda/flashinfer_workspace.h"
#include "core/platform/device.h"
#include "core/runtime/cuda_graph_executor_impl.h"
#include "core/runtime/options.h"
#include "layers/common/attention_metadata_builder.h"

namespace xllm {
namespace {

const KVCache& first_full_attention_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& kv_cache : kv_caches) {
    if (!kv_cache.empty()) {
      const auto& k_cache = kv_cache.get_k_cache();
      const auto& v_cache = kv_cache.get_v_cache();
      if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
          v_cache.numel() > 0) {
        return kv_cache;
      }
    }
  }
  LOG(FATAL) << "No full-attention KV cache found";
  std::abort();
}

std::vector<KVCache> MakeSingleKvCaches(torch::Tensor k_cache,
                                        torch::Tensor v_cache) {
  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(KVCacheTensors{k_cache, v_cache});
  return kv_caches;
}

class ScopedConfigSnapshot final {
 public:
  ScopedConfigSnapshot()
      : execution_config_(ExecutionConfig::get_instance()),
        kv_cache_config_(KVCacheConfig::get_instance()),
        model_config_(ModelConfig::get_instance()),
        scheduler_config_(SchedulerConfig::get_instance()) {}

  ~ScopedConfigSnapshot() {
    ExecutionConfig::get_instance() = execution_config_;
    KVCacheConfig::get_instance() = kv_cache_config_;
    ModelConfig::get_instance() = model_config_;
    SchedulerConfig::get_instance() = scheduler_config_;
  }

 private:
  ExecutionConfig execution_config_;
  KVCacheConfig kv_cache_config_;
  ModelConfig model_config_;
  SchedulerConfig scheduler_config_;
};

class CudaGraphExecutorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("cuda_graph_executor_test");
    google::SetStderrLogging(google::INFO);

    if (torch::cuda::is_available()) {
      xllm::Device xllm_device(0);
      xllm_device.set_device();
    }

    // Keep the test minimal and deterministic.
    KVCacheConfig::get_instance().block_size(1);
    SchedulerConfig::get_instance().max_tokens_per_batch(128);
    ExecutionConfig::get_instance().enable_graph_mode_decode_no_padding(true);

    // Seed all RNGs once per test environment.
    torch::manual_seed(0);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new CudaGraphExecutorTestEnvironment);

torch::Device InitXllmCudaDeviceForTest(int32_t device_index = 0) {
  xllm::Device xllm_device(device_index);
  return xllm_device.unwrap();
}

// A minimal CausalLM whose forward contains a FlashInfer batch-decode attention
// call. This model is used to compare eager vs CUDA-graph execution under the
// same inputs.
class FakeAttnCausalLM final : public CausalLM {
 public:
  FakeAttnCausalLM(const ModelArgs& args, const torch::Device& device)
      : args_(args),
        device_(device),
        options_(
            torch::TensorOptions().dtype(torch::kBFloat16).device(device)) {
    const int64_t vocab_size = std::max<int64_t>(args_.vocab_size(), 1024);
    embedding_ =
        register_module("embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(
                            vocab_size, args_.hidden_size())));
    const int64_t q_hidden_size = args_.hidden_size();
    const int64_t n_kv_heads = args_.n_kv_heads().value_or(args_.n_heads());
    const int64_t kv_hidden_size = n_kv_heads * args_.head_dim();

    q_proj_ = register_module("q_proj",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  q_hidden_size, q_hidden_size)));
    k_proj_ = register_module("k_proj",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  q_hidden_size, kv_hidden_size)));
    v_proj_ = register_module("v_proj",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  q_hidden_size, kv_hidden_size)));

    // Move modules to target device before initializing weights.
    this->to(device_);

    // Initialize projections with random bf16 weights/bias.
    q_proj_->to(torch::kBFloat16);
    k_proj_->to(torch::kBFloat16);
    v_proj_->to(torch::kBFloat16);
    q_proj_->weight.data().normal_();
    k_proj_->weight.data().normal_();
    v_proj_->weight.data().normal_();
    if (q_proj_->bias.defined()) {
      q_proj_->bias.data().normal_();
    }
    if (k_proj_->bias.defined()) {
      k_proj_->bias.data().normal_();
    }
    if (v_proj_->bias.defined()) {
      v_proj_->bias.data().normal_();
    }

    const int n_heads = args_.n_heads();
    const int head_dim = args_.head_dim();
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_ = std::make_unique<layer::AttentionImpl>(
        n_heads,
        head_dim,
        scale,
        /*num_kv_heads=*/args_.n_kv_heads().value_or(n_heads),
        /*sliding_window=*/-1);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    (void)positions;
    CHECK(!kv_caches.empty());

    // Use the executor-provided metadata when available (CUDA graph mode).
    layer::AttentionMetadata attn_meta;
    if (params.attn_metadata) {
      attn_meta = *params.attn_metadata;
    } else {
      attn_meta =
          layer::AttentionMetadataBuilder::build(params, /*enable_mla=*/false);
    }
    CHECK(attn_meta.plan_info) << "attn_meta.plan_info must be set";
    attn_meta.plan_info->layer_id = 0;

    // Query/Key/Value are identical embeddings. For kv_len=1, attention output
    // equals value exactly.
    auto token_ids = tokens.to(torch::kInt64);
    auto x = embedding_->forward(token_ids).to(torch::kBFloat16);
    torch::Tensor q = q_proj_->forward(x);
    torch::Tensor k = k_proj_->forward(x);
    torch::Tensor v = v_proj_->forward(x);

    auto& full_kv_cache =
        const_cast<KVCache&>(first_full_attention_cache(kv_caches));
    auto [out, out_lse] = attn_->forward(attn_meta, q, k, v, full_kv_cache);
    (void)out_lse;
    return ModelOutput(out);
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
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

 private:
  ModelArgs args_;
  torch::Device device_;
  torch::TensorOptions options_;
  torch::nn::Embedding embedding_{nullptr};
  torch::nn::Linear q_proj_{nullptr};
  torch::nn::Linear k_proj_{nullptr};
  torch::nn::Linear v_proj_{nullptr};
  std::unique_ptr<layer::AttentionImpl> attn_;
};

ModelInputParams MakeDecodeParams(const torch::Device& device) {
  ModelInputParams p;
  p.meta.batch_forward_type = BatchForwardType::DECODE;
  p.meta.num_sequences = 1;
  p.meta.kv_max_seq_len = 4;
  p.meta.q_max_seq_len = 1;
  p.enable_graph = false;  // executor will set metadata->enable_cuda_graph

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  // cumulative lengths (cu_seq_lens)
  p.attention.device.q_seq_lens = torch::tensor({0, 1}, iopt);
  p.attention.device.kv_seq_lens = torch::tensor({0, 4}, iopt);
  p.attention.device.q_cu_seq_lens = p.attention.device.q_seq_lens;

  // slot mapping for the single token -> last slot in the 4-length kv cache
  p.attention.device.new_cache_slots = torch::tensor({3}, iopt);
  // block table is required by AttentionMetadataBuilder for decode path
  p.attention.device.block_tables = torch::tensor({{0, 1, 2, 3}}, iopt);

  // FlashInfer paged-kv metadata: one page (block) per sequence.
  p.attention.device.paged_kv_indptr = torch::tensor({0, 4}, iopt);
  p.attention.device.paged_kv_indices = torch::tensor({0, 1, 2, 3}, iopt);
  p.attention.device.paged_kv_last_page_len = torch::tensor({1}, iopt);

  return p;
}

ModelInputParams MakeBatchDecodeParams(const torch::Device& device,
                                       int32_t num_sequences) {
  CHECK_GT(num_sequences, 0);

  ModelInputParams p;
  p.meta.batch_forward_type = BatchForwardType::DECODE;
  p.meta.num_sequences = num_sequences;
  p.meta.kv_max_seq_len = 1;
  p.meta.q_max_seq_len = 1;
  p.enable_graph = false;

  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor cumulative_lens = torch::arange(0, num_sequences + 1, iopt);
  p.attention.device.q_seq_lens = cumulative_lens;
  p.attention.device.kv_seq_lens = cumulative_lens.clone();
  p.attention.device.q_cu_seq_lens = p.attention.device.q_seq_lens;
  p.attention.device.new_cache_slots = torch::arange(0, num_sequences, iopt);
  p.attention.device.block_tables =
      torch::arange(0, num_sequences, iopt).unsqueeze(/*dim=*/1);
  p.attention.device.paged_kv_indptr = cumulative_lens.clone();
  p.attention.device.paged_kv_indices = torch::arange(0, num_sequences, iopt);
  p.attention.device.paged_kv_last_page_len =
      torch::ones({num_sequences}, iopt);
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
  p.enable_graph = false;  // executor will set metadata->enable_cuda_graph

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  // cumulative lengths (cu_seq_lens)
  p.attention.device.q_seq_lens = torch::tensor({0, num_tokens}, iopt);
  p.attention.device.kv_seq_lens = torch::tensor({0, num_tokens}, iopt);
  p.attention.device.q_cu_seq_lens = p.attention.device.q_seq_lens;

  // prefill writes all tokens into kv-cache slots [0, num_tokens)
  p.attention.device.new_cache_slots = torch::arange(0, num_tokens, iopt);
  p.attention.device.block_tables =
      torch::arange(0, num_tokens, iopt).unsqueeze(0);

  // FlashInfer paged-kv metadata: one page (block) per token since block_size=1
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
  return MakeSingleKvCaches(k_cache, v_cache);
}

std::vector<KVCache> MakeHybridKvCaches(const torch::Device& device,
                                        int64_t num_pages,
                                        int64_t page_size,
                                        int64_t num_kv_heads,
                                        int64_t head_dim) {
  std::vector<KVCache> full_kv_caches =
      MakeKvCaches(device, num_pages, page_size, num_kv_heads, head_dim);
  KVCache full_kv = std::move(full_kv_caches.front());
  auto linear_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto conv_cache =
      torch::zeros({8, num_kv_heads * page_size * head_dim, 3}, linear_options);
  auto ssm_cache =
      torch::zeros({8, num_kv_heads, head_dim, head_dim}, linear_options);
  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(LinearAttentionKVCacheTensors{conv_cache, ssm_cache});
  kv_caches.emplace_back(
      KVCacheTensors{full_kv.get_k_cache(), full_kv.get_v_cache()});
  return kv_caches;
}

ModelArgs make_test_model_args() {
  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.max_position_embeddings(32);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(2);
  args.head_dim(128);
  args.n_kv_heads(1);
  return args;
}

runtime::Options make_test_runtime_options(int64_t max_seqs_per_batch) {
  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(max_seqs_per_batch);
  return options;
}

ModelInputParams make_multi_sequence_decode_params(
    const torch::Device& device) {
  ModelInputParams p;
  p.meta.batch_forward_type = BatchForwardType::DECODE;
  p.meta.num_sequences = 2;
  p.meta.kv_max_seq_len = 9;
  p.meta.q_max_seq_len = 1;
  p.enable_graph = false;

  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  p.attention.device.q_seq_lens = torch::tensor({0, 1, 2}, iopt);
  p.attention.device.kv_seq_lens = torch::tensor({0, 4, 9}, iopt);
  p.attention.device.q_cu_seq_lens = p.attention.device.q_seq_lens;
  p.attention.device.new_cache_slots = torch::tensor({5, 7}, iopt);
  p.attention.device.block_tables =
      torch::tensor({{0, 1, 2, 3}, {4, 5, 6, 7}}, iopt);
  p.attention.device.paged_kv_indptr = torch::tensor({0, 1, 3}, iopt);
  p.attention.device.paged_kv_indices = torch::tensor({2, 4, 6}, iopt);
  p.attention.device.paged_kv_last_page_len = torch::tensor({1, 2}, iopt);
  return p;
}

TEST(CudaGraphExecutorTest, DecodeMetadataFastPathUpdatesPersistentBuffers) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);
  ModelArgs args = make_test_model_args();
  runtime::Options options =
      make_test_runtime_options(/*max_seqs_per_batch=*/4);
  runtime::cuda::CudaGraphPersistentParam persistent(args, device, options);

  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor tokens = torch::tensor({10, 11}, iopt);
  torch::Tensor positions = torch::tensor({20, 21}, iopt);
  ModelInputParams params = make_multi_sequence_decode_params(device);
  std::vector<KVCache> kv = MakeKvCaches(device,
                                         /*num_pages=*/16,
                                         /*page_size=*/1,
                                         /*num_kv_heads=*/1,
                                         /*head_dim=*/128);

  std::optional<ModelInputParams> updated =
      persistent.update(tokens,
                        kv[0].get_k_cache(),
                        kv[0].get_v_cache(),
                        positions,
                        params,
                        /*padded_num_tokens=*/4,
                        /*return_capture_params=*/true);

  ASSERT_TRUE(updated.has_value());
  ASSERT_TRUE(updated->attn_metadata);

  EXPECT_TRUE(
      torch::equal(persistent.persistent_tokens(/*actual_tokens=*/4).cpu(),
                   torch::tensor({10, 11, 0, 0}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(persistent.persistent_positions(/*actual_tokens=*/4).cpu(),
                   torch::tensor({20, 21, 0, 0}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(
      persistent.persistent_new_cache_slots(/*actual_tokens=*/4).cpu(),
      torch::tensor({5, 7, -1, -1}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(
      persistent.kv_seq_lens(/*actual_batch_size=*/5).cpu(),
      torch::tensor({0, 4, 9, 9, 9}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(
      persistent.persistent_kv_seq_lens_delta(/*actual_batch_size=*/4).cpu(),
      torch::tensor({4, 5, 0, 0}, torch::dtype(torch::kInt32))));

  EXPECT_TRUE(torch::equal(
      updated->attn_metadata->q_cu_seq_lens.cpu(),
      torch::tensor({0, 1, 2, 3, 4}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(
      updated->attn_metadata->kv_cu_seq_lens.cpu(),
      torch::tensor({0, 4, 9, 9, 9}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(updated->attn_metadata->kv_seq_lens.cpu(),
                   torch::tensor({4, 5, 0, 0}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(updated->attn_metadata->slot_mapping.cpu(),
                   torch::tensor({5, 7, -1, -1}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(
      updated->attn_metadata->paged_kv_indptr.cpu(),
      torch::tensor({0, 1, 3, 3, 3}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(updated->attn_metadata->paged_kv_indices
                       .slice(/*dim=*/0,
                              /*start=*/0,
                              /*end=*/3)
                       .cpu(),
                   torch::tensor({2, 4, 6}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(updated->attn_metadata->paged_kv_last_page_len.cpu(),
                   torch::tensor({1, 2, 1, 1}, torch::dtype(torch::kInt32))));
  ASSERT_TRUE(updated->attn_metadata->qo_indptr.has_value());
  EXPECT_TRUE(torch::equal(
      updated->attn_metadata->qo_indptr.value().cpu(),
      torch::tensor({0, 1, 2, 3, 4}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(updated->attn_metadata->block_table.cpu(),
                           params.attention.device.block_tables.cpu()));
}

TEST(CudaGraphExecutorTest, DecodeMetadataFastPathUpdatesLinearStateIndices) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);
  ModelArgs args = make_test_model_args();
  runtime::Options options =
      make_test_runtime_options(/*max_seqs_per_batch=*/4);
  runtime::cuda::CudaGraphPersistentParam persistent(args, device, options);

  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor tokens = torch::tensor({10, 11}, iopt);
  torch::Tensor positions = torch::tensor({20, 21}, iopt);
  ModelInputParams params = make_multi_sequence_decode_params(device);
  params.embedding.linear_state_ids = {8, 6};
  params.embedding.linear_state_indices = torch::tensor({8, 6}, iopt);
  std::vector<KVCache> kv = MakeKvCaches(device,
                                         /*num_pages=*/16,
                                         /*page_size=*/1,
                                         /*num_kv_heads=*/1,
                                         /*head_dim=*/128);

  std::optional<ModelInputParams> updated =
      persistent.update(tokens,
                        kv[0].get_k_cache(),
                        kv[0].get_v_cache(),
                        positions,
                        params,
                        /*padded_num_tokens=*/4,
                        /*return_capture_params=*/true);

  ASSERT_TRUE(updated.has_value());
  EXPECT_TRUE(torch::equal(
      persistent.persistent_linear_state_indices(/*actual_batch_size=*/2).cpu(),
      torch::tensor({8, 6}, torch::dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(updated->embedding.linear_state_indices.cpu(),
                           torch::tensor({8, 6}, torch::dtype(torch::kInt32))));
}

TEST(CudaGraphExecutorTest, DecodeMetadataFastPathFallbackMatchesLegacyPath) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);
  ModelArgs args = make_test_model_args();
  runtime::Options options =
      make_test_runtime_options(/*max_seqs_per_batch=*/4);
  runtime::cuda::CudaGraphPersistentParam fast_path_persistent(
      args, device, options);
  runtime::cuda::CudaGraphPersistentParam fallback_persistent(
      args, device, options);

  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor tokens = torch::tensor({10, 11, 12}, iopt);
  torch::Tensor positions = torch::tensor({20, 21, 22}, iopt);
  ModelInputParams fast_params = MakeBatchDecodeParams(device, 3);
  ModelInputParams fallback_params = MakeBatchDecodeParams(device, 3);
  torch::Tensor new_cache_slots_base =
      torch::tensor({0, 99, 1, 88, 2, 77}, iopt).view({3, 2});
  fallback_params.attention.device.new_cache_slots =
      new_cache_slots_base.select(1, 0);
  ASSERT_FALSE(
      fallback_params.attention.device.new_cache_slots.is_contiguous());

  std::vector<KVCache> kv = MakeKvCaches(device,
                                         /*num_pages=*/16,
                                         /*page_size=*/1,
                                         /*num_kv_heads=*/1,
                                         /*head_dim=*/128);

  std::optional<ModelInputParams> fast_updated =
      fast_path_persistent.update(tokens,
                                  kv[0].get_k_cache(),
                                  kv[0].get_v_cache(),
                                  positions,
                                  fast_params,
                                  /*padded_num_tokens=*/4,
                                  /*return_capture_params=*/true);
  std::optional<ModelInputParams> fallback_updated =
      fallback_persistent.update(tokens,
                                 kv[0].get_k_cache(),
                                 kv[0].get_v_cache(),
                                 positions,
                                 fallback_params,
                                 /*padded_num_tokens=*/4,
                                 /*return_capture_params=*/true);

  ASSERT_TRUE(fast_updated.has_value());
  ASSERT_TRUE(fallback_updated.has_value());
  ASSERT_TRUE(fast_updated->attn_metadata);
  ASSERT_TRUE(fallback_updated->attn_metadata);

  EXPECT_TRUE(torch::equal(
      fast_path_persistent.persistent_tokens(/*actual_tokens=*/4).cpu(),
      fallback_persistent.persistent_tokens(/*actual_tokens=*/4).cpu()));
  EXPECT_TRUE(torch::equal(
      fast_path_persistent.persistent_positions(/*actual_tokens=*/4).cpu(),
      fallback_persistent.persistent_positions(/*actual_tokens=*/4).cpu()));
  EXPECT_TRUE(torch::equal(
      fast_path_persistent.persistent_new_cache_slots(/*actual_tokens=*/4)
          .cpu(),
      fallback_persistent.persistent_new_cache_slots(/*actual_tokens=*/4)
          .cpu()));
  EXPECT_TRUE(torch::equal(
      fast_path_persistent.kv_seq_lens(/*actual_batch_size=*/5).cpu(),
      fallback_persistent.kv_seq_lens(/*actual_batch_size=*/5).cpu()));
  EXPECT_TRUE(torch::equal(
      fast_path_persistent.persistent_kv_seq_lens_delta(/*actual_batch_size=*/4)
          .cpu(),
      fallback_persistent.persistent_kv_seq_lens_delta(/*actual_batch_size=*/4)
          .cpu()));
  EXPECT_TRUE(torch::equal(fast_updated->attn_metadata->kv_seq_lens.cpu(),
                           fallback_updated->attn_metadata->kv_seq_lens.cpu()));
  EXPECT_TRUE(
      torch::equal(fast_updated->attn_metadata->q_cu_seq_lens.cpu(),
                   fallback_updated->attn_metadata->q_cu_seq_lens.cpu()));
  EXPECT_TRUE(
      torch::equal(fast_updated->attn_metadata->kv_cu_seq_lens.cpu(),
                   fallback_updated->attn_metadata->kv_cu_seq_lens.cpu()));
  EXPECT_TRUE(
      torch::equal(fast_updated->attn_metadata->slot_mapping.cpu(),
                   fallback_updated->attn_metadata->slot_mapping.cpu()));
  EXPECT_TRUE(
      torch::equal(fast_updated->attn_metadata->paged_kv_indptr.cpu(),
                   fallback_updated->attn_metadata->paged_kv_indptr.cpu()));
  EXPECT_TRUE(
      torch::equal(fast_updated->attn_metadata->paged_kv_indices.cpu(),
                   fallback_updated->attn_metadata->paged_kv_indices.cpu()));
  EXPECT_TRUE(torch::equal(
      fast_updated->attn_metadata->paged_kv_last_page_len.cpu(),
      fallback_updated->attn_metadata->paged_kv_last_page_len.cpu()));
  ASSERT_TRUE(fast_updated->attn_metadata->qo_indptr.has_value());
  ASSERT_TRUE(fallback_updated->attn_metadata->qo_indptr.has_value());
  EXPECT_TRUE(
      torch::equal(fast_updated->attn_metadata->qo_indptr.value().cpu(),
                   fallback_updated->attn_metadata->qo_indptr.value().cpu()));
}

TEST(CudaGraphExecutorTest, BatchDecodeCaptureAndReplay) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().enable_graph_vmm_pool(false);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.max_position_embeddings(16);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(2);
  args.head_dim(128);
  args.n_kv_heads(1);

  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(1);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  auto tokens = torch::tensor(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto positions = torch::tensor(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto params = MakeDecodeParams(device);

  // Eager baseline and CUDA graph runs share the same KVCache.
  auto kv = MakeKvCaches(device,
                         /*num_pages=*/4,
                         /*page_size=*/1,
                         /*num_kv_heads=*/1,
                         /*head_dim=*/128);
  auto eager_out =
      model->forward(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  // LOG(INFO) << "eager_out: " << eager_out;
  // Graph capture (first run) + replay (second run).
  auto out1 = graph_exec->run(tokens, positions, kv, params).hidden_states;
  torch::cuda::synchronize();
  // LOG(INFO) << "out1: " << out1;
  EXPECT_TRUE(torch::allclose(out1, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "graph capture output should match eager output";

  auto out2 = graph_exec->run(tokens, positions, kv, params).hidden_states;
  torch::cuda::synchronize();
  EXPECT_TRUE(torch::allclose(out2, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "graph replay output should match eager output";
}

TEST(CudaGraphExecutorTest, BatchDecodePaddedBucket100CaptureAndReplay) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .enable_graph_vmm_pool(false)
      .enable_graph_mode_decode_no_padding(false);
  SchedulerConfig::get_instance().max_tokens_per_batch(128);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args = make_test_model_args();
  args.max_position_embeddings(16);
  runtime::Options options =
      make_test_runtime_options(/*max_seqs_per_batch=*/100);

  runtime::cuda::CudaGraphPersistentParam persistent(args, device, options);
  EXPECT_EQ(persistent.q_seq_lens(/*actual_batch_size=*/0).size(0), 113);
  EXPECT_EQ(persistent.kv_seq_lens(/*actual_batch_size=*/0).size(0), 113);
  EXPECT_EQ(
      persistent.persistent_paged_kv_indptr(/*actual_batch_size=*/0).size(0),
      113);
  EXPECT_EQ(
      persistent.persistent_paged_kv_last_page_len(/*actual_batch_size=*/0)
          .size(0),
      112);
  EXPECT_EQ(
      persistent.persistent_decode_qo_indptr(/*actual_batch_size=*/0).size(0),
      113);
  EXPECT_EQ(persistent.persistent_block_tables(/*actual_batch_size=*/0).size(0),
            112);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  constexpr int32_t kActualBatchSize = 100;
  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor tokens = torch::arange(1, kActualBatchSize + 1, iopt);
  torch::Tensor positions = torch::zeros({kActualBatchSize}, iopt);
  ModelInputParams params = MakeBatchDecodeParams(device, kActualBatchSize);

  std::vector<KVCache> kv = MakeKvCaches(device,
                                         /*num_pages=*/kActualBatchSize,
                                         /*page_size=*/1,
                                         /*num_kv_heads=*/1,
                                         /*head_dim=*/128);
  std::vector<KVCache> eager_kv = MakeSingleKvCaches(
      kv[0].get_k_cache().clone(), kv[0].get_v_cache().clone());
  torch::Tensor eager_out =
      model->forward(tokens, positions, eager_kv, params).hidden_states.clone();
  torch::cuda::synchronize();

  torch::Tensor first_out =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  EXPECT_EQ(first_out.size(0), kActualBatchSize);
  EXPECT_TRUE(
      torch::allclose(first_out, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "100 -> 112 graph capture and first replay should match eager";

  torch::Tensor second_out =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  EXPECT_TRUE(
      torch::allclose(second_out, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "100 -> 112 graph replay should match eager";
}

TEST(CudaGraphExecutorTest, BatchDecodeNoPaddingBucket100CaptureAndReplay) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .enable_graph_vmm_pool(false)
      .enable_graph_mode_decode_no_padding(true);
  SchedulerConfig::get_instance().max_tokens_per_batch(128);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args = make_test_model_args();
  args.max_position_embeddings(16);
  runtime::Options options =
      make_test_runtime_options(/*max_seqs_per_batch=*/100);

  runtime::cuda::CudaGraphPersistentParam persistent(args, device, options);
  EXPECT_EQ(persistent.q_seq_lens(/*actual_batch_size=*/0).size(0), 101);
  EXPECT_EQ(persistent.kv_seq_lens(/*actual_batch_size=*/0).size(0), 101);
  EXPECT_EQ(
      persistent.persistent_paged_kv_indptr(/*actual_batch_size=*/0).size(0),
      101);
  EXPECT_EQ(
      persistent.persistent_paged_kv_last_page_len(/*actual_batch_size=*/0)
          .size(0),
      100);
  EXPECT_EQ(
      persistent.persistent_decode_qo_indptr(/*actual_batch_size=*/0).size(0),
      101);
  EXPECT_EQ(persistent.persistent_block_tables(/*actual_batch_size=*/0).size(0),
            100);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  constexpr int32_t kActualBatchSize = 100;
  torch::TensorOptions iopt =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor tokens = torch::arange(1, kActualBatchSize + 1, iopt);
  torch::Tensor positions = torch::zeros({kActualBatchSize}, iopt);
  ModelInputParams params = MakeBatchDecodeParams(device, kActualBatchSize);

  std::vector<KVCache> kv = MakeKvCaches(device,
                                         /*num_pages=*/kActualBatchSize,
                                         /*page_size=*/1,
                                         /*num_kv_heads=*/1,
                                         /*head_dim=*/128);
  std::vector<KVCache> eager_kv = MakeSingleKvCaches(
      kv[0].get_k_cache().clone(), kv[0].get_v_cache().clone());
  torch::Tensor eager_out =
      model->forward(tokens, positions, eager_kv, params).hidden_states.clone();
  torch::cuda::synchronize();

  torch::Tensor first_out =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  EXPECT_EQ(first_out.size(0), kActualBatchSize);
  EXPECT_TRUE(
      torch::allclose(first_out, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "100-token no-padding graph capture and first replay should match "
         "eager";

  torch::Tensor second_out =
      graph_exec->run(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  EXPECT_TRUE(
      torch::allclose(second_out, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "100-token no-padding graph replay should match eager";
}

TEST(CudaGraphExecutorTest,
     BatchDecodeCaptureAndReplayWithLinearOnlyLayerZero) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance().enable_graph_vmm_pool(false);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.max_position_embeddings(16);
  args.vocab_size(2048);
  args.n_layers(2);
  args.n_heads(2);
  args.head_dim(128);
  args.n_kv_heads(1);
  args.layer_types({"linear_attention", "full_attention"});

  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(1);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  auto tokens = torch::tensor(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto positions = torch::tensor(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto params = MakeDecodeParams(device);
  auto kv = MakeHybridKvCaches(device,
                               /*num_pages=*/4,
                               /*page_size=*/1,
                               /*num_kv_heads=*/1,
                               /*head_dim=*/128);

  auto eager_out =
      model->forward(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  auto out1 = graph_exec->run(tokens, positions, kv, params).hidden_states;
  torch::cuda::synchronize();
  auto out2 = graph_exec->run(tokens, positions, kv, params).hidden_states;
  torch::cuda::synchronize();

  EXPECT_TRUE(torch::allclose(out1, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3));
  EXPECT_TRUE(torch::allclose(out2, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3));
}

TEST(CudaGraphExecutorTest, PrefillPiecewiseCaptureAndReplay) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .enable_graph_mode_decode_no_padding(true)
      .enable_prefill_piecewise_graph(true)
      .enable_graph_vmm_pool(false)
      .max_tokens_for_graph_mode(128);
  SchedulerConfig::get_instance().max_tokens_per_batch(128);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.max_position_embeddings(256);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(2);
  args.head_dim(128);
  args.n_kv_heads(1);

  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(1);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto graph_exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  constexpr int32_t kNumTokens = 113;
  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto tokens = torch::arange(1, kNumTokens + 1, iopt);
  auto positions = torch::arange(0, kNumTokens, iopt);
  auto params = MakePrefillParams(device, kNumTokens);

  auto kv = MakeKvCaches(device,
                         /*num_pages=*/128,
                         /*page_size=*/1,
                         /*num_kv_heads=*/1,
                         /*head_dim=*/128);

  auto kv_eager = MakeSingleKvCaches(kv[0].get_k_cache().clone(),
                                     kv[0].get_v_cache().clone());
  auto kv_graph_first = MakeSingleKvCaches(kv[0].get_k_cache().clone(),
                                           kv[0].get_v_cache().clone());
  auto kv_graph_second = MakeSingleKvCaches(kv[0].get_k_cache().clone(),
                                            kv[0].get_v_cache().clone());

  auto eager_out =
      model->forward(tokens, positions, kv_eager, params).hidden_states.clone();
  torch::cuda::synchronize();

  auto out1 =
      graph_exec->run(tokens, positions, kv_graph_first, params).hidden_states;
  out1 = out1.clone();
  torch::cuda::synchronize();

  auto out2 =
      graph_exec->run(tokens, positions, kv_graph_second, params).hidden_states;
  out2 = out2.clone();
  torch::cuda::synchronize();

  EXPECT_EQ(out1.size(0), kNumTokens);
  EXPECT_EQ(out1.size(1), args.hidden_size());
  EXPECT_EQ(out2.size(0), kNumTokens);
  EXPECT_EQ(out2.size(1), args.hidden_size());

  EXPECT_TRUE(torch::isfinite(eager_out).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(out1).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(out2).all().item<bool>());

  EXPECT_TRUE(torch::allclose(out1, eager_out, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "prefill first run (capture + replay) should match eager";
  EXPECT_TRUE(torch::allclose(out2, eager_out, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "prefill second run (replay) should match eager";
  EXPECT_TRUE(torch::allclose(out1, out2, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "prefill first and second runs should be consistent";
}

TEST(CudaGraphExecutorTest, CompareMqa2v1AndMqa8v1) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .enable_prefill_piecewise_graph(true)
      .enable_graph_vmm_pool(false)
      .max_tokens_for_graph_mode(128);
  SchedulerConfig::get_instance().max_tokens_per_batch(128);
  ModelConfig::get_instance().flashinfer_workspace_buffer_size(256 * 1024 *
                                                               1024);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  constexpr int32_t kNumTokens = 113;
  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto tokens = torch::arange(1, kNumTokens + 1, iopt);
  auto positions = torch::arange(0, kNumTokens, iopt);

  struct PrefillRunOutputs {
    torch::Tensor eager_out;
    torch::Tensor graph_out;
  };

  auto run_mqa_prefill = [&](int64_t n_heads, int64_t n_kv_heads) {
    ModelArgs args;
    args.model_type("fake_attn");
    args.dtype("bfloat16");
    args.hidden_size(n_heads * 128);
    args.max_position_embeddings(256);
    args.vocab_size(2048);
    args.n_layers(1);
    args.n_heads(n_heads);
    args.head_dim(128);
    args.n_kv_heads(n_kv_heads);

    runtime::Options options;
    options.block_size(1);
    options.max_seqs_per_batch(1);

    auto model = std::make_unique<FakeAttnCausalLM>(args, device);
    auto graph_exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
        model.get(), args, device, options);

    auto params = MakePrefillParams(device, kNumTokens);
    auto kv = MakeKvCaches(device,
                           /*num_pages=*/128,
                           /*page_size=*/1,
                           /*num_kv_heads=*/n_kv_heads,
                           /*head_dim=*/128);
    auto kv_eager = MakeSingleKvCaches(kv[0].get_k_cache().clone(),
                                       kv[0].get_v_cache().clone());
    auto kv_graph = MakeSingleKvCaches(kv[0].get_k_cache().clone(),
                                       kv[0].get_v_cache().clone());

    auto eager_out = model->forward(tokens, positions, kv_eager, params)
                         .hidden_states.clone();
    torch::cuda::synchronize();

    auto graph_out =
        graph_exec->run(tokens, positions, kv_graph, params).hidden_states;
    graph_out = graph_out.clone();
    torch::cuda::synchronize();

    return PrefillRunOutputs{std::move(eager_out), std::move(graph_out)};
  };

  auto out_mqa_2v1 = run_mqa_prefill(/*n_heads=*/2, /*n_kv_heads=*/1);
  auto out_mqa_8v1 = run_mqa_prefill(/*n_heads=*/8, /*n_kv_heads=*/1);

  EXPECT_EQ(out_mqa_2v1.eager_out.size(0), kNumTokens);
  EXPECT_EQ(out_mqa_2v1.graph_out.size(0), kNumTokens);
  EXPECT_EQ(out_mqa_8v1.eager_out.size(0), kNumTokens);
  EXPECT_EQ(out_mqa_8v1.graph_out.size(0), kNumTokens);
  EXPECT_EQ(out_mqa_2v1.eager_out.size(1), 256);
  EXPECT_EQ(out_mqa_2v1.graph_out.size(1), 256);
  EXPECT_EQ(out_mqa_8v1.eager_out.size(1), 1024);
  EXPECT_EQ(out_mqa_8v1.graph_out.size(1), 1024);

  EXPECT_TRUE(torch::isfinite(out_mqa_2v1.eager_out).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(out_mqa_2v1.graph_out).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(out_mqa_8v1.eager_out).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(out_mqa_8v1.graph_out).all().item<bool>());

  EXPECT_TRUE(torch::allclose(out_mqa_2v1.graph_out,
                              out_mqa_2v1.eager_out,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "MQA(2/1) graph output should match eager output";
  EXPECT_TRUE(torch::allclose(out_mqa_8v1.graph_out,
                              out_mqa_8v1.eager_out,
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "MQA(8/1) graph output should match eager output";

  const auto out_mqa_8v1_first_256 = out_mqa_8v1.graph_out.slice(/*dim=*/1,
                                                                 /*start=*/0,
                                                                 /*end=*/256);
  const float mean_abs_delta = (out_mqa_2v1.graph_out - out_mqa_8v1_first_256)
                                   .abs()
                                   .mean()
                                   .item<float>();
  EXPECT_TRUE(std::isfinite(mean_abs_delta));
}

TEST(CudaGraphExecutorTest, GraphVmmPoolMemoryReuseAcrossMultiShape) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .enable_prefill_piecewise_graph(true)
      .max_tokens_for_graph_mode(256);
  SchedulerConfig::get_instance().max_tokens_per_batch(256);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.max_position_embeddings(512);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(2);
  args.head_dim(128);
  args.n_kv_heads(1);

  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(1);

  auto create_executor = [&]() {
    auto model = std::make_unique<FakeAttnCausalLM>(args, device);
    auto exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
        model.get(), args, device, options);
    return std::make_pair(std::move(model), std::move(exec));
  };

  auto run_prefill_capture_sweep = [&](bool enable_graph_vmm_pool) {
    ExecutionConfig::get_instance().enable_graph_vmm_pool(
        enable_graph_vmm_pool);
    auto [model, exec] = create_executor();

    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto all_tokens = torch::arange(1, 257, iopt);
    auto all_positions = torch::arange(0, 256, iopt);
    auto kv = MakeKvCaches(device,
                           /*num_pages=*/256,
                           /*page_size=*/1,
                           /*num_kv_heads=*/1,
                           /*head_dim=*/128);

    std::vector<size_t> memory_usage_bytes;
    memory_usage_bytes.reserve(9);
    for (int32_t num_tokens = 256; num_tokens >= 128; num_tokens -= 16) {
      auto tokens =
          all_tokens.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens);
      auto positions =
          all_positions.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens);
      auto params = MakePrefillParams(device, num_tokens);
      auto out = exec->run(tokens, positions, kv, params).hidden_states;
      (void)out;
      torch::cuda::synchronize();
      memory_usage_bytes.push_back(exec->get_graph_memory_usage_bytes());
    }
    return memory_usage_bytes;
  };

  const auto vmm_usage =
      run_prefill_capture_sweep(/*enable_graph_vmm_pool=*/true);
  ASSERT_FALSE(vmm_usage.empty());
  for (size_t i = 1; i < vmm_usage.size(); ++i) {
    EXPECT_EQ(vmm_usage[i], vmm_usage.front())
        << "With enable_graph_vmm_pool=true, memory should remain stable "
        << "during 256->128 multi-shape capture sweep, step=" << i
        << ", baseline=" << vmm_usage.front() << ", current=" << vmm_usage[i];
  }

  const auto no_vmm_usage =
      run_prefill_capture_sweep(/*enable_graph_vmm_pool=*/false);
  ASSERT_FALSE(no_vmm_usage.empty());

  size_t grow_steps = 0;
  for (size_t i = 1; i < no_vmm_usage.size(); ++i) {
    if (no_vmm_usage[i] > no_vmm_usage[i - 1]) {
      ++grow_steps;
    }
  }

  EXPECT_GT(no_vmm_usage.back(), no_vmm_usage.front())
      << "With enable_graph_vmm_pool=false, memory should grow during "
      << "256->128 multi-shape capture sweep, first=" << no_vmm_usage.front()
      << ", last=" << no_vmm_usage.back();
  EXPECT_GT(grow_steps, 0U)
      << "With enable_graph_vmm_pool=false, expected at least one increasing "
      << "step during 256->128 sweep.";
}

TEST(CudaGraphExecutorTest, GraphVmmPoolEnabledPrefillCorrectness) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  ScopedConfigSnapshot config_snapshot;
  ExecutionConfig::get_instance()
      .enable_graph(true)
      .enable_prefill_piecewise_graph(true)
      .max_tokens_for_graph_mode(256)
      .enable_graph_vmm_pool(true);
  SchedulerConfig::get_instance().max_tokens_per_batch(256);

  const torch::Device device = InitXllmCudaDeviceForTest(/*device_index=*/0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(256);
  args.max_position_embeddings(512);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(2);
  args.head_dim(128);
  args.n_kv_heads(1);

  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(1);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto exec = std::make_unique<runtime::cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto all_tokens = torch::arange(1, 257, iopt);
  auto all_positions = torch::arange(0, 256, iopt);

  for (int32_t num_tokens = 128; num_tokens <= 256; num_tokens += 16) {
    auto tokens = all_tokens.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens);
    auto positions =
        all_positions.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens);
    auto params = MakePrefillParams(device, num_tokens);

    auto kv_base = MakeKvCaches(device,
                                /*num_pages=*/256,
                                /*page_size=*/1,
                                /*num_kv_heads=*/1,
                                /*head_dim=*/128);
    auto kv_eager = MakeSingleKvCaches(kv_base[0].get_k_cache().clone(),
                                       kv_base[0].get_v_cache().clone());
    auto kv_graph = MakeSingleKvCaches(kv_base[0].get_k_cache().clone(),
                                       kv_base[0].get_v_cache().clone());

    auto eager_out = model->forward(tokens, positions, kv_eager, params)
                         .hidden_states.clone();
    auto graph_out =
        exec->run(tokens, positions, kv_graph, params).hidden_states.clone();
    torch::cuda::synchronize();

    EXPECT_TRUE(torch::isfinite(graph_out).all().item<bool>())
        << "graph output contains non-finite values at num_tokens="
        << num_tokens;
    EXPECT_TRUE(torch::allclose(graph_out,
                                eager_out,
                                /*rtol=*/1e-2,
                                /*atol=*/1e-2))
        << "With enable_graph_vmm_pool=true, graph output should match eager "
        << "output at num_tokens=" << num_tokens;
  }
}

}  // namespace
}  // namespace xllm
