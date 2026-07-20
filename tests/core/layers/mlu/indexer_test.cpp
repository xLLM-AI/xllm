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

#include "layers/mlu/indexer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <sstream>

#include "core/framework/config/kv_cache_config.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/attention.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
class MockDeepseekScalingRotaryEmbedding
    : public DeepseekScalingRotaryEmbeddingImpl {
 public:
  MockDeepseekScalingRotaryEmbedding(int64_t rotary_dim,
                                     int64_t max_position_embeddings,
                                     int64_t rope_theta,
                                     bool interleaved,
                                     const torch::TensorOptions& options)
      : DeepseekScalingRotaryEmbeddingImpl(rotary_dim,
                                           rotary_dim,
                                           max_position_embeddings,
                                           max_position_embeddings,
                                           rope_theta,
                                           interleaved,
                                           /*scaling_factor=*/2.5,
                                           /*extrapolation_factor=*/1.,
                                           /*attn_factor=*/40,
                                           /*beta_fast=*/32,
                                           /*beta_slow=*/1,
                                           /*mscale=*/1.,
                                           /*mscale_all_dim=*/1.,
                                           options) {
    mock_rope_ = std::make_shared<RotaryEmbeddingImpl>(
        rotary_dim, max_position_embeddings, rope_theta, interleaved, options);
  }
  void forward(torch::Tensor& q,
               torch::Tensor& k,
               const torch::Tensor& positions,
               const torch::Tensor& cu_query_lens,
               int64_t max_query_len,
               bool is_prompt) {
    return mock_rope_->forward(
        q, k, positions, cu_query_lens, max_query_len, is_prompt);
  }

 private:
  std::shared_ptr<RotaryEmbeddingImpl> mock_rope_;
};

class IndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (Platform::device_count() < 1) {
      GTEST_SKIP() << "MLU device is required for indexer kernel tests.";
    }
    torch::Device torch_device(Platform::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(torch_device)
                   .requires_grad(false);
    int_option_ = options_.dtype(torch::kInt32);

    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
    KVCacheConfig::get_instance().block_size(1);
  }

  void TearDown() override {}

  torch::Tensor create_random_tensor(
      const std::vector<int64_t>& shape,
      float min_val = -1.0f,
      float max_val = 1.0f,
      std::optional<torch::ScalarType> dtype = std::nullopt) {
    auto opts = dtype.has_value() ? options_.dtype(dtype.value()) : options_;
    return torch::rand(shape, opts) * (max_val - min_val) + min_val;
  }

  std::unordered_map<std::string, torch::Tensor> create_random_weights(
      int64_t dim,
      int64_t index_n_heads,
      int64_t index_head_dim,
      int64_t q_lora_rank) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    weight_dict["wq_b.weight"] = create_random_tensor(
        {index_n_heads * index_head_dim, q_lora_rank}, -0.1f, 0.1f);
    weight_dict["wk.weight"] =
        create_random_tensor({index_head_dim, dim}, -0.1f, 0.1f);
    weight_dict["weights_proj.weight"] =
        create_random_tensor({index_n_heads, dim}, -0.1f, 0.1f);
    weight_dict["k_norm.weight"] =
        create_random_tensor({index_head_dim}, -0.5f, 0.5f, torch::kFloat32);
    weight_dict["k_norm.bias"] =
        create_random_tensor({index_head_dim}, -0.5f, 0.5f, torch::kFloat32);

    return weight_dict;
  }

  void populate_attention_metadata(AttentionMetadata& metadata,
                                   int64_t batch_size,
                                   int64_t max_query_len,
                                   int64_t max_seq_len,
                                   bool is_prefill,
                                   int64_t max_num_blocks_per_seq) {
    // q_cu_seq_lens
    metadata.q_cu_seq_lens = torch::arange(
        0, (batch_size + 1) * max_query_len, max_query_len, int_option_);

    // kv_cu_seq_lens
    metadata.kv_cu_seq_lens = torch::arange(
        0, (batch_size + 1) * max_query_len, max_query_len, int_option_);

    metadata.kv_seq_lens =
        torch::full({batch_size}, max_query_len, int_option_);

    metadata.block_table =
        torch::zeros({batch_size, max_num_blocks_per_seq}, int_option_);

    for (int64_t b = 0; b < batch_size; ++b) {
      auto seq = torch::arange(b * max_query_len + 1,
                               b * max_query_len + 1 + max_query_len,
                               int_option_);
      metadata.block_table[b].index_put_(
          {torch::indexing::Slice(0, max_query_len)}, seq);
    }

    // slot_mapping
    metadata.slot_mapping =
        torch::arange(1, batch_size * max_query_len + 1, int_option_);

    metadata.max_query_len = max_query_len;
    metadata.max_seq_len = max_seq_len;
    metadata.total_kv_len = batch_size * max_query_len;
    metadata.compute_dtype = "bfloat16";
    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = false;
  }

  void populate_chunked_attention_metadata(AttentionMetadata& metadata,
                                           int64_t batch_size,
                                           int64_t history_len,
                                           int64_t current_len,
                                           int64_t block_size,
                                           bool use_noncontiguous_blocks) {
    int64_t total_len = history_len + current_len;
    int64_t blocks_per_seq = (total_len + block_size - 1) / block_size;

    metadata.q_cu_seq_lens = torch::arange(
        0, (batch_size + 1) * current_len, current_len, int_option_);

    metadata.kv_cu_seq_lens =
        torch::arange(0, (batch_size + 1) * total_len, total_len, int_option_);

    metadata.kv_seq_lens = torch::full({batch_size}, total_len, int_option_);

    metadata.block_table =
        torch::zeros({batch_size, blocks_per_seq}, int_option_);

    std::vector<int32_t> slot_mapping;
    slot_mapping.reserve(batch_size * current_len);

    for (int64_t b = 0; b < batch_size; ++b) {
      std::vector<int32_t> block_ids;
      block_ids.reserve(blocks_per_seq);
      for (int64_t logical_block = 0; logical_block < blocks_per_seq;
           ++logical_block) {
        int64_t contiguous_block = b * blocks_per_seq + logical_block;
        int64_t physical_block = use_noncontiguous_blocks
                                     ? contiguous_block * 2 + 1
                                     : contiguous_block;
        block_ids.emplace_back(static_cast<int32_t>(physical_block));
      }
      metadata.block_table[b].copy_(torch::tensor(block_ids, int_option_));

      for (int64_t position = history_len; position < total_len; ++position) {
        int64_t logical_block = position / block_size;
        int64_t block_offset = position % block_size;
        int64_t slot =
            static_cast<int64_t>(block_ids[logical_block]) * block_size +
            block_offset;
        slot_mapping.emplace_back(static_cast<int32_t>(slot));
      }
    }
    metadata.slot_mapping = torch::tensor(slot_mapping, int_option_);

    metadata.max_query_len = current_len;
    metadata.max_seq_len = total_len;
    metadata.total_kv_len = batch_size * total_len;
    metadata.compute_dtype = "bfloat16";
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = true;
  }

  struct TestConfig {
    int64_t dim = 7168;
    int64_t index_n_heads = 64;
    int64_t index_head_dim = 128;
    int64_t qk_rope_head_dim = 64;
    int64_t index_topk = 2048;
    int64_t q_lora_rank = 1536;
    int64_t max_position_embeddings = 8192;
    int64_t rope_theta = 10000;
    bool rope_interleaved = true;
    int64_t head_kv = 1;
    int64_t block_size = 1;
    int64_t block_num = 10240;
  };

  struct TestInputs {
    torch::Tensor x;
    torch::Tensor q_norm;
    torch::Tensor positions;
    torch::Tensor k_cache;
    std::optional<torch::Tensor> k_cache_scale;
    std::unordered_map<std::string, torch::Tensor> weights;
    AttentionMetadata metadata;
  };

  TestInputs create_inputs(int64_t batch_size,
                           int64_t max_query_len,
                           bool is_prefill,
                           bool chunked_prefill = false,
                           int64_t history_len = 0,
                           bool use_default_rope = false,
                           bool quantized_cache = false,
                           int64_t cache_block_size = 1,
                           bool use_noncontiguous_blocks = false) {
    test_config_ = TestConfig();
    test_config_.block_size = cache_block_size;
    KVCacheConfig::get_instance().block_size(cache_block_size);
    if (use_default_rope) {
      rotary_emb_ = std::make_shared<RotaryEmbeddingImpl>(
          test_config_.qk_rope_head_dim,
          test_config_.max_position_embeddings,
          test_config_.rope_theta,
          test_config_.rope_interleaved,
          options_);
    } else {
      rotary_emb_ = std::make_shared<MockDeepseekScalingRotaryEmbedding>(
          test_config_.qk_rope_head_dim,
          test_config_.max_position_embeddings,
          test_config_.rope_theta,
          test_config_.rope_interleaved,
          options_);
    }

    TestInputs inputs;
    int64_t num_tokens = batch_size * max_query_len;

    inputs.x =
        create_random_tensor({num_tokens, test_config_.dim}, -1.0f, 1.0f);
    inputs.q_norm = create_random_tensor(
        {num_tokens, test_config_.q_lora_rank}, -1.0f, 1.0f);

    inputs.positions =
        torch::randint(0, max_query_len, {num_tokens}, int_option_);

    const std::vector<int64_t> cache_shape = {test_config_.block_num,
                                              test_config_.head_kv,
                                              test_config_.block_size,
                                              test_config_.index_head_dim};
    if (quantized_cache) {
      inputs.k_cache = torch::zeros(cache_shape, options_.dtype(torch::kChar));
      inputs.k_cache_scale = torch::zeros({test_config_.block_num,
                                           test_config_.head_kv,
                                           test_config_.block_size},
                                          options_.dtype(torch::kFloat32));
    } else {
      inputs.k_cache = create_random_tensor(cache_shape, -0.5f, 0.5f);
    }

    inputs.weights = create_random_weights(test_config_.dim,
                                           test_config_.index_n_heads,
                                           test_config_.index_head_dim,
                                           test_config_.q_lora_rank);

    if (chunked_prefill) {
      populate_chunked_attention_metadata(inputs.metadata,
                                          batch_size,
                                          history_len,
                                          max_query_len,
                                          cache_block_size,
                                          use_noncontiguous_blocks);
    } else {
      populate_attention_metadata(inputs.metadata,
                                  batch_size,
                                  max_query_len,
                                  test_config_.max_position_embeddings,
                                  is_prefill,
                                  num_tokens);
    }
    return inputs;
  }

  TestInputs create_quantized_inputs(int64_t batch_size,
                                     int64_t max_query_len,
                                     bool is_prefill,
                                     bool chunked_prefill = false,
                                     int64_t history_len = 0,
                                     int64_t cache_block_size = 1,
                                     bool use_noncontiguous_blocks = false) {
    return create_inputs(batch_size,
                         max_query_len,
                         is_prefill,
                         chunked_prefill,
                         history_len,
                         /*use_default_rope=*/false,
                         /*quantized_cache=*/true,
                         cache_block_size,
                         use_noncontiguous_blocks);
  }

  void fill_quantized_cache(TestInputs& inputs) {
    CHECK(inputs.k_cache_scale.has_value());
    torch::Tensor cache_values =
        torch::randint(
            -64, 64, inputs.k_cache.sizes(), options_.dtype(torch::kInt32))
            .to(torch::kChar);
    inputs.k_cache.copy_(cache_values);
    inputs.k_cache_scale->copy_(torch::rand(inputs.k_cache_scale->sizes(),
                                            options_.dtype(torch::kFloat32)) +
                                0.01f);
  }

  Indexer create_indexer(TestInputs& inputs, bool enable_fused_qk) {
    StateDict state_dict(inputs.weights);
    QuantArgs quant_args;
    Indexer indexer = Indexer(IndexerImpl(test_config_.dim,
                                          test_config_.index_n_heads,
                                          test_config_.index_head_dim,
                                          test_config_.qk_rope_head_dim,
                                          test_config_.index_topk,
                                          test_config_.q_lora_rank,
                                          enable_fused_qk,
                                          rotary_emb_,
                                          quant_args,
                                          parallel_args_,
                                          options_));
    indexer->load_state_dict(state_dict);
    return indexer;
  }

  std::tuple<torch::Tensor, torch::Tensor> run_indexer(TestInputs& inputs,
                                                       bool is_prefill,
                                                       bool enable_fused_qk) {
    Indexer indexer = create_indexer(inputs, enable_fused_qk);
    return indexer->forward(inputs.x,
                            inputs.q_norm,
                            inputs.positions,
                            inputs.k_cache,
                            inputs.metadata,
                            is_prefill,
                            inputs.k_cache_scale);
  }

  void expect_select_output(const torch::Tensor& block_tables,
                            const torch::Tensor& context_lens,
                            int64_t num_tokens) const {
    EXPECT_EQ(block_tables.scalar_type(), torch::kInt32);
    EXPECT_EQ(block_tables.sizes(),
              (torch::IntArrayRef{num_tokens, test_config_.index_topk}));
    EXPECT_EQ(context_lens.scalar_type(), torch::kInt32);
    EXPECT_EQ(context_lens.sizes(), (torch::IntArrayRef{num_tokens}));
  }

  void expect_quantized_cache_updated(const TestInputs& inputs) const {
    EXPECT_EQ(inputs.k_cache.scalar_type(), torch::kChar);
    torch::Tensor k_cache_cpu = inputs.k_cache.cpu();
    EXPECT_TRUE(k_cache_cpu.ne(0).any().item<bool>());
    ASSERT_TRUE(inputs.k_cache_scale.has_value());
    EXPECT_EQ(inputs.k_cache_scale->scalar_type(), torch::kFloat32);
    torch::Tensor k_cache_scale_cpu = inputs.k_cache_scale->cpu();
    EXPECT_TRUE(torch::isfinite(k_cache_scale_cpu).all().item<bool>());
    EXPECT_TRUE(k_cache_scale_cpu.ne(0).any().item<bool>());
  }

  v32_cp::DeepseekV32CPContext make_single_rank_sp_context(
      const TestInputs& inputs,
      int64_t token_num) const {
    const int32_t token_num_i32 = static_cast<int32_t>(token_num);
    const int32_t context_len = static_cast<int32_t>(
        inputs.metadata.is_chunked_prefill ? inputs.metadata.total_kv_len
                                           : token_num);
    v32_sp::DeepseekV32SPSegment segment;
    segment.req_idx = 0;
    segment.rank = 0;
    segment.q_tokens = token_num_i32;
    segment.suffix_k_len = token_num_i32;
    segment.ctx_k_len = context_len;
    segment.world_begin = 0;

    torch::Tensor segment_prefix =
        torch::tensor({0, token_num_i32}, int_option_).view({1, 2});
    torch::Tensor context_prefix =
        torch::tensor({0, context_len}, int_option_).view({1, 2});
    v32_cp::DeepseekV32CPContext sp_ctx;
    sp_ctx.local_attn_metadata = inputs.metadata;
    sp_ctx.batch_forward_type = inputs.metadata.is_chunked_prefill
                                    ? BatchForwardType::CHUNKED_PREFILL
                                    : BatchForwardType::PREFILL;
    sp_ctx.local_segments = {segment};
    sp_ctx.seg_q_starts_cpu = {0};
    sp_ctx.req_q_offsets_cpu = {0};
    sp_ctx.req_ctx_offsets_cpu = {0};
    sp_ctx.seg_q_cu_lens_2col = segment_prefix;
    sp_ctx.seg_suffix_k_cu_lens_2col = segment_prefix;
    sp_ctx.seg_ctx_k_cu_lens_2col = context_prefix;
    sp_ctx.seg_ctx_lens_1col = torch::tensor({context_len}, int_option_);
    sp_ctx.gathered_reorder_index =
        torch::arange(token_num, options_.dtype(torch::kInt64));
    sp_ctx.gathered_slot_mapping = inputs.metadata.slot_mapping;
    sp_ctx.total_tokens = token_num_i32;
    sp_ctx.rank = 0;
    return sp_ctx;
  }

  ParallelArgs parallel_args_{0, 1, nullptr};
  TestConfig test_config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_option_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
};

TEST_F(IndexerTest, PrefillBatch) {
  LOG(INFO) << "Testing Prefill (Small Batch)";
  int64_t batch_size = 2;
  int64_t max_query_len = 4096;
  const bool is_prefill = true;
  const bool enable_fused_qk = false;
  int64_t num_tokens = batch_size * max_query_len;
  TestInputs inputs = create_inputs(batch_size, max_query_len, is_prefill);
  auto [new_block_tables, new_context_lens] =
      run_indexer(inputs, is_prefill, enable_fused_qk);

  EXPECT_EQ(new_block_tables.sizes().size(), 2)
      << "new_block_tables should be 2D tensor";
  EXPECT_EQ(new_context_lens.sizes().size(), 1)
      << "new_context_lens should be 1D tensor";
  EXPECT_EQ(new_block_tables.size(0), num_tokens) << "Batch size should match";
  EXPECT_EQ(new_block_tables.size(1), test_config_.index_topk)
      << "Top-k should match";

  // Verify that the first value in new_block_tables is 1 (calculated via vLLM
  // MLU)
  EXPECT_EQ(new_block_tables.index({0, 0}).item<int64_t>(), 1)
      << "The first value in new_block_tables should be 1";
}

TEST_F(IndexerTest, ChunkedPrefillBatch) {
  LOG(INFO) << "Testing Chunked Prefill";
  const int64_t batch_size = 2;
  const int64_t history_len = 128;
  const int64_t current_len = 64;
  int64_t num_new_tokens = batch_size * current_len;
  const bool is_prefill = true;
  const bool is_chunked = true;
  const bool enable_fused_qk = false;
  TestInputs inputs = create_inputs(
      batch_size, current_len, is_prefill, is_chunked, history_len);
  auto [new_block_tables, new_context_lens] =
      run_indexer(inputs, is_prefill, enable_fused_qk);

  // Validations
  // Shape Verification
  EXPECT_EQ(new_block_tables.dim(), 2);
  EXPECT_EQ(new_block_tables.size(0), num_new_tokens);  // [batch * current_len]
  EXPECT_EQ(new_block_tables.size(1), test_config_.index_topk);

  // Value Verification
  auto top1_indices = new_block_tables.index({torch::indexing::Slice(), 0})
                          .to(torch::kInt64)
                          .cpu();
  auto top1_sum = top1_indices.sum().item<int64_t>();
  auto top1_max = top1_indices.max().item<int64_t>();

  LOG(INFO) << "[top-1 block index] sum: " << top1_sum << ", max: " << top1_max;

  // The expected value is calculated via vLLM MLU
  int64_t expected_sum = 12288;
  int64_t expected_max = 192;
  EXPECT_EQ(top1_sum, expected_sum)
      << "top-1 block index sum does not match ground truth";
  EXPECT_EQ(top1_max, expected_max)
      << "top-1 block index max does not match ground truth";
}

TEST_F(IndexerTest, CompareFusedVsNonFusedDecode) {
  LOG(INFO) << "Testing Decode";
  TestInputs inputs = create_inputs(128, 1, false);

  auto [base_block_tables, base_context_lens] =
      run_indexer(inputs, false, false);
  auto [fused_block_tables, fused_context_lens] =
      run_indexer(inputs, false, true);

  auto fused_block_tables_slice = fused_block_tables.slice(1, 0, 1);
  auto base_block_tables_slice = base_block_tables.slice(1, 0, 1);
  test::verify_tensor_close(fused_context_lens.to(torch::kFloat32),
                            base_context_lens.to(torch::kFloat32));
  test::verify_tensor_close(fused_block_tables_slice.to(torch::kFloat32),
                            base_block_tables_slice.to(torch::kFloat32));
}

TEST_F(IndexerTest, CompareFusedVsNonFusedMultipleRuns) {
  LOG(INFO) << "Testing with multiple random seeds";

  Device device(options_.device());
  for (int i = 0; i < 3; ++i) {
    LOG(INFO) << "Random seed iteration: " << i;
    device.set_seed(i * 100);
    TestInputs inputs = create_inputs(128, 1, false);

    auto [base_block_tables, base_context_lens] =
        run_indexer(inputs, false, false);
    auto [fused_block_tables, fused_context_lens] =
        run_indexer(inputs, false, true);

    auto fused_block_tables_slice = fused_block_tables.slice(1, 0, 1);
    auto base_block_tables_slice = base_block_tables.slice(1, 0, 1);
    test::verify_tensor_close(fused_context_lens.to(torch::kFloat32),
                              base_context_lens.to(torch::kFloat32));
    test::verify_tensor_close(fused_block_tables_slice.to(torch::kFloat32),
                              base_block_tables_slice.to(torch::kFloat32));
  }
}

TEST_F(IndexerTest, CompareFusedVsNonFusedEdgeCaseSmall) {
  LOG(INFO) << "Testing Edge Case (Very Small Input)";
  TestInputs inputs = create_inputs(16, 1, false);

  auto [base_block_tables, base_context_lens] =
      run_indexer(inputs, false, false);
  auto [fused_block_tables, fused_context_lens] =
      run_indexer(inputs, false, true);

  auto fused_block_tables_slice = fused_block_tables.slice(1, 0, 1);
  auto base_block_tables_slice = base_block_tables.slice(1, 0, 1);
  test::verify_tensor_close(fused_context_lens.to(torch::kFloat32),
                            base_context_lens.to(torch::kFloat32));
  test::verify_tensor_close(fused_block_tables_slice.to(torch::kFloat32),
                            base_block_tables_slice.to(torch::kFloat32));
}

TEST_F(IndexerTest, DefaultRopeDecodePath) {
  LOG(INFO) << "Testing default rope decode path";
  TestInputs inputs = create_inputs(32, 1, false, false, 0, true);

  auto [block_tables, context_lens] = run_indexer(inputs, false, false);

  EXPECT_EQ(block_tables.dim(), 2);
  EXPECT_EQ(block_tables.size(0), 32);
  EXPECT_EQ(block_tables.size(1), test_config_.index_topk);
  EXPECT_EQ(context_lens.dim(), 1);
  EXPECT_EQ(context_lens.size(0), 32);
}

TEST_F(IndexerTest, Int8NormalPrefillWritesCacheScaleAndSelectsBlocks) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kQueryLen = 128;
  TestInputs inputs =
      create_quantized_inputs(kBatchSize, kQueryLen, /*is_prefill=*/true);

  auto [block_tables, context_lens] =
      run_indexer(inputs, /*is_prefill=*/true, /*enable_fused_qk=*/true);

  expect_select_output(block_tables, context_lens, kBatchSize * kQueryLen);
  expect_quantized_cache_updated(inputs);
}

TEST_F(IndexerTest, Int8ChunkedPrefillSelectsAcrossNoncontiguousPages) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kHistoryLen = 24;
  constexpr int64_t kQueryLen = 24;
  constexpr int64_t kBlockSize = 16;
  TestInputs inputs =
      create_quantized_inputs(kBatchSize,
                              kQueryLen,
                              /*is_prefill=*/true,
                              /*chunked_prefill=*/true,
                              kHistoryLen,
                              kBlockSize,
                              /*use_noncontiguous_blocks=*/true);
  fill_quantized_cache(inputs);

  auto [block_tables, context_lens] =
      run_indexer(inputs, /*is_prefill=*/true, /*enable_fused_qk=*/true);

  expect_select_output(block_tables, context_lens, kBatchSize * kQueryLen);
  expect_quantized_cache_updated(inputs);
}

TEST_F(IndexerTest, Int8DecodeWritesCacheScaleAndSelectsBlocks) {
  constexpr int64_t kBatchSize = 16;
  TestInputs inputs = create_quantized_inputs(
      kBatchSize, /*max_query_len=*/1, /*is_prefill=*/false);

  auto [block_tables, context_lens] =
      run_indexer(inputs, /*is_prefill=*/false, /*enable_fused_qk=*/true);

  expect_select_output(block_tables, context_lens, kBatchSize);
  expect_quantized_cache_updated(inputs);
}

TEST_F(IndexerTest, Int8SpPrefillWritesCacheScaleAndSelectsBlocks) {
  constexpr int64_t kTokenNum = 128;
  TestInputs inputs = create_quantized_inputs(
      /*batch_size=*/1, kTokenNum, /*is_prefill=*/true);
  Indexer indexer = create_indexer(inputs, /*enable_fused_qk=*/true);
  v32_cp::DeepseekV32CPContext sp_ctx =
      make_single_rank_sp_context(inputs, kTokenNum);

  IndexerSPPreOut pre_out = indexer->sp_pre(inputs.x,
                                            inputs.q_norm,
                                            inputs.positions,
                                            inputs.metadata,
                                            sp_ctx,
                                            /*quantize_output=*/false);
  EXPECT_EQ(pre_out.q.scalar_type(), torch::kBFloat16);
  EXPECT_FALSE(pre_out.q_scale.has_value());

  auto [block_tables, context_lens] =
      indexer->sp_post(pre_out,
                       pre_out.k_local,
                       inputs.k_cache,
                       inputs.metadata,
                       sp_ctx.gathered_slot_mapping,
                       sp_ctx,
                       inputs.k_cache_scale);

  expect_select_output(block_tables, context_lens, kTokenNum);
  expect_quantized_cache_updated(inputs);
}

TEST_F(IndexerTest, Int8SpChunkedPrefillMatchesSingleRankNormalPath) {
  constexpr int64_t kBatchSize = 1;
  constexpr int64_t kHistoryLen = 24;
  constexpr int64_t kQueryLen = 24;
  constexpr int64_t kBlockSize = 16;
  TestInputs normal_inputs =
      create_quantized_inputs(kBatchSize,
                              kQueryLen,
                              /*is_prefill=*/true,
                              /*chunked_prefill=*/true,
                              kHistoryLen,
                              kBlockSize,
                              /*use_noncontiguous_blocks=*/true);
  fill_quantized_cache(normal_inputs);

  TestInputs sp_inputs = normal_inputs;
  sp_inputs.k_cache = normal_inputs.k_cache.clone();
  sp_inputs.k_cache_scale = normal_inputs.k_cache_scale->clone();

  auto [normal_block_tables, normal_context_lens] =
      run_indexer(normal_inputs,
                  /*is_prefill=*/true,
                  /*enable_fused_qk=*/true);

  Indexer indexer = create_indexer(sp_inputs, /*enable_fused_qk=*/true);
  v32_cp::DeepseekV32CPContext sp_ctx =
      make_single_rank_sp_context(sp_inputs, kQueryLen);
  IndexerSPPreOut pre_out = indexer->sp_pre(sp_inputs.x,
                                            sp_inputs.q_norm,
                                            sp_inputs.positions,
                                            sp_inputs.metadata,
                                            sp_ctx,
                                            /*quantize_output=*/false);
  auto [sp_block_tables, sp_context_lens] =
      indexer->sp_post(pre_out,
                       pre_out.k_local,
                       sp_inputs.k_cache,
                       sp_inputs.metadata,
                       sp_ctx.gathered_slot_mapping,
                       sp_ctx,
                       sp_inputs.k_cache_scale);

  expect_select_output(sp_block_tables, sp_context_lens, kQueryLen);
  expect_quantized_cache_updated(sp_inputs);
  EXPECT_TRUE(torch::equal(sp_context_lens, normal_context_lens));
  EXPECT_TRUE(torch::equal(
      sp_block_tables.slice(/*dim=*/1, /*start=*/0, /*end=*/1),
      normal_block_tables.slice(/*dim=*/1, /*start=*/0, /*end=*/1)));
}

}  // namespace layer
}  // namespace xllm
