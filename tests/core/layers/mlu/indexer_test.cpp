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

#include "layers/mlu/indexer.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <sstream>

#include "core/framework/config/kv_cache_config.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quantization/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/common/kv_shard_batch_metadata.h"
#include "layers/mlu/attention.h"
#include "layers/mlu/dcp_decode_context.h"
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

  void expect_dcp_local_prefill_candidates(TestInputs& inputs) {
    parallel_args_.world_size() = 2;
    parallel_args_.kv_split_size() = 2;
    constexpr int32_t kDcpRank = 0;
    const KVShardLayout layout(
        static_cast<int32_t>(KVCacheConfig::get_instance().block_size()),
        /*dcp_size=*/2,
        kDcpRank);
    const KVShardCausalSelectorMetadata causal_selector =
        build_kv_shard_causal_selector_metadata(inputs.metadata, layout);

    AttentionMetadata prefill_metadata = inputs.metadata;
    prefill_metadata.slot_mapping =
        localize_kv_shard_slots(inputs.metadata.slot_mapping, layout);
    AttentionMetadata selector_metadata = prefill_metadata;
    selector_metadata.q_cu_seq_lens = causal_selector.q_cu_seq_lens;
    selector_metadata.kv_cu_seq_lens = causal_selector.q_cu_seq_lens;
    selector_metadata.kv_seq_lens = causal_selector.local_context_lens;
    selector_metadata.block_table = causal_selector.block_table;
    selector_metadata.max_query_len = 1;
    selector_metadata.max_seq_len = test_config_.index_topk;
    selector_metadata.total_kv_len = 0;
    selector_metadata.is_prefill = false;
    selector_metadata.is_chunked_prefill = false;

    Indexer indexer = create_indexer(inputs, /*enable_fused_qk=*/true);
    const DcpIndexerLocalCandidates candidates =
        indexer->forward_dcp_local_prefill(inputs.x,
                                           inputs.q_norm,
                                           inputs.positions,
                                           inputs.k_cache,
                                           prefill_metadata,
                                           selector_metadata,
                                           inputs.k_cache_scale);

    const int64_t token_count = inputs.x.size(0);
    EXPECT_EQ(candidates.scores.sizes(),
              (torch::IntArrayRef{token_count, test_config_.index_topk}));
    EXPECT_EQ(candidates.global_slots.sizes(), candidates.scores.sizes());

    torch::Tensor columns =
        torch::arange(test_config_.index_topk, options_.dtype(torch::kInt32));
    torch::Tensor invalid_columns =
        columns.unsqueeze(0) >= causal_selector.local_context_lens.unsqueeze(1);
    EXPECT_TRUE(candidates.global_slots.masked_select(invalid_columns)
                    .eq(KVShardLayout::kInvalidSlot)
                    .all()
                    .item<bool>());
    EXPECT_TRUE(candidates.scores.masked_select(invalid_columns)
                    .isneginf()
                    .all()
                    .item<bool>());
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

  v32_cp::DeepseekV32CPContext make_single_rank_cp_context(
      const TestInputs& inputs,
      int64_t token_num) const {
    const int32_t token_num_i32 = static_cast<int32_t>(token_num);
    const int32_t context_len = static_cast<int32_t>(
        inputs.metadata.is_chunked_prefill ? inputs.metadata.total_kv_len
                                           : token_num);
    v32_cp::DeepseekV32CPSegment segment;
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
    v32_cp::DeepseekV32CPContext cp_ctx;
    cp_ctx.local_attn_metadata = inputs.metadata;
    cp_ctx.batch_forward_type = inputs.metadata.is_chunked_prefill
                                    ? BatchForwardType::CHUNKED_PREFILL
                                    : BatchForwardType::PREFILL;
    cp_ctx.local_segments = {segment};
    cp_ctx.seg_q_starts_cpu = {0};
    cp_ctx.req_q_offsets_cpu = {0};
    cp_ctx.req_ctx_offsets_cpu = {0};
    cp_ctx.seg_q_cu_lens_2col = segment_prefix;
    cp_ctx.seg_suffix_k_cu_lens_2col = segment_prefix;
    cp_ctx.seg_ctx_k_cu_lens_2col = context_prefix;
    cp_ctx.seg_ctx_lens_1col = torch::tensor({context_len}, int_option_);
    cp_ctx.gathered_reorder_index =
        torch::arange(token_num, options_.dtype(torch::kInt64));
    cp_ctx.gathered_slot_mapping = inputs.metadata.slot_mapping;
    cp_ctx.total_tokens = token_num_i32;
    cp_ctx.rank = 0;
    return cp_ctx;
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

TEST_F(IndexerTest, DcpLocalCausalPrefillSelectsRankLocalCandidates) {
  test_config_.index_n_heads = 32;
  TestInputs inputs = create_inputs(
      /*batch_size=*/1,
      /*max_query_len=*/24,
      /*is_prefill=*/true,
      /*chunked_prefill=*/false,
      /*history_len=*/0,
      /*use_default_rope=*/false,
      /*quantized_cache=*/false,
      /*cache_block_size=*/16);
  expect_dcp_local_prefill_candidates(inputs);
}

TEST_F(IndexerTest, DcpLocalCausalChunkedPrefillSelectsRankLocalCandidates) {
  test_config_.index_n_heads = 32;
  TestInputs inputs = create_inputs(
      /*batch_size=*/1,
      /*max_query_len=*/24,
      /*is_prefill=*/true,
      /*chunked_prefill=*/true,
      /*history_len=*/24,
      /*use_default_rope=*/false,
      /*quantized_cache=*/false,
      /*cache_block_size=*/16);
  expect_dcp_local_prefill_candidates(inputs);
}

class DcpIndexerScoreTest : public IndexerTest {
 protected:
  void SetUp() override {
    IndexerTest::SetUp();
    test_config_.index_n_heads = 32;
  }
  void expect_scores(const DcpIndexerLocalCandidates& candidates,
                     const IndexerCPPreOut& pre_out,
                     const torch::Tensor& cache,
                     const std::optional<torch::Tensor>& cache_scale,
                     const KVShardLayout& layout,
                     double tolerance = 1e-4) {
    torch::Tensor q = pre_out.q.cpu().to(torch::kFloat32);
    if (pre_out.q_scale.has_value()) {
      q *= pre_out.q_scale->cpu().unsqueeze(-1);
    }
    torch::Tensor weights = pre_out.weights.cpu().to(torch::kFloat32);
    torch::Tensor keys = cache.cpu().flatten(0, 2).to(torch::kFloat32);
    if (cache_scale.has_value()) {
      keys *= cache_scale->cpu().flatten().unsqueeze(-1);
    }
    torch::Tensor slots =
        localize_kv_shard_slots(candidates.global_slots.cpu(), layout);
    torch::Tensor scores = candidates.scores.cpu();
    const auto q_data = q.accessor<float, 3>();
    const auto weight_data = weights.accessor<float, 2>();
    const auto key_data = keys.accessor<float, 2>();
    const auto slot_data = slots.accessor<int32_t, 2>();
    const auto score_data = scores.accessor<float, 2>();
    int64_t valid_count = 0;
    int64_t invalid_count = 0;
    for (int64_t row = 0; row < slots.size(0); ++row) {
      for (int64_t col = 0; col < slots.size(1); ++col) {
        const int32_t slot = slot_data[row][col];
        if (slot < 0) {
          EXPECT_TRUE(std::isinf(score_data[row][col]));
          EXPECT_LT(score_data[row][col], 0);
          ++invalid_count;
          continue;
        }
        double expected = 0;
        for (int64_t head = 0; head < q.size(1); ++head) {
          double dot = 0;
          for (int64_t dim = 0; dim < q.size(2); ++dim) {
            dot += static_cast<double>(q_data[row][head][dim]) *
                   key_data[slot][dim];
          }
          expected += std::max(dot, 0.0) * weight_data[row][head];
        }
        ASSERT_NEAR(score_data[row][col],
                    expected,
                    tolerance * (1.0 + std::abs(expected)))
            << "row=" << row << " slot=" << slot;
        ++valid_count;
      }
    }
    EXPECT_GT(valid_count, 0);
    EXPECT_GT(invalid_count, 0);
  }

  void expect_entry_scores(bool is_prefill,
                           bool chunked,
                           bool quantized_cache = false) {
    TestInputs inputs = create_inputs(
        /*batch_size=*/1,
        /*max_query_len=*/is_prefill ? 8 : 1,
        is_prefill,
        chunked,
        /*history_len=*/chunked ? 8 : 0,
        /*use_default_rope=*/true,
        quantized_cache,
        /*cache_block_size=*/1);
    if (quantized_cache) {
      fill_quantized_cache(inputs);
    }
    parallel_args_.world_size() = 2;
    parallel_args_.kv_split_size() = 2;
    const KVShardLayout layout(
        /*physical_block_size=*/1, /*dcp_size=*/2, /*dcp_rank=*/0);
    inputs.positions.zero_();
    // Fixed projections produce opposite query heads and signed head weights.
    // Position zero avoids differences between fused and eager rotary paths.
    inputs.x.fill_(1);
    inputs.q_norm.fill_(1);
    inputs.weights.at("wq_b.weight").zero_();
    inputs.weights.at("wq_b.weight").index_put_({0, 0}, 1);
    inputs.weights.at("wq_b.weight")
        .index_put_({test_config_.index_head_dim, 0}, -1);
    inputs.weights.at("weights_proj.weight").zero_();
    inputs.weights.at("weights_proj.weight").index_put_({0, 0}, 1);
    inputs.weights.at("weights_proj.weight").index_put_({1, 0}, -2);
    Indexer indexer = create_indexer(inputs, /*enable_fused_qk=*/true);
    IndexerCPPreOut reference = indexer->cp_pre(
        inputs.x, inputs.q_norm, inputs.positions, inputs.metadata);
    if (quantized_cache) {
      const torch::Tensor smooth =
          torch::ones({reference.q.size(-1)},
                      torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(reference.q.device()));
      std::tie(reference.q, reference.q_scale) =
          kernel::mlu::scaled_quantize(reference.q, smooth);
    }
    DcpIndexerLocalCandidates candidates;
    if (is_prefill) {
      const KVShardCausalSelectorMetadata causal =
          build_kv_shard_causal_selector_metadata(inputs.metadata, layout);
      AttentionMetadata selector = inputs.metadata;
      selector.q_cu_seq_lens = causal.q_cu_seq_lens;
      selector.kv_cu_seq_lens = causal.q_cu_seq_lens;
      selector.kv_seq_lens = causal.local_context_lens;
      selector.block_table = causal.block_table;
      selector.max_query_len = 1;
      selector.is_prefill = false;
      selector.is_chunked_prefill = false;
      inputs.metadata.slot_mapping =
          localize_kv_shard_slots(inputs.metadata.slot_mapping, layout);
      candidates = indexer->forward_dcp_local_prefill(inputs.x,
                                                      inputs.q_norm,
                                                      inputs.positions,
                                                      inputs.k_cache,
                                                      inputs.metadata,
                                                      selector,
                                                      inputs.k_cache_scale);
    } else {
      inputs.metadata.kv_seq_lens.fill_(8);
      inputs.metadata.block_table = torch::arange(8, int_option_).view({1, 8});
      candidates = indexer->forward_dcp_local(inputs.x,
                                              inputs.q_norm,
                                              inputs.positions,
                                              inputs.k_cache,
                                              inputs.metadata,
                                              inputs.k_cache_scale);
    }
    expect_scores(candidates,
                  reference,
                  inputs.k_cache,
                  inputs.k_cache_scale,
                  layout,
                  /*tolerance=*/1e-4);
  }
};

TEST_F(DcpIndexerScoreTest, DecodeMatchesReluReference) {
  expect_entry_scores(/*is_prefill=*/false, /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, PrefillMatchesReluReference) {
  expect_entry_scores(/*is_prefill=*/true, /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, ChunkedPrefillMatchesReluReference) {
  expect_entry_scores(/*is_prefill=*/true, /*chunked=*/true);
}

TEST_F(DcpIndexerScoreTest, CpCandidatesPreserveGlobalTopk) {
  parallel_args_.world_size() = 2;
  parallel_args_.kv_split_size() = 2;
  rotary_emb_ = std::make_shared<RotaryEmbeddingImpl>(
      test_config_.qk_rope_head_dim,
      test_config_.max_position_embeddings,
      test_config_.rope_theta,
      test_config_.rope_interleaved,
      options_);
  IndexerCPPreOut pre_out;
  pre_out.q = torch::zeros({2, 32, 128}, options_);
  pre_out.q.select(/*dim=*/2, /*index=*/0).fill_(1);
  pre_out.q.select(/*dim=*/1, /*index=*/1).zero_();
  pre_out.q.index_put_({torch::indexing::Slice(), 1, 1}, 1);
  pre_out.k_local = torch::zeros({2, 128}, options_);
  pre_out.weights = torch::zeros({2, 32}, options_);
  pre_out.weights.index_put_({0, 0}, 1);
  pre_out.weights.index_put_({0, 1}, 1);
  pre_out.weights.index_put_({1, 0}, -1);
  pre_out.weights.index_put_({1, 1}, 2);
  std::vector<torch::Tensor> rank_scores;
  std::vector<torch::Tensor> rank_slots;
  rank_scores.reserve(2);
  rank_slots.reserve(2);
  for (int32_t rank = 0; rank < 2; ++rank) {
    SCOPED_TRACE(rank);
    parallel_args_.rank() = rank;
    const KVShardLayout layout(
        /*physical_block_size=*/1, /*dcp_size=*/2, rank);
    TestInputs inputs;
    Indexer indexer = create_indexer(inputs, /*enable_fused_qk=*/true);
    const int32_t ctx_len = 1025 - rank;
    torch::Tensor keys = torch::zeros({ctx_len, 1, 1, 128}, options_);
    keys.select(/*dim=*/3, /*index=*/0).fill_(20);
    keys.index_put_({0, 0, 0, 0}, rank == 0 ? 10 : 6);
    keys.index_put_({0, 0, 0, 1}, rank == 0 ? -9 : 0);
    std::optional<torch::Tensor> scales = std::nullopt;
    AttentionMetadata prefill;
    prefill.slot_mapping = torch::full({2}, -1, int_option_);
    AttentionMetadata selector;
    selector.block_table =
        torch::arange(ctx_len, int_option_).unsqueeze(0).repeat({2, 1});
    selector.kv_seq_lens = torch::full({2}, ctx_len, int_option_);
    selector.q_cu_seq_lens = torch::arange(3, int_option_);
    const DcpIndexerLocalCandidates candidates =
        indexer->forward_dcp_local_prefill_from_cp(
            pre_out, keys, prefill, selector);
    expect_scores(candidates, pre_out, keys, scales, layout);
    rank_scores.emplace_back(candidates.scores);
    rank_slots.emplace_back(candidates.global_slots);
  }
  DcpIndexerGatherAsyncCtx gathered;
  gathered.gathered_scores = torch::stack(rank_scores);
  gathered.gathered_global_slots = torch::stack(rank_slots);
  const DcpDecodeContext context(
      KVShardLayout(/*physical_block_size=*/1, /*dcp_size=*/2, /*dcp_rank=*/0),
      /*dcp_group=*/nullptr);
  const DsaTopkState merged = context.finish_indexer_candidate_merge(
      std::move(gathered), /*topk=*/2048, torch::zeros({2}, int_option_));
  torch::Tensor selected = merged.block_tables()[0].cpu();
  torch::Tensor expected = torch::arange(2049, torch::kInt32);
  expected = expected.masked_select(expected.ne(1));
  EXPECT_TRUE(torch::equal(std::get<0>(selected.sort()), expected));
  EXPECT_TRUE(selected.eq(0).any().item<bool>());
  EXPECT_FALSE(selected.eq(1).any().item<bool>());
  EXPECT_EQ(merged.context_lens()[0].item<int32_t>(), 2048);
}

TEST_F(DcpIndexerScoreTest, GenericCpPreservesInt8Scales) {
  test_config_.index_n_heads = 64;
  parallel_args_.world_size() = 2;
  parallel_args_.kv_split_size() = 2;
  const KVShardLayout layout(
      /*physical_block_size=*/1, /*dcp_size=*/2, /*dcp_rank=*/0);
  TestInputs empty_inputs;
  Indexer indexer = create_indexer(empty_inputs, /*enable_fused_qk=*/true);
  for (const bool quantized_query : {false, true}) {
    SCOPED_TRACE(quantized_query);
    IndexerCPPreOut pre_out;
    pre_out.q = torch::zeros(
        {2, test_config_.index_n_heads, test_config_.index_head_dim}, options_);
    pre_out.q.select(1, 0).select(1, 0).fill_(1);
    pre_out.q.select(1, 1).select(1, 0).fill_(-1);
    if (quantized_query) {
      pre_out.q = pre_out.q.to(torch::kChar);
      pre_out.q_scale = torch::full({2, test_config_.index_n_heads},
                                    0.5,
                                    options_.dtype(torch::kFloat32));
    }
    pre_out.weights = torch::zeros({2, test_config_.index_n_heads}, options_);
    pre_out.weights.select(1, 0).fill_(1);
    pre_out.weights.select(1, 1).fill_(-2);
    pre_out.k_local = torch::zeros({2, test_config_.index_head_dim}, options_);
    torch::Tensor cache = torch::zeros({3, 1, 1, test_config_.index_head_dim},
                                       options_.dtype(torch::kChar));
    cache.index_put_({0, 0, 0, 0}, 4);
    cache.index_put_({1, 0, 0, 0}, -3);
    cache.index_put_({2, 0, 0, 0}, 2);
    const torch::Tensor cache_scale =
        torch::tensor({0.25f, 1.0f, 2.0f}, options_.dtype(torch::kFloat32))
            .view({3, 1, 1});
    AttentionMetadata prefill;
    // Score resident history without overwriting it during the CP cache write.
    prefill.slot_mapping = torch::full({2}, -1, int_option_);
    AttentionMetadata selector;
    selector.is_prefill = false;
    selector.q_cu_seq_lens = torch::arange(3, int_option_);
    selector.kv_cu_seq_lens = selector.q_cu_seq_lens;
    selector.kv_seq_lens = torch::full({2}, 3, int_option_);
    selector.block_table = torch::arange(3, int_option_).repeat({2, 1});
    const DcpIndexerLocalCandidates candidates =
        indexer->forward_dcp_local_prefill_from_cp(
            pre_out, cache, prefill, selector, cache_scale);
    expect_scores(candidates,
                  pre_out,
                  cache,
                  cache_scale,
                  layout,
                  /*tolerance=*/1e-4);
  }
}

TEST_F(DcpIndexerScoreTest, GenericDecodeSupports64Heads) {
  test_config_.index_n_heads = 64;
  expect_entry_scores(/*is_prefill=*/false, /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, GenericPrefillSupports64Heads) {
  test_config_.index_n_heads = 64;
  expect_entry_scores(/*is_prefill=*/true, /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, GenericDecodeSupportsFloat16) {
  options_ = options_.dtype(torch::kFloat16);
  expect_entry_scores(/*is_prefill=*/false, /*chunked=*/false);
}

TEST_F(DcpIndexerScoreTest, GenericDecodePreservesInt8Scales) {
  expect_entry_scores(/*is_prefill=*/false,
                      /*chunked=*/false,
                      /*quantized_cache=*/true);
}

TEST_F(DcpIndexerScoreTest, GenericPrefillPreservesInt8Scales) {
  expect_entry_scores(/*is_prefill=*/true,
                      /*chunked=*/false,
                      /*quantized_cache=*/true);
}

TEST_F(DcpIndexerScoreTest, GenericChunkedPrefillPreservesInt8Scales) {
  expect_entry_scores(/*is_prefill=*/true,
                      /*chunked=*/true,
                      /*quantized_cache=*/true);
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

TEST_F(IndexerTest, Int8CpPrefillWritesCacheScaleAndSelectsBlocks) {
  constexpr int64_t kTokenNum = 128;
  TestInputs inputs = create_quantized_inputs(
      /*batch_size=*/1, kTokenNum, /*is_prefill=*/true);
  Indexer indexer = create_indexer(inputs, /*enable_fused_qk=*/true);
  v32_cp::DeepseekV32CPContext cp_ctx =
      make_single_rank_cp_context(inputs, kTokenNum);

  IndexerCPPreOut pre_out = indexer->cp_pre(
      inputs.x, inputs.q_norm, inputs.positions, inputs.metadata);
  EXPECT_EQ(pre_out.q.scalar_type(), torch::kBFloat16);
  EXPECT_FALSE(pre_out.q_scale.has_value());

  auto [block_tables, context_lens] =
      indexer->cp_post(pre_out,
                       pre_out.k_local,
                       inputs.k_cache,
                       inputs.metadata,
                       cp_ctx.gathered_slot_mapping,
                       cp_ctx,
                       inputs.k_cache_scale);

  expect_select_output(block_tables, context_lens, kTokenNum);
  expect_quantized_cache_updated(inputs);
}

TEST_F(IndexerTest, Int8CpChunkedPrefillMatchesSingleRankNormalPath) {
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

  TestInputs cp_inputs = normal_inputs;
  cp_inputs.k_cache = normal_inputs.k_cache.clone();
  cp_inputs.k_cache_scale = normal_inputs.k_cache_scale->clone();

  auto [normal_block_tables, normal_context_lens] =
      run_indexer(normal_inputs,
                  /*is_prefill=*/true,
                  /*enable_fused_qk=*/true);

  Indexer indexer = create_indexer(cp_inputs, /*enable_fused_qk=*/true);
  v32_cp::DeepseekV32CPContext cp_ctx =
      make_single_rank_cp_context(cp_inputs, kQueryLen);
  IndexerCPPreOut pre_out = indexer->cp_pre(
      cp_inputs.x, cp_inputs.q_norm, cp_inputs.positions, cp_inputs.metadata);
  auto [cp_block_tables, cp_context_lens] =
      indexer->cp_post(pre_out,
                       pre_out.k_local,
                       cp_inputs.k_cache,
                       cp_inputs.metadata,
                       cp_ctx.gathered_slot_mapping,
                       cp_ctx,
                       cp_inputs.k_cache_scale);

  expect_select_output(cp_block_tables, cp_context_lens, kQueryLen);
  expect_quantized_cache_updated(cp_inputs);
  EXPECT_TRUE(torch::equal(cp_context_lens, normal_context_lens));
  EXPECT_TRUE(torch::equal(
      cp_block_tables.slice(/*dim=*/1, /*start=*/0, /*end=*/1),
      normal_block_tables.slice(/*dim=*/1, /*start=*/0, /*end=*/1)));
}

}  // namespace layer
}  // namespace xllm
