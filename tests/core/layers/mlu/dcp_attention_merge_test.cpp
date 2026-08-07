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

#include "layers/mlu/dcp_attention_merge.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>

#include "framework/kv_cache/kv_cache.h"
#include "layers/common/attention_metadata.h"
#include "layers/mlu/attention.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm::layer {
namespace {

TEST(DcpAttentionMergeTest, WeightsPartialOutputsByNaturalLogLse) {
  torch::Tensor partial_outputs =
      torch::tensor({2.0f, 4.0f, 6.0f, 8.0f}).reshape({2, 1, 1, 1, 2});
  torch::Tensor partial_lse =
      torch::tensor({std::log(2.0f), std::log(6.0f)}).reshape({2, 1, 1, 1});

  const DcpAttentionResult result =
      merge_dcp_attention_shards(partial_outputs, partial_lse);

  EXPECT_TRUE(torch::allclose(
      result.output, torch::tensor({5.0f, 7.0f}).reshape({1, 1, 1, 2})));
  EXPECT_TRUE(torch::allclose(
      result.lse, torch::tensor({std::log(8.0f)}).reshape({1, 1, 1})));
}

TEST(DcpAttentionMergeTest, IgnoresEmptyShardAndPreservesFiniteShard) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  torch::Tensor partial_outputs =
      torch::tensor({0.0f, 0.0f, 3.0f, 9.0f}).reshape({2, 1, 1, 1, 2});
  torch::Tensor partial_lse =
      torch::tensor({negative_infinity, std::log(4.0f)}).reshape({2, 1, 1, 1});

  const DcpAttentionResult result =
      merge_dcp_attention_shards(partial_outputs, partial_lse);

  EXPECT_TRUE(torch::equal(result.output,
                           torch::tensor({3.0f, 9.0f}).reshape({1, 1, 1, 2})));
  EXPECT_TRUE(torch::allclose(
      result.lse, torch::tensor({std::log(4.0f)}).reshape({1, 1, 1})));
}

TEST(DcpAttentionMergeTest, ReturnsZeroAndNegativeInfinityWhenAllShardsEmpty) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  torch::Tensor partial_outputs = torch::ones({2, 1, 1, 2, 3});
  torch::Tensor partial_lse = torch::full({2, 1, 2, 1}, negative_infinity);

  const DcpAttentionResult result =
      merge_dcp_attention_shards(partial_outputs, partial_lse);

  EXPECT_TRUE(torch::equal(result.output, torch::zeros({1, 1, 2, 3})));
  EXPECT_TRUE(torch::isneginf(result.lse).all().item<bool>());
}

TEST(MluMlaDecodeLseTest, ReturnsFloat32NaturalLogNormalizer) {
  constexpr int64_t kNumHeads = 2;
  constexpr int64_t kHeadSize = 576;
  constexpr int64_t kValueHeadSize = 512;
  constexpr int64_t kContextLength = 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  const torch::Device device(Platform::type_torch(), 0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions int_options = options.dtype(torch::kInt32);

  Attention attention(kNumHeads,
                      kHeadSize,
                      /*num_kv_heads=*/1,
                      kValueHeadSize,
                      /*sliding_window=*/-1,
                      scale,
                      /*use_fused_mla_qkv=*/true,
                      /*enable_lighting_indexer=*/false,
                      /*enable_mla=*/true);
  torch::Tensor query = test::seeded_tensor("dcp_attention_merge/query",
                                            {1, kNumHeads, kHeadSize},
                                            torch::kBFloat16,
                                            device);
  torch::Tensor cache_tensor = test::seeded_tensor("dcp_attention_merge/cache",
                                                   {1, 1, 16, kHeadSize},
                                                   torch::kBFloat16,
                                                   device);
  KVCache kv_cache(KVCacheTensors{cache_tensor, torch::Tensor()});

  AttentionMetadata metadata;
  metadata.block_table = torch::zeros({1, 1}, int_options);
  metadata.kv_seq_lens = torch::tensor({kContextLength}, int_options);
  metadata.slot_mapping = torch::full({1}, -1, int_options);
  metadata.max_seq_len = kContextLength;
  metadata.compute_dtype = "float";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  torch::Tensor unused_key = torch::empty({1, 1, kHeadSize}, options);
  torch::Tensor unused_value;

  auto [output, output_lse] = attention->forward(metadata,
                                                 query,
                                                 unused_key,
                                                 unused_value,
                                                 kv_cache,
                                                 /*return_lse=*/true);
  Device(device).synchronize_default_stream();

  ASSERT_TRUE(output_lse.has_value());
  EXPECT_EQ(output_lse->scalar_type(), torch::kFloat32);
  EXPECT_EQ(output_lse->sizes(), torch::IntArrayRef({1, kNumHeads, 1}));
  torch::Tensor query_float = query.to(torch::kFloat32).squeeze(/*dim=*/0);
  torch::Tensor keys_float = cache_tensor.index({0, 0})
                                 .to(torch::kFloat32)
                                 .slice(/*dim=*/0, 0, kContextLength);
  torch::Tensor expected_lse = torch::logsumexp(
      torch::matmul(query_float, keys_float.transpose(0, 1)) * scale,
      /*dim=*/-1,
      /*keepdim=*/true);
  EXPECT_TRUE(torch::allclose(output_lse.value(),
                              expected_lse,
                              /*rtol=*/1e-3,
                              /*atol=*/1e-3))
      << "actual=" << output_lse.value().cpu()
      << ", expected=" << expected_lse.cpu();
}

TEST(MluMlaDecodeLseTest, ArtificialShardMergeMatchesFullCacheDecode) {
  constexpr int64_t kNumHeads = 2;
  constexpr int64_t kHeadSize = 576;
  constexpr int64_t kValueHeadSize = 512;
  constexpr int64_t kContextLength = 4;
  constexpr int64_t kShardLength = kContextLength / 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  const torch::Device device(Platform::type_torch(), 0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions int_options = options.dtype(torch::kInt32);
  Attention attention(kNumHeads,
                      kHeadSize,
                      /*num_kv_heads=*/1,
                      kValueHeadSize,
                      /*sliding_window=*/-1,
                      scale,
                      /*use_fused_mla_qkv=*/true,
                      /*enable_lighting_indexer=*/false,
                      /*enable_mla=*/true);
  torch::Tensor query = test::seeded_tensor("dcp_attention_merge/shard_query",
                                            {1, kNumHeads, kHeadSize},
                                            torch::kBFloat16,
                                            device);
  torch::Tensor full_cache =
      test::seeded_tensor("dcp_attention_merge/full_cache",
                          {1, 1, 16, kHeadSize},
                          torch::kBFloat16,
                          device);
  torch::Tensor first_shard = torch::zeros_like(full_cache);
  torch::Tensor second_shard = torch::zeros_like(full_cache);
  first_shard.slice(/*dim=*/2, 0, kShardLength)
      .copy_(full_cache.slice(/*dim=*/2, 0, kShardLength));
  second_shard.slice(/*dim=*/2, 0, kShardLength)
      .copy_(full_cache.slice(/*dim=*/2, kShardLength, kContextLength));

  auto run_decode = [&](const torch::Tensor& cache_tensor,
                        int64_t context_length) {
    KVCache kv_cache(KVCacheTensors{cache_tensor, torch::Tensor()});
    AttentionMetadata metadata;
    metadata.block_table = torch::zeros({1, 1}, int_options);
    metadata.kv_seq_lens = torch::tensor({context_length}, int_options);
    metadata.slot_mapping = torch::full({1}, -1, int_options);
    metadata.max_seq_len = context_length;
    metadata.compute_dtype = "float";
    metadata.is_prefill = false;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    torch::Tensor query_input = query.clone();
    torch::Tensor unused_key = torch::empty({1, 1, kHeadSize}, options);
    torch::Tensor unused_value;
    auto [output, output_lse] = attention->forward(metadata,
                                                   query_input,
                                                   unused_key,
                                                   unused_value,
                                                   kv_cache,
                                                   /*return_lse=*/true);
    CHECK(output_lse.has_value());
    return DcpAttentionResult{output.view({1, 1, kNumHeads, kValueHeadSize}),
                              output_lse.value()};
  };

  const DcpAttentionResult full = run_decode(full_cache, kContextLength);
  const DcpAttentionResult first = run_decode(first_shard, kShardLength);
  const DcpAttentionResult second = run_decode(second_shard, kShardLength);
  const DcpAttentionResult merged =
      merge_dcp_attention_shards(torch::stack({first.output, second.output}),
                                 torch::stack({first.lse, second.lse}));
  Device(device).synchronize_default_stream();

  EXPECT_TRUE(torch::allclose(merged.output.to(torch::kFloat32),
                              full.output.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2));
  EXPECT_TRUE(torch::allclose(merged.lse,
                              full.lse,
                              /*rtol=*/1e-3,
                              /*atol=*/1e-3));
}

TEST(MluMlaDecodeLseTest, SupportsInt8CacheAndUnequalBatchContexts) {
  constexpr int64_t kNumHeads = 2;
  constexpr int64_t kHeadSize = 576;
  constexpr int64_t kValueHeadSize = 512;
  constexpr int64_t kBatchSize = 2;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  const torch::Device device(Platform::type_torch(), 0);
  const torch::TensorOptions options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions int_options = options.dtype(torch::kInt32);
  Attention attention(kNumHeads,
                      kHeadSize,
                      /*num_kv_heads=*/1,
                      kValueHeadSize,
                      /*sliding_window=*/-1,
                      scale,
                      /*use_fused_mla_qkv=*/true,
                      /*enable_lighting_indexer=*/false,
                      /*enable_mla=*/true);
  torch::Tensor query = test::seeded_tensor("dcp_attention_merge/int8_query",
                                            {kBatchSize, kNumHeads, kHeadSize},
                                            torch::kBFloat16,
                                            device);
  torch::Tensor cache_tensor =
      test::seeded_tensor("dcp_attention_merge/int8_cache",
                          {kBatchSize, 1, 16, kHeadSize},
                          torch::kInt8,
                          device);
  torch::Tensor cache_scale =
      torch::full({kBatchSize, 1, 16}, 0.01f, options.dtype(torch::kFloat32));
  KVCache kv_cache(
      QuantizedKVCacheTensors{KVCacheTensors{cache_tensor, torch::Tensor()},
                              cache_scale,
                              torch::Tensor()});

  AttentionMetadata metadata;
  metadata.block_table = torch::tensor({0, 1}, int_options).reshape({2, 1});
  metadata.kv_seq_lens = torch::tensor({2, 4}, int_options);
  metadata.slot_mapping = torch::full({kBatchSize}, -1, int_options);
  metadata.max_seq_len = 4;
  metadata.compute_dtype = "float";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  torch::Tensor unused_key = torch::empty({kBatchSize, 1, kHeadSize}, options);
  torch::Tensor unused_value;

  auto [output, output_lse] = attention->forward(metadata,
                                                 query,
                                                 unused_key,
                                                 unused_value,
                                                 kv_cache,
                                                 /*return_lse=*/true);
  Device(device).synchronize_default_stream();

  ASSERT_TRUE(output_lse.has_value());
  EXPECT_EQ(output.sizes(),
            torch::IntArrayRef({kBatchSize, kNumHeads * kValueHeadSize}));
  EXPECT_EQ(output_lse->sizes(),
            torch::IntArrayRef({kBatchSize, kNumHeads, 1}));
  EXPECT_TRUE(torch::isfinite(output.to(torch::kFloat32)).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(output_lse.value()).all().item<bool>());
}

}  // namespace
}  // namespace xllm::layer
