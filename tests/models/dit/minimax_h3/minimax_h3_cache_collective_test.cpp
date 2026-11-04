/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core/framework/dit_cache/similarity_residual_cache.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {
namespace {

TEST(H3CacheDiTWorld16Test,
     DistinctLocalMetricsAndEmptyTargetsReachOneDecision) {
  const char* enabled = std::getenv("XLLM_H3_WORLD_TEST");
  if (enabled == nullptr || std::string(enabled) != "1") {
    GTEST_SKIP()
        << "Run the isolated 16-process H3 harness with XLLM_H3_WORLD_TEST=1";
  }
  const char* rank_text = std::getenv("RANK");
  const char* size_text = std::getenv("WORLD_SIZE");
  const char* port_text = std::getenv("XLLM_H3_TEST_PORT");
  ASSERT_NE(rank_text, nullptr);
  ASSERT_NE(size_text, nullptr);
  ASSERT_NE(port_text, nullptr);
  const int32_t rank = std::stoi(rank_text);
  ASSERT_EQ(std::stoi(size_text), 16);
  ASSERT_GE(rank, 0);
  ASSERT_LT(rank, 16);
  const char* local_text = std::getenv("LOCAL_RANK");
  const char* visible_text = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  ASSERT_NE(local_text, nullptr);
  ASSERT_NE(visible_text, nullptr);
  ASSERT_EQ(std::stoi(local_text), 0);
  ASSERT_EQ(std::string(visible_text).find(','), std::string::npos);
  RecordProperty("global_rank", rank);
  RecordProperty("local_rank", 0);
  RecordProperty("visible_device", std::string(visible_text));
  const char* host_text = std::getenv("XLLM_H3_TEST_MASTER_ADDR");
  const std::string host = host_text == nullptr ? "127.0.0.1" : host_text;
  const torch::Device device("npu:0");
  std::unique_ptr<ProcessGroup> world =
      create_process_group(rank,
                           16,
                           16,
                           std::stoi(port_text),
                           false,
                           host,
                           "h3_cachedit_world16_test",
                           device);
  CacheDiTOptions options;
  options.residual_diff_threshold = 0.08F;
  SimilarityResidualCacheState cache(options);
  const torch::Tensor previous =
      torch::full({2, 2},
                  rank + 1.0F,
                  torch::TensorOptions().device(device).dtype(torch::kFloat32));
  const torch::Tensor indices =
      rank == 0
          ? torch::empty(
                {0}, torch::TensorOptions().device(device).dtype(torch::kInt64))
          : torch::zeros(
                {1},
                torch::TensorOptions().device(device).dtype(torch::kInt64));
  ASSERT_FALSE(cache.decide(0, previous, 2, world.get(), indices).hit);
  cache.record_dense(0, previous, previous + 3.0F);
  const float change = rank >= 8 ? 0.1F : 0.001F;
  const torch::Tensor current = previous * (1.0F + change);
  const SimilarityResidualCacheDecision decision =
      cache.decide(1, current, 2, world.get(), indices);
  // Ranks 8..15 have local relative L1=0.1 (>0.08), while others have 0.001;
  // rank0 contributes no values. The raw global sum is 10.035 / 135.
  EXPECT_TRUE(decision.hit);
  EXPECT_NEAR(decision.relative_l1, 10.035F / 135.0F, 1e-6F);
  EXPECT_TRUE(torch::allclose(cache.apply(1, current), current + 3.0F));
  torch::Tensor agreement =
      torch::tensor({decision.hit ? 1 : 0,
                     static_cast<int32_t>(cache.similarity_checks()),
                     static_cast<int32_t>(cache.dense_forwards()),
                     static_cast<int32_t>(cache.cache_hits())},
                    torch::TensorOptions().device(device).dtype(torch::kInt32));
  world->allreduce(agreement);
  EXPECT_TRUE(
      torch::equal(agreement.cpu(), torch::full({4}, 16, torch::kInt32)));
  SimilarityResidualCacheState next_request(options);
  EXPECT_FALSE(next_request.decide(0, current, 2, world.get(), indices).hit);
  EXPECT_EQ(next_request.cache_hits(), 0);
  EXPECT_FALSE(next_request.has_stack_residual());
  EXPECT_EQ(next_request.dense_forwards(), 0);
  EXPECT_EQ(next_request.similarity_checks(), 0);
  const bool request_reset = !next_request.has_stack_residual() &&
                             next_request.cache_hits() == 0 &&
                             next_request.dense_forwards() == 0 &&
                             next_request.similarity_checks() == 0;
  torch::Tensor evidence =
      torch::tensor({indices.numel() == 0 ? 1 : 0, request_reset ? 1 : 0},
                    torch::TensorOptions().device(device).dtype(torch::kInt32));
  world->allreduce(evidence);
  const torch::Tensor host_evidence = evidence.cpu();
  EXPECT_EQ(host_evidence[0].item<int32_t>(), 1);
  EXPECT_EQ(host_evidence[1].item<int32_t>(), 16);
  RecordProperty("cache_hits", cache.cache_hits());
  RecordProperty("dense_forwards", cache.dense_forwards());
  RecordProperty("similarity_checks", cache.similarity_checks());
  RecordProperty("global_hit_votes", agreement.cpu()[0].item<int32_t>());
  RecordProperty("empty_target_ranks", host_evidence[0].item<int32_t>());
  RecordProperty("request_reset_ranks", host_evidence[1].item<int32_t>());
  world->barrier();
}

}  // namespace
}  // namespace xllm
