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

#include "core/runtime/acl_graph_bucket_policy.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(AclGraphBucketPolicyUnitTest, ConvertsLocalLimitToGlobalDpCapacity) {
  EXPECT_EQ(npu::acl_graph_max_global_batch_size(3000, 64, 32), 2048);
  EXPECT_EQ(npu::acl_graph_max_global_batch_size(64, 64, 32), 64);
  EXPECT_EQ(npu::acl_graph_max_global_batch_size(3000, 16, 32), 512);
}

TEST(AclGraphBucketPolicyUnitTest, MapsGlobalWarmupBucketsToLocalDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/2,
      /*max_batch_size=*/16,
      /*dp_size=*/4,
      /*is_graph_warmup=*/false));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/3,
      /*max_batch_size=*/16,
      /*dp_size=*/4,
      /*is_graph_warmup=*/false));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/10,
      /*max_batch_size=*/40,
      /*dp_size=*/4,
      /*is_graph_warmup=*/false));
}

TEST(AclGraphBucketPolicyUnitTest, RejectsNonCanonicalSingleDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/20,
      /*max_batch_size=*/20,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/12,
      /*max_batch_size=*/20,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false));
}

TEST(AclGraphBucketPolicyUnitTest, AdmitsNoPaddingMtpTokenBuckets) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/12,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*num_speculative_tokens=*/3));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/11,
      /*max_batch_size=*/16,
      /*dp_size=*/1,
      /*is_graph_warmup=*/false,
      /*num_speculative_tokens=*/3));
}

TEST(AclGraphBucketPolicyUnitTest, AdmitsNonDivisibleDpMtpTailBucket) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/3,
      /*max_batch_size=*/7,
      /*dp_size=*/2,
      /*is_graph_warmup=*/false,
      3));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(
      /*batch_size=*/4,
      /*max_batch_size=*/7,
      /*dp_size=*/2,
      /*is_graph_warmup=*/false,
      3));
}

TEST(AclGraphBucketPolicyUnitTest, AppliesWarmupCapacityInLocalDpUnits) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(4, 16, 4, true));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(5, 16, 4, true));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(15, 16, 1, false));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(15, 16, 1, true));
}

}  // namespace
}  // namespace xllm
