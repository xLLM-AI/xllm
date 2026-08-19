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

#include "core/runtime/acl_graph_bucket_policy.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(AclGraphBucketPolicyUnitTest, MapsGlobalWarmupBucketsToLocalDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(1, 16, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(2, 16, 4));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(3, 16, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(4, 16, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(12, 64, 4));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(10, 40, 4));
}

TEST(AclGraphBucketPolicyUnitTest, RejectsNonCanonicalSingleDpBatches) {
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(16, 20, 1));
  EXPECT_TRUE(npu::is_acl_graph_warmup_batch_size(20, 20, 1));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(3, 20, 1));
  EXPECT_FALSE(npu::is_acl_graph_warmup_batch_size(12, 20, 1));
}

TEST(AclGraphBucketPolicyUnitTest, AppliesWarmupCapacityInLocalDpUnits) {
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(4, 16, 4, true));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(5, 16, 4, true));
  EXPECT_FALSE(npu::is_acl_graph_decode_capture_allowed(15, 16, 1, false));
  EXPECT_TRUE(npu::is_acl_graph_decode_capture_allowed(15, 16, 1, true));
}

}  // namespace
}  // namespace xllm
