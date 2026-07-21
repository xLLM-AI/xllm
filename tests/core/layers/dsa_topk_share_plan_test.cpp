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

#include "core/layers/common/dsa_topk_share_plan.h"

#include <gtest/gtest.h>

namespace xllm::layer {
namespace {

ModelArgs make_dsa_mtp_args() {
  ModelArgs args;
  args.model_type() = "glm_moe_dsa_mtp";
  args.index_share_for_mtp_iteration() = true;
  args.index_head_dim() = 128;
  args.index_n_heads() = 32;
  args.index_topk() = 2048;
  return args;
}

TEST(DsaTopkSharePlanTest, EnablesCrossStepReuseForMtpModelWithIndexer) {
  EXPECT_TRUE(is_mtp_dsa_topk_reuse_enabled(make_dsa_mtp_args()));
}

TEST(DsaTopkSharePlanTest, DisablesCrossStepReuseWhenConfigFlagIsOff) {
  ModelArgs args = make_dsa_mtp_args();
  args.index_share_for_mtp_iteration() = false;
  EXPECT_FALSE(is_mtp_dsa_topk_reuse_enabled(args));
}

TEST(DsaTopkSharePlanTest, DisablesCrossStepReuseWithoutCompleteIndexer) {
  ModelArgs args = make_dsa_mtp_args();
  args.index_head_dim() = 0;
  EXPECT_FALSE(is_mtp_dsa_topk_reuse_enabled(args));

  args = make_dsa_mtp_args();
  args.index_n_heads() = 0;
  EXPECT_FALSE(is_mtp_dsa_topk_reuse_enabled(args));

  args = make_dsa_mtp_args();
  args.index_topk() = 0;
  EXPECT_FALSE(is_mtp_dsa_topk_reuse_enabled(args));
}

TEST(DsaTopkSharePlanTest, DisablesCrossStepReuseForTargetModel) {
  ModelArgs args = make_dsa_mtp_args();
  args.model_type() = "glm_moe_dsa";
  EXPECT_FALSE(is_mtp_dsa_topk_reuse_enabled(args));
}

}  // namespace
}  // namespace xllm::layer
