/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/model/mtp_draft_model_args.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache_estimation.h"

namespace xllm {
namespace {

ModelArgs make_target_args(const std::string& model_type,
                           int32_t num_nextn_predict_layers) {
  ModelArgs args;
  args.model_type(model_type)
      .num_nextn_predict_layers(num_nextn_predict_layers)
      .n_layers(48)
      .layer_types({"linear_attention", "full_attention"})
      .full_attention_interval(4);
  return args;
}

TEST(MtpDraftModelArgsTest, NormalizesQwen35DenseAndMoeAliases) {
  const std::vector<std::pair<std::string, std::string>> model_types = {
      {"qwen3_5", "qwen3_5_mtp"},
      {"qwen3_5_text", "qwen3_5_mtp"},
      {"qwen3_5_moe", "qwen3_5_moe_mtp"},
      {"qwen3_5_moe_text", "qwen3_5_moe_mtp"},
  };

  for (const auto& [target_type, draft_type] : model_types) {
    ModelArgs args = make_target_args(target_type, 2);

    EXPECT_EQ(normalize_mtp_draft_model_args(args, args),
              MtpDraftModelArgsStatus::NORMALIZED);
    EXPECT_EQ(args.model_type(), draft_type);
    EXPECT_EQ(args.n_layers(), 2);
    EXPECT_EQ(args.layer_types(),
              std::vector<std::string>({"full_attention", "full_attention"}));
    EXPECT_EQ(args.full_attention_interval(), 1);
  }
}

TEST(MtpDraftModelArgsTest, PreservesExistingDeepSeekMapping) {
  ModelArgs args = make_target_args("deepseek_v4", 1);

  EXPECT_EQ(normalize_mtp_draft_model_args(args, args),
            MtpDraftModelArgsStatus::NORMALIZED);
  EXPECT_EQ(args.model_type(), "deepseek_v4_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.layer_types(), std::vector<std::string>({"full_attention"}));
  EXPECT_EQ(args.full_attention_interval(), 1);
}

TEST(MtpDraftModelArgsTest, LeavesZeroLayerInputUnchanged) {
  ModelArgs args = make_target_args("qwen3_5", 0);
  const ModelArgs original = args;

  EXPECT_EQ(normalize_mtp_draft_model_args(args, args),
            MtpDraftModelArgsStatus::NOT_APPLICABLE);
  EXPECT_EQ(args.model_type(), original.model_type());
  EXPECT_EQ(args.n_layers(), original.n_layers());
  EXPECT_EQ(args.layer_types(), original.layer_types());
  EXPECT_EQ(args.full_attention_interval(), original.full_attention_interval());
}

TEST(MtpDraftModelArgsTest, LeavesUnsupportedModelInputUnchanged) {
  ModelArgs args = make_target_args("unsupported_model", 2);
  const ModelArgs original = args;

  EXPECT_EQ(normalize_mtp_draft_model_args(args, args),
            MtpDraftModelArgsStatus::UNSUPPORTED);
  EXPECT_EQ(args.model_type(), original.model_type());
  EXPECT_EQ(args.n_layers(), original.n_layers());
  EXPECT_EQ(args.layer_types(), original.layer_types());
  EXPECT_EQ(args.full_attention_interval(), original.full_attention_interval());
}

TEST(MtpDraftModelArgsTest, NormalizedDraftKvSizingMatchesLoadedLayerShape) {
  ModelArgs target_args = make_target_args("qwen3_5", 2);
  target_args.n_layers(8)
      .head_dim(16)
      .layer_types({"linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                    "full_attention"})
      .linear_num_key_heads(2)
      .linear_num_value_heads(2)
      .linear_key_head_dim(4)
      .linear_value_head_dim(8)
      .linear_conv_kernel_dim(3);
  ModelArgs draft_args = target_args;
  ASSERT_EQ(normalize_mtp_draft_model_args(target_args, draft_args),
            MtpDraftModelArgsStatus::NORMALIZED);

  KVCacheEstimateOptions target_options;
  target_options.dtype = torch::kFloat16;
  target_options.kv_cache_dtype = "auto";
  target_options.cache_size_in_bytes = 1024 * 1024;
  target_options.block_size = 16;
  target_options.world_size = 1;
  target_options.n_local_kv_heads = 2;
  target_options.n_local_linear_k_heads = 2;
  target_options.n_local_linear_v_heads = 2;
  target_options.max_seqs_per_batch = 8;

  KVCacheEstimateOptions draft_options = target_options;
  draft_options.is_draft_engine = true;
  const KVCacheCapacity target_capacity =
      estimate_kv_cache_capacity(target_args, target_options);
  const KVCacheCapacity draft_capacity =
      estimate_kv_cache_capacity(draft_args, draft_options);

  EXPECT_EQ(target_capacity.n_layers(), 8);
  EXPECT_EQ(target_capacity.num_full_attention_layers(), 4);
  EXPECT_EQ(target_capacity.num_linear_attention_layers(), 4);
  EXPECT_EQ(draft_args.model_type(), "qwen3_5_mtp");
  EXPECT_EQ(draft_args.n_layers(), 2);
  EXPECT_EQ(draft_args.layer_types(),
            std::vector<std::string>({"full_attention", "full_attention"}));
  EXPECT_EQ(draft_args.full_attention_interval(), 1);
  EXPECT_EQ(draft_capacity.n_layers(), 2);
  EXPECT_EQ(draft_capacity.num_full_attention_layers(), 2);
  EXPECT_EQ(draft_capacity.num_linear_attention_layers(), 0);
}

}  // namespace
}  // namespace xllm
