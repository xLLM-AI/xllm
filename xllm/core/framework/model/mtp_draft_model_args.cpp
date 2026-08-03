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

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {
namespace {

const std::unordered_map<std::string, std::string> kModelTypeToMtpType = {
    {"deepseek_v3", "deepseek_v3_mtp"},
    {"deepseek_v32", "deepseek_v32_mtp"},
    {"deepseek_v4", "deepseek_v4_mtp"},
    {"glm_moe_dsa", "glm_moe_dsa_mtp"},
    {"joyai_llm_flash", "joyai_llm_flash_mtp"},
    {"mimo", "mimo_mtp"},
    {"qwen3_5", "qwen3_5_mtp"},
    {"qwen3_5_text", "qwen3_5_mtp"},
    {"qwen3_5_moe", "qwen3_5_moe_mtp"},
    {"qwen3_5_moe_text", "qwen3_5_moe_mtp"},
};

}  // namespace

MtpDraftModelArgsStatus normalize_mtp_draft_model_args(
    const ModelArgs& target_args,
    ModelArgs& draft_args) {
  const int32_t mtp_layers = target_args.num_nextn_predict_layers();
  if (mtp_layers <= 0) {
    return MtpDraftModelArgsStatus::NOT_APPLICABLE;
  }

  const auto model_type_it = kModelTypeToMtpType.find(target_args.model_type());
  if (model_type_it == kModelTypeToMtpType.end()) {
    return MtpDraftModelArgsStatus::UNSUPPORTED;
  }

  ModelArgs normalized_args = target_args;
  normalized_args.model_type(model_type_it->second)
      .n_layers(mtp_layers)
      .layer_types(std::vector<std::string>(static_cast<size_t>(mtp_layers),
                                            "full_attention"))
      .full_attention_interval(1);
  draft_args = std::move(normalized_args);
  return MtpDraftModelArgsStatus::NORMALIZED;
}

}  // namespace xllm
