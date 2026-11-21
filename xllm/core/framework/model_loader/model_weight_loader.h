/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/framework/state_dict/state_dict.h"

namespace xllm {

template <typename ModelType, typename Head>
void load_model_weights(
    ModelType& model,
    const std::vector<std::unique_ptr<StateDict>>& state_dicts,
    Head& lm_head,
    const std::string& language_prefix,
    bool encoder_embedding_mode = false) {
  constexpr bool has_vision = requires { model.vision_model(); };
  auto* language_model = [&] {
    if constexpr (has_vision) {
      return encoder_embedding_mode ? nullptr : &model.language_model();
    } else {
      return &model;
    }
  }();

  auto load_language_shard = [&](const StateDict& state_dict,
                                 const std::string& weight_prefix) {
    if constexpr (requires {
                    lm_head->load_state_dict(language_model->lm_head_weights(
                        state_dict, weight_prefix));
                  }) {
      if (!lm_head.is_empty()) {
        lm_head->load_state_dict(
            language_model->lm_head_weights(state_dict, weight_prefix));
      }
    } else {
      auto weight = language_model->lm_head_weight(state_dict);
      if (weight.defined()) {
        lm_head->weight = weight.to(lm_head->weight.device())
                              .to(lm_head->weight.scalar_type());
      }
    }
    language_model->load_body_weights(state_dict, weight_prefix);
  };

  for (const auto& state_dict : state_dicts) {
    if constexpr (has_vision) {
      model.vision_model().load_state_dict(
          state_dict->get_dict_with_prefix(ModelType::kVisionWeightPrefix));
      if (!language_model) {
        continue;
      }
    }
    if constexpr (requires { language_model->kNormalizeCheckpoint; }) {
      load_language_shard(
          language_model->kNormalizeCheckpoint(*state_dict, language_prefix),
          {});
    } else {
      load_language_shard(*state_dict, language_prefix);
    }
  }
  if (!language_model) {
    return;
  }
  if constexpr (requires {
                  language_model->verify_model_weights(language_prefix);
                }) {
    language_model->verify_model_weights(language_prefix);
  }
  if constexpr (requires {
                  language_model->verify_lm_head(lm_head->is_weight_loaded());
                }) {
    if (!lm_head.is_empty()) {
      language_model->verify_lm_head(lm_head->is_weight_loaded());
    }
  }
}

}  // namespace xllm
