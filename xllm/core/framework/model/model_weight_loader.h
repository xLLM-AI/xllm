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

#include <glog/logging.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/lm_head.h"

namespace xllm {

void load_draft_token_offsets(const StateDict& state_dict,
                              torch::Tensor& offsets);

template <typename ModelType, typename Head>
torch::Tensor load_model_weights(
    ModelType& model,
    ModelLoader& loader,
    Head& lm_head,
    const ModelContext* lm_head_context = nullptr) {
  constexpr bool has_vision = requires { model.vision_model(); };
  const auto [language_model, prefix] = [&] {
    if constexpr (has_vision) {
      return std::pair{
          model.encoder_embedding_mode() ? nullptr : &*model.language_model(),
          ModelType::kLanguageWeightPrefix};
    } else {
      return std::pair{&model, ModelType::kWeightPrefix};
    }
  }();
  const std::string language_prefix = prefix;

  using LanguageModel = std::remove_pointer_t<decltype(language_model)>;
  constexpr bool normalize = requires { LanguageModel::kNormalizeCheckpoint; };
  const std::string weight_prefix = normalize ? "" : language_prefix;
  constexpr auto checkpoint_prefix = []() -> const char* {
    if constexpr (requires { LanguageModel::kDecoderCheckpointPrefix; }) {
      return LanguageModel::kDecoderCheckpointPrefix;
    } else {
      return nullptr;
    }
  }();
  const std::string decoder_prefix = checkpoint_prefix ? checkpoint_prefix : "";
  const auto decoder_aliases =
      checkpoint_prefix ? std::vector<std::string>{}
                        : std::vector<std::string>{"model.language_model.",
                                                   "language_model.model.",
                                                   weight_prefix,
                                                   "model.",
                                                   ""};

  auto load_shard = [&](const StateDict& state_dict) {
    if constexpr (std::is_same_v<Head, layer::LmHead>) {
      if (lm_head_context != nullptr || !lm_head.is_empty()) {
        auto weights = [&] {
          if constexpr (requires {
                          language_model->select_output_weights(state_dict);
                        }) {
            return language_model->select_output_weights(state_dict);
          } else if constexpr (requires {
                                 language_model->select_output_weights(
                                     state_dict, weight_prefix);
                               }) {
            return language_model->select_output_weights(state_dict,
                                                         weight_prefix);
          } else if constexpr (requires {
                                 LanguageModel::kOutputWeightPrefixes;
                               }) {
            return state_dict.get_dict_with_prefix(
                /*prefixes=*/LanguageModel::kOutputWeightPrefixes);
          } else {
            const std::vector<std::string> prefixes =
                language_model->tie_word_embeddings()
                    ? std::vector<std::string>{weight_prefix + "embed_tokens.",
                                               "model.embed_tokens.",
                                               "embed_tokens.",
                                               "embed."}
                    : std::vector<std::string>{"lm_head.",
                                               "model.lm_head.",
                                               "model.head.",
                                               "head.",
                                               weight_prefix + "lm_head.",
                                               weight_prefix + "head.",
                                               weight_prefix};
            return state_dict.get_dict_with_prefix(prefixes);
          }
        }();
        if (lm_head.is_empty() && weights.has(/*tensor_name=*/"weight")) {
          lm_head = layer::LmHead(*lm_head_context);
          lm_head->eval();
        }
        if (!lm_head.is_empty()) {
          lm_head->load_state_dict(weights);
        }
      }
    } else {
      auto weight = language_model->lm_head_weight(state_dict);
      if (weight.defined()) {
        lm_head->weight = weight.to(lm_head->weight.options());
      }
    }
    if constexpr (checkpoint_prefix == nullptr) {
      language_model->load_state_dict(
          state_dict.get_dict_with_prefix(decoder_aliases));
    } else if constexpr (std::string_view(checkpoint_prefix).empty()) {
      language_model->load_state_dict(state_dict);
    } else {
      language_model->load_state_dict(
          state_dict.get_dict_with_prefix(decoder_prefix));
    }
  };

  torch::Tensor draft_token_offsets;
  for (const auto& state_dict : loader.get_state_dicts()) {
    if constexpr (has_vision) {
      model.vision_model().load_state_dict(
          state_dict->get_dict_with_prefix(/*prefix=*/ModelType::kVisionWeightPrefix));
      if (!language_model) {
        continue;
      }
    } else {
      load_draft_token_offsets(*state_dict, draft_token_offsets);
    }
    if constexpr (normalize) {
      load_shard(
          LanguageModel::kNormalizeCheckpoint(*state_dict, language_prefix));
    } else {
      load_shard(*state_dict);
    }
  }
  if (!language_model) {
    return draft_token_offsets;
  }
  if constexpr (requires {
                  language_model->verify_model_weights(language_prefix);
                }) {
    language_model->verify_model_weights(language_prefix);
  }
  if constexpr (requires { LanguageModel::kMissingLmHeadError; }) {
    if (!lm_head.is_empty()) {
      CHECK(lm_head->is_weight_loaded()) << LanguageModel::kMissingLmHeadError;
    }
  }
  return draft_token_offsets;
}

}  // namespace xllm
