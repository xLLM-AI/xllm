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

#pragma once

#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

#include "core/framework/model/model_args.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

// Configuration only: registering a loader must not advertise runnable OLMo2.
REGISTER_MODEL_ARGS_LOADER(olmo2, [](const JsonReader& json, ModelArgs* args) {
  if (args == nullptr) {
    LOG(ERROR) << "OLMo2 ModelArgs must not be null";
    return false;
  }
  ModelArgs parsed;
  parsed.model_type("olmo2");
  auto read = [&json]<typename T>(
                  const char* key, T* value, bool required = false) {
    const auto* node = json.resolve(key);
    if (node == nullptr || node->is_null()) {
      if (required) {
        LOG(ERROR) << "OLMo2 requires config field: " << key;
      }
      return !required;
    }
    bool valid = false;
    if constexpr (std::is_same_v<T, bool>) {
      valid = node->is_boolean();
    } else if constexpr (std::is_integral_v<T>) {
      // Compare in the stored integer domain: JSON signed/unsigned comparison
      // can otherwise wrap UINT64_MAX to the valid absent-token sentinel -1.
      if (node->is_number_unsigned()) {
        valid = node->get<uint64_t>() <=
                static_cast<uint64_t>(std::numeric_limits<T>::max());
      } else if (node->is_number_integer()) {
        const int64_t number = node->get<int64_t>();
        valid = number >= std::numeric_limits<T>::min() &&
                number <= std::numeric_limits<T>::max();
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      valid = node->is_number() && std::isfinite(node->get<double>()) &&
              std::abs(node->get<double>()) <= std::numeric_limits<T>::max();
    } else {
      valid = node->is_string();
    }
    if (!valid) {
      LOG(ERROR) << "Invalid OLMo2 config field: " << key;
      return false;
    }
    *value = node->get<T>();
    return true;
  };

  // Checkpoint dimensions are required; HF's architecture defaults must not
  // silently turn a truncated checkpoint config into a different model.
  if (!read("hidden_size", &parsed.hidden_size(), /*required=*/true) ||
      !read("num_hidden_layers", &parsed.n_layers(), /*required=*/true) ||
      !read("num_attention_heads", &parsed.n_heads(), /*required=*/true) ||
      !read("intermediate_size",
            &parsed.intermediate_size(),
            /*required=*/true) ||
      !read("vocab_size", &parsed.vocab_size(), /*required=*/true) ||
      !read("max_position_embeddings",
            &parsed.max_position_embeddings(),
            /*required=*/true)) {
    return false;
  }
  if (parsed.hidden_size() <= 0 || parsed.n_layers() <= 0 ||
      parsed.n_heads() <= 0 || parsed.intermediate_size() <= 0 ||
      parsed.vocab_size() <= 0 || parsed.max_position_embeddings() <= 0 ||
      parsed.hidden_size() % parsed.n_heads() != 0) {
    LOG(ERROR)
        << "OLMo2 requires positive dimensions and hidden_size divisible "
           "by num_attention_heads";
    return false;
  }

  int64_t kv_heads = parsed.n_heads();
  parsed.head_dim(parsed.hidden_size() / parsed.n_heads());
  if (!read("num_key_value_heads", &kv_heads) ||
      !read("head_dim", &parsed.head_dim()) || kv_heads <= 0 ||
      parsed.n_heads() % kv_heads != 0 || parsed.head_dim() <= 0 ||
      parsed.head_dim() != parsed.hidden_size() / parsed.n_heads() ||
      parsed.head_dim() % 2 != 0) {
    LOG(ERROR) << "Invalid OLMo2 attention head configuration";
    return false;
  }
  parsed.n_kv_heads(kv_heads);

  // Defaults follow Olmo2Config; explicit null special tokens use -1.
  parsed.hidden_act("silu");
  parsed.rms_norm_eps(1e-5f);
  parsed.rope_theta(10000.0f);
  parsed.bos_token_id(-1);
  parsed.pad_token_id(1);
  parsed.eos_token_id(50279);
  if (!read("hidden_act", &parsed.hidden_act()) ||
      !read("rms_norm_eps", &parsed.rms_norm_eps()) ||
      !read("rope_theta", &parsed.rope_theta()) ||
      !read("rope_parameters.rope_theta", &parsed.rope_theta()) ||
      !read("tie_word_embeddings", &parsed.tie_word_embeddings()) ||
      !read("attention_bias", &parsed.attention_bias()) ||
      !read("attention_dropout", &parsed.attention_dropout())) {
    return false;
  }
  // Only absent/null aliases mean unspecified; explicit empty strings are
  // invalid.
  for (const char* key : {"torch_dtype", "dtype"}) {
    const auto* node = json.resolve(key);
    if (node == nullptr || node->is_null()) {
      continue;
    }
    if (!read(key, &parsed.dtype()) || parsed.dtype().empty()) {
      LOG(ERROR) << "Invalid OLMo2 dtype field: " << key;
      return false;
    }
  }
  if (parsed.hidden_act().empty() || parsed.rms_norm_eps() <= 0 ||
      parsed.rope_theta() <= 0 || parsed.attention_dropout() < 0 ||
      parsed.attention_dropout() >= 1 ||
      (!parsed.dtype().empty() && parsed.dtype() != "float16" &&
       parsed.dtype() != "bfloat16" && parsed.dtype() != "float32")) {
    LOG(ERROR) << "Invalid OLMo2 activation, norm, RoPE, dropout or dtype";
    return false;
  }
  parsed.qkv_bias(parsed.attention_bias());
  const auto* rope_scaling = json.resolve("rope_scaling");
  if (rope_scaling != nullptr && !rope_scaling->is_null()) {
    LOG(ERROR) << "OLMo2 config loader currently requires unscaled RoPE";
    return false;
  }
  const auto* rope_parameters = json.resolve("rope_parameters");
  if (rope_parameters != nullptr && !rope_parameters->is_null()) {
    std::string rope_type = "default";
    // Match Transformers: the canonical rope_type overrides the legacy type.
    // A default type must not silently discard explicitly supplied scaling.
    if (!rope_parameters->is_object() ||
        !read("rope_parameters.type", &rope_type) ||
        !read("rope_parameters.rope_type", &rope_type) ||
        rope_type != "default" ||
        (rope_parameters->contains("factor") &&
         !rope_parameters->at("factor").is_null()) ||
        (rope_parameters->contains("short_factor") &&
         !rope_parameters->at("short_factor").is_null()) ||
        (rope_parameters->contains("long_factor") &&
         !rope_parameters->at("long_factor").is_null()) ||
        (rope_parameters->contains("original_max_position_embeddings") &&
         !rope_parameters->at("original_max_position_embeddings").is_null())) {
      LOG(ERROR) << "OLMo2 config loader requires unscaled rope_parameters";
      return false;
    }
  }

  for (const char* key : {"bos_token_id", "pad_token_id"}) {
    int32_t token_id = std::string(key) == "bos_token_id"
                           ? parsed.bos_token_id()
                           : parsed.pad_token_id();
    const auto* node = json.resolve(key);
    if (node != nullptr && node->is_null()) {
      token_id = -1;
    } else if (!read(key, &token_id)) {
      return false;
    }
    if (token_id < -1 || token_id >= parsed.vocab_size()) {
      LOG(ERROR) << "OLMo2 token ID outside vocabulary: " << key;
      return false;
    }
    if (std::string(key) == "bos_token_id") {
      parsed.bos_token_id(token_id);
    } else {
      parsed.pad_token_id(token_id);
    }
  }
  const auto* eos = json.resolve("eos_token_id");
  if (eos != nullptr && eos->is_null()) {
    parsed.eos_token_id(-1);
  } else if (eos != nullptr && eos->is_array()) {
    if (eos->empty()) {
      LOG(ERROR) << "OLMo2 eos_token_id list must not be empty";
      return false;
    }
    parsed.eos_token_id_vec().reserve(eos->size());
    for (const auto& token : *eos) {
      if (!token.is_number_integer() || token < 0 ||
          token >= parsed.vocab_size() ||
          token > std::numeric_limits<int32_t>::max()) {
        LOG(ERROR) << "Invalid OLMo2 eos_token_id list";
        return false;
      }
      parsed.eos_token_id_vec().emplace_back(token.get<int32_t>());
    }
    parsed.eos_token_id(parsed.eos_token_id_vec().front());
  } else if (!read("eos_token_id", &parsed.eos_token_id())) {
    return false;
  }
  if (parsed.eos_token_id() < -1 ||
      parsed.eos_token_id() >= parsed.vocab_size()) {
    LOG(ERROR) << "OLMo2 EOS token ID outside vocabulary";
    return false;
  }
  if (parsed.eos_token_id() >= 0) {
    parsed.stop_token_ids().insert(parsed.eos_token_id());
  }
  parsed.stop_token_ids().insert(parsed.eos_token_id_vec().begin(),
                                 parsed.eos_token_id_vec().end());
  *args = std::move(parsed);
  return true;
});

}  // namespace
}  // namespace xllm
