/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <ostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {

// ── Shared Utilities ───────────────────────────────────────────────────────

inline std::string to_lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

// ── Quant Method Checks ─────────────────────────────────────────────────────

inline bool is_w8a8_dynamic_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8_dynamic";
}

inline bool is_w8a8_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8";
}

// ── Quantization method identifiers ─────────────────────────────────────────
static const std::string kQuantMethodFp8 = "fp8";
static const std::string kQuantMethodSmoothquant = "smoothquant";
static const std::string kQuantMethodAscendInt4 = "ascend_int4";
static const std::string kQuantMethodAscendInt8 = "ascend_int8";

struct QuantArgs {
  using QuantDescs = std::unordered_map<std::string, std::string>;

  PROPERTY(std::string, quant_method);

  PROPERTY(std::string, quantize_type);
  PROPERTY(std::string, torch_dtype) = "bfloat16";
  // quantization bits
  PROPERTY(int64_t, bits) = 0;
  // MoE routed experts weight bits for DeepSeek-style SmoothQuant mixed W4A8.
  PROPERTY(int64_t, moe_weight_bits) = 8;

  // quantization group size
  PROPERTY(int64_t, group_size) = 0;

  // aka act_order, true results in better quantisation accuracy.
  PROPERTY(bool, desc_act) = false;

  // whether the input is symmetric
  PROPERTY(bool, is_sym) = false;

  // whether activation scheme is dynamic
  PROPERTY(bool, activation_dynamic) = true;

  // whether weights use compressed-tensors W8A8 dynamic key naming convention
  // (weight/weight_scale instead of qweight/per_channel_scale, no smooth)
  PROPERTY(bool, is_compressed_tensors_w8a8_dynamic) = false;

  // FP8 format : e4m3, e5m2
  PROPERTY(std::string, fmt) = "e4m3";

  // weight block size
  PROPERTY(std::vector<int64_t>, weight_block_size) = {};

  // exact module names or regexes prefixed with "re:" that should bypass
  // quantization for compressed-tensors models.
  PROPERTY(std::vector<std::string>, ignored_modules) = {};

  // Optional quantization format version from Ascend
  // quant_model_description.json. For example, W4A8_DYNAMIC version "1.0.0"
  // stores two int4 values packed into int8 before the runtime int32 pack.
  PROPERTY(std::string, quant_version) = "";

  // Optional per-weight quant type map loaded from quant_model_description.json
  // key: full tensor name, e.g. "layers.0.attn.wq_a.weight"
  // value: quant type, e.g. "W8A8_DYNAMIC"
  PROPERTY(QuantDescs, quant_descs) = {};

  // ── Convenience helpers for weight-loading prefix resolution ──────────

  bool is_quantized() const { return !quant_descs_.empty(); }

  // Returns `quant_key` when the model is quantized (quant_descs is
  // non-empty), otherwise returns `fallback`.  Use this to pick between
  // checkpoint key names / prefixes at load time without repeated if/else.
  std::string resolve_prefix(const std::string& quant_key,
                             const std::string& fallback) const {
    return quant_descs_.empty() ? fallback : quant_key;
  }

  // ── Module-level helpers ────────────────────────────────────────────────

  bool should_ignore_module(const std::string& module_name) const {
    for (const auto& pattern : ignored_modules()) {
      if (pattern == module_name) {
        return true;
      }
      if (pattern.size() > 3 && pattern.rfind("re:", 0) == 0) {
        try {
          if (std::regex_match(module_name, std::regex(pattern.substr(3)))) {
            return true;
          }
        } catch (const std::regex_error&) {
        }
      }
    }
    return false;
  }

  QuantArgs for_module(const std::string& module_name) const {
    QuantArgs local_args = *this;
    if (should_ignore_module(module_name)) {
      local_args.quant_method().clear();
      local_args.is_compressed_tensors_w8a8_dynamic() = false;
    }
    return local_args;
  }

  // check if weights can be fused
  bool can_be_fused() const {
    // can't fuse quantized weights if desc_act is true
    return quant_method().empty() || !desc_act();
  }

  // Query quant type by full tensor key.
  std::optional<std::string> get_quant_method(
      const std::string& weight_name) const {
    const auto it = quant_descs_.find(weight_name);
    if (it == quant_descs_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // Query quant type by (prefix, suffix), e.g. ("layers.0.attn.wq_a",
  // "weight") -> lookup "layers.0.attn.wq_a.weight".
  // Also tries adding/removing "model." prefix for compatibility.
  std::optional<std::string> get_quant_method(const std::string& prefix,
                                              const std::string& suffix) const {
    if (suffix.empty()) {
      return std::nullopt;
    }

    std::vector<std::string> candidate_keys;
    if (prefix.empty()) {
      candidate_keys.push_back(suffix);
    } else {
      candidate_keys.push_back(prefix + "." + suffix);
      static const std::string kModelPrefix = "model.";
      if (prefix.size() >= kModelPrefix.size() &&
          prefix.compare(0, kModelPrefix.size(), kModelPrefix) == 0) {
        candidate_keys.push_back(prefix.substr(kModelPrefix.size()) + "." +
                                 suffix);
      } else {
        candidate_keys.push_back(kModelPrefix + prefix + "." + suffix);
      }
    }

    for (const auto& key : candidate_keys) {
      if (auto quant = get_quant_method(key); quant.has_value()) {
        return quant;
      }
    }
    return std::nullopt;
  }

  // Query quant type for merged prefixes; only returns a quant type when all
  // prefixes resolve successfully and share the same type.
  std::optional<std::string> get_quant_method(
      const std::vector<std::string>& prefixes,
      const std::string& suffix) const {
    if (prefixes.empty()) {
      return std::nullopt;
    }

    std::optional<std::string> resolved_quant;
    for (const auto& prefix : prefixes) {
      auto quant = get_quant_method(prefix, suffix);
      if (!quant.has_value()) {
        return std::nullopt;
      }
      if (!resolved_quant.has_value()) {
        resolved_quant = quant;
      } else if (resolved_quant.value() != quant.value()) {
        return std::nullopt;
      }
    }
    return resolved_quant;
  }

  // Query quant type from state_dict prefix + local prefixes.
  // full_prefix = normalize(state_dict.prefix()) + "." +
  // normalize(prefixes[i]). Returns quant_method when all queried prefixes
  // resolve to the same value. Returns nullopt when all queried prefixes are
  // unresolved. Fails when only part of prefixes resolve, or resolved values
  // are inconsistent.
  std::optional<std::string> get_quant_method_from_prefixes(
      const StateDict& state_dict,
      const std::vector<std::string>& prefixes) const {
    auto to_lower = [](std::string value) {
      std::transform(
          value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
          });
      return value;
    };
    auto normalize_prefix = [](std::string value) {
      while (!value.empty() && value.back() == '.') {
        value.pop_back();
      }
      return value;
    };
    auto join_prefix = [&](const std::string& lhs, const std::string& rhs) {
      if (lhs.empty()) {
        return rhs;
      }
      if (rhs.empty()) {
        return lhs;
      }
      return lhs + "." + rhs;
    };

    const std::string base_prefix =
        normalize_prefix(std::string(state_dict.prefix()));
    std::vector<std::string> query_prefixes;
    if (prefixes.empty()) {
      if (!base_prefix.empty()) {
        query_prefixes.push_back(base_prefix);
      }
    } else {
      query_prefixes.reserve(prefixes.size());
      for (const auto& prefix : prefixes) {
        std::string full_prefix =
            join_prefix(base_prefix, normalize_prefix(prefix));
        if (!full_prefix.empty()) {
          query_prefixes.push_back(std::move(full_prefix));
        }
      }
    }
    if (query_prefixes.empty()) {
      return std::nullopt;
    }

    std::optional<std::string> unresolved_prefix;
    std::optional<std::pair<std::string, std::string>> resolved;
    for (const auto& prefix : query_prefixes) {
      auto quant = get_quant_method(prefix, "weight");
      if (!quant.has_value()) {
        if (resolved.has_value()) {
          LOG(FATAL) << "Inconsistent quant_method for prefixes: prefix '"
                     << prefix << "' unresolved, but prefix '"
                     << resolved->first << "' resolved to '" << resolved->second
                     << "'.";
        }
        if (!unresolved_prefix.has_value()) {
          unresolved_prefix = prefix;
        }
        continue;
      }
      const std::string normalized_quant = to_lower(quant.value());
      if (unresolved_prefix.has_value()) {
        LOG(FATAL) << "Inconsistent quant_method for prefixes: prefix '"
                   << unresolved_prefix.value() << "' unresolved, but prefix '"
                   << prefix << "' resolved to '" << normalized_quant << "'.";
      }
      if (!resolved.has_value()) {
        resolved = std::make_pair(prefix, normalized_quant);
        continue;
      }
      CHECK_EQ(resolved->second, normalized_quant)
          << "Inconsistent quant_method for prefixes: prefix '" << prefix
          << "' resolved to '" << normalized_quant << "', but prefix '"
          << resolved->first << "' resolved to '" << resolved->second << "'.";
    }
    if (!resolved.has_value()) {
      return std::nullopt;
    }
    return resolved->second;
  }
};

inline std::ostream& operator<<(std::ostream& os, const QuantArgs& args) {
  os << "QuantArgs: [";
  os << "quant_method: " << args.quant_method();
  os << ", quantize_type: " << args.quantize_type();
  os << ", bits: " << args.bits();
  os << ", moe_weight_bits: " << args.moe_weight_bits();
  os << ", group_size: " << args.group_size();
  os << ", desc_act: " << args.desc_act();
  os << ", is_sym: " << args.is_sym();
  os << ", activation_dynamic: " << args.activation_dynamic();
  os << ", fmt: " << args.fmt();
  os << ", ignored_modules: " << args.ignored_modules().size();
  os << ", quant_version: " << args.quant_version();
  os << ", quant_desc_count: " << args.quant_descs().size();
  os << "]";
  return os;
}

}  // namespace xllm
