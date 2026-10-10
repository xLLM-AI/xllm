/* Copyright 2025-2026 The xLLM Authors.
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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <ostream>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"

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
static const std::string kQuantMethodW8a8Dynamic = "w8a8_dynamic";

struct CompressedQuantGroup {
  std::vector<std::string> targets;
  int64_t bits = 8;
  int64_t group_size = 0;
  bool preserve_smooth = false;
};

class QuantArgs final {
 private:
  struct RegexCache {
    std::unordered_map<std::string, std::shared_ptr<const std::regex>> patterns;
    std::unordered_set<std::string> invalid;
    std::deque<std::string> order;
  };

  static constexpr size_t kRegexCacheLimit = 128;

  std::shared_ptr<const std::regex> cached_regex(
      const std::string& expression) const {
    if (regex_cache_ == nullptr) {
      regex_cache_ = std::make_shared<RegexCache>();
    }
    if (const auto it = regex_cache_->patterns.find(expression);
        it != regex_cache_->patterns.end()) {
      return it->second;
    }
    if (regex_cache_->invalid.contains(expression)) {
      return nullptr;
    }
    if (regex_cache_->order.size() >= kRegexCacheLimit) {
      const std::string& evicted = regex_cache_->order.front();
      regex_cache_->patterns.erase(evicted);
      regex_cache_->invalid.erase(evicted);
      regex_cache_->order.pop_front();
    }
    regex_cache_->order.push_back(expression);
    try {
      auto pattern = std::make_shared<const std::regex>(expression);
      regex_cache_->patterns.emplace(expression, pattern);
      return pattern;
    } catch (const std::regex_error&) {
      regex_cache_->invalid.emplace(expression);
      return nullptr;
    }
  }

  bool matches_module_target_cached(const std::string& module_name,
                                    const std::string& target,
                                    bool match_prefix = true) const {
    if (!match_prefix && target == module_name) {
      return true;
    }
    if (target.starts_with("re:") && (match_prefix || target.size() > 3)) {
      const auto pattern = cached_regex(target.substr(3));
      if (pattern == nullptr) {
        return false;
      }
      return match_prefix
                 ? std::regex_search(module_name,
                                     *pattern,
                                     std::regex_constants::match_continuous)
                 : std::regex_match(module_name, *pattern);
    }
    return target == module_name;
  }

  mutable std::shared_ptr<RegexCache> regex_cache_ =
      std::make_shared<RegexCache>();

 public:
  using QuantDescs = std::unordered_map<std::string, std::string>;

  PROPERTY(std::string, quant_method);

  PROPERTY(std::string, quantize_type);
  PROPERTY(std::string, torch_dtype) = "bfloat16";
  // quantization bits
  PROPERTY(int64_t, bits) = 0;
  // Weight bits for routed MoE experts.
  PROPERTY(int64_t, moe_weight_bits) = 8;
  // Whether per-group quantization applies only to routed experts.
  PROPERTY(bool, only_expert_per_group) = false;

  // quantization group size
  PROPERTY(int64_t, group_size) = 0;

  // aka act_order, true results in better quantisation accuracy.
  PROPERTY(bool, desc_act) = false;

  // whether the input is symmetric
  PROPERTY(bool, is_sym) = false;

  // whether activation scheme is dynamic
  PROPERTY(bool, activation_dynamic) = true;

  // Whether integer weights with dynamic A8 use compressed-tensors keys
  // (weight/weight_scale instead of qweight/per_channel_scale).
  PROPERTY(bool, is_compressed_tensors_w8a8_dynamic) = false;

  // Smooth is a per-module checkpoint property, independent of weight bits.
  PROPERTY(bool, preserve_smooth) = false;
  PROPERTY(std::vector<CompressedQuantGroup>, compressed_groups) = {};

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

  // ── Module-level helpers ────────────────────────────────────────────────

  bool should_ignore_module(const std::string& module_name) const {
    return std::any_of(ignored_modules().begin(),
                       ignored_modules().end(),
                       [this, &module_name](const std::string& target) {
                         return matches_module_target_cached(
                             module_name, target, /*match_prefix=*/false);
                       });
  }

  static bool matches_module_target(const std::string& module_name,
                                    const std::string& target,
                                    bool match_prefix = true) {
    // Legacy ignore entries preserve exact-match precedence.
    if (!match_prefix && target == module_name) {
      return true;
    }
    if (target.starts_with("re:") && (match_prefix || target.size() > 3)) {
      try {
        const std::regex pattern(target.substr(3));
        return match_prefix
                   ? std::regex_search(module_name,
                                       pattern,
                                       std::regex_constants::match_continuous)
                   : std::regex_match(module_name, pattern);
      } catch (const std::regex_error&) {
        return false;
      }
    }
    return target == module_name;
  }

  static bool matches_any_target(const std::string& module_name,
                                 const std::vector<std::string>& targets) {
    return std::any_of(targets.begin(),
                       targets.end(),
                       [&module_name](const std::string& target) {
                         return matches_module_target(module_name, target);
                       });
  }

  bool matches_any_target_cached(
      const std::string& module_name,
      const std::vector<std::string>& targets) const {
    return std::any_of(targets.begin(),
                       targets.end(),
                       [this, &module_name](const std::string& target) {
                         return matches_module_target_cached(module_name,
                                                             target);
                       });
  }

  const CompressedQuantGroup* module_quant_scheme(
      const std::string& module_name) const {
    if (should_ignore_module(module_name)) {
      return nullptr;
    }
    const CompressedQuantGroup* resolved = nullptr;
    for (const auto& group : compressed_groups()) {
      const bool matches =
          std::find(group.targets.begin(), group.targets.end(), "Linear") !=
              group.targets.end() ||
          matches_any_target_cached(module_name, group.targets);
      if (!matches) {
        continue;
      }
      CHECK(resolved == nullptr ||
            (resolved->bits == group.bits &&
             resolved->group_size == group.group_size &&
             resolved->preserve_smooth == group.preserve_smooth))
          << "Conflicting compressed-tensors groups for " << module_name;
      resolved = &group;
    }
    return resolved;
  }

  QuantArgs for_module(const std::string& module_name) const {
    QuantArgs local_args = *this;
    const CompressedQuantGroup* scheme = compressed_groups().empty()
                                             ? nullptr
                                             : module_quant_scheme(module_name);
    if (scheme != nullptr) {
      local_args.bits() = scheme->bits;
      local_args.moe_weight_bits() = scheme->bits;
      local_args.group_size() = scheme->group_size;
      local_args.preserve_smooth() = scheme->preserve_smooth;
    } else if (!compressed_groups().empty() ||
               should_ignore_module(module_name)) {
      local_args.quant_method().clear();
      local_args.is_compressed_tensors_w8a8_dynamic() = false;
      local_args.preserve_smooth() = false;
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

  // Query quant type from a checkpoint prefix + local prefixes.
  // full_prefix = normalize(checkpoint_prefix) + "." +
  // normalize(prefixes[i]). Returns quant_method when all queried prefixes
  // resolve to the same value. Returns nullopt when all queried prefixes are
  // unresolved. Fails when only part of prefixes resolve, or resolved values
  // are inconsistent.
  std::optional<std::string> get_quant_method_from_prefixes(
      std::string_view checkpoint_prefix,
      const std::vector<std::string>& prefixes) const {
    auto normalize_prefix = [](std::string value) {
      while (!value.empty() && value.back() == '.') {
        value.pop_back();
      }
      return value;
    };
    const std::string base_prefix =
        normalize_prefix(std::string(checkpoint_prefix));
    std::vector<std::string> query_prefixes;
    if (prefixes.empty()) {
      if (!base_prefix.empty()) {
        query_prefixes.push_back(base_prefix);
      }
    } else {
      query_prefixes.reserve(prefixes.size());
      for (const auto& prefix : prefixes) {
        std::string full_prefix = normalize_prefix(prefix);
        if (!base_prefix.empty()) {
          full_prefix = full_prefix.empty() ? base_prefix
                                            : base_prefix + "." + full_prefix;
        }
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
      const std::string normalized_quant = to_lower_copy(quant.value());
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
  os << ", only_expert_per_group: " << args.only_expert_per_group();
  os << ", group_size: " << args.group_size();
  os << ", desc_act: " << args.desc_act();
  os << ", is_sym: " << args.is_sym();
  os << ", activation_dynamic: " << args.activation_dynamic();
  os << ", preserve_smooth: " << args.preserve_smooth();
  os << ", compressed_groups: " << args.compressed_groups().size();
  os << ", fmt: " << args.fmt();
  os << ", ignored_modules: " << args.ignored_modules().size();
  os << ", quant_version: " << args.quant_version();
  os << ", quant_desc_count: " << args.quant_descs().size();
  os << "]";
  return os;
}

}  // namespace xllm
