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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/framework/tokenizer/tokenizer.h"

namespace xllm {

class JsonObjectGrammar;

enum class JsonObjectMaskBuildPhase : uint8_t {
  NORMAL = 0,
  DRAFT = 1,
  TARGET = 2,
};

struct JsonObjectGrammarSnapshot final {
  bool enabled = false;
  bool reasoning_enabled = false;
  std::vector<int32_t> token_ids;
};

class JsonObjectGrammarState final {
 public:
  JsonObjectGrammarState() = default;

  bool can_accept_token(int32_t token_id) const;
  bool accept_token(int32_t token_id);

  bool can_accept_piece(std::string_view piece) const;
  bool accept_piece(std::string_view piece);

  bool is_valid() const { return valid_; }
  bool is_complete() const { return valid_ && root_complete_; }
  bool in_reasoning() const { return reasoning_phase_; }
  bool reasoning_enabled() const { return reasoning_enabled_; }
  bool initialized() const { return grammar_ != nullptr; }
  const JsonObjectGrammar* grammar() const { return grammar_; }
  JsonObjectGrammarSnapshot snapshot() const;
  // Fingerprint of the matcher FSM only (not committed token history). Used for
  // mask caching and debug; identical FSMs share the same mask.
  uint64_t fingerprint() const;

 private:
  friend class JsonObjectGrammar;

  enum class ParseMode : int8_t {
    NONE = 0,
    STRING = 1,
    STRING_ESCAPE = 2,
    STRING_UNICODE = 3,
    NUMBER = 4,
    LITERAL = 5,
  };

  enum class StringRole : int8_t { VALUE = 0, OBJECT_KEY = 1 };

  enum class ContainerType : int8_t { OBJECT = 0, ARRAY = 1 };

  enum class ContainerState : int8_t {
    OBJECT_KEY_OR_END = 0,
    OBJECT_KEY_AFTER_COMMA = 1,
    OBJECT_COLON = 2,
    OBJECT_VALUE = 3,
    OBJECT_COMMA_OR_END = 4,
    ARRAY_VALUE_OR_END = 5,
    ARRAY_VALUE_AFTER_COMMA = 6,
    ARRAY_COMMA_OR_END = 7,
  };

  enum class NumberState : int8_t {
    AFTER_MINUS = 0,
    ZERO = 1,
    INTEGER = 2,
    FRACTION_POINT = 3,
    FRACTION = 4,
    EXPONENT = 5,
    EXPONENT_SIGN = 6,
    EXPONENT_DIGITS = 7,
  };

  struct ContainerFrame {
    ContainerType type = ContainerType::OBJECT;
    ContainerState state = ContainerState::OBJECT_KEY_OR_END;
  };

  JsonObjectGrammarState(const JsonObjectGrammar* grammar,
                         bool reasoning_phase);

  bool consume_character(char character);
  bool consume_string_character(char character);
  bool consume_number_character(char character);
  bool consume_literal_character(char character);
  bool start_value(char character);
  void complete_value();
  bool close_container(ContainerType type);
  bool is_value_delimiter(char character) const;
  void invalidate() { valid_ = false; }
  bool has_complete_number() const;
  // Copy matcher fields for trial accepts without cloning committed_token_ids_.
  void copy_trial_state_from(const JsonObjectGrammarState& other);
  std::string transition_cache_key() const;

  const JsonObjectGrammar* grammar_ = nullptr;
  std::vector<ContainerFrame> containers_;
  ParseMode parse_mode_ = ParseMode::NONE;
  StringRole string_role_ = StringRole::VALUE;
  NumberState number_state_ = NumberState::AFTER_MINUS;
  std::string literal_target_;
  size_t literal_index_ = 0;
  uint8_t unicode_digits_ = 0;
  bool valid_ = true;
  bool root_started_ = false;
  bool root_complete_ = false;
  bool reasoning_phase_ = false;
  bool reasoning_enabled_ = false;
  size_t reasoning_marker_index_ = 0;
  std::vector<int32_t> committed_token_ids_;
};

class JsonObjectGrammar final {
 public:
  JsonObjectGrammar(std::vector<std::string> token_pieces,
                    std::unordered_set<int32_t> stop_token_ids = {},
                    std::vector<int32_t> reasoning_end_token_ids = {});

  static std::shared_ptr<const JsonObjectGrammar> create_from_tokenizer(
      const Tokenizer& tokenizer,
      int32_t eos_token_id,
      const std::unordered_set<int32_t>& stop_token_ids,
      int64_t model_vocab_size,
      const std::vector<int32_t>& reasoning_end_token_ids,
      std::string* error);

  static std::shared_ptr<const JsonObjectGrammar> create_from_tokenizer(
      const Tokenizer& tokenizer,
      int32_t eos_token_id,
      const std::unordered_set<int32_t>& stop_token_ids,
      int64_t model_vocab_size,
      bool reasoning_enabled,
      std::string* error);

  JsonObjectGrammarState initial_state(bool reasoning_phase = false) const;

  JsonObjectGrammarState restore_state(
      const JsonObjectGrammarSnapshot& snapshot) const;

  std::vector<int32_t> allowed_token_ids(
      const JsonObjectGrammarState& state) const;

  // Packed allowed-token bitmask, little-endian words, bit i => token i
  // allowed.
  std::vector<uint32_t> allowed_token_bitmask(
      const JsonObjectGrammarState& state) const;

  torch::Tensor build_filter_mask(
      const JsonObjectGrammarState& state,
      const torch::Device& device = torch::kCPU,
      torch::ScalarType dtype = torch::kFloat32) const;

  // Int32 bitmask row: shape [ceil(vocab/32)]. Preferred compact transfer form.
  torch::Tensor build_filter_bitmask(
      const JsonObjectGrammarState& state,
      const torch::Device& device = torch::kCPU) const;

  size_t vocab_size() const { return token_pieces_.size(); }
  size_t bitmask_num_words() const {
    return (token_pieces_.size() + 31U) / 32U;
  }
  const std::string& token_piece(int32_t token_id) const;

 private:
  friend class JsonObjectGrammarState;
  friend torch::Tensor build_json_object_filter_mask(
      const std::vector<JsonObjectGrammarState>& states,
      const torch::Device& device,
      torch::ScalarType dtype);
  friend torch::Tensor build_json_object_filter_bitmask(
      const std::vector<JsonObjectGrammarState>& states,
      const torch::Device& device,
      JsonObjectMaskBuildPhase phase);

  struct CachedMask final {
    std::vector<uint32_t> bitmask;
    torch::Tensor float_mask_cpu;
  };

  struct FilterMaskCacheEntry final {
    std::shared_ptr<const CachedMask> mask;
    std::list<std::string>::iterator recency_iterator;
  };

  struct FilterMaskCache final {
    std::mutex mutex;
    std::list<std::string> recency;
    std::unordered_map<std::string, FilterMaskCacheEntry> entries;
  };

  torch::Tensor get_cpu_filter_mask(const JsonObjectGrammarState& state) const;
  std::vector<uint32_t> compute_allowed_bitmask(
      const JsonObjectGrammarState& state) const;
  std::shared_ptr<const CachedMask> cached_mask_for_state(
      const JsonObjectGrammarState& state) const;
  static torch::Tensor float_mask_from_bitmask(
      const std::vector<uint32_t>& bitmask,
      size_t vocab_size);

  std::vector<std::string> token_pieces_;
  std::unordered_set<int32_t> stop_token_ids_;
  std::vector<int32_t> reasoning_end_token_ids_;
  // Precomputed additive mask for reasoning: allow all tokens except stops.
  torch::Tensor reasoning_filter_mask_cpu_;
  std::vector<uint32_t> reasoning_bitmask_;
  std::shared_ptr<const CachedMask> reasoning_cached_mask_;

  std::shared_ptr<FilterMaskCache> filter_mask_cache_;
};

torch::Tensor build_json_object_filter_mask(
    const std::vector<JsonObjectGrammarState>& states,
    const torch::Device& device = torch::kCPU,
    torch::ScalarType dtype = torch::kFloat32);

torch::Tensor build_json_object_filter_bitmask(
    const std::vector<JsonObjectGrammarState>& states,
    const torch::Device& device = torch::kCPU,
    JsonObjectMaskBuildPhase phase = JsonObjectMaskBuildPhase::NORMAL);

// Advances each initialized state with its corresponding accepted token.
std::vector<JsonObjectGrammarState> advance_json_object_states(
    const std::vector<JsonObjectGrammarState>& states,
    const std::vector<int32_t>& token_ids);

// Apply a [batch, ceil(vocab/32)] int32 bitmask in-place on logits [batch,
// vocab].
void apply_token_bitmask_inplace(torch::Tensor& logits,
                                 const torch::Tensor& bitmask);

}  // namespace xllm
