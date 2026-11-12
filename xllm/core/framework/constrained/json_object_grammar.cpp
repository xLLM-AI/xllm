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

#include "core/framework/constrained/json_object_grammar.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <optional>

#include "core/common/metrics.h"

namespace xllm {
namespace {

constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr size_t kMaxFilterMaskCacheEntries = 64;

uint64_t hash_byte(uint64_t hash, uint8_t value) {
  return (hash ^ value) * kFnvPrime;
}

uint64_t hash_uint64(uint64_t hash, uint64_t value) {
  for (int32_t byte_index = 0; byte_index < 8; ++byte_index) {
    const int32_t shift = byte_index * 8;
    hash = hash_byte(
        hash,
        static_cast<uint8_t>((value >> shift) & static_cast<uint64_t>(0xFF)));
  }
  return hash;
}

void append_cache_key_byte(std::string* key, uint8_t value) {
  key->push_back(static_cast<char>(value));
}

void append_cache_key_uint64(std::string* key, uint64_t value) {
  for (int32_t byte_index = 0; byte_index < 8; ++byte_index) {
    const int32_t shift = byte_index * 8;
    append_cache_key_byte(
        key,
        static_cast<uint8_t>((value >> shift) & static_cast<uint64_t>(0xFF)));
  }
}

bool is_hex_digit(char character) {
  return (character >= '0' && character <= '9') ||
         (character >= 'a' && character <= 'f') ||
         (character >= 'A' && character <= 'F');
}

bool is_json_whitespace(char character) {
  return character == ' ' || character == '\n' || character == '\r' ||
         character == '\t';
}

bool is_json_delimiter(char character) {
  return is_json_whitespace(character) || character == ',' ||
         character == ']' || character == '}';
}

bool is_json_escape(char character) {
  return character == '"' || character == '\\' || character == '/' ||
         character == 'b' || character == 'f' || character == 'n' ||
         character == 'r' || character == 't' || character == 'u';
}

}  // namespace

JsonObjectGrammarState::JsonObjectGrammarState(const JsonObjectGrammar* grammar,
                                               bool reasoning_phase)
    : grammar_(grammar),
      reasoning_phase_(reasoning_phase),
      reasoning_enabled_(reasoning_phase) {}

void JsonObjectGrammarState::copy_state_from(
    const JsonObjectGrammarState& other) {
  grammar_ = other.grammar_;
  containers_ = other.containers_;
  parse_mode_ = other.parse_mode_;
  string_role_ = other.string_role_;
  number_state_ = other.number_state_;
  literal_target_ = other.literal_target_;
  literal_index_ = other.literal_index_;
  unicode_digits_ = other.unicode_digits_;
  valid_ = other.valid_;
  root_started_ = other.root_started_;
  root_complete_ = other.root_complete_;
  reasoning_phase_ = other.reasoning_phase_;
  reasoning_enabled_ = other.reasoning_enabled_;
  reasoning_marker_index_ = other.reasoning_marker_index_;
  // Intentionally leave committed_token_ids_ empty: acceptance never reads it.
}

bool JsonObjectGrammarState::accept_token(int32_t token_id) {
  const JsonObjectGrammarState checkpoint = clone_state();
  const bool accepted = advance(token_id);
  if (!accepted) {
    copy_state_from(checkpoint);
  }
  return accepted;
}

void JsonObjectGrammarState::advance_reasoning(int32_t token_id) {
  const auto& marker = grammar_->reasoning_end_token_ids_;
  if (!marker.empty()) {
    if (token_id != marker[reasoning_marker_index_]) {
      reasoning_marker_index_ = token_id == marker.front() ? 1 : 0;
    } else if (++reasoning_marker_index_ == marker.size()) {
      reasoning_phase_ = false;
      reasoning_marker_index_ = 0;
      containers_.clear();
      parse_mode_ = ParseMode::NONE;
      root_started_ = false;
      root_complete_ = false;
    }
  }
  commit_token(token_id);
}

bool JsonObjectGrammarState::advance(int32_t token_id) {
  if (!valid_ || grammar_ == nullptr ||
      static_cast<size_t>(token_id) >= grammar_->vocab_size()) {
    return false;
  }

  if (grammar_->stop_token_ids_.contains(token_id)) {
    if (reasoning_phase_ || !root_complete_ || parse_mode_ != ParseMode::NONE) {
      return false;
    }
    commit_token(token_id);
    return true;
  }

  if (reasoning_phase_) {
    advance_reasoning(token_id);
    return true;
  }

  if (root_complete_ && parse_mode_ == ParseMode::NONE &&
      !grammar_->stop_token_ids_.empty()) {
    return false;
  }

  const std::string& piece = grammar_->token_pieces_[token_id];
  if (piece.empty() || !accept_piece(piece)) {
    return false;
  }
  commit_token(token_id);
  return true;
}

JsonObjectGrammarSnapshot JsonObjectGrammarState::snapshot() const {
  JsonObjectGrammarSnapshot snapshot;
  snapshot.enabled = initialized();
  snapshot.reasoning_enabled = reasoning_enabled_;
  snapshot.token_ids = committed_token_ids_;
  return snapshot;
}

JsonObjectGrammarState JsonObjectGrammarState::clone_state() const {
  JsonObjectGrammarState state;
  state.copy_state_from(*this);
  return state;
}

uint64_t JsonObjectGrammarState::fingerprint() const {
  // Matcher-only fingerprint: identical FSM states share the same allowed mask
  // regardless of how many tokens were committed to reach this state.
  uint64_t hash = kFnvOffsetBasis;
  hash = hash_uint64(hash, initialized() ? 1U : 0U);
  hash = hash_uint64(hash, valid_ ? 1U : 0U);
  hash = hash_uint64(hash, root_started_ ? 1U : 0U);
  hash = hash_uint64(hash, root_complete_ ? 1U : 0U);
  hash = hash_uint64(hash, reasoning_phase_ ? 1U : 0U);
  hash = hash_uint64(hash, reasoning_enabled_ ? 1U : 0U);
  hash = hash_uint64(hash, static_cast<uint64_t>(reasoning_marker_index_));
  hash = hash_uint64(hash, static_cast<uint64_t>(parse_mode_));
  hash = hash_uint64(hash, static_cast<uint64_t>(string_role_));
  hash = hash_uint64(hash, static_cast<uint64_t>(number_state_));
  hash = hash_uint64(hash, static_cast<uint64_t>(unicode_digits_));
  hash = hash_uint64(hash, static_cast<uint64_t>(literal_index_));
  hash = hash_uint64(hash, static_cast<uint64_t>(literal_target_.size()));
  for (const unsigned char character : literal_target_) {
    hash = hash_byte(hash, character);
  }
  hash = hash_uint64(hash, static_cast<uint64_t>(containers_.size()));
  for (const ContainerFrame& frame : containers_) {
    hash = hash_uint64(hash, static_cast<uint64_t>(frame.type));
    hash = hash_uint64(hash, static_cast<uint64_t>(frame.state));
  }
  return hash;
}

std::string JsonObjectGrammarState::transition_cache_key() const {
  std::string key;
  key.reserve(40 + literal_target_.size() + containers_.size() * 2);
  append_cache_key_byte(&key, valid_ ? 1U : 0U);
  append_cache_key_byte(&key, root_started_ ? 1U : 0U);
  append_cache_key_byte(&key, root_complete_ ? 1U : 0U);
  append_cache_key_byte(&key, reasoning_phase_ ? 1U : 0U);
  append_cache_key_byte(&key, static_cast<uint8_t>(parse_mode_));
  if (reasoning_phase_) {
    append_cache_key_uint64(&key,
                            static_cast<uint64_t>(reasoning_marker_index_));
  }
  switch (parse_mode_) {
    case ParseMode::STRING:
    case ParseMode::STRING_ESCAPE:
      append_cache_key_byte(&key, static_cast<uint8_t>(string_role_));
      break;
    case ParseMode::STRING_UNICODE:
      append_cache_key_byte(&key, static_cast<uint8_t>(string_role_));
      append_cache_key_byte(&key, unicode_digits_);
      break;
    case ParseMode::NUMBER:
      append_cache_key_byte(&key, static_cast<uint8_t>(number_state_));
      break;
    case ParseMode::LITERAL:
      append_cache_key_uint64(&key, static_cast<uint64_t>(literal_index_));
      append_cache_key_uint64(&key,
                              static_cast<uint64_t>(literal_target_.size()));
      key.append(literal_target_);
      break;
    case ParseMode::NONE:
      break;
  }
  append_cache_key_uint64(&key, static_cast<uint64_t>(containers_.size()));
  for (const ContainerFrame& frame : containers_) {
    append_cache_key_byte(&key, static_cast<uint8_t>(frame.type));
    append_cache_key_byte(&key, static_cast<uint8_t>(frame.state));
  }
  return key;
}

bool JsonObjectGrammarState::accept_piece(std::string_view piece) {
  if (!valid_ || grammar_ == nullptr || piece.empty()) {
    return false;
  }
  for (const char character : piece) {
    if (!consume_character(character)) {
      invalidate();
      return false;
    }
  }
  return true;
}

bool JsonObjectGrammarState::consume_character(char character) {
  if (reasoning_phase_) {
    return true;
  }
  if (parse_mode_ == ParseMode::STRING ||
      parse_mode_ == ParseMode::STRING_ESCAPE ||
      parse_mode_ == ParseMode::STRING_UNICODE) {
    return consume_string_character(character);
  }
  if (parse_mode_ == ParseMode::NUMBER) {
    return consume_number_character(character);
  }
  if (parse_mode_ == ParseMode::LITERAL) {
    return consume_literal_character(character);
  }

  if (root_complete_) {
    return is_json_whitespace(character);
  }

  if (containers_.empty()) {
    if (is_json_whitespace(character)) {
      return !root_started_;
    }
    if (!root_started_ && character == '{') {
      root_started_ = true;
      containers_.push_back(
          {ContainerType::OBJECT, ContainerState::OBJECT_KEY_OR_END});
      return true;
    }
    return false;
  }

  ContainerFrame& frame = containers_.back();
  switch (frame.state) {
    case ContainerState::OBJECT_KEY_OR_END:
      if (is_json_whitespace(character)) {
        return true;
      }
      if (frame.type == ContainerType::OBJECT && character == '}') {
        return close_container(ContainerType::OBJECT);
      }
      if (frame.type == ContainerType::OBJECT && character == '"') {
        parse_mode_ = ParseMode::STRING;
        string_role_ = StringRole::OBJECT_KEY;
        return true;
      }
      return false;
    case ContainerState::OBJECT_KEY_AFTER_COMMA:
      if (is_json_whitespace(character)) {
        return true;
      }
      if (frame.type == ContainerType::OBJECT && character == '"') {
        parse_mode_ = ParseMode::STRING;
        string_role_ = StringRole::OBJECT_KEY;
        return true;
      }
      return false;
    case ContainerState::OBJECT_COLON:
      if (is_json_whitespace(character)) {
        return true;
      }
      if (character != ':') {
        return false;
      }
      frame.state = ContainerState::OBJECT_VALUE;
      return true;
    case ContainerState::OBJECT_VALUE:
      if (is_json_whitespace(character)) {
        return true;
      }
      return start_value(character);
    case ContainerState::OBJECT_COMMA_OR_END:
      if (is_json_whitespace(character)) {
        return true;
      }
      if (character == ',') {
        frame.state = ContainerState::OBJECT_KEY_AFTER_COMMA;
        return true;
      }
      if (character == '}') {
        return close_container(ContainerType::OBJECT);
      }
      return false;
    case ContainerState::ARRAY_VALUE_OR_END:
      if (is_json_whitespace(character)) {
        return true;
      }
      if (character == ']') {
        return close_container(ContainerType::ARRAY);
      }
      return start_value(character);
    case ContainerState::ARRAY_VALUE_AFTER_COMMA:
      if (is_json_whitespace(character)) {
        return true;
      }
      return start_value(character);
    case ContainerState::ARRAY_COMMA_OR_END:
      if (is_json_whitespace(character)) {
        return true;
      }
      if (character == ',') {
        frame.state = ContainerState::ARRAY_VALUE_AFTER_COMMA;
        return true;
      }
      if (character == ']') {
        return close_container(ContainerType::ARRAY);
      }
      return false;
  }
  return false;
}

bool JsonObjectGrammarState::consume_string_character(char character) {
  if (parse_mode_ == ParseMode::STRING_ESCAPE) {
    if (!is_json_escape(character)) {
      return false;
    }
    if (character == 'u') {
      parse_mode_ = ParseMode::STRING_UNICODE;
      unicode_digits_ = 0;
    } else {
      parse_mode_ = ParseMode::STRING;
    }
    return true;
  }
  if (parse_mode_ == ParseMode::STRING_UNICODE) {
    if (!is_hex_digit(character)) {
      return false;
    }
    ++unicode_digits_;
    if (unicode_digits_ == 4) {
      parse_mode_ = ParseMode::STRING;
      unicode_digits_ = 0;
    }
    return true;
  }
  if (character == '"') {
    parse_mode_ = ParseMode::NONE;
    if (string_role_ == StringRole::OBJECT_KEY) {
      containers_.back().state = ContainerState::OBJECT_COLON;
    } else {
      complete_value();
    }
    return true;
  }
  if (character == '\\') {
    parse_mode_ = ParseMode::STRING_ESCAPE;
    return true;
  }
  return static_cast<unsigned char>(character) >= 0x20;
}

bool JsonObjectGrammarState::consume_number_character(char character) {
  const NumberState current_state = number_state_;
  if (current_state == NumberState::AFTER_MINUS) {
    if (character == '0') {
      number_state_ = NumberState::ZERO;
      return true;
    }
    if (character >= '1' && character <= '9') {
      number_state_ = NumberState::INTEGER;
      return true;
    }
    return false;
  }
  if (current_state == NumberState::ZERO) {
    if (character == '.') {
      number_state_ = NumberState::FRACTION_POINT;
      return true;
    }
    if (character == 'e' || character == 'E') {
      number_state_ = NumberState::EXPONENT;
      return true;
    }
  } else if (current_state == NumberState::INTEGER) {
    if (character >= '0' && character <= '9') {
      return true;
    }
    if (character == '.') {
      number_state_ = NumberState::FRACTION_POINT;
      return true;
    }
    if (character == 'e' || character == 'E') {
      number_state_ = NumberState::EXPONENT;
      return true;
    }
  } else if (current_state == NumberState::FRACTION_POINT) {
    if (character >= '0' && character <= '9') {
      number_state_ = NumberState::FRACTION;
      return true;
    }
    return false;
  } else if (current_state == NumberState::FRACTION) {
    if (character >= '0' && character <= '9') {
      return true;
    }
    if (character == 'e' || character == 'E') {
      number_state_ = NumberState::EXPONENT;
      return true;
    }
  } else if (current_state == NumberState::EXPONENT) {
    if (character == '+' || character == '-') {
      number_state_ = NumberState::EXPONENT_SIGN;
      return true;
    }
    if (character >= '0' && character <= '9') {
      number_state_ = NumberState::EXPONENT_DIGITS;
      return true;
    }
    return false;
  } else if (current_state == NumberState::EXPONENT_SIGN) {
    if (character >= '0' && character <= '9') {
      number_state_ = NumberState::EXPONENT_DIGITS;
      return true;
    }
    return false;
  } else if (current_state == NumberState::EXPONENT_DIGITS &&
             character >= '0' && character <= '9') {
    return true;
  }

  if (has_complete_number() && is_json_delimiter(character)) {
    parse_mode_ = ParseMode::NONE;
    complete_value();
    return consume_character(character);
  }
  return false;
}

bool JsonObjectGrammarState::consume_literal_character(char character) {
  if (literal_index_ >= literal_target_.size() ||
      character != literal_target_[literal_index_]) {
    return false;
  }
  ++literal_index_;
  if (literal_index_ == literal_target_.size()) {
    parse_mode_ = ParseMode::NONE;
    complete_value();
  }
  return true;
}

bool JsonObjectGrammarState::start_value(char character) {
  if (character == '{') {
    containers_.push_back(
        {ContainerType::OBJECT, ContainerState::OBJECT_KEY_OR_END});
    return true;
  }
  if (character == '[') {
    containers_.push_back(
        {ContainerType::ARRAY, ContainerState::ARRAY_VALUE_OR_END});
    return true;
  }
  if (character == '"') {
    parse_mode_ = ParseMode::STRING;
    string_role_ = StringRole::VALUE;
    return true;
  }
  if (character == '-') {
    parse_mode_ = ParseMode::NUMBER;
    number_state_ = NumberState::AFTER_MINUS;
    return true;
  }
  if (character == '0') {
    parse_mode_ = ParseMode::NUMBER;
    number_state_ = NumberState::ZERO;
    return true;
  }
  if (character >= '1' && character <= '9') {
    parse_mode_ = ParseMode::NUMBER;
    number_state_ = NumberState::INTEGER;
    return true;
  }
  if (character == 't' || character == 'f' || character == 'n') {
    parse_mode_ = ParseMode::LITERAL;
    literal_target_ = character == 't'   ? "true"
                      : character == 'f' ? "false"
                                         : "null";
    literal_index_ = 1;
    return true;
  }
  return false;
}

void JsonObjectGrammarState::complete_value() {
  if (containers_.empty()) {
    root_complete_ = true;
    return;
  }
  ContainerFrame& frame = containers_.back();
  if (frame.type == ContainerType::OBJECT) {
    frame.state = ContainerState::OBJECT_COMMA_OR_END;
  } else {
    frame.state = ContainerState::ARRAY_COMMA_OR_END;
  }
}

bool JsonObjectGrammarState::close_container(ContainerType type) {
  if (containers_.empty() || containers_.back().type != type) {
    return false;
  }
  containers_.pop_back();
  complete_value();
  return true;
}

bool JsonObjectGrammarState::has_complete_number() const {
  return number_state_ == NumberState::ZERO ||
         number_state_ == NumberState::INTEGER ||
         number_state_ == NumberState::FRACTION ||
         number_state_ == NumberState::EXPONENT_DIGITS;
}

JsonObjectGrammar::JsonObjectGrammar(
    std::vector<std::string> token_pieces,
    std::unordered_set<int32_t> stop_token_ids,
    std::vector<int32_t> reasoning_end_token_ids)
    : token_pieces_(std::move(token_pieces)),
      stop_token_ids_(std::move(stop_token_ids)),
      reasoning_end_token_ids_(std::move(reasoning_end_token_ids)),
      filter_mask_cache_(std::make_shared<FilterMaskCache>()) {
  std::vector<uint32_t> reasoning_bitmask(bitmask_num_words(), 0xFFFFFFFFu);
  for (const int32_t stop_token_id : stop_token_ids_) {
    if (stop_token_id < 0 ||
        static_cast<size_t>(stop_token_id) >= token_pieces_.size()) {
      continue;
    }
    const size_t word_index = static_cast<size_t>(stop_token_id) / 32U;
    const uint32_t bit = 1U << (static_cast<uint32_t>(stop_token_id) & 31U);
    reasoning_bitmask[word_index] &= ~bit;
  }
  // Clear unused high bits in the last word so padding bits stay disallowed.
  const size_t remainder = token_pieces_.size() % 32U;
  if (remainder != 0U && !reasoning_bitmask.empty()) {
    const uint32_t keep_mask = (1U << static_cast<uint32_t>(remainder)) - 1U;
    reasoning_bitmask.back() &= keep_mask;
  }
  reasoning_cached_mask_ = std::make_shared<const std::vector<uint32_t>>(
      std::move(reasoning_bitmask));
}

std::shared_ptr<const JsonObjectGrammar>
JsonObjectGrammar::create_from_tokenizer(
    const Tokenizer& tokenizer,
    int32_t eos_token_id,
    const std::unordered_set<int32_t>& stop_token_ids,
    int64_t model_vocab_size,
    const std::vector<int32_t>& reasoning_end_token_ids,
    std::string* error) {
  const size_t tokenizer_vocab_size = tokenizer.vocab_size();
  if (tokenizer_vocab_size == 0) {
    if (error != nullptr) {
      *error =
          "JSON object constraint requires a non-empty tokenizer vocabulary";
    }
    return nullptr;
  }

  if (model_vocab_size <= 0) {
    model_vocab_size = static_cast<int64_t>(tokenizer_vocab_size);
  }
  if (model_vocab_size < static_cast<int64_t>(tokenizer_vocab_size)) {
    if (error != nullptr) {
      *error = "model vocabulary (" + std::to_string(model_vocab_size) +
               ") is smaller than tokenizer vocabulary (" +
               std::to_string(tokenizer_vocab_size) + ")";
    }
    return nullptr;
  }
  const size_t model_vocab_size_value = static_cast<size_t>(model_vocab_size);

  std::vector<std::string> token_pieces;
  token_pieces.reserve(model_vocab_size_value);
  size_t non_empty_piece_count = 0;
  for (size_t token_id = 0; token_id < tokenizer_vocab_size; ++token_id) {
    const int32_t id = static_cast<int32_t>(token_id);
    std::string piece = tokenizer.decode_token(id);
    if (piece.empty()) {
      piece = tokenizer.id_to_token(id);
    }
    if (!piece.empty()) {
      ++non_empty_piece_count;
    }
    token_pieces.push_back(std::move(piece));
  }
  token_pieces.resize(model_vocab_size_value);
  if (non_empty_piece_count == 0) {
    if (error != nullptr) {
      *error =
          "JSON object constraint requires stable decoded tokenizer pieces";
    }
    return nullptr;
  }
  std::unordered_set<int32_t> terminal_token_ids = stop_token_ids;
  if (eos_token_id >= 0) {
    terminal_token_ids.insert(eos_token_id);
  }
  return std::make_shared<const JsonObjectGrammar>(
      std::move(token_pieces), terminal_token_ids, reasoning_end_token_ids);
}

std::shared_ptr<const JsonObjectGrammar>
JsonObjectGrammar::create_from_tokenizer(
    const Tokenizer& tokenizer,
    int32_t eos_token_id,
    const std::unordered_set<int32_t>& stop_token_ids,
    int64_t model_vocab_size,
    bool reasoning_enabled,
    std::string* error) {
  std::vector<int32_t> reasoning_end_token_ids;
  if (reasoning_enabled && (!tokenizer.encode("</think>",
                                              &reasoning_end_token_ids,
                                              /*add_special_tokens=*/false) ||
                            reasoning_end_token_ids.empty())) {
    if (error != nullptr) {
      *error = "reasoning end marker </think> is not available";
    }
    return nullptr;
  }
  return create_from_tokenizer(tokenizer,
                               eos_token_id,
                               stop_token_ids,
                               model_vocab_size,
                               reasoning_end_token_ids,
                               error);
}

JsonObjectGrammarState JsonObjectGrammar::initial_state(
    bool reasoning_phase) const {
  return JsonObjectGrammarState(this, reasoning_phase);
}

JsonObjectGrammarState JsonObjectGrammar::restore_state(
    const JsonObjectGrammarSnapshot& snapshot) const {
  JsonObjectGrammarState state = initial_state(snapshot.reasoning_enabled);
  if (!snapshot.enabled) {
    return JsonObjectGrammarState();
  }
  for (const int32_t token_id : snapshot.token_ids) {
    if (!state.advance(token_id)) {
      state.invalidate();
      break;
    }
  }
  return state;
}

std::vector<uint32_t> JsonObjectGrammar::compute_allowed_bitmask(
    const JsonObjectGrammarState& state) const {
  std::vector<uint32_t> bitmask(bitmask_num_words(), 0U);
  if (!state.is_valid()) {
    return bitmask;
  }
  for (size_t token_id = 0; token_id < token_pieces_.size(); ++token_id) {
    if (!state.clone_state().advance(static_cast<int32_t>(token_id))) {
      continue;
    }
    bitmask[token_id / 32U] |= 1U << (static_cast<uint32_t>(token_id) & 31U);
  }
  return bitmask;
}

std::shared_ptr<const std::vector<uint32_t>>
JsonObjectGrammar::cached_mask_for_state(
    const JsonObjectGrammarState& state) const {
  CHECK(state.grammar_ == this)
      << "JSON grammar state belongs to a different grammar";
  CHECK(state.is_valid()) << "JSON object grammar state is invalid";
  if (state.in_reasoning()) {
    return reasoning_cached_mask_;
  }

  const std::string cache_key = state.transition_cache_key();
  {
    const std::lock_guard<std::mutex> lock(filter_mask_cache_->mutex);
    const auto it = filter_mask_cache_->entries.find(cache_key);
    if (it != filter_mask_cache_->entries.end()) {
      filter_mask_cache_->recency.splice(filter_mask_cache_->recency.begin(),
                                         filter_mask_cache_->recency,
                                         it->second.recency_iterator);
      COUNTER_INC(json_object_mask_cache_hits_total);
      return it->second.mask;
    }
  }

  COUNTER_INC(json_object_mask_cache_misses_total);
  Timer scan_timer;
  std::vector<uint32_t> bitmask = compute_allowed_bitmask(state);
  HISTOGRAM_OBSERVE(json_object_mask_vocab_scan_latency_microseconds,
                    static_cast<int64_t>(scan_timer.elapsed_microseconds()));
  CHECK(std::any_of(bitmask.begin(), bitmask.end(), [](uint32_t word) {
    return word != 0U;
  })) << "JSON object grammar has no allowed token; refusing unrestricted mask";

  auto cached =
      std::make_shared<const std::vector<uint32_t>>(std::move(bitmask));

  const std::lock_guard<std::mutex> lock(filter_mask_cache_->mutex);
  const auto existing = filter_mask_cache_->entries.find(cache_key);
  if (existing != filter_mask_cache_->entries.end()) {
    filter_mask_cache_->recency.splice(filter_mask_cache_->recency.begin(),
                                       filter_mask_cache_->recency,
                                       existing->second.recency_iterator);
    return existing->second.mask;
  }
  if (filter_mask_cache_->entries.size() >= kMaxFilterMaskCacheEntries) {
    CHECK(!filter_mask_cache_->recency.empty());
    filter_mask_cache_->entries.erase(filter_mask_cache_->recency.back());
    filter_mask_cache_->recency.pop_back();
  }
  filter_mask_cache_->recency.emplace_front(cache_key);
  filter_mask_cache_->entries.emplace(
      cache_key,
      FilterMaskCacheEntry{cached, filter_mask_cache_->recency.begin()});
  return cached;
}

std::vector<int32_t> JsonObjectGrammar::allowed_token_ids(
    const JsonObjectGrammarState& state) const {
  const std::vector<uint32_t> bitmask = allowed_token_bitmask(state);
  std::vector<int32_t> allowed;
  allowed.reserve(token_pieces_.size());
  for (size_t token_id = 0; token_id < token_pieces_.size(); ++token_id) {
    const uint32_t word = bitmask[token_id / 32U];
    if ((word & (1U << (static_cast<uint32_t>(token_id) & 31U))) != 0U) {
      allowed.push_back(static_cast<int32_t>(token_id));
    }
  }
  return allowed;
}

std::vector<uint32_t> JsonObjectGrammar::allowed_token_bitmask(
    const JsonObjectGrammarState& state) const {
  if (!state.is_valid()) {
    return std::vector<uint32_t>(bitmask_num_words(), 0U);
  }
  return *cached_mask_for_state(state);
}

torch::Tensor build_json_object_filter_bitmask(
    const std::vector<JsonObjectGrammarState>& states,
    const torch::Device& device) {
  if (states.empty()) {
    return torch::Tensor();
  }

  const JsonObjectGrammar* grammar = nullptr;
  for (const auto& state : states) {
    if (state.initialized()) {
      grammar = state.grammar();
      break;
    }
  }
  if (grammar == nullptr) {
    return torch::Tensor();
  }

  Timer batch_timer;
  int64_t constrained_rows = 0;
  const int64_t num_words = static_cast<int64_t>(grammar->bitmask_num_words());
  torch::Tensor mask = torch::empty(
      {static_cast<int64_t>(states.size()), num_words},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  int32_t* mask_data = mask.data_ptr<int32_t>();
  const size_t row_bytes = static_cast<size_t>(num_words) * sizeof(int32_t);
  for (size_t row = 0; row < states.size(); ++row) {
    const auto& state = states[row];
    int32_t* row_data = mask_data + row * static_cast<size_t>(num_words);
    if (state.initialized()) {
      ++constrained_rows;
      const JsonObjectGrammar* state_grammar = state.grammar();
      CHECK_EQ(state_grammar->vocab_size(), grammar->vocab_size())
          << "mixed JSON grammar vocabularies in one batch";
      if (state.is_valid()) {
        const auto cached_mask = state_grammar->cached_mask_for_state(state);
        std::memcpy(row_data, cached_mask->data(), row_bytes);
      } else {
        std::fill_n(row_data, num_words, 0);
      }
    } else {
      std::fill_n(row_data, num_words, static_cast<int32_t>(-1));
    }
  }
  HISTOGRAM_OBSERVE(json_object_mask_batch_build_latency_microseconds,
                    static_cast<int64_t>(batch_timer.elapsed_microseconds()));
  if (!device.is_cpu()) {
    Timer transfer_timer;
    mask = mask.to(device);
    HISTOGRAM_OBSERVE(
        json_object_mask_device_copy_latency_microseconds,
        static_cast<int64_t>(transfer_timer.elapsed_microseconds()));
  }
  return mask;
}

std::optional<JsonTokenMismatch> JsonObjectGrammarState::advance_tokens(
    size_t base_committed_tokens,
    size_t row_width,
    const int64_t* row_tokens,
    int64_t token_stride) {
  for (size_t token_offset = 0; token_offset < row_width; ++token_offset) {
    const int64_t raw_token_id =
        row_tokens[static_cast<int64_t>(token_offset) * token_stride];
    if (raw_token_id < 0) {
      break;
    }
    const int32_t token_id = static_cast<int32_t>(raw_token_id);
    if (!accept_token(token_id)) {
      return JsonTokenMismatch{token_offset,
                               token_id,
                               base_committed_tokens + token_offset,
                               fingerprint()};
    }
  }
  return std::nullopt;
}

std::vector<JsonTokenMismatch> apply_json_constraints(
    std::vector<JsonObjectGrammarState>& states,
    const torch::Tensor& accepted_tokens,
    const std::vector<int32_t>& token_rows) {
  const size_t num_states = token_rows.size();
  CHECK_EQ(num_states, states.size())
      << "JSON constraint states must align with accepted-token rows";
  if (!accepted_tokens.defined()) {
    for (size_t state_index = 0; state_index < num_states; ++state_index) {
      if (states[state_index].initialized()) {
        CHECK_EQ(token_rows[state_index], -1)
            << "JSON constraint advancement requires accepted tokens";
      }
    }
    return {};
  }

  CHECK(accepted_tokens.device().is_cpu())
      << "JSON constraint tokens must already be ready on CPU";
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kInt64)
      << "JSON constraint tokens must have int64 dtype";
  CHECK(accepted_tokens.dim() == 1 || accepted_tokens.dim() == 2)
      << "JSON constraint tokens must be 1-D or 2-D";

  const int64_t* token_data = accepted_tokens.const_data_ptr<int64_t>();
  const int64_t row_stride = accepted_tokens.stride(0);
  const bool has_multiple_tokens = accepted_tokens.dim() == 2;
  const int64_t tokens_per_row =
      has_multiple_tokens ? accepted_tokens.size(1) : 1;
  const int64_t token_stride =
      has_multiple_tokens ? accepted_tokens.stride(1) : 0;
  std::vector<JsonTokenMismatch> violations;
  for (size_t state_index = 0; state_index < num_states; ++state_index) {
    JsonObjectGrammarState& state = states[state_index];
    const int32_t token_row = token_rows[state_index];
    if (!state.initialized() || token_row == -1) {
      continue;
    }
    CHECK_LT(token_row, accepted_tokens.size(0))
        << "JSON constraint token row is outside the accepted-token tensor";
    const int64_t row_offset = static_cast<int64_t>(token_row) * row_stride;
    std::optional<JsonTokenMismatch> mismatch =
        state.advance_tokens(state.num_accepted_tokens(),
                             static_cast<size_t>(tokens_per_row),
                             token_data + row_offset,
                             token_stride);
    if (mismatch.has_value()) {
      if (violations.empty()) {
        violations.reserve(num_states);
      }
      mismatch->sequence_index = state_index;
      violations.push_back(*mismatch);
      state = JsonObjectGrammarState();
    }
  }
  return violations;
}

}  // namespace xllm
