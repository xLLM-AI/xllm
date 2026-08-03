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

#include "core/framework/sampling/json_object_grammar.h"

#include <glog/logging.h>

#include "core/util/slice.h"

namespace xllm {
namespace {

constexpr float kDisallowedTokenMask = -1.0e9F;
constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

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

bool JsonObjectGrammarState::can_accept_token(int32_t token_id) const {
  JsonObjectGrammarState candidate = *this;
  return candidate.accept_token(token_id);
}

bool JsonObjectGrammarState::accept_token(int32_t token_id) {
  if (!valid_ || grammar_ == nullptr || token_id < 0 ||
      static_cast<size_t>(token_id) >= grammar_->vocab_size()) {
    return false;
  }

  if (grammar_->stop_token_ids_.find(token_id) !=
      grammar_->stop_token_ids_.end()) {
    if (reasoning_phase_ || !root_complete_ || parse_mode_ != ParseMode::NONE) {
      return false;
    }
    committed_token_ids_.push_back(token_id);
    return true;
  }

  if (reasoning_phase_) {
    if (grammar_->reasoning_end_token_ids_.empty()) {
      committed_token_ids_.push_back(token_id);
      return true;
    }
    const auto& marker = grammar_->reasoning_end_token_ids_;
    if (token_id == marker[reasoning_marker_index_]) {
      ++reasoning_marker_index_;
      if (reasoning_marker_index_ == marker.size()) {
        reasoning_phase_ = false;
        reasoning_marker_index_ = 0;
        containers_.clear();
        parse_mode_ = ParseMode::NONE;
        root_started_ = false;
        root_complete_ = false;
      }
    } else {
      reasoning_marker_index_ = token_id == marker.front() ? 1 : 0;
    }
    committed_token_ids_.push_back(token_id);
    return true;
  }

  if (root_complete_ && parse_mode_ == ParseMode::NONE &&
      !grammar_->stop_token_ids_.empty()) {
    return false;
  }

  const std::string& piece = grammar_->token_piece(token_id);
  if (piece.empty()) {
    return false;
  }
  if (!accept_piece(piece)) {
    return false;
  }
  committed_token_ids_.push_back(token_id);
  return true;
}

JsonObjectGrammarSnapshot JsonObjectGrammarState::snapshot() const {
  JsonObjectGrammarSnapshot snapshot;
  snapshot.enabled = initialized();
  snapshot.reasoning_enabled = reasoning_enabled_;
  snapshot.token_ids = committed_token_ids_;
  return snapshot;
}

uint64_t JsonObjectGrammarState::fingerprint() const {
  const JsonObjectGrammarSnapshot state_snapshot = snapshot();
  uint64_t hash = kFnvOffsetBasis;
  hash = hash_uint64(hash, state_snapshot.enabled ? 1U : 0U);
  hash = hash_uint64(hash, state_snapshot.reasoning_enabled ? 1U : 0U);
  hash =
      hash_uint64(hash, static_cast<uint64_t>(state_snapshot.token_ids.size()));
  for (const int32_t token_id : state_snapshot.token_ids) {
    hash = hash_uint64(hash,
                       static_cast<uint64_t>(static_cast<uint32_t>(token_id)));
  }
  return hash;
}

bool JsonObjectGrammarState::can_accept_piece(std::string_view piece) const {
  JsonObjectGrammarState candidate = *this;
  return candidate.accept_piece(piece);
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

  if (has_complete_number() && is_value_delimiter(character)) {
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

bool JsonObjectGrammarState::is_value_delimiter(char character) const {
  return is_json_delimiter(character);
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
      reasoning_end_token_ids_(std::move(reasoning_end_token_ids)) {}

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
  if (reasoning_enabled) {
    if (!tokenizer.encode("</think>",
                          &reasoning_end_token_ids,
                          /*add_special_tokens=*/false) ||
        reasoning_end_token_ids.empty()) {
      if (error != nullptr) {
        *error = "reasoning end marker </think> is not available";
      }
      return nullptr;
    }
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
    if (!state.accept_token(token_id)) {
      state.invalidate();
      break;
    }
  }
  return state;
}

std::vector<int32_t> JsonObjectGrammar::allowed_token_ids(
    const JsonObjectGrammarState& state) const {
  CHECK(state.grammar_ == this)
      << "JSON grammar state belongs to a different grammar";
  std::vector<int32_t> allowed;
  if (!state.is_valid()) {
    return allowed;
  }
  allowed.reserve(token_pieces_.size());
  for (size_t token_id = 0; token_id < token_pieces_.size(); ++token_id) {
    if (state.can_accept_token(static_cast<int32_t>(token_id))) {
      allowed.push_back(static_cast<int32_t>(token_id));
    }
  }
  return allowed;
}

torch::Tensor JsonObjectGrammar::build_filter_mask(
    const JsonObjectGrammarState& state,
    const torch::Device& device,
    torch::ScalarType dtype) const {
  const std::vector<int32_t> allowed = allowed_token_ids(state);
  CHECK(!allowed.empty())
      << "JSON object grammar has no allowed token; refusing unrestricted mask";
  auto mask = torch::full({static_cast<int64_t>(vocab_size())},
                          kDisallowedTokenMask,
                          torch::TensorOptions().dtype(torch::kFloat32));
  const auto allowed_ids = torch::tensor(allowed, torch::kLong);
  mask.index_fill_(0, allowed_ids, 0.0F);
  if (dtype != torch::kFloat32) {
    mask = mask.to(dtype);
  }
  if (!device.is_cpu()) {
    mask = mask.to(device);
  }
  return mask;
}

torch::Tensor build_json_object_filter_mask(
    const std::vector<JsonObjectGrammarState>& states,
    const torch::Device& device,
    torch::ScalarType dtype) {
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

  std::vector<torch::Tensor> masks;
  masks.reserve(states.size());
  for (const auto& state : states) {
    if (state.initialized()) {
      CHECK_EQ(state.grammar(), grammar)
          << "mixed JSON grammar definitions in one batch";
      masks.push_back(grammar->build_filter_mask(state, device, dtype));
    } else {
      masks.push_back(
          torch::zeros({static_cast<int64_t>(grammar->vocab_size())},
                       torch::TensorOptions().device(device).dtype(dtype)));
    }
  }
  return torch::stack(masks, /*dim=*/0);
}

std::vector<JsonObjectGrammarState> advance_json_object_states(
    const std::vector<JsonObjectGrammarState>& states,
    const std::vector<int32_t>& token_ids) {
  CHECK_EQ(states.size(), token_ids.size())
      << "JSON grammar state/token count mismatch";
  std::vector<JsonObjectGrammarState> next_states = states;
  for (size_t state_idx = 0; state_idx < next_states.size(); ++state_idx) {
    if (!next_states[state_idx].initialized() ||
        !next_states[state_idx].can_accept_token(token_ids[state_idx])) {
      continue;
    }
    CHECK(next_states[state_idx].accept_token(token_ids[state_idx]))
        << "JSON grammar state transition failed, token_id="
        << token_ids[state_idx];
  }
  return next_states;
}

const std::string& JsonObjectGrammar::token_piece(int32_t token_id) const {
  CHECK_GE(token_id, 0);
  CHECK_LT(static_cast<size_t>(token_id), token_pieces_.size());
  return token_pieces_[token_id];
}

}  // namespace xllm
