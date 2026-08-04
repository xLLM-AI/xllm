/* Copyright 2025-2026 The xLLM Authors.

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

#include "rwkv_tokenizer.h"

#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <absl/strings/str_cat.h>
#include <glog/logging.h>

#include <cctype>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string_view>

namespace xllm {
namespace {

struct RwkvTrieNode {
  absl::flat_hash_map<uint8_t, std::unique_ptr<RwkvTrieNode>> children;
  std::optional<int32_t> token_id;
};

const RwkvTrieNode* find_child(const RwkvTrieNode* node, uint8_t byte) {
  const auto it = node->children.find(byte);
  if (it == node->children.end()) {
    return nullptr;
  }
  return it->second.get();
}

RwkvTrieNode* get_or_create_child(RwkvTrieNode* node, uint8_t byte) {
  std::unique_ptr<RwkvTrieNode>& child = node->children[byte];
  if (child == nullptr) {
    child = std::make_unique<RwkvTrieNode>();
  }
  return child.get();
}

void utf8_append_codepoint(std::string* out, uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    out->push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out->push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out->push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0x10FFFF) {
    out->push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
}

bool parse_hex_digits(const std::string& repr,
                      size_t pos,
                      int32_t digits,
                      uint32_t* value) {
  if (value == nullptr || pos + static_cast<size_t>(digits) > repr.size()) {
    return false;
  }
  uint32_t parsed = 0;
  for (int32_t i = 0; i < digits; ++i) {
    const char ch = repr[pos + static_cast<size_t>(i)];
    if (!std::isxdigit(static_cast<unsigned char>(ch))) {
      return false;
    }
    parsed = (parsed << 4) +
             static_cast<uint32_t>(std::isdigit(ch) != 0
                                       ? ch - '0'
                                       : std::tolower(ch) - 'a' + 10);
  }
  *value = parsed;
  return true;
}

bool append_codepoint(bool bytes_mode, std::string* out, uint32_t codepoint) {
  if (bytes_mode) {
    if (codepoint > 0xFF) {
      return false;
    }
    out->push_back(static_cast<char>(codepoint));
    return true;
  }
  utf8_append_codepoint(out, codepoint);
  return true;
}

bool decode_utf8_codepoint(const std::string& repr,
                           size_t pos,
                           uint32_t* codepoint,
                           size_t* consumed) {
  if (codepoint == nullptr || consumed == nullptr || pos >= repr.size()) {
    return false;
  }
  const unsigned char lead = static_cast<unsigned char>(repr[pos]);
  if ((lead & 0x80) == 0) {
    *codepoint = lead;
    *consumed = 1;
    return true;
  }
  if ((lead & 0xE0) == 0xC0) {
    if (pos + 1 >= repr.size()) {
      return false;
    }
    const unsigned char b1 = static_cast<unsigned char>(repr[pos + 1]);
    if ((b1 & 0xC0) != 0x80) {
      return false;
    }
    *codepoint = ((lead & 0x1F) << 6) | (b1 & 0x3F);
    *consumed = 2;
    return true;
  }
  if ((lead & 0xF0) == 0xE0) {
    if (pos + 2 >= repr.size()) {
      return false;
    }
    const unsigned char b1 = static_cast<unsigned char>(repr[pos + 1]);
    const unsigned char b2 = static_cast<unsigned char>(repr[pos + 2]);
    if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) {
      return false;
    }
    *codepoint = ((lead & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
    *consumed = 3;
    return true;
  }
  if ((lead & 0xF8) == 0xF0) {
    if (pos + 3 >= repr.size()) {
      return false;
    }
    const unsigned char b1 = static_cast<unsigned char>(repr[pos + 1]);
    const unsigned char b2 = static_cast<unsigned char>(repr[pos + 2]);
    const unsigned char b3 = static_cast<unsigned char>(repr[pos + 3]);
    if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) {
      return false;
    }
    *codepoint = ((lead & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                 ((b2 & 0x3F) << 6) | (b3 & 0x3F);
    *consumed = 4;
    return true;
  }
  return false;
}

bool parse_escape(bool bytes_mode,
                  const std::string& repr,
                  size_t* pos,
                  std::string* out) {
  if (pos == nullptr || out == nullptr || *pos >= repr.size()) {
    return false;
  }
  const char esc = repr[(*pos)++];
  switch (esc) {
    case '\\':
      return append_codepoint(bytes_mode, out, '\\');
    case '\'':
      return append_codepoint(bytes_mode, out, '\'');
    case '"':
      return append_codepoint(bytes_mode, out, '"');
    case 'a':
      return append_codepoint(bytes_mode, out, '\a');
    case 'b':
      return append_codepoint(bytes_mode, out, '\b');
    case 'f':
      return append_codepoint(bytes_mode, out, '\f');
    case 'n':
      return append_codepoint(bytes_mode, out, '\n');
    case 'r':
      return append_codepoint(bytes_mode, out, '\r');
    case 't':
      return append_codepoint(bytes_mode, out, '\t');
    case 'v':
      return append_codepoint(bytes_mode, out, '\v');
    case 'x': {
      uint32_t value = 0;
      if (!parse_hex_digits(repr, *pos, 2, &value)) {
        return false;
      }
      *pos += 2;
      return append_codepoint(bytes_mode, out, value);
    }
    case 'u': {
      uint32_t value = 0;
      if (!parse_hex_digits(repr, *pos, 4, &value)) {
        return false;
      }
      *pos += 4;
      return append_codepoint(bytes_mode, out, value);
    }
    case 'U': {
      uint32_t value = 0;
      if (!parse_hex_digits(repr, *pos, 8, &value)) {
        return false;
      }
      *pos += 8;
      return append_codepoint(bytes_mode, out, value);
    }
    default:
      return false;
  }
}

// Parse Python str/bytes literals used in rwkv_vocab_v20230424.txt.
// str literals are UTF-8 encoded after parsing, matching official rwkv eval().
bool parse_python_literal(const std::string& repr, std::string* out) {
  if (out == nullptr || repr.empty()) {
    return false;
  }
  out->clear();

  bool bytes_mode = false;
  char quote = 0;
  size_t pos = 0;
  if (repr.size() >= 2 && repr[0] == 'b' &&
      (repr[1] == '\'' || repr[1] == '"')) {
    bytes_mode = true;
    quote = repr[1];
    pos = 2;
  } else if (repr[0] == '\'' || repr[0] == '"') {
    quote = repr[0];
    pos = 1;
  } else {
    return false;
  }

  while (pos < repr.size()) {
    const char ch = repr[pos];
    if (ch == quote) {
      return pos + 1 == repr.size();
    }
    if (ch == '\\') {
      ++pos;
      if (!parse_escape(bytes_mode, repr, &pos, out)) {
        return false;
      }
      continue;
    }
    if (bytes_mode) {
      out->push_back(ch);
      ++pos;
      continue;
    }
    uint32_t codepoint = 0;
    size_t consumed = 0;
    if (!decode_utf8_codepoint(repr, pos, &codepoint, &consumed)) {
      return false;
    }
    utf8_append_codepoint(out, codepoint);
    pos += consumed;
  }
  return false;
}

void add_to_trie(RwkvTrieNode* root,
                 const std::string& token_bytes,
                 int32_t token_id) {
  RwkvTrieNode* node = root;
  for (const unsigned char byte : token_bytes) {
    node = get_or_create_child(node, byte);
  }
  node->token_id = token_id;
}

}  // namespace

struct RwkvTokenMapHash {
  using is_transparent = void;

  size_t operator()(std::string_view key) const {
    return absl::Hash<std::string_view>{}(key);
  }

  size_t operator()(const std::string& key) const {
    return absl::Hash<std::string_view>{}(std::string_view(key));
  }
};

struct RwkvTokenMapEq {
  using is_transparent = void;

  bool operator()(std::string_view lhs, std::string_view rhs) const {
    return lhs == rhs;
  }

  bool operator()(const std::string& lhs, const std::string& rhs) const {
    return lhs == rhs;
  }

  bool operator()(std::string_view lhs, const std::string& rhs) const {
    return lhs == rhs;
  }

  bool operator()(const std::string& lhs, std::string_view rhs) const {
    return lhs == rhs;
  }
};

struct RwkvTokenizer::VocabData {
  std::unique_ptr<RwkvTrieNode> root;
  absl::flat_hash_map<int32_t, std::string> idx_to_token;
  absl::flat_hash_map<std::string, int32_t, RwkvTokenMapHash, RwkvTokenMapEq>
      token_to_idx;
  size_t vocab_size = 0;
};

std::shared_ptr<const RwkvTokenizer::VocabData> RwkvTokenizer::build_vocab_data(
    const std::string& vocab_file_path) {
  auto data = std::make_shared<RwkvTokenizer::VocabData>();
  data->root = std::make_unique<RwkvTrieNode>();

  std::ifstream input(vocab_file_path);
  CHECK(input) << "Failed to open RWKV vocab file: " << vocab_file_path;

  int32_t skipped_lines = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    const size_t first_space = line.find(' ');
    const size_t last_space = line.rfind(' ');
    if (first_space == std::string::npos || last_space == first_space) {
      ++skipped_lines;
      continue;
    }

    int32_t token_id = 0;
    try {
      token_id = std::stoi(line.substr(0, first_space));
    } catch (const std::exception&) {
      ++skipped_lines;
      continue;
    }

    const std::string repr =
        line.substr(first_space + 1, last_space - first_space - 1);
    int32_t expected_len = 0;
    try {
      expected_len = std::stoi(line.substr(last_space + 1));
    } catch (const std::exception&) {
      ++skipped_lines;
      continue;
    }

    std::string token_bytes;
    if (!parse_python_literal(repr, &token_bytes) ||
        static_cast<int32_t>(token_bytes.size()) != expected_len) {
      ++skipped_lines;
      continue;
    }

    data->idx_to_token[token_id] = token_bytes;
    data->token_to_idx[token_bytes] = token_id;
    add_to_trie(data->root.get(), token_bytes, token_id);
    data->vocab_size =
        std::max(data->vocab_size, static_cast<size_t>(token_id) + 1);
  }

  CHECK(!data->idx_to_token.empty())
      << "RWKV vocab file is empty: " << vocab_file_path;
  if (skipped_lines > 0) {
    LOG(WARNING) << "RWKV vocab skipped " << skipped_lines
                 << " invalid lines from " << vocab_file_path;
  }
  LOG(INFO) << "Loaded RWKV trie tokenizer from " << vocab_file_path
            << ", entries=" << data->idx_to_token.size()
            << ", vocab_size=" << data->vocab_size;
  return data;
}

std::shared_ptr<const RwkvTokenizer::VocabData> RwkvTokenizer::load_vocab_data(
    const std::string& vocab_file_path) {
  static std::mutex cache_mutex;
  static absl::flat_hash_map<std::string,
                             std::weak_ptr<const RwkvTokenizer::VocabData>>
      cache;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto it = cache.find(vocab_file_path);
    if (it != cache.end()) {
      if (auto cached = it->second.lock()) {
        return cached;
      }
      cache.erase(it);
    }
  }

  auto data = RwkvTokenizer::build_vocab_data(vocab_file_path);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[vocab_file_path] = data;
  }
  return data;
}

RwkvTokenizer::RwkvTokenizer(const std::string_view& dir_path,
                             const TokenizerArgs& args)
    : dir_path_(dir_path), args_(args) {
  const std::string vocab_file_path =
      dir_path_.empty() ? args_.vocab_file()
                        : absl::StrCat(dir_path_, "/", args_.vocab_file());
  vocab_ = load_vocab_data(vocab_file_path);
}

std::unique_ptr<Tokenizer> RwkvTokenizer::clone() const {
  return std::make_unique<RwkvTokenizer>(dir_path_, args_);
}

bool RwkvTokenizer::encode_bytes(const std::string_view& text,
                                 std::vector<int32_t>* ids) const {
  if (ids == nullptr) {
    return false;
  }
  ids->clear();

  size_t index = 0;
  while (index < text.size()) {
    const RwkvTrieNode* node = vocab_->root.get();
    size_t best_end = index;
    std::optional<int32_t> best_id;

    size_t cursor = index;
    while (cursor < text.size()) {
      const uint8_t byte = static_cast<uint8_t>(text[cursor]);
      const RwkvTrieNode* child = find_child(node, byte);
      if (child == nullptr) {
        break;
      }
      node = child;
      ++cursor;
      if (node->token_id.has_value()) {
        best_end = cursor;
        best_id = node->token_id;
      }
    }

    if (!best_id.has_value()) {
      LOG(ERROR) << "Failed to tokenize RWKV text at byte offset " << index;
      ids->clear();
      return false;
    }
    ids->push_back(best_id.value());
    index = best_end;
  }
  return true;
}

bool RwkvTokenizer::encode(const std::string_view& text,
                           std::vector<int32_t>* ids,
                           bool add_special_tokens) const {
  if (ids == nullptr) {
    return false;
  }

  ids->clear();
  if (!encode_bytes(text, ids)) {
    return false;
  }
  if (add_special_tokens && args_.add_bos_token() &&
      !args_.bos_token().empty()) {
    const auto bos_id = token_to_id(args_.bos_token());
    if (bos_id.has_value()) {
      ids->insert(ids->begin(), bos_id.value());
    }
  }
  if (add_special_tokens && args_.add_eos_token() &&
      !args_.eos_token().empty()) {
    const auto eos_id = token_to_id(args_.eos_token());
    if (eos_id.has_value()) {
      ids->push_back(eos_id.value());
    }
  }
  return true;
}

std::string RwkvTokenizer::decode(const Slice<int32_t>& ids,
                                  bool skip_special_tokens) const {
  std::optional<int32_t> bos_id;
  std::optional<int32_t> eos_id;
  std::optional<int32_t> pad_id;
  if (skip_special_tokens) {
    if (!args_.bos_token().empty()) {
      bos_id = token_to_id(args_.bos_token());
    }
    if (!args_.eos_token().empty()) {
      eos_id = token_to_id(args_.eos_token());
    }
    if (!args_.pad_token().empty()) {
      pad_id = token_to_id(args_.pad_token());
    }
  }

  std::string bytes;
  bytes.reserve(ids.size());
  for (const int32_t id : ids) {
    if (skip_special_tokens) {
      if (bos_id.has_value() && id == bos_id.value()) {
        continue;
      }
      if (eos_id.has_value() && id == eos_id.value()) {
        continue;
      }
      if (pad_id.has_value() && id == pad_id.value()) {
        continue;
      }
    }

    const auto it = vocab_->idx_to_token.find(id);
    if (it == vocab_->idx_to_token.end()) {
      LOG(WARNING) << "Unknown RWKV token id during decode: " << id;
      continue;
    }
    bytes.append(it->second);
  }

  // Match official RWKV tokenizer behavior: replace invalid UTF-8 with U+FFFD.
  std::string output;
  output.reserve(bytes.size());
  size_t index = 0;
  while (index < bytes.size()) {
    const unsigned char byte = static_cast<unsigned char>(bytes[index]);
    if (byte < 0x80) {
      output.push_back(static_cast<char>(byte));
      ++index;
      continue;
    }

    size_t seq_len = 1;
    if ((byte & 0xE0) == 0xC0) {
      seq_len = 2;
    } else if ((byte & 0xF0) == 0xE0) {
      seq_len = 3;
    } else if ((byte & 0xF8) == 0xF0) {
      seq_len = 4;
    } else {
      output.append("\xEF\xBF\xBD");
      ++index;
      continue;
    }

    if (index + seq_len > bytes.size()) {
      output.append("\xEF\xBF\xBD");
      break;
    }

    bool valid = true;
    for (size_t i = 1; i < seq_len; ++i) {
      const unsigned char next = static_cast<unsigned char>(bytes[index + i]);
      if ((next & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }

    if (!valid) {
      output.append("\xEF\xBF\xBD");
      ++index;
      continue;
    }

    output.append(bytes.data() + index, seq_len);
    index += seq_len;
  }
  return output;
}

std::optional<int32_t> RwkvTokenizer::token_to_id(
    const std::string_view& token) const {
  const auto it = vocab_->token_to_idx.find(token);
  if (it == vocab_->token_to_idx.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::string RwkvTokenizer::id_to_token(int32_t id) const {
  const auto it = vocab_->idx_to_token.find(id);
  if (it == vocab_->idx_to_token.end()) {
    return "";
  }
  return it->second;
}

size_t RwkvTokenizer::vocab_size() const { return vocab_->vocab_size; }

}  // namespace xllm
