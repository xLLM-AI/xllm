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

#include "api_service/multipart_form_data.h"

#include <absl/strings/match.h>
#include <butil/iobuf.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace xllm::api_service {
namespace {

// Maximum boundary length in bytes (RFC 2046 section 5.1.1).
constexpr size_t kMaxBoundaryLength = 70;

constexpr std::string_view kBoundaryExtraChars = "'()+_,-./:=? ";

constexpr size_t kNpos = std::string_view::npos;

std::string_view trim(std::string_view value) {
  constexpr std::string_view kWhitespace = " \t";
  const size_t start = value.find_first_not_of(kWhitespace);
  if (start == std::string_view::npos) {
    return {};
  }
  const size_t end = value.find_last_not_of(kWhitespace);
  return value.substr(start, end - start + 1);
}

bool parse_param_value(std::string_view params,
                       size_t& pos,
                       std::string& value) {
  value.clear();
  if (pos >= params.size()) {
    return false;
  }
  if (params[pos] == '"') {
    ++pos;
    while (pos < params.size()) {
      const char c = params[pos];
      if (c == '\\' && pos + 1 < params.size()) {
        value.push_back(params[pos + 1]);
        pos += 2;
      } else if (c == '"') {
        ++pos;
        return true;
      } else {
        value.push_back(c);
        ++pos;
      }
    }
    return false;
  }
  const size_t start = pos;
  while (pos < params.size() && params[pos] != ';') {
    ++pos;
  }
  value = std::string(trim(params.substr(start, pos - start)));
  return true;
}

bool parse_boundary(std::string_view content_type, std::string& boundary) {
  size_t pos = 0;
  while (pos < content_type.size()) {
    if (content_type[pos] == ';' || content_type[pos] == ' ' ||
        content_type[pos] == '\t') {
      ++pos;
      continue;
    }
    const size_t key_start = pos;
    while (pos < content_type.size() && content_type[pos] != '=' &&
           content_type[pos] != ';') {
      ++pos;
    }
    if (pos >= content_type.size() || content_type[pos] != '=') {
      // Valueless token such as the media type itself; skip to the next one.
      while (pos < content_type.size() && content_type[pos] != ';') {
        ++pos;
      }
      continue;
    }
    const std::string_view key =
        content_type.substr(key_start, pos - key_start);
    ++pos;
    std::string param_value;
    if (!parse_param_value(content_type, pos, param_value)) {
      return false;
    }
    if (absl::EqualsIgnoreCase(key, "boundary")) {
      boundary = std::move(param_value);
      if (boundary.empty() || boundary.size() > kMaxBoundaryLength) {
        return false;
      }
      for (const char c : boundary) {
        if (std::isalnum(static_cast<unsigned char>(c)) == 0 &&
            kBoundaryExtraChars.find(c) == std::string_view::npos) {
          return false;
        }
      }
      return boundary.back() != ' ';
    }
  }
  return false;
}

bool parse_content_disposition(std::string_view value, MultipartPart& part) {
  constexpr std::string_view kDispositionType = "form-data";
  if (value.substr(0, kDispositionType.size()) != kDispositionType) {
    return false;
  }
  size_t pos = kDispositionType.size();
  while (pos < value.size()) {
    if (value[pos] == ';' || value[pos] == ' ' || value[pos] == '\t') {
      ++pos;
      continue;
    }
    const size_t key_start = pos;
    while (pos < value.size() && value[pos] != '=' && value[pos] != ';') {
      ++pos;
    }
    if (pos >= value.size() || value[pos] != '=') {
      return false;
    }
    const std::string_view key = value.substr(key_start, pos - key_start);
    ++pos;
    std::string param_value;
    if (!parse_param_value(value, pos, param_value)) {
      return false;
    }
    if (absl::EqualsIgnoreCase(key, "name")) {
      part.name = std::move(param_value);
    } else if (absl::EqualsIgnoreCase(key, "filename")) {
      part.filename = std::move(param_value);
    }
  }
  return true;
}

bool parse_part_headers(std::string_view headers, MultipartPart& part) {
  size_t pos = 0;
  while (pos < headers.size()) {
    const size_t line_end = headers.find("\r\n", pos);
    const size_t line_length =
        (line_end == std::string_view::npos ? headers.size() : line_end) - pos;
    const std::string_view line = headers.substr(pos, line_length);
    pos = line_end == std::string_view::npos ? headers.size() : line_end + 2;

    const size_t colon = line.find(':');
    if (colon == std::string_view::npos) {
      return false;
    }
    const std::string_view key = trim(line.substr(0, colon));
    const std::string_view value = trim(line.substr(colon + 1));
    if (absl::EqualsIgnoreCase(key, "content-disposition")) {
      if (!parse_content_disposition(value, part)) {
        return false;
      }
    } else if (absl::EqualsIgnoreCase(key, "content-type")) {
      part.content_type = std::string(value);
    }
  }
  return !part.name.empty();
}

// Random-access view over the block chain of an IOBuf.
class BodyView {
 public:
  explicit BodyView(const butil::IOBuf& body) : slice_it_(body) {
    butil::IOBufBytesIterator it(body);
    const void* data = nullptr;
    size_t size = 0;
    size_t offset = 0;
    while (it.forward_one_block(&data, &size)) {
      if (size == 0) {
        continue;
      }
      spans_.push_back(Span{static_cast<const char*>(data), size, offset});
      offset += size;
    }
    size_ = offset;
  }

  size_t size() const { return size_; }

  char byte_at(size_t pos) const {
    const Span& span = spans_[span_index_for(pos)];
    return span.data[pos - span.offset];
  }

  bool equals_at(size_t pos, std::string_view bytes) const {
    if (pos > size_ || bytes.size() > size_ - pos) {
      return false;
    }
    size_t span_idx = span_index_for(pos);
    size_t in_span = pos - spans_[span_idx].offset;
    size_t compared = 0;
    size_t remaining = bytes.size();
    while (remaining > 0) {
      const Span& span = spans_[span_idx];
      const size_t n = std::min(remaining, span.len - in_span);
      if (std::memcmp(span.data + in_span, bytes.data() + compared, n) != 0) {
        return false;
      }
      compared += n;
      remaining -= n;
      in_span = 0;
      ++span_idx;
    }
    return true;
  }

  size_t find(std::string_view pattern, size_t from) const {
    if (pattern.empty()) {
      return from;
    }
    for (const Span& span : spans_) {
      const size_t span_end = span.offset + span.len;
      if (span_end <= from) {
        continue;
      }
      size_t in_span = from > span.offset ? from - span.offset : 0;
      while (in_span < span.len) {
        const void* hit =
            std::memchr(span.data + in_span,
                        static_cast<unsigned char>(pattern.front()),
                        span.len - in_span);
        if (hit == nullptr) {
          break;
        }
        const size_t pos =
            span.offset + (static_cast<const char*>(hit) - span.data);
        if (pos + pattern.size() > size_) {
          return kNpos;
        }
        if (equals_at(pos, pattern)) {
          return pos;
        }
        in_span = pos + 1 - span.offset;
      }
    }
    return kNpos;
  }

  void slice(size_t pos, size_t len, butil::IOBuf* out) {
    slice_it_.forward(pos - sliced_upto_);
    slice_it_.append_and_forward(out, len);
    sliced_upto_ = pos + len;
  }

 private:
  struct Span {
    const char* data;
    size_t len;
    size_t offset;
  };

  size_t span_index_for(size_t pos) const {
    const auto it = std::upper_bound(
        spans_.begin(), spans_.end(), pos, [](size_t value, const Span& span) {
          return value < span.offset;
        });
    return static_cast<size_t>(it - spans_.begin()) - 1;
  }

  std::vector<Span> spans_;
  size_t size_ = 0;
  butil::IOBufBytesIterator slice_it_;
  size_t sliced_upto_ = 0;
};

}  // namespace

std::optional<std::string> find_multipart_field_value(
    const MultipartFormData& form,
    std::string_view name) {
  for (const auto& part : form.parts) {
    if (part.filename.empty() && part.name == name) {
      return part.value.to_string();
    }
  }
  return std::nullopt;
}

Status parse_multipart_form_data(std::string_view content_type_header,
                                 const butil::IOBuf& body,
                                 size_t max_part_bytes,
                                 MultipartFormData& out) {
  std::string boundary;
  if (!parse_boundary(content_type_header, boundary)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "Missing or invalid boundary in Content-Type header.");
  }

  const std::string first_delimiter = "--" + boundary;
  const std::string delimiter = "\r\n--" + boundary;
  BodyView view(body);
  if (!view.equals_at(0, first_delimiter)) {
    return Status(StatusCode::INVALID_ARGUMENT, "Malformed multipart body.");
  }
  size_t pos = first_delimiter.size();

  while (true) {
    // A boundary line is either terminated ("--") or followed by a part.
    if (view.equals_at(pos, "--")) {
      break;
    }
    // Skip optional transport padding, then require the CRLF line ending.
    while (pos < view.size() &&
           (view.byte_at(pos) == ' ' || view.byte_at(pos) == '\t')) {
      ++pos;
    }
    if (!view.equals_at(pos, "\r\n")) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Malformed multipart boundary.");
    }
    pos += 2;

    const size_t header_end = view.find("\r\n\r\n", pos);
    if (header_end == kNpos) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Malformed multipart part headers.");
    }
    // Part headers are small; materialize them for the line-based parser.
    std::string headers;
    body.copy_to(&headers, header_end - pos, pos);
    pos = header_end + 4;

    MultipartPart part;
    if (!parse_part_headers(headers, part)) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Malformed multipart part headers.");
    }

    const size_t next_delimiter = view.find(delimiter, pos);
    if (next_delimiter == kNpos) {
      return Status(StatusCode::INVALID_ARGUMENT, "Truncated multipart body.");
    }
    if (max_part_bytes > 0 && next_delimiter - pos > max_part_bytes) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Maximum part size exceeded.");
    }
    view.slice(pos, next_delimiter - pos, &part.value);
    pos = next_delimiter + delimiter.size();
    out.parts.push_back(std::move(part));
  }
  return Status();
}

}  // namespace xllm::api_service
