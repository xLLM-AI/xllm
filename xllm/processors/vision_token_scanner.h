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

#include <cstddef>
#include <string_view>
#include <utility>

namespace xllm {

// Borrows the prompt and token strings, which must outlive the scanner.
// Calls to find_next must use nondecreasing offsets into the original prompt.
class VisionTokenScanner final {
 public:
  enum class TokenType { INVALID, IMAGE, VIDEO };

  VisionTokenScanner(std::string_view prompt,
                     std::string_view image_token,
                     std::string_view video_token)
      : prompt_(prompt),
        image_token_(image_token),
        video_token_(video_token),
        image_pos_(prompt.find(image_token)),
        video_pos_(prompt.find(video_token)) {}

  std::pair<TokenType, size_t> find_next(size_t begin) {
    // Retain the unconsumed match (including npos) instead of rescanning the
    // remaining prompt whenever only the other token type is consumed.
    if (image_pos_ != std::string_view::npos && image_pos_ < begin) {
      image_pos_ = prompt_.find(image_token_, begin);
    }
    if (video_pos_ != std::string_view::npos && video_pos_ < begin) {
      video_pos_ = prompt_.find(video_token_, begin);
    }
    if (image_pos_ == std::string_view::npos &&
        video_pos_ == std::string_view::npos) {
      return {TokenType::INVALID, std::string_view::npos};
    }
    return image_pos_ < video_pos_
               ? std::make_pair(TokenType::IMAGE, image_pos_)
               : std::make_pair(TokenType::VIDEO, video_pos_);
  }

 private:
  std::string_view prompt_;
  std::string_view image_token_;
  std::string_view video_token_;
  size_t image_pos_;
  size_t video_pos_;
};

}  // namespace xllm
