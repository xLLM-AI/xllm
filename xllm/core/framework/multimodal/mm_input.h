/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "core/framework/multimodal/embedding_output.h"
#include "mm_type.h"

namespace xllm {

using EmbeddingInput = EmbeddingOutput;

struct MMInputItem {
  class IVisitor {
   public:
    virtual ~IVisitor() = default;
    virtual bool visit(const MMInputItem& item) = 0;
  };

  void clear() {
    type = MMType::NONE;
    raw_data.clear();
  }

  std::optional<torch::Tensor> get_decode_data(MMType type_) const {
    if (type_ == MMType::IMAGE) {
      return decode_image;
    } else if (type_ == MMType::VIDEO) {
      return decode_video;
    } else if (type_ == MMType::AUDIO) {
      return decode_audio;
    } else {
      return std::nullopt;
    }
  }

  bool has_type(MMType type_) const { return (type & type_) != 0; }

  bool is_embedding() const { return embedding.embedding.defined(); }

  uint32_t type = MMType::NONE;

  std::string raw_data;  // binary

  torch::Tensor decode_image;  // image: rgb, [c,h,w], uint8
  torch::Tensor decode_video;  // video: rgb, [t,c,h,w], uint8
  torch::Tensor decode_audio;  // audio: mono, [t], float32

  VideoMetadata video_meta;
  AudioMetadata audio_meta;
  EmbeddingInput embedding;
};

struct MMPayload {
  MMPayload() = default;
  explicit MMPayload(std::string data, size_t offset = 0)
      : data(std::move(data)), offset(offset) {}

  bool get(std::string& value, size_t len) {
    if (len == data.size()) {
      value = std::move(data);
      return true;
    }

    if (data.size() - offset < len) {
      return false;
    }

    value = data.substr(offset, len);
    offset += len;

    return true;
  }

  std::string data;
  size_t offset;
};

class MMInput {
 public:
  MMInput() = default;
  explicit MMInput(std::string payload) : payload_(std::move(payload)) {}

  bool empty() const { return items_.empty(); }
  void clear() { items_.clear(); }
  size_t size() const { return items_.size(); }

  const std::vector<MMInputItem>& items() const { return items_; }

  void insert(const std::vector<MMInputItem>& inputs) {
    items_.insert(items_.end(), inputs.begin(), inputs.end());
  }

  std::vector<torch::Tensor> get_decode_data(MMType type) const {
    std::vector<torch::Tensor> vec;

    for (const auto& item : items_) {
      if (item.has_type(type)) {
        if (item.is_embedding()) {
          continue;
        }
        auto t = item.get_decode_data(type);
        if (t) {
          vec.emplace_back(*t);
        }
      }
    }
    return vec;
  }

  bool foreach (MMInputItem::IVisitor& v) const;

  MMPayload& payload() { return payload_; }
  const MMPayload& payload() const { return payload_; }

 private:
  MMPayload payload_;
  std::vector<MMInputItem> items_;
};

enum class MMErrCode : uint8_t {
  SUCCESS = 0,
  LOAD_DATA_ERR = 1,
  LOAD_HTTP_ERR = 2,
  LOAD_LOCAL_ERR = 3,
  INVALID_URL_ERR = 4,
  PARSE_EMB_ERR = 5,
  DECODE_ERR = 6,
  HANDLER_ERR = 7,
};

inline const char* MMErrToString(MMErrCode code) {
  switch (code) {
    case MMErrCode::LOAD_DATA_ERR:
      return "Failed to load multimodal input from data url.";
    case MMErrCode::LOAD_HTTP_ERR:
      return "Failed to download multimodal input from http url.";
    case MMErrCode::LOAD_LOCAL_ERR:
      return "Failed to download multimodal input from local url.";
    case MMErrCode::PARSE_EMB_ERR:
      return "Failed to parse input multimodal embedding.";
    case MMErrCode::DECODE_ERR:
      return "Failed to decode multimodal input.";
    case MMErrCode::INVALID_URL_ERR:
      return "Multimodal input url must be data URL (base64 or binary) / "
             "http(s) URL / local file URL.";
    case MMErrCode::HANDLER_ERR:
      return "Unsupported multimodal input type.";
    default:
      return "Unknown error occurred when loading/decoding multimodal input.";
  }
}

class MMHandlerSet;
class ThreadPool;
class MMInputTransfer {
 public:
  MMInputTransfer();
  ~MMInputTransfer();

  MMErrCode trans(const std::vector<Message>& messages, MMInput& inputs);

 private:
  MMErrCode trans_parallel(const MMContentVec& mmc,
                           std::vector<MMInputItem>& inputs,
                           MMPayload& payload);
  MMErrCode trans(const MMContentVec& mmc,
                  std::vector<MMInputItem>& inputs,
                  MMPayload& payload);

  std::unique_ptr<MMHandlerSet> mm_handlers_;
  std::unique_ptr<ThreadPool> threadpool_;
};

}  // namespace xllm
