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

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "multimodal.pb.h"

namespace xllm {
namespace mm_service_utils {
namespace detail {

// The proto arrives as const& but is a mutable RPC request object that is no
// longer read after this call, so we const_cast once and move its large fields
// (urls, header values, embeddings) into MMContent instead of copying them.
inline bool append_mm_content(const ::xllm::proto::MMInputData& item,
                              MMContentVec& contents,
                              std::string& error_message) {
  auto& mutable_item = const_cast<::xllm::proto::MMInputData&>(item);
  const std::string& type = mutable_item.type();
  if (type == "text") {
    if (!mutable_item.has_text()) {
      error_message = "text content requires text.";
      return false;
    }
    contents.emplace_back(type, *mutable_item.release_text());
    return true;
  }

  if (type == "image_url") {
    if (!mutable_item.has_image_url()) {
      error_message = "image_url content requires image_url.";
      return false;
    }
    ImageURL image_url;
    auto* proto_url = mutable_item.mutable_image_url();
    image_url.url = std::move(*proto_url->release_url());
    for (auto& [key, value] : *proto_url->mutable_headers()) {
      image_url.headers[key] = std::move(value);
    }
    contents.emplace_back(type, std::move(image_url));
    contents.back().uuid = std::move(*mutable_item.mutable_uuid());
    return true;
  }

  if (type == "video_url") {
    if (!mutable_item.has_video_url()) {
      error_message = "video_url content requires video_url.";
      return false;
    }
    VideoURL video_url;
    auto* proto_url = mutable_item.mutable_video_url();
    video_url.url = std::move(*proto_url->release_url());
    for (auto& [key, value] : *proto_url->mutable_headers()) {
      video_url.headers[key] = std::move(value);
    }
    contents.emplace_back(type, std::move(video_url));
    contents.back().uuid = std::move(*mutable_item.mutable_uuid());
    return true;
  }

  if (type == "audio_url") {
    if (!mutable_item.has_audio_url()) {
      error_message = "audio_url content requires audio_url.";
      return false;
    }
    AudioURL audio_url;
    auto* proto_url = mutable_item.mutable_audio_url();
    audio_url.url = std::move(*proto_url->release_url());
    for (auto& [key, value] : *proto_url->mutable_headers()) {
      audio_url.headers[key] = std::move(value);
    }
    contents.emplace_back(type, std::move(audio_url));
    contents.back().uuid = std::move(*mutable_item.mutable_uuid());
    return true;
  }

  if (type == "image_embedding") {
    if (!mutable_item.has_image_embedding()) {
      error_message = "image_embedding content requires image_embedding data.";
      return false;
    }
    contents.emplace_back(type,
                          std::move(*mutable_item.mutable_image_embedding()));
    contents.back().uuid = std::move(*mutable_item.mutable_uuid());
    return true;
  }

  if (type == "video_embedding") {
    if (!mutable_item.has_video_embedding()) {
      error_message = "video_embedding content requires video_embedding data.";
      return false;
    }
    contents.emplace_back(type,
                          std::move(*mutable_item.mutable_video_embedding()));
    contents.back().uuid = std::move(*mutable_item.mutable_uuid());
    return true;
  }

  if (type == "audio_embedding") {
    if (!mutable_item.has_audio_embedding()) {
      error_message = "audio_embedding content requires audio_embedding data.";
      return false;
    }
    contents.emplace_back(type,
                          std::move(*mutable_item.mutable_audio_embedding()));
    contents.back().uuid = std::move(*mutable_item.mutable_uuid());
    return true;
  }

  error_message = "message content type is invalid.";
  return false;
}

}  // namespace detail

template <typename Call>
bool build_messages(const google::protobuf::RepeatedPtrField<
                        xllm::proto::MMChatMessage>& req_messages,
                    std::vector<Message>& out_messages,
                    std::shared_ptr<Call> call,
                    int image_limit) {
  out_messages.clear();
  out_messages.reserve(req_messages.size());

  for (const auto& req_message : req_messages) {
    MMContentVec contents;
    contents.reserve(req_message.content_size());
    std::string error_message;

    for (const auto& input : req_message.content()) {
      if (!detail::append_mm_content(input, contents, error_message)) {
        call->finish_with_error(StatusCode::INVALID_ARGUMENT, error_message);
        return false;
      }
    }

    out_messages.emplace_back(req_message.role(), std::move(contents));
    auto& msg = out_messages.back();

    if (req_message.has_tool_call_id()) {
      msg.tool_call_id = req_message.tool_call_id();
    }

    if (req_message.tool_calls_size() > 0) {
      Message::ToolCallVec tool_calls;
      tool_calls.reserve(req_message.tool_calls_size());
      for (const auto& tool_call : req_message.tool_calls()) {
        tool_calls.emplace_back();
        auto& tc = tool_calls.back();
        if (tool_call.has_id()) {
          tc.id = tool_call.id();
        }
        tc.type = tool_call.type();
        tc.function.name = tool_call.function().name();
        tc.function.arguments = tool_call.function().arguments();
      }
      msg.tool_calls = std::move(tool_calls);
    }
  }

  for (auto& msg : out_messages) {
    if (msg.calc_count("image_url") > image_limit) {
      call->finish_with_error(StatusCode::INVALID_ARGUMENT,
                              "Number of images in a single message exceeds "
                              "the allowed image limit.");
      return false;
    }
  }

  return true;
};

}  // namespace mm_service_utils
}  // namespace xllm
