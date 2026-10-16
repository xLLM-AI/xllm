/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/request/dit_source_decoder.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include "butil/base64.h"
#include "core/util/threadpool.h"

namespace xllm {

DiTSourceDecoder::DiTSourceDecoder(const BinaryPayload& request_payload)
    : request_payload_(request_payload) {}

ThreadPool& DiTSourceDecoder::thread_pool() {
  static ThreadPool thread_pool(/*num_threads=*/8,
                                /*cpu_binding=*/false,
                                /*pool_name=*/"DiT.media_decode");
  return thread_pool;
}

namespace {

// Canonical name<->modality mapping: an unnamed source gets the canonical name
// of its modality, and a canonical name requires its matching modality.
struct CanonicalMediaName {
  std::string_view modality;
  std::string_view name;
};
constexpr std::array<CanonicalMediaName, 3> kCanonicalMediaNames{{
    {"image", "image"},
    {"video", "prompt_video"},
    {"audio", "prompt_audio"},
}};

}  // namespace

bool DiTSourceDecoder::add_source(const proto::MediaSource& source,
                                  Status& status) {
  Input input;
  input.modality = source.modality().empty() ? "image" : source.modality();
  const auto* canonical =
      std::find_if(kCanonicalMediaNames.begin(),
                   kCanonicalMediaNames.end(),
                   [&input](const CanonicalMediaName& entry) {
                     return entry.modality == input.modality;
                   });
  if (canonical == kCanonicalMediaNames.end()) {
    status = Status(StatusCode::INVALID_ARGUMENT,
                    "modality must be image, video, or audio");
    return false;
  }
  if (source.name().empty()) {
    input.name = std::string(canonical->name);
  } else {
    input.name = source.name();
    const auto* named = std::find_if(kCanonicalMediaNames.begin(),
                                     kCanonicalMediaNames.end(),
                                     [&input](const CanonicalMediaName& entry) {
                                       return entry.name == input.name;
                                     });
    if (named != kCanonicalMediaNames.end() &&
        input.modality != named->modality) {
      status = Status(StatusCode::INVALID_ARGUMENT,
                      "media source name and modality do not match");
      return false;
    }
  }
  if (source.type() == "base64") {
    input.encoded_data = source.base64();
  } else if (source.type() == "binary" && source.has_binary()) {
    const uint64_t offset = source.binary().offset();
    const uint64_t length = source.binary().length();
    const uint64_t payload_size = request_payload_.size();
    if (offset > payload_size || length > payload_size - offset) {
      status = Status(StatusCode::INVALID_ARGUMENT, "invalid media source");
      return false;
    }
    input.encoding = Encoding::BINARY;
    input.binary_offset = static_cast<size_t>(offset);
    input.binary_length = static_cast<size_t>(length);
  } else {
    status = Status(StatusCode::INVALID_ARGUMENT, "invalid media source");
    return false;
  }
  inputs_.emplace_back(std::move(input));
  return true;
}

bool DiTSourceDecoder::add_sources(
    const google::protobuf::RepeatedPtrField<proto::MediaSource>& sources,
    Status& status) {
  inputs_.reserve(inputs_.size() + sources.size());
  for (const proto::MediaSource& source : sources) {
    if (!add_source(source, status)) {
      return false;
    }
  }
  return true;
}

void DiTSourceDecoder::add_sources(
    const google::protobuf::RepeatedPtrField<std::string>& sources,
    std::string_view default_name) {
  inputs_.reserve(inputs_.size() + sources.size());
  for (const std::string& source : sources) {
    inputs_.emplace_back(Input{.name = std::string(default_name),
                               .modality = "image",
                               .encoded_data = source});
  }
}

bool DiTSourceDecoder::decode(const DecodeFn& decode_fn,
                              std::vector<MediaNamedTensor>& outputs,
                              Status& status) const {
  if (inputs_.empty()) {
    return true;
  }

  std::vector<std::vector<MediaNamedTensor>> decoded(inputs_.size());
  std::vector<uint8_t> is_decoded(inputs_.size(), false);
  const auto decode_one = [&](size_t index) {
    const Input& input = inputs_[index];
    std::string decoded_bytes;
    if (input.encoding == Encoding::BASE64) {
      const butil::StringPiece encoded(input.encoded_data.data(),
                                       input.encoded_data.size());
      if (!butil::Base64Decode(encoded, &decoded_bytes)) {
        return;
      }
    } else {
      if (!request_payload_.copy_to(
              input.binary_offset, input.binary_length, decoded_bytes)) {
        return;
      }
    }
    is_decoded[index] =
        decode_fn(input.name, input.modality, decoded_bytes, decoded[index]);
  };

  if (inputs_.size() == 1) {
    decode_one(/*index=*/0);
  } else {
    TaskGroup tasks(static_cast<int32_t>(inputs_.size()));
    for (size_t index = 0; index < inputs_.size(); ++index) {
      thread_pool().schedule(
          tasks.wrap([&decode_one, index]() { decode_one(index); }));
    }
    tasks.wait();
  }

  for (size_t index = 0; index < inputs_.size(); ++index) {
    if (!is_decoded[index]) {
      status = Status(
          StatusCode::INVALID_ARGUMENT,
          "failed to decode media source at index " + std::to_string(index));
      return false;
    }
  }
  size_t output_count = outputs.size();
  for (const auto& source_outputs : decoded) {
    output_count += source_outputs.size();
  }
  outputs.reserve(output_count);
  for (std::vector<MediaNamedTensor>& source_outputs : decoded) {
    outputs.insert(outputs.end(),
                   std::make_move_iterator(source_outputs.begin()),
                   std::make_move_iterator(source_outputs.end()));
  }
  return true;
}

}  // namespace xllm
