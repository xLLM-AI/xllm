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

#include <cstdint>
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

bool DiTSourceDecoder::add_source(const proto::MediaSource& source,
                                  std::string default_name,
                                  Status& status) {
  Input input;
  input.name = !source.has_name() || source.name().empty()
                   ? std::move(default_name)
                   : source.name();
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
    std::string_view default_name,
    Status& status) {
  inputs_.reserve(inputs_.size() + sources.size());
  for (const proto::MediaSource& source : sources) {
    if (!add_source(source, std::string(default_name), status)) {
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
    inputs_.emplace_back(
        Input{.name = std::string(default_name), .encoded_data = source});
  }
}

bool DiTSourceDecoder::decode(const DecodeFn& decode_fn,
                              std::vector<NamedTensor>& outputs,
                              Status& status) const {
  if (inputs_.empty()) {
    return true;
  }

  std::vector<torch::Tensor> tensors(inputs_.size());
  std::vector<uint8_t> decoded(inputs_.size(), 0);
  const auto decode_one = [&](size_t index) {
    const Input& input = inputs_[index];
    std::string decoded_bytes;
    std::string_view raw_bytes;
    if (input.encoding == Encoding::BASE64) {
      const butil::StringPiece encoded(input.encoded_data.data(),
                                       input.encoded_data.size());
      if (!butil::Base64Decode(encoded, &decoded_bytes)) {
        return;
      }
      raw_bytes = decoded_bytes;
    } else {
      if (!request_payload_.copy_to(
              input.binary_offset, input.binary_length, decoded_bytes)) {
        return;
      }
      raw_bytes = decoded_bytes;
    }
    if (decode_fn(raw_bytes, tensors[index])) {
      decoded[index] = 1;
    }
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
    if (decoded[index] == 0) {
      status = Status(
          StatusCode::INVALID_ARGUMENT,
          "failed to decode media source at index " + std::to_string(index));
      return false;
    }
  }
  outputs.reserve(outputs.size() + inputs_.size());
  for (size_t index = 0; index < inputs_.size(); ++index) {
    outputs.emplace_back(NamedTensor{.name = inputs_[index].name,
                                     .tensor = std::move(tensors[index])});
  }
  return true;
}

}  // namespace xllm
