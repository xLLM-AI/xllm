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

#pragma once

#include <google/protobuf/repeated_ptr_field.h>

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "common.pb.h"
#include "core/common/types.h"
#include "core/util/binary_payload.h"
#include "framework/request/dit_input_sources.h"

namespace xllm {

class ThreadPool;

class DiTSourceDecoder final {
 public:
  using DecodeFn =
      std::function<bool(std::string_view raw_bytes, torch::Tensor& tensor)>;

  // Source views refer to the request proto or request_payload. Keep both
  // alive until this decoder and its synchronous decode calls are finished.
  explicit DiTSourceDecoder(const BinaryPayload& request_payload);

  bool add_source(const proto::MediaSource& source,
                  std::string default_name,
                  Status& status);

  bool add_sources(
      const google::protobuf::RepeatedPtrField<proto::MediaSource>& sources,
      std::string_view default_name,
      Status& status);

  void add_sources(
      const google::protobuf::RepeatedPtrField<std::string>& sources,
      std::string_view default_name);

  bool decode(const DecodeFn& decode_fn,
              std::vector<NamedTensor>& outputs,
              Status& status) const;

 private:
  enum class Encoding : uint8_t {
    BASE64,
    BINARY,
  };

  struct Input {
    std::string name;
    std::string_view encoded_data;
    size_t binary_offset = 0;
    size_t binary_length = 0;
    Encoding encoding = Encoding::BASE64;
  };

  static ThreadPool& thread_pool();

  const BinaryPayload& request_payload_;
  std::vector<Input> inputs_;
};

}  // namespace xllm
