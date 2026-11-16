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

#include <butil/iobuf.h>

#include <mutex>

#include "worker.pb.h"

namespace xllm {

// One immutable input per DP group and engine step. Encoding is lazy so a
// successful SHM dispatch does not allocate or pack a BRPC payload. The input's
// borrowed tensor storage must remain immutable until every RPC completes.
class ForwardInputEncoding final {
 public:
  template <typename Input>
  bool encode(const Input& input,
              proto::PackedForwardInput* metadata,
              butil::IOBuf& attachment) {
    std::call_once(once_, [&] {
      valid_ = input.to_packed_attachment(&metadata_, attachment_);
    });
    if (!valid_) {
      return false;
    }
    *metadata = metadata_;
    // IOBuf copies retain blocks, including the tensor owners, without copying
    // payload bytes. Each RPC can release its attachment independently.
    attachment.append(attachment_);
    return true;
  }

 private:
  std::once_flag once_;
  bool valid_ = false;
  proto::PackedForwardInput metadata_;
  butil::IOBuf attachment_;
};

}  // namespace xllm
