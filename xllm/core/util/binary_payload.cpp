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

#include "core/util/binary_payload.h"

#include <utility>

namespace xllm {

BinaryPayload::BinaryPayload(const butil::IOBuf& data) : data_(data) {}

BinaryPayload::BinaryPayload(butil::IOBuf&& data)
    : data_(butil::IOBuf::Movable(data)) {}

BinaryPayload::BinaryPayload(std::string_view data) {
  data_.append(data.data(), data.size());
}

size_t BinaryPayload::size() const { return data_.size(); }

bool BinaryPayload::empty() const { return data_.empty(); }

bool BinaryPayload::valid_range(size_t offset, size_t length) const {
  return offset <= data_.size() && length <= data_.size() - offset;
}

bool BinaryPayload::copy_to(size_t offset, size_t length, void* target) const {
  if (!valid_range(offset, length) || (length > 0 && target == nullptr)) {
    return false;
  }
  return data_.copy_to(target, length, offset) == length;
}

bool BinaryPayload::copy_to(size_t offset,
                            size_t length,
                            std::string& target) const {
  if (!valid_range(offset, length)) {
    return false;
  }
  target.resize(length);
  if (length == 0) {
    return true;
  }
  if (data_.copy_to(target.data(), length, offset) != length) {
    target.clear();
    return false;
  }
  return true;
}

}  // namespace xllm
