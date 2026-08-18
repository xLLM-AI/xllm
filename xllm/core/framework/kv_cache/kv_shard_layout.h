/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>

namespace xllm {

class KVShardLayout final {
 public:
  static constexpr int64_t kInvalidSlot = -1;

  KVShardLayout(int32_t physical_block_size,
                int32_t dcp_size,
                int32_t dcp_rank);

  int32_t physical_block_size() const { return physical_block_size_; }
  int32_t dcp_size() const { return dcp_size_; }
  int32_t dcp_rank() const { return dcp_rank_; }
  int64_t logical_block_size() const {
    return static_cast<int64_t>(physical_block_size_) * dcp_size_;
  }

  int32_t owner_of(int64_t global_slot) const;
  bool owns(int64_t global_slot) const;
  int64_t localize(int64_t global_slot) const;
  int64_t globalize(int64_t local_slot) const;

 private:
  int32_t physical_block_size_ = 1;
  int32_t dcp_size_ = 1;
  int32_t dcp_rank_ = 0;
};

}  // namespace xllm
