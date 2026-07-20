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

#include "platform/mlu/mlu_rdma_memory_plan.h"

#include <glog/logging.h>

#include <limits>

namespace xllm::mlu {

RdmaMemoryPlan make_rdma_memory_plan(size_t logical_bytes, size_t block_count) {
  CHECK_GT(logical_bytes, static_cast<size_t>(0))
      << "logical byte size must be positive";
  CHECK_GT(block_count, static_cast<size_t>(0))
      << "block count must be positive";
  CHECK_EQ(logical_bytes % block_count, static_cast<size_t>(0))
      << "logical bytes must be divisible by block count";

  const size_t block_bytes = logical_bytes / block_count;
  CHECK_GT(block_bytes, static_cast<size_t>(0))
      << "block byte size must be positive";

  size_t registered_bytes = logical_bytes;
  if (kMinRdmaRegisterBytes > logical_bytes) {
    CHECK_LE(kMinRdmaRegisterBytes,
             std::numeric_limits<size_t>::max() - block_bytes + 1)
        << "registered byte size overflow";
    registered_bytes =
        ((kMinRdmaRegisterBytes + block_bytes - 1) / block_bytes) * block_bytes;
  }
  return RdmaMemoryPlan{logical_bytes, block_bytes, registered_bytes};
}

}  // namespace xllm::mlu
