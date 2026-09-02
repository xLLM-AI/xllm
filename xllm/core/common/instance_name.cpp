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

#include "core/common/instance_name.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

namespace xllm {
namespace {

constexpr uint64_t kRequestIdCounterBits = 48;
constexpr uint64_t kRequestIdCounterMask = (1ULL << kRequestIdCounterBits) - 1;

std::atomic<uint64_t> g_request_id_seq{1};

}  // namespace

int64_t next_request_id() {
  const uint64_t seq = g_request_id_seq.fetch_add(1, std::memory_order_relaxed);
  const uint64_t instance_key =
      InstanceName::name()->get_name_hash_value() & 0xFFFFULL;
  return static_cast<int64_t>((instance_key << kRequestIdCounterBits) |
                              (seq & kRequestIdCounterMask));
}

std::string generate_request_id(std::string_view prefix) {
  const uint64_t request_id = static_cast<uint64_t>(next_request_id());
  std::string encoded;
  encoded.reserve(prefix.size() + 20);
  encoded.append(prefix);
  encoded.append(std::to_string(request_id));
  return encoded;
}

}  // namespace xllm
