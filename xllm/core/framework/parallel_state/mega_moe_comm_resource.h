/* Copyright 2026 The xLLM Authors.

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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "hccl/hccl.h"

namespace at {
class Tensor;
}  // namespace at

namespace xllm {

struct MegaMoeCommSpec {
  std::string group_name;
  HcclComm hccl_comm = nullptr;
  int32_t ep_world_size = 0;
  int32_t device_index = -1;
  int64_t max_num_tokens_per_rank = 0;
};

enum class MegaMoeCommRejectReason : int32_t {
  NONE = 0,
  EMPTY_GROUP,
  NULL_COMM,
  UNSUPPORTED_EP_WORLD_SIZE,
  INVALID_DEVICE_INDEX,
  INVALID_MAX_NUM_TOKENS_PER_RANK,
};

struct MegaMoeCommValidation {
  bool valid = false;
  MegaMoeCommRejectReason reason = MegaMoeCommRejectReason::EMPTY_GROUP;
};

struct MegaMoeCommSymbolStatus {
  bool available = false;
  std::string missing_symbol;
};

MegaMoeCommValidation validate_mega_moe_comm_spec(
    const MegaMoeCommSpec& spec);

MegaMoeCommSymbolStatus probe_mega_moe_comm_symbols();

struct MegaMoeBufferSpanValidation {
  bool valid = false;
  int32_t mismatched_rank = -1;
  uint64_t required_payload_size = 0;
  uint64_t accessible_span = 0;
};

// HcclGetHcclBuffer returns the local communication payload while
// HcclGetRemoteIpcHcclBuf may return a larger accessible span that includes
// auxiliary MC2 memory. Validate that every returned span covers the complete
// local payload before copying the communication context to an NPU tensor.
MegaMoeBufferSpanValidation validate_mega_moe_buffer_accessible_spans(
    uint64_t local_payload_size,
    const std::vector<uint64_t>& rank_accessible_spans);

class MegaMoeCommResource final {
 public:
  ~MegaMoeCommResource();

  MegaMoeCommResource(const MegaMoeCommResource&) = delete;
  MegaMoeCommResource& operator=(const MegaMoeCommResource&) = delete;
  MegaMoeCommResource(MegaMoeCommResource&&) = delete;
  MegaMoeCommResource& operator=(MegaMoeCommResource&&) = delete;

  static std::unique_ptr<MegaMoeCommResource> create(
      const MegaMoeCommSpec& spec);

  const at::Tensor& context_tensor() const;
  int64_t ccl_buffer_size() const;
  int64_t max_num_tokens_per_rank() const;

 private:
  class Impl;

  explicit MegaMoeCommResource(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

// A bounded, ProcessGroup-owned cache. One strong entry follows the HCCL
// communicator lifetime; layers retain only weak references. There is no
// process-global unbounded key map.
class MegaMoeCommResourceSlot final {
 public:
  using Factory = std::function<std::shared_ptr<MegaMoeCommResource>(
      const MegaMoeCommSpec&)>;

  std::shared_ptr<MegaMoeCommResource> acquire(
      const MegaMoeCommSpec& spec);
  std::shared_ptr<MegaMoeCommResource> acquire(
      const MegaMoeCommSpec& spec,
      const Factory& factory);
  void reset();
  void reset_for_teardown();

 private:
  static bool same_key(const MegaMoeCommSpec& lhs,
                       const MegaMoeCommSpec& rhs);

  std::mutex mutex_;
  std::optional<MegaMoeCommSpec> cached_spec_;
  std::shared_ptr<MegaMoeCommResource> resource_;
};

}  // namespace xllm
