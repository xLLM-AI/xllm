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

// Type-erased RAII holder for a device completion primitive. Production code
// stores an aclrtEvent; the fence only owns its lifetime and invokes the
// supplied wait operation, which keeps ACL details out of this header.
using MegaMoeCompletionToken = std::shared_ptr<void>;

// Protects a ProcessGroup-owned KFC context from overlapping host launches.
//
// aclnnMegaMoe is asynchronous: returning from the host API only means that
// device work has been enqueued. The next host launch must therefore wait for
// the previous device launch before it reuses the same KFC context and HCCL
// windows. Otherwise, EP ranks can consume context mutated by a later layer,
// which was observed as nondeterministic EP8 output rather than a shape error.
// The fence is scoped to one communication resource, so unrelated NPU work
// remains asynchronous.
class MegaMoeLaunchFence final {
 public:
  using WaitForCompletion =
      std::function<bool(const MegaMoeCompletionToken& completion)>;
  using SynchronizeAbandonedLaunch = std::function<bool()>;

  class Lease final {
   public:
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&&) noexcept = delete;
    ~Lease();

    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    // Transfers ownership of the recorded device completion to the fence.
    // If this is not called (for example, because the host launch throws),
    // the destructor synchronizes the abandoned launch before unlocking.
    void record_completion(MegaMoeCompletionToken completion);

   private:
    friend class MegaMoeLaunchFence;

    Lease(MegaMoeLaunchFence* fence,
          std::unique_lock<std::mutex>&& lock,
          SynchronizeAbandonedLaunch synchronize_abandoned_launch);

    MegaMoeLaunchFence* fence_ = nullptr;
    std::unique_lock<std::mutex> lock_;
    SynchronizeAbandonedLaunch synchronize_abandoned_launch_;
  };

  Lease acquire(const WaitForCompletion& wait_for_completion,
                SynchronizeAbandonedLaunch synchronize_abandoned_launch);
  void drain(const WaitForCompletion& wait_for_completion);

 private:
  std::mutex mutex_;
  MegaMoeCompletionToken completion_;
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
  MegaMoeLaunchFence::Lease acquire_launch_lease();
  void record_launch_completion(MegaMoeLaunchFence::Lease& lease);

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
