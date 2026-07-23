/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "lora_request.h"

namespace xllm {

// LoRARegistry owns the name<->int_id map of loaded adapters and refcounts
// them so unload can drain in-flight requests before releasing weights.
//
// Threading model:
//   - Reads (lookup / list / has) hold a shared_lock.
//   - Writes (register / unregister) hold a unique_lock.
// This lets a hot request path resolve model="biz-A" -> id concurrently
// while the API handler serialises adapter mutations.
//
// Reference-count semantics (drain-on-unload, mirrors vLLM behaviour):
//   - lookup_and_pin() atomically resolves the name and increments refcount.
//     Callers MUST call unpin() when the request finishes.
//   - unregister() marks the adapter as "unloading"; new lookup_and_pin()
//     for that name returns nullopt. The actual weight removal is postponed
//     to the moment refcount drops to zero.
class LoRARegistry {
 public:
  LoRARegistry() = default;

  // Register a name -> id binding. Returns the assigned int_id.
  //
  // Idempotency (matches vLLM /v1/load_lora_adapter): if `lora_name` is
  // already registered and points to the same path, we return the existing
  // int_id and no state changes. If the same name is registered with a
  // different path, this is a caller error and we log + reject unless the
  // caller opted into "load_inplace" (P1).
  //
  // int_id numbering is monotonically increasing and never reuses freed ids
  // for the lifetime of the process, so an adapter that was unloaded and
  // later reloaded gets a fresh id. This keeps stale references safe.
  std::optional<uint64_t> register_adapter(const LoRARequest& req);

  // Return the int_id + full LoRARequest for a given name if the adapter is
  // currently loaded and NOT marked for unload. Also increments the pin
  // count so the adapter cannot be freed underneath the caller.
  //
  // Returns std::nullopt if the name is unknown, or if the adapter is being
  // drained.
  struct PinnedAdapter {
    uint64_t int_id;
    LoRARequest req;
  };
  std::optional<PinnedAdapter> lookup_and_pin(const std::string& lora_name);

  // Decrement the pin count of an adapter previously returned by
  // lookup_and_pin. When the count hits zero AND the adapter is marked for
  // unload, the entry is finally removed.
  void unpin(uint64_t int_id);

  // Read-only lookup; does NOT modify refcount. Used by the /v1/models and
  // /v1/lora_adapters endpoints to enumerate.
  std::optional<LoRARequest> lookup(const std::string& lora_name) const;
  std::optional<LoRARequest> lookup(uint64_t int_id) const;

  // Mark adapter as unloading. If refcount is zero, remove immediately;
  // otherwise defer removal until unpin() drops the last reference.
  //
  // Returns true if the adapter existed (whether removed synchronously or
  // scheduled for drain).
  bool unregister(const std::string& lora_name);

  // P1-A.3: register a callback fired at the moment an adapter entry is
  // finally erased from the map (either synchronously from unregister when
  // refcount is 0, or deferred from unpin when the last ref drops after
  // an unloading mark). The callback receives the int_id so downstream
  // pools (per_proj_device_pool_, host cache, metric bvars) can free
  // their per-adapter state at the same moment the registry forgets it.
  //
  // Called with the registry mutex held; keep the callback body short.
  using FinalRemovalCallback = std::function<void(uint64_t int_id)>;
  void set_on_final_removal(FinalRemovalCallback cb) {
    std::unique_lock lock(mu_);
    on_final_removal_ = std::move(cb);
  }

  // P1-D: symmetric hook fired when register_adapter first creates an
  // entry (idempotent no-ops on re-register skip the callback). Used by
  // LoRAMetrics to allocate its per-adapter bvars. Called with the
  // registry mutex held; keep the callback body short.
  using RegisterCallback =
      std::function<void(uint64_t int_id, const std::string& lora_name)>;
  void set_on_register(RegisterCallback cb) {
    std::unique_lock lock(mu_);
    on_register_ = std::move(cb);
  }

  // Enumerate currently-loaded adapter names. Used by /v1/models.
  std::vector<LoRARequest> list() const;

  size_t size() const;

  // P1-A.4: probe whether an int_id still has an entry (may be unloading).
  // Used by /v1/unload_lora_adapter drain polling to know when the last
  // in-flight request finished and the entry was erased.
  bool contains(uint64_t int_id) const;

 private:
  struct Entry {
    LoRARequest req;
    // Live in-flight requests. When >0, the entry cannot be freed.
    int32_t pin_count = 0;
    // A caller has requested unregister but requests are still using it.
    bool unloading = false;
  };

  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, uint64_t> name_to_id_;
  std::unordered_map<uint64_t, Entry> id_to_entry_;
  std::atomic<uint64_t> next_id_{1};
  FinalRemovalCallback on_final_removal_;
  RegisterCallback on_register_;
};

}  // namespace xllm
