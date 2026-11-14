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

#include <absl/hash/hash.h>

#include <chrono>
#include <iterator>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>

namespace xllm {

// Remembers cancelled reservation identities so a delayed AddNewRequests cannot
// resurrect them. The scheduler serializes access with its waiting-request
// lock.
class CancelledPDReservations final {
 public:
  using Clock = std::chrono::steady_clock;

  CancelledPDReservations() = default;
  CancelledPDReservations(const CancelledPDReservations&) = delete;
  CancelledPDReservations& operator=(const CancelledPDReservations&) = delete;

  // Explicit times support deterministic expiry tests; callers supply monotonic
  // times under the same lock, keeping the list ordered by expiry.
  void record(const std::string& request_id,
              const std::string& reservation_id,
              Clock::time_point now = Clock::now()) {
    prune(now);
    Identity identity{request_id, reservation_id};
    auto it = index_.find(identity);
    if (it != index_.end()) {
      it->second->expires_at = now + kRetention;
      cancellations_.splice(cancellations_.end(), cancellations_, it->second);
      return;
    }
    cancellations_.emplace_back(Entry{std::move(identity), now + kRetention});
    auto entry = std::prev(cancellations_.end());
    index_.emplace(entry->identity, entry);
  }

  bool contains(const std::string& request_id,
                const std::string& reservation_id,
                Clock::time_point now = Clock::now()) {
    prune(now);
    return index_.contains(Identity{request_id, reservation_id});
  }

 private:
  using Identity = std::pair<std::string, std::string>;
  struct Entry {
    Identity identity;
    Clock::time_point expires_at;
  };

  void prune(Clock::time_point now) {
    while (!cancellations_.empty() &&
           cancellations_.front().expires_at <= now) {
      index_.erase(cancellations_.front().identity);
      cancellations_.pop_front();
    }
  }

  static constexpr std::chrono::minutes kRetention{5};
  std::list<Entry> cancellations_;
  std::unordered_map<Identity, std::list<Entry>::iterator, absl::Hash<Identity>>
      index_;
};

}  // namespace xllm
