/* Copyright 2025-2026 The xLLM Authors.

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
#include <absl/time/time.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "common.pb.h"
#include "framework/request/request.h"

namespace xllm {
class PriorityComparator {
 public:
  virtual bool operator()(const std::shared_ptr<Request>& a,
                          const std::shared_ptr<Request>& b) const = 0;
  virtual ~PriorityComparator() = default;
};

struct FCFSComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct StrictPriorityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DeadlineComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct SJFComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeDeadlineComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DensityWithAntiStarveComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeDensityWithAntiStarveComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeDensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct UrgencyDensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct UrgencyPriorityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

struct DecodeUrgencyDensityComparator : public PriorityComparator {
  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;
};

// ShortRequestFirst request class for PD-prefill waiting requests.
enum class ShortRequestFirstRequestClass : int8_t {
  SHORT = 0,
  LONG = 1,
};

// Classify a PD-prefill waiting request for ShortRequestFirst ordering by
// prompt length.
ShortRequestFirstRequestClass classify_short_request_first(Request& request,
                                                           int32_t threshold);

// Select the single LONG request that may be promoted ahead of waiting SHORT
// requests: the LONG request with the oldest created_time, promoted only when
// its wait reaches long_max_wait_ms and both SHORT and LONG requests are
// waiting. Returns nullptr when no promotion applies. `requests` is any range
// yielding std::shared_ptr<Request> (e.g. RequestPriorityQueue or a vector),
// so the caller can scan the live queue without copying it.
template <typename RequestRange>
std::shared_ptr<Request> select_short_request_first_promoted(
    const RequestRange& requests,
    int32_t threshold,
    double long_max_wait_ms,
    absl::Time now) {
  std::shared_ptr<Request> oldest_long;
  bool has_short = false;
  bool has_long = false;
  for (const auto& request : requests) {
    const ShortRequestFirstRequestClass request_class =
        classify_short_request_first(*request, threshold);
    if (request_class == ShortRequestFirstRequestClass::SHORT) {
      has_short = true;
    } else if (request_class == ShortRequestFirstRequestClass::LONG) {
      has_long = true;
      if (oldest_long == nullptr ||
          request->created_time() < oldest_long->created_time()) {
        oldest_long = request;
      }
    }
  }
  if (long_max_wait_ms > 0.0 && has_short && has_long &&
      oldest_long != nullptr &&
      now - oldest_long->created_time() >=
          absl::Milliseconds(long_max_wait_ms)) {
    return oldest_long;
  }
  return nullptr;
}

// ShortRequestFirst ordering: the promoted LONG head first (when non-null),
// then SHORT requests, then the remaining LONG requests. Within the same
// class, older requests (created_time) come first.
class ShortRequestFirstComparator final : public PriorityComparator {
 public:
  ShortRequestFirstComparator(int32_t threshold,
                              std::shared_ptr<Request> promoted);

  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;

 private:
  int32_t short_request_first_rank(
      const std::shared_ptr<Request>& request) const;

  int32_t threshold_;
  std::shared_ptr<Request> promoted_;
};

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy, bool reverse = false);

}  // namespace xllm
