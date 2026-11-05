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

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/model/model_input_types.h"
#include "util/timer.h"

namespace xllm {
namespace proto {
class PrefetchRequest;
}  // namespace proto

enum class PrefetchControl : uint8_t {
  CONTINUE = 0,
  STOP = 1,
};

struct PrefetchUnit final {
  // Offsets into StoragePrefetchRequest::transfer_infos. The unit starts at
  // the previous unit's end; gates occupy [begin, gated_end) and optional
  // blocks occupy [gated_end, end).
  size_t gated_end = 0;
  size_t end = 0;
  // A unit may have an optional cache whose allocation failed. Keeping this
  // bit separate from the flat range lets the worker report an optional miss
  // instead of treating an empty range as "not applicable".
  bool has_non_gated = false;
};

class StoragePrefetchRequest final {
 public:
  bool to_proto(proto::PrefetchRequest* request) const;
  bool from_proto(const proto::PrefetchRequest& request);

  std::vector<BlockTransferInfo> transfer_infos;
  std::vector<PrefetchUnit> units;

  bool valid() const {
    if (units.empty()) {
      return false;
    }
    size_t begin = 0;
    for (const PrefetchUnit& unit : units) {
      if (unit.gated_end <= begin || unit.gated_end > unit.end ||
          unit.end > transfer_infos.size()) {
        return false;
      }
      begin = unit.end;
    }
    return begin == transfer_infos.size() &&
           std::all_of(transfer_infos.begin(),
                       transfer_infos.end(),
                       [](const BlockTransferInfo& info) {
                         return info.transfer_type == TransferType::G2H;
                       });
  }

  size_t unit_count() const { return units.size(); }

  size_t batch_count(size_t batch_size) const {
    return (units.size() + batch_size - 1) / batch_size;
  }

  size_t batch_unit_count(size_t batch_index, size_t batch_size) const {
    const size_t begin = batch_index * batch_size;
    return std::min(batch_size, units.size() - begin);
  }
};

struct PrefetchSummary final {
  // Per-unit AND across all workers. gated_hits is used for the contiguous
  // full-cache prefix; non_gated_hits selects the deepest optional checkpoint
  // inside that prefix.
  std::vector<uint8_t> gated_hits;
  std::vector<uint8_t> non_gated_hits;
};

class PrefetchResult final {
 public:
  using StopPredicate = std::function<bool()>;
  // The second argument certifies that no worker can still write host blocks.
  using DoneCallback = std::function<void(PrefetchSummary, bool)>;

  PrefetchResult(size_t worker_count,
                 const StoragePrefetchRequest& request,
                 size_t batch_size,
                 int64_t timeout_ms,
                 StopPredicate stop_requested,
                 DoneCallback done)
      : workers_(worker_count),
        remaining_workers_(worker_count),
        unit_count_(request.unit_count()),
        batch_size_(batch_size),
        timeout_ms_(timeout_ms),
        stop_requested_(std::move(stop_requested)),
        done_(std::move(done)) {
    CHECK_GT(worker_count, 0u);
    CHECK_GT(batch_size_, 0u);
    CHECK(request.valid());
    CHECK(timeout_ms_ == -1 || timeout_ms_ > 0);
    CHECK(stop_requested_ != nullptr);
    CHECK(done_ != nullptr);
    for (WorkerProgress& worker : workers_) {
      worker.gated_hits.assign(unit_count_, 0);
      worker.non_gated_hits.assign(unit_count_, 0);
    }
  }

  size_t worker_count() const { return workers_.size(); }
  int64_t stream_idle_timeout_ms() const { return timeout_ms_; }
  size_t batch_wire_size() const { return batch_size_; }
  size_t batch_count() const {
    return (unit_count_ + batch_size_ - 1) / batch_size_;
  }
  void mark_worker_failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
    continue_prefetch_ = false;
  }

  // The wire response always has 2 * batch_size bytes, including zero padding
  // for a short final batch. Only the first real unit_count bytes from each
  // vector are committed to the aggregate summary.
  std::optional<PrefetchControl> record_batch_result(
      size_t worker_index,
      const std::vector<uint8_t>& gated_hits,
      const std::vector<uint8_t>& non_gated_hits) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_index >= workers_.size()) {
      continue_prefetch_ = false;
      return std::nullopt;
    }
    WorkerProgress& worker = workers_[worker_index];
    if (worker.state != WorkerState::WAITING_RESULT ||
        worker.batch_index >= batch_count() ||
        gated_hits.size() != batch_size_ ||
        non_gated_hits.size() != batch_size_) {
      continue_prefetch_ = false;
      return std::nullopt;
    }
    const size_t begin = worker.batch_index * batch_size_;
    const size_t count = std::min(batch_size_, unit_count_ - begin);
    std::copy(gated_hits.begin(),
              gated_hits.begin() + static_cast<std::ptrdiff_t>(count),
              worker.gated_hits.begin() + static_cast<std::ptrdiff_t>(begin));
    std::copy(
        non_gated_hits.begin(),
        non_gated_hits.begin() + static_cast<std::ptrdiff_t>(count),
        worker.non_gated_hits.begin() + static_cast<std::ptrdiff_t>(begin));

    const bool gated_complete =
        std::all_of(gated_hits.begin(),
                    gated_hits.begin() + static_cast<std::ptrdiff_t>(count),
                    [](uint8_t hit) { return hit != 0; });
    const bool timed_out =
        timeout_ms_ > 0 && timer_.elapsed_milliseconds() >= timeout_ms_;
    if (!gated_complete || timed_out || stop_requested_()) {
      continue_prefetch_ = false;
    }
    if (!continue_prefetch_ || worker.batch_index + 1 >= batch_count()) {
      worker.state = WorkerState::WAITING_CLOSE;
      return PrefetchControl::STOP;
    }
    ++worker.batch_index;
    return PrefetchControl::CONTINUE;
  }

  void begin_worker_rpc(size_t worker_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LT(worker_index, workers_.size());
    WorkerProgress& worker = workers_[worker_index];
    CHECK(worker.state == WorkerState::WAITING_RESULT);
    CHECK(!worker.rpc_pending);
    worker.rpc_pending = true;
    ++pending_worker_rpcs_;
  }

  void complete_worker_rpc(size_t worker_index, bool rejected_before_start) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CHECK_LT(worker_index, workers_.size());
      WorkerProgress& worker = workers_[worker_index];
      CHECK(worker.rpc_pending);
      worker.rpc_pending = false;
      CHECK_GT(pending_worker_rpcs_, 0u);
      --pending_worker_rpcs_;
      if (rejected_before_start) {
        failed_ = true;
        worker.quiescent = true;
        if (worker.state != WorkerState::ENDED) {
          worker.state = WorkerState::ENDED;
          CHECK_GT(remaining_workers_, 0u);
          --remaining_workers_;
        }
      }
    }
    finish_if_ready();
  }

  void mark_worker_ended(size_t worker_index,
                         bool worker_ok,
                         bool worker_quiescent) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      CHECK_LT(worker_index, workers_.size());
      WorkerProgress& worker = workers_[worker_index];
      if (worker.state == WorkerState::ENDED) {
        return;
      }
      if (!worker_ok || !worker_quiescent) {
        failed_ = true;
        continue_prefetch_ = false;
      }
      worker.quiescent = worker_quiescent;
      worker.state = WorkerState::ENDED;
      CHECK_GT(remaining_workers_, 0u);
      --remaining_workers_;
    }
    finish_if_ready();
  }

 private:
  enum class WorkerState : uint8_t {
    WAITING_RESULT = 0,
    WAITING_CLOSE = 1,
    ENDED = 2,
  };

  struct WorkerProgress {
    size_t batch_index = 0;
    WorkerState state = WorkerState::WAITING_RESULT;
    std::vector<uint8_t> gated_hits;
    std::vector<uint8_t> non_gated_hits;
    bool rpc_pending = false;
    bool quiescent = false;
  };

  void finish_if_ready() {
    DoneCallback done;
    PrefetchSummary summary;
    bool workers_quiescent = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // A close can precede the admission RPC reply. Wait for that reply
      // before deciding whether the reserved blocks need quarantine.
      if (remaining_workers_ != 0 || pending_worker_rpcs_ != 0 || !done_) {
        return;
      }

      // A failed worker cannot certify even the prefix it reported earlier.
      // Preserve the fail-closed behavior of the previous bitmap protocol.
      if (!failed_) {
        summary.gated_hits.assign(unit_count_, 1);
        summary.non_gated_hits.assign(unit_count_, 1);
        for (const WorkerProgress& progress : workers_) {
          for (size_t index = 0; index < unit_count_; ++index) {
            summary.gated_hits[index] =
                summary.gated_hits[index] && progress.gated_hits[index];
            summary.non_gated_hits[index] =
                summary.non_gated_hits[index] && progress.non_gated_hits[index];
          }
        }
      }
      workers_quiescent = std::all_of(
          workers_.begin(), workers_.end(), [](const WorkerProgress& worker) {
            return worker.quiescent;
          });
      done = std::move(done_);
    }
    done(std::move(summary), workers_quiescent);
  }

  mutable std::mutex mutex_;
  std::vector<WorkerProgress> workers_;
  size_t remaining_workers_ = 0;
  size_t pending_worker_rpcs_ = 0;
  size_t unit_count_ = 0;
  size_t batch_size_ = 0;
  int64_t timeout_ms_ = -1;
  StopPredicate stop_requested_;
  DoneCallback done_;
  Timer timer_;
  bool failed_ = false;
  bool continue_prefetch_ = true;
};

}  // namespace xllm
