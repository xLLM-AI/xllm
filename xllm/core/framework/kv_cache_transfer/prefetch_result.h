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
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "framework/model/model_input_params.h"
#include "util/timer.h"

namespace xllm {

enum class PrefetchControl : uint8_t {
  CONTINUE = 0,
  STOP = 1,
};

class StoragePrefetchRequest final {
 public:
  std::vector<BlockTransferInfo> transfer_infos;
  std::vector<uint32_t> unit_end_offsets;
  std::vector<uint32_t> batch_end_unit_offsets;

  bool valid() const {
    if (transfer_infos.empty() || unit_end_offsets.empty() ||
        batch_end_unit_offsets.empty()) {
      return false;
    }

    uint32_t previous = 0;
    for (uint32_t offset : unit_end_offsets) {
      if (offset <= previous || offset > transfer_infos.size()) {
        return false;
      }
      previous = offset;
    }
    if (unit_end_offsets.back() != transfer_infos.size()) {
      return false;
    }

    previous = 0;
    for (uint32_t offset : batch_end_unit_offsets) {
      if (offset <= previous || offset > unit_end_offsets.size() ||
          offset - previous > std::numeric_limits<uint8_t>::max()) {
        return false;
      }
      previous = offset;
    }
    if (batch_end_unit_offsets.back() != unit_end_offsets.size()) {
      return false;
    }

    return std::all_of(transfer_infos.begin(),
                       transfer_infos.end(),
                       [](const BlockTransferInfo& info) {
                         return info.transfer_type == TransferType::G2H;
                       });
  }

  size_t batch_count() const { return batch_end_unit_offsets.size(); }

  size_t batch_unit_begin(size_t batch_index) const {
    return batch_index == 0 ? 0 : batch_end_unit_offsets[batch_index - 1];
  }

  size_t batch_unit_count(size_t batch_index) const {
    CHECK_LT(batch_index, batch_end_unit_offsets.size());
    return batch_end_unit_offsets[batch_index] - batch_unit_begin(batch_index);
  }

  std::pair<size_t, size_t> batch_transfer_range(size_t batch_index) const {
    CHECK_LT(batch_index, batch_end_unit_offsets.size());
    const size_t unit_begin = batch_unit_begin(batch_index);
    const size_t unit_end = batch_end_unit_offsets[batch_index];
    const size_t transfer_begin =
        unit_begin == 0 ? 0 : unit_end_offsets[unit_begin - 1];
    return {transfer_begin, unit_end_offsets[unit_end - 1]};
  }

  std::optional<uint8_t> count_prefix_hit_units(
      size_t batch_index,
      const std::vector<uint8_t>& logical_hits) const {
    if (batch_index >= batch_count()) {
      return std::nullopt;
    }
    const auto [transfer_begin, transfer_end] =
        batch_transfer_range(batch_index);
    if (logical_hits.size() != transfer_end - transfer_begin) {
      return std::nullopt;
    }

    const size_t unit_begin = batch_unit_begin(batch_index);
    const size_t unit_end = batch_end_unit_offsets[batch_index];
    size_t hit_units = 0;
    size_t local_begin = 0;
    for (size_t unit = unit_begin; unit < unit_end; ++unit) {
      const size_t local_end = unit_end_offsets[unit] - transfer_begin;
      const bool unit_hit = std::all_of(
          logical_hits.begin() + static_cast<std::ptrdiff_t>(local_begin),
          logical_hits.begin() + static_cast<std::ptrdiff_t>(local_end),
          [](uint8_t hit) { return hit != 0; });
      if (!unit_hit) {
        break;
      }
      ++hit_units;
      local_begin = local_end;
    }
    return static_cast<uint8_t>(hit_units);
  }
};

class PrefetchResult final {
 public:
  using StopPredicate = std::function<bool()>;
  // The second argument certifies that no worker can still write host blocks.
  using DoneCallback = std::function<void(size_t, bool)>;

  PrefetchResult(size_t worker_count,
                 std::vector<uint32_t> batch_end_unit_offsets,
                 int64_t timeout_ms,
                 StopPredicate stop_requested,
                 DoneCallback done)
      : workers_(worker_count),
        remaining_workers_(worker_count),
        batch_end_unit_offsets_(std::move(batch_end_unit_offsets)),
        timeout_ms_(timeout_ms),
        stop_requested_(std::move(stop_requested)),
        done_(std::move(done)) {
    CHECK_GT(worker_count, 0u);
    CHECK(!batch_end_unit_offsets_.empty());
    CHECK(timeout_ms_ == -1 || timeout_ms_ > 0);
    CHECK(stop_requested_ != nullptr);
    CHECK(done_ != nullptr);
  }

  size_t worker_count() const { return workers_.size(); }
  int64_t stream_idle_timeout_ms() const { return timeout_ms_; }

  std::optional<PrefetchControl> record_batch_result(size_t worker_index,
                                                     uint8_t prefix_hit_units) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_index >= workers_.size()) {
      return std::nullopt;
    }

    WorkerProgress& worker = workers_[worker_index];
    if (worker.state != WorkerState::WAITING_RESULT ||
        worker.batch_index >= batch_end_unit_offsets_.size()) {
      return std::nullopt;
    }

    const size_t batch_begin =
        worker.batch_index == 0
            ? 0
            : batch_end_unit_offsets_[worker.batch_index - 1];
    const size_t batch_end = batch_end_unit_offsets_[worker.batch_index];
    const size_t batch_units = batch_end - batch_begin;
    if (prefix_hit_units > batch_units) {
      return std::nullopt;
    }

    worker.hit_units += prefix_hit_units;
    const bool last_batch =
        worker.batch_index + 1 == batch_end_unit_offsets_.size();
    const bool timed_out =
        timeout_ms_ > 0 && timer_.elapsed_milliseconds() >= timeout_ms_;
    const bool stopped = stop_requested_();

    if (prefix_hit_units != batch_units || last_batch || timed_out || stopped ||
        failed_) {
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
    size_t hit_units = 0;
    WorkerState state = WorkerState::WAITING_RESULT;
    bool rpc_pending = false;
    bool quiescent = false;
  };

  void finish_if_ready() {
    DoneCallback done;
    size_t common_hit_units = 0;
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
        common_hit_units = workers_.front().hit_units;
        for (const WorkerProgress& progress : workers_) {
          common_hit_units = std::min(common_hit_units, progress.hit_units);
        }
      }
      workers_quiescent = std::all_of(
          workers_.begin(), workers_.end(), [](const WorkerProgress& worker) {
            return worker.quiescent;
          });
      done = std::move(done_);
    }
    done(common_hit_units, workers_quiescent);
  }

  mutable std::mutex mutex_;
  std::vector<WorkerProgress> workers_;
  size_t remaining_workers_ = 0;
  size_t pending_worker_rpcs_ = 0;
  std::vector<uint32_t> batch_end_unit_offsets_;
  int64_t timeout_ms_ = -1;
  StopPredicate stop_requested_;
  DoneCallback done_;
  Timer timer_;
  bool failed_ = false;
};

}  // namespace xllm
