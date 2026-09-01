/* Copyright 2025-2026 The xLLM Authors.

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

// Residual-aware SJF request class for PD-prefill waiting requests. Mirrors
// the vLLM residual_sjf policy ordering: recovery requests first, then
// max-wait aged requests in FCFS order, then fresh requests by residual local
// prefill cost (prompt tokens minus local cached tokens, quantized to KV
// blocks).
enum class ResidualSJFRequestClass : int8_t {
  RECOVERY = 0,
  AGED = 1,
  FRESH = 2,
};

// Residual local prefill cost in KV blocks: floor(prompt tokens / block_size)
// minus the locally cached full blocks, matching the vLLM residual_sjf cost so
// equal residual block counts tie-break by arrival. When block_size is
// unavailable (0) the cost degrades to 0 (FCFS by arrival); production always
// supplies a real KV block size.
size_t residual_sjf_cost_blocks(size_t num_prompt_tokens,
                                size_t num_local_computed_blocks,
                                size_t block_size);

// Residual-aware SJF ordering for PD-prefill waiting requests:
// RECOVERY first, then AGED (max-wait) requests in FCFS order, then FRESH
// requests by residual cost with arrival-time tie-break.
class ResidualSJFComparator final : public PriorityComparator {
 public:
  using ResidualCostFn = std::function<size_t(const std::shared_ptr<Request>&)>;

  ResidualSJFComparator(ResidualCostFn residual_cost_blocks,
                        int32_t max_wait_ms,
                        absl::Time now);

  bool operator()(const std::shared_ptr<Request>& a,
                  const std::shared_ptr<Request>& b) const override;

 private:
  ResidualSJFRequestClass rank(const std::shared_ptr<Request>& request) const;

  ResidualCostFn residual_cost_blocks_;
  int32_t max_wait_ms_;
  absl::Time now_;
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

std::function<bool(const std::shared_ptr<Request>&,
                   const std::shared_ptr<Request>&)>
create_comparator(const std::string& priority_strategy, bool reverse = false);

}  // namespace xllm
