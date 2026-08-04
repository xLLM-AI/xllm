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

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/eplb/eplb_options.h"

namespace xllm {

// Enumerates the concrete EPLB rebalance strategies that MakeEplbPolicy can
// materialize. Adding a new kind: extend this enum, the switch in
// MakeEplbPolicy, and the string parser eplb_policy_kind_from_string() so
// operators can select it from --eplb_policy_kind / eplb config JSON.
enum class EplbPolicyKind : int8_t {
  // Historical xLLM greedy replica selection and LPT balanced_pack.
  GREEDY = 0,
  // Max-load-reduction replica selection followed by strict equal-cardinality
  // LPT packing across the all-connected HCCS super-node.
  BALANCED = 1,
};

// Parse a case-insensitive policy string into the enum. Historical policy
// names map to BALANCED. Unknown input maps to GREEDY so a bad flag does not
// stop the rebalance loop.
EplbPolicyKind eplb_policy_kind_from_string(const std::string& kind);

// Abstract base for pluggable EPLB rebalance strategies. Concrete
// implementations own their prior-load state and produce a new
// [layer_num x device_num x device_experts_num] int32 distribution plus a
// per-layer update mask on each call.
class IEplbPolicy {
 public:
  virtual ~IEplbPolicy() = default;

  // Recalculate expert distribution based on the latest workload snapshot.
  // Input:  expert_load - shape [layer_num, num_experts] int64
  //         physical_expert_load - optional measured shape
  //             [layer_num, device_num, device_experts_num] int64
  // Output: pair<expert_distribution, update_flags>
  //           expert_distribution: shape [layer_num, device_num,
  //                                       device_experts_num], int32
  //                                stable snapshot across subsequent calls
  //           update_flags:        per-layer bool, true = new plan differs
  //                                enough from previous and should be pushed
  virtual std::pair<torch::Tensor, std::vector<bool>> rebalance_experts(
      torch::Tensor expert_load,
      torch::Tensor physical_expert_load = torch::Tensor()) = 0;

  // Seed the policy with the placement already active in the model. Managers
  // call this once before their worker threads start so the first rebalance
  // compares against the real current layout instead of publishing blindly.
  virtual void initialize_distribution(
      const torch::Tensor& current_distribution) {
    (void)current_distribution;
  }

  // Finalize or roll back the most recently proposed update for one layer.
  // Stateless policies may keep the default no-op implementation.
  virtual void commit_layer(int32_t layer_id) { (void)layer_id; }
  virtual void abort_layer(int32_t layer_id) { (void)layer_id; }

  // Short human-readable name of the concrete policy. Used only for
  // observability (heartbeat / rebalance logs) so operators can see which
  // strategy actually planned the round without cross-referencing the flag.
  // Default returns "unknown"; every concrete subclass MUST override.
  virtual std::string name() const { return "unknown"; }
};

// Shared orchestration for placement policies. Concrete policies only build a
// candidate assignment for one layer; this class owns publication, benefit
// gating, and rollback state.
class EplbPolicyBase : public IEplbPolicy {
 public:
  EplbPolicyBase(int32_t device_experts_num,
                 int32_t device_num,
                 int32_t layer_num,
                 EplbOptions options);

  std::pair<torch::Tensor, std::vector<bool>> rebalance_experts(
      torch::Tensor expert_load,
      torch::Tensor physical_expert_load = torch::Tensor()) final;

  void initialize_distribution(const torch::Tensor& current_distribution) final;

  void commit_layer(int32_t layer_id) final;
  void abort_layer(int32_t layer_id) final;

 protected:
  virtual std::vector<int64_t> plan_layer(
      const std::vector<int64_t>& expert_loads_host,
      const std::vector<int64_t>& previous_assignment) = 0;

  int32_t device_experts_num_;
  int32_t device_num_;
  int32_t layer_num_;
  torch::Tensor expert_distribution_;
  EplbOptions options_;
  std::vector<std::vector<int64_t>> prev_layer_assignment_;
  struct LayerRollbackState {
    bool valid = false;
    torch::Tensor expert_distribution;
    std::vector<int64_t> prev_assignment;
  };
  std::vector<LayerRollbackState> rollback_states_;
};

// Historical xLLM greedy replica selection and LPT packing policy.
class GreedyEplbPolicy final : public EplbPolicyBase {
 public:
  GreedyEplbPolicy(int32_t device_experts_num,
                   int32_t device_num,
                   int32_t layer_num,
                   EplbOptions options);

  std::string name() const override { return "greedy"; }

 protected:
  std::vector<int64_t> plan_layer(
      const std::vector<int64_t>& expert_loads_host,
      const std::vector<int64_t>& previous_assignment) override;

 private:
  // Pure host-vector LPT balanced pack. Returns per-device assignment as a
  // flat row-major vector of length `device_num_ * device_experts_num_`
  // (unassigned == -1), or empty on failure.
  std::vector<int64_t> compute_balanced_pack_host(
      const std::vector<int64_t>& expert_loads_host);

  // Companion redundancy allocator. Returns
  //   (updated_weights, redundancy_map)
  // where redundancy_map is row-major of shape
  // [num_experts, redundancy_experts] with unfilled slots == -1.
  std::pair<std::vector<int64_t>, std::vector<int64_t>>
  update_origin_weights_host(const std::vector<int64_t>& expert_loads_host,
                             int32_t redundancy_experts);
};

// Uses max-load-reduction replica selection followed by strict
// equal-cardinality LPT packing across the all-connected HCCS super-node.
class BalancedEplbPolicy final : public EplbPolicyBase {
 public:
  BalancedEplbPolicy(int32_t device_experts_num,
                     int32_t device_num,
                     int32_t layer_num,
                     EplbOptions options);

  std::string name() const override { return "balanced"; }

 protected:
  std::vector<int64_t> plan_layer(
      const std::vector<int64_t>& expert_loads_host,
      const std::vector<int64_t>& previous_assignment) override;

 private:
  // Strict equal-cardinality LPT: each device receives exactly
  // `device_experts_num_` physical experts and we choose the assignment
  // greedily on descending effective per-replica load. Fails (returns empty)
  // if slot count != device_num_ * device_experts_num_.
  std::vector<int64_t> balanced_packing(
      const std::vector<int32_t>& phy_to_log,
      const std::vector<int32_t>& replica_count,
      const std::vector<int64_t>& expert_loads_host);
};

// Factory. Constructs the requested policy or falls back to Greedy when the
// enum value is unrecognised (kept explicit so future additions do not
// silently degrade to a wrong plan).
std::unique_ptr<IEplbPolicy> MakeEplbPolicy(EplbPolicyKind kind,
                                            int32_t device_experts_num,
                                            int32_t device_num,
                                            int32_t layer_num,
                                            EplbOptions options);

// Compatibility facade for the current EplbManager API. It snapshots global
// EPLB options, selects the configured concrete policy, and seeds that policy
// with the identity placement used by EplbManager before its lifecycle is
// migrated to explicit dependency injection.
class EplbPolicy final : public IEplbPolicy {
 public:
  EplbPolicy(int32_t device_experts_num, int32_t device_num, int32_t layer_num);

  std::pair<torch::Tensor, std::vector<bool>> rebalance_experts(
      torch::Tensor expert_load,
      torch::Tensor physical_expert_load = torch::Tensor()) override;

  void initialize_distribution(
      const torch::Tensor& current_distribution) override;
  void commit_layer(int32_t layer_id) override;
  void abort_layer(int32_t layer_id) override;
  std::string name() const override;

 private:
  std::unique_ptr<IEplbPolicy> impl_;
};

}  // namespace xllm
