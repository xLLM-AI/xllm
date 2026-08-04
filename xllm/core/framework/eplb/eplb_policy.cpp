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

#include "core/framework/eplb/eplb_policy.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace xllm {

namespace {

bool ratio_greater(int64_t lhs_numerator,
                   int64_t lhs_denominator,
                   int64_t rhs_numerator,
                   int64_t rhs_denominator) {
  CHECK_GT(lhs_denominator, 0);
  CHECK_GT(rhs_denominator, 0);
  return static_cast<__int128>(lhs_numerator) * rhs_denominator >
         static_cast<__int128>(rhs_numerator) * lhs_denominator;
}

// Copy `[device_num, device_experts_num]` host int32 layout back into
// `expert_distribution` at row `layer_index`. Shared between the greedy and
// flat policies so the tensor-shaping code lives in one place.
void write_layer_assignment_to_tensor(const std::vector<int64_t>& balanced,
                                      int32_t device_num,
                                      int32_t device_experts_num,
                                      int64_t layer_index,
                                      torch::Tensor& expert_distribution) {
  std::vector<int32_t> balanced_i32;
  balanced_i32.reserve(balanced.size());
  for (int64_t v : balanced) {
    balanced_i32.emplace_back(static_cast<int32_t>(v));
  }
  torch::Tensor layer_assignment =
      torch::from_blob(
          balanced_i32.data(), {device_num, device_experts_num}, torch::kInt32)
          .clone();
  expert_distribution.index_put_({layer_index}, layer_assignment);
}

double compute_peak_device_load(const std::vector<int64_t>& layer_assignment,
                                const std::vector<int64_t>& expert_loads,
                                int32_t device_num,
                                int32_t device_experts_num) {
  CHECK_EQ(layer_assignment.size(),
           static_cast<size_t>(device_num * device_experts_num));
  std::vector<int32_t> replica_count(expert_loads.size(), 0);
  for (int64_t expert_id : layer_assignment) {
    CHECK_GE(expert_id, 0);
    CHECK_LT(static_cast<size_t>(expert_id), expert_loads.size());
    ++replica_count[static_cast<size_t>(expert_id)];
  }

  std::vector<double> device_loads(static_cast<size_t>(device_num), 0.0);
  for (size_t slot = 0; slot < layer_assignment.size(); ++slot) {
    const size_t expert_id = static_cast<size_t>(layer_assignment[slot]);
    const int32_t replicas = replica_count[expert_id];
    CHECK_GT(replicas, 0);
    const size_t device = slot / static_cast<size_t>(device_experts_num);
    device_loads[device] +=
        static_cast<double>(expert_loads[expert_id]) / replicas;
  }
  return *std::max_element(device_loads.begin(), device_loads.end());
}

struct PeakLoadComparison {
  double current_peak = 0.0;
  double proposed_peak = 0.0;
  double improvement_ratio = 0.0;
};

PeakLoadComparison compare_peak_load(
    const std::vector<int64_t>& previous_assignment,
    const std::vector<int64_t>& proposed_assignment,
    const std::vector<int64_t>& expert_loads,
    int32_t device_num,
    int32_t device_experts_num,
    const torch::Tensor& physical_expert_load,
    int32_t layer_id) {
  double current_peak = compute_peak_device_load(
      previous_assignment, expert_loads, device_num, device_experts_num);
  double proposed_peak = compute_peak_device_load(
      proposed_assignment, expert_loads, device_num, device_experts_num);
  if (physical_expert_load.defined()) {
    CHECK_EQ(physical_expert_load.dim(), 3)
        << "physical_expert_load must be [layer, device, slot].";
    CHECK_GT(layer_id, -1);
    CHECK_LT(layer_id, physical_expert_load.size(0));
    CHECK_EQ(physical_expert_load.size(1), device_num);
    CHECK_EQ(physical_expert_load.size(2), device_experts_num);
    const torch::Tensor device_loads = torch::sum(
        physical_expert_load[layer_id].to(torch::kFloat64), /*dim=*/1);
    current_peak = torch::max(device_loads).item<double>();
    const double physical_total = torch::sum(device_loads).item<double>();
    const double logical_total =
        std::accumulate(expert_loads.begin(), expert_loads.end(), 0.0);
    if (logical_total > 0.0) {
      proposed_peak *= physical_total / logical_total;
    }
  }
  if (current_peak <= 0.0) {
    return {current_peak, proposed_peak, 0.0};
  }
  return {current_peak,
          proposed_peak,
          (current_peak - proposed_peak) / current_peak};
}

std::vector<std::vector<int64_t>> copy_distribution_to_host(
    const torch::Tensor& distribution,
    int32_t layer_num,
    int32_t device_num,
    int32_t device_experts_num) {
  const torch::Tensor distribution_cpu =
      distribution.to(torch::kCPU).to(torch::kInt32).contiguous();
  CHECK_EQ(distribution_cpu.dim(), 3);
  CHECK_EQ(distribution_cpu.size(0), layer_num);
  CHECK_EQ(distribution_cpu.size(1), device_num);
  CHECK_EQ(distribution_cpu.size(2), device_experts_num);
  const int32_t* data = distribution_cpu.data_ptr<int32_t>();
  const size_t layer_stride =
      static_cast<size_t>(device_num) * device_experts_num;
  std::vector<std::vector<int64_t>> assignments(
      static_cast<size_t>(layer_num), std::vector<int64_t>(layer_stride));
  for (int32_t layer = 0; layer < layer_num; ++layer) {
    for (size_t slot = 0; slot < layer_stride; ++slot) {
      assignments[static_cast<size_t>(layer)][slot] = static_cast<int64_t>(
          data[static_cast<size_t>(layer) * layer_stride + slot]);
    }
  }
  return assignments;
}

std::string to_lower_ascii(const std::string& in) {
  std::string out(in.size(), '\0');
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(in[i])));
  }
  return out;
}

// Max-load-reduction argmax replica selection used by the balanced policy.
// Given per-origin loads and a number of redundancy slots, returns
//   (phy_to_log, replica_count)
// where phy_to_log has length num_experts + redundancy_experts and the first
// num_experts entries are the primaries (identity mapping). replica_count[e]
// starts at 1 and grows as replicate_experts hands e additional copies. All
// comparisons run on int64 cross-multiplication so plans stay bitwise stable
// across ranks — the driver-broadcast layout must be deterministic.
std::pair<std::vector<int32_t>, std::vector<int32_t>> compute_replicate_experts(
    const std::vector<int64_t>& expert_loads_host,
    int32_t redundancy_experts,
    int32_t max_replica_count) {
  const int32_t num_experts = static_cast<int32_t>(expert_loads_host.size());
  const int32_t total_physical = num_experts + redundancy_experts;
  if (num_experts <= 0 || redundancy_experts < 0 || max_replica_count <= 0) {
    return {};
  }
  std::vector<int32_t> phy_to_log;
  phy_to_log.reserve(static_cast<size_t>(total_physical));
  for (int32_t e = 0; e < num_experts; ++e) {
    phy_to_log.emplace_back(e);
  }
  std::vector<int32_t> replica_count(static_cast<size_t>(num_experts), 1);
  for (int32_t r = 0; r < redundancy_experts; ++r) {
    int32_t best = -1;
    int64_t best_num = 0;
    int64_t best_den = 1;
    for (int32_t e = 0; e < num_experts; ++e) {
      const int64_t load = expert_loads_host[static_cast<size_t>(e)];
      const int64_t rc = replica_count[static_cast<size_t>(e)];
      if (rc >= max_replica_count) {
        continue;
      }
      const int64_t num = load;
      const int64_t den = rc * (rc + 1);
      if (best < 0 || ratio_greater(num, den, best_num, best_den)) {
        best = e;
        best_num = num;
        best_den = den;
      }
    }
    if (best < 0) {
      return {};
    }
    ++replica_count[static_cast<size_t>(best)];
    phy_to_log.emplace_back(best);
  }
  return {std::move(phy_to_log), std::move(replica_count)};
}

void stabilize_local_slots(std::vector<int64_t>& layer_assignment,
                           const std::vector<int64_t>& prev_assignment,
                           int32_t device_num,
                           int32_t device_experts_num) {
  if (prev_assignment.empty() ||
      prev_assignment.size() != layer_assignment.size()) {
    return;
  }

  const std::vector<int64_t> packed_assignment = layer_assignment;
  for (int32_t device = 0; device < device_num; ++device) {
    const size_t row_start = static_cast<size_t>(device) * device_experts_num;
    std::vector<bool> source_used(static_cast<size_t>(device_experts_num),
                                  false);
    std::vector<bool> destination_filled(
        static_cast<size_t>(device_experts_num), false);

    for (int32_t destination = 0; destination < device_experts_num;
         ++destination) {
      const size_t destination_index =
          row_start + static_cast<size_t>(destination);
      const int64_t previous_expert = prev_assignment[destination_index];
      for (int32_t source = 0; source < device_experts_num; ++source) {
        const size_t source_offset = static_cast<size_t>(source);
        if (source_used[source_offset] ||
            packed_assignment[row_start + source_offset] != previous_expert) {
          continue;
        }
        layer_assignment[destination_index] = previous_expert;
        source_used[source_offset] = true;
        destination_filled[static_cast<size_t>(destination)] = true;
        break;
      }
    }

    int32_t next_source = 0;
    for (int32_t destination = 0; destination < device_experts_num;
         ++destination) {
      if (destination_filled[static_cast<size_t>(destination)]) {
        continue;
      }
      while (next_source < device_experts_num &&
             source_used[static_cast<size_t>(next_source)]) {
        ++next_source;
      }
      CHECK_LT(next_source, device_experts_num)
          << "EPLB local slot stabilization exhausted source slots.";
      const size_t source_offset = static_cast<size_t>(next_source);
      layer_assignment[row_start + static_cast<size_t>(destination)] =
          packed_assignment[row_start + source_offset];
      source_used[source_offset] = true;
    }
  }
}

}  // namespace

EplbPolicyKind eplb_policy_kind_from_string(const std::string& kind) {
  const std::string normalized = to_lower_ascii(kind);
  if (normalized == "balanced" || normalized == "flat" ||
      normalized == "deepseek_flat" || normalized == "deepseek-flat" ||
      normalized == "deepseek_hier" || normalized == "deepseek-hier" ||
      normalized == "hier" || normalized == "hierarchical") {
    return EplbPolicyKind::BALANCED;
  }
  return EplbPolicyKind::GREEDY;
}

// -----------------------------------------------------------------------------
// EplbPolicyBase
// -----------------------------------------------------------------------------

EplbPolicyBase::EplbPolicyBase(int32_t device_experts_num,
                               int32_t device_num,
                               int32_t layer_num,
                               EplbOptions options)
    : device_experts_num_(device_experts_num),
      device_num_(device_num),
      layer_num_(layer_num),
      options_(std::move(options)) {
  expert_distribution_ = torch::full(
      {layer_num_, device_num_, device_experts_num_}, -1, torch::kInt32);
  rollback_states_.resize(static_cast<size_t>(layer_num_));
}

void EplbPolicyBase::initialize_distribution(
    const torch::Tensor& current_distribution) {
  const torch::Tensor distribution_cpu =
      current_distribution.to(torch::kCPU).to(torch::kInt32).contiguous();
  expert_distribution_.copy_(distribution_cpu);
  prev_layer_assignment_ = copy_distribution_to_host(
      distribution_cpu, layer_num_, device_num_, device_experts_num_);
  rollback_states_.assign(static_cast<size_t>(layer_num_),
                          LayerRollbackState{});
}

std::pair<torch::Tensor, std::vector<bool>> EplbPolicyBase::rebalance_experts(
    torch::Tensor expert_load,
    torch::Tensor physical_expert_load) {
  std::vector<bool> enable_update_vec(static_cast<size_t>(layer_num_), false);
  if (prev_layer_assignment_.size() != static_cast<size_t>(layer_num_)) {
    prev_layer_assignment_.assign(static_cast<size_t>(layer_num_),
                                  std::vector<int64_t>());
  }

  for (int64_t i = 0; i < layer_num_; ++i) {
    const size_t layer_index = static_cast<size_t>(i);
    torch::Tensor layer_load = expert_load[i].to(torch::kInt64).contiguous();
    CHECK_EQ(layer_load.dim(), 1) << "expert_loads must be 1D tensor";
    const int64_t num_experts = layer_load.size(0);
    std::vector<int64_t> expert_loads_host(
        layer_load.data_ptr<int64_t>(),
        layer_load.data_ptr<int64_t>() + num_experts);
    std::vector<int64_t> balanced =
        plan_layer(expert_loads_host, prev_layer_assignment_[layer_index]);
    if (balanced.empty()) {
      continue;
    }
    CHECK_EQ(static_cast<int32_t>(balanced.size()),
             device_num_ * device_experts_num_)
        << "EPLB balanced pack size mismatch";
    bool placement_changed = prev_layer_assignment_[layer_index].empty() ||
                             balanced != prev_layer_assignment_[layer_index];
    if (placement_changed && !prev_layer_assignment_[layer_index].empty()) {
      const PeakLoadComparison comparison =
          compare_peak_load(prev_layer_assignment_[layer_index],
                            balanced,
                            expert_loads_host,
                            device_num_,
                            device_experts_num_,
                            physical_expert_load,
                            static_cast<int32_t>(i));
      if (comparison.improvement_ratio <= 0.0 ||
          comparison.improvement_ratio <
              options_.eplb_min_peak_load_improvement) {
        placement_changed = false;
        balanced = prev_layer_assignment_[layer_index];
      }
      LOG(INFO) << "EPLB placement benefit | policy=" << name()
                << " | layer=" << i
                << " | current_peak=" << comparison.current_peak
                << " | proposed_peak=" << comparison.proposed_peak
                << " | improvement_ratio=" << comparison.improvement_ratio
                << " | min_improvement_ratio="
                << options_.eplb_min_peak_load_improvement
                << " | placement_changed=" << placement_changed;
    }
    if (!placement_changed) {
      continue;
    }
    LayerRollbackState& rollback = rollback_states_[layer_index];
    rollback.valid = true;
    rollback.expert_distribution = expert_distribution_[i].clone();
    rollback.prev_assignment = prev_layer_assignment_[layer_index];
    prev_layer_assignment_[layer_index] = balanced;
    write_layer_assignment_to_tensor(
        balanced, device_num_, device_experts_num_, i, expert_distribution_);
    enable_update_vec[layer_index] = true;
  }
  expert_distribution_ = expert_distribution_.contiguous();
  return {expert_distribution_.clone(), enable_update_vec};
}

void EplbPolicyBase::commit_layer(int32_t layer_id) {
  CHECK_GE(layer_id, 0);
  CHECK_LT(layer_id, layer_num_);
  rollback_states_[static_cast<size_t>(layer_id)] = LayerRollbackState{};
}

void EplbPolicyBase::abort_layer(int32_t layer_id) {
  CHECK_GE(layer_id, 0);
  CHECK_LT(layer_id, layer_num_);
  const size_t layer_index = static_cast<size_t>(layer_id);
  LayerRollbackState& rollback = rollback_states_[layer_index];
  if (!rollback.valid) {
    return;
  }
  expert_distribution_[layer_id].copy_(rollback.expert_distribution);
  prev_layer_assignment_[layer_index] = std::move(rollback.prev_assignment);
  rollback = LayerRollbackState{};
}

// -----------------------------------------------------------------------------
// GreedyEplbPolicy
// -----------------------------------------------------------------------------

GreedyEplbPolicy::GreedyEplbPolicy(int32_t device_experts_num,
                                   int32_t device_num,
                                   int32_t layer_num,
                                   EplbOptions options)
    : EplbPolicyBase(device_experts_num,
                     device_num,
                     layer_num,
                     std::move(options)) {}

std::vector<int64_t> GreedyEplbPolicy::plan_layer(
    const std::vector<int64_t>& expert_loads_host,
    const std::vector<int64_t>& previous_assignment) {
  (void)previous_assignment;
  return compute_balanced_pack_host(expert_loads_host);
}

std::vector<int64_t> GreedyEplbPolicy::compute_balanced_pack_host(
    const std::vector<int64_t>& expert_loads_host) {
  const int32_t redundant_experts_num = options_.redundant_experts_num;
  auto [updated_weights, redundancy_map] = update_origin_weights_host(
      expert_loads_host, device_num_ * redundant_experts_num);

  const int64_t num_experts = static_cast<int64_t>(expert_loads_host.size());
  const int64_t total_slots =
      static_cast<int64_t>(device_num_) * device_experts_num_;

  std::vector<int64_t> device_assignments(static_cast<size_t>(total_slots), -1);
  std::vector<int64_t> device_loads(static_cast<size_t>(device_num_), 0);
  std::vector<int32_t> free_slots(static_cast<size_t>(device_num_),
                                  device_experts_num_);

  auto place_on_device =
      [&](int32_t device_idx, int64_t expert_id, int64_t weight) -> bool {
    const int64_t row_start =
        static_cast<int64_t>(device_idx) * device_experts_num_;
    for (int64_t slot = 0; slot < device_experts_num_; ++slot) {
      if (device_assignments[static_cast<size_t>(row_start + slot)] == -1) {
        device_assignments[static_cast<size_t>(row_start + slot)] = expert_id;
        device_loads[static_cast<size_t>(device_idx)] += weight;
        --free_slots[static_cast<size_t>(device_idx)];
        return true;
      }
    }
    return false;
  };

  auto min_loaded_device_with_capacity = [&]() -> int32_t {
    int32_t best = -1;
    int64_t best_load = 0;
    for (int32_t d = 0; d < device_num_; ++d) {
      if (free_slots[static_cast<size_t>(d)] <= 0) {
        continue;
      }
      const int64_t load = device_loads[static_cast<size_t>(d)];
      if (best < 0 || load < best_load) {
        best = d;
        best_load = load;
      }
    }
    return best;
  };

  // Phase 1: redundant expert placement.
  for (int64_t origin_id = 0; origin_id < num_experts; ++origin_id) {
    const size_t redundancy_row =
        static_cast<size_t>(origin_id) *
        static_cast<size_t>(device_num_ * redundant_experts_num);
    for (int64_t i = 0; i < device_num_ * redundant_experts_num; ++i) {
      const int64_t redundant_slot_id =
          redundancy_map[redundancy_row + static_cast<size_t>(i)];
      if (redundant_slot_id == -1) {
        break;
      }
      const int32_t target = min_loaded_device_with_capacity();
      if (target < 0) {
        LOG(ERROR) << "EPLB rebalance skipped: no device has a free slot for "
                      "redundant experts.";
        return {};
      }
      if (!place_on_device(target,
                           origin_id,
                           updated_weights[static_cast<size_t>(origin_id)])) {
        LOG(ERROR) << "EPLB rebalance skipped: device " << target
                   << " unexpectedly full during redundant placement.";
        return {};
      }
    }
  }

  // Phase 2: primary experts sorted desc by updated weight, LPT-place.
  std::vector<int64_t> sorted_indices(static_cast<size_t>(num_experts));
  for (int64_t i = 0; i < num_experts; ++i) {
    sorted_indices[static_cast<size_t>(i)] = i;
  }
  std::sort(
      sorted_indices.begin(), sorted_indices.end(), [&](int64_t a, int64_t b) {
        return updated_weights[static_cast<size_t>(a)] >
               updated_weights[static_cast<size_t>(b)];
      });

  for (int64_t expert_id : sorted_indices) {
    const int32_t target = min_loaded_device_with_capacity();
    if (target < 0) {
      break;
    }
    if (!place_on_device(target,
                         expert_id,
                         updated_weights[static_cast<size_t>(expert_id)])) {
      LOG(ERROR) << "EPLB rebalance skipped: target device " << target
                 << " has no free slot for primary expert " << expert_id << ".";
      return {};
    }
  }
  return device_assignments;
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
GreedyEplbPolicy::update_origin_weights_host(
    const std::vector<int64_t>& expert_loads_host,
    int32_t redundancy_experts) {
  const int64_t num_experts = static_cast<int64_t>(expert_loads_host.size());
  std::vector<int64_t> redundancy_map(
      static_cast<size_t>(num_experts) *
          static_cast<size_t>(redundancy_experts),
      -1);
  std::vector<int64_t> current_weights = expert_loads_host;

  for (int32_t i = 0; i < redundancy_experts; ++i) {
    int64_t max_idx = 0;
    int64_t max_val = current_weights.empty() ? 0 : current_weights[0];
    for (int64_t j = 1; j < num_experts; ++j) {
      const int64_t v = current_weights[static_cast<size_t>(j)];
      if (v > max_val) {
        max_val = v;
        max_idx = j;
      }
    }
    const size_t row_start =
        static_cast<size_t>(max_idx) * static_cast<size_t>(redundancy_experts);
    int32_t redundancy_count = 1;
    for (int32_t k = 0; k < redundancy_experts; ++k) {
      if (redundancy_map[row_start + static_cast<size_t>(k)] != -1) {
        ++redundancy_count;
      }
    }
    redundancy_map[row_start + static_cast<size_t>(redundancy_count - 1)] =
        num_experts + i;
    const int64_t new_weight =
        static_cast<int64_t>(current_weights[static_cast<size_t>(max_idx)] *
                             redundancy_count / (redundancy_count + 1.0));
    current_weights[static_cast<size_t>(max_idx)] = new_weight;
  }
  return {std::move(current_weights), std::move(redundancy_map)};
}

// -----------------------------------------------------------------------------
// BalancedEplbPolicy
// -----------------------------------------------------------------------------

BalancedEplbPolicy::BalancedEplbPolicy(int32_t device_experts_num,
                                       int32_t device_num,
                                       int32_t layer_num,
                                       EplbOptions options)
    : EplbPolicyBase(device_experts_num,
                     device_num,
                     layer_num,
                     std::move(options)) {}

std::vector<int64_t> BalancedEplbPolicy::plan_layer(
    const std::vector<int64_t>& expert_loads_host,
    const std::vector<int64_t>& previous_assignment) {
  const int32_t redundancy_experts =
      device_num_ * options_.redundant_experts_num;
  auto [phy_to_log, replica_count] = compute_replicate_experts(
      expert_loads_host, redundancy_experts, device_num_);
  if (phy_to_log.empty()) {
    return {};
  }
  std::vector<int64_t> balanced =
      balanced_packing(phy_to_log, replica_count, expert_loads_host);
  if (balanced.empty()) {
    return {};
  }
  stabilize_local_slots(
      balanced, previous_assignment, device_num_, device_experts_num_);
  return balanced;
}

std::vector<int64_t> BalancedEplbPolicy::balanced_packing(
    const std::vector<int32_t>& phy_to_log,
    const std::vector<int32_t>& replica_count,
    const std::vector<int64_t>& expert_loads_host) {
  const int64_t total_slots =
      static_cast<int64_t>(device_num_) * device_experts_num_;
  if (static_cast<int64_t>(phy_to_log.size()) != total_slots) {
    LOG(ERROR) << "BalancedEplbPolicy: physical expert count "
               << phy_to_log.size() << " != device_num_ * device_experts_num_ ("
               << total_slots << "); redundancy configuration mismatch.";
    return {};
  }
  std::vector<int64_t> device_assignments(static_cast<size_t>(total_slots), -1);
  std::vector<int64_t> device_loads(static_cast<size_t>(device_num_), 0);
  std::vector<int32_t> free_slots(static_cast<size_t>(device_num_),
                                  device_experts_num_);
  std::vector<std::vector<bool>> device_has_expert(
      static_cast<size_t>(device_num_),
      std::vector<bool>(expert_loads_host.size(), false));

  // Sort physical experts by effective per-replica load descending. Effective
  // load = load[e] / replica_count[e]; compared via int64 cross-multiplication
  // for the same determinism reason as in replicate_experts.
  std::vector<int32_t> order(static_cast<size_t>(total_slots));
  for (int64_t s = 0; s < total_slots; ++s) {
    order[static_cast<size_t>(s)] = static_cast<int32_t>(s);
  }
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    const int32_t log_a = phy_to_log[static_cast<size_t>(a)];
    const int32_t log_b = phy_to_log[static_cast<size_t>(b)];
    const int64_t load_a = expert_loads_host[static_cast<size_t>(log_a)];
    const int64_t load_b = expert_loads_host[static_cast<size_t>(log_b)];
    const int64_t rc_a = replica_count[static_cast<size_t>(log_a)];
    const int64_t rc_b = replica_count[static_cast<size_t>(log_b)];
    return ratio_greater(load_a, rc_a, load_b, rc_b);
  });

  const auto build_cyclic_fallback = [&]() -> std::vector<int64_t> {
    std::vector<int32_t> expert_order;
    expert_order.reserve(replica_count.size());
    for (int32_t expert = 0;
         expert < static_cast<int32_t>(replica_count.size());
         ++expert) {
      if (replica_count[static_cast<size_t>(expert)] > 0) {
        expert_order.emplace_back(expert);
      }
    }
    std::sort(
        expert_order.begin(), expert_order.end(), [&](int32_t a, int32_t b) {
          const int64_t load_a = expert_loads_host[static_cast<size_t>(a)];
          const int64_t load_b = expert_loads_host[static_cast<size_t>(b)];
          const int64_t rc_a = replica_count[static_cast<size_t>(a)];
          const int64_t rc_b = replica_count[static_cast<size_t>(b)];
          if (ratio_greater(load_a, rc_a, load_b, rc_b)) {
            return true;
          }
          if (ratio_greater(load_b, rc_b, load_a, rc_a)) {
            return false;
          }
          return a < b;
        });

    std::vector<int64_t> fallback(static_cast<size_t>(total_slots), -1);
    std::vector<int32_t> next_slot(static_cast<size_t>(device_num_), 0);
    int64_t placement_index = 0;
    for (int32_t expert : expert_order) {
      const int32_t replicas = replica_count[static_cast<size_t>(expert)];
      if (replicas > device_num_) {
        return {};
      }
      for (int32_t replica = 0; replica < replicas; ++replica) {
        const int32_t device =
            static_cast<int32_t>(placement_index % device_num_);
        const int32_t slot = next_slot[static_cast<size_t>(device)]++;
        if (slot >= device_experts_num_) {
          return {};
        }
        fallback[static_cast<size_t>(device) * device_experts_num_ + slot] =
            expert;
        ++placement_index;
      }
    }
    if (placement_index != total_slots ||
        std::any_of(fallback.begin(), fallback.end(), [](int64_t expert) {
          return expert < 0;
        })) {
      return {};
    }
    return fallback;
  };

  auto min_loaded_device_with_capacity = [&](int32_t expert_id) -> int32_t {
    int32_t best = -1;
    int64_t best_load = 0;
    for (int32_t d = 0; d < device_num_; ++d) {
      if (free_slots[static_cast<size_t>(d)] <= 0 ||
          device_has_expert[static_cast<size_t>(d)]
                           [static_cast<size_t>(expert_id)]) {
        continue;
      }
      const int64_t load = device_loads[static_cast<size_t>(d)];
      if (best < 0 || load < best_load) {
        best = d;
        best_load = load;
      }
    }
    return best;
  };

  auto place_on_device = [&](int32_t device_idx,
                             int64_t expert_id,
                             int64_t effective_load) -> bool {
    const int64_t row_start =
        static_cast<int64_t>(device_idx) * device_experts_num_;
    for (int64_t slot = 0; slot < device_experts_num_; ++slot) {
      if (device_assignments[static_cast<size_t>(row_start + slot)] == -1) {
        device_assignments[static_cast<size_t>(row_start + slot)] = expert_id;
        device_loads[static_cast<size_t>(device_idx)] += effective_load;
        --free_slots[static_cast<size_t>(device_idx)];
        device_has_expert[static_cast<size_t>(device_idx)]
                         [static_cast<size_t>(expert_id)] = true;
        return true;
      }
    }
    return false;
  };

  for (int32_t phy_idx : order) {
    const int32_t log_id = phy_to_log[static_cast<size_t>(phy_idx)];
    const int64_t load = expert_loads_host[static_cast<size_t>(log_id)];
    const int64_t rc = replica_count[static_cast<size_t>(log_id)];
    // Skip zero-replica log ids defensively; replicate_experts should never
    // hand these out but the CHECK is cheap.
    CHECK_GT(rc, 0)
        << "BalancedEplbPolicy: replica_count must be positive for every "
           "physical slot.";
    const int64_t effective_load = load / rc;
    const int32_t target = min_loaded_device_with_capacity(log_id);
    if (target < 0) {
      LOG(WARNING) << "BalancedEplbPolicy: LPT placement reached a dead end at "
                   << "physical expert " << phy_idx << " (log id " << log_id
                   << "); using deterministic cyclic fallback.";
      return build_cyclic_fallback();
    }
    if (!place_on_device(target, log_id, effective_load)) {
      LOG(WARNING) << "BalancedEplbPolicy: LPT placement failed on device "
                   << target << " for physical expert " << phy_idx
                   << "; using deterministic cyclic fallback.";
      return build_cyclic_fallback();
    }
  }
  return device_assignments;
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

std::unique_ptr<IEplbPolicy> MakeEplbPolicy(EplbPolicyKind kind,
                                            int32_t device_experts_num,
                                            int32_t device_num,
                                            int32_t layer_num,
                                            EplbOptions options) {
  switch (kind) {
    case EplbPolicyKind::BALANCED:
      return std::make_unique<BalancedEplbPolicy>(
          device_experts_num, device_num, layer_num, std::move(options));
    case EplbPolicyKind::GREEDY:
    default:
      return std::make_unique<GreedyEplbPolicy>(
          device_experts_num, device_num, layer_num, std::move(options));
  }
}

EplbPolicy::EplbPolicy(int32_t device_experts_num,
                       int32_t device_num,
                       int32_t layer_num) {
  EplbOptions options = EplbOptions::from_global_config();
  const int32_t routed_experts_num =
      device_experts_num - options.redundant_experts_num;
  CHECK_GT(routed_experts_num, 0)
      << "EPLB routed experts per device must be positive.";

  const EplbPolicyKind policy_kind =
      eplb_policy_kind_from_string(options.eplb_policy_kind);
  impl_ = MakeEplbPolicy(policy_kind,
                         device_experts_num,
                         device_num,
                         layer_num,
                         std::move(options));
  torch::Tensor initial_distribution =
      torch::zeros({layer_num, device_num, device_experts_num}, torch::kInt32);
  for (int32_t layer = 0; layer < layer_num; ++layer) {
    for (int32_t device = 0; device < device_num; ++device) {
      const int32_t base = device * routed_experts_num;
      for (int32_t slot = 0; slot < device_experts_num; ++slot) {
        const int32_t routed_slot =
            slot < routed_experts_num ? slot : routed_experts_num - 1;
        initial_distribution[layer][device][slot] = base + routed_slot;
      }
    }
  }
  impl_->initialize_distribution(initial_distribution);
}

std::pair<torch::Tensor, std::vector<bool>> EplbPolicy::rebalance_experts(
    torch::Tensor expert_load,
    torch::Tensor physical_expert_load) {
  return impl_->rebalance_experts(std::move(expert_load),
                                  std::move(physical_expert_load));
}

void EplbPolicy::initialize_distribution(
    const torch::Tensor& current_distribution) {
  impl_->initialize_distribution(current_distribution);
}

void EplbPolicy::commit_layer(int32_t layer_id) {
  impl_->commit_layer(layer_id);
}

void EplbPolicy::abort_layer(int32_t layer_id) { impl_->abort_layer(layer_id); }

std::string EplbPolicy::name() const { return impl_->name(); }

}  // namespace xllm
