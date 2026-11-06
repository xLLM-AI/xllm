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

#include <glog/logging.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <vector>

#include "core/runtime/task_execution_pipeline_speculative.h"

namespace xllm {

std::optional<std::string> TaskExecutionPipeline::spec_calibration(
    SpecCalibrationCommand command) {
  check_external_thread();
  folly::Promise<std::optional<std::string>> promise;
  auto future = promise.getFuture();
  state_executor_.schedule([this,
                            command = std::move(command),
                            promise = std::move(promise)]() mutable {
    if (speculative_capacity_ && speculative_capacity_->adaptive_draft_step &&
        command.action == SpecCalibrationAction::CAPABILITIES &&
        accepted_.empty() && completed_.empty()) {
      promise.setValue(draft_step_calibration_capabilities());
      return;
    }
    if (!speculative_capacity_ || !speculative_capacity_->startup_calibration ||
        !adaptive_verify_ || adaptive_verify_->startup_calibration_complete ||
        !accepted_.empty()) {
      promise.setValue(std::nullopt);
      return;
    }
    CHECK(completed_.empty());
    promise.setValue(spec_calibration_impl(command));
  });
  return std::move(future).get();
}

std::optional<std::string>
TaskExecutionPipeline::draft_step_calibration_capabilities() const {
  if (slots_.empty() || captured_graph_batch_sizes_.empty()) {
    return std::nullopt;
  }
  std::vector<uint32_t> requests;
  std::vector<uint32_t> graphs;
  const auto& first = *slots_.front()->speculative->drafts.front();
  requests.reserve(first.captured_graph_batch_sizes.size());
  graphs.reserve(captured_graph_batch_sizes_.size());
  uint64_t max_kv = std::min(capacity_.max_kv_seq_len, capacity_.max_positions);
  for (const auto& task_slot : slots_) {
    const auto& slot = *task_slot->speculative;
    const auto bound = [&](const auto& input) {
      const auto& tables = input.model_params().attention.device.block_tables;
      CHECK(tables.defined() && tables.dim() == 2);
      max_kv = std::min(
          max_kv,
          static_cast<uint64_t>(tables.size(1)) * capacity_.logical_block_size);
    };
    bound(*slot.validate_input);
    for (const auto& draft : slot.drafts) {
      bound(*draft->input);
    }
    if (slot.prefix_refresh) {
      bound(*slot.prefix_refresh->input);
    }
  }
  for (const int64_t rows : first.captured_graph_batch_sizes) {
    if (rows <= 0 || rows % 2 != 0) {
      continue;
    }
    const bool common =
        std::all_of(slots_.begin(), slots_.end(), [&](const auto& task_slot) {
          const auto& drafts = task_slot->speculative->drafts;
          for (size_t step = 0; step < drafts.size(); ++step) {
            const auto& captured = drafts[step]->captured_graph_batch_sizes;
            const int64_t required = step == 0 ? rows : rows / 2;
            if (!std::binary_search(
                    captured.begin(), captured.end(), required)) {
              return false;
            }
          }
          return true;
        });
    if (common) {
      requests.emplace_back(static_cast<uint32_t>(rows / 2));
    }
  }
  for (const int64_t rows : captured_graph_batch_sizes_) {
    const bool common =
        std::all_of(slots_.begin(), slots_.end(), [&](const auto& task_slot) {
          const auto& refresh = task_slot->speculative->prefix_refresh;
          return !refresh ||
                 std::binary_search(refresh->captured_graph_batch_sizes.begin(),
                                    refresh->captured_graph_batch_sizes.end(),
                                    rows);
        });
    if (common) {
      graphs.emplace_back(static_cast<uint32_t>(rows));
    }
  }
  return nlohmann::json{
      {"algorithm", "mtp"},
      {"request_buckets", requests},
      {"target_graph_rows", graphs},
      {"max_kv_length", max_kv},
      {"draft_width", speculative_capacity_->num_speculative_tokens},
      {"slot_count", capacity_.slot_count}}
      .dump();
}

std::optional<std::string> TaskExecutionPipeline::spec_calibration_impl(
    const SpecCalibrationCommand& command) {
  std::vector<uint32_t> requests;
  std::vector<uint32_t> graphs;
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  const auto& first = *slots_.front()->speculative->drafts.front();
  const uint32_t draft_rows_per_request = block_draft() ? draft_width : 2;
  for (const int64_t rows : first.captured_graph_batch_sizes) {
    if (rows <= 0 || rows % draft_rows_per_request != 0) {
      return std::nullopt;
    }
    const bool common =
        std::all_of(slots_.begin(), slots_.end(), [&](const auto& slot) {
          const auto& captured =
              slot->speculative->drafts.front()->captured_graph_batch_sizes;
          return std::binary_search(captured.begin(), captured.end(), rows);
        });
    if (common) {
      requests.emplace_back(
          static_cast<uint32_t>(rows / draft_rows_per_request));
    }
  }
  graphs.reserve(captured_graph_batch_sizes_.size());
  for (const int64_t rows : captured_graph_batch_sizes_) {
    graphs.emplace_back(static_cast<uint32_t>(rows));
  }
  const auto graph_coverage = adaptive_verify_graph_coverage();
  const auto kv_bound = [&](uint32_t batch) {
    uint64_t bound =
        std::min(capacity_.max_kv_seq_len, capacity_.max_positions);
    if (!block_draft()) {
      const auto capabilities = draft_step_calibration_capabilities();
      if (!capabilities) {
        return uint32_t{0};
      }
      bound = std::min(bound,
                       nlohmann::json::parse(*capabilities)
                           .at("max_kv_length")
                           .get<uint64_t>());
      return static_cast<uint32_t>(bound);
    }
    for (const auto& slot : slots_) {
      const auto& draft = *slot->speculative->drafts.front();
      const auto captured =
          draft.captured_graph_kv_capacities.find(batch * draft_width);
      if (captured == draft.captured_graph_kv_capacities.end()) {
        return uint32_t{0};
      }
      bound = std::min(
          {bound,
           captured->second + 1,
           static_cast<uint64_t>(draft.input->model_params()
                                     .attention.device.block_tables.size(1)) *
               capacity_.logical_block_size});
    }
    return static_cast<uint32_t>(bound);
  };
  switch (command.action) {
    case SpecCalibrationAction::CAPABILITIES: {
      nlohmann::json bounds = nlohmann::json::array();
      for (const uint32_t batch : requests) {
        bounds.emplace_back(nlohmann::json{{"requests", batch},
                                           {"max_kv_length", kv_bound(batch)}});
      }
      nlohmann::json coverage = nlohmann::json::array();
      for (const auto& entry : graph_coverage) {
        coverage.emplace_back(
            nlohmann::json{{"requests", entry.requests},
                           {"target_graph_rows", entry.target_graph_rows}});
      }
      return nlohmann::json{{"request_buckets", requests},
                            {"draft_width", draft_width},
                            {"target_graph_rows", graphs},
                            {"graph_coverage", std::move(coverage)},
                            {"kv_bounds", std::move(bounds)},
                            {"slot_count", capacity_.slot_count}}
          .dump();
    }
    case SpecCalibrationAction::BEGIN_SAMPLE: {
      if (adaptive_verify_->calibration_point) {
        return std::nullopt;
      }
      const auto data = nlohmann::json::parse(command.payload, nullptr, false);
      if (!data.is_object() || !data.contains("configuration") ||
          !data.contains("point") || !data["point"].is_number_unsigned()) {
        return std::nullopt;
      }
      const auto config = parse_spec_startup_calibration(
          data["configuration"].dump(), draft_width);
      if (!config) {
        return std::nullopt;
      }
      const auto points =
          plan_spec_calibration(*config, requests, graphs, graph_coverage);
      const uint64_t index = data["point"].get<uint64_t>();
      if (!points || index >= points->size()) {
        return std::nullopt;
      }
      const auto& point = (*points)[index];
      if (point.context.max_kv_length > kv_bound(point.requests)) {
        return std::nullopt;
      }
      adaptive_verify_->calibration_point = point;
      if (speculative_capacity_->calibration_leader) {
        adaptive_verify_->calibration_session =
            std::make_unique<SpecCalibrationSession>(*config, point);
      }
      return std::string("sample_started");
    }
    case SpecCalibrationAction::FINISH_SAMPLE: {
      if (!adaptive_verify_->calibration_point) {
        return std::nullopt;
      }
      nlohmann::json result = nlohmann::json::array();
      const auto& session = adaptive_verify_->calibration_session;
      if (speculative_capacity_->calibration_leader) {
        if (!session || !session->finished()) {
          return std::nullopt;
        }
        for (const auto& window : session->windows()) {
          result.emplace_back(
              nlohmann::json{{"first_generation", window.first_generation},
                             {"last_generation", window.last_generation},
                             {"logical_requests", window.logical_requests},
                             {"draft_requests", window.draft_requests},
                             {"target_graph_rows", window.target_graph_rows},
                             {"min_kv_length", window.min_kv_length},
                             {"max_kv_length", window.max_kv_length},
                             {"steps", window.steps},
                             {"accepted_tokens", window.accepted_tokens},
                             {"draft_tokens", window.draft_tokens},
                             {"elapsed_ms", window.elapsed_ms}});
        }
      }
      adaptive_verify_->calibration_session.reset();
      adaptive_verify_->calibration_point.reset();
      return result.dump();
    }
    case SpecCalibrationAction::INSTALL_COSTS: {
      if (adaptive_verify_->calibration_point) {
        return std::nullopt;
      }
      const auto bundle = SpecCostTableBundle::from_json(command.payload);
      if (!bundle) {
        return std::nullopt;
      }
      auto costs = std::make_shared<SpecCostProfileStore>();
      size_t installed_profiles = 0;
      for (const auto& entry : bundle->entries()) {
        const auto& covered_requests =
            entry.request_buckets.empty() ? requests : entry.request_buckets;
        for (const uint32_t batch : covered_requests) {
          const auto full = std::lower_bound(
              graphs.begin(), graphs.end(), batch * (draft_width + 1));
          if (!std::binary_search(requests.begin(), requests.end(), batch) ||
              full == graphs.end() ||
              (!entry.request_buckets.empty() &&
               entry.max_kv_length > kv_bound(batch))) {
            return std::nullopt;
          }
          if (entry.request_buckets.empty() &&
              !entry.model.step_time_ms(batch, *full)) {
            continue;
          }
          SpecCostProfile profile;
          profile.draft_width = draft_width;
          profile.draft_requests = batch;
          profile.min_kv_length = entry.min_kv_length;
          profile.max_kv_length =
              std::min(entry.max_kv_length, kv_bound(batch));
          profile.cost_scope = SpecCostScope::FULL_STEP;
          auto measured_costs =
              entry.model.verification_costs(batch, draft_width, graphs);
          if (!measured_costs) {
            return std::nullopt;
          }
          profile.target_graph_costs = std::move(*measured_costs);
          if (!costs->store(std::move(profile))) {
            return std::nullopt;
          }
          ++installed_profiles;
        }
      }
      if (installed_profiles == 0) {
        return std::nullopt;
      }
      adaptive_verify_->owned_costs = std::move(costs);
      speculative_capacity_->adaptive_costs = adaptive_verify_->owned_costs;
      adaptive_verify_->histories.assign(adaptive_verify_->histories.size(),
                                         {});
      adaptive_verify_->retired_plans = {};
      std::fill(context_->published_.begin(), context_->published_.end(), 0);
      context_->request_ids_.assign(context_->request_ids_.size(), {});
      adaptive_verify_->startup_calibration_complete = true;
      // Echo the canonical table, allowing the master to confirm every rank
      // installed exactly the same immutable costs before declaring ready.
      return command.payload;
    }
  }
  return std::nullopt;
}

}  // namespace xllm
