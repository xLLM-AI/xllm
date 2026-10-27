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
#include <utility>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/speculative/adaptive/draft_step_controller.h"
#include "core/scheduler/profile/profile_manager.h"

namespace xllm {

void ProfileManager::calibrate_draft_step_startup() {
  auto* calibration = draft_step_controller_;
  CHECK(calibration != nullptr && distributed_worker_manager_ != nullptr);
  CHECK(options_.enable_task_pipeline() && options_.enable_schedule_overlap());
  CHECK_EQ(options_.dp_size(), 1);
  const auto& execution = ExecutionConfig::get_instance();
  CHECK(execution.enable_graph() && !execution.disable_graph_warmup());
  // Reuse the startup-only capability RPC. Unlike DSpark's Device-side
  // pruning, MTP depth selection lives in the engine, so costs need not be
  // installed in, or broadcast among, the workers.
  const auto capabilities = distributed_worker_manager_->spec_calibration(
      {SpecCalibrationAction::CAPABILITIES, {}});
  CHECK(capabilities && !capabilities->empty());
  CHECK(std::all_of(
      capabilities->begin(),
      capabilities->end(),
      [&](const std::string& value) { return value == capabilities->front(); }))
      << "Draft-step calibration geometry differs across workers.";
  const auto geometry = nlohmann::json::parse(capabilities->front());
  CHECK_EQ(geometry.at("algorithm").get<std::string>(), "mtp");
  CHECK_EQ(geometry.at("slot_count").get<uint32_t>(), 2U);
  const auto& config = calibration->configuration(
      static_cast<uint32_t>(model_args_->max_position_embeddings()));
  CHECK_EQ(geometry.at("draft_width").get<uint32_t>(), config.draft_width);
  CHECK_LE(config.contexts.back().max_kv_length,
           geometry.at("max_kv_length").get<uint64_t>());
  CHECK(!decode_graph_warmup_plan_.batch_sizes.empty());
  const uint32_t logical_capacity = static_cast<uint32_t>(
      *std::max_element(decode_graph_warmup_plan_.batch_sizes.begin(),
                        decode_graph_warmup_plan_.batch_sizes.end()));
  auto requests = geometry.at("request_buckets").get<std::vector<uint32_t>>();
  std::erase_if(requests, [&](uint32_t count) {
    return count > 128 || count > logical_capacity ||
           count > static_cast<uint32_t>(options_.max_seqs_per_batch()) ||
           count > static_cast<uint32_t>(options_.max_tokens_per_batch());
  });
  calibration->set_geometry(
      std::move(requests),
      geometry.at("target_graph_rows").get<std::vector<uint32_t>>());
  const auto points = calibration->points();
  CHECK(!points.empty());
  LOG(INFO) << "Draft-step effective startup calibration: configuration="
            << serialize_spec_startup_calibration(config)
            << ", captured_geometry=" << geometry.dump()
            << ", sample_points=" << points.size();
  std::vector<std::vector<SpecCostTableEntry>> costs(config.draft_width);
  for (auto& profiles : costs) {
    profiles.reserve(config.contexts.size());
  }
  std::vector<SpecSteadyWindow> windows;
  for (size_t index = 0; index < points.size(); ++index) {
    const auto& point = points[index];
    run_startup_speculative_sample(
        config,
        point.batch,
        index,
        [&]() { calibration->begin_sample(point); },
        [&]() {
          auto samples = calibration->finish_sample();
          windows.reserve(windows.size() + samples.size());
          for (auto& sample : samples) {
            windows.emplace_back(std::move(sample));
          }
        });
    LOG(INFO) << "Draft-step internal calibration sample complete: depth="
              << point.draft_steps << ", Bcap=" << point.batch.requests
              << ", G=" << point.batch.target_graph_rows
              << ", windows=" << config.windows_per_point;
    const bool last_profile_point =
        index + 1 == points.size() ||
        points[index + 1].draft_steps != point.draft_steps ||
        points[index + 1].batch.context.max_kv_length !=
            point.batch.context.max_kv_length;
    if (!last_profile_point) {
      continue;
    }
    const auto& context = point.batch.context;
    const auto fitted = fit_draft_step_cost_table(windows,
                                                  config.model,
                                                  context.min_kv_length,
                                                  context.max_kv_length,
                                                  point.draft_steps);
    CHECK(fitted)
        << "Cannot fit an identifiable positive draft-step full-cycle model; "
           "check the sampled request/target axes and KV coverage.";
    auto model = SpecCostModel::from_json(*fitted);
    CHECK(model);
    costs[point.draft_steps - 1].emplace_back(
        SpecCostTableEntry{context.min_kv_length,
                           context.max_kv_length,
                           config.request_buckets,
                           std::move(*model)});
    LOG(INFO) << "Draft-step full-cycle cost fitted: depth="
              << point.draft_steps << ", KV=[" << context.min_kv_length << ", "
              << context.max_kv_length << "]";
    windows.clear();
  }
  calibration->install_costs(std::move(costs));
  LOG(INFO) << "Draft-step startup calibration complete; measured per-depth "
               "full-cycle "
               "costs ready for throughput-based selection.";
}

}  // namespace xllm
