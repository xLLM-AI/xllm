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

#include "core/framework/speculative/adaptive/spec_startup_calibration.h"

#include <glog/logging.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <system_error>
#include <utility>

#include "core/distributed_runtime/distributed_worker_manager.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/request/request_state.h"
#include "core/framework/speculative/adaptive/draft_step_controller.h"
#include "core/scheduler/profile/profile_manager.h"

namespace xllm {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  CHECK(input.is_open()) << "Cannot read DSpark calibration input: " << path;
  const std::string text{std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>()};
  CHECK(!input.bad()) << "Failed to read DSpark calibration input: " << path;
  return text;
}

nlohmann::json checkpoint_identity(const std::filesystem::path& path) {
  CHECK(std::filesystem::is_directory(path));
  nlohmann::json result;
  result["path"] = std::filesystem::canonical(path).string();
  for (const char* filename : {"config.json",
                               "quant_model_description.json",
                               "model.safetensors.index.json"}) {
    const auto file = path / filename;
    if (!std::filesystem::exists(file)) {
      continue;
    }
    const auto data = nlohmann::json::parse(read_file(file), nullptr, false);
    CHECK(data.is_object()) << "Invalid checkpoint identity input: " << file;
    result[filename] = data;
  }
  CHECK(result.contains("config.json"));
  // Stat-only weight identity avoids reading hundreds of GB during startup.
  // Paths, sizes and modification timestamps are checked in addition to the
  // canonical model/quantization configurations; this is not a content hash.
  nlohmann::json weights = nlohmann::json::object();
  for (const auto& file : std::filesystem::recursive_directory_iterator(path)) {
    const std::string extension = file.path().extension().string();
    if (!file.is_regular_file() ||
        (extension != ".safetensors" && extension != ".bin" &&
         extension != ".pt" && extension != ".pth")) {
      continue;
    }
    weights[std::filesystem::relative(file.path(), path).string()] = {
        {"bytes", file.file_size()},
        {"mtime", file.last_write_time().time_since_epoch().count()}};
  }
  CHECK(!weights.empty()) << "Checkpoint has no identifiable weight files.";
  result["weights"] = std::move(weights);
  return result;
}

void save_new_table(const std::filesystem::path& path,
                    const std::string& text) {
  CHECK(!path.empty());
  std::string pattern = path.string() + ".tmp.XXXXXX";
  std::vector<char> temporary(pattern.begin(), pattern.end());
  temporary.emplace_back('\0');
  const int descriptor = ::mkstemp(temporary.data());
  CHECK_GE(descriptor, 0) << "Cannot create DSpark cost-table output: " << path;
  size_t offset = 0;
  while (offset < text.size()) {
    const ssize_t written =
        ::write(descriptor, text.data() + offset, text.size() - offset);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    CHECK_GT(written, 0) << "Cannot write DSpark cost-table output: " << path;
    offset += static_cast<size_t>(written);
  }
  CHECK_EQ(::fsync(descriptor), 0);
  CHECK_EQ(::close(descriptor), 0);
  // Atomic publication without replacing an existing user file, including a
  // concurrent writer. The temporary is in the same filesystem as the target.
  const int linked = ::link(temporary.data(), path.c_str());
  const int link_error = errno;
  CHECK_EQ(::unlink(temporary.data()), 0);
  CHECK_EQ(linked, 0) << "Cannot publish new DSpark cost table (errno="
                      << link_error << "): " << path;
}

SpecSteadyWindow read_window(const nlohmann::json& data) {
  SpecSteadyWindow window;
  window.first_generation = data.at("first_generation").get<uint64_t>();
  window.last_generation = data.at("last_generation").get<uint64_t>();
  window.logical_requests = data.at("logical_requests").get<uint32_t>();
  window.draft_requests = data.at("draft_requests").get<uint32_t>();
  window.target_graph_rows = data.at("target_graph_rows").get<uint32_t>();
  window.min_kv_length = data.at("min_kv_length").get<uint32_t>();
  window.max_kv_length = data.at("max_kv_length").get<uint32_t>();
  window.steps = data.at("steps").get<uint32_t>();
  window.accepted_tokens = data.at("accepted_tokens").get<uint64_t>();
  window.draft_tokens = data.at("draft_tokens").get<uint64_t>();
  window.elapsed_ms = data.at("elapsed_ms").get<double>();
  return window;
}

}  // namespace

std::string spec_calibration_compatibility(const runtime::Options& options) {
  CHECK(options.draft_model_path().has_value());
  return nlohmann::json{
      {"implementation", "async-steady-full-step-v1"},
      {"model_impl", ModelConfig::get_instance().model_impl()},
      {"target", checkpoint_identity(options.model_path())},
      {"draft", checkpoint_identity(*options.draft_model_path())},
      {"world_size", options.nnodes() * options.devices().size()},
      {"dp_size", options.dp_size()},
      {"ep_size", options.ep_size()},
      {"cp_size", options.cp_size()},
      {"graph", options.enable_graph()},
      {"task_pipeline", options.enable_task_pipeline()},
      {"schedule_overlap", options.enable_schedule_overlap()},
      {"max_tokens_per_batch", options.max_tokens_per_batch()},
      {"max_seqs_per_batch", options.max_seqs_per_batch()},
      {"block_size", options.block_size()},
      {"kv_cache_dtype", options.kv_cache_dtype()},
      {"npu_kernel_backend", options.npu_kernel_backend()},
      {"python_graph_backend",
       ExecutionConfig::get_instance().python_graph_backend()},
      {"unified_graph",
       ExecutionConfig::get_instance().enable_unified_mtp_graph()},
      {"fia_decode", ExecutionConfig::get_instance().enable_fia_decode()},
      {"num_speculative_tokens", options.num_speculative_tokens()}}
      .dump();
}

const SpecStartupCalibrationConfig&
ProfileManager::target_verify_calibration_config() {
  if (!target_verify_calibration_config_) {
    if (options_.adaptive_target_verify_calibration().empty()) {
      target_verify_calibration_config_ = default_spec_startup_calibration(
          static_cast<uint32_t>(
              engine_->model_args().max_position_embeddings()),
          options_.decode_graph_warmup_config().num_speculative_tokens);
    } else {
      target_verify_calibration_config_ = parse_spec_startup_calibration(
          read_file(options_.adaptive_target_verify_calibration()),
          options_.decode_graph_warmup_config().num_speculative_tokens);
    }
    CHECK(target_verify_calibration_config_)
        << "Invalid DSpark calibration configuration or insufficient context "
           "for defaults; provide an explicit calibration JSON with fewer "
           "steps.";
    LOG(INFO) << "DSpark startup calibration configuration: source="
              << (options_.adaptive_target_verify_calibration().empty()
                      ? "built-in defaults"
                      : options_.adaptive_target_verify_calibration())
              << ", configuration="
              << serialize_spec_startup_calibration(
                     *target_verify_calibration_config_)
              << "; default request buckets are resolved from captured graphs.";
  }
  return *target_verify_calibration_config_;
}

int32_t ProfileManager::adaptive_calibration_warmup_seq_len(
    int32_t default_length) {
  if (options_.enable_adaptive_draft_step()) {
    auto* calibration = engine_->draft_step_controller();
    CHECK(calibration != nullptr);
    const uint32_t max_positions =
        static_cast<uint32_t>(engine_->model_args().max_position_embeddings());
    const auto& config = calibration->configuration(max_positions);
    const auto length = spec_calibration_warmup_length(
        config.contexts.back().max_kv_length,
        max_positions,
        static_cast<uint32_t>(block_manager_pool_->block_size()),
        static_cast<uint32_t>(default_length),
        config.draft_width,
        /*autoregressive_draft=*/true);
    CHECK(length) << "MTP calibration exceeds safe graph warmup KV coverage.";
    return static_cast<int32_t>(*length);
  }
  if (!options_.enable_adaptive_target_verify() &&
      options_.adaptive_target_verify_calibration().empty() &&
      options_.adaptive_target_verify_cost_table().empty()) {
    return default_length;
  }
  uint32_t max_kv_length = 0;
  if (options_.adaptive_target_verify_cost_table().empty()) {
    max_kv_length =
        target_verify_calibration_config().contexts.back().max_kv_length;
  } else {
    const auto bundle = SpecCostTableBundle::from_json(
        read_file(options_.adaptive_target_verify_cost_table()),
        options_.spec_calibration_compatibility());
    CHECK(bundle) << "Invalid or incompatible DSpark cost table.";
    // Legacy tables may declare an unlimited KV domain. Retain their previous
    // warmup geometry, and cap installation to the actual captured coverage.
    if (bundle->entries().front().request_buckets.empty()) {
      return default_length;
    }
    max_kv_length = bundle->entries().back().max_kv_length;
  }
  const auto length = spec_calibration_warmup_length(
      max_kv_length,
      static_cast<uint32_t>(engine_->model_args().max_position_embeddings()),
      static_cast<uint32_t>(block_manager_pool_->block_size()),
      static_cast<uint32_t>(default_length),
      options_.decode_graph_warmup_config().num_speculative_tokens);
  CHECK(length) << "DSpark calibration KV bound exceeds safe graph warmup "
                   "capacity, including both in-flight allocations.";
  return static_cast<int32_t>(*length);
}

void ProfileManager::calibrate_spec_startup() {
  CHECK(distributed_worker_manager_ != nullptr);
  CHECK(options_.enable_task_pipeline());
  CHECK_GT(options_.dp_size(), 0);
  const auto& execution = ExecutionConfig::get_instance();
  CHECK(execution.enable_graph() && !execution.disable_graph_warmup());
  const auto capabilities = distributed_worker_manager_->spec_calibration(
      {SpecCalibrationAction::CAPABILITIES, {}});
  CHECK(capabilities && !capabilities->empty());
  CHECK(std::all_of(
      capabilities->begin(),
      capabilities->end(),
      [&](const std::string& value) { return value == capabilities->front(); }))
      << "DSpark captured calibration geometry differs across workers.";
  const auto geometry = nlohmann::json::parse(capabilities->front());
  const auto compatibility = nlohmann::json::parse(
      options_.spec_calibration_compatibility(), nullptr, false);
  CHECK(compatibility.is_object() && !compatibility.empty());
  const auto install = [&](const std::string& serialized) {
    CHECK(SpecCostTableBundle::from_json(serialized, compatibility.dump()));
    const auto installed = distributed_worker_manager_->spec_calibration(
        {SpecCalibrationAction::INSTALL_COSTS, serialized});
    CHECK(installed && installed->size() == capabilities->size());
    CHECK(std::all_of(
        installed->begin(),
        installed->end(),
        [&](const std::string& value) { return value == serialized; }))
        << "Workers did not install identical DSpark cost tables.";
  };
  if (!options_.adaptive_target_verify_cost_table().empty()) {
    const std::string serialized =
        read_file(options_.adaptive_target_verify_cost_table());
    install(serialized);
    const auto bundle =
        SpecCostTableBundle::from_json(serialized, compatibility.dump());
    CHECK(bundle);
    nlohmann::json coverage = nlohmann::json::array();
    for (const auto& entry : bundle->entries()) {
      coverage.emplace_back(nlohmann::json{
          {"min_kv_length", entry.min_kv_length},
          {"max_kv_length", entry.max_kv_length},
          {"request_buckets", entry.request_buckets},
          {"model", SpecCostModel::kind_name(entry.model.kind())}});
    }
    const auto data = nlohmann::json::parse(serialized);
    LOG(INFO) << "DSpark adaptive effective configuration: "
              << nlohmann::json{{"source", "saved_cost_table"},
                                {"path",
                                 options_.adaptive_target_verify_cost_table()},
                                {"calibration",
                                 data.value("calibration",
                                            nlohmann::json::object())},
                                {"declared_coverage", std::move(coverage)},
                                {"captured_geometry", geometry}}
                     .dump();
    LOG(INFO) << "DSpark saved full-step costs installed; startup sampling and "
                 "fitting skipped.";
    return;
  }
  CHECK(options_.enable_schedule_overlap());
  SpecStartupCalibrationConfig config = target_verify_calibration_config();
  if (!options_.adaptive_target_verify_cost_table_output().empty()) {
    const std::filesystem::path output =
        options_.adaptive_target_verify_cost_table_output();
    const std::filesystem::path parent = output.has_parent_path()
                                             ? output.parent_path()
                                             : std::filesystem::path(".");
    std::error_code error;
    CHECK(std::filesystem::is_directory(parent, error))
        << "DSpark cost-table output directory is missing or inaccessible: "
        << parent << ": " << error.message();
    CHECK_EQ(::access(parent.c_str(), W_OK | X_OK), 0)
        << "DSpark cost-table output directory is not writable/searchable: "
        << parent;
    CHECK(!std::filesystem::exists(output, error) && !error)
        << "DSpark cost-table output already exists or cannot be inspected; "
           "choose a new path: "
        << output;
  }
  CHECK_EQ(geometry.at("slot_count").get<uint32_t>(), 2U);
  CHECK_EQ(geometry.at("draft_width").get<uint32_t>(), config.draft_width);
  const auto requests =
      geometry.at("request_buckets").get<std::vector<uint32_t>>();
  const auto graphs =
      geometry.at("target_graph_rows").get<std::vector<uint32_t>>();
  if (options_.adaptive_target_verify_calibration().empty()) {
    config.request_buckets = requests;
    std::erase_if(config.request_buckets, [&](uint32_t count) {
      const uint64_t total = static_cast<uint64_t>(count) * options_.dp_size();
      return total > static_cast<uint64_t>(options_.max_seqs_per_batch()) ||
             total > static_cast<uint64_t>(options_.max_tokens_per_batch());
    });
  }
  const auto configuration =
      nlohmann::json::parse(serialize_spec_startup_calibration(config));
  LOG(INFO)
      << "DSpark adaptive effective configuration: "
      << nlohmann::json{{"source",
                         options_.adaptive_target_verify_calibration().empty()
                             ? "built-in defaults"
                             : "JSON file"},
                        {"path", options_.adaptive_target_verify_calibration()},
                        {"calibration", configuration},
                        {"captured_geometry", geometry}}
             .dump();
  const auto& coverage = geometry.at("graph_coverage");
  std::vector<SpecCalibrationGraphCoverage> graph_coverage;
  graph_coverage.reserve(coverage.size());
  for (const auto& entry : coverage) {
    graph_coverage.emplace_back(SpecCalibrationGraphCoverage{
        entry.at("requests").get<uint32_t>(),
        entry.at("target_graph_rows").get<std::vector<uint32_t>>()});
  }
  const auto points =
      plan_spec_calibration(config, requests, graphs, graph_coverage);
  CHECK(points)
      << "Calibration requires captured buckets and full-width coverage.";
  for (const auto& point : *points) {
    CHECK_LE(static_cast<uint64_t>(point.requests) * options_.dp_size(),
             static_cast<uint64_t>(options_.max_seqs_per_batch()));
    CHECK_LE(static_cast<uint64_t>(point.requests) * options_.dp_size(),
             static_cast<uint64_t>(options_.max_tokens_per_batch()));
    CHECK_LE(
        static_cast<uint64_t>(point.context.max_kv_length),
        static_cast<uint64_t>(engine_->model_args().max_position_embeddings()));
    const auto& bounds = geometry.at("kv_bounds");
    const auto captured =
        std::find_if(bounds.begin(), bounds.end(), [&](const auto& bound) {
          return bound.at("requests").template get<uint32_t>() ==
                 point.requests;
        });
    CHECK(captured != bounds.end());
    CHECK_LE(point.context.max_kv_length,
             captured->at("max_kv_length").get<uint32_t>());
  }
  CHECK_GT(engine_->model_args().vocab_size(), 2);
  nlohmann::json bundle{{"schema_version", 1},
                        {"measurement", "steady_full_step"},
                        {"compatibility", compatibility},
                        {"profiles", nlohmann::json::array()}};
  std::vector<SpecSteadyWindow> context_windows;
  bundle["calibration"] = configuration;
  bundle["captured_geometry"] = geometry;
  for (size_t index = 0; index < points->size(); ++index) {
    const auto& point = (*points)[index];
    const auto started = distributed_worker_manager_->spec_calibration(
        {SpecCalibrationAction::BEGIN_SAMPLE,
         nlohmann::json{{"configuration", configuration}, {"point", index}}
             .dump()});
    CHECK(started && started->size() == capabilities->size());
    CHECK(std::all_of(
        started->begin(), started->end(), [](const std::string& value) {
          return value == "sample_started";
        }));
    run_startup_speculative_sample(
        config,
        point,
        index,
        []() {},
        [&]() {
          // Every rank's Slots are retired before finishing the measurement.
          const auto finished = distributed_worker_manager_->spec_calibration(
              {SpecCalibrationAction::FINISH_SAMPLE, {}});
          CHECK(finished && finished->size() == capabilities->size());
          CHECK(std::all_of(
              std::next(finished->begin()),
              finished->end(),
              [](const std::string& value) { return value == "[]"; }))
              << "Only the designated DSpark leader may return calibration "
                 "samples.";
          const auto windows = nlohmann::json::parse(finished->front());
          CHECK(windows.is_array() &&
                windows.size() == config.windows_per_point);
          for (const auto& sample : windows) {
            context_windows.emplace_back(read_window(sample));
          }
        });
    LOG(INFO) << "DSpark internal calibration sample complete: Bcap="
              << point.requests << ", G=" << point.target_graph_rows
              << ", windows=" << config.windows_per_point;
    const bool last_context_point =
        index + 1 == points->size() ||
        (*points)[index + 1].context.max_kv_length !=
            point.context.max_kv_length;
    if (last_context_point) {
      const auto fitted = fit_spec_cost_table(context_windows,
                                              config.model,
                                              point.context.min_kv_length,
                                              point.context.max_kv_length,
                                              config.draft_width);
      CHECK(fitted) << "Cannot fit an identifiable positive DSpark cost table.";
      bundle["profiles"].emplace_back(
          nlohmann::json{{"min_kv_length", point.context.min_kv_length},
                         {"max_kv_length", point.context.max_kv_length},
                         {"request_buckets", config.request_buckets},
                         {"table", nlohmann::json::parse(*fitted)}});
      context_windows.clear();
    }
  }
  const std::string serialized = bundle.dump();
  install(serialized);
  if (!options_.adaptive_target_verify_cost_table_output().empty()) {
    save_new_table(options_.adaptive_target_verify_cost_table_output(),
                   serialized + '\n');
  }
  LOG(INFO) << "DSpark startup calibration complete; immutable full-step costs "
               "installed.";
}

void ProfileManager::run_startup_speculative_sample(
    const SpecStartupCalibrationConfig& config,
    const SpecCalibrationPoint& point,
    uint64_t sample_id,
    const std::function<void()>& after_prefill,
    const std::function<void()>& after_drain) {
  const uint64_t decode_rounds =
      static_cast<uint64_t>(config.warmup_steps) +
      static_cast<uint64_t>(config.window_steps) * config.windows_per_point + 1;
  CHECK_GT(engine_->model_args().vocab_size(), 2);
  std::mt19937 random(config.token_seed);
  std::uniform_int_distribution<int32_t> token(
      1, engine_->model_args().vocab_size() - 1);
  std::vector<std::shared_ptr<Request>> owned;
  std::vector<Sequence*> sequences;
  const uint32_t total_requests = point.requests * options_.dp_size();
  owned.reserve(total_requests);
  sequences.reserve(total_requests);
  const size_t capacity = point.context.prefill_tokens +
                          (decode_rounds + 3) * (config.draft_width + 1);
  for (uint32_t row = 0; row < total_requests; ++row) {
    std::vector<int32_t> prompt(point.context.prefill_tokens);
    std::generate(
        prompt.begin(), prompt.end(), [&]() { return token(random); });
    RequestState state(std::move(prompt));
    state.seq_capacity = capacity;
    state.n = 1;
    state.best_of = 1;
    state.logprobs = false;
    state.enable_schedule_overlap = true;
    state.is_graph_warmup = false;
    state.sampling_param.temperature = 0.0;
    state.sampling_param.top_k = 0;
    state.sampling_param.top_p = 1.0;
    state.stopping_checker.set_ignore_eos(true);
    state.stopping_checker.set_max_generated_tokens((decode_rounds + 3) *
                                                    (config.draft_width + 1));
    state.stopping_checker.set_max_context_len(capacity);
    auto request = std::make_shared<Request>("spec-calibration-" +
                                                 std::to_string(sample_id) +
                                                 "-" + std::to_string(row),
                                             "",
                                             "",
                                             std::move(state));
    Sequence* sequence = request->sequences().front().get();
    sequence->set_dp_rank(static_cast<int32_t>(row / point.requests));
    CHECK(block_manager_pool_->allocate(sequence, capacity))
        << "Insufficient KV capacity for internal speculative calibration.";
    sequences.emplace_back(sequence);
    owned.emplace_back(std::move(request));
  }
  // Use the real prefill path to bootstrap target/draft KV and hidden state.
  // Drain it before measuring the steady two-Slot decode path.
  while (sequences.front()->stage() != SequenceStage::DECODE) {
    std::vector<size_t> budgets;
    budgets.reserve(sequences.size());
    for (Sequence* sequence : sequences) {
      const size_t remaining = sequence->num_prompt_tokens() -
                               sequence->kv_state().kv_cache_tokens_num();
      budgets.emplace_back(
          std::min(remaining,
                   static_cast<size_t>(options_.max_tokens_per_batch()) /
                       sequences.size()));
    }
    auto batch =
        batch_factory_.create_batches(owned, sequences, budgets, nullptr);
    step_callback_(batch);
    result_callback_(batch);
  }
  after_prefill();
  std::optional<BatchGroup> previous;
  const std::vector<size_t> budgets(sequences.size(), 1);
  for (uint64_t round = 0; round < decode_rounds; ++round) {
    auto batch =
        batch_factory_.create_batches(owned, sequences, budgets, nullptr);
    step_callback_(batch);
    if (previous) {
      result_callback_(*previous);
    }
    previous = std::move(batch);
  }
  CHECK(previous.has_value());
  result_callback_(*previous);
  previous.reset();
  after_drain();
  for (Sequence* sequence : sequences) {
    block_manager_pool_->deallocate_without_cache(sequence);
  }
}

}  // namespace xllm
