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

#include "profile_manager.h"

#include <absl/time/time.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/mtp_utils.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request_state.h"
#include "scheduler/profile/graph_warmup.h"
#include "util/rec_model_utils.h"
#include "util/utils.h"

namespace xllm {

ProfileManager::ProfileManager(Engine* engine, const Options& options)
    : options_(options), engine_(engine) {
  CHECK(engine_ != nullptr);
  block_manager_pool_ = engine_->block_manager_pool();
  CHECK(block_manager_pool_ != nullptr);
  prefill_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), true /*is_prefill*/);
  decode_time_predictor_ = std::make_unique<TimePredictor>(
      options.enable_profile_kv_blocks(), false /*is_prefill*/);
  if (options.enable_profile_step_time()) {
    LOG(INFO) << "Starting profiliing step time.";
    profile_step_time(false);
    // test accuracy
    // eval_sequence_latency_prediction();
    // eval_batch_latency_prediction("only_prefill");
    // eval_batch_latency_prediction("only_decode");
    // eval_batch_latency_prediction("mix");
  }
  if (options.enable_profile_token_budget()) {
    LOG(INFO) << "Starting profiliing token budget.";
    profile_token_budget();
  }
  // more profile here, such as token_budget profile and decode length
  // prediction.

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  // Warmup ACL graph executor if enabled
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    if (!is_rec_multi_round_mode()) {
      warmup_for_graph();
    }
  }
#endif
}

// --------------------- for test only ---------------------------
void ProfileManager::eval_sequence_latency_prediction() {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;
  int32_t token_step = 500;
  int32_t prefix_step = 500;
  int32_t upper_bound = 4000;

  LOG(INFO) << "Starting testing sequence latency prediction";
  for (int32_t token_length = token_step; token_length < upper_bound;
       token_length += token_step) {
    for (int32_t prefix_length = 0; prefix_length < token_length;
         prefix_length += prefix_step) {
      target_vec.emplace_back(run_request(token_length, prefix_length));
      pred_vec.emplace_back(predict_step_time(token_length, prefix_length));
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
void ProfileManager::eval_batch_latency_prediction(const std::string mode) {
  std::vector<double> pred_vec;
  std::vector<double> target_vec;

  LOG(INFO) << "Starting testing batch latency prediction for " << mode;
  if (mode == "only_prefill") {
    int32_t max_batch_size = 10;
    int32_t token_step = 500;
    int32_t prefix_step = 500;
    int32_t upper_bound = 4000;
    for (int32_t token_length = token_step; token_length < upper_bound;
         token_length += token_step) {
      for (int32_t prefix_length = 0; prefix_length < token_length;
           prefix_length += prefix_step) {
        target_vec.emplace_back(
            run_request(token_length, prefix_length, max_batch_size));
        pred_vec.emplace_back(
            predict_step_time(token_length, prefix_length, max_batch_size));
      }
    }
  }
  if (mode == "only_decode") {
    int32_t max_batch_size = 200;
    int32_t token_length = 500;
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size++) {
      target_vec.emplace_back(
          run_request(token_length, token_length - 1, batch_size));
      pred_vec.emplace_back(
          predict_step_time(token_length, token_length - 1, batch_size));
    }
  }
  if (mode == "mix") {
    if (!::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
      LOG(WARNING) << "When chunked prefill is disabled, mixed prefill and "
                      "decode scenarios will not be tested.";
      return;
    }
    int32_t max_batch_size = 100;
    int32_t max_prefill_cnt = 5;
    int32_t token_length = 500;
    for (int32_t batch_size = 50; batch_size <= max_batch_size;
         batch_size += 10) {
      for (int32_t prefill_cnt = 0; prefill_cnt <= max_prefill_cnt;
           prefill_cnt++) {
        std::vector<int32_t> token_length_vec;
        std::vector<int32_t> prefix_length_vec;
        token_length_vec.insert(
            token_length_vec.end(), prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(), prefill_cnt, 0);
        // token_length_vec.insert(token_length_vec.end(), batch_size/5,
        // token_length); prefix_length_vec.insert(prefix_length_vec.end(),
        // batch_size/5, token_length-1);
        token_length_vec.insert(
            token_length_vec.end(), batch_size - prefill_cnt, token_length);
        prefix_length_vec.insert(prefix_length_vec.end(),
                                 batch_size - prefill_cnt,
                                 token_length - 1);
        target_vec.emplace_back(
            run_request(token_length_vec, prefix_length_vec));
        pred_vec.emplace_back(
            predict_step_time(token_length_vec, prefix_length_vec));
      }
    }
  }

  // print for debug
  for (const auto& element : pred_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;
  for (const auto& element : target_vec) {
    std::cout << static_cast<int32_t>(element) << " ";
  }
  std::cout << std::endl;

  double sum_error = 0.0;
  double sum_percentage_error = 0.0;

  for (size_t i = 0; i < pred_vec.size(); ++i) {
    double error = std::abs(pred_vec[i] - target_vec[i]);
    sum_error += error;
    sum_percentage_error += error / std::abs(target_vec[i]);
  }
  double mae = sum_error / pred_vec.size();
  double mape = (sum_percentage_error / pred_vec.size()) * 100.0;

  LOG(INFO) << "Mean Absolute Error (MAE) of latency prediction: " << mae
            << " ms";
  LOG(INFO) << "Mean Absolute Percentage Error (MAPE) of latency prediction: "
            << mape << " %";
}
// -------------------------------------------------------------

// ---------------------- dump to file-----------------------
std::string ProfileManager::generate_filename(const std::string& file_suffix) {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");

  std::string filename;
  filename = ss.str() + "_" + file_suffix + ".txt";

  return filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::pair<int32_t, double>>& time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << data.first << "," << data.second << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}

void ProfileManager::dump_step_time_profile_to_file(
    const std::vector<std::tuple<int32_t, int32_t, double>>&
        time_profiling_data,
    bool is_prefill) {
  std::string filename = is_prefill
                             ? generate_filename("profile_prefill_step_time")
                             : generate_filename("profile_decode_step_time");
  std::ofstream outfile(filename);
  if (!outfile.is_open()) {
    LOG(FATAL) << "Could not open file " << filename << " for writing.";
    return;
  }
  // write data
  for (const auto& data : time_profiling_data) {
    outfile << std::get<0>(data) << "," << std::get<1>(data) << ","
            << std::get<2>(data) << std::endl;
  }
  outfile.close();
  LOG(INFO) << "Profile data saved to: " << filename;
}
// -------------------------------------------------------------

void ProfileManager::profile_step_time(bool if_dump_to_file) {
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  // TODO: support length for decode request profile
  int32_t profile_max_prompt_length =
      std::min(max_context_len, options_.profile_max_prompt_length());
  auto block_size = block_manager_pool_->options().block_size();
  bool enable_profile_kv_blocks = options_.enable_profile_kv_blocks();

  // warm up
  run_request(profile_max_prompt_length, 0);

  // prefill time profile
  if (options_.enable_profile_kv_blocks()) {
    // starting from max_context_len, dividing the token length by 2 in
    // each loop iteration
    // consider to generate kv blocks for prompt
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length >>= 1) {
      // increase prefix length according to block size
      auto block_step = (profile_length_step_ + block_size - 1) / block_size;
      for (int32_t prefix_length = 0;
           prefix_length < token_length - 1 + (block_step * block_size);
           prefix_length += (block_step * block_size)) {
        if (prefix_length > token_length - 1) {
          // avoid kv_cache_token_num == token_length
          prefix_length = token_length - 1;
        }
        double latency_mean = 0;

        for (int32_t k = 0; k < profile_count_per_step_; k++) {
          latency_mean += run_request(token_length, prefix_length);
        }
        latency_mean /= profile_count_per_step_;
        // use token_length and prefix_length to predict
        time_profiling_data.emplace_back(
            token_length, prefix_length, latency_mean);
      }
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  } else {
    // not consider kv cache
    std::vector<std::pair<int32_t, double>> time_profiling_data;
    for (int32_t token_length = profile_max_prompt_length; token_length > 1;
         token_length *= 0.8) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, 0);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, latency_mean);
    }
    if (if_dump_to_file) {
      dump_step_time_profile_to_file(time_profiling_data, true /*is_prefill*/);
    }
    train_prefill_time_predictor(time_profiling_data);
  }
  if (::xllm::DisaggPDConfig::get_instance().enable_disagg_pd()) {
    LOG(INFO) << "Disagg PD enabled, skip decode time profile.";
    return;
  }
  // decode time profile

  std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data;
  int32_t max_batch_size = 25;
  // for (int32_t token_length = profile_max_prompt_length; token_length >
  // 1;token_length >>= 1)
  for (int32_t token_length = 2; token_length < profile_max_prompt_length;
       token_length += profile_length_step_) {
    for (int32_t batch_size = 1; batch_size < max_batch_size; batch_size += 2) {
      double latency_mean = 0;
      for (int32_t k = 0; k < profile_count_per_step_; k++) {
        latency_mean += run_request(token_length, token_length - 1, batch_size);
      }
      latency_mean /= profile_count_per_step_;
      time_profiling_data.emplace_back(token_length, batch_size, latency_mean);
    }
  }
  if (if_dump_to_file) {
    dump_step_time_profile_to_file(time_profiling_data, false /*is_prefill*/);
  }
  train_decode_time_predictor(time_profiling_data);
}

void ProfileManager::train_prefill_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_prefill_time_predictor(
    std::vector<std::pair<int32_t, double>> time_profiling_data) {
  prefill_time_predictor_->fit_for_prefill(time_profiling_data);
}
void ProfileManager::train_decode_time_predictor(
    std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data) {
  decode_time_predictor_->fit_for_decode(time_profiling_data);
}

// ----------------------predict step time-----------------------
std::vector<double> ProfileManager::get_coefficients(bool is_prefill) {
  if (is_prefill) {
    return prefill_time_predictor_->get_coefficients();
  } else {
    return decode_time_predictor_->get_coefficients();
  }
}

double ProfileManager::get_constant_overhead() {
  if (prefill_time_predictor_->is_trained() &&
      decode_time_predictor_->is_trained()) {
    return (prefill_time_predictor_->get_constant_overhead() +
            decode_time_predictor_->get_constant_overhead()) /
           2;
  } else if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_constant_overhead();
  } else if (decode_time_predictor_->is_trained()) {
    return decode_time_predictor_->get_constant_overhead();
  }
  return 0.0;
}

int32_t ProfileManager::get_quadratic_root(Sequence* sequence, double budget) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_state().kv_cache_tokens_num();
  if (prefill_time_predictor_->is_trained()) {
    return prefill_time_predictor_->get_quadratic_root(prefix_length, budget);
  }
  LOG(ERROR) << "Prefill time predictor is not trained yet.";
  return 0;
}

// for single sequence
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  CHECK(length > prefix_length)
      << "Token length (" << length << ") must be greater than prefix length "
      << " (" << prefix_length << ").";
  double ratio = 1.0;
  if (force_use_prefill_predictor) {
    return ratio * prefill_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  }
  if (length - 1 == prefix_length) {
    return ratio * decode_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  } else {
    return ratio * prefill_time_predictor_->predict_time(
                       length, prefix_length, if_need_add_constant_term);
  }
}

double ProfileManager::predict_step_time(Sequence* sequence,
                                         bool if_need_add_constant_term,
                                         bool force_use_prefill_predictor) {
  auto length = sequence->num_tokens();
  auto prefix_length = sequence->kv_cache_tokens_num();
  double latency = predict_step_time(length,
                                     prefix_length,
                                     if_need_add_constant_term,
                                     force_use_prefill_predictor);
  return latency;
}
// for single batch or sequences
double ProfileManager::predict_step_time(
    const std::vector<int32_t>& length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(length_vec.size() == prefix_length_vec.size());
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < length_vec.size(); i++) {
    // predict for each sequence
    int32_t length = length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}

// for seq in batch with the same token and prefix length
double ProfileManager::predict_step_time(int32_t length,
                                         int32_t prefix_length,
                                         int32_t batch_size) {
  double total_latency = get_constant_overhead();
  for (int32_t i = 0; i < batch_size; i++) {
    // predict for each sequence
    total_latency += predict_step_time(length, prefix_length, false);
  }
  return total_latency;
}
// ---------------------------------------------

// ----------------------for profile token budget-----------------------
void ProfileManager::profile_token_budget() {
  // use token budget means defaultly ignoring prefix cache and decode request's
  // kv cache load overhead
  // warm up
  run_request(options_.profile_max_prompt_length(), 0);
  profile_token_budget_ =
      binary_search_max_tokens(options_.max_global_tpot_ms(), 1, 4096);
  LOG(INFO) << "Profile token budget: " << profile_token_budget_
            << "for TPOT SLO: " << options_.max_global_tpot_ms();
}

bool ProfileManager::check_if_satisfy_slo(int32_t num_tokens,
                                          int32_t tpot_slo_ms) {
  // int32_t prompt_tokens_per_batch = 1024;
  // auto batch_size = num_tokens / prompt_tokens_per_batch;
  // int32_t extra_token_length = num_tokens % prompt_tokens_per_batch;
  // double batch_latency = 0;
  // for (int32_t k = 0; k < profile_count_per_step_; k++) {
  //   batch_latency +=
  //       run_request(prompt_tokens_per_batch, 0, batch_size,
  //       extra_token_length);
  // }
  double batch_latency = 0;
  for (int32_t k = 0; k < profile_count_per_step_; k++) {
    batch_latency += run_request(num_tokens, 0, 1, 0);
  }
  batch_latency /= profile_count_per_step_;
  if (batch_latency <= tpot_slo_ms) {
    return true;
  } else {
    return false;
  }
}

int32_t ProfileManager::binary_search_max_tokens(int32_t tpot_slo_ms,
                                                 int32_t lower_bound,
                                                 int32_t upper_bound) {
  int32_t left = lower_bound;
  int32_t right = upper_bound;
  // [left, right)
  while (left < right) {
    int32_t mid = left + (right - left) / 2;
    if (check_if_satisfy_slo(mid, tpot_slo_ms)) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

int32_t ProfileManager::get_token_budget() { return profile_token_budget_; }

// ---------------------------------------------

const std::vector<ProfileManager::CopyBlockProfile>&
ProfileManager::get_copy_block_profile() {
  // NOTE: Add more model profiles here
  static const std::vector<CopyBlockProfile> profiles = {
      // offline copy block profile
      {"Qwen2-7B", 128, 0.48, 0.24, "Qwen2-7B, block_size=128"},
      {"Qwen2-7B", 64, 0.20, 0.25, "Qwen2-7B, block_size=128"},
  };

  return profiles;
}

const ProfileManager::CopyBlockProfile* ProfileManager::find_profile(
    const std::string& model_name,
    int32_t block_size) const {
  const auto& profiles = get_copy_block_profile();
  for (const auto& profile : profiles) {
    if ((profile.model_name == model_name ||
         model_name.find(profile.model_name) != std::string::npos) &&
        profile.block_size == block_size) {
      return &profile;
    }
  }
  LOG(ERROR) << "No profile found for " << model_name
             << " with block_size=" << block_size << ", using default values";
  return nullptr;
}

int32_t ProfileManager::get_max_copy_block_num(double latency_budget) {
  auto block_size = block_manager_pool_->options().block_size();
  const CopyBlockProfile* profile =
      find_profile(::xllm::ModelConfig::get_instance().model_id(), block_size);

  double a = 1, b = 0;  // default values
  if (profile) {
    a = profile->slope;
    b = profile->intercept;
  }

  double max_blocks = std::max((latency_budget - b) / a, 0.0);
  return static_cast<int32_t>(max_blocks);
}

double ProfileManager::predict_copy_blocks_time(
    size_t num_copy_blocks,
    bool if_need_add_constant_term) {
  auto block_size = block_manager_pool_->options().block_size();
  const CopyBlockProfile* profile =
      find_profile(::xllm::ModelConfig::get_instance().model_id(), block_size);

  double a = 1, b = 0;  // default values
  if (profile) {
    a = profile->slope;
    b = profile->intercept;
  }
  return if_need_add_constant_term ? a * num_copy_blocks + b
                                   : a * num_copy_blocks;
}

std::shared_ptr<Request> ProfileManager::generate_single_request(
    int32_t token_length,
    int32_t prefix_length) {
  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t eos_token_id = model_args.eos_token_id();

  std::random_device rd;
  std::mt19937_64 gen(rd());

  // If req_state does not initialize the stopchecker, default eos_token_id = 0,
  // need to skip it
  std::uniform_int_distribution<int32_t> dis(1, vocab_size - 2);

  std::vector<int32_t> token_ids(token_length);
  std::generate(token_ids.begin(), token_ids.end(), [&]() {
    int32_t token = dis(gen);
    return token == eos_token_id ? token + 1 : token;  // skip eos
  });

  RequestState req_state(token_ids);
  req_state.enable_schedule_overlap = options_.enable_schedule_overlap();
  auto request = std::make_shared<Request>(
      /*request_id=*/next_warmup_request_id(),
      /*x_request_id=*/"",
      /*x_request_time=*/"",
      req_state);

  // TODO: better disable prefix cache
  if (prefix_length > 0) {
    if (!block_manager_pool_->BlockManagerPool::allocate(
            request->sequences()[0].get(), prefix_length)) {
      LOG(FATAL) << "Profiling time failed! Not enough blocks, prefix length : "
                 << prefix_length;
    }
    request->sequences()[0]->kv_state().incr_kv_cache_tokens_num(prefix_length);
  }

  if (!block_manager_pool_->BlockManagerPool::allocate(
          request->sequences()[0].get(), token_length)) {
    LOG(FATAL) << "Profiling time failed! Not enough blocks, token length : "
               << token_length;
  }

  return request;
}

std::shared_ptr<Request> ProfileManager::generate_single_decode_request(
    int32_t total_length,
    std::optional<int32_t> dp_rank) {
  CHECK_GT(total_length, 1) << "Decode profiling requires total_length > 1.";

  auto& model_args = engine_->model_args();
  int32_t vocab_size = model_args.vocab_size();
  int32_t eos_token_id = model_args.eos_token_id();

  std::random_device rd;
  std::mt19937_64 gen(rd());

  // If req_state does not initialize the stopchecker, default eos_token_id = 0,
  // need to skip it
  std::uniform_int_distribution<int32_t> dis(1, vocab_size - 2);

  const int32_t prompt_length = total_length - 1;
  std::vector<int32_t> prompt_token_ids(prompt_length);
  std::generate(prompt_token_ids.begin(), prompt_token_ids.end(), [&]() {
    int32_t token = dis(gen);
    return token == eos_token_id ? token + 1 : token;  // skip eos
  });

  RequestState req_state(prompt_token_ids);
  req_state.enable_schedule_overlap = options_.enable_schedule_overlap();
  const int32_t num_speculative_tokens =
      ::xllm::SpeculativeConfig::get_instance().num_speculative_tokens();
  size_t seq_capacity = static_cast<size_t>(total_length) +
                        static_cast<size_t>(num_speculative_tokens) + 1;
  if (options_.enable_schedule_overlap()) {
    seq_capacity += static_cast<size_t>(num_speculative_tokens) + 1;
  }
  req_state.seq_capacity = seq_capacity;
  auto request = std::make_shared<Request>(
      /*request_id=*/next_warmup_request_id(),
      /*x_request_id=*/"",
      /*x_request_time=*/"",
      req_state);

  auto* sequence = request->sequences()[0].get();
  if (dp_rank.has_value()) {
    CHECK_GE(dp_rank.value(), 0);
    CHECK_LT(dp_rank.value(), options_.dp_size());
    sequence->set_dp_rank(dp_rank.value());
  }
  if (!block_manager_pool_->BlockManagerPool::allocate(sequence,
                                                       seq_capacity)) {
    LOG(FATAL) << "Profiling decode step time failed! Not enough blocks, total "
                  "length: "
               << total_length;
  }
  sequence->kv_state().incr_kv_cache_tokens_num(prompt_length);

  int32_t generated_token = dis(gen);
  generated_token =
      generated_token == eos_token_id ? generated_token + 1 : generated_token;
  sequence->append_token(generated_token);

  // With MTP speculative decoding the worker's decode path requires a valid
  // decode state written via the MTP bootstrap channel before validating the
  // per-token decode state. Inject a placeholder bootstrap embedding so the
  // synthetic warmup/profile request takes the same bootstrap path as a real
  // disagg PD decode request instead of reading stale recycled decode state.
  const int64_t bootstrap_width = mtp_hidden_state_width(model_args);
  prepare_warmup_decode_sequence(
      sequence, bootstrap_width, num_speculative_tokens);

  CHECK(sequence->stage() == SequenceStage::DECODE)
      << "Decode profiling request is not in DECODE stage. total_length: "
      << total_length << ", prompt_length: " << prompt_length
      << ", kv_cache_tokens_num: " << sequence->kv_state().kv_cache_tokens_num()
      << ", num_tokens: " << sequence->num_tokens();
  CHECK_EQ(sequence->num_generated_tokens(), 1)
      << "Decode profiling request should start with one generated token.";

  return request;
}

// collect the latency of each step
double ProfileManager::run_request(int32_t token_length,
                                   int32_t prefix_length,
                                   int32_t batch_size,
                                   int32_t extra_token_length) {
  CHECK(token_length >= prefix_length);
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(batch_size);
  sequences_budget.reserve(batch_size);
  requests.reserve(batch_size);

  // batch sequences with the same kv cahce and token length
  for (int32_t i = 0; i < batch_size; i++) {
    // generate random token ids and request
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // maybe another sequence for extra token length (< token_length) for token
  // budget profiling
  if (extra_token_length > 0) {
    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(requests, sequences, sequences_budget);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

// currently for test only
double ProfileManager::run_request(
    const std::vector<int32_t>& token_length_vec,
    const std::vector<int32_t>& prefix_length_vec) {
  CHECK(token_length_vec.size() == prefix_length_vec.size());
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(token_length_vec.size());
  sequences_budget.reserve(token_length_vec.size());
  requests.reserve(token_length_vec.size());

  // batch sequences with the same kv cahce and token length
  for (int32_t i = 0; i < token_length_vec.size(); i++) {
    // generate random token ids and request
    int32_t token_length = token_length_vec[i];
    int32_t prefix_length = prefix_length_vec[i];

    std::shared_ptr<Request> request =
        generate_single_request(token_length, prefix_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(token_length - prefix_length);
  }
  // build batch
  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

double ProfileManager::run_decode_request(
    const std::vector<int32_t>& total_length_vec) {
  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;

  for (int32_t total_length : total_length_vec) {
    std::shared_ptr<Request> request =
        generate_single_decode_request(total_length);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(1);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

double ProfileManager::run_graph_decode_request(
    const std::vector<int32_t>& total_length_vec) {
  CHECK_GT(options_.dp_size(), 0);

  std::vector<Sequence*> sequences;
  std::vector<size_t> sequences_budget;
  std::vector<std::shared_ptr<Request>> requests;
  sequences.reserve(total_length_vec.size());
  sequences_budget.reserve(total_length_vec.size());
  requests.reserve(total_length_vec.size());

  for (size_t i = 0; i < total_length_vec.size(); ++i) {
    int32_t dp_rank = static_cast<int32_t>(i % options_.dp_size());
    std::shared_ptr<Request> request =
        generate_single_decode_request(total_length_vec[i], dp_rank);
    requests.emplace_back(request);
    sequences.emplace_back(request->sequences()[0].get());
    sequences_budget.emplace_back(1);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(requests, sequences, sequences_budget, nullptr);

  absl::Time start_time = absl::Now();
  engine_->step(batches);
  if (options_.enable_schedule_overlap()) {
    engine_->update_last_step_result(batches);
  }
  double latency = absl::ToDoubleMilliseconds(absl::Now() - start_time);
  for (auto& request : requests) {
    block_manager_pool_->deallocate_without_cache(
        request->sequences()[0].get());
  }

  return latency;
}

// Generate a batch of decode requests in DECODE stage and execute one decode
// step, then return the step latency.
double ProfileManager::profile_decode_step_time(int32_t token_length,
                                                int32_t batch_size,
                                                int32_t min_context_len,
                                                int32_t max_context_len) {
  double total_latency = 0.0;
  for (int32_t i = 0; i < profile_count_per_step_; ++i) {
    std::vector<int32_t> token_length_vec;
    generate_random_decode_batch(batch_size * token_length,
                                 batch_size,
                                 min_context_len,
                                 max_context_len,
                                 token_length_vec);
    total_latency += run_decode_request(token_length_vec);
  }
  return total_latency / profile_count_per_step_;
}

// Generate a batch of random decode requests with an average total sequence
// length of token_length.
void ProfileManager::generate_random_decode_batch(
    int32_t total_length,
    int32_t batch_size,
    int32_t min_context_len,
    int32_t max_context_len,
    std::vector<int32_t>& token_length_vec) {
  CHECK(total_length >= batch_size * min_context_len);
  CHECK(total_length <= batch_size * max_context_len);

  token_length_vec.resize(batch_size, min_context_len);
  int remain = total_length - batch_size * min_context_len;

  std::random_device rd;
  std::mt19937_64 gen(rd());

  for (int i = 0; i < batch_size; ++i) {
    if (remain == 0) break;

    int max = remain > (max_context_len - min_context_len)
                  ? (max_context_len - min_context_len)
                  : remain;

    std::uniform_int_distribution<int> dis(0, max);
    int add = dis(gen);
    token_length_vec[i] += add;
    remain -= add;
  }

  int idx = 0;
  while (remain > 0) {
    if (token_length_vec[idx % batch_size] < max_context_len) {
      token_length_vec[idx % batch_size] += 1;
      --remain;
    }
    ++idx;
  }
}

void ProfileManager::warmup_for_graph() {
  const GraphWarmupPlan plan = graph_warmup_plan(options_.instance_role());
  if (plan == GraphWarmupPlan::PREFILL_ONLY) {
    LOG(INFO) << "PREFILL graph warmup: prefill only";
    warmup_prefill_for_graph();
    return;
  }
  if (plan == GraphWarmupPlan::DECODE_ONLY) {
    LOG(INFO) << "DECODE graph warmup: decode buckets only";
    warmup_decode_for_graph();
    return;
  }

  warmup_unified_for_graph();
}

void ProfileManager::warmup_prefill_for_graph() {
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();

  int32_t prefill_tokens =
      std::min(options_.max_tokens_per_batch(), max_context_len);
  double prefill_latency =
      run_request(prefill_tokens, /*prefix_length=*/0, /*batch_size=*/1);
  LOG(INFO) << "Prefill warmup completed: tokens=" << prefill_tokens
            << ", latency=" << prefill_latency << " ms";
}

void ProfileManager::warmup_unified_for_graph() {
  warmup_prefill_for_graph();
  warmup_decode_for_graph();
}

void ProfileManager::warmup_decode_for_graph() {
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();
  int32_t max_decode_batch_size = options_.max_seqs_per_batch();
  const int32_t max_concurrent_requests =
      ::xllm::ServiceConfig::get_instance().max_concurrent_requests();
  if (max_concurrent_requests > 0) {
    max_decode_batch_size =
        std::min(max_decode_batch_size, max_concurrent_requests);
  }
  int32_t decode_seq_len = std::min(16, max_context_len);

  std::vector<int32_t> decode_batch_sizes =
      graph_decode_buckets(max_decode_batch_size, options_.dp_size());
  const int32_t decode_bucket_count =
      static_cast<int32_t>(decode_batch_sizes.size());

  LOG(INFO) << "Graph warmup started: bucket_count=" << decode_bucket_count
            << ", decode_seq_len=" << decode_seq_len;

  // Capture from the largest bucket down to the smallest so every smaller
  // bucket reuses the scratch blocks freed by the largest capture within the
  // shared graph mempool. Ascending capture forces each larger bucket to grab
  // fresh scratch (the freed smaller blocks cannot satisfy it), which makes the
  // pool grow linearly with the bucket count.
  double decode_total_latency = 0.0;
  for (int32_t bucket_index = decode_bucket_count - 1; bucket_index >= 0;
       --bucket_index) {
    const int32_t batch_size =
        decode_batch_sizes[static_cast<size_t>(bucket_index)];
    std::vector<int32_t> total_length_vec(batch_size, decode_seq_len);
    const double decode_latency = run_graph_decode_request(total_length_vec);
    decode_total_latency += decode_latency;
    LOG(INFO) << graph_warmup_progress(
        /*completed=*/decode_bucket_count - bucket_index,
        /*total=*/decode_bucket_count,
        /*bucket=*/batch_size,
        /*latency_ms=*/decode_latency);
  }

  LOG(INFO) << "Decode warmup completed: bucket_count=" << decode_bucket_count
            << ", decode_max_batch_size="
            << (decode_batch_sizes.empty() ? 0 : decode_batch_sizes.back())
            << ", decode_seq_len=" << decode_seq_len
            << ", decode_total_latency=" << decode_total_latency << " ms";
}

}  // namespace xllm
