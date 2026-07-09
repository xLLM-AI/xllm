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

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "common/macros.h"
#include "common/types.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/xservice_client.h"
#include "time_predictor.h"

namespace xllm {
class Engine;
class ProfileManager {
 public:
  struct Options {
    PROPERTY(bool, enable_schedule_overlap) = false;

    PROPERTY(int32_t, dp_size) = 1;
    // config for profile
    PROPERTY(bool, enable_profile_step_time) = false;

    PROPERTY(bool, enable_profile_token_budget) = false;

    PROPERTY(int32_t, profile_max_prompt_length) = 2048;

    PROPERTY(bool, enable_profile_kv_blocks) = true;

    PROPERTY(int32_t,
             max_tokens_per_batch) = std::numeric_limits<int32_t>::max();

    PROPERTY(int32_t, max_seqs_per_batch) = 1024;

    PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();

    PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();

    PROPERTY(InstanceRole, instance_role) = InstanceRole::DEFAULT;
  };

  struct CopyBlockProfile {
    std::string model_name;
    int block_size;
    double slope;      // milliseconds per block
    double intercept;  // milliseconds constant overhead
    std::string note;
  };
  ProfileManager(Engine* engine, const Options& options);

  int32_t get_token_budget();

  int32_t get_max_copy_block_num(double latency_budget);

  double predict_copy_blocks_time(size_t num_copy_blocks,
                                  bool if_need_add_constant_term = true);

  // for single sequence
  double predict_step_time(int32_t length,
                           int32_t prefix_length = 0,
                           bool if_need_add_constant_term = true,
                           bool force_use_prefill_predictor = false);

  double predict_step_time(Sequence* sequence,
                           bool if_need_add_constant_term = true,
                           bool force_use_prefill_predictor = false);
  // for single batch or sequences
  double predict_step_time(int32_t length,
                           int32_t prefix_length,
                           int32_t batch_size);

  double predict_step_time(const std::vector<int32_t>& length_vec,
                           const std::vector<int32_t>& prefix_length_vec);
  // Generate a request of token_length and prefix_length, finally
  // executing and returning the inference time.
  double run_request(int32_t token_length,
                     int32_t prefix_length,
                     int32_t batch_size = 1,
                     int32_t extra_token_length = 0);
  double run_request(const std::vector<int32_t>& token_length_vec,
                     const std::vector<int32_t>& prefix_length_vec);

  // Generate a batch of decode requests in DECODE stage and execute one decode
  // step, then return the step latency.
  double profile_decode_step_time(int32_t token_length,
                                  int32_t batch_size,
                                  int32_t min_context_len,
                                  int32_t max_context_len);

  void train_prefill_time_predictor(
      std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data);

  void train_decode_time_predictor(
      std::vector<std::tuple<int32_t, int32_t, double>> time_profiling_data);

  void train_prefill_time_predictor(
      std::vector<std::pair<int32_t, double>> time_profiling_data);

  void train_speculative_validate_time_predictor(
      const std::vector<std::tuple<int32_t, int32_t, int32_t, double>>&
          time_profiling_data);

  double get_constant_overhead();

  int32_t get_quadratic_root(Sequence* sequence, double budget);

  std::vector<double> get_coefficients(bool is_prefill = true);

  void profile_step_time(bool if_dump_to_file);

 private:
  void dump_step_time_profile_to_file(
      const std::vector<std::pair<int32_t, double>>& time_profiling_data,
      bool is_prefill);

  void dump_step_time_profile_to_file(
      const std::vector<std::tuple<int32_t, int32_t, double>>&
          time_profiling_data,
      bool is_prefill);

  std::shared_ptr<Request> generate_single_request(int32_t token_length,
                                                   int32_t prefix_length);
  std::shared_ptr<Request> generate_single_decode_request(
      int32_t total_length,
      std::optional<int32_t> dp_rank = std::nullopt);

  std::string generate_filename(const std::string& file_suffix);

  void eval_sequence_latency_prediction();

  void eval_batch_latency_prediction(const std::string mode);

  void profile_token_budget();

  void profile_speculative_validate_time();

  // Warmup ACL graph executor according to the instance role.
  void warmup_for_graph();

  void warmup_prefill_for_graph();

  void warmup_decode_for_graph();

  void warmup_unified_for_graph();

  bool check_if_satisfy_slo(int32_t num_tokens, int32_t tpot_slo_ms);

  int32_t binary_search_max_tokens(int32_t tpot_slo_ms,
                                   int32_t lower_bound,
                                   int32_t upper_bound);

  // Generate a batch of random decode requests with an average total sequence
  // length of token_length.
  void generate_random_decode_batch(int32_t total_length,
                                    int32_t batch_size,
                                    int32_t min_context_len,
                                    int32_t max_context_len,
                                    std::vector<int32_t>& token_length_vec);

  double run_decode_request(const std::vector<int32_t>& total_length_vec);

  double run_graph_decode_request(const std::vector<int32_t>& total_length_vec);

  static const std::vector<ProfileManager::CopyBlockProfile>&
  get_copy_block_profile();

  const ProfileManager::CopyBlockProfile* find_profile(
      const std::string& model_name,
      int block_size) const;

  std::unique_ptr<TimePredictor> prefill_time_predictor_;
  std::unique_ptr<TimePredictor> decode_time_predictor_;

  const Options options_;

  Engine* engine_;

  BlockManagerPool* block_manager_pool_;

  int32_t profile_length_step_ = 256;

  int32_t profile_count_per_step_ = 3;

  int32_t profile_token_budget_ = std::numeric_limits<int32_t>::max();
};

}  // namespace xllm
