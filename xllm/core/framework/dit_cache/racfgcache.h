/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "dit_cache_impl.h"

namespace xllm {

class RACFGCache : public DitCacheImpl {
 public:
  RACFGCache() = default;
  ~RACFGCache() override = default;

  RACFGCache(const RACFGCache&) = delete;
  RACFGCache& operator=(const RACFGCache&) = delete;
  RACFGCache(RACFGCache&&) = default;
  RACFGCache& operator=(RACFGCache&&) = default;

  void init(const DiTCacheConfig& cfg) override;

  void set_runtime_context(const DiTCacheRuntimeContext& ctx) override;

  bool on_before_step(const CacheStepIn& stepin) override;
  CacheStepOut on_after_step(const CacheStepIn& stepin) override;

  bool on_before_block(const CacheBlockIn& blockin) override;
  CacheBlockOut on_after_block(const CacheBlockIn& blockin) override;

 private:
  static constexpr float kEps = 1e-6f;

  enum class ProxyErrorType : int64_t {
    DeltaY = 0,
    DeltaMinus = 1,
  };

  struct BranchLocalState {
    // residual cache
    torch::Tensor previous_residual;
    torch::Tensor previous_encoder_residual;

    // base state for this step
    torch::Tensor base_hidden_states;
    torch::Tensor base_encoder_hidden_states;

    torch::Tensor first_probe_input_prev;
    torch::Tensor probe_hidden_prev;
    // previous probe reference
    torch::Tensor proxy_prev_input;
    torch::Tensor proxy_prev_probe_states;

    // current-step probe tensors
    torch::Tensor current_probe_input;
    torch::Tensor current_probe_hidden;

    // current-step probe scalars
    float current_dx = std::numeric_limits<float>::quiet_NaN();
    float current_dy = std::numeric_limits<float>::quiet_NaN();
    float current_branch_error = std::numeric_limits<float>::quiet_NaN();

    bool probe_ready_this_step = false;
  };

  struct JointState {
    int64_t anchor_step = 0;
    float accumulated_risk = 0.0f;
    int64_t consecutive_reuse = 0;
    bool last_reuse = false;

    // precomputed propagation-aware weights, size = infer_steps
    std::vector<float> prop_weight_schedule;
  };

  struct JointDecision {
    bool ready = false;
    bool reuse_both = false;

    float ec = std::numeric_limits<float>::quiet_NaN();
    float eu = std::numeric_limits<float>::quiet_NaN();

    float dhat = std::numeric_limits<float>::quiet_NaN();
    float ghat = 1.0f;
    float rhat = std::numeric_limits<float>::quiet_NaN();
  };

 private:
  // config
  int64_t probe_depth_ = 2;
  float tau_ = 0.24f;
  float true_cfg_scale_ = 3.0f;

  bool use_prop_weight_ = true;
  float prop_a_ = 0.4806166f;
  float prop_alpha_ = 0.4782565f;
  float prop_b_ = 0.0641170f;

  ProxyErrorType proxy_error_type_ = ProxyErrorType::DeltaY;

  std::string rho_table_path_;
  std::string model_name_;

  // runtime state
  bool use_cache_ = false;
  bool joint_decision_ready_ = false;
  bool force_full_this_step_ = false;

  BranchLocalState local_;
  JointState joint_;
  JointDecision current_decision_;

  // rho_table_at[a, t], stored on CPU
  torch::Tensor rho_table_at_cpu_;

 private:
  void reset_all_state_();
  void reset_step_state_(int64_t step_id);
  torch::Tensor apply_prev_hidden_states_residual_(
      const torch::Tensor& hidden_states) const;

  std::pair<torch::Tensor, torch::Tensor> apply_prev_residual_pair_(
      const torch::Tensor& original_hidden_states,
      const torch::Tensor& original_encoder_hidden_states) const;
  float compute_rel_l1_(const torch::Tensor& curr,
                        const torch::Tensor& prev,
                        float eps = kEps) const;

  float compute_branch_error_(float dx, float dy) const;

  bool local_probe_available_() const;
  bool local_history_available_() const;

  void prepare_probe_at_block_(const CacheBlockIn& blockin);
  bool cfg_parallel_enabled_() const;
  bool is_cond_rank_() const;
  bool is_uncond_rank_() const;

  std::pair<float, float> exchange_branch_errors_(float local_err) const;

  float lookup_rho_(int64_t anchor_step, int64_t step_id) const;
  float build_prop_weight_(int64_t step_i) const;
  void build_prop_weight_schedule_();

  JointDecision make_joint_decision_(int64_t step_id);
  void update_local_history_after_full_();
  void update_joint_state_after_full_();
  void update_joint_state_after_reuse_();
  void validate_config_() const;
  void load_rho_table_();
};

}  // namespace xllm