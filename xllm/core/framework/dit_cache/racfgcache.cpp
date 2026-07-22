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

#include "racfgcache.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <utility>

#include "calibration/racfgcache_calibration_tables.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {

namespace {
inline bool is_nan_f(float x) { return std::isnan(x); }
}  // namespace

void RACFGCache::init(const DiTCacheConfig& cfg) {
  warmup_steps_ = cfg.racfgcache.warmup_steps;
  probe_depth_ = cfg.racfgcache.probe_depth;
  tau_ = cfg.racfgcache.tau;
  true_cfg_scale_ = cfg.racfgcache.true_cfg_scale;

  use_prop_weight_ = cfg.racfgcache.use_prop_weight;
  prop_a_ = cfg.racfgcache.prop_a;
  prop_alpha_ = cfg.racfgcache.prop_alpha;
  prop_b_ = cfg.racfgcache.prop_b;

  proxy_error_type_ =
      static_cast<ProxyErrorType>(cfg.racfgcache.proxy_error_type);

  rho_table_path_ = cfg.racfgcache.rho_table_path;
  model_name_ = cfg.racfgcache.model_name;
  validate_config_();
  reset_all_state_();
  build_prop_weight_schedule_();
}

void RACFGCache::set_runtime_context(const DiTCacheRuntimeContext& ctx) {
  DitCacheImpl::set_runtime_context(ctx);
  if (ctx.infer_steps > 0) {
    infer_steps_ = ctx.infer_steps;
  }
  if (ctx.num_blocks > 0) {
    num_blocks_ = ctx.num_blocks;
  }
  if (ctx.true_cfg_scale > 0.0f) {
    true_cfg_scale_ = ctx.true_cfg_scale;
  }
  build_prop_weight_schedule_();
  load_rho_table_();
}

bool RACFGCache::on_before_step(const CacheStepIn& stepin) {
  current_step_ = stepin.step_id;

  if (current_step_ == 0) {
    reset_all_state_();
    build_prop_weight_schedule_();
  } else {
    reset_step_state_(current_step_);
  }

  auto hidden_states = get_tensor_or_empty(stepin.tensors, "hidden_states");
  auto original_hidden_states =
      get_tensor_or_empty(stepin.tensors, "original_hidden_states");
  auto encoder_hidden_states =
      get_tensor_or_empty(stepin.tensors, "encoder_hidden_states");
  auto original_encoder_hidden_states =
      get_tensor_or_empty(stepin.tensors, "original_encoder_hidden_states");

  local_.base_hidden_states =
      original_hidden_states.defined() ? original_hidden_states : hidden_states;
  local_.base_encoder_hidden_states = original_encoder_hidden_states.defined()
                                          ? original_encoder_hidden_states
                                          : encoder_hidden_states;

  if (!cfg_parallel_enabled_()) {
    force_full_this_step_ = true;
  }

  if (current_step_ <= warmup_steps_) {
    force_full_this_step_ = true;
  }
  return false;
}

CacheStepOut RACFGCache::on_after_step(const CacheStepIn& stepin) {
  auto hidden_states = get_tensor_or_empty(stepin.tensors, "hidden_states");
  auto original_hidden_states =
      get_tensor_or_empty(stepin.tensors, "original_hidden_states");
  auto encoder_hidden_states =
      get_tensor_or_empty(stepin.tensors, "encoder_hidden_states");
  auto original_encoder_hidden_states =
      get_tensor_or_empty(stepin.tensors, "original_encoder_hidden_states");

  if (!use_cache_) {
    if (hidden_states.defined() && original_hidden_states.defined()) {
      local_.previous_residual =
          (hidden_states - original_hidden_states).detach().contiguous();
    }

    if (encoder_hidden_states.defined() &&
        original_encoder_hidden_states.defined()) {
      local_.previous_encoder_residual =
          (encoder_hidden_states - original_encoder_hidden_states)
              .detach()
              .contiguous();
    }

    update_local_history_after_full_();
    update_joint_state_after_full_();
  } else {
    update_joint_state_after_reuse_();
  }

  TensorMap out_map;
  if (hidden_states.defined()) {
    out_map["hidden_states"] = hidden_states;
  }
  if (encoder_hidden_states.defined()) {
    out_map["encoder_hidden_states"] = encoder_hidden_states;
  }
  return CacheStepOut(out_map);
}

bool RACFGCache::on_before_block(const CacheBlockIn& blockin) {
  if (force_full_this_step_) {
    return false;
  }

  if (!joint_decision_ready_) {
    return false;
  }

  if (!use_cache_) {
    return false;
  }

  if (blockin.block_id < probe_depth_) {
    return false;
  }
  return true;
}

CacheBlockOut RACFGCache::on_after_block(const CacheBlockIn& blockin) {
  auto hidden_states = get_tensor_or_empty(blockin.tensors, "hidden_states");
  auto encoder_hidden_states =
      get_tensor_or_empty(blockin.tensors, "encoder_hidden_states");
  auto original_hidden_states =
      get_tensor_or_empty(blockin.tensors, "original_hidden_states");
  auto original_encoder_hidden_states =
      get_tensor_or_empty(blockin.tensors, "original_encoder_hidden_states");

  TensorMap out_map;
  out_map["hidden_states"] = hidden_states;
  if (encoder_hidden_states.defined()) {
    out_map["encoder_hidden_states"] = encoder_hidden_states;
  }

  if (force_full_this_step_) {
    return CacheBlockOut(out_map);
  }

  if (blockin.block_id != probe_depth_ - 1) {
    return CacheBlockOut(out_map);
  }

  prepare_probe_at_block_(blockin);

  if (!local_probe_available_() || !local_history_available_()) {
    current_decision_.ready = true;
    current_decision_.reuse_both = false;
    joint_decision_ready_ = true;
    use_cache_ = false;
    return CacheBlockOut(out_map);
  }

  current_decision_ = make_joint_decision_(current_step_);
  joint_decision_ready_ = true;
  use_cache_ = current_decision_.ready && current_decision_.reuse_both;

  if (!use_cache_) {
    return CacheBlockOut(out_map);
  }

  auto residual_applied = apply_prev_residual_pair_(
      original_hidden_states, original_encoder_hidden_states);

  TensorMap cached_out_map;
  cached_out_map["hidden_states"] = std::move(residual_applied.first);
  if (residual_applied.second.defined()) {
    cached_out_map["encoder_hidden_states"] =
        std::move(residual_applied.second);
  }

  return CacheBlockOut(cached_out_map);
}

void RACFGCache::reset_all_state_() {
  use_cache_ = false;
  joint_decision_ready_ = false;
  force_full_this_step_ = false;

  local_ = BranchLocalState{};
  joint_ = JointState{};
  current_decision_ = JointDecision{};

  buffers.clear();
}

void RACFGCache::reset_step_state_(int64_t step_id) {
  use_cache_ = false;
  joint_decision_ready_ = false;
  force_full_this_step_ = false;

  current_decision_ = JointDecision{};

  local_.base_hidden_states = torch::Tensor();
  local_.base_encoder_hidden_states = torch::Tensor();

  local_.current_probe_input = torch::Tensor();
  local_.current_probe_hidden = torch::Tensor();

  local_.current_dx = std::numeric_limits<float>::quiet_NaN();
  local_.current_dy = std::numeric_limits<float>::quiet_NaN();
  local_.current_branch_error = std::numeric_limits<float>::quiet_NaN();

  local_.probe_ready_this_step = false;
}

torch::Tensor RACFGCache::apply_prev_hidden_states_residual_(
    const torch::Tensor& hidden_states) const {
  if (!hidden_states.defined() || !local_.previous_residual.defined()) {
    return hidden_states;
  }
  return (hidden_states + local_.previous_residual).contiguous();
}

std::pair<torch::Tensor, torch::Tensor> RACFGCache::apply_prev_residual_pair_(
    const torch::Tensor& original_hidden_states,
    const torch::Tensor& original_encoder_hidden_states) const {
  torch::Tensor new_hidden = original_hidden_states;
  torch::Tensor new_encoder = original_encoder_hidden_states;

  if (original_hidden_states.defined() && local_.previous_residual.defined()) {
    new_hidden =
        (original_hidden_states + local_.previous_residual).contiguous();
  }

  if (original_encoder_hidden_states.defined() &&
      local_.previous_encoder_residual.defined()) {
    new_encoder =
        (original_encoder_hidden_states + local_.previous_encoder_residual)
            .contiguous();
  }

  return {new_hidden, new_encoder};
}

float RACFGCache::compute_rel_l1_(const torch::Tensor& curr,
                                  const torch::Tensor& prev,
                                  float eps) const {
  if (!curr.defined() || !prev.defined()) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  torch::Device dev = curr.device();
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

  auto sum_abs_diff = (curr - prev).abs().sum().to(torch::kFloat32);
  auto sum_abs_prev = prev.abs().sum().to(torch::kFloat32);
  auto count = torch::tensor({static_cast<float>(curr.numel())}, opts);

  if (runtime_ctx_.sp_enabled && runtime_ctx_.sp_world_size > 1 &&
      runtime_ctx_.sp_group != nullptr) {
    auto* sp_group = static_cast<ProcessGroup*>(runtime_ctx_.sp_group);
    CHECK(sp_group != nullptr) << "sp_group is null in RACFGCache";

    sum_abs_diff = xllm::parallel_state::reduce(sum_abs_diff, sp_group);
    sum_abs_prev = xllm::parallel_state::reduce(sum_abs_prev, sp_group);
    count = xllm::parallel_state::reduce(count, sp_group);
  }

  auto denom = sum_abs_prev / count;
  float denom_f = denom.item<float>();
  if (std::abs(denom_f) < eps) {
    return 0.0f;
  }

  auto mean_abs_diff = sum_abs_diff / count;
  auto rel = mean_abs_diff / (denom + eps);
  return rel.item<float>();
}

float RACFGCache::compute_branch_error_(float dx, float dy) const {
  if (is_nan_f(dy)) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  if (proxy_error_type_ == ProxyErrorType::DeltaMinus) {
    if (is_nan_f(dx)) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return std::abs(dy - dx);
  }

  // default: delta_y
  return dy;
}

bool RACFGCache::local_probe_available_() const {
  return local_.probe_ready_this_step && !is_nan_f(local_.current_branch_error);
}

bool RACFGCache::local_history_available_() const {
  return local_.previous_residual.defined() &&
         local_.first_probe_input_prev.defined() &&
         local_.probe_hidden_prev.defined();
}

void RACFGCache::prepare_probe_at_block_(const CacheBlockIn& blockin) {
  auto hidden_states = get_tensor_or_empty(blockin.tensors, "hidden_states");
  auto original_hidden_states =
      get_tensor_or_empty(blockin.tensors, "original_hidden_states");

  local_.current_probe_input =
      original_hidden_states.defined() ? original_hidden_states : hidden_states;
  local_.current_probe_hidden = hidden_states;

  local_.current_dx = compute_rel_l1_(local_.current_probe_input,
                                      local_.first_probe_input_prev);
  local_.current_dy =
      compute_rel_l1_(local_.current_probe_hidden, local_.probe_hidden_prev);

  local_.current_branch_error =
      compute_branch_error_(local_.current_dx, local_.current_dy);

  local_.probe_ready_this_step = true;
}

bool RACFGCache::cfg_parallel_enabled_() const {
  return runtime_ctx_.cfg_enabled && runtime_ctx_.cfg_world_size == 2 &&
         runtime_ctx_.cfg_group != nullptr;
}

bool RACFGCache::is_cond_rank_() const { return runtime_ctx_.cfg_rank == 0; }

bool RACFGCache::is_uncond_rank_() const { return runtime_ctx_.cfg_rank == 1; }

std::pair<float, float> RACFGCache::exchange_branch_errors_(
    float local_err) const {
  if (!cfg_parallel_enabled_()) {
    return {local_err, local_err};
  }

  auto* cfg_group = static_cast<ProcessGroup*>(runtime_ctx_.cfg_group);
  CHECK(cfg_group != nullptr) << "cfg_group is null in RACFGCache";

  torch::Device dev = local_.base_hidden_states.defined()
                          ? local_.base_hidden_states.device()
                          : torch::Device(torch::kCPU);

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  auto local_tensor = torch::tensor({local_err}, opts);

  auto gathered = xllm::parallel_state::gather(local_tensor, cfg_group, 0);
  auto gathered_cpu = gathered.to(torch::kCPU).contiguous().view({-1});

  CHECK_EQ(gathered_cpu.numel(), 2)
      << "RACFGCache expects cfg world size = 2, but got gathered numel = "
      << gathered_cpu.numel();

  // rank 0 = cond, rank 1 = uncond
  float ec = gathered_cpu[0].item<float>();
  float eu = gathered_cpu[1].item<float>();
  return {ec, eu};
}

float RACFGCache::lookup_rho_(int64_t anchor_step, int64_t step_id) const {
  if (!rho_table_at_cpu_.defined() || rho_table_at_cpu_.numel() == 0) {
    return 0.0f;
  }

  CHECK_EQ(rho_table_at_cpu_.dim(), 2)
      << "rho_table_at is expected to be a 2D tensor";

  int64_t a_max = rho_table_at_cpu_.size(0) - 1;
  int64_t t_max = rho_table_at_cpu_.size(1) - 1;

  int64_t a = std::max<int64_t>(0, std::min<int64_t>(anchor_step, a_max));
  int64_t t = std::max<int64_t>(0, std::min<int64_t>(step_id, t_max));

  return rho_table_at_cpu_.index({a, t}).item<float>();
}

float RACFGCache::build_prop_weight_(int64_t step_i) const {
  if (!use_prop_weight_) {
    return 1.0f;
  }

  if (!joint_.prop_weight_schedule.empty()) {
    int64_t idx = std::max<int64_t>(
        0,
        std::min<int64_t>(
            step_i,
            static_cast<int64_t>(joint_.prop_weight_schedule.size()) - 1));
    return joint_.prop_weight_schedule[idx];
  }

  int64_t T = infer_steps_ > 0 ? infer_steps_ : runtime_ctx_.infer_steps;
  if (T <= 1) {
    return 1.0f;
  }

  int64_t k = std::max<int64_t>(0, std::min<int64_t>(step_i, T - 1));
  double x = static_cast<double>(T - 1 - k) / static_cast<double>(T - 1);
  double raw_t = static_cast<double>(prop_a_) * std::pow(x, prop_alpha_) +
                 static_cast<double>(prop_b_);

  double raw_sum = 0.0;
  for (int64_t i = 0; i < T; ++i) {
    double xi = static_cast<double>(T - 1 - i) / static_cast<double>(T - 1);
    raw_sum += static_cast<double>(prop_a_) * std::pow(xi, prop_alpha_) +
               static_cast<double>(prop_b_);
  }
  double raw_mean = std::max(raw_sum / static_cast<double>(T), 1e-12);

  return static_cast<float>(raw_t / raw_mean);
}

void RACFGCache::build_prop_weight_schedule_() {
  joint_.prop_weight_schedule.clear();

  if (!use_prop_weight_) {
    return;
  }

  int64_t T = infer_steps_ > 0 ? infer_steps_ : runtime_ctx_.infer_steps;
  if (T <= 0) {
    return;
  }

  joint_.prop_weight_schedule.resize(T, 1.0f);

  if (T == 1) {
    joint_.prop_weight_schedule[0] = 1.0f;
    return;
  }

  std::vector<double> raws(T, 0.0);
  double raw_sum = 0.0;
  for (int64_t i = 0; i < T; ++i) {
    double x = static_cast<double>(T - 1 - i) / static_cast<double>(T - 1);
    double raw = static_cast<double>(prop_a_) * std::pow(x, prop_alpha_) +
                 static_cast<double>(prop_b_);
    raws[i] = raw;
    raw_sum += raw;
  }
  double raw_mean = std::max(raw_sum / static_cast<double>(T), 1e-12);

  for (int64_t i = 0; i < T; ++i) {
    joint_.prop_weight_schedule[i] = static_cast<float>(raws[i] / raw_mean);
  }
}

RACFGCache::JointDecision RACFGCache::make_joint_decision_(int64_t step_id) {
  JointDecision out;
  out.ready = true;

  if (force_full_this_step_) {
    out.reuse_both = false;
    return out;
  }

  if (!local_history_available_()) {
    out.reuse_both = false;
    return out;
  }

  auto local_err = local_.current_branch_error;
  if (is_nan_f(local_err)) {
    out.reuse_both = false;
    return out;
  }

  auto branch_errors = exchange_branch_errors_(local_err);
  out.ec = branch_errors.first;
  out.eu = branch_errors.second;

  if (is_nan_f(out.ec) || is_nan_f(out.eu)) {
    out.reuse_both = false;
    return out;
  }

  float s = true_cfg_scale_;
  float rho = lookup_rho_(joint_.anchor_step, step_id);

  float term_u = (1.0f - s) * (1.0f - s) * out.eu * out.eu;
  float term_c = s * s * out.ec * out.ec;
  float term_cross = 2.0f * s * (1.0f - s) * out.eu * out.ec * rho;

  float val_raw = term_u + term_c + term_cross;
  float val_clamped = std::max(val_raw, 0.0f);

  out.dhat = std::sqrt(val_clamped);
  out.ghat = build_prop_weight_(step_id);
  out.rhat = std::max(out.dhat * out.ghat, 0.0f);

  bool budget_ok = (joint_.accumulated_risk + out.rhat <= tau_);
  out.reuse_both = budget_ok;
  return out;
}

void RACFGCache::update_local_history_after_full_() {
  if (local_.current_probe_input.defined()) {
    local_.first_probe_input_prev =
        local_.current_probe_input.detach().contiguous();
  }

  if (local_.current_probe_hidden.defined()) {
    local_.probe_hidden_prev =
        local_.current_probe_hidden.detach().contiguous();
  }
}

void RACFGCache::update_joint_state_after_full_() {
  joint_.anchor_step = current_step_;
  joint_.accumulated_risk = 0.0f;
  joint_.consecutive_reuse = 0;
  joint_.last_reuse = false;
}

void RACFGCache::update_joint_state_after_reuse_() {
  if (!is_nan_f(current_decision_.rhat)) {
    joint_.accumulated_risk += current_decision_.rhat;
  }
  joint_.consecutive_reuse += 1;
  joint_.last_reuse = true;
}

void RACFGCache::validate_config_() const {
  CHECK_GE(warmup_steps_, 0) << "warmup_steps must be >= 0";
  CHECK_GE(probe_depth_, 1) << "probe_depth must be >= 1";
  CHECK_GT(tau_, 0.0f) << "tau must be > 0";
  CHECK_GE(true_cfg_scale_, 1.0f) << "true_cfg_scale must be >= 1.0";
  CHECK(proxy_error_type_ == ProxyErrorType::DeltaY ||
        proxy_error_type_ == ProxyErrorType::DeltaMinus)
      << "unsupported proxy_error_type";
}

void RACFGCache::load_rho_table_() {
  if (rho_table_path_.empty()) {
    rho_table_at_cpu_ = torch::Tensor();
    LOG(INFO) << "[RACFG][RHO] no rho key configured, fallback to rho=0";
    return;
  }

  RhoTableSpec spec;
  spec.model_name = model_name_;
  spec.cfg_scale = true_cfg_scale_;
  spec.infer_steps = infer_steps_ > 0 ? infer_steps_ : runtime_ctx_.infer_steps;
  LOG(INFO) << "[RACFG][RHO] loading rho table with spec: "
            << "model_name=" << spec.model_name
            << " cfg_scale=" << spec.cfg_scale
            << " infer_steps=" << spec.infer_steps;
  auto hardcoded = get_hardcoded_rho_table(spec);
  if (hardcoded.defined()) {
    CHECK_EQ(hardcoded.dim(), 2) << "hardcoded rho table must be 2D";
    rho_table_at_cpu_ =
        hardcoded.to(torch::kCPU).to(torch::kFloat32).contiguous();
    LOG(INFO) << "[RACFG][RHO] loaded hardcoded rho table with key="
              << rho_table_path_ << " shape=" << rho_table_at_cpu_.sizes();
    return;
  }

  // fallback: old file-based load
  torch::Tensor table;
  torch::load(table, rho_table_path_);
  CHECK(table.defined()) << "failed to load rho table from " << rho_table_path_;
  CHECK_EQ(table.dim(), 2) << "rho table must be a 2D tensor";

  rho_table_at_cpu_ = table.to(torch::kCPU).to(torch::kFloat32).contiguous();

  LOG(INFO) << "[RACFG][RHO] loaded file rho table from path="
            << rho_table_path_ << " shape=" << rho_table_at_cpu_.sizes();
}

}  // namespace xllm