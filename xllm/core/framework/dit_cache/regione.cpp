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

#include "regione.h"

#include <glog/logging.h>
#include <torch/nn/functional/pooling.h>

#include <algorithm>

namespace xllm {
namespace {
bool tensor_cache_ready(const std::vector<torch::Tensor>& cache,
                        int64_t block_id) {
  return block_id >= 0 && block_id < static_cast<int64_t>(cache.size()) &&
         cache[block_id].defined();
}

torch::Tensor regione_to_cpu_cache(const torch::Tensor& tensor) {
  auto src = tensor.detach().contiguous();
  auto cpu_options = src.options().device(torch::kCPU).pinned_memory(true);
  auto cpu_tensor = torch::empty(src.sizes(), cpu_options);
  cpu_tensor.copy_(src, /*non_blocking=*/false);
  return cpu_tensor;
}
}  // namespace

void RegionECache::init(const DiTCacheConfig& cfg) {
  config_ = cfg;
  regione_enabled_ = cfg.selected_policy == PolicyType::RegionE;
  regione_velocity_cache_ = torch::Tensor();
  regione_current_block_ = -1;
  regione_current_use_cfg_ = false;
  regione_current_step_ = 0;
  regione_infer_steps_ = 0;
  regione_num_blocks_ = 0;
  regione_partial_mode_ = false;
  regione_target_seq_len_ = 0;
  regione_grid_h_ = 0;
  regione_grid_w_ = 0;
  regione_image_seq_len_ = 0;
  regione_sp_rank_ = 0;
  regione_sp_size_ = 1;
  regione_local_start_ = 0;
  regione_local_end_ = 0;
  regione_condition_latents_ = torch::Tensor();
  regione_edited_ids_ = torch::Tensor();
  regione_unedited_ids_ = torch::Tensor();
  regione_local_edited_global_ids_ = torch::Tensor();
  regione_local_edited_cache_ids_ = torch::Tensor();
  regione_local_image_global_ids_ = torch::Tensor();
  regione_clear_all_prefetch_slots();
}

void RegionECache::set_num_blocks(int64_t num_blocks) {
  regione_num_blocks_ = num_blocks;
  ensure_regione_kv_size(num_blocks);
}

bool RegionECache::regione_is_refresh_step(int64_t step) const {
  for (const auto refresh_step : config_.regione.refresh_steps) {
    const auto refresh_index =
        refresh_step > 0 ? refresh_step - 1 : refresh_step;
    if (step == refresh_index) return true;
  }
  return false;
}

bool RegionECache::regione_is_tail_step(int64_t step) const {
  return config_.regione.tail_steps > 0 && regione_infer_steps_ > 0 &&
         step >= regione_infer_steps_ - config_.regione.tail_steps;
}

bool RegionECache::regione_should_run_full_step(int64_t step) const {
  if (!regione_enabled_) return true;
  if (!regione_has_regions()) return true;
  if (step < config_.regione.warmup_steps) return true;
  if (regione_is_tail_step(step)) return true;
  return regione_is_refresh_step(step);
}

bool RegionECache::regione_should_compute_velocity(int64_t step,
                                                   double timestep,
                                                   double prev_timestep) {
  if (!regione_enabled_) {
    regione_avd_ratio_ = 1.0;
    return true;
  }
  // STS / SMS / forced refresh: always run DiT and reset AVD accumulator.
  // Original inplace.py also disables AVD at step == warmup (partial DiT still
  // runs); keep that separate from regione_should_run_full_step so ARP→partial
  // transition at warmup is unchanged.
  if (regione_should_run_full_step(step) ||
      step <= config_.regione.warmup_steps) {
    regione_avd_accumulate_ = 1.0;
    regione_avd_ratio_ = 1.0;
    return true;
  }

  // Diffusers 28-step RegionE transition gamma (inplace.py), 27 values for
  // transitions between 28 steps. Linearly upsample/downsample onto the
  // actual (infer_steps - 1) transitions.
  static constexpr double kRegionEGammaRef[] = {
      1.0186, 1.0241, 1.0236, 1.0205, 1.0298, 1.0221, 1.0248, 1.0246, 1.0269,
      1.0275, 1.0323, 1.0311, 1.0298, 1.0353, 1.0343, 1.0397, 1.0387, 1.0393,
      1.0404, 1.0458, 1.0507, 1.0418, 1.0518, 1.0426, 1.0311, 1.0068, 0.7628};
  static constexpr int64_t kGammaRefLen = static_cast<int64_t>(
      sizeof(kRegionEGammaRef) / sizeof(kRegionEGammaRef[0]));

  auto sample_gamma = [&](int64_t cur_step) -> double {
    const int64_t n_steps =
        regione_infer_steps_ > 1 ? regione_infer_steps_ : 28;
    const int64_t n_trans = std::max<int64_t>(1, n_steps - 1);
    const int64_t idx = std::max<int64_t>(0, cur_step - 1);
    const double pos = static_cast<double>(idx) *
                       static_cast<double>(kGammaRefLen - 1) /
                       static_cast<double>(std::max<int64_t>(1, n_trans - 1));
    const int64_t lo = std::min(static_cast<int64_t>(pos), kGammaRefLen - 1);
    const int64_t hi = std::min(lo + 1, kGammaRefLen - 1);
    const double frac = pos - static_cast<double>(lo);
    return kRegionEGammaRef[lo] * (1.0 - frac) + kRegionEGammaRef[hi] * frac;
  };

  if (!config_.regione.use_avd_gamma || step < 1) {
    const auto interval =
        std::max<int64_t>(1, config_.regione.skip_interval_steps);
    const bool compute =
        ((step - config_.regione.warmup_steps) % interval) == 0;
    regione_avd_ratio_ = 1.0;
    if (compute) regione_avd_accumulate_ = 1.0;
    return compute;
  }

  // AVDCache (paper Eq.7-9 / inplace.py), step-count agnostic via resampled γ:
  //   ratio = gamma(step) * (1 + (t - t_prev) / 1000)
  //   accumulate *= ratio; error = 1 - accumulate
  //   reuse velocity while error <= cache_threshold and ratio < 1
  const double gamma = sample_gamma(step);
  const double ratio = gamma * (1.0 + (timestep - prev_timestep) / 1000.0);
  regione_avd_ratio_ = ratio;

  if (ratio >= 1.0) {
    regione_avd_accumulate_ = 1.0;
    return true;  // recompute DiT
  }

  regione_avd_accumulate_ *= ratio;
  const double error = 1.0 - regione_avd_accumulate_;
  if (error > static_cast<double>(config_.regione.cache_threshold)) {
    regione_avd_accumulate_ = 1.0;
    return true;  // recompute DiT
  }
  return false;  // reuse velocity cache * ratio
}

bool RegionECache::regione_should_direct_unedited(int64_t step) const {
  if (!regione_enabled_ || !regione_has_regions()) return false;
  if (regione_is_tail_step(step)) return false;
  return step == config_.regione.warmup_steps - 1 ||
         regione_is_refresh_step(step);
}

int64_t RegionECache::regione_next_direct_step(int64_t step) const {
  int64_t tail_start = regione_infer_steps_;
  if (config_.regione.tail_steps > 0 && regione_infer_steps_ > 0) {
    tail_start =
        std::max<int64_t>(0, regione_infer_steps_ - config_.regione.tail_steps);
  }
  int64_t next_step = tail_start;
  for (const auto refresh_step : config_.regione.refresh_steps) {
    const auto refresh_index =
        refresh_step > 0 ? refresh_step - 1 : refresh_step;
    if (refresh_index > step && refresh_index < next_step)
      next_step = refresh_index;
  }
  if (next_step <= step) next_step = step + 1;
  if (regione_infer_steps_ > 0)
    next_step = std::min<int64_t>(next_step, regione_infer_steps_);
  return next_step;
}

void RegionECache::regione_prepare_inference(
    const torch::Tensor& latents,
    const torch::Tensor& condition_latents,
    int64_t grid_h,
    int64_t grid_w,
    int64_t sp_rank,
    int64_t sp_size) {
  if (!regione_enabled_) return;
  regione_target_seq_len_ =
      latents.defined() && latents.dim() > 1 ? latents.size(1) : 0;
  const auto condition_seq_len =
      condition_latents.defined() && condition_latents.dim() > 1
          ? condition_latents.size(1)
          : 0;
  regione_image_seq_len_ = regione_target_seq_len_ + condition_seq_len;
  regione_grid_h_ = grid_h;
  regione_grid_w_ = grid_w;
  regione_sp_rank_ = sp_rank;
  regione_sp_size_ = std::max<int64_t>(1, sp_size);
  const auto shard = regione_sp_size_ > 0
                         ? regione_image_seq_len_ / regione_sp_size_
                         : regione_image_seq_len_;
  regione_local_start_ = regione_sp_rank_ * shard;
  regione_local_end_ = regione_sp_rank_ == regione_sp_size_ - 1
                           ? regione_image_seq_len_
                           : regione_local_start_ + shard;
  if (regione_image_seq_len_ > 0 && regione_local_end_ > regione_local_start_) {
    regione_local_image_global_ids_ =
        torch::arange(regione_local_start_,
                      regione_local_end_,
                      latents.options().dtype(torch::kLong));
  } else {
    regione_local_image_global_ids_ = torch::Tensor();
  }
  regione_condition_latents_ = condition_latents;
  regione_edited_ids_ = torch::Tensor();
  regione_unedited_ids_ = torch::Tensor();
  regione_velocity_cache_ = torch::Tensor();
  regione_avd_accumulate_ = 1.0;
  regione_avd_ratio_ = 1.0;
  regione_partial_mode_ = false;
  regione_local_edited_global_ids_ = torch::Tensor();
  regione_local_edited_cache_ids_ = torch::Tensor();
  for (auto& cache : regione_k_cache_cpu_) cache = torch::Tensor();
  for (auto& cache : regione_v_cache_cpu_) cache = torch::Tensor();
  for (auto& cache : regione_cond_k_cache_cpu_) cache = torch::Tensor();
  for (auto& cache : regione_cond_v_cache_cpu_) cache = torch::Tensor();
  regione_clear_all_prefetch_slots();
}

torch::Tensor RegionECache::regione_normalize_ids(
    const torch::Tensor& ids,
    const torch::Device& device) const {
  if (!ids.defined()) return torch::Tensor();
  auto out = ids;
  if (out.dim() > 1) out = out.reshape({-1});
  return out.to(device, torch::kLong, /*non_blocking=*/false, /*copy=*/false);
}

torch::Tensor RegionECache::regione_gather_ids(const torch::Tensor& tensor,
                                               const torch::Tensor& ids,
                                               int64_t dim) const {
  if (!tensor.defined() || !ids.defined()) return tensor;
  return tensor.index_select(dim, regione_normalize_ids(ids, tensor.device()));
}

torch::Tensor RegionECache::regione_scatter_ids(const torch::Tensor& values,
                                                const torch::Tensor& ids,
                                                const torch::Tensor& base,
                                                int64_t dim) const {
  if (!values.defined() || !ids.defined() || !base.defined()) return base;
  auto out = base.clone();
  out.index_copy_(dim, regione_normalize_ids(ids, base.device()), values);
  return out;
}

torch::Tensor RegionECache::regione_active_edited_ids() const {
  // Partial+SP feeds the full edited set into every rank, then relies on the
  // normal SP split/all-to-all path. Do not return per-shard local ids here —
  // a rank with 0 local edited tokens would produce an empty sequence and
  // break equal-length all_to_all_4D.
  return regione_edited_ids_;
}

torch::Tensor RegionECache::regione_kv_update_ids() const {
  // Full-mode image K/V is stored after SP QKV all-to-all, so the cache is
  // indexed by global sequence positions. Partial-step `key` is likewise
  // assembled across SP ranks into global edited_ids_ order — update with
  // those global indices rather than per-shard local cache ids.
  return regione_edited_ids_;
}

torch::Tensor RegionECache::regione_gather_edited(
    const torch::Tensor& tensor) const {
  return regione_gather_ids(tensor, regione_active_edited_ids(), 1);
}

torch::Tensor RegionECache::regione_gather_unedited(
    const torch::Tensor& tensor) const {
  return regione_gather_ids(tensor, regione_unedited_ids_, 1);
}

torch::Tensor RegionECache::regione_scatter_edited(
    const torch::Tensor& edited,
    const torch::Tensor& base) const {
  return regione_scatter_ids(edited, regione_active_edited_ids(), base, 1);
}

torch::Tensor RegionECache::regione_scatter_unedited(
    const torch::Tensor& unedited,
    const torch::Tensor& base) const {
  return regione_scatter_ids(unedited, regione_unedited_ids_, base, 1);
}

torch::Tensor RegionECache::regione_gather_query_rope(
    const torch::Tensor& image_rope) const {
  // SP attention (non-CMO) runs QKV all-to-all before RoPE, so img_query holds
  // every rank's edited tokens concatenated in contiguous-shard order. With
  // sorted edited_ids_ that order matches the global edited set — gather by
  // global ids, not the per-rank local edited subset.
  auto ids = regione_edited_ids_;
  if (!image_rope.defined() || !ids.defined()) return image_rope;
  if (image_rope.size(0) == ids.numel()) return image_rope;
  return regione_gather_ids(image_rope, ids, 0);
}

torch::Tensor RegionECache::regione_gather_key_rope(
    const torch::Tensor& image_rope,
    int64_t key_len) const {
  if (!image_rope.defined() || !regione_is_partial_sp_mode()) return image_rope;
  // Patched image K is the full-sequence cache (post all-to-all store). Prefer
  // a length-matched rope; only fall back to per-shard gather when the cache
  // is still local-shard sized.
  if (image_rope.size(0) == key_len) return image_rope;
  if (!regione_local_image_global_ids_.defined() ||
      regione_local_image_global_ids_.numel() != key_len) {
    return image_rope;
  }
  return regione_gather_ids(image_rope, regione_local_image_global_ids_, 0);
}

torch::Tensor RegionECache::regione_local_update_mask(
    const torch::Tensor& base) const {
  if (!base.defined()) return torch::Tensor();
  auto mask = torch::zeros({base.size(0), base.size(1), 1}, base.options());
  // SP partial DiT runs the full edited set on every rank; when reducing
  // latent updates, each rank must only own its image shard's edited tokens
  // so reduce(SUM) does not double-count.
  torch::Tensor ids;
  if (regione_sp_size_ > 1 && regione_local_edited_global_ids_.defined()) {
    ids = regione_local_edited_global_ids_;
  } else {
    ids = regione_edited_ids_;
  }
  if (ids.defined() && ids.numel() > 0) {
    auto ones = torch::ones({base.size(0), ids.numel(), 1}, base.options());
    mask.index_copy_(1, regione_normalize_ids(ids, base.device()), ones);
  }
  return mask;
}

void RegionECache::regione_select_regions(const torch::Tensor& sample,
                                          const torch::Tensor& model_output,
                                          const torch::Tensor& sigmas,
                                          int64_t step) {
  if (!regione_enabled_ || regione_has_regions()) return;
  if (!sample.defined() || !model_output.defined() ||
      !regione_condition_latents_.defined())
    return;
  if (sample.dim() != 3 || sample.size(0) != 1) {
    regione_edited_ids_ =
        torch::arange(sample.size(1), sample.options().dtype(torch::kLong));
    regione_unedited_ids_ =
        torch::empty({0}, sample.options().dtype(torch::kLong));
    regione_update_local_ids();
    return;
  }
  auto condition = regione_condition_latents_;
  if (condition.dim() != 3 || condition.size(1) < sample.size(1)) {
    regione_edited_ids_ =
        torch::arange(sample.size(1), sample.options().dtype(torch::kLong));
    regione_unedited_ids_ =
        torch::empty({0}, sample.options().dtype(torch::kLong));
    regione_update_local_ids();
    return;
  }
  condition = condition.slice(1, 0, sample.size(1)).to(sample.dtype());
  auto sigma = sigmas.index({step}).to(sample.device()).to(sample.dtype());
  auto sigma_final = sigmas.index({-1}).to(sample.device()).to(sample.dtype());
  auto estimate = sample + (sigma_final - sigma) * model_output;
  auto estimate_norm =
      estimate /
      torch::sqrt(torch::sum(estimate * estimate, -1, true)).clamp_min(1e-6);
  auto condition_norm =
      condition /
      torch::sqrt(torch::sum(condition * condition, -1, true)).clamp_min(1e-6);
  auto similarity = torch::sum(estimate_norm * condition_norm, -1);
  auto selected_mask = similarity <= config_.regione.region_threshold;
  if (config_.regione.erosion_dilation && regione_grid_h_ > 0 &&
      regione_grid_w_ > 0 &&
      regione_grid_h_ * regione_grid_w_ == sample.size(1)) {
    auto mask2d = selected_mask.to(torch::kFloat)
                      .view({1, 1, regione_grid_h_, regione_grid_w_});
    auto vertical_pool_opts =
        torch::nn::functional::MaxPool2dFuncOptions({3, 1}).stride(1).padding(
            {1, 0});
    auto horizontal_pool_opts =
        torch::nn::functional::MaxPool2dFuncOptions({1, 3}).stride(1).padding(
            {0, 1});
    auto eroded_vertical =
        -torch::nn::functional::max_pool2d(-mask2d, vertical_pool_opts);
    auto eroded_horizontal =
        -torch::nn::functional::max_pool2d(-mask2d, horizontal_pool_opts);
    auto eroded = eroded_vertical * eroded_horizontal;
    auto dilation_pool_opts =
        torch::nn::functional::MaxPool2dFuncOptions({5, 5}).stride(1).padding(
            2);
    auto dilated =
        torch::nn::functional::max_pool2d(eroded, dilation_pool_opts);
    selected_mask = dilated.view({1, -1}) > 0.5;
  }
  auto edited = torch::nonzero(selected_mask[0])
                    .reshape({-1})
                    .to(sample.device(), torch::kLong);
  if (edited.numel() == 0) {
    edited = std::get<1>(similarity[0].min(0, false))
                 .reshape({1})
                 .to(sample.device(), torch::kLong);
  }
  auto unedited_mask =
      torch::ones({sample.size(1)}, sample.options().dtype(torch::kBool));
  unedited_mask.index_fill_(0, edited, false);
  auto unedited = torch::nonzero(unedited_mask)
                      .reshape({-1})
                      .to(sample.device(), torch::kLong);
  regione_edited_ids_ = edited;
  regione_unedited_ids_ = unedited;
  regione_update_local_ids();
}

void RegionECache::regione_update_local_ids() {
  regione_local_edited_global_ids_ = torch::Tensor();
  regione_local_edited_cache_ids_ = torch::Tensor();
  if (!regione_edited_ids_.defined() || regione_sp_size_ <= 1) return;
  auto ids =
      regione_normalize_ids(regione_edited_ids_, regione_edited_ids_.device());
  auto mask = (ids >= regione_local_start_) & (ids < regione_local_end_);
  auto local_global = ids.index({mask});
  regione_local_edited_global_ids_ = local_global;
  regione_local_edited_cache_ids_ = local_global - regione_local_start_;
}

void RegionECache::regione_update_velocity_cache(const torch::Tensor& value) {
  if (regione_enabled_ && value.defined()) regione_velocity_cache_ = value;
}

torch::Tensor RegionECache::regione_velocity_cache() const {
  return regione_velocity_cache_;
}

RegionEStepPlan RegionECache::begin_step(int64_t step,
                                         double timestep,
                                         double prev_timestep) {
  RegionEStepPlan plan;
  plan.enabled = regione_enabled_;
  if (!regione_enabled_) return plan;

  regione_set_current_step(step);
  plan.full_step = regione_should_run_full_step(step);
  plan.partial_step = regione_has_regions() && !plan.full_step;
  regione_set_partial_mode(plan.partial_step);
  plan.use_velocity_cache =
      !regione_should_compute_velocity(step, timestep, prev_timestep) &&
      regione_velocity_cache_.defined();
  plan.run_partition = !regione_has_regions() &&
                       step == regione_warmup_steps() - 1 &&
                       !plan.use_velocity_cache;
  plan.direct_unedited = regione_should_direct_unedited(step);
  return plan;
}

RegionEStepInput RegionECache::prepare_step_input(
    const torch::Tensor& latents,
    const torch::Tensor& condition_latents,
    const std::vector<std::vector<int64_t>>& main_shape,
    const RegionEStepPlan& plan) const {
  RegionEStepInput input;
  input.step_latents =
      plan.partial_step ? regione_gather_edited(latents) : latents;
  input.latent_model_input = input.step_latents;
  if (!plan.partial_step && condition_latents.defined()) {
    input.latent_model_input = torch::cat({latents, condition_latents}, 1);
  }
  input.main_shape = main_shape;
  if (plan.partial_step) {
    input.main_shape = {{1, input.step_latents.size(1), 1}};
  }
  input.use_cached_velocity = plan.use_velocity_cache;
  if (plan.use_velocity_cache) {
    // inplace.py: noise_pred = cache * ratio
    input.cached_velocity = regione_velocity_cache_ * regione_avd_ratio_;
    if (plan.partial_step && input.cached_velocity.defined() &&
        input.cached_velocity.size(1) != input.step_latents.size(1)) {
      input.cached_velocity = regione_gather_edited(input.cached_velocity);
    }
  }
  return input;
}

void RegionECache::observe_velocity(const torch::Tensor& latents,
                                    const torch::Tensor& noise_pred,
                                    const torch::Tensor& sigmas,
                                    int64_t step,
                                    const RegionEStepPlan& plan) {
  if (!regione_enabled_ || plan.use_velocity_cache) return;
  if (plan.run_partition) {
    regione_select_regions(latents, noise_pred, sigmas, step);
  }
  regione_update_velocity_cache(noise_pred);
}

torch::Tensor RegionECache::apply_direct_unedited(
    const torch::Tensor& prev_latents,
    const torch::Tensor& latents,
    const torch::Tensor& noise_pred,
    const torch::Tensor& sigmas,
    int64_t step) const {
  auto sigma = sigmas.index({step}).to(latents.device()).to(latents.dtype());
  auto next_direct_step = regione_next_direct_step(step);
  auto sigma_direct =
      sigmas.index({next_direct_step}).to(latents.device()).to(latents.dtype());
  auto unedited_direct =
      regione_gather_unedited(latents) +
      (sigma_direct - sigma) * regione_gather_unedited(noise_pred);
  return regione_scatter_unedited(unedited_direct, prev_latents);
}

void RegionECache::regione_prefetch_img_kv(int64_t block_id,
                                           bool use_cfg,
                                           const torch::Tensor& reference) {
  if (!reference.defined()) return;
  regione_prefetch_img_kv(
      block_id, use_cfg, reference.device(), reference.scalar_type());
}

void RegionECache::regione_set_current_block(int64_t block_id,
                                             bool use_cfg,
                                             const torch::Tensor& reference) {
  regione_current_block_ = block_id;
  regione_current_use_cfg_ = use_cfg;
  if (reference.defined()) {
    regione_prefetch_img_kv(
        block_id + 1, use_cfg, reference.device(), reference.scalar_type());
  }
}

void RegionECache::regione_set_current_step(int64_t step) {
  regione_current_step_ = step;
}

void RegionECache::regione_set_partial_mode(bool partial_mode) {
  regione_partial_mode_ = regione_enabled_ && partial_mode;
}

bool RegionECache::regione_is_partial_sp_mode() const {
  return regione_enabled_ && regione_partial_mode_ && regione_sp_size_ > 1;
}

bool RegionECache::regione_should_store_kv() const {
  return regione_enabled_ && !regione_partial_mode_;
}

bool RegionECache::regione_should_patch_kv() const {
  return regione_enabled_ && regione_partial_mode_;
}

void RegionECache::ensure_regione_kv_size(int64_t num_blocks) {
  if (num_blocks <= 0) return;
  regione_k_cache_cpu_.resize(num_blocks);
  regione_v_cache_cpu_.resize(num_blocks);
  regione_cond_k_cache_cpu_.resize(num_blocks);
  regione_cond_v_cache_cpu_.resize(num_blocks);
}

std::vector<RegionECache::RegionEPrefetchedKV>&
RegionECache::regione_prefetch_slots(bool use_cfg) {
  auto& slots =
      use_cfg ? regione_cond_prefetch_slots_ : regione_prefetch_slots_;
  if (slots.empty()) slots.resize(2);
  return slots;
}

void RegionECache::regione_clear_prefetch_slot(RegionEPrefetchedKV& slot) {
  slot.block_id = -1;
  slot.key = torch::Tensor();
  slot.value = torch::Tensor();
#if defined(USE_NPU)
  slot.ready_event.reset();
#endif
}

void RegionECache::regione_clear_prefetch_block(bool use_cfg,
                                                int64_t block_id) {
  for (auto& slot : regione_prefetch_slots(use_cfg)) {
    if (slot.block_id == block_id) regione_clear_prefetch_slot(slot);
  }
}

void RegionECache::regione_clear_all_prefetch_slots() {
  for (auto& slot : regione_prefetch_slots_) {
    regione_clear_prefetch_slot(slot);
  }
  for (auto& slot : regione_cond_prefetch_slots_) {
    regione_clear_prefetch_slot(slot);
  }
}

void RegionECache::regione_prefetch_img_kv(int64_t block_id,
                                           bool use_cfg,
                                           const torch::Device& device,
                                           c10::ScalarType dtype) {
  if (!regione_enabled_ || !regione_partial_mode_ || block_id < 0 ||
      block_id >= regione_num_blocks_) {
    return;
  }
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  if (!tensor_cache_ready(k_cache, block_id) ||
      !tensor_cache_ready(v_cache, block_id)) {
    return;
  }

  auto& slots = regione_prefetch_slots(use_cfg);
  for (const auto& slot : slots) {
    if (slot.block_id == block_id && slot.key.defined() &&
        slot.value.defined() && slot.key.device() == device &&
        slot.key.scalar_type() == dtype) {
      return;
    }
  }

  RegionEPrefetchedKV* target = nullptr;
  for (auto& slot : slots) {
    if (slot.block_id < 0 || !slot.key.defined() || !slot.value.defined()) {
      target = &slot;
      break;
    }
  }
  if (target == nullptr) {
    target = &slots[static_cast<size_t>(block_id) % slots.size()];
  }
  regione_clear_prefetch_slot(*target);

#if defined(USE_NPU)
  if (device.is_privateuseone()) {
    auto stream = c10_npu::getStreamFromPool(false, device.index());
    {
      c10_npu::NPUStreamGuard stream_guard(stream);
      target->key = k_cache[block_id]
                        .to(device, dtype, /*non_blocking=*/true, /*copy=*/true)
                        .contiguous();
      target->value =
          v_cache[block_id]
              .to(device, dtype, /*non_blocking=*/true, /*copy=*/true)
              .contiguous();
      target->ready_event =
          std::make_shared<c10_npu::NPUEvent>(ACL_EVENT_EXTERNAL);
      target->ready_event->record(stream);
    }
    target->block_id = block_id;
    return;
  }
#endif

  target->key = k_cache[block_id]
                    .to(device, dtype, /*non_blocking=*/false, /*copy=*/true)
                    .contiguous();
  target->value = v_cache[block_id]
                      .to(device, dtype, /*non_blocking=*/false, /*copy=*/true)
                      .contiguous();
  target->block_id = block_id;
}

bool RegionECache::regione_take_prefetched_img_kv(int64_t block_id,
                                                  bool use_cfg,
                                                  const torch::Device& device,
                                                  c10::ScalarType dtype,
                                                  torch::Tensor* key,
                                                  torch::Tensor* value) {
  for (auto& slot : regione_prefetch_slots(use_cfg)) {
    if (slot.block_id != block_id || !slot.key.defined() ||
        !slot.value.defined() || slot.key.device() != device ||
        slot.key.scalar_type() != dtype) {
      continue;
    }
#if defined(USE_NPU)
    if (slot.ready_event != nullptr && device.is_privateuseone()) {
      auto current_stream = c10_npu::getCurrentNPUStream(device.index());
      slot.ready_event->block(current_stream);
    }
#endif
    *key = slot.key;
    *value = slot.value;
    regione_clear_prefetch_slot(slot);
    return true;
  }
  return false;
}

void RegionECache::regione_store_img_kv(int64_t block_id,
                                        bool use_cfg,
                                        const torch::Tensor& key,
                                        const torch::Tensor& value) {
  if (!regione_enabled_ || block_id < 0) return;
  ensure_regione_kv_size(std::max<int64_t>(regione_num_blocks_, block_id + 1));
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  k_cache[block_id] = regione_to_cpu_cache(key);
  v_cache[block_id] = regione_to_cpu_cache(value);
  regione_clear_prefetch_block(use_cfg, block_id);
}

std::pair<torch::Tensor, torch::Tensor> RegionECache::process_image_kv(
    const torch::Tensor& key,
    const torch::Tensor& value) {
  if (!regione_enabled_) return {key, value};
  if (regione_should_store_kv()) {
    regione_store_img_kv(
        regione_current_block_, regione_current_use_cfg_, key, value);
    return {key, value};
  }
  if (regione_should_patch_kv()) {
    return regione_patch_img_kv(
        regione_current_block_, regione_current_use_cfg_, key, value);
  }
  return {key, value};
}

std::pair<torch::Tensor, torch::Tensor> RegionECache::adjust_image_rope(
    const torch::Tensor& image_rope,
    int64_t key_len) const {
  auto query_rope = regione_partial_mode_
                        ? regione_gather_query_rope(image_rope)
                        : image_rope;
  auto key_rope = regione_gather_key_rope(image_rope, key_len);
  return {query_rope, key_rope};
}

std::pair<torch::Tensor, torch::Tensor> RegionECache::regione_patch_img_kv(
    int64_t block_id,
    bool use_cfg,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  if (!regione_enabled_ || block_id < 0) return {key, value};
  auto& k_cache = use_cfg ? regione_cond_k_cache_cpu_ : regione_k_cache_cpu_;
  auto& v_cache = use_cfg ? regione_cond_v_cache_cpu_ : regione_v_cache_cpu_;
  if (!tensor_cache_ready(k_cache, block_id) ||
      !tensor_cache_ready(v_cache, block_id)) {
    regione_store_img_kv(block_id, use_cfg, key, value);
    return {key, value};
  }
  torch::Tensor full_key;
  torch::Tensor full_value;
  const auto took_prefetched = regione_take_prefetched_img_kv(block_id,
                                                              use_cfg,
                                                              key.device(),
                                                              key.scalar_type(),
                                                              &full_key,
                                                              &full_value);
  if (!took_prefetched) {
    full_key = k_cache[block_id]
                   .to(key.device(),
                       key.scalar_type(),
                       /*non_blocking=*/true,
                       /*copy=*/true)
                   .contiguous();
    full_value = v_cache[block_id]
                     .to(value.device(),
                         value.scalar_type(),
                         /*non_blocking=*/true,
                         /*copy=*/true)
                     .contiguous();
  }
  if (full_key.sizes() == key.sizes()) {
    regione_store_img_kv(block_id, use_cfg, key, value);
    return {key, value};
  }
  auto update_ids = regione_kv_update_ids();
  if (update_ids.defined() && key.dim() >= 2 &&
      full_key.size(0) == key.size(0)) {
    CHECK_EQ(key.size(1), update_ids.numel())
        << "RegionE partial KV length must match edited_ids before scatter; "
           "key_len="
        << key.size(1) << " edited=" << update_ids.numel()
        << " full_key_len=" << full_key.size(1)
        << " (likely SP pad was applied after KV patch)";
    full_key = regione_scatter_ids(key, update_ids, full_key, 1);
    full_value = regione_scatter_ids(value, update_ids, full_value, 1);
  }
  return {full_key, full_value};
}

bool RegionECache::regione_profile_enabled() const {
  return regione_enabled_ && config_.regione.profile;
}

void RegionECache::regione_profile_reset_step(int64_t step,
                                              bool partial_step,
                                              bool full_step,
                                              bool velocity_cache,
                                              int64_t step_tokens,
                                              int64_t full_tokens) {
  regione_profile_step_ = step;
  regione_profile_partial_step_ = partial_step;
  regione_profile_full_step_ = full_step;
  regione_profile_velocity_cache_ = velocity_cache;
  regione_profile_step_tokens_ = step_tokens;
  regione_profile_full_tokens_ = full_tokens;
  regione_profile_kv_store_count_ = 0;
  regione_profile_prefetch_issue_count_ = 0;
  regione_profile_prefetch_hit_count_ = 0;
  regione_profile_prefetch_miss_count_ = 0;
  regione_profile_fallback_h2d_count_ = 0;
  regione_profile_patch_scatter_count_ = 0;
  regione_profile_kv_store_cpu_ms_ = 0.0;
  regione_profile_prefetch_issue_ms_ = 0.0;
  regione_profile_prefetch_wait_ms_ = 0.0;
  regione_profile_fallback_h2d_ms_ = 0.0;
  regione_profile_patch_scatter_ms_ = 0.0;
}

void RegionECache::regione_profile_log_step(double transformer_ms,
                                            double arp_ms,
                                            double scheduler_ms,
                                            double total_ms) const {
  if (!regione_profile_enabled()) return;
  LOG(INFO) << "[RegionEProfile] step=" << regione_profile_step_ << " mode="
            << (regione_profile_partial_step_
                    ? "partial"
                    : (regione_profile_full_step_ ? "full" : "reuse"))
            << " velocity_cache=" << regione_profile_velocity_cache_
            << " tokens=" << regione_profile_step_tokens_ << "/"
            << regione_profile_full_tokens_ << " total_ms=" << total_ms
            << " transformer_ms=" << transformer_ms
            << " scheduler_ms=" << scheduler_ms << " arp_ms=" << arp_ms
            << " kv_store_cpu_ms=" << regione_profile_kv_store_cpu_ms_
            << " kv_store_count=" << regione_profile_kv_store_count_
            << " kv_prefetch_issue_ms=" << regione_profile_prefetch_issue_ms_
            << " kv_prefetch_issue_count="
            << regione_profile_prefetch_issue_count_
            << " kv_prefetch_wait_ms=" << regione_profile_prefetch_wait_ms_
            << " kv_prefetch_hit_count=" << regione_profile_prefetch_hit_count_
            << " kv_prefetch_miss_count="
            << regione_profile_prefetch_miss_count_
            << " kv_fallback_h2d_ms=" << regione_profile_fallback_h2d_ms_
            << " kv_fallback_h2d_count=" << regione_profile_fallback_h2d_count_
            << " kv_patch_scatter_ms=" << regione_profile_patch_scatter_ms_
            << " kv_patch_scatter_count="
            << regione_profile_patch_scatter_count_;
}

void RegionECache::regione_profile_add_kv_store(double ms) {
  if (!regione_profile_enabled()) return;
  regione_profile_kv_store_cpu_ms_ += ms;
  ++regione_profile_kv_store_count_;
}

void RegionECache::regione_profile_add_prefetch_issue(double ms) {
  if (!regione_profile_enabled()) return;
  regione_profile_prefetch_issue_ms_ += ms;
  ++regione_profile_prefetch_issue_count_;
}

void RegionECache::regione_profile_add_prefetch_hit(double wait_ms) {
  if (!regione_profile_enabled()) return;
  regione_profile_prefetch_wait_ms_ += wait_ms;
  ++regione_profile_prefetch_hit_count_;
}

void RegionECache::regione_profile_add_prefetch_miss() {
  if (!regione_profile_enabled()) return;
  ++regione_profile_prefetch_miss_count_;
}

void RegionECache::regione_profile_add_fallback_h2d(double ms) {
  if (!regione_profile_enabled()) return;
  regione_profile_fallback_h2d_ms_ += ms;
  ++regione_profile_fallback_h2d_count_;
}

void RegionECache::regione_profile_add_patch_scatter(double ms) {
  if (!regione_profile_enabled()) return;
  regione_profile_patch_scatter_ms_ += ms;
  ++regione_profile_patch_scatter_count_;
}

}  // namespace xllm
