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

#include <memory>
#include <utility>
#include <vector>

#include "dit_cache_config.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

namespace xllm {

struct RegionEStepPlan {
  bool enabled = false;
  bool full_step = true;
  bool partial_step = false;
  bool use_velocity_cache = false;
  bool run_partition = false;
  bool direct_unedited = false;
};

struct RegionEStepInput {
  torch::Tensor step_latents;
  torch::Tensor latent_model_input;
  std::vector<std::vector<int64_t>> main_shape;
  torch::Tensor cached_velocity;
  bool use_cached_velocity = false;
};

class RegionECache {
 private:
  struct RegionEPrefetchedKV {
    int64_t block_id = -1;
    torch::Tensor key;
    torch::Tensor value;
#if defined(USE_NPU)
    std::shared_ptr<c10_npu::NPUEvent> ready_event;
#endif
  };

 public:
  RegionECache() = default;
  ~RegionECache() = default;

  RegionECache(const RegionECache&) = delete;
  RegionECache& operator=(const RegionECache&) = delete;
  RegionECache(RegionECache&&) = default;
  RegionECache& operator=(RegionECache&&) = default;

  void init(const DiTCacheConfig& cfg);
  void set_infer_steps(int64_t infer_steps) {
    regione_infer_steps_ = infer_steps;
  }
  void set_num_blocks(int64_t num_blocks);

  bool is_enabled() const { return regione_enabled_; }
  int64_t regione_warmup_steps() const { return config_.regione.warmup_steps; }
  bool regione_has_regions() const { return regione_edited_ids_.defined(); }
  bool regione_is_partial_mode() const { return regione_partial_mode_; }
  bool regione_is_partial_sp_mode() const;
  bool regione_should_compute_velocity(int64_t step) const;
  bool regione_should_run_full_step(int64_t step) const;
  bool regione_should_direct_unedited(int64_t step) const;
  int64_t regione_next_direct_step(int64_t step) const;
  void regione_prepare_inference(const torch::Tensor& latents,
                                 const torch::Tensor& condition_latents,
                                 int64_t grid_h,
                                 int64_t grid_w,
                                 int64_t sp_rank = 0,
                                 int64_t sp_size = 1);

  RegionEStepPlan begin_step(int64_t step);
  RegionEStepInput prepare_step_input(
      const torch::Tensor& latents,
      const torch::Tensor& condition_latents,
      const std::vector<std::vector<int64_t>>& main_shape,
      const RegionEStepPlan& plan) const;
  void observe_velocity(const torch::Tensor& latents,
                        const torch::Tensor& noise_pred,
                        const torch::Tensor& sigmas,
                        int64_t step,
                        const RegionEStepPlan& plan);
  torch::Tensor apply_direct_unedited(const torch::Tensor& prev_latents,
                                      const torch::Tensor& latents,
                                      const torch::Tensor& noise_pred,
                                      const torch::Tensor& sigmas,
                                      int64_t step) const;

  torch::Tensor regione_gather_edited(const torch::Tensor& tensor) const;
  torch::Tensor regione_gather_unedited(const torch::Tensor& tensor) const;
  torch::Tensor regione_scatter_edited(const torch::Tensor& edited,
                                       const torch::Tensor& base) const;
  torch::Tensor regione_scatter_unedited(const torch::Tensor& unedited,
                                         const torch::Tensor& base) const;
  torch::Tensor regione_local_update_mask(const torch::Tensor& base) const;

  void regione_prefetch_img_kv(int64_t block_id,
                               bool use_cfg,
                               const torch::Tensor& reference);
  void regione_set_current_block(
      int64_t block_id,
      bool use_cfg,
      const torch::Tensor& reference = torch::Tensor());
  std::pair<torch::Tensor, torch::Tensor> process_image_kv(
      const torch::Tensor& key,
      const torch::Tensor& value);
  std::pair<torch::Tensor, torch::Tensor> adjust_image_rope(
      const torch::Tensor& image_rope,
      int64_t key_len) const;

 private:
  void regione_select_regions(const torch::Tensor& sample,
                              const torch::Tensor& model_output,
                              const torch::Tensor& sigmas,
                              int64_t step);
  torch::Tensor regione_gather_query_rope(
      const torch::Tensor& image_rope) const;
  torch::Tensor regione_gather_key_rope(const torch::Tensor& image_rope,
                                        int64_t key_len) const;
  void regione_update_velocity_cache(const torch::Tensor& value);
  torch::Tensor regione_velocity_cache() const;
  void regione_set_current_step(int64_t step);
  void regione_set_partial_mode(bool partial_mode);
  int64_t regione_current_block() const { return regione_current_block_; }
  bool regione_current_use_cfg() const { return regione_current_use_cfg_; }
  bool regione_should_store_kv() const;
  bool regione_should_patch_kv() const;
  void regione_store_img_kv(int64_t block_id,
                            bool use_cfg,
                            const torch::Tensor& key,
                            const torch::Tensor& value);
  std::pair<torch::Tensor, torch::Tensor> regione_patch_img_kv(
      int64_t block_id,
      bool use_cfg,
      const torch::Tensor& key,
      const torch::Tensor& value);

 private:
  bool regione_is_refresh_step(int64_t step) const;
  bool regione_is_tail_step(int64_t step) const;
  torch::Tensor regione_normalize_ids(const torch::Tensor& ids,
                                      const torch::Device& device) const;
  torch::Tensor regione_active_edited_ids() const;
  torch::Tensor regione_kv_update_ids() const;
  void regione_update_local_ids();
  torch::Tensor regione_gather_ids(const torch::Tensor& tensor,
                                   const torch::Tensor& ids,
                                   int64_t dim) const;
  torch::Tensor regione_scatter_ids(const torch::Tensor& values,
                                    const torch::Tensor& ids,
                                    const torch::Tensor& base,
                                    int64_t dim) const;
  void ensure_regione_kv_size(int64_t num_blocks);
  std::vector<RegionEPrefetchedKV>& regione_prefetch_slots(bool use_cfg);
  void regione_clear_prefetch_slot(RegionEPrefetchedKV& slot);
  void regione_clear_prefetch_block(bool use_cfg, int64_t block_id);
  void regione_clear_all_prefetch_slots();
  void regione_prefetch_img_kv(int64_t block_id,
                               bool use_cfg,
                               const torch::Device& device,
                               c10::ScalarType dtype);
  bool regione_take_prefetched_img_kv(int64_t block_id,
                                      bool use_cfg,
                                      const torch::Device& device,
                                      c10::ScalarType dtype,
                                      torch::Tensor* key,
                                      torch::Tensor* value);

  DiTCacheConfig config_;
  bool regione_enabled_ = false;
  int64_t regione_infer_steps_ = 0;
  int64_t regione_num_blocks_ = 0;
  int64_t regione_current_step_ = 0;
  int64_t regione_current_block_ = -1;
  bool regione_current_use_cfg_ = false;
  bool regione_partial_mode_ = false;
  int64_t regione_target_seq_len_ = 0;
  int64_t regione_grid_h_ = 0;
  int64_t regione_grid_w_ = 0;
  int64_t regione_image_seq_len_ = 0;
  int64_t regione_sp_rank_ = 0;
  int64_t regione_sp_size_ = 1;
  int64_t regione_local_start_ = 0;
  int64_t regione_local_end_ = 0;
  torch::Tensor regione_condition_latents_;
  torch::Tensor regione_edited_ids_;
  torch::Tensor regione_unedited_ids_;
  torch::Tensor regione_local_edited_global_ids_;
  torch::Tensor regione_local_edited_cache_ids_;
  torch::Tensor regione_local_image_global_ids_;
  torch::Tensor regione_velocity_cache_;
  std::vector<torch::Tensor> regione_k_cache_cpu_;
  std::vector<torch::Tensor> regione_v_cache_cpu_;
  std::vector<torch::Tensor> regione_cond_k_cache_cpu_;
  std::vector<torch::Tensor> regione_cond_v_cache_cpu_;
  std::vector<RegionEPrefetchedKV> regione_prefetch_slots_;
  std::vector<RegionEPrefetchedKV> regione_cond_prefetch_slots_;
};

}  // namespace xllm
