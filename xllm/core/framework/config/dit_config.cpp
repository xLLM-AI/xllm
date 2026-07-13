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

#include "core/framework/config/dit_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_int32(max_requests_per_batch, 1, "Max number of request per batch.");

DEFINE_string(dit_cache_policy,
              "TaylorSeer",
              "The policy of dit cache(e.g. None, FBCache, TaylorSeer, "
              "FBCacheTaylorSeer, ResidualCache).");

DEFINE_int64(dit_cache_warmup_steps, 0, "The number of warmup steps.");

DEFINE_int64(dit_cache_n_derivatives,
             3,
             "The number of derivatives to use in TaylorSeer.");

DEFINE_int64(dit_cache_skip_interval_steps,
             3,
             "The interval steps to skip for derivative calculation.");

DEFINE_double(dit_cache_residual_diff_threshold,
              0.09,
              "The residual difference threshold for cache reuse.");

DEFINE_int64(dit_cache_start_steps,
             5,
             "The number of steps to skip at the start");

DEFINE_int64(dit_cache_end_steps, 5, "The number of steps to skip at the end.");

DEFINE_int64(dit_cache_start_blocks,
             5,
             "The number of blocks to skip at the start.");

DEFINE_int64(dit_cache_end_blocks,
             5,
             "The number of blocks to skip at the end.");

DEFINE_bool(dit_sp_communication_overlap,
            true,
            "Communication & Computation overlap for sequence parallel");

DEFINE_bool(dit_debug_print,
            false,
            "whether print the debug info for dit models");

DEFINE_bool(dit_laser_attention_enabled,
            false,
            "whether to use the laser attention kernel (MindIE-SD, tuned for "
            "Wan2.2) in place of npu_fusion_attention for DiT attention.");

DEFINE_int64(dit_generation_image_area_max,
             0,
             "Maximum allowed image area (width * height) for image generation "
             "requests. If set to 0, there is no limit.");
// --- dit vae tiling ---

DEFINE_bool(
    dit_enable_vae_tiling,
    false,
    "whether enable vae tiling, currently only support qwen-image-edit-plus");

DEFINE_int64(
    dit_vae_image_size,
    1048576,
    "Qwen Image Edit Plus VAE image size used to calculate dimensions.");

DEFINE_bool(dit_sparse_attention_enabled,
            false,
            "Enable block-wise sparse attention / RainFusion for WAN.");

DEFINE_double(dit_sparse_attention_sparsity,
              0.5,
              "Sparse attention sparsity ratio in [0.0, 1.0). 0.0 = dense "
              "attention, 0.5 = drop 50 percent blocks.");

DEFINE_int64(
    dit_sparse_attention_pool_size,
    128,
    "Sparse attention pooling window size for block-wise mask generation.");

DEFINE_int64(dit_sparse_attention_sparse_start_step,
             0,
             "Sparse attention step index to start sparse attention. "
             "Steps before this use dense attention.");

DEFINE_string(
    dit_sparse_attention_version,
    "rain_fusion",
    "Sparse attention version: 'rain_fusion' (frame-pairing + "
    "aclnnRainFusionAttention) or 'sparse_attention' (block-decompose + "
    "aclnnBlockSparseAttention).");

DEFINE_int64(dit_sparse_attention_mask_refresh_steps,
             1,
             "Sparse attention: recompute block sparse mask every N diffusion "
             "steps. 1 = every step (default), higher = reuse mask longer.");

namespace xllm {

void DiTConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(max_requests_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_policy);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_warmup_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_n_derivatives);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_skip_interval_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_residual_diff_threshold);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_start_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_end_steps);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_start_blocks);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_cache_end_blocks);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sp_communication_overlap);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_debug_print);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_laser_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_generation_image_area_max);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_vae_image_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_enable_vae_tiling);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_sparsity);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_pool_size);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_sparse_start_step);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_version);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(dit_sparse_attention_mask_refresh_steps);
}

void DiTConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(max_requests_per_batch);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_policy);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_warmup_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_n_derivatives);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_skip_interval_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_residual_diff_threshold);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_start_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_end_steps);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_start_blocks);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_cache_end_blocks);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sp_communication_overlap);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_debug_print);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_laser_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_generation_image_area_max);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_vae_image_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_enable_vae_tiling);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_enabled);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_sparsity);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_pool_size);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_sparse_start_step);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_version);
  XLLM_CONFIG_ASSIGN_FROM_JSON(dit_sparse_attention_mask_refresh_steps);
}

void DiTConfig::append_config_json(nlohmann::ordered_json& config_json) const {
  const DiTConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, max_requests_per_batch);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_policy);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_warmup_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_n_derivatives);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_skip_interval_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_residual_diff_threshold);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_start_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_end_steps);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_start_blocks);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_cache_end_blocks);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sp_communication_overlap);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_debug_print);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_laser_attention_enabled);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_generation_image_area_max);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_vae_image_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_enable_vae_tiling);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_enabled);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_sparsity);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_pool_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_sparse_start_step);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_version);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, dit_sparse_attention_mask_refresh_steps);
}

DiTConfig& DiTConfig::get_instance() {
  static DiTConfig config;
  return config;
}

void DiTConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
