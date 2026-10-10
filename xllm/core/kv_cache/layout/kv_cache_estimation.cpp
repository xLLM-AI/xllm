/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/kv_cache/layout/kv_cache_estimation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/kv_cache/block/block_utils.h"
#include "core/kv_cache/layout/deepseek_v4_cache_policy.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "framework/model/model_args.h"
#include "util/pretty_print.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {

constexpr int32_t kNzAlignment = 16;

int64_t kpool_tail_slot_size_bytes(int64_t tail_len, int64_t index_head_dim) {
  return 2 * static_cast<int64_t>(sizeof(uint16_t)) * tail_len * index_head_dim;
}

int64_t kv_cache_dtype_size(const std::string& kv_cache_dtype,
                            int64_t model_dtype_size) {
  if (kv_cache_dtype == "auto") {
    return model_dtype_size;
  }
  if (kv_cache_dtype == "int8") {
    return 1;
  }
  if (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2") {
    return 1;
  }
  return model_dtype_size;
}

int64_t kv_slot_size(const ModelArgs& model_args,
                     const KVCacheEstimateOptions& options,
                     int64_t cache_dtype_size) {
  if (model_args.enable_mla()) {
    if (util::enable_mla_packed_c8(options.kv_cache_dtype == "int8",
                                   model_args.model_type())) {
      return mla_packed_c8_row_bytes(model_args.kv_lora_rank(),
                                     model_args.qk_rope_head_dim());
    }
#if defined(USE_NPU)
    if ((model_args.model_type() == "deepseek_v3" ||
         model_args.model_type() == "deepseek_v3_mtp") &&
        options.enable_prefix_cache) {
      return cache_dtype_size *
             ((model_args.kv_lora_rank() + kNzAlignment - 1) / kNzAlignment +
              (model_args.qk_rope_head_dim() + kNzAlignment - 1) /
                  kNzAlignment) *
             kNzAlignment;
    }
#endif
    return cache_dtype_size *
           (model_args.kv_lora_rank() + model_args.qk_rope_head_dim());
  }

  return 2 * cache_dtype_size * model_args.head_dim() *
         options.n_local_kv_heads;
}

int64_t index_slot_size(const ModelArgs& model_args,
                        bool enable_indexer_cache_quantization,
                        int64_t dtype_size,
                        KPoolCacheLayout kpool_layout) {
  if (model_args.index_n_heads() <= 0) {
    return 0;
  }

  const int64_t index_n_head = 1;
  if (model_args.index_kpool_compress()) {
    CHECK(!enable_indexer_cache_quantization)
        << "KPool requires BF16 compressed index cache.";
    CHECK_GT(model_args.index_kpool(), 0) << "KPool requires index_kpool > 0.";
  }
  int64_t split_factor = 1;
  if (Platform::supports_dsa_indexer_cache_sharding() &&
      util::kv_split_size_effective() > 1) {
    split_factor = util::kv_split_size_effective();
  }
  if (enable_indexer_cache_quantization) {
    // int8 index cache: one byte per element, plus an independent per-token
    // fp32 scale (kept separate from the main-KV scale_slot_size path).
    return split_factor * (static_cast<int64_t>(sizeof(int8_t)) * index_n_head *
                               model_args.index_head_dim() +
                           static_cast<int64_t>(sizeof(float)));
  }
  if (kpool_layout == KPoolCacheLayout::COMPRESSED_WITH_TAIL &&
      model_args.index_kpool_compress()) {
    CHECK_EQ(model_args.index_head_dim() % model_args.index_kpool(), 0)
        << "KPool index head dim must be divisible by index_kpool.";
    return split_factor * dtype_size * index_n_head *
           model_args.index_head_dim() / model_args.index_kpool();
  }
  return split_factor * dtype_size * index_n_head *
         (model_args.index_kpool_compress()
              ? 2 * model_args.index_head_dim() + 1
              : model_args.index_head_dim());
}

int64_t scale_slot_size(const ModelArgs& model_args,
                        const KVCacheEstimateOptions& options) {
  if (options.kv_cache_dtype == "auto") {
    return 0;
  }
  if (model_args.enable_mla()) {
    // Packed main KV rows already include their per-tile scales.
    if (util::enable_mla_packed_c8(options.kv_cache_dtype == "int8",
                                   model_args.model_type())) {
      return 0;
    }
    return sizeof(float);
  }
  return 2 * sizeof(float) * options.n_local_kv_heads;
}

int64_t standard_full_cache_block_size_in_bytes(
    const KVCacheCapacity& kv_cache_cap) {
  const int64_t full_attention_layers =
      std::max<int64_t>(kv_cache_cap.num_full_attention_layers(), 1);
  const int64_t indexer_layers = kv_cache_cap.num_indexer_layers();
  CHECK_GE(indexer_layers, 0) << "num_indexer_layers must be non-negative";
  CHECK_LE(indexer_layers, full_attention_layers)
      << "num_indexer_layers cannot exceed full-attention layers";
  const int64_t logical_block_bytes =
      kv_cache_cap.block_size() *
      (full_attention_layers *
           (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size()) +
       indexer_layers * kv_cache_cap.index_slot_size());
  CHECK_GT(logical_block_bytes, 0) << "logical block bytes must be positive";
  return logical_block_bytes;
}

int64_t layerwise_split_block_count(const ModelArgs& model_args,
                                    int32_t layerwise_split_size,
                                    const KVCacheCapacity& kv_cache_cap,
                                    int64_t available_bytes,
                                    int64_t additional_block_bytes) {
  const int64_t num_layers = kv_cache_cap.n_layers();
  const std::vector<bool> indexer_layer_mask =
      resolve_indexer_cache_enabled_layers(model_args, num_layers);

  std::vector<int64_t> layer_bytes(static_cast<size_t>(num_layers), 0);
  bool any_indexer_layer = false;
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    if (!is_full_attention_layer(model_args, layer_id)) {
      continue;
    }
    const bool has_indexer =
        kv_cache_cap.index_slot_size() > 0 &&
        (indexer_layer_mask.empty() ||
         indexer_layer_mask[static_cast<size_t>(layer_id)]);
    any_indexer_layer = any_indexer_layer || has_indexer;
    const int64_t bytes =
        kv_cache_cap.block_size() *
        (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size() +
         (has_indexer ? kv_cache_cap.index_slot_size() : 0));
    layer_bytes[static_cast<size_t>(layer_id)] = bytes;
  }

  // Scratch matches an owned layer, including indexer when any layer has one.
  const int64_t scratch_bytes_per_block =
      kv_cache_cap.block_size() *
      (kv_cache_cap.slot_size() + kv_cache_cap.scale_slot_size() +
       (any_indexer_layer ? kv_cache_cap.index_slot_size() : 0));
  CHECK_GT(scratch_bytes_per_block, 0);

  int64_t common_block_count = std::numeric_limits<int64_t>::max();
  for (int32_t split_rank = 0; split_rank < layerwise_split_size;
       ++split_rank) {
    const LayerwiseSplitLayout layout(
        /*enabled=*/true, layerwise_split_size, split_rank);
    const std::vector<bool> layer_cache_owned =
        build_layer_cache_owned(model_args, layout, num_layers);
    int64_t owned_bytes = 0;
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      if (layer_cache_owned[static_cast<size_t>(layer_id)]) {
        owned_bytes += layer_bytes[static_cast<size_t>(layer_id)];
      }
    }
    if (owned_bytes == 0) {
      continue;
    }
    const int64_t per_block_bytes =
        owned_bytes + scratch_bytes_per_block + additional_block_bytes;
    common_block_count =
        std::min(common_block_count, available_bytes / per_block_bytes);
  }

  CHECK_NE(common_block_count, std::numeric_limits<int64_t>::max())
      << "No layerwise split rank owns a model layer.";
  CHECK_GT(common_block_count, 0) << "No memory for one layerwise split block.";
  return common_block_count;
}

bool enable_qwen3_5_spec_verify(const ModelArgs& model_args,
                                const KVCacheEstimateOptions& options) {
  return options.num_speculative_tokens > 0 && !options.is_draft_engine &&
         is_qwen3_5_target_model_type(model_args.model_type());
}

int64_t linear_slot_size(const ModelArgs& model_args,
                         const KVCacheEstimateOptions& options,
                         int64_t dtype_size) {
  if (model_args.linear_num_value_heads() <= 0) {
    return 0;
  }
  const int64_t num_speculative_tokens =
      enable_qwen3_5_spec_verify(model_args, options)
          ? options.num_speculative_tokens
          : 0;

  const int64_t head_k_dim = model_args.linear_key_head_dim();
  const int64_t head_v_dim = model_args.linear_value_head_dim();
  const int64_t ssm_dtype_size =
      resolve_ssm_dtype_size(model_args.mamba_ssm_dtype(), dtype_size);

  const int64_t linear_ssm_slot_size =
      ssm_dtype_size * options.n_local_linear_v_heads * head_k_dim * head_v_dim;
  const int64_t linear_conv_state_len =
      model_args.linear_conv_kernel_dim() - 1 + num_speculative_tokens;
  const int64_t linear_conv_slot_size =
      dtype_size *
      (head_k_dim * options.n_local_linear_k_heads * 2 +
       head_v_dim * options.n_local_linear_v_heads) *
      linear_conv_state_len;
  return linear_conv_slot_size +
         linear_ssm_slot_size * (num_speculative_tokens + 1);
}

int64_t max_linear_state_blocks(int64_t cache_size_in_bytes,
                                int64_t state_slot_bytes,
                                int64_t full_cache_block_size_in_bytes) {
  if (state_slot_bytes <= 0) {
    return kPaddingLinearStateBlocks;
  }

  CHECK_GT(cache_size_in_bytes, 0);
  CHECK_GT(full_cache_block_size_in_bytes, 0);

  int64_t max_linear_blocks = (cache_size_in_bytes - 1) / state_slot_bytes;
  const int64_t balanced_max_linear_blocks =
      (cache_size_in_bytes +
       kPaddingLinearStateBlocks * full_cache_block_size_in_bytes) /
      (state_slot_bytes + full_cache_block_size_in_bytes);
  max_linear_blocks = std::min(max_linear_blocks, balanced_max_linear_blocks);

  return std::max<int64_t>(max_linear_blocks, kPaddingLinearStateBlocks);
}

int64_t calculate_linear_state_blocks(int64_t cache_size_in_bytes,
                                      int64_t state_slot_bytes,
                                      int64_t full_cache_block_size_in_bytes,
                                      int64_t max_seqs_per_batch,
                                      int64_t max_concurrent_requests,
                                      int64_t max_linear_state_cache_slots,
                                      bool enable_prefix_cache) {
  CHECK_GE(max_linear_state_cache_slots, 0)
      << "max_linear_state_cache_slots must be greater than or equal to 0.";
  if (state_slot_bytes <= 0) {
    return kPaddingLinearStateBlocks;
  }
  const int64_t max_blocks = max_linear_state_blocks(
      cache_size_in_bytes, state_slot_bytes, full_cache_block_size_in_bytes);
  if (max_linear_state_cache_slots > 0) {
    const int64_t requested_blocks =
        max_linear_state_cache_slots + kPaddingLinearStateBlocks;
    CHECK_LE(requested_blocks, max_blocks)
        << "max_linear_state_cache_slots requires " << requested_blocks
        << " linear-state blocks, but only " << max_blocks
        << " fit in the configured KV cache budget.";
    return requested_blocks;
  }

  if (!enable_prefix_cache) {
    // Slots must cover every simultaneously running sequence, bounded by
    // both the scheduler batch limit and the service concurrency cap.
    int64_t running_seqs_upper_bound = max_seqs_per_batch;
    if (max_concurrent_requests > 0) {
      running_seqs_upper_bound =
          std::min<int64_t>(running_seqs_upper_bound, max_concurrent_requests);
    }
    const int64_t live_slot_blocks =
        running_seqs_upper_bound + kPaddingLinearStateBlocks;
    return std::max<int64_t>(std::min<int64_t>(live_slot_blocks, max_blocks),
                             kPaddingLinearStateBlocks);
  }

  // Auto-size: allocate ~47% of cache bytes to linear-state slots (ratio 0.9
  // means linear fraction = 0.9 / 1.9).
  constexpr double kLinearStateFullKvMemoryRatio = 0.9;
  const double linear_memory_fraction =
      kLinearStateFullKvMemoryRatio / (1.0 + kLinearStateFullKvMemoryRatio);
  const double linear_memory_bytes =
      static_cast<double>(cache_size_in_bytes) * linear_memory_fraction;
  const int64_t auto_blocks = std::max<int64_t>(
      static_cast<int64_t>(linear_memory_bytes / state_slot_bytes),
      kPaddingLinearStateBlocks);
  // Both bounds already sit at or above kPaddingLinearStateBlocks (auto_blocks
  // is floored above; max_blocks is floored in max_linear_state_blocks), so the
  // min of the two cannot drop below it.
  return std::min<int64_t>(auto_blocks, max_blocks);
}

int64_t dsv4_common_unit_bytes(const Dsv4KVCacheEstimateCost& cache_cost,
                               int64_t common_blocks_per_unit) {
  CHECK_GT(cache_cost.manager_blocks_per_unit, 0);
  CHECK_EQ(common_blocks_per_unit % cache_cost.manager_blocks_per_unit, 0);
  const int64_t compressed_units =
      common_blocks_per_unit / cache_cost.manager_blocks_per_unit;
  return compressed_units * cache_cost.token_unit_bytes;
}

void set_dsv4_compressed_counts(const Dsv4KVCacheEstimateCost& cache_cost,
                                int64_t token_unit_count,
                                KVCacheCapacity* kv_cache_cap) {
  CHECK(kv_cache_cap != nullptr);
  if (cache_cost.n_c4_layers > 0 && cache_cost.n_c128_layers > 0) {
    kv_cache_cap->c128_count(token_unit_count);
    kv_cache_cap->c4_count(32 * token_unit_count);
  } else if (cache_cost.n_c4_layers > 0) {
    kv_cache_cap->c4_count(token_unit_count);
  } else if (cache_cost.n_c128_layers > 0) {
    kv_cache_cap->c128_count(token_unit_count);
  }
}

Dsv4KVCacheEstimateCost estimate_dsv4_kv_cache_cost(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  const int64_t block_size = options.block_size;
  const int64_t head_dim = model_args.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(model_args.index_head_dim(), 1);
  const std::vector<int32_t>& compress_ratios = model_args.compress_ratios();
  const int64_t float32_size = 4;
  const int64_t dtype_size =
      static_cast<int64_t>(torch::elementSize(options.dtype));

  Dsv4KVCacheEstimateCost cache_cost;
  const int64_t swa_blocks_per_seq =
      get_swa_blocks_per_seq(model_args.window_size(), block_size);
  const int64_t max_seqs =
      std::max(options.max_seqs_per_batch, static_cast<int64_t>(1));
  int64_t burst_budget =
      std::max(options.max_tokens_per_batch, static_cast<int64_t>(0));
  int64_t publish_unit_blocks = 0;
  if (options.enable_dp_fair_token_budget && options.dp_size > 1 &&
      options.instance_role == InstanceRole::PREFILL) {
    // The scheduler caps each DP group at max_tokens_per_batch / dp_size
    // tokens per scheduling round, floored at one prefill chunk, so the
    // prefill burst landing on any single rank is bounded by the same
    // expression. Keep this identical to the scheduler-side cap or the ring
    // under-reserves.
    int64_t per_group_cap =
        (burst_budget + options.dp_size - 1) / options.dp_size;
    if (options.enable_chunked_prefill) {
      per_group_cap =
          std::max(per_group_cap,
                   std::min<int64_t>(options.max_tokens_per_chunk_for_prefill,
                                     burst_budget));
    } else {
      per_group_cap = burst_budget;
    }
    burst_budget = per_group_cap;
    if (options.enable_prefix_cache) {
      // Prefix reuse needs the SWA rows of a complete compressed checkpoint.
      for (const int32_t ratio : compress_ratios) {
        if (ratio > 1) {
          publish_unit_blocks =
              std::max(publish_unit_blocks, static_cast<int64_t>(ratio));
        }
      }
    }
  }
  const int64_t burst_blocks =
      std::max(util::ceil_div(burst_budget, block_size), publish_unit_blocks);
  cache_cost.swa_count =
      swa_blocks_per_seq * max_seqs + burst_blocks + max_seqs + 2;

  for (int64_t i = 0; i < model_args.n_layers(); ++i) {
    const int32_t ratio = i < static_cast<int64_t>(compress_ratios.size())
                              ? compress_ratios[static_cast<size_t>(i)]
                              : 1;
    if (ratio == 4) {
      ++cache_cost.n_c4_layers;
    } else if (ratio == 128) {
      ++cache_cost.n_c128_layers;
    }
  }
  const int64_t n_c1_layers =
      model_args.n_layers() - cache_cost.n_c4_layers - cache_cost.n_c128_layers;

  const int64_t swa_bytes_per_c1_block = block_size * head_dim * dtype_size;
  const int64_t swa_bytes_per_c4_block =
      block_size * head_dim * dtype_size +
      block_size * (2 * head_dim * float32_size) * 2 +
      block_size * (2 * index_head_dim * float32_size) * 2;
  const int64_t swa_bytes_per_c128_block =
      block_size * head_dim * dtype_size +
      block_size * head_dim * float32_size * 2;

  cache_cost.swa_bytes_per_block =
      n_c1_layers * swa_bytes_per_c1_block +
      cache_cost.n_c4_layers * swa_bytes_per_c4_block +
      cache_cost.n_c128_layers * swa_bytes_per_c128_block;
  cache_cost.constant_swa_bytes =
      cache_cost.swa_count * cache_cost.swa_bytes_per_block;

  const DeepSeekV4CachePolicy cache_policy =
      get_dsv4_cache_policy(options.dtype);
  const int64_t scale_bytes =
      cache_policy.has_indexer_cache_scale ? cache_policy.scale_dtype_size : 0;
  const int64_t bytes_per_c4_block =
      block_size *
      (head_dim * dtype_size + index_head_dim * cache_policy.index_dtype_size +
       scale_bytes);
  const int64_t bytes_per_c128_block = block_size * head_dim * dtype_size;

  if (cache_cost.n_c4_layers > 0 && cache_cost.n_c128_layers > 0) {
    cache_cost.token_unit_bytes =
        32 * cache_cost.n_c4_layers * bytes_per_c4_block +
        cache_cost.n_c128_layers * bytes_per_c128_block;
    cache_cost.manager_blocks_per_unit = 128;
  } else if (cache_cost.n_c4_layers > 0) {
    cache_cost.token_unit_bytes = cache_cost.n_c4_layers * bytes_per_c4_block;
    cache_cost.manager_blocks_per_unit = 4;
  } else if (cache_cost.n_c128_layers > 0) {
    cache_cost.token_unit_bytes =
        cache_cost.n_c128_layers * bytes_per_c128_block;
    cache_cost.manager_blocks_per_unit = 128;
  }
  return cache_cost;
}

void init_dsv4_counts(const ModelArgs& model_args,
                      const KVCacheEstimateOptions& options,
                      KVCacheCapacity* kv_cache_cap) {
  CHECK(kv_cache_cap != nullptr);
  const Dsv4KVCacheEstimateCost cache_cost =
      estimate_dsv4_kv_cache_cost(model_args, options);
  CHECK_GE(kv_cache_cap->cache_size_in_bytes(), cache_cost.constant_swa_bytes)
      << "no memory for the minimum DSV4 SWA cache, required="
      << readable_size(cache_cost.constant_swa_bytes)
      << ", available=" << readable_size(kv_cache_cap->cache_size_in_bytes());
  int64_t token_mem =
      kv_cache_cap->cache_size_in_bytes() - cache_cost.constant_swa_bytes;
  if (options.draft_model_args != nullptr) {
    CHECK(options.draft_options != nullptr)
        << "DSV4 draft options must be provided with draft model args";
    CHECK(
        util::is_deepseek_v4_model_type(options.draft_model_args->model_type()))
        << "DSV4 speculative kv cache estimation only supports DeepSeek V4 "
           "draft";
    const Dsv4KVCacheEstimateCost draft_cost = estimate_dsv4_kv_cache_cost(
        *options.draft_model_args, *options.draft_options);
    const int64_t constant_bytes =
        cache_cost.constant_swa_bytes + draft_cost.constant_swa_bytes;
    CHECK_GE(kv_cache_cap->cache_size_in_bytes(), constant_bytes)
        << "no memory left for speculative target/draft fixed kv cache "
           "allocation";

    const int64_t common_blocks_per_unit = std::max(
        cache_cost.manager_blocks_per_unit, draft_cost.manager_blocks_per_unit);
    const int64_t target_common_unit_bytes =
        dsv4_common_unit_bytes(cache_cost, common_blocks_per_unit);
    const int64_t draft_common_unit_bytes =
        dsv4_common_unit_bytes(draft_cost, common_blocks_per_unit);
    const int64_t combined_unit_bytes =
        target_common_unit_bytes + draft_common_unit_bytes;
    const int64_t remaining_bytes =
        kv_cache_cap->cache_size_in_bytes() - constant_bytes;
    if (combined_unit_bytes > 0) {
      CHECK_GE(remaining_bytes, combined_unit_bytes)
          << "minimum DSV4 target/draft SWA caches leave insufficient memory "
             "for one compressed cache unit, swa_required="
          << readable_size(constant_bytes)
          << ", compressed_unit_required=" << readable_size(combined_unit_bytes)
          << ", available="
          << readable_size(kv_cache_cap->cache_size_in_bytes());
    }
    const int64_t token_unit_count =
        combined_unit_bytes > 0 ? remaining_bytes / combined_unit_bytes : 0;

    const int64_t adjusted_cache_size_in_bytes =
        cache_cost.constant_swa_bytes +
        token_unit_count * target_common_unit_bytes;
    CHECK_GT(adjusted_cache_size_in_bytes, 0)
        << "no memory left for speculative target/draft kv cache allocation";
    LOG(INFO) << "speculative kv cache capacity adjusted from "
              << readable_size(kv_cache_cap->cache_size_in_bytes()) << " to "
              << readable_size(adjusted_cache_size_in_bytes)
              << ", target_constant_bytes=" << cache_cost.constant_swa_bytes
              << ", draft_constant_bytes=" << draft_cost.constant_swa_bytes
              << ", target_common_unit_bytes=" << target_common_unit_bytes
              << ", draft_common_unit_bytes=" << draft_common_unit_bytes
              << ", common_blocks_per_unit=" << common_blocks_per_unit
              << ", token_unit_count=" << token_unit_count;
    kv_cache_cap->cache_size_in_bytes(adjusted_cache_size_in_bytes);
    token_mem = token_unit_count * target_common_unit_bytes;
  } else {
    CHECK(options.draft_options == nullptr)
        << "DSV4 draft options require draft model args";
    if (cache_cost.token_unit_bytes > 0) {
      CHECK_GE(token_mem, cache_cost.token_unit_bytes)
          << "minimum DSV4 SWA cache leaves insufficient memory for one "
             "compressed cache unit, swa_required="
          << readable_size(cache_cost.constant_swa_bytes)
          << ", compressed_unit_required="
          << readable_size(cache_cost.token_unit_bytes) << ", available="
          << readable_size(kv_cache_cap->cache_size_in_bytes());
    }
  }

  kv_cache_cap->swa_count(cache_cost.swa_count);
  kv_cache_cap->c4_count(0);
  kv_cache_cap->c128_count(0);
  // Keep SWA at the operational minimum calculated above. Prefix-cache entries
  // share this pool; any remaining memory is reserved for compressed history.
  if (cache_cost.token_unit_bytes > 0 && token_mem > 0) {
    const int64_t token_unit_count = token_mem / cache_cost.token_unit_bytes;
    set_dsv4_compressed_counts(cache_cost, token_unit_count, kv_cache_cap);
  }

  CHECK_GT(kv_cache_cap->swa_count(), 0) << "DSV4 swa_count must be > 0";
  if (cache_cost.n_c4_layers > 0) {
    CHECK_GT(kv_cache_cap->c4_count(), 0)
        << "DSV4 c4_count must be > 0 when compress_ratio=4 layers exist";
  }
  if (cache_cost.n_c128_layers > 0) {
    CHECK_GT(kv_cache_cap->c128_count(), 0)
        << "DSV4 c128_count must be > 0 when compress_ratio=128 layers "
           "exist";
  }

  int64_t manager_base_blocks = 0;
  if (cache_cost.n_c4_layers > 0) {
    manager_base_blocks =
        std::max(manager_base_blocks, kv_cache_cap->c4_count() * 4);
  }
  if (cache_cost.n_c128_layers > 0) {
    manager_base_blocks =
        std::max(manager_base_blocks, kv_cache_cap->c128_count() * 128);
  }
  kv_cache_cap->n_blocks(std::max<int64_t>(manager_base_blocks, 1));
}

void init_standard_counts(const ModelArgs& model_args,
                          const KVCacheEstimateOptions& options,
                          KVCacheCapacity* kv_cache_cap) {
  for (int64_t layer_id = 0; layer_id < kv_cache_cap->n_layers(); ++layer_id) {
    if (is_full_attention_layer(model_args, layer_id)) {
      ++kv_cache_cap->num_full_attention_layers();
    } else {
      ++kv_cache_cap->num_linear_attention_layers();
    }
  }

  if (kv_cache_cap->index_slot_size() > 0) {
    int64_t num_indexer_layers = kv_cache_cap->num_full_attention_layers();
    const std::vector<bool> indexer_layer_mask =
        resolve_indexer_cache_enabled_layers(model_args,
                                             kv_cache_cap->n_layers());
    if (!indexer_layer_mask.empty()) {
      num_indexer_layers = static_cast<int64_t>(std::count(
          indexer_layer_mask.begin(), indexer_layer_mask.end(), true));
    }
    kv_cache_cap->num_indexer_layers(num_indexer_layers);
  }

  CHECK_GE(options.embedding_context_bytes_per_block, 0);
  const int64_t full_cache_block_size_in_bytes =
      standard_full_cache_block_size_in_bytes(*kv_cache_cap) +
      options.embedding_context_bytes_per_block;
  const int64_t gdn_state_bytes = kv_cache_cap->num_linear_attention_layers() *
                                  kv_cache_cap->linear_slot_size();
  const int64_t kpool_state_bytes =
      kv_cache_cap->num_indexer_layers() * kv_cache_cap->kpool_tail_slot_size();
  const int64_t total_state_bytes = gdn_state_bytes + kpool_state_bytes;
  int64_t num_linear_state_blocks =
      calculate_linear_state_blocks(kv_cache_cap->cache_size_in_bytes(),
                                    total_state_bytes,
                                    full_cache_block_size_in_bytes,
                                    options.max_seqs_per_batch,
                                    options.max_concurrent_requests,
                                    options.max_linear_state_cache_slots,
                                    options.enable_prefix_cache);
  CHECK_GE(options.linear_state_cache_block_limit, 0);
  if (options.linear_state_cache_block_limit > 0 &&
      kv_cache_cap->num_linear_attention_layers() > 0) {
    CHECK_LE(options.max_linear_state_cache_slots + kPaddingLinearStateBlocks,
             options.linear_state_cache_block_limit)
        << "Selected backend supports at most "
        << options.linear_state_cache_block_limit
        << " physical linear-state slots, including "
        << kPaddingLinearStateBlocks << " padding slots.";
    num_linear_state_blocks = std::min(num_linear_state_blocks,
                                       options.linear_state_cache_block_limit);
  }
  kv_cache_cap->num_linear_state_blocks(num_linear_state_blocks);
  kv_cache_cap->linear_cache_size_in_bytes(
      kv_cache_cap->num_linear_state_blocks() * total_state_bytes);
  const int64_t available_full_cache_size_in_bytes =
      kv_cache_cap->cache_size_in_bytes() -
      kv_cache_cap->linear_cache_size_in_bytes();
  if (total_state_bytes > 0) {
    CHECK_GT(kv_cache_cap->cache_size_in_bytes(),
             kv_cache_cap->linear_cache_size_in_bytes())
        << "failed to reserve linear state cache for linear-attention "
           "layers: "
        << "max_seqs_per_batch (" << options.max_seqs_per_batch
        << ") is too large. Please reduce max_seqs_per_batch to less than "
        << kv_cache_cap->cache_size_in_bytes() / total_state_bytes - 2;
  }
  CHECK_GT(available_full_cache_size_in_bytes, 0)
      << "no memory left for full-attention kv cache after reserving linear "
         "state cache";
  if (options.layerwise_split_size > 1) {
    kv_cache_cap->n_blocks(
        layerwise_split_block_count(model_args,
                                    options.layerwise_split_size,
                                    *kv_cache_cap,
                                    available_full_cache_size_in_bytes,
                                    options.embedding_context_bytes_per_block));
  } else {
    kv_cache_cap->n_blocks(available_full_cache_size_in_bytes /
                           full_cache_block_size_in_bytes);
  }
  CHECK_GT(kv_cache_cap->n_blocks(), 0) << "no n_blocks for kv cache";
}

}  // namespace

KPoolCacheLayout default_kpool_layout() {
  return Platform::is_mlu() ? KPoolCacheLayout::COMPRESSED_WITH_TAIL
                            : KPoolCacheLayout::PACKED;
}

int64_t estimate_shared_kpool_blocks(const KVCacheCapacity& target_cap,
                                     const KVCacheCapacity& draft_cap,
                                     const ModelArgs& draft_args) {
  CHECK_EQ(target_cap.block_size(), draft_cap.block_size());
  CHECK_GT(target_cap.kpool_tail_slot_size(), 0);
  CHECK_GT(draft_cap.kpool_tail_slot_size(), 0);
  CHECK_EQ(draft_cap.num_linear_attention_layers(), 0);
  // Draft request slots and the verify window are inherited from the target
  // shape, rather than from the draft's standalone capacity estimate.
  const int64_t draft_tail_len =
      std::max<int64_t>(target_cap.kpool_tail_len(), draft_args.index_kpool());
  const int64_t draft_state_bytes =
      draft_cap.num_indexer_layers() * target_cap.num_linear_state_blocks() *
      kpool_tail_slot_size_bytes(draft_tail_len, draft_args.index_head_dim());
  const int64_t state_bytes =
      target_cap.linear_cache_size_in_bytes() + draft_state_bytes;
  const int64_t budget = std::min(target_cap.cache_size_in_bytes(),
                                  draft_cap.cache_size_in_bytes());
  CHECK_GT(budget, state_bytes)
      << "no memory left after reserving target and draft KPool request state";
  const int64_t block_bytes =
      standard_full_cache_block_size_in_bytes(target_cap) +
      standard_full_cache_block_size_in_bytes(draft_cap);
  const int64_t n_blocks = (budget - state_bytes) / block_bytes;
  CHECK_GT(n_blocks, 0) << "no memory for a shared KPool block";
  return n_blocks;
}

std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& model_args,
    int64_t num_cache_layers) {
  if (!Platform::supports_dsa_indexer_cache_elision()) {
    return {};
  }
  return layer::get_dsa_indexer_layer_mask(model_args, num_cache_layers);
}

std::vector<bool> build_layer_cache_owned(const ModelArgs& model_args,
                                          const LayerwiseSplitLayout& layout,
                                          int64_t num_layers) {
  std::vector<bool> layer_cache_owned;
  layer_cache_owned.reserve(static_cast<size_t>(num_layers));
  for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
    layer_cache_owned.emplace_back(
        !is_full_attention_layer(model_args, layer_id) ||
        layout.owns(layer_id));
  }
  return layer_cache_owned;
}

int64_t estimate_layerwise_split_block_count(
    const ModelArgs& model_args,
    int32_t layerwise_split_size,
    const KVCacheCapacity& kv_cache_cap,
    int64_t available_bytes,
    int64_t additional_block_bytes) {
  return layerwise_split_block_count(model_args,
                                     layerwise_split_size,
                                     kv_cache_cap,
                                     available_bytes,
                                     additional_block_bytes);
}

int64_t KVCacheEstimator::estimate_memory_budget(
    const runtime::Options& options,
    const KVCacheEstimateContext& context) const {
  if (context.xtensor_cache_size.has_value()) {
    CHECK_GT(*context.xtensor_cache_size, 0)
        << "XTensor KV cache budget must be positive";
    LOG(INFO) << "XTensor mode: available memory from PhyPagePool: "
              << readable_size(*context.xtensor_cache_size);
    return *context.xtensor_cache_size;
  }

  CHECK(!context.worker_memory.empty())
      << "KV cache estimation requires worker memory snapshots";
  const int64_t encoder_cache_reserved_bytes =
      context.is_multimodal ? options.max_encoder_cache_size() * 1024 * 1024
                            : 0;
  int64_t cache_size_in_bytes = std::numeric_limits<int64_t>::max();
  for (size_t i = 0; i < context.worker_memory.size(); ++i) {
    const KVCacheMemorySnapshot& memory = context.worker_memory[i];
    int64_t available_memory = memory.available_memory;
    const int64_t total_memory = memory.total_memory;
    LOG(INFO) << "worker #" << i
              << ": available memory: " << readable_size(available_memory)
              << ", total memory: " << readable_size(total_memory)
              << ". Using max_memory_utilization: "
              << options.max_memory_utilization()
              << ", max_cache_size: " << readable_size(options.max_cache_size())
              << ", encoder_cache_reserved: "
              << readable_size(encoder_cache_reserved_bytes);
    GAUGE_SET(weight_size_in_kilobytes,
              (total_memory - available_memory) / 1024);
    GAUGE_SET(total_memory_size_in_kilobytes, total_memory / 1024);
    if (options.max_memory_utilization() < 1.0) {
      const int64_t buffer_memory = static_cast<int64_t>(
          std::ceil(total_memory * (1.0 - options.max_memory_utilization())));
      available_memory -= buffer_memory;
    }
    if (options.max_cache_size() > 0) {
      available_memory = std::min(available_memory, options.max_cache_size());
    }
    available_memory -= encoder_cache_reserved_bytes;
    cache_size_in_bytes = std::min(cache_size_in_bytes, available_memory);
  }
  return cache_size_in_bytes;
}

KVCacheCapacity KVCacheEstimator::estimate(
    const runtime::Options& options,
    const KVCacheEstimateContext& context) const {
  KVCacheEstimateOptions estimate_options;
  estimate_options.dtype = context.dtype;
  estimate_options.kv_cache_dtype = options.kv_cache_dtype();
  estimate_options.indexer_cache_dtype =
      ::xllm::KVCacheConfig::get_instance().indexer_cache_dtype();
  estimate_options.cache_size_in_bytes =
      estimate_memory_budget(options, context);
  estimate_options.block_size = options.block_size();
  estimate_options.world_size = context.world_size;
  estimate_options.max_seqs_per_batch =
      static_cast<int64_t>(options.max_seqs_per_batch());
  estimate_options.max_concurrent_requests = static_cast<int64_t>(
      ::xllm::ServiceConfig::get_instance().max_concurrent_requests());
  estimate_options.max_tokens_per_batch =
      static_cast<int64_t>(options.max_tokens_per_batch());
  estimate_options.max_tokens_per_chunk_for_prefill =
      static_cast<int64_t>(options.max_tokens_per_chunk_for_prefill());
  estimate_options.max_linear_state_cache_slots =
      options.max_linear_state_cache_slots();
  estimate_options.linear_state_cache_block_limit =
      context.linear_state_cache_block_limit;
  estimate_options.is_draft_engine = options.is_draft_engine();
  estimate_options.enable_chunked_prefill = options.enable_chunked_prefill();
  estimate_options.enable_schedule_overlap = options.enable_schedule_overlap();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  const bool is_mtp_draft_engine =
      options.is_draft_engine() &&
      SpeculativeConfig::is_mtp_algorithm(options.speculative_algorithm());
  estimate_options.enable_prefix_cache = options.enable_prefix_cache() &&
                                         !kv_cache_config.enable_xtensor() &&
                                         !is_mtp_draft_engine;
  estimate_options.enable_disagg_pd = options.enable_disagg_pd();
  estimate_options.instance_role = options.instance_role();
  if (!context.is_multimodal) {
    estimate_options.num_speculative_tokens =
        static_cast<int64_t>(options.num_speculative_tokens());
    estimate_options.dp_size = options.dp_size();
    estimate_options.enable_dp_fair_token_budget =
        ::xllm::SchedulerConfig::get_instance().enable_dp_fair_token_budget();
    if (options.enable_mtp_draft_body_tp1() && options.is_draft_engine()) {
      estimate_options.world_size = 1;
    }
    estimate_options.layerwise_split_size =
        options.is_draft_engine()
            ? 1
            : ParallelConfig::get_instance().layerwise_split_size();

    if (options.enable_task_pipeline() &&
        options.num_speculative_tokens() > 0 && !options.is_draft_engine()) {
      estimate_options.embedding_context_bytes_per_block =
          sizeof(int64_t) + 2 * sizeof(int32_t);
      if (!SpeculativeConfig::is_block_diffusion_algorithm(
              options.speculative_algorithm())) {
        CHECK_GT(model_args_.hidden_size(), 0);
        const int64_t element_bytes = static_cast<int64_t>(
            torch::scalarTypeToTypeMeta(context.dtype).itemsize());
        estimate_options.embedding_context_bytes_per_block +=
            sizeof(int64_t) + 2 * model_args_.hidden_size() * element_bytes +
            sizeof(bool);
      }
    }
  }

  return estimate(std::move(estimate_options));
}

KVCacheCapacity KVCacheEstimator::estimate(
    KVCacheEstimateOptions options) const {
  CHECK_GT(options.world_size, 0) << "world_size must be positive";
  const int64_t n_heads = model_args_.n_heads();
  const int64_t n_kv_heads = model_args_.n_kv_heads().value_or(n_heads);
  options.n_local_kv_heads =
      std::max<int64_t>(1, n_kv_heads / options.world_size);
  if (has_linear_attention_layers(model_args_) &&
      model_args_.linear_num_value_heads() > 0) {
    options.n_local_linear_k_heads = std::max<int64_t>(
        1, model_args_.linear_num_key_heads() / options.world_size);
    options.n_local_linear_v_heads = std::max<int64_t>(
        1, model_args_.linear_num_value_heads() / options.world_size);
  } else {
    options.n_local_linear_k_heads = 0;
    options.n_local_linear_v_heads = 0;
  }
  return estimate_kv_cache_capacity(model_args_, options);
}

KVCacheCapacity estimate_kv_cache_capacity(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  KVCacheCapacity kv_cache_cap;
  kv_cache_cap
      .cache_size_in_bytes(
          std::max(options.cache_size_in_bytes, static_cast<int64_t>(0)))
      .block_size(options.block_size);
  CHECK_GT(kv_cache_cap.cache_size_in_bytes(), 0)
      << "Available kv cache size must be greater than 0";
  const bool enable_dsv4_estimation =
      util::is_deepseek_v4_model_type(model_args.model_type());
  if (options.draft_model_args != nullptr) {
    CHECK(enable_dsv4_estimation)
        << "DSV4 speculative kv cache estimation only supports DeepSeek V4 "
           "target";
  }

  const int64_t dtype_size = static_cast<int64_t>(
      torch::scalarTypeToTypeMeta(options.dtype).itemsize());
  const int64_t cache_dtype_size =
      kv_cache_dtype_size(options.kv_cache_dtype, dtype_size);
  const bool enable_indexer_cache_quantization =
      options.indexer_cache_dtype == "int8";
  const bool enable_mla_kv_cache_quantization =
      model_args.enable_mla() &&
      util::enable_mla_packed_c8(options.kv_cache_dtype == "int8",
                                 model_args.model_type());

  kv_cache_cap.slot_size(kv_slot_size(model_args, options, cache_dtype_size))
      .index_slot_size(index_slot_size(model_args,
                                       enable_indexer_cache_quantization,
                                       dtype_size,
                                       options.kpool_layout))
      .kpool_layout(options.kpool_layout)
      .enable_indexer_cache_quant(enable_indexer_cache_quantization)
      .enable_mla_kv_cache_quant(enable_mla_kv_cache_quantization)
      .scale_slot_size(scale_slot_size(model_args, options))
      .linear_slot_size(linear_slot_size(model_args, options, dtype_size))
      .n_layers(model_args.n_layers())
      .block_size(options.block_size);
  const int64_t num_speculative_tokens =
      enable_qwen3_5_spec_verify(model_args, options)
          ? options.num_speculative_tokens
          : 0;
  kv_cache_cap.linear_conv_state_len(model_args.linear_conv_kernel_dim() - 1 +
                                     num_speculative_tokens);
  kv_cache_cap.linear_ssm_checkpoint_stride(num_speculative_tokens + 1);
  if (options.is_draft_engine && model_args.num_nextn_predict_layers() > 0) {
    kv_cache_cap.n_layers(model_args.num_nextn_predict_layers());
  }

  if (model_args.index_kpool_compress() &&
      options.kpool_layout == KPoolCacheLayout::COMPRESSED_WITH_TAIL) {
    CHECK_EQ(options.layerwise_split_size, 1)
        << "Compressed KPool requires owned layer caches.";
    CHECK_EQ(options.dtype, torch::kBFloat16);
    CHECK_EQ(options.block_size % model_args.index_kpool(), 0);
    CHECK_GE(options.num_speculative_tokens, 0);
    // One verify window includes its base token. Reserve a second window for
    // the asynchronous next-first-draft handoff before the commit is observed.
    const int64_t window = options.num_speculative_tokens > 0
                               ? 2 * (options.num_speculative_tokens + 1)
                               : 0;
    kv_cache_cap.kpool_tail_len(model_args.index_kpool() + window);
    kv_cache_cap.kpool_tail_slot_size(kpool_tail_slot_size_bytes(
        kv_cache_cap.kpool_tail_len(), model_args.index_head_dim()));
  }

  if (enable_dsv4_estimation) {
    init_dsv4_counts(model_args, options, &kv_cache_cap);
  } else {
    init_standard_counts(model_args, options, &kv_cache_cap);
  }
  return kv_cache_cap;
}

}  // namespace xllm
