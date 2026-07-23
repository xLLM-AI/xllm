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

#include "framework/kv_cache/kv_cache_estimation.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/layers/common/dsa_topk_share_plan.h"
#include "core/platform/platform.h"
#include "framework/block/block_utils.h"
#include "framework/kv_cache/deepseek_v4_cache_policy.h"
#include "framework/model/model_args.h"
#include "platform/mlu/mlu_rdma_memory_plan.h"
#include "util/pretty_print.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

namespace xllm {

namespace {

constexpr int32_t kNzAlignment = 16;

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
                        int64_t dtype_size) {
  if (model_args.index_n_heads() <= 0) {
    return 0;
  }

  const int64_t index_n_head = 1;
  if (enable_indexer_cache_quantization) {
    // int8 index cache: one byte per element, plus an independent per-token
    // fp32 scale (kept separate from the main-KV scale_slot_size path).
    return static_cast<int64_t>(sizeof(int8_t)) * index_n_head *
               model_args.index_head_dim() +
           static_cast<int64_t>(sizeof(float));
  }
  return dtype_size * index_n_head * model_args.index_head_dim();
}

int64_t scale_slot_size(const ModelArgs& model_args,
                        const KVCacheEstimateOptions& options) {
  if (options.kv_cache_dtype == "auto") {
    return 0;
  }
  if (model_args.enable_mla()) {
    return sizeof(float);
  }
  return 2 * sizeof(float) * options.n_local_kv_heads;
}

bool use_rdma_indexer_scale_padding(const KVCacheEstimateOptions& options,
                                    const KVCacheCapacity& kv_cache_cap) {
#if defined(USE_MLU)
  return options.enable_rdma_scale_padding &&
         kv_cache_cap.enable_indexer_cache_quant();
#else
  (void)options;
  (void)kv_cache_cap;
  return false;
#endif
}

size_t checked_product(size_t lhs, size_t rhs, const char* description) {
  if (lhs > static_cast<size_t>(0)) {
    CHECK_LE(rhs, std::numeric_limits<size_t>::max() / lhs)
        << description << " overflow";
  }
  return lhs * rhs;
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

size_t standard_full_cache_allocation_bytes(const KVCacheCapacity& kv_cache_cap,
                                            int64_t n_blocks,
                                            bool enable_rdma_scale_padding) {
  CHECK_GT(n_blocks, 0) << "n_blocks must be positive";
  const int64_t logical_block_bytes =
      standard_full_cache_block_size_in_bytes(kv_cache_cap);

  const size_t logical_bytes =
      checked_product(static_cast<size_t>(n_blocks),
                      static_cast<size_t>(logical_block_bytes),
                      "full cache logical bytes");
  if (!enable_rdma_scale_padding) {
    return logical_bytes;
  }

  const size_t scale_block_bytes =
      checked_product(static_cast<size_t>(kv_cache_cap.block_size()),
                      static_cast<size_t>(sizeof(float)),
                      "indexer scale block bytes");
  const size_t scale_logical_bytes =
      checked_product(static_cast<size_t>(n_blocks),
                      scale_block_bytes,
                      "indexer scale logical bytes");
  const mlu::RdmaMemoryPlan scale_plan = mlu::make_rdma_memory_plan(
      scale_logical_bytes, static_cast<size_t>(n_blocks));
  const size_t padding_per_layer =
      scale_plan.registered_bytes - scale_plan.logical_bytes;
  const size_t total_padding =
      checked_product(static_cast<size_t>(kv_cache_cap.num_indexer_layers()),
                      padding_per_layer,
                      "indexer scale RDMA padding");
  CHECK_LE(total_padding, std::numeric_limits<size_t>::max() - logical_bytes)
      << "full cache allocation bytes overflow";
  return logical_bytes + total_padding;
}

bool is_qwen3_5_target_model_type(const std::string& model_type) {
  return model_type == "qwen3_5" || model_type == "qwen3_5_moe" ||
         model_type == "qwen3_5_text" || model_type == "qwen3_5_moe_text" ||
         model_type.rfind("qwen3_5_", 0) == 0;
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
                                int64_t num_linear_attention_layers,
                                int64_t linear_slot_size,
                                int64_t full_cache_block_size_in_bytes) {
  if (linear_slot_size <= 0 || num_linear_attention_layers <= 0) {
    return kPaddingLinearStateBlocks;
  }

  CHECK_GT(cache_size_in_bytes, 0);
  CHECK_GT(full_cache_block_size_in_bytes, 0);
  const int64_t linear_bytes_per_block =
      num_linear_attention_layers * linear_slot_size;
  CHECK_GT(linear_bytes_per_block, 0);

  int64_t max_linear_blocks =
      (cache_size_in_bytes - 1) / linear_bytes_per_block;
  const int64_t balanced_max_linear_blocks =
      (cache_size_in_bytes +
       kPaddingLinearStateBlocks * full_cache_block_size_in_bytes) /
      (linear_bytes_per_block + full_cache_block_size_in_bytes);
  max_linear_blocks = std::min(max_linear_blocks, balanced_max_linear_blocks);

  return std::max<int64_t>(max_linear_blocks, kPaddingLinearStateBlocks);
}

int64_t calculate_linear_state_blocks(int64_t cache_size_in_bytes,
                                      int64_t num_linear_attention_layers,
                                      int64_t linear_slot_size,
                                      int64_t full_cache_block_size_in_bytes,
                                      int64_t max_seqs_per_batch,
                                      int64_t max_linear_state_cache_slots,
                                      bool enable_prefix_cache) {
  CHECK_GE(max_linear_state_cache_slots, 0)
      << "max_linear_state_cache_slots must be greater than or equal to 0.";
  if (num_linear_attention_layers <= 0 || linear_slot_size <= 0) {
    return kPaddingLinearStateBlocks;
  }
  const int64_t max_blocks =
      max_linear_state_blocks(cache_size_in_bytes,
                              num_linear_attention_layers,
                              linear_slot_size,
                              full_cache_block_size_in_bytes);
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
    const int64_t live_slot_blocks =
        max_seqs_per_batch + kPaddingLinearStateBlocks;
    return std::max<int64_t>(std::min<int64_t>(live_slot_blocks, max_blocks),
                             kPaddingLinearStateBlocks);
  }

  // Auto-size: allocate ~47% of cache bytes to linear-state slots (ratio 0.9
  // means linear fraction = 0.9 / 1.9).
  constexpr double kLinearStateFullKvMemoryRatio = 0.9;
  const int64_t linear_bytes_per_block =
      num_linear_attention_layers * linear_slot_size;
  int64_t auto_blocks = kPaddingLinearStateBlocks;
  if (linear_slot_size > 0 && num_linear_attention_layers > 0 &&
      linear_bytes_per_block > 0) {
    const double linear_memory_fraction =
        kLinearStateFullKvMemoryRatio / (1.0 + kLinearStateFullKvMemoryRatio);
    const double linear_memory_bytes =
        static_cast<double>(cache_size_in_bytes) * linear_memory_fraction;
    auto_blocks = std::max<int64_t>(
        static_cast<int64_t>(linear_memory_bytes / linear_bytes_per_block),
        kPaddingLinearStateBlocks);
  }
  // Both bounds already sit at or above kPaddingLinearStateBlocks (auto_blocks
  // is floored above; max_blocks is floored in max_linear_state_blocks), so the
  // min of the two cannot drop below it.
  return std::min<int64_t>(auto_blocks, max_blocks);
}

Dsv4KVCacheEstimateCost estimate_dsv4_kv_cache_cost(
    const ModelArgs& model_args,
    const KVCacheEstimateOptions& options) {
  const int64_t max_seqs =
      std::max(options.max_seqs_per_batch, static_cast<int64_t>(1));
  const int64_t block_size = options.block_size;
  const int64_t semantic_window = std::max(model_args.window_size(), 1);
  const int64_t max_model_len = model_args.max_seq_len();
  const int64_t window_size =
      max_model_len > 0 ? std::min<int64_t>(semantic_window, max_model_len)
                        : semantic_window;
  const int64_t swa_blocks_per_seq =
      get_swa_blocks_per_seq(window_size, block_size);
  const int64_t burst_blocks = util::ceil_div(
      std::max(options.max_tokens_per_batch, static_cast<int64_t>(1)),
      block_size);
  const int64_t head_dim = model_args.head_dim();
  const int64_t index_head_dim =
      std::max<int64_t>(model_args.index_head_dim(), 1);
  const std::vector<int32_t>& compress_ratios = model_args.compress_ratios();
  const int64_t float32_size = 4;
  const int64_t dtype_size =
      static_cast<int64_t>(torch::elementSize(options.dtype));

  Dsv4KVCacheEstimateCost cache_cost;
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

  const int64_t swa_bytes_per_c1_layer =
      cache_cost.swa_count * block_size * head_dim * dtype_size;
  const int64_t swa_bytes_per_c4_layer =
      cache_cost.swa_count *
      (block_size * head_dim * dtype_size +
       block_size * (2 * head_dim * float32_size) * 2 +
       block_size * (2 * index_head_dim * float32_size) * 2);
  const int64_t swa_bytes_per_c128_layer =
      cache_cost.swa_count * (block_size * head_dim * dtype_size +
                              block_size * head_dim * float32_size * 2);

  cache_cost.constant_swa_bytes =
      n_c1_layers * swa_bytes_per_c1_layer +
      cache_cost.n_c4_layers * swa_bytes_per_c4_layer +
      cache_cost.n_c128_layers * swa_bytes_per_c128_layer;

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
  int64_t token_mem = std::max(
      static_cast<int64_t>(0),
      kv_cache_cap->cache_size_in_bytes() - cache_cost.constant_swa_bytes);

  if (options.draft_model_args != nullptr) {
    CHECK(options.draft_options != nullptr)
        << "DSV4 draft options must be provided with draft model args";
    CHECK(util::is_target_model_type(options.draft_model_args->model_type(),
                                     /*target_type=*/"deepseek_v4",
                                     /*match_mtp=*/true))
        << "DSV4 MTP kv cache estimation only supports DeepSeek V4 draft";
    const Dsv4KVCacheEstimateCost draft_cost = estimate_dsv4_kv_cache_cost(
        *options.draft_model_args, *options.draft_options);
    const int64_t constant_bytes =
        cache_cost.constant_swa_bytes + draft_cost.constant_swa_bytes;
    CHECK_GT(kv_cache_cap->cache_size_in_bytes(), constant_bytes)
        << "no memory left for mtp target/draft fixed kv cache allocation";

    const int64_t token_unit_bytes =
        cache_cost.token_unit_bytes + draft_cost.token_unit_bytes;
    CHECK_GT(token_unit_bytes, 0)
        << "mtp target and draft token unit bytes must be positive";
    const int64_t token_unit_count =
        (kv_cache_cap->cache_size_in_bytes() - constant_bytes) /
        token_unit_bytes;
    CHECK_GT(token_unit_count, 0)
        << "no memory left for mtp target/draft kv cache token blocks";

    const int64_t adjusted_cache_size_in_bytes =
        cache_cost.constant_swa_bytes +
        token_unit_count * cache_cost.token_unit_bytes;
    CHECK_GT(adjusted_cache_size_in_bytes, 0)
        << "no memory left for mtp target/draft kv cache allocation";
    LOG(INFO) << "mtp kv cache capacity adjusted from "
              << readable_size(kv_cache_cap->cache_size_in_bytes()) << " to "
              << readable_size(adjusted_cache_size_in_bytes)
              << ", target_constant_bytes=" << cache_cost.constant_swa_bytes
              << ", draft_constant_bytes=" << draft_cost.constant_swa_bytes
              << ", target_token_unit_bytes=" << cache_cost.token_unit_bytes
              << ", draft_token_unit_bytes=" << draft_cost.token_unit_bytes
              << ", token_unit_count=" << token_unit_count;
    kv_cache_cap->cache_size_in_bytes(adjusted_cache_size_in_bytes);
    token_mem = token_unit_count * cache_cost.token_unit_bytes;
  } else {
    CHECK(options.draft_options == nullptr)
        << "DSV4 draft options require draft model args";
  }

  kv_cache_cap->swa_count(cache_cost.swa_count);
  kv_cache_cap->c4_count(0);
  kv_cache_cap->c128_count(0);
  if (cache_cost.n_c4_layers > 0 && cache_cost.n_c128_layers > 0) {
    if (cache_cost.token_unit_bytes > 0 && token_mem > 0) {
      kv_cache_cap->c128_count(token_mem / cache_cost.token_unit_bytes);
      kv_cache_cap->c4_count(32 * kv_cache_cap->c128_count());
    }
  } else if (cache_cost.n_c4_layers > 0) {
    if (cache_cost.token_unit_bytes > 0 && token_mem > 0) {
      kv_cache_cap->c4_count(token_mem / cache_cost.token_unit_bytes);
    }
  } else if (cache_cost.n_c128_layers > 0) {
    if (cache_cost.token_unit_bytes > 0 && token_mem > 0) {
      kv_cache_cap->c128_count(token_mem / cache_cost.token_unit_bytes);
    }
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

  const int64_t block_size = kv_cache_cap->block_size();
  const int64_t full_cache_block_size_in_bytes =
      standard_full_cache_block_size_in_bytes(*kv_cache_cap);
  kv_cache_cap->num_linear_state_blocks(
      calculate_linear_state_blocks(kv_cache_cap->cache_size_in_bytes(),
                                    kv_cache_cap->num_linear_attention_layers(),
                                    kv_cache_cap->linear_slot_size(),
                                    full_cache_block_size_in_bytes,
                                    options.max_seqs_per_batch,
                                    options.max_linear_state_cache_slots,
                                    options.enable_prefix_cache));
  kv_cache_cap->linear_cache_size_in_bytes(
      kv_cache_cap->num_linear_attention_layers() *
      kv_cache_cap->num_linear_state_blocks() *
      kv_cache_cap->linear_slot_size());
  const int64_t available_full_cache_size_in_bytes =
      kv_cache_cap->cache_size_in_bytes() -
      kv_cache_cap->linear_cache_size_in_bytes();
  if (kv_cache_cap->linear_slot_size() > 0) {
    CHECK_GT(kv_cache_cap->cache_size_in_bytes(),
             kv_cache_cap->linear_cache_size_in_bytes())
        << "failed to reserve linear state cache for linear-attention "
           "layers: "
        << "max_seqs_per_batch (" << options.max_seqs_per_batch
        << ") is too large. Please reduce max_seqs_per_batch to less than "
        << kv_cache_cap->cache_size_in_bytes() /
                   (kv_cache_cap->num_linear_attention_layers() *
                    kv_cache_cap->linear_slot_size()) -
               2;
  }
  CHECK_GT(available_full_cache_size_in_bytes, 0)
      << "no memory left for full-attention kv cache after reserving linear "
         "state cache";
  const int64_t logical_n_blocks =
      available_full_cache_size_in_bytes / full_cache_block_size_in_bytes;
  const bool enable_rdma_scale_padding =
      use_rdma_indexer_scale_padding(options, *kv_cache_cap);
  if (!enable_rdma_scale_padding) {
    kv_cache_cap->n_blocks(logical_n_blocks);
    CHECK_GT(kv_cache_cap->n_blocks(), 0) << "no n_blocks for kv cache";
    return;
  }

  CHECK_GT(logical_n_blocks, 0) << "no logical n_blocks for kv cache";
  const size_t available_bytes =
      static_cast<size_t>(available_full_cache_size_in_bytes);
  const size_t minimum_allocation_bytes =
      standard_full_cache_allocation_bytes(*kv_cache_cap,
                                           /*n_blocks=*/1,
                                           enable_rdma_scale_padding);
  const mlu::RdmaMemoryPlan minimum_scale_plan = mlu::make_rdma_memory_plan(
      static_cast<size_t>(block_size) * sizeof(float),
      /*block_count=*/1);
  CHECK_LE(minimum_allocation_bytes, available_bytes)
      << "no memory for one KV cache block with RDMA-registerable indexer "
         "scales: available_bytes="
      << available_full_cache_size_in_bytes
      << ", full_attention_layers=" << kv_cache_cap->num_full_attention_layers()
      << ", indexer_layers=" << kv_cache_cap->num_indexer_layers()
      << ", scale_registered_bytes_per_layer="
      << minimum_scale_plan.registered_bytes
      << ", minimum_allocation_bytes=" << minimum_allocation_bytes;

  int64_t low = 1;
  int64_t high = logical_n_blocks;
  while (low < high) {
    const int64_t mid = low + (high - low + 1) / 2;
    const size_t allocation_bytes = standard_full_cache_allocation_bytes(
        *kv_cache_cap, mid, enable_rdma_scale_padding);
    if (allocation_bytes <= available_bytes) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  kv_cache_cap->n_blocks(low);

  const size_t scale_logical_bytes =
      checked_product(static_cast<size_t>(kv_cache_cap->n_blocks()),
                      static_cast<size_t>(block_size) * sizeof(float),
                      "indexer scale logical bytes");
  const mlu::RdmaMemoryPlan scale_plan = mlu::make_rdma_memory_plan(
      scale_logical_bytes, static_cast<size_t>(kv_cache_cap->n_blocks()));
  LOG(INFO) << "RDMA indexer scale capacity adjusted from " << logical_n_blocks
            << " to " << kv_cache_cap->n_blocks()
            << " blocks, scale_logical_bytes_per_layer="
            << scale_plan.logical_bytes << ", scale_registered_bytes_per_layer="
            << scale_plan.registered_bytes << ", full_attention_layers="
            << kv_cache_cap->num_full_attention_layers()
            << ", indexer_layers=" << kv_cache_cap->num_indexer_layers();
  CHECK_GT(kv_cache_cap->n_blocks(), 0) << "no n_blocks for kv cache";
}

}  // namespace

std::vector<bool> resolve_indexer_cache_enabled_layers(
    const ModelArgs& model_args,
    int64_t num_cache_layers) {
  if (!Platform::supports_dsa_indexer_cache_elision()) {
    return {};
  }
  return layer::get_dsa_indexer_layer_mask(model_args, num_cache_layers);
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
      util::is_target_model_type(model_args.model_type(),
                                 /*target_type=*/"deepseek_v4",
                                 /*match_mtp=*/true);
  if (options.draft_model_args != nullptr) {
    CHECK(enable_dsv4_estimation)
        << "DSV4 MTP kv cache estimation only supports DeepSeek V4 target";
  }

  const int64_t dtype_size = static_cast<int64_t>(
      torch::scalarTypeToTypeMeta(options.dtype).itemsize());
  const int64_t cache_dtype_size =
      kv_cache_dtype_size(options.kv_cache_dtype, dtype_size);
  const bool enable_indexer_cache_quantization =
      options.indexer_cache_dtype == "int8";

  kv_cache_cap.slot_size(kv_slot_size(model_args, options, cache_dtype_size))
      .index_slot_size(index_slot_size(
          model_args, enable_indexer_cache_quantization, dtype_size))
      .enable_indexer_cache_quant(enable_indexer_cache_quantization)
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
#if !defined(USE_NPU)
  if (options.is_draft_engine) {
    kv_cache_cap.n_layers(model_args.num_nextn_predict_layers());
  }
#endif

  if (enable_dsv4_estimation) {
    init_dsv4_counts(model_args, options, &kv_cache_cap);
  } else {
    init_standard_counts(model_args, options, &kv_cache_cap);
  }
  return kv_cache_cap;
}

}  // namespace xllm
