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

#include "layers/mlu/deepseek_v4/deepseek_v4_attention.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/block/block_utils.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/dsa_cache_mapping.h"
#include "layers/mlu/deepseek_v4_ref_utils.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kLayerId = 0;
constexpr int64_t kBlockSize = 16;

torch::Tensor seeded(const std::string& key,
                     torch::IntArrayRef shape,
                     torch::ScalarType dtype,
                     const torch::Device& device) {
  return (test::seeded_tensor(key, shape, dtype, device) - 0.5) * 0.2;
}

DSACacheMapping make_cache_mapping(int64_t ratio) {
  DSACacheMapping mapping;
  if (ratio == 1) {
    mapping.ori_cache_idx = 0;
    return mapping;
  }
  if (ratio == 4) {
    mapping.cmp_cache_idx = 0;
    mapping.index_cache_idx = 1;
    mapping.ori_cache_idx = 2;
    mapping.kv_state_cache_idx = 3;
    mapping.score_state_cache_idx = 4;
    mapping.index_kv_state_cache_idx = 5;
    mapping.index_score_state_cache_idx = 6;
    return mapping;
  }
  if (ratio == 128) {
    mapping.cmp_cache_idx = 0;
    mapping.ori_cache_idx = 1;
    mapping.kv_state_cache_idx = 2;
    mapping.score_state_cache_idx = 3;
  }
  return mapping;
}

torch::Tensor build_decode_slot_mapping(const torch::Tensor& cache_block_table,
                                        const DSAMetadata& dsa,
                                        int64_t compress_ratio,
                                        int64_t block_size,
                                        const torch::Device& device) {
  const int64_t batch_size = static_cast<int64_t>(dsa.start_pos_vec.size());
  std::vector<torch::Tensor> per_seq_slots;
  per_seq_slots.reserve(static_cast<size_t>(batch_size));
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_begin = dsa.query_start_offsets[seq_idx];
    const int64_t q_end = dsa.query_start_offsets[seq_idx + 1];
    if (q_end <= q_begin) {
      continue;
    }
    torch::Tensor seq_positions = dsa.input_positions.slice(0, q_begin, q_end);
    torch::Tensor compressed_row = seq_positions / compress_ratio;
    torch::Tensor block_col =
        (compressed_row / block_size).to(torch::kInt64).unsqueeze(0);
    torch::Tensor block_offset = compressed_row % block_size;
    torch::Tensor block_id =
        cache_block_table[seq_idx].unsqueeze(0).gather(1, block_col).squeeze(0);
    per_seq_slots.emplace_back(
        (block_id * block_size + block_offset).to(torch::kInt32));
  }
  if (per_seq_slots.empty()) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  }
  return torch::cat(per_seq_slots, 0).to(device);
}

std::vector<int64_t> offsets_from_lens(const std::vector<int64_t>& q_lens) {
  std::vector<int64_t> offsets{0};
  offsets.reserve(q_lens.size() + 1);
  for (int64_t q_len : q_lens) {
    offsets.emplace_back(offsets.back() + q_len);
  }
  return offsets;
}

std::vector<int64_t> token_positions(const std::vector<int64_t>& start_pos,
                                     const std::vector<int64_t>& q_lens) {
  std::vector<int64_t> positions;
  int64_t total_tokens = 0;
  for (int64_t q_len : q_lens) {
    total_tokens += q_len;
  }
  positions.reserve(static_cast<size_t>(total_tokens));
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
      positions.emplace_back(start_pos[seq_idx] + token_idx);
    }
  }
  return positions;
}

torch::Tensor positions_tensor(const std::vector<int64_t>& positions,
                               const torch::Device& device) {
  std::vector<int32_t> values;
  values.reserve(positions.size());
  for (int64_t position : positions) {
    values.emplace_back(static_cast<int32_t>(position));
  }
  return torch::tensor(
      values, torch::TensorOptions().dtype(torch::kInt32).device(device));
}

std::vector<int64_t> compressed_positions(const std::vector<int64_t>& start_pos,
                                          const std::vector<int64_t>& q_lens,
                                          int64_t ratio) {
  std::vector<int64_t> positions;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      if ((pos + 1) % ratio == 0) {
        positions.emplace_back(pos + 1 - ratio);
      }
    }
  }
  return positions;
}

torch::Tensor rows_from_table(const torch::Tensor& table,
                              const std::vector<int64_t>& positions,
                              int64_t rope_dim) {
  if (positions.empty()) {
    return torch::empty({0, rope_dim}, table.options());
  }
  torch::Tensor index = torch::tensor(
      positions,
      torch::TensorOptions().dtype(torch::kInt64).device(table.device()));
  return table.index_select(/*dim=*/0, index);
}

torch::Tensor compressed_cos_sin_table(
    const test::Dsv4AttentionRefConfig& config,
    const torch::TensorOptions& options) {
  auto [sin_table, cos_table] =
      test::make_dsv4_freqs_ref(config.max_seq_len + 1,
                                config.rope_head_dim,
                                config.original_seq_len,
                                config.compress_rope_theta,
                                config.rope_factor,
                                config.beta_fast,
                                config.beta_slow,
                                options);
  return torch::cat({cos_table, sin_table}, /*dim=*/-1);
}

torch::Tensor default_cos_sin_table(const test::Dsv4AttentionRefConfig& config,
                                    const torch::TensorOptions& options) {
  auto [sin_table, cos_table] =
      test::make_dsv4_freqs_ref(config.max_seq_len + 1,
                                config.rope_head_dim,
                                /*original_seq_len=*/0,
                                config.rope_theta,
                                config.rope_factor,
                                config.beta_fast,
                                config.beta_slow,
                                options);
  return torch::cat({cos_table, sin_table}, /*dim=*/-1);
}

}  // namespace

class DeepseekV4AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device torch_device(Platform::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();
    device_ = torch_device;
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(device_)
                   .requires_grad(false);
    cpu_options_ =
        torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
    parallel_args_ = std::make_unique<ParallelArgs>(
        test::create_default_parallel_args(process_group_));
  }

  test::Dsv4AttentionRefConfig make_config(int64_t compress_ratio) const {
    test::Dsv4AttentionRefConfig config;
    config.hidden_dim = 32;
    config.q_lora_rank = 16;
    config.n_heads = 4;
    config.head_dim = 512;
    config.rope_head_dim = 64;
    config.o_groups = 2;
    config.o_lora_rank = 8;
    config.window_size = 4;
    config.compress_ratio = compress_ratio;
    config.index_n_heads = 32;
    config.index_head_dim = 128;
    config.index_topk = 3;
    config.max_seq_len = 260;
    config.norm_eps = 1e-6;
    config.rope_theta = 10000.0;
    config.compress_rope_theta = 40000.0;
    config.rope_factor = 40.0;
    config.original_seq_len = 128;
    config.beta_fast = 32;
    config.beta_slow = 1;
    return config;
  }

  ModelArgs make_model_args(const test::Dsv4AttentionRefConfig& config) const {
    ModelArgs args;
    args.hidden_size() = config.hidden_dim;
    args.head_dim() = config.head_dim;
    args.n_heads() = config.n_heads;
    args.n_kv_heads() = 1;
    args.q_lora_rank() = config.q_lora_rank;
    args.rope_head_dim() = config.rope_head_dim;
    args.o_groups() = config.o_groups;
    args.o_lora_rank() = config.o_lora_rank;
    args.compress_ratios() = {static_cast<int32_t>(config.compress_ratio)};
    args.window_size() = config.window_size;
    args.index_n_heads() = config.index_n_heads;
    args.index_head_dim() = config.index_head_dim;
    args.index_topk() = config.index_topk;
    args.rms_norm_eps() = static_cast<float>(config.norm_eps);
    return args;
  }

  test::Dsv4CompressorRefWeights make_compressor_weights(
      const std::string& prefix,
      int64_t compress_ratio,
      int64_t hidden_dim,
      int64_t head_dim) const {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    return {seeded(prefix + ".wkv",
                   {coff * head_dim, hidden_dim},
                   torch::kBFloat16,
                   torch::kCPU),
            seeded(prefix + ".wgate",
                   {coff * head_dim, hidden_dim},
                   torch::kBFloat16,
                   torch::kCPU),
            test::seeded_tensor(
                prefix + ".norm", {head_dim}, torch::kFloat32, torch::kCPU) +
                0.5,
            seeded(prefix + ".ape",
                   {compress_ratio, coff * head_dim},
                   torch::kFloat32,
                   torch::kCPU)};
  }

  test::Dsv4AttentionRefWeights make_weights(
      const test::Dsv4AttentionRefConfig& config) const {
    const std::string prefix =
        "dsv4.attention.ref." + std::to_string(config.compress_ratio);
    test::Dsv4AttentionRefWeights weights;
    weights.wq_a = seeded(prefix + ".wq_a",
                          {config.q_lora_rank, config.hidden_dim},
                          torch::kBFloat16,
                          torch::kCPU);
    weights.q_norm = test::seeded_tensor(prefix + ".q_norm",
                                         {config.q_lora_rank},
                                         torch::kFloat32,
                                         torch::kCPU) +
                     0.5;
    weights.wq_b =
        seeded(prefix + ".wq_b",
               {config.n_heads * config.head_dim, config.q_lora_rank},
               torch::kBFloat16,
               torch::kCPU);
    weights.wkv = seeded(prefix + ".wkv",
                         {config.head_dim, config.hidden_dim},
                         torch::kBFloat16,
                         torch::kCPU);
    weights.kv_norm = test::seeded_tensor(prefix + ".kv_norm",
                                          {config.head_dim},
                                          torch::kFloat32,
                                          torch::kCPU) +
                      0.5;
    weights.wo_a = seeded(prefix + ".wo_a",
                          {config.o_groups * config.o_lora_rank,
                           config.n_heads * config.head_dim / config.o_groups},
                          torch::kBFloat16,
                          torch::kCPU);
    weights.wo_b =
        seeded(prefix + ".wo_b",
               {config.hidden_dim, config.o_groups * config.o_lora_rank},
               torch::kBFloat16,
               torch::kCPU);
    weights.attn_sink = seeded(
        prefix + ".attn_sink", {config.n_heads}, torch::kFloat32, torch::kCPU);
    if (config.compress_ratio == 4 || config.compress_ratio == 128) {
      weights.compressor = make_compressor_weights(prefix + ".compressor",
                                                   config.compress_ratio,
                                                   config.hidden_dim,
                                                   config.head_dim);
    }
    if (config.compress_ratio == 4) {
      weights.indexer.wq_b = seeded(
          prefix + ".indexer.wq_b",
          {config.index_n_heads * config.index_head_dim, config.q_lora_rank},
          torch::kBFloat16,
          torch::kCPU);
      weights.indexer.weights_proj =
          seeded(prefix + ".indexer.weights",
                 {config.index_n_heads, config.hidden_dim},
                 torch::kBFloat16,
                 torch::kCPU) +
          0.25;
      weights.indexer.compressor =
          make_compressor_weights(prefix + ".indexer.compressor",
                                  config.compress_ratio,
                                  config.hidden_dim,
                                  config.index_head_dim);
    }
    return weights;
  }

  StateDict make_state_dict(const test::Dsv4AttentionRefWeights& weights,
                            int64_t compress_ratio) const {
    std::unordered_map<std::string, torch::Tensor> tensors;
    tensors["wq_a.weight"] = weights.wq_a.to(device_);
    tensors["q_norm.weight"] = weights.q_norm.to(device_);
    tensors["wq_b.weight"] = weights.wq_b.to(device_);
    tensors["wkv.weight"] = weights.wkv.to(device_);
    tensors["kv_norm.weight"] = weights.kv_norm.to(device_);
    tensors["wo_a.weight"] = weights.wo_a.to(device_);
    tensors["wo_b.weight"] = weights.wo_b.to(device_);
    tensors["attn_sink"] = weights.attn_sink.to(device_);
    if (compress_ratio == 4 || compress_ratio == 128) {
      tensors["compressor.wkv.weight"] = weights.compressor.wkv.to(device_);
      tensors["compressor.wgate.weight"] = weights.compressor.wgate.to(device_);
      tensors["compressor.norm.weight"] = weights.compressor.norm.to(device_);
      tensors["compressor.ape"] = weights.compressor.ape.to(device_);
    }
    if (compress_ratio == 4) {
      tensors["indexer.wq_b.weight"] = weights.indexer.wq_b.to(device_);
      tensors["indexer.weights_proj.weight"] =
          weights.indexer.weights_proj.to(device_);
      tensors["indexer.compressor.wkv.weight"] =
          weights.indexer.compressor.wkv.to(device_);
      tensors["indexer.compressor.wgate.weight"] =
          weights.indexer.compressor.wgate.to(device_);
      tensors["indexer.compressor.norm.weight"] =
          weights.indexer.compressor.norm.to(device_);
      tensors["indexer.compressor.ape"] =
          weights.indexer.compressor.ape.to(device_);
    }
    return StateDict(tensors);
  }

  KVCache make_kv_cache(int64_t batch_size,
                        const test::Dsv4AttentionRefConfig& config) const {
    const int64_t compressed_len =
        config.compress_ratio > 1
            ? config.max_seq_len / config.compress_ratio + 1
            : 1;
    const int64_t swa_blocks =
        (config.max_seq_len + kBlockSize - 1) / kBlockSize;
    const int64_t compressed_blocks =
        (compressed_len + kBlockSize - 1) / kBlockSize;
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = torch::zeros(
        {batch_size * swa_blocks, 1, kBlockSize, config.head_dim}, options_);
    if (config.compress_ratio == 4 || config.compress_ratio == 128) {
      tensors.key_cache = torch::zeros(
          {batch_size * compressed_blocks, 1, kBlockSize, config.head_dim},
          options_);
      const int64_t coff = config.compress_ratio == 4 ? 2 : 1;
      tensors.compress_kv_state = torch::zeros(
          {batch_size, coff * config.compress_ratio, coff * config.head_dim},
          options_.dtype(torch::kFloat32));
      tensors.compress_score_state = torch::full(
          {batch_size, coff * config.compress_ratio, coff * config.head_dim},
          -std::numeric_limits<float>::infinity(),
          options_.dtype(torch::kFloat32));
    }
    if (config.compress_ratio == 4) {
      tensors.index_cache = torch::zeros({batch_size * compressed_blocks,
                                          1,
                                          kBlockSize,
                                          config.index_head_dim},
                                         options_);
      tensors.compress_index_kv_state =
          torch::zeros({batch_size, 8, 2 * config.index_head_dim},
                       options_.dtype(torch::kFloat32));
      tensors.compress_index_score_state =
          torch::full({batch_size, 8, 2 * config.index_head_dim},
                      -std::numeric_limits<float>::infinity(),
                      options_.dtype(torch::kFloat32));
    }
    return KVCache(tensors);
  }

  torch::Tensor make_paged_table(int64_t batch_size,
                                 int64_t token_capacity) const {
    const int64_t blocks_per_seq =
        (token_capacity + kBlockSize - 1) / kBlockSize;
    std::vector<int32_t> values;
    values.reserve(static_cast<size_t>(batch_size * blocks_per_seq));
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      for (int64_t col = 0; col < blocks_per_seq; ++col) {
        values.emplace_back(
            static_cast<int32_t>(seq_idx * blocks_per_seq + col));
      }
    }
    return torch::tensor(
               values,
               torch::TensorOptions().dtype(torch::kInt32).device(device_))
        .view({batch_size, blocks_per_seq});
  }

  torch::Tensor make_state_table(int64_t batch_size) const {
    std::vector<int32_t> values;
    values.reserve(static_cast<size_t>(batch_size));
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      values.emplace_back(static_cast<int32_t>(seq_idx));
    }
    return torch::tensor(
               values,
               torch::TensorOptions().dtype(torch::kInt32).device(device_))
        .view({batch_size, 1});
  }

  torch::Tensor make_slots(const std::vector<int64_t>& start_pos,
                           const std::vector<int64_t>& q_lens,
                           int64_t ratio,
                           int64_t token_capacity) const {
    const int64_t blocks_per_seq =
        (token_capacity + kBlockSize - 1) / kBlockSize;
    std::vector<int32_t> slots;
    for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
         ++seq_idx) {
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        const int64_t pos = start_pos[seq_idx] + token_idx;
        if (ratio == 1 || (pos + 1) % ratio == 0) {
          const int64_t logical_pos = ratio == 1 ? pos : pos / ratio;
          const int64_t block_id =
              seq_idx * blocks_per_seq + logical_pos / kBlockSize;
          const int64_t block_offset = logical_pos % kBlockSize;
          slots.emplace_back(
              static_cast<int32_t>(block_id * kBlockSize + block_offset));
        }
      }
    }
    return torch::tensor(
        slots, torch::TensorOptions().dtype(torch::kInt32).device(device_));
  }

  void fill_c128_meta(const std::shared_ptr<DSAMetadata>& dsa,
                      const torch::Tensor& block_table,
                      const std::vector<int64_t>& start_pos,
                      const std::vector<int64_t>& q_lens) const {
    std::vector<int32_t> lens;
    int64_t total_tokens = 0;
    for (int64_t q_len : q_lens) {
      total_tokens += q_len;
    }
    lens.reserve(static_cast<size_t>(total_tokens));

    int64_t max_context_len = 0;
    for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
         ++seq_idx) {
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        const int64_t pos = start_pos[seq_idx] + token_idx;
        const int64_t context_len = (pos + 1) / 128;
        lens.emplace_back(static_cast<int32_t>(context_len));
        max_context_len = std::max(max_context_len, context_len);
      }
    }

    torch::TensorOptions int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    dsa->c128_attn_metadata.context_lens = torch::tensor(lens, int_options);
    dsa->c128_attn_metadata.max_context_len = max_context_len;

    const int64_t table_cols =
        std::max<int64_t>((max_context_len + kBlockSize - 1) / kBlockSize, 1);
    torch::Tensor table =
        torch::full({total_tokens, table_cols},
                    -1,
                    torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor src = block_table.cpu().to(torch::kInt32).contiguous();
    auto table_acc = table.accessor<int32_t, 2>();
    auto src_acc = src.accessor<int32_t, 2>();

    int64_t row = 0;
    for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
         ++seq_idx) {
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        const int64_t context_len =
            static_cast<int64_t>(lens[static_cast<size_t>(row)]);
        const int64_t blocks = (context_len + kBlockSize - 1) / kBlockSize;
        const int64_t cols = std::min<int64_t>(blocks, src.size(1));
        for (int64_t col = 0; col < cols; ++col) {
          table_acc[row][col] = src_acc[seq_idx][col];
        }
        ++row;
      }
    }
    dsa->c128_attn_metadata.block_table_for_attn = table.to(device_);
  }

  torch::Tensor make_window_table(const std::vector<int64_t>& kv_lens,
                                  const std::vector<int64_t>& first_blocks,
                                  int64_t live_blocks,
                                  int32_t expired_block_id = 0) const {
    int64_t max_blocks = 1;
    for (int64_t kv_len : kv_lens) {
      max_blocks = std::max(max_blocks, (kv_len + kBlockSize - 1) / kBlockSize);
    }
    const int64_t batch_size = static_cast<int64_t>(kv_lens.size());
    std::vector<int32_t> values(static_cast<size_t>(batch_size * max_blocks),
                                expired_block_id);
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      const int64_t seq_base = seq_idx * live_blocks;
      for (int64_t col = 0; col < live_blocks && col < max_blocks; ++col) {
        const int64_t logical_col = first_blocks[seq_idx] + col;
        if (logical_col >= max_blocks) {
          continue;
        }
        values[static_cast<size_t>(seq_idx * max_blocks + logical_col)] =
            static_cast<int32_t>(seq_base + col);
      }
    }
    torch::TensorOptions int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    return torch::tensor(values, int_options).view({batch_size, max_blocks});
  }

  torch::Tensor make_window_slots(const std::vector<int64_t>& start_pos,
                                  const std::vector<int64_t>& q_lens,
                                  const std::vector<int64_t>& first_blocks,
                                  int64_t live_blocks) const {
    std::vector<int32_t> slots;
    for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
         ++seq_idx) {
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        const int64_t pos = start_pos[seq_idx] + token_idx;
        const int64_t block_col = pos / kBlockSize - first_blocks[seq_idx];
        const int64_t block_offset = pos % kBlockSize;
        CHECK_GE(block_col, 0) << "SWA window does not cover write block.";
        CHECK_LT(block_col, live_blocks)
            << "SWA window does not cover write block.";
        const int64_t block_id = seq_idx * live_blocks + block_col;
        slots.emplace_back(
            static_cast<int32_t>(block_id * kBlockSize + block_offset));
      }
    }
    return torch::tensor(
        slots, torch::TensorOptions().dtype(torch::kInt32).device(device_));
  }

  void use_swa_window(const std::shared_ptr<DSAMetadata>& dsa,
                      const std::vector<int64_t>& first_blocks,
                      int64_t live_blocks,
                      int32_t expired_block_id = 0) const {
    std::vector<int64_t> kv_lens;
    kv_lens.reserve(static_cast<size_t>(dsa->kv_seq_lens.size(0)));
    torch::Tensor kv_lens_cpu =
        dsa->kv_seq_lens.to(torch::kCPU).to(torch::kInt64);
    for (int64_t seq_idx = 0; seq_idx < dsa->kv_seq_lens.size(0); ++seq_idx) {
      kv_lens.emplace_back(kv_lens_cpu[seq_idx].item<int64_t>());
    }
    dsa->block_tables[kLayerId][0] =
        make_window_table(kv_lens, first_blocks, live_blocks, expired_block_id);
    std::vector<int64_t> q_lens;
    q_lens.reserve(dsa->start_pos_vec.size());
    for (int64_t seq_idx = 0;
         seq_idx < static_cast<int64_t>(dsa->start_pos_vec.size());
         ++seq_idx) {
      q_lens.emplace_back(dsa->query_start_offsets[seq_idx + 1] -
                          dsa->query_start_offsets[seq_idx]);
    }
    dsa->slot_mappings[kLayerId][0] = make_window_slots(
        dsa->start_pos_vec, q_lens, first_blocks, live_blocks);
  }

  std::shared_ptr<DSAMetadata> make_metadata(
      const test::Dsv4AttentionRefConfig& config,
      const std::vector<int64_t>& start_pos,
      const std::vector<int64_t>& q_lens) {
    const int64_t batch_size = static_cast<int64_t>(q_lens.size());
    const int64_t compressed_len =
        config.compress_ratio > 1
            ? config.max_seq_len / config.compress_ratio + 1
            : 1;
    auto dsa = std::make_shared<DSAMetadata>();
    dsa->layer_id = kLayerId;
    dsa->start_pos_vec = start_pos;
    dsa->query_start_offsets = offsets_from_lens(q_lens);

    std::vector<int32_t> q_cu{0};
    std::vector<int32_t> kv_cu{0};
    std::vector<int32_t> q_seq;
    std::vector<int32_t> kv_seq;
    std::vector<int32_t> c4_seq;
    q_seq.reserve(q_lens.size());
    kv_seq.reserve(q_lens.size());
    int64_t total_q = 0;
    dsa->index_total_c4_len = 0;
    dsa->index_max_c4_len = 0;
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      total_q += q_lens[seq_idx];
      const int64_t kv_len = start_pos[seq_idx] + q_lens[seq_idx];
      q_cu.emplace_back(static_cast<int32_t>(total_q));
      kv_cu.emplace_back(kv_cu.back() + static_cast<int32_t>(kv_len));
      q_seq.emplace_back(static_cast<int32_t>(q_lens[seq_idx]));
      kv_seq.emplace_back(static_cast<int32_t>(kv_len));
      const int64_t c4_len = kv_len / 4;
      c4_seq.emplace_back(static_cast<int32_t>(c4_len));
      dsa->index_total_c4_len += c4_len;
      dsa->index_max_c4_len = std::max(dsa->index_max_c4_len, c4_len);
    }

    torch::TensorOptions int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    dsa->q_cu_seq_lens = torch::tensor(q_cu, int_options);
    dsa->kv_cu_seq_lens = torch::tensor(kv_cu, int_options);
    dsa->q_seq_lens = torch::tensor(q_seq, int_options);
    dsa->kv_seq_lens = torch::tensor(kv_seq, int_options);
    dsa->seq_lens_q = dsa->q_seq_lens;
    dsa->seq_lens = dsa->kv_seq_lens;
    dsa->actual_seq_lengths_query = dsa->q_cu_seq_lens;
    dsa->actual_seq_lengths_kv = dsa->kv_seq_lens;
    dsa->index_c4_seq_lens = torch::tensor(c4_seq, int_options);
    dsa->max_seqlen_q = torch::max(dsa->q_seq_lens);
    dsa->max_seqlen_kv = torch::max(dsa->kv_seq_lens);

    const int64_t window_left = std::max<int64_t>(config.window_size - 1, 0);
    std::vector<int32_t> swa_history_lens;
    std::vector<int32_t> swa_context_lens;
    swa_history_lens.reserve(q_lens.size());
    swa_context_lens.reserve(static_cast<size_t>(total_q));
    dsa->swa_start_pos_vec.clear();
    dsa->swa_start_pos_vec.reserve(q_lens.size());
    dsa->swa_max_history_len = 0;
    dsa->swa_max_context_len = 0;
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      const int64_t swa_start =
          std::max<int64_t>(0, start_pos[seq_idx] - window_left);
      const int64_t history_len = start_pos[seq_idx] - swa_start;
      dsa->swa_start_pos_vec.emplace_back(swa_start);
      swa_history_lens.emplace_back(static_cast<int32_t>(history_len));
      dsa->swa_max_history_len =
          std::max(dsa->swa_max_history_len, history_len);
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        const int64_t token_abs_pos = start_pos[seq_idx] + token_idx;
        const int64_t context_len = token_abs_pos - swa_start + 1;
        swa_context_lens.emplace_back(static_cast<int32_t>(context_len));
        dsa->swa_max_context_len =
            std::max(dsa->swa_max_context_len, context_len);
      }
    }
    dsa->swa_history_lens = torch::tensor(swa_history_lens, int_options);
    dsa->swa_context_lens = torch::tensor(swa_context_lens, int_options);

    const bool compressed =
        config.compress_ratio == 4 || config.compress_ratio == 128;
    const std::vector<int64_t> positions = token_positions(start_pos, q_lens);
    dsa->input_positions = positions_tensor(positions, device_);
    if (compressed) {
      const std::vector<int64_t> c_positions =
          compressed_positions(start_pos, q_lens, config.compress_ratio);
      if (config.compress_ratio == 4) {
        dsa->c4_pad_positions = positions_tensor(c_positions, device_);
      } else {
        dsa->c128_pad_positions = positions_tensor(c_positions, device_);
      }
    }

    torch::Tensor ori_table = make_paged_table(batch_size, config.max_seq_len);
    torch::Tensor ori_slots =
        make_slots(start_pos, q_lens, /*ratio=*/1, config.max_seq_len);
    std::vector<torch::Tensor> layer_tables;
    std::vector<torch::Tensor> layer_slots;
    caches_info_.clear();
    if (config.compress_ratio == 1) {
      layer_tables = {ori_table};
      layer_slots = {ori_slots};
      caches_info_ = {{{0,
                        DSACacheType::SLIDING_WINDOW,
                        1,
                        static_cast<int32_t>(kBlockSize)}}};
    } else {
      torch::Tensor cmp_table = make_paged_table(batch_size, compressed_len);
      torch::Tensor cmp_slots =
          make_slots(start_pos, q_lens, config.compress_ratio, compressed_len);
      const int64_t state_len =
          (config.compress_ratio == 4 ? 2 : 1) * config.compress_ratio;
      torch::Tensor state_table = make_state_table(batch_size);
      layer_tables = {cmp_table, ori_table, state_table, state_table};
      layer_slots = {cmp_slots, ori_slots, torch::Tensor(), torch::Tensor()};
      caches_info_ = {{{0,
                        DSACacheType::TOKEN,
                        static_cast<int32_t>(config.compress_ratio),
                        static_cast<int32_t>(kBlockSize)},
                       {1,
                        DSACacheType::SLIDING_WINDOW,
                        1,
                        static_cast<int32_t>(kBlockSize)},
                       {2,
                        DSACacheType::SLIDING_WINDOW,
                        1,
                        static_cast<int32_t>(state_len)},
                       {2,
                        DSACacheType::SLIDING_WINDOW,
                        1,
                        static_cast<int32_t>(state_len)}}};
      if (config.compress_ratio == 4) {
        torch::Tensor index_table =
            make_paged_table(batch_size, compressed_len);
        torch::Tensor index_slots =
            make_slots(start_pos, q_lens, /*ratio=*/4, compressed_len);
        torch::Tensor index_state_table = make_state_table(batch_size);
        layer_tables = {cmp_table,
                        index_table,
                        ori_table,
                        state_table,
                        state_table,
                        index_state_table,
                        index_state_table};
        layer_slots = {cmp_slots,
                       index_slots,
                       ori_slots,
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor(),
                       torch::Tensor()};
        caches_info_ = {
            {{0, DSACacheType::TOKEN, 4, static_cast<int32_t>(kBlockSize)},
             {0, DSACacheType::TOKEN, 4, static_cast<int32_t>(kBlockSize)},
             {1,
              DSACacheType::SLIDING_WINDOW,
              1,
              static_cast<int32_t>(kBlockSize)},
             {2, DSACacheType::SLIDING_WINDOW, 1, 8},
             {2, DSACacheType::SLIDING_WINDOW, 1, 8},
             {3, DSACacheType::SLIDING_WINDOW, 1, 8},
             {3, DSACacheType::SLIDING_WINDOW, 1, 8}}};
      }
    }
    dsa->block_tables = {layer_tables};
    dsa->slot_mappings = {layer_slots};
    if (config.compress_ratio == 128) {
      fill_c128_meta(dsa, layer_tables[0], start_pos, q_lens);
    }
    dsa->caches_info = &caches_info_;
    dsa->cmp_slots_dict[config.compress_ratio] =
        build_decode_slot_mapping(layer_tables[0],  // cmp_table
                                  *dsa,
                                  config.compress_ratio,
                                  kBlockSize,
                                  options_.device());
    return dsa;
  }

  AttentionMetadata make_attention_metadata(
      const std::shared_ptr<DSAMetadata>& dsa,
      bool is_prefill,
      bool is_chunked_prefill) const {
    AttentionMetadata attn_metadata;
    attn_metadata.is_prefill = is_prefill;
    attn_metadata.is_chunked_prefill = is_chunked_prefill;
    attn_metadata.is_dummy = false;
    attn_metadata.q_cu_seq_lens = dsa->q_cu_seq_lens;
    attn_metadata.kv_cu_seq_lens = dsa->kv_cu_seq_lens;
    attn_metadata.q_seq_lens = dsa->q_seq_lens;
    attn_metadata.kv_seq_lens = dsa->kv_seq_lens;
    attn_metadata.max_query_len = dsa->q_seq_lens.max().item<int64_t>();
    attn_metadata.max_seq_len = dsa->kv_seq_lens.max().item<int64_t>();
    attn_metadata.total_kv_len = dsa->kv_seq_lens.sum().item<int64_t>();
    attn_metadata.compute_dtype = "float";
    attn_metadata.dsa_metadata = dsa;
    return attn_metadata;
  }

  KVCache make_window_cache(KVCache& source,
                            int64_t batch_size,
                            const test::Dsv4AttentionRefConfig& config,
                            const std::vector<int64_t>& first_blocks,
                            int64_t live_blocks) const {
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache = torch::zeros(
        {batch_size * live_blocks, 1, kBlockSize, config.head_dim}, options_);
    torch::Tensor source_swa = source.get_swa_cache();
    const int64_t source_blocks = source_swa.size(0) / batch_size;
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      for (int64_t col = 0; col < live_blocks; ++col) {
        const int64_t src_block = first_blocks[seq_idx] + col;
        if (src_block >= source_blocks) {
          continue;
        }
        tensors.swa_cache[seq_idx * live_blocks + col].copy_(
            source_swa[seq_idx * source_blocks + src_block]);
      }
    }
    tensors.key_cache = source.get_k_cache();
    tensors.index_cache = source.get_index_cache();
    tensors.indexer_cache_scale =
        source.get_indexer_cache_scale().value_or(torch::Tensor());
    tensors.compress_kv_state = source.get_compress_kv_state();
    tensors.compress_score_state = source.get_compress_score_state();
    tensors.compress_index_kv_state = source.get_compress_index_kv_state();
    tensors.compress_index_score_state =
        source.get_compress_index_score_state();
    return KVCache(tensors);
  }

  KVCache make_swa_burst_cache(
      KVCache& source,
      const test::Dsv4AttentionRefConfig& config,
      int64_t total_blocks,
      const std::vector<int64_t>& source_blocks,
      const std::vector<int64_t>& target_blocks) const {
    CHECK_EQ(source_blocks.size(), target_blocks.size());
    DeepSeekV4KVCacheTensors tensors;
    tensors.swa_cache =
        torch::zeros({total_blocks, 1, kBlockSize, config.head_dim}, options_);
    torch::Tensor source_swa = source.get_swa_cache();
    for (size_t idx = 0; idx < source_blocks.size(); ++idx) {
      const int64_t source_block = source_blocks[idx];
      const int64_t target_block = target_blocks[idx];
      CHECK_GE(source_block, 0);
      CHECK_LT(source_block, source_swa.size(0));
      CHECK_GE(target_block, 0);
      CHECK_LT(target_block, total_blocks);
      tensors.swa_cache[target_block].copy_(source_swa[source_block]);
    }
    return KVCache(tensors);
  }

  torch::Tensor make_sparse_swa_table(
      int64_t kv_len,
      const std::vector<int64_t>& logical_blocks,
      const std::vector<int64_t>& cache_blocks) const {
    CHECK_EQ(logical_blocks.size(), cache_blocks.size());
    const int64_t max_blocks =
        std::max<int64_t>((kv_len + kBlockSize - 1) / kBlockSize, 1);
    std::vector<int32_t> values(static_cast<size_t>(max_blocks), -1);
    for (size_t idx = 0; idx < logical_blocks.size(); ++idx) {
      const int64_t logical_block = logical_blocks[idx];
      CHECK_GE(logical_block, 0);
      CHECK_LT(logical_block, max_blocks);
      values[static_cast<size_t>(logical_block)] =
          static_cast<int32_t>(cache_blocks[idx]);
    }
    return torch::tensor(
               values,
               torch::TensorOptions().dtype(torch::kInt32).device(device_))
        .view({1, max_blocks});
  }

  torch::Tensor make_swa_slots_from_table(
      const std::vector<int64_t>& start_pos,
      const std::vector<int64_t>& q_lens,
      const torch::Tensor& block_table) const {
    torch::Tensor table_cpu = block_table.cpu().to(torch::kInt32).contiguous();
    auto table_acc = table_cpu.accessor<int32_t, 2>();
    std::vector<int32_t> slots;
    for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(q_lens.size());
         ++seq_idx) {
      for (int64_t token_idx = 0; token_idx < q_lens[seq_idx]; ++token_idx) {
        const int64_t pos = start_pos[seq_idx] + token_idx;
        const int64_t logical_block = pos / kBlockSize;
        CHECK_LT(logical_block, table_cpu.size(1));
        const int32_t block_id = table_acc[seq_idx][logical_block];
        CHECK_GE(block_id, 0) << "SWA table does not cover write block.";
        const int64_t block_offset = pos % kBlockSize;
        slots.emplace_back(
            static_cast<int32_t>(block_id * kBlockSize + block_offset));
      }
    }
    return torch::tensor(
        slots, torch::TensorOptions().dtype(torch::kInt32).device(device_));
  }

  torch::Tensor run_actual_with_dsa(
      const test::Dsv4AttentionRefConfig& config,
      const test::Dsv4AttentionRefWeights& weights,
      KVCache& kv_cache,
      const torch::Tensor& hidden_cpu,
      const std::shared_ptr<DSAMetadata>& dsa,
      bool is_prefill,
      bool is_chunked_prefill) {
    ModelArgs args = make_model_args(config);
    QuantArgs quant_args;
    torch::Tensor cos_sin =
        default_cos_sin_table(config, cpu_options_).to(device_);
    torch::Tensor compressed_cos_sin =
        compressed_cos_sin_table(config, cpu_options_).to(device_);
    DeepseekV4Attention attention = DeepseekV4Attention(DeepseekV4AttentionImpl(
        args, quant_args, *parallel_args_, options_, kLayerId));
    if (cos_sin.defined()) {
      std::vector<torch::Tensor> chunks =
          cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      dsa->cos_table = chunks[0].contiguous();
      dsa->sin_table = chunks[1].contiguous();
      dsa->inverse_sin_table = -dsa->sin_table;
    }
    if (compressed_cos_sin.defined()) {
      std::vector<torch::Tensor> chunks =
          compressed_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      dsa->compressed_cos_table = chunks[0].contiguous();
      dsa->compressed_sin_table = chunks[1].contiguous();
      dsa->compressed_inverse_sin_table = -dsa->compressed_sin_table;
    }
    const int64_t ratio =
        config.compress_ratio <= 1 ? 1 : config.compress_ratio;
    DSACacheMapping mapping = make_cache_mapping(ratio);
    if (!is_prefill && !is_chunked_prefill && ratio > 1) {
      torch::Tensor decode_slots = dsa->cmp_slots_dict[ratio];
      if (decode_slots.defined()) {
        dsa->slot_mappings[kLayerId][mapping.cmp_cache_idx] = decode_slots;
        if (ratio == 4) {
          dsa->slot_mappings[kLayerId][mapping.index_cache_idx] = decode_slots;
        }
      }
    }
    attention->set_cache_mapping(mapping);
    attention->load_state_dict(make_state_dict(weights, config.compress_ratio));
    torch::Tensor hidden = hidden_cpu.to(device_);
    AttentionMetadata attn_metadata =
        make_attention_metadata(dsa, is_prefill, is_chunked_prefill);
    auto [actual, lse] = attention->forward(attn_metadata, hidden, kv_cache);
    (void)lse;
    return actual.cpu();
  }

  torch::Tensor run_actual(const test::Dsv4AttentionRefConfig& config,
                           const test::Dsv4AttentionRefWeights& weights,
                           KVCache& kv_cache,
                           const torch::Tensor& hidden_cpu,
                           const std::vector<int64_t>& start_pos,
                           const std::vector<int64_t>& q_lens,
                           bool is_prefill,
                           bool is_chunked_prefill) {
    std::shared_ptr<DSAMetadata> dsa = make_metadata(config, start_pos, q_lens);
    return run_actual_with_dsa(config,
                               weights,
                               kv_cache,
                               hidden_cpu,
                               dsa,
                               is_prefill,
                               is_chunked_prefill);
  }

  test::Dsv4AttentionRefResult run_ref(
      const test::Dsv4AttentionRefConfig& config,
      const test::Dsv4AttentionRefWeights& weights,
      const torch::Tensor& hidden,
      const std::vector<int64_t>& start_pos,
      const std::vector<int64_t>& q_lens,
      test::Dsv4AttentionRefState state = test::Dsv4AttentionRefState()) const {
    if (!state.token_cache.defined()) {
      state = test::make_dsv4_attention_state_ref(
          static_cast<int64_t>(start_pos.size()), config, cpu_options_);
    }
    return test::dsv4_attention_ref(
        hidden, weights, state, start_pos, offsets_from_lens(q_lens), config);
  }

  void expect_close(const torch::Tensor& actual,
                    const torch::Tensor& expected) const {
    test::verify_tensor_close(actual.to(torch::kFloat32),
                              expected.to(torch::kFloat32),
                              /*rtol=*/2e-2,
                              /*atol=*/2e-2);
  }

  torch::Device device_{torch::kCPU};
  torch::TensorOptions options_;
  torch::TensorOptions cpu_options_;
  std::unique_ptr<ProcessGroup> process_group_;
  std::unique_ptr<ParallelArgs> parallel_args_;
  std::vector<std::vector<DSACacheInfo>> caches_info_;
};

TEST_F(DeepseekV4AttentionTest, Ratio1PrefillMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/1);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.r1.hidden",
                                {6, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  test::Dsv4AttentionRefResult expected =
      run_ref(config, weights, hidden, /*start_pos=*/{0}, /*q_lens=*/{6});
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden,
                                    /*start_pos=*/{0},
                                    /*q_lens=*/{6},
                                    /*is_prefill=*/true,
                                    /*is_chunked_prefill=*/false);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, RawRatio0PrefillMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/1);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.r0.hidden",
                                {5, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  const std::vector<int64_t> start_pos{0};
  const std::vector<int64_t> q_lens{5};
  test::Dsv4AttentionRefResult expected =
      run_ref(config, weights, hidden, start_pos, q_lens);
  std::shared_ptr<DSAMetadata> dsa = make_metadata(config, start_pos, q_lens);

  ModelArgs args = make_model_args(config);
  args.compress_ratios() = {0};
  QuantArgs quant_args;
  torch::Tensor cos_sin =
      default_cos_sin_table(config, cpu_options_).to(device_);
  torch::Tensor compressed_cos_sin =
      compressed_cos_sin_table(config, cpu_options_).to(device_);
  DeepseekV4Attention attention = DeepseekV4Attention(DeepseekV4AttentionImpl(
      args, quant_args, *parallel_args_, options_, kLayerId));
  attention->set_cache_mapping(make_cache_mapping(/*ratio=*/1));
  if (cos_sin.defined()) {
    std::vector<torch::Tensor> chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa->cos_table = chunks[0].contiguous();
    dsa->sin_table = chunks[1].contiguous();
    dsa->inverse_sin_table = -dsa->sin_table;
  }
  if (compressed_cos_sin.defined()) {
    std::vector<torch::Tensor> chunks =
        compressed_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa->compressed_cos_table = chunks[0].contiguous();
    dsa->compressed_sin_table = chunks[1].contiguous();
    dsa->compressed_inverse_sin_table = -dsa->compressed_sin_table;
  }
  attention->load_state_dict(make_state_dict(weights, /*compress_ratio=*/1));

  AttentionMetadata attn_metadata =
      make_attention_metadata(dsa,
                              /*is_prefill=*/true,
                              /*is_chunked_prefill=*/false);

  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  torch::Tensor hidden_device = hidden.to(device_);
  auto [actual, lse] =
      attention->forward(attn_metadata, hidden_device, kv_cache);
  (void)lse;
  expect_close(actual.cpu(), expected.output);
}

TEST_F(DeepseekV4AttentionTest, Ratio4FullPrefillMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/4);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.r4.hidden",
                                {8, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  test::Dsv4AttentionRefResult expected =
      run_ref(config, weights, hidden, /*start_pos=*/{0}, /*q_lens=*/{8});
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden,
                                    /*start_pos=*/{0},
                                    /*q_lens=*/{8},
                                    /*is_prefill=*/true,
                                    /*is_chunked_prefill=*/false);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, Ratio4ChunkedContinuationMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/4);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.chunk.hidden",
                                {8, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefState state = test::make_dsv4_attention_state_ref(
      /*batch_size=*/1, config, cpu_options_);
  test::Dsv4AttentionRefResult first = run_ref(config,
                                               weights,
                                               hidden.slice(0, 0, 5),
                                               /*start_pos=*/{0},
                                               /*q_lens=*/{5},
                                               state);
  run_actual(config,
             weights,
             kv_cache,
             hidden.slice(0, 0, 5),
             /*start_pos=*/{0},
             /*q_lens=*/{5},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);
  test::Dsv4AttentionRefResult expected = run_ref(config,
                                                  weights,
                                                  hidden.slice(0, 5, 8),
                                                  /*start_pos=*/{5},
                                                  /*q_lens=*/{3},
                                                  first.state);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden.slice(0, 5, 8),
                                    /*start_pos=*/{5},
                                    /*q_lens=*/{3},
                                    /*is_prefill=*/false,
                                    /*is_chunked_prefill=*/true);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest,
       Ratio4DecodeWithoutCompressedContextMatchesRef) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/4);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.r4.no_cmp.hidden",
                                {3, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefResult warmup = run_ref(config,
                                                weights,
                                                hidden.slice(0, 0, 2),
                                                /*start_pos=*/{0},
                                                /*q_lens=*/{2});
  run_actual(config,
             weights,
             kv_cache,
             hidden.slice(0, 0, 2),
             /*start_pos=*/{0},
             /*q_lens=*/{2},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);
  test::Dsv4AttentionRefResult expected = run_ref(config,
                                                  weights,
                                                  hidden.slice(0, 2, 3),
                                                  /*start_pos=*/{2},
                                                  /*q_lens=*/{1},
                                                  warmup.state);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden.slice(0, 2, 3),
                                    /*start_pos=*/{2},
                                    /*q_lens=*/{1},
                                    /*is_prefill=*/false,
                                    /*is_chunked_prefill=*/false);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, SwaWrapContinuationIgnoresExpiredBlocks) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/1);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  const int64_t warmup_len = 224;
  const int64_t chunk_len = 8;
  torch::Tensor hidden = seeded("dsv4.attn.swa.wrap.hidden",
                                {warmup_len + chunk_len, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);

  KVCache full_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefResult warmup =
      run_ref(config,
              weights,
              hidden.slice(/*dim=*/0, /*start=*/0, /*end=*/warmup_len),
              /*start_pos=*/{0},
              /*q_lens=*/{warmup_len});
  run_actual(config,
             weights,
             full_cache,
             hidden.slice(/*dim=*/0, /*start=*/0, /*end=*/warmup_len),
             /*start_pos=*/{0},
             /*q_lens=*/{warmup_len},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);

  const int64_t live_blocks = 3;
  const int64_t first_live_block = warmup_len / kBlockSize - 2;
  const std::vector<int64_t> first_blocks = {first_live_block};
  KVCache window_cache = make_window_cache(
      full_cache, /*batch_size=*/1, config, first_blocks, live_blocks);
  std::shared_ptr<DSAMetadata> dsa = make_metadata(config,
                                                   /*start_pos=*/{warmup_len},
                                                   /*q_lens=*/{chunk_len});
  use_swa_window(dsa,
                 first_blocks,
                 live_blocks,
                 /*expired_block_id=*/-1);

  test::Dsv4AttentionRefResult expected =
      run_ref(config,
              weights,
              hidden.slice(/*dim=*/0,
                           /*start=*/warmup_len,
                           /*end=*/warmup_len + chunk_len),
              /*start_pos=*/{warmup_len},
              /*q_lens=*/{chunk_len},
              warmup.state);
  torch::Tensor actual =
      run_actual_with_dsa(config,
                          weights,
                          window_cache,
                          hidden.slice(/*dim=*/0,
                                       /*start=*/warmup_len,
                                       /*end=*/warmup_len + chunk_len),
                          dsa,
                          /*is_prefill=*/false,
                          /*is_chunked_prefill=*/true);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, RuntimeSwaBurstTableMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/1);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  const int64_t warmup_len = 224;
  const int64_t chunk_len = 8;
  torch::Tensor hidden = seeded("dsv4.attn.swa.burst.hidden",
                                {warmup_len + chunk_len, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);

  KVCache full_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefResult warmup =
      run_ref(config,
              weights,
              hidden.slice(/*dim=*/0, /*start=*/0, /*end=*/warmup_len),
              /*start_pos=*/{0},
              /*q_lens=*/{warmup_len});
  run_actual(config,
             weights,
             full_cache,
             hidden.slice(/*dim=*/0, /*start=*/0, /*end=*/warmup_len),
             /*start_pos=*/{0},
             /*q_lens=*/{warmup_len},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);

  const int64_t swa_blocks_per_seq =
      get_swa_blocks_per_seq(config.window_size, kBlockSize);
  const int64_t burst_blocks = (chunk_len + kBlockSize - 1) / kBlockSize;
  const int64_t total_blocks =
      swa_blocks_per_seq + burst_blocks + /*max_seqs=*/1 + 2;
  const int64_t history_block = warmup_len / kBlockSize - 1;
  const int64_t burst_block = warmup_len / kBlockSize;
  KVCache burst_cache = make_swa_burst_cache(full_cache,
                                             config,
                                             total_blocks,
                                             /*source_blocks=*/{history_block},
                                             /*target_blocks=*/{0});

  std::shared_ptr<DSAMetadata> dsa = make_metadata(config,
                                                   /*start_pos=*/{warmup_len},
                                                   /*q_lens=*/{chunk_len});
  torch::Tensor block_table =
      make_sparse_swa_table(/*kv_len=*/warmup_len + chunk_len,
                            /*logical_blocks=*/{history_block, burst_block},
                            /*cache_blocks=*/{0, 1});
  dsa->block_tables[kLayerId][0] = block_table;
  dsa->slot_mappings[kLayerId][0] =
      make_swa_slots_from_table(dsa->start_pos_vec, {chunk_len}, block_table);

  test::Dsv4AttentionRefResult expected =
      run_ref(config,
              weights,
              hidden.slice(/*dim=*/0,
                           /*start=*/warmup_len,
                           /*end=*/warmup_len + chunk_len),
              /*start_pos=*/{warmup_len},
              /*q_lens=*/{chunk_len},
              warmup.state);
  torch::Tensor actual =
      run_actual_with_dsa(config,
                          weights,
                          burst_cache,
                          hidden.slice(/*dim=*/0,
                                       /*start=*/warmup_len,
                                       /*end=*/warmup_len + chunk_len),
                          dsa,
                          /*is_prefill=*/false,
                          /*is_chunked_prefill=*/true);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest,
       Ratio128DecodeWithoutCompressedContextMatchesRef) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/128);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.c128.no_cmp.hidden",
                                {11, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefResult warmup = run_ref(config,
                                                weights,
                                                hidden.slice(0, 0, 10),
                                                /*start_pos=*/{0},
                                                /*q_lens=*/{10});
  run_actual(config,
             weights,
             kv_cache,
             hidden.slice(0, 0, 10),
             /*start_pos=*/{0},
             /*q_lens=*/{10},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);
  test::Dsv4AttentionRefResult expected = run_ref(config,
                                                  weights,
                                                  hidden.slice(0, 10, 11),
                                                  /*start_pos=*/{10},
                                                  /*q_lens=*/{1},
                                                  warmup.state);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden.slice(0, 10, 11),
                                    /*start_pos=*/{10},
                                    /*q_lens=*/{1},
                                    /*is_prefill=*/false,
                                    /*is_chunked_prefill=*/false);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, Ratio128DecodeBoundaryMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/128);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.decode.hidden",
                                {128, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefResult warmup = run_ref(config,
                                                weights,
                                                hidden.slice(0, 0, 127),
                                                /*start_pos=*/{0},
                                                /*q_lens=*/{127});
  run_actual(config,
             weights,
             kv_cache,
             hidden.slice(0, 0, 127),
             /*start_pos=*/{0},
             /*q_lens=*/{127},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);
  test::Dsv4AttentionRefResult expected = run_ref(config,
                                                  weights,
                                                  hidden.slice(0, 127, 128),
                                                  /*start_pos=*/{127},
                                                  /*q_lens=*/{1},
                                                  warmup.state);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden.slice(0, 127, 128),
                                    /*start_pos=*/{127},
                                    /*q_lens=*/{1},
                                    /*is_prefill=*/false,
                                    /*is_chunked_prefill=*/false);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, Ratio128SecondCompressedRowMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/128);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.c128.second.hidden",
                                {256, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  KVCache kv_cache = make_kv_cache(/*batch_size=*/1, config);
  test::Dsv4AttentionRefResult warmup = run_ref(config,
                                                weights,
                                                hidden.slice(0, 0, 255),
                                                /*start_pos=*/{0},
                                                /*q_lens=*/{255});
  run_actual(config,
             weights,
             kv_cache,
             hidden.slice(0, 0, 255),
             /*start_pos=*/{0},
             /*q_lens=*/{255},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);
  test::Dsv4AttentionRefResult expected = run_ref(config,
                                                  weights,
                                                  hidden.slice(0, 255, 256),
                                                  /*start_pos=*/{255},
                                                  /*q_lens=*/{1},
                                                  warmup.state);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    hidden.slice(0, 255, 256),
                                    /*start_pos=*/{255},
                                    /*q_lens=*/{1},
                                    /*is_prefill=*/false,
                                    /*is_chunked_prefill=*/false);
  expect_close(actual, expected.output);
}

TEST_F(DeepseekV4AttentionTest, VariableBatchContinuationMatchesReference) {
  test::Dsv4AttentionRefConfig config = make_config(/*compress_ratio=*/4);
  test::Dsv4AttentionRefWeights weights = make_weights(config);
  torch::Tensor hidden = seeded("dsv4.attn.varlen.hidden",
                                {8, config.hidden_dim},
                                torch::kBFloat16,
                                torch::kCPU);
  KVCache kv_cache = make_kv_cache(/*batch_size=*/2, config);
  torch::Tensor first_hidden =
      torch::cat({hidden.slice(0, 0, 2), hidden.slice(0, 4, 7)}, 0);
  torch::Tensor second_hidden =
      torch::cat({hidden.slice(0, 2, 4), hidden.slice(0, 7, 8)}, 0);
  test::Dsv4AttentionRefResult first = run_ref(config,
                                               weights,
                                               first_hidden,
                                               /*start_pos=*/{0, 0},
                                               /*q_lens=*/{2, 3});
  run_actual(config,
             weights,
             kv_cache,
             first_hidden,
             /*start_pos=*/{0, 0},
             /*q_lens=*/{2, 3},
             /*is_prefill=*/true,
             /*is_chunked_prefill=*/false);
  test::Dsv4AttentionRefResult expected = run_ref(config,
                                                  weights,
                                                  second_hidden,
                                                  /*start_pos=*/{2, 3},
                                                  /*q_lens=*/{2, 1},
                                                  first.state);
  torch::Tensor actual = run_actual(config,
                                    weights,
                                    kv_cache,
                                    second_hidden,
                                    /*start_pos=*/{2, 3},
                                    /*q_lens=*/{2, 1},
                                    /*is_prefill=*/false,
                                    /*is_chunked_prefill=*/true);
  expect_close(actual, expected.output);
}

}  // namespace layer
}  // namespace xllm
