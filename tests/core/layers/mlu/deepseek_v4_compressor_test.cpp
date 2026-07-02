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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "layers/common/dsa_metadata.h"
#include "layers/mlu/deepseek_v4/compressor.h"
#include "layers/mlu/deepseek_v4_ref_utils.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/linalg.h"

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kBlockSize = 16;
constexpr int64_t kStateBlockSize = 16;

struct CompressorConfig {
  int64_t hidden_dim = 16;
  int64_t head_dim = 512;
  int64_t rope_head_dim = 64;
  double norm_eps = 1e-6;
};

torch::Tensor seeded(const std::string& key,
                     torch::IntArrayRef shape,
                     torch::ScalarType dtype,
                     const torch::Device& device) {
  return (test::seeded_tensor(key, shape, dtype, device) - 0.5) * 0.2;
}

torch::Tensor make_paged_table(int64_t batch_size,
                               int64_t blocks_per_seq,
                               const torch::TensorOptions& options) {
  std::vector<int32_t> values;
  values.reserve(static_cast<size_t>(batch_size * blocks_per_seq));
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    for (int64_t block_idx = 0; block_idx < blocks_per_seq; ++block_idx) {
      values.emplace_back(
          static_cast<int32_t>(seq_idx * blocks_per_seq + block_idx));
    }
  }
  return torch::tensor(values, options).view({batch_size, blocks_per_seq});
}

torch::Tensor make_compressed_slots(const std::vector<int64_t>& start_pos,
                                    int64_t q_len,
                                    int64_t compress_ratio,
                                    int64_t blocks_per_seq,
                                    const torch::TensorOptions& options) {
  std::vector<int32_t> slots;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      if ((pos + 1) % compress_ratio != 0) {
        continue;
      }
      const int64_t logical_pos = pos / compress_ratio;
      const int64_t block_col = logical_pos / kBlockSize;
      const int64_t block_offset = logical_pos % kBlockSize;
      const int64_t slot =
          (seq_idx * blocks_per_seq + block_col) * kBlockSize + block_offset;
      slots.emplace_back(static_cast<int32_t>(slot));
    }
  }
  return torch::tensor(slots, options);
}

torch::Tensor make_decode_compressed_slots(
    const std::vector<int64_t>& start_pos,
    int64_t q_len,
    int64_t compress_ratio,
    int64_t blocks_per_seq,
    const torch::TensorOptions& options) {
  std::vector<int32_t> slots;
  slots.reserve(start_pos.size() * static_cast<size_t>(q_len));
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      const int64_t logical_pos = pos / compress_ratio;
      const int64_t block_col = logical_pos / kBlockSize;
      const int64_t block_offset = logical_pos % kBlockSize;
      const int64_t slot =
          (seq_idx * blocks_per_seq + block_col) * kBlockSize + block_offset;
      slots.emplace_back(static_cast<int32_t>(slot));
    }
  }
  return torch::tensor(slots, options);
}

std::vector<int64_t> make_compressed_positions(
    const std::vector<int64_t>& start_pos,
    int64_t q_len,
    int64_t compress_ratio) {
  std::vector<int64_t> positions;
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
      const int64_t pos = start_pos[seq_idx] + token_idx;
      if ((pos + 1) % compress_ratio != 0) {
        continue;
      }
      positions.emplace_back(pos + 1 - compress_ratio);
    }
  }
  return positions;
}

torch::Tensor gather_paged_state(const torch::Tensor& paged_state,
                                 const torch::Tensor& block_table,
                                 int64_t batch_size,
                                 int64_t state_len) {
  torch::Tensor dense = torch::empty(
      {batch_size, state_len, paged_state.size(2)}, paged_state.options());
  torch::Tensor block_table_cpu =
      block_table.to(torch::kCPU).to(torch::kInt64).contiguous();
  const int64_t block_size = paged_state.size(1);
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    for (int64_t row_idx = 0; row_idx < state_len; ++row_idx) {
      const int64_t block_col = row_idx / block_size;
      const int64_t block_offset = row_idx % block_size;
      const int64_t block_id =
          block_table_cpu[seq_idx][block_col].item<int64_t>();
      dense[seq_idx]
          .slice(/*dim=*/0, /*start=*/row_idx, /*end=*/row_idx + 1)
          .copy_(paged_state[block_id].slice(/*dim=*/0,
                                             /*start=*/block_offset,
                                             /*end=*/block_offset + 1));
    }
  }
  return dense;
}

void scatter_paged_state(const torch::Tensor& dense,
                         const torch::Tensor& paged_state,
                         const torch::Tensor& block_table,
                         int64_t state_len) {
  torch::Tensor block_table_cpu =
      block_table.to(torch::kCPU).to(torch::kInt64).contiguous();
  const int64_t batch_size = dense.size(0);
  const int64_t block_size = paged_state.size(1);
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    for (int64_t row_idx = 0; row_idx < state_len; ++row_idx) {
      const int64_t block_col = row_idx / block_size;
      const int64_t block_offset = row_idx % block_size;
      const int64_t block_id =
          block_table_cpu[seq_idx][block_col].item<int64_t>();
      paged_state[block_id]
          .slice(/*dim=*/0, /*start=*/block_offset, /*end=*/block_offset + 1)
          .copy_(dense[seq_idx].slice(
              /*dim=*/0, /*start=*/row_idx, /*end=*/row_idx + 1));
    }
  }
}

torch::Tensor gather_cache_rows(const torch::Tensor& paged_cache,
                                const torch::Tensor& slot_mapping,
                                int64_t head_dim) {
  if (slot_mapping.numel() == 0) {
    return torch::empty({0, head_dim}, paged_cache.options());
  }
  torch::Tensor dense = paged_cache.squeeze(/*dim=*/1).reshape({-1, head_dim});
  return dense.index_select(/*dim=*/0, slot_mapping.to(torch::kLong));
}

void verify_state_rows(const torch::Tensor& actual,
                       const torch::Tensor& expected,
                       int64_t seq_idx,
                       int64_t begin,
                       int64_t end) {
  if (begin == end) {
    return;
  }
  test::verify_tensor_close(actual[seq_idx]
                                .slice(/*dim=*/0, /*start=*/begin, /*end=*/end)
                                .to(torch::kFloat32),
                            expected[seq_idx]
                                .slice(/*dim=*/0, /*start=*/begin, /*end=*/end)
                                .to(torch::kFloat32),
                            /*rtol=*/7e-2,
                            /*atol=*/7e-2);
}

void verify_live_state(const torch::Tensor& actual,
                       const torch::Tensor& expected,
                       const std::vector<int64_t>& start_pos,
                       int64_t q_len,
                       int64_t compress_ratio) {
  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(start_pos.size());
       ++seq_idx) {
    const int64_t ctx_len = start_pos[seq_idx] + q_len;
    const int64_t remainder = ctx_len % compress_ratio;
    if (compress_ratio == 4 && ctx_len >= compress_ratio) {
      verify_state_rows(actual, expected, seq_idx, 0, compress_ratio);
      verify_state_rows(actual,
                        expected,
                        seq_idx,
                        compress_ratio,
                        compress_ratio + remainder);
      continue;
    }
    verify_state_rows(actual, expected, seq_idx, 0, remainder);
  }
}

}  // namespace

class DeepseekV4CompressorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device torch_device(Platform::type_torch(), 0);
    Device device(torch_device);
    device.set_seed();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(torch_device)
                   .requires_grad(false);
    int_options_ = options_.dtype(torch::kInt32);
  }

  std::unordered_map<std::string, torch::Tensor> make_weights(
      int64_t compress_ratio,
      bool rotate) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    std::unordered_map<std::string, torch::Tensor> weights;
    const std::string prefix = "deepseek_v4_compressor." +
                               std::to_string(compress_ratio) +
                               (rotate ? ".rotate" : ".plain");
    weights["wkv.weight"] =
        seeded(prefix + ".wkv",
               {coff * config_.head_dim, config_.hidden_dim},
               torch::kBFloat16,
               options_.device());
    weights["wgate.weight"] =
        seeded(prefix + ".wgate",
               {coff * config_.head_dim, config_.hidden_dim},
               torch::kBFloat16,
               options_.device());
    weights["norm.weight"] = test::seeded_tensor(prefix + ".norm",
                                                 {config_.head_dim},
                                                 torch::kFloat32,
                                                 options_.device()) +
                             0.5;
    weights["ape"] = seeded(prefix + ".ape",
                            {compress_ratio, coff * config_.head_dim},
                            torch::kFloat32,
                            options_.device());
    return weights;
  }

  test::Dsv4CompressorRefResult run_case(
      int64_t compress_ratio,
      int64_t batch_size,
      int64_t q_len,
      const std::vector<int64_t>& start_pos,
      bool rotate,
      torch::Tensor initial_kv_state = torch::Tensor(),
      torch::Tensor initial_score_state = torch::Tensor(),
      torch::Tensor initial_paged_state = torch::Tensor(),
      int64_t rope_padding_rows = 0) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t state_len = coff * compress_ratio;
    const int64_t state_dim = coff * config_.head_dim;
    const int64_t total_tokens = batch_size * q_len;
    std::unordered_map<std::string, torch::Tensor> weights =
        make_weights(compress_ratio, rotate);
    torch::Tensor hidden_states =
        seeded("deepseek_v4_compressor.hidden." + std::to_string(q_len),
               {total_tokens, config_.hidden_dim},
               torch::kBFloat16,
               options_.device());
    torch::Tensor kv_state;
    torch::Tensor score_state;
    torch::Tensor paged_state;
    if (initial_kv_state.defined()) {
      kv_state = initial_kv_state.clone();
      score_state = initial_score_state.clone();
      paged_state = initial_paged_state.clone();
    } else {
      kv_state = torch::zeros({batch_size, state_len, state_dim},
                              options_.dtype(torch::kFloat32));
      score_state = torch::zeros({batch_size, state_len, state_dim},
                                 options_.dtype(torch::kFloat32));
    }

    AttentionMetadata attn_metadata;
    attn_metadata.dsa_metadata = std::make_shared<DSAMetadata>();
    DSAMetadata& dsa_metadata = *attn_metadata.dsa_metadata;
    dsa_metadata.start_pos_vec = start_pos;
    dsa_metadata.query_start_offsets.reserve(batch_size + 1);
    std::vector<int32_t> q_cu_seq_lens;
    std::vector<int32_t> kv_cu_seq_lens;
    std::vector<int32_t> q_seq_lens;
    std::vector<int32_t> kv_seq_lens;
    std::vector<int32_t> input_positions;
    q_cu_seq_lens.reserve(static_cast<size_t>(batch_size + 1));
    kv_cu_seq_lens.reserve(static_cast<size_t>(batch_size + 1));
    q_seq_lens.reserve(static_cast<size_t>(batch_size));
    kv_seq_lens.reserve(static_cast<size_t>(batch_size));
    input_positions.reserve(static_cast<size_t>(total_tokens));
    q_cu_seq_lens.emplace_back(0);
    kv_cu_seq_lens.emplace_back(0);
    int64_t max_kv_len = 0;
    int64_t total_kv_len = 0;
    for (int64_t seq_idx = 0; seq_idx <= batch_size; ++seq_idx) {
      dsa_metadata.query_start_offsets.emplace_back(seq_idx * q_len);
      if (seq_idx == batch_size) {
        continue;
      }
      const int64_t kv_len = start_pos[seq_idx] + q_len;
      max_kv_len = std::max(max_kv_len, kv_len);
      total_kv_len += kv_len;
      q_seq_lens.emplace_back(static_cast<int32_t>(q_len));
      kv_seq_lens.emplace_back(static_cast<int32_t>(kv_len));
      q_cu_seq_lens.emplace_back(static_cast<int32_t>((seq_idx + 1) * q_len));
      kv_cu_seq_lens.emplace_back(static_cast<int32_t>(total_kv_len));
      for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
        input_positions.emplace_back(
            static_cast<int32_t>(start_pos[seq_idx] + token_idx));
      }
    }
    int64_t output_rows = 0;
    for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
      output_rows += (start_pos[seq_idx] + q_len) / compress_ratio -
                     start_pos[seq_idx] / compress_ratio;
    }
    std::vector<int64_t> compressed_pos =
        make_compressed_positions(start_pos, q_len, compress_ratio);
    const int64_t max_compressed_pos =
        compressed_pos.empty()
            ? 0
            : *std::max_element(compressed_pos.begin(), compressed_pos.end());
    auto [sin_table, cos_table] =
        test::make_dsv4_rope_ref(max_compressed_pos + 1 + rope_padding_rows,
                                 config_.rope_head_dim,
                                 options_);
    torch::Tensor compressed_positions =
        torch::tensor(compressed_pos, int_options_);
    torch::Tensor ref_sin =
        compressed_positions.numel() == 0
            ? torch::empty({0, config_.rope_head_dim}, options_)
            : sin_table.index_select(
                  /*dim=*/0, compressed_positions.to(torch::kLong));
    torch::Tensor ref_cos =
        compressed_positions.numel() == 0
            ? torch::empty({0, config_.rope_head_dim}, options_)
            : cos_table.index_select(
                  /*dim=*/0, compressed_positions.to(torch::kLong));
    dsa_metadata.c4_pad_positions = compressed_positions;
    dsa_metadata.c128_pad_positions = compressed_positions;
    dsa_metadata.input_positions = torch::tensor(input_positions, int_options_);

    const int64_t compressed_cache_len = std::max<int64_t>(
        1, (max_kv_len + compress_ratio - 1) / compress_ratio);
    const int64_t compressed_blocks =
        (compressed_cache_len + kBlockSize - 1) / kBlockSize;
    torch::Tensor compressed_kv_cache = torch::zeros(
        {batch_size * compressed_blocks, 1, kBlockSize, config_.head_dim},
        options_);
    torch::Tensor compressed_slot_mapping = make_compressed_slots(
        start_pos, q_len, compress_ratio, compressed_blocks, int_options_);
    bool has_prefix = false;
    for (int64_t pos : start_pos) {
      has_prefix = has_prefix || pos > 0;
    }
    const bool is_decode = q_len == 1 && has_prefix;
    torch::Tensor slot_mapping =
        is_decode ? make_decode_compressed_slots(start_pos,
                                                 q_len,
                                                 compress_ratio,
                                                 compressed_blocks,
                                                 int_options_)
                  : compressed_slot_mapping;
    attn_metadata.is_prefill = !is_decode && !has_prefix;
    attn_metadata.is_chunked_prefill = !is_decode && has_prefix;
    attn_metadata.is_dummy = false;
    attn_metadata.is_causal =
        attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
    attn_metadata.max_query_len = q_len;
    attn_metadata.max_seq_len = max_kv_len;
    attn_metadata.total_kv_len = total_kv_len;
    attn_metadata.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, int_options_);
    attn_metadata.kv_cu_seq_lens = torch::tensor(kv_cu_seq_lens, int_options_);
    attn_metadata.q_seq_lens = torch::tensor(q_seq_lens, int_options_);
    attn_metadata.kv_seq_lens = torch::tensor(kv_seq_lens, int_options_);
    attn_metadata.block_table =
        make_paged_table(batch_size, compressed_blocks, int_options_);
    attn_metadata.slot_mapping = slot_mapping;
    dsa_metadata.q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
    dsa_metadata.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
    dsa_metadata.q_seq_lens = attn_metadata.q_seq_lens;
    dsa_metadata.kv_seq_lens = attn_metadata.kv_seq_lens;
    dsa_metadata.seq_lens_q = attn_metadata.q_seq_lens;
    dsa_metadata.seq_lens = attn_metadata.kv_seq_lens;

    torch::Tensor hadamard;
    if (rotate) {
      hadamard = util::create_hadamard_matrix(config_.head_dim,
                                              torch::kFloat32,
                                              torch::Device(torch::kCPU),
                                              /*normalize=*/true)
                     .to(options_.device(), options_.dtype().toScalarType());
    }
    test::Dsv4CompressorRefWeights ref_weights{weights["wkv.weight"],
                                               weights["wgate.weight"],
                                               weights["norm.weight"],
                                               weights["ape"]};
    test::Dsv4CompressorRefConfig ref_config{compress_ratio,
                                             config_.head_dim,
                                             config_.rope_head_dim,
                                             rotate,
                                             config_.norm_eps};
    test::Dsv4CompressorRefResult expected =
        test::dsv4_compressor_ref(hidden_states,
                                  ref_weights,
                                  kv_state.clone(),
                                  score_state.clone(),
                                  start_pos,
                                  dsa_metadata.query_start_offsets,
                                  ref_sin,
                                  ref_cos,
                                  hadamard,
                                  ref_config);
    StateDict state_dict(weights);
    Compressor compressor = Compressor(CompressorImpl(compress_ratio,
                                                      config_.hidden_dim,
                                                      config_.head_dim,
                                                      config_.rope_head_dim,
                                                      rotate,
                                                      config_.norm_eps,
                                                      options_));
    compressor->load_state_dict(state_dict);

    int64_t max_len = std::max(max_kv_len, state_len);
    const int64_t state_blocks =
        (max_len + kStateBlockSize - 1) / kStateBlockSize + 2;
    torch::Tensor block_table =
        make_paged_table(batch_size, state_blocks, int_options_);
    // Owning merged state: [batch * state_blocks, kStateBlockSize,
    // 2 * state_dim] with kv in the [0, state_dim) lanes and score in the
    // [state_dim, 2 * state_dim) lanes. The compressor forward consumes this
    // owning tensor directly (the decode op checks state_cache[0].
    // is_contiguous(), so it must be the owning tensor, never a narrow view).
    if (!initial_kv_state.defined()) {
      paged_state = torch::zeros(
          {batch_size * state_blocks, kStateBlockSize, 2 * state_dim},
          options_.dtype(torch::kFloat32));
    }
    torch::Tensor actual = compressor->forward(attn_metadata,
                                               hidden_states,
                                               compressed_kv_cache,
                                               slot_mapping,
                                               paged_state,
                                               block_table,
                                               sin_table,
                                               cos_table);
    test::Dsv4CompressorRefResult actual_result;
    actual_result.output = actual.numel() == 0
                               ? gather_cache_rows(compressed_kv_cache,
                                                   compressed_slot_mapping,
                                                   config_.head_dim)
                               : actual;

    if (expected.output.numel() == 0) {
      EXPECT_EQ(actual_result.output.sizes(), expected.output.sizes());
    } else {
      test::verify_tensor_close(actual_result.output.to(torch::kFloat32),
                                expected.output.to(torch::kFloat32),
                                /*rtol=*/2e-2,
                                /*atol=*/2e-2);
    }
    actual_result.kv_state = expected.kv_state;
    actual_result.score_state = expected.score_state;
    actual_result.paged_state = paged_state;
    return actual_result;
  }

  CompressorConfig config_;
  torch::TensorOptions options_;
  torch::TensorOptions int_options_;
};

TEST_F(DeepseekV4CompressorTest, Ratio4PrefillWithRemainder) {
  run_case(/*compress_ratio=*/4,
           /*batch_size=*/1,
           /*q_len=*/9,
           /*start_pos=*/{0},
           /*rotate=*/false);
}

TEST_F(DeepseekV4CompressorTest, Ratio128PrefillWithRemainder) {
  run_case(/*compress_ratio=*/128,
           /*batch_size=*/1,
           /*q_len=*/130,
           /*start_pos=*/{0},
           /*rotate=*/false);
}

TEST_F(DeepseekV4CompressorTest, Ratio4DecodeAcrossBoundary) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/3,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 0);
  state = run_case(/*compress_ratio=*/4,
                   /*batch_size=*/1,
                   /*q_len=*/1,
                   /*start_pos=*/{3},
                   /*rotate=*/false,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio4DecodeMixedBoundaryBatch) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/2,
                                                 /*q_len=*/1,
                                                 /*start_pos=*/{0, 3},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio128ChunkedPrefillContinuesState) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/128,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/127,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 0);
  state = run_case(/*compress_ratio=*/128,
                   /*batch_size=*/1,
                   /*q_len=*/2,
                   /*start_pos=*/{127},
                   /*rotate=*/false,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio128ChunkedPrefillAlignedWindow) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/128,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/128,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 1);
  state = run_case(/*compress_ratio=*/128,
                   /*batch_size=*/1,
                   /*q_len=*/128,
                   /*start_pos=*/{128},
                   /*rotate=*/false,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, RotatePathMatchesReference) {
  run_case(/*compress_ratio=*/4,
           /*batch_size=*/1,
           /*q_len=*/8,
           /*start_pos=*/{0},
           /*rotate=*/true);
}

TEST_F(DeepseekV4CompressorTest, AcceptsPaddedCompressedRopeRows) {
  run_case(/*compress_ratio=*/4,
           /*batch_size=*/2,
           /*q_len=*/5,
           /*start_pos=*/{0, 2},
           /*rotate=*/false,
           /*initial_kv_state=*/torch::Tensor(),
           /*initial_score_state=*/torch::Tensor(),
           /*initial_paged_state=*/torch::Tensor(),
           /*rope_padding_rows=*/3);
}

TEST_F(DeepseekV4CompressorTest, Ratio4PrefillStartsMidFirstWindow) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/5,
                                                 /*start_pos=*/{2},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio4ChunkedPrefillAlignedOverlap) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/4,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 1);
  state = run_case(/*compress_ratio=*/4,
                   /*batch_size=*/1,
                   /*q_len=*/4,
                   /*start_pos=*/{4},
                   /*rotate=*/false,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio4ChunkedPrefillCompletesOverlap) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/6,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/false);
  EXPECT_EQ(state.output.size(0), 1);
  state = run_case(/*compress_ratio=*/4,
                   /*batch_size=*/1,
                   /*q_len=*/2,
                   /*start_pos=*/{6},
                   /*rotate=*/false,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio4ChunkedPrefillRotatesOverlap) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/6,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/true);
  EXPECT_EQ(state.output.size(0), 1);
  state = run_case(/*compress_ratio=*/4,
                   /*batch_size=*/1,
                   /*q_len=*/2,
                   /*start_pos=*/{6},
                   /*rotate=*/true,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

TEST_F(DeepseekV4CompressorTest, Ratio4ChunkedPrefillRotatesIndexerShape) {
  test::Dsv4CompressorRefResult state = run_case(/*compress_ratio=*/4,
                                                 /*batch_size=*/1,
                                                 /*q_len=*/5,
                                                 /*start_pos=*/{0},
                                                 /*rotate=*/true);
  EXPECT_EQ(state.output.size(0), 1);
  state = run_case(/*compress_ratio=*/4,
                   /*batch_size=*/1,
                   /*q_len=*/3,
                   /*start_pos=*/{5},
                   /*rotate=*/true,
                   state.kv_state,
                   state.score_state,
                   state.paged_state);
  EXPECT_EQ(state.output.size(0), 1);
}

}  // namespace layer
}  // namespace xllm
