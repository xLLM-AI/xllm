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

#include "layers/cuda/xattention.h"

#include <gtest/gtest.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "layers/cuda/flashinfer_workspace.h"
#include "layers/cuda/xattention_workspace.h"

namespace xllm::layer::test {
namespace {

struct DecodeTestInput {
  AttentionMetadata attn_metadata;
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
};

class XAttentionDecodeCompareTest : public ::testing::Test {
 protected:
  static constexpr int32_t kBatchSize = 4;
  static constexpr int32_t kBeamWidth = 128;
  static constexpr int32_t kNumHeads = 16;
  static constexpr int32_t kNumKvHeads = 8;
  static constexpr int32_t kHeadDim = 128;
  static constexpr int32_t kSharedSeqLen = 300;
  static constexpr int32_t kMaxDecodeStep = 2;
  static constexpr int32_t kCurrentStep = 1;

  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA is not available.";
    }

    device_ = torch::Device(torch::kCUDA, 0);
    flashinfer::FlashinferWorkspace::get_instance().initialize(device_);
    xattention::XAttentionWorkspace::get_instance().initialize(device_);
  }

  DecodeTestInput create_decode_test_input(torch::ScalarType dtype) {
    const int32_t total_beam = kBatchSize * kBeamWidth;
    const int32_t unshared_offset = kSharedSeqLen;
    const int32_t full_kv_len = unshared_offset + total_beam * kMaxDecodeStep;
    const int32_t kv_len_per_beam = kSharedSeqLen + (kCurrentStep + 1);

    auto float_opts = torch::TensorOptions().dtype(dtype).device(device_);
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(device_);

    DecodeTestInput input;
    auto& meta = input.attn_metadata;

    meta.is_prefill = false;
    meta.is_chunked_prefill = false;
    meta.is_dummy = false;
    meta.is_causal = false;
    meta.max_query_len = 1;
    meta.max_seq_len = kv_len_per_beam;
    meta.compute_dtype = dtype == torch::kBFloat16 ? "bfloat16" : "half";
    meta.enable_cuda_graph = false;

    meta.plan_info = std::make_shared<PlanInfo>();
    meta.plan_info->layer_id = 0;
    meta.shared_plan_info = std::make_shared<PlanInfo>();
    meta.shared_plan_info->layer_id = 0;
    meta.unshared_plan_info = std::make_shared<PlanInfo>();
    meta.unshared_plan_info->layer_id = 0;

    meta.full_k_cache =
        torch::zeros({full_kv_len, kNumKvHeads, kHeadDim}, float_opts);
    meta.full_v_cache =
        torch::zeros({full_kv_len, kNumKvHeads, kHeadDim}, float_opts);

    meta.full_k_cache.slice(0, 0, kSharedSeqLen)
        .copy_(
            torch::randn({kSharedSeqLen, kNumKvHeads, kHeadDim}, float_opts) *
            0.001);
    meta.full_v_cache.slice(0, 0, kSharedSeqLen)
        .copy_(
            torch::randn({kSharedSeqLen, kNumKvHeads, kHeadDim}, float_opts) *
            0.001);

    meta.unshared_k_cache =
        meta.full_k_cache.slice(0, unshared_offset, full_kv_len)
            .view({kBatchSize,
                   kBeamWidth,
                   kMaxDecodeStep,
                   kNumKvHeads,
                   kHeadDim});
    meta.unshared_v_cache =
        meta.full_v_cache.slice(0, unshared_offset, full_kv_len)
            .view({kBatchSize,
                   kBeamWidth,
                   kMaxDecodeStep,
                   kNumKvHeads,
                   kHeadDim});

    meta.unshared_k_cache.slice(2, 0, 1).copy_(
        torch::randn({kBatchSize, kBeamWidth, 1, kNumKvHeads, kHeadDim},
                     float_opts) *
        0.001);
    meta.unshared_v_cache.slice(2, 0, 1).copy_(
        torch::randn({kBatchSize, kBeamWidth, 1, kNumKvHeads, kHeadDim},
                     float_opts) *
        0.001);

    meta.block_table =
        torch::arange(total_beam, int_opts).view({total_beam, 1});
    meta.step_tensor = torch::tensor({kCurrentStep}, int_opts);

    std::vector<int32_t> paged_kv_indptr(total_beam + 1, 0);
    std::vector<int32_t> paged_kv_indices;
    paged_kv_indices.reserve(total_beam * kv_len_per_beam);

    int32_t cursor = 0;
    for (int32_t beam_id = 0; beam_id < total_beam; ++beam_id) {
      paged_kv_indptr[beam_id] = cursor;
      for (int32_t t = 0; t < kSharedSeqLen; ++t) {
        paged_kv_indices.push_back(t);
        ++cursor;
      }
      for (int32_t s = 0; s <= kCurrentStep; ++s) {
        paged_kv_indices.push_back(unshared_offset + beam_id * kMaxDecodeStep +
                                   s);
        ++cursor;
      }
      paged_kv_indptr[beam_id + 1] = cursor;
    }

    meta.paged_kv_indptr = torch::tensor(paged_kv_indptr, int_opts);
    meta.paged_kv_indices = torch::tensor(paged_kv_indices, int_opts);
    meta.paged_kv_last_page_len = torch::ones({total_beam}, int_opts);

    meta.kv_seq_lens = torch::full({total_beam}, kv_len_per_beam, int_opts);
    meta.q_seq_lens = torch::ones({total_beam}, int_opts);
    meta.q_cu_seq_lens = torch::arange(0, total_beam + 1, 1, int_opts);
    std::vector<int32_t> kv_cu_seq_lens(kBatchSize + 1, 0);
    for (int32_t i = 1; i <= kBatchSize; ++i) {
      kv_cu_seq_lens[i] = i * kSharedSeqLen;
    }
    meta.kv_cu_seq_lens = torch::tensor(kv_cu_seq_lens, int_opts);
    meta.qo_indptr = torch::arange(0, total_beam + 1, 1, int_opts);

    XAttentionTwoStageDecodeCache two_stage_cache;
    auto fp32_opts =
        torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    two_stage_cache.shared_lse =
        torch::zeros({total_beam, kNumHeads, 1}, fp32_opts);
    two_stage_cache.shared_o =
        torch::zeros({total_beam, kNumHeads, kHeadDim}, float_opts);
    two_stage_cache.unshared_lse =
        torch::zeros({total_beam, kNumHeads, 1}, fp32_opts);
    two_stage_cache.unshared_o =
        torch::zeros({total_beam, kNumHeads, kHeadDim}, float_opts);

    two_stage_cache.q_cu_seq_lens_shared =
        torch::arange(0, (kBatchSize + 1) * kBeamWidth, kBeamWidth, int_opts);
    two_stage_cache.qo_indptr_expanded =
        torch::arange(total_beam + 1, int_opts);
    two_stage_cache.paged_kv_indptr_expanded =
        torch::arange(total_beam + 1, int_opts);
    two_stage_cache.paged_kv_indices_expanded =
        torch::arange(total_beam, int_opts);
    two_stage_cache.paged_kv_last_page_len_expanded =
        torch::full({total_beam}, kCurrentStep + 1, int_opts);

    meta.xattention_two_stage_decode_cache = std::move(two_stage_cache);

    input.query =
        torch::randn({total_beam, kNumHeads * kHeadDim}, float_opts) * 0.001;
    input.key =
        torch::randn({total_beam, kNumKvHeads * kHeadDim}, float_opts) * 0.001;
    input.value =
        torch::randn({total_beam, kNumKvHeads * kHeadDim}, float_opts) * 0.001;

    return input;
  }

  torch::Tensor run_decode_once(DecodeTestInput& input, bool enable_two_stage) {
    RecConfig::get_instance().enable_xattention_one_stage(!enable_two_stage);
    SchedulerConfig::get_instance().max_tokens_per_batch(kSharedSeqLen);

    XAttentionImpl attention(
        /*num_heads=*/kNumHeads,
        /*head_size=*/kHeadDim,
        /*scale=*/1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        /*num_kv_heads=*/kNumKvHeads,
        /*sliding_window=*/-1);

    torch::Tensor output = torch::zeros_like(input.query);
    KVCache dummy_cache;

    auto result = attention.forward(input.attn_metadata,
                                    input.query,
                                    input.key,
                                    input.value,
                                    output,
                                    dummy_cache);

    torch::cuda::synchronize();
    return std::get<0>(result).clone();
  }

  void compare_single_and_two_stage(torch::ScalarType dtype,
                                    double atol,
                                    double rtol) {
    constexpr int64_t kSeed = 20260303;
    torch::manual_seed(kSeed);
    torch::cuda::manual_seed_all(kSeed);
    auto single_input = create_decode_test_input(dtype);
    torch::manual_seed(kSeed);
    torch::cuda::manual_seed_all(kSeed);
    auto two_stage_input = create_decode_test_input(dtype);

    two_stage_input.query.copy_(single_input.query);
    two_stage_input.key.copy_(single_input.key);
    two_stage_input.value.copy_(single_input.value);

    auto single_output =
        run_decode_once(single_input, /*enable_two_stage=*/false);
    auto two_stage_output =
        run_decode_once(two_stage_input, /*enable_two_stage=*/true);

    auto abs_diff =
        (single_output - two_stage_output).abs().to(torch::kFloat32);
    const double max_abs_diff = abs_diff.max().item<double>();
    const double mean_abs_diff = abs_diff.mean().item<double>();

    EXPECT_TRUE(torch::allclose(single_output, two_stage_output, rtol, atol))
        << "single-stage and two-stage decode outputs mismatch: "
        << "max_abs_diff=" << max_abs_diff
        << ", mean_abs_diff=" << mean_abs_diff << ", atol=" << atol
        << ", rtol=" << rtol;

    EXPECT_LT(max_abs_diff, atol)
        << "max_abs_diff exceeds threshold: "
        << "max_abs_diff=" << max_abs_diff << ", threshold=" << atol;
  }

  torch::Device device_{torch::kCPU};
};

TEST_F(XAttentionDecodeCompareTest, SingleVsTwoStageFp16) {
  compare_single_and_two_stage(torch::kFloat16,
                               /*atol=*/2e-3,
                               /*rtol=*/2e-3);
}

TEST_F(XAttentionDecodeCompareTest, SingleVsTwoStageBf16) {
  compare_single_and_two_stage(torch::kBFloat16,
                               /*atol=*/2e-2,
                               /*rtol=*/2e-2);
}

}  // namespace
}  // namespace xllm::layer::test
