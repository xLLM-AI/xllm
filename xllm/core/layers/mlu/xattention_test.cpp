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

#include "layers/mlu/xattention.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/mlu/attention.h"
#include "runtime/rec_beam_utils.h"

namespace xllm::layer::test {
namespace {

constexpr int64_t kNumHeads = 4;
constexpr int64_t kNumKvHeads = 2;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kMaxTokens = 20;
constexpr int64_t kBatchSize = 2;
constexpr int64_t kBeamWidth = 2;
constexpr int64_t kMaxDecodeSteps = 2;
constexpr float kPoison = 7.0f;

class ScopedRecFlags {
 public:
  ScopedRecFlags(int32_t max_decode_rounds, bool one_stage)
      : old_max_decode_rounds_(FLAGS_max_decode_rounds),
        old_one_stage_(FLAGS_enable_xattention_one_stage) {
    FLAGS_max_decode_rounds = max_decode_rounds;
    FLAGS_enable_xattention_one_stage = one_stage;
  }

  ~ScopedRecFlags() {
    FLAGS_max_decode_rounds = old_max_decode_rounds_;
    FLAGS_enable_xattention_one_stage = old_one_stage_;
  }

 private:
  int32_t old_max_decode_rounds_;
  bool old_one_stage_;
};

torch::TensorOptions cpu_float() {
  return torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
}

torch::TensorOptions cpu_int() {
  return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
}

torch::Tensor to_mlu(const torch::Tensor& tensor,
                     torch::ScalarType dtype = torch::kBFloat16) {
  return tensor.to(torch::Device("mlu:0"), dtype);
}

torch::Tensor attention_ref(const torch::Tensor& query,
                            const torch::Tensor& key,
                            const torch::Tensor& value,
                            float scale,
                            bool causal) {
  const int64_t groups = query.size(1) / key.size(1);
  auto expanded_k = key.repeat_interleave(groups, /*dim=*/1);
  auto expanded_v = value.repeat_interleave(groups, /*dim=*/1);
  auto q = query.permute({1, 0, 2});
  auto k = expanded_k.permute({1, 0, 2});
  auto v = expanded_v.permute({1, 0, 2});
  auto scores = torch::matmul(q, k.transpose(1, 2)) * scale;
  if (causal) {
    auto mask = torch::ones({query.size(0), key.size(0)},
                            torch::TensorOptions().dtype(torch::kBool))
                    .triu(/*diagonal=*/1);
    scores.masked_fill_(mask, -std::numeric_limits<float>::infinity());
  }
  return torch::matmul(torch::softmax(scores, -1), v).permute({1, 0, 2});
}

torch::Tensor packed_prefill_ref(const torch::Tensor& query,
                                 const torch::Tensor& key,
                                 const torch::Tensor& value,
                                 const std::vector<int64_t>& seq_lens,
                                 float scale) {
  std::vector<torch::Tensor> outputs;
  int64_t offset = 0;
  for (int64_t seq_len : seq_lens) {
    outputs.push_back(attention_ref(query.slice(0, offset, offset + seq_len),
                                    key.slice(0, offset, offset + seq_len),
                                    value.slice(0, offset, offset + seq_len),
                                    scale,
                                    /*causal=*/true));
    offset += seq_len;
  }
  return torch::cat(outputs, 0);
}

template <typename AttentionType>
struct XAttentionFixture {
  explicit XAttentionFixture(int64_t beam_width = kBeamWidth,
                             int64_t max_tokens = kMaxTokens,
                             int64_t num_heads = kNumHeads,
                             int64_t num_kv_heads = kNumKvHeads,
                             int64_t head_dim = kHeadDim,
                             int64_t max_decode_steps = kMaxDecodeSteps)
      : beam_width(beam_width),
        max_tokens(max_tokens),
        num_heads(num_heads),
        num_kv_heads(num_kv_heads),
        head_dim(head_dim),
        max_decode_steps(max_decode_steps),
        scale(1.0f / std::sqrt(static_cast<float>(head_dim))),
        attention(num_heads,
                  head_dim,
                  scale,
                  num_kv_heads,
                  /*sliding_window=*/-1) {}

  void init(const std::vector<int64_t>& lens, bool with_decode_cache = true) {
    prompt_lens = lens;
    batch_size = static_cast<int64_t>(lens.size());
    total_beam = batch_size * beam_width;
    total_prompt = 0;
    max_prompt = 0;
    std::vector<int32_t> cu_lens = {0};
    for (int64_t len : lens) {
      total_prompt += len;
      max_prompt = std::max(max_prompt, len);
      cu_lens.push_back(static_cast<int32_t>(total_prompt));
    }

    auto mlu_bf16 = torch::TensorOptions()
                        .dtype(torch::kBFloat16)
                        .device(torch::Device("mlu:0"));
    auto mlu_fp32 = mlu_bf16.dtype(torch::kFloat32);
    auto mlu_int = mlu_bf16.dtype(torch::kInt32);
    const int64_t full_len =
        max_tokens + (with_decode_cache ? total_beam * max_decode_steps : 0);
    full_k = torch::full({full_len, num_kv_heads, head_dim}, kPoison, mlu_bf16);
    full_v = torch::full({full_len, num_kv_heads, head_dim}, kPoison, mlu_bf16);

    meta.full_k_cache = full_k;
    meta.full_v_cache = full_v;
    meta.q_cu_seq_lens =
        torch::tensor(cu_lens, cpu_int()).to(torch::Device("mlu:0"));
    meta.kv_cu_seq_lens = meta.q_cu_seq_lens;
    meta.total_kv_len = total_prompt;
    meta.max_query_len = max_prompt;
    meta.max_seq_len = max_prompt;
    meta.compute_dtype = "float";
    meta.is_chunked_prefill = false;
    meta.is_dummy = false;

    if (with_decode_cache) {
      unshared_k = full_k.slice(0, max_tokens, full_len)
                       .view({batch_size,
                              beam_width,
                              num_kv_heads,
                              max_decode_steps,
                              head_dim});
      unshared_v = full_v.slice(0, max_tokens, full_len)
                       .view({batch_size,
                              beam_width,
                              num_kv_heads,
                              max_decode_steps,
                              head_dim});
      ASSERT_TRUE(unshared_k.is_contiguous());
      ASSERT_TRUE(unshared_v.is_contiguous());
      meta.unshared_k_cache = unshared_k;
      meta.unshared_v_cache = unshared_v;

      XAttentionTwoStageDecodeCache cache;
      cache.shared_o =
          torch::empty({total_beam, num_heads, head_dim}, mlu_bf16);
      cache.shared_lse = torch::empty({total_beam, num_heads, 1}, mlu_fp32);
      cache.unshared_o =
          torch::empty({total_beam, num_heads, head_dim}, mlu_bf16);
      cache.unshared_lse = torch::empty({total_beam, num_heads, 1}, mlu_fp32);
      cache.shared_lse_kernel = torch::empty({num_heads, total_beam}, mlu_fp32);
      cache.q_cu_seq_lens_shared =
          torch::arange(0, (batch_size + 1) * beam_width, beam_width, mlu_int);
      cache.decode_slot_mapping = torch::empty({total_beam}, mlu_int);
      cache.unshared_seq_lens = torch::empty({total_beam}, mlu_int);
      meta.xattention_two_stage_decode_cache = std::move(cache);
      meta.block_table =
          torch::arange(total_beam, mlu_int).view({total_beam, 1});
    }
  }

  torch::Tensor run_prefill(torch::Tensor& query,
                            torch::Tensor& key,
                            torch::Tensor& value) {
    meta.is_prefill = true;
    return std::get<0>(attention.forward(meta, query, key, value, dummy_cache));
  }

  torch::Tensor run_decode(torch::Tensor& query,
                           torch::Tensor& key,
                           torch::Tensor& value,
                           int64_t step) {
    meta.is_prefill = false;
    auto& cache = meta.xattention_two_stage_decode_cache.value();
    auto block_ids = meta.block_table.view({-1});
    cache.decode_slot_mapping =
        block_ids * max_decode_steps +
        torch::full({total_beam}, step, block_ids.options());
    cache.unshared_seq_lens =
        torch::full({total_beam}, step + 1, block_ids.options());
    return std::get<0>(attention.forward(meta, query, key, value, dummy_cache));
  }

  int64_t beam_width;
  int64_t max_tokens;
  int64_t num_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
  int64_t max_decode_steps;
  int64_t batch_size = 0;
  int64_t total_beam = 0;
  int64_t total_prompt = 0;
  int64_t max_prompt = 0;
  float scale;
  std::vector<int64_t> prompt_lens;
  AttentionType attention;
  AttentionMetadata meta;
  KVCache dummy_cache;
  torch::Tensor full_k;
  torch::Tensor full_v;
  torch::Tensor unshared_k;
  torch::Tensor unshared_v;
};

using ComponentFixture = XAttentionFixture<MluXAttentionImpl>;
using WrapperFixture = XAttentionFixture<AttentionImpl>;

ModelInputParams make_rec_params(const WrapperFixture& fixture,
                                 BatchForwardType forward_type) {
  auto mlu_int = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .device(torch::Device("mlu:0"));
  auto mlu_fp32 = mlu_int.dtype(torch::kFloat32);
  ModelInputParams params;
  params.batch_forward_type = forward_type;
  params.num_sequences =
      forward_type.is_decode() ? fixture.total_beam : fixture.batch_size;
  params.q_seq_lens = forward_type.is_decode()
                          ? torch::arange(fixture.total_beam + 1, mlu_int)
                          : fixture.meta.q_cu_seq_lens;
  params.kv_seq_lens = fixture.meta.kv_cu_seq_lens;
  params.q_max_seq_len = forward_type.is_decode() ? 1 : fixture.max_prompt;
  params.kv_max_seq_len = fixture.max_prompt;
  params.kv_seq_lens_vec = {0, 5, 13};
  params.block_tables = fixture.meta.block_table;
  params.new_cache_slots = torch::zeros(
      {forward_type.is_decode() ? fixture.total_beam : fixture.total_prompt},
      mlu_int);

  auto& rec = params.mutable_llmrec_params();
  rec.batch_size = fixture.batch_size;
  rec.beam_width = fixture.beam_width;
  rec.total_round = fixture.max_decode_steps + 1;
  rec.current_round_tensor = torch::zeros({1}, mlu_int);
  if (!forward_type.is_decode()) {
    return params;
  }

  const auto& cache = fixture.meta.xattention_two_stage_decode_cache.value();
  rec.two_stage_shared_lse = cache.shared_lse;
  rec.two_stage_shared_o = cache.shared_o;
  rec.two_stage_unshared_lse = cache.unshared_lse;
  rec.two_stage_unshared_o = cache.unshared_o;
  rec.two_stage_q_cu_seq_lens_shared = cache.q_cu_seq_lens_shared;
  rec.two_stage_shared_lse_kernel = cache.shared_lse_kernel;
  rec.two_stage_decode_slot_mapping = cache.decode_slot_mapping;
  rec.two_stage_unshared_seq_lens = cache.unshared_seq_lens;
  rec.unshared_k_caches = {fixture.unshared_k};
  rec.unshared_v_caches = {fixture.unshared_v};
  EXPECT_EQ(rec.two_stage_shared_lse_kernel.scalar_type(), torch::kFloat32);
  EXPECT_EQ(rec.two_stage_shared_lse_kernel.options().device(),
            mlu_fp32.device());
  return params;
}

TEST(MluRecParamsTest, ToMovesAllMluTwoStageFields) {
  LlmRecMultiRoundParams params;
  auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
  auto int32 = torch::TensorOptions().dtype(torch::kInt32);
  params.two_stage_shared_lse_kernel = torch::zeros({4, 6}, fp32);
  params.two_stage_decode_slot_mapping = torch::arange(6, int32);
  params.two_stage_unshared_seq_lens = torch::ones({6}, int32);

  auto moved = params.to(torch::Device("mlu:0"));
  EXPECT_EQ(moved.two_stage_shared_lse_kernel.device().type(),
            torch::DeviceType::PrivateUse1);
  EXPECT_EQ(moved.two_stage_decode_slot_mapping.device().type(),
            torch::DeviceType::PrivateUse1);
  EXPECT_EQ(moved.two_stage_unshared_seq_lens.device().type(),
            torch::DeviceType::PrivateUse1);
  EXPECT_FALSE(moved.two_stage_qo_indptr_expanded.defined());
}

TEST(MluAttentionMetadataBuilderTest, PrefillOmitsTwoStageDecodeCache) {
  ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
  WrapperFixture fixture;
  fixture.init({5, 8});
  auto params = make_rec_params(fixture, BatchForwardType::PREFILL);

  auto metadata = AttentionMetadataBuilder::build(params);

  EXPECT_FALSE(metadata.xattention_two_stage_decode_cache.has_value());
  EXPECT_EQ(metadata.total_kv_len, 13);
}

TEST(MluAttentionMetadataBuilderTest, DecodeMapsCompleteCacheWithoutCopies) {
  ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
  WrapperFixture fixture;
  fixture.init({5, 8});
  auto params = make_rec_params(fixture, BatchForwardType::DECODE);

  auto metadata = AttentionMetadataBuilder::build(params);
  ASSERT_TRUE(metadata.xattention_two_stage_decode_cache.has_value());
  const auto& rec = *params.llmrec_params();
  const auto& cache = metadata.xattention_two_stage_decode_cache.value();

  EXPECT_EQ(cache.shared_lse.data_ptr(), rec.two_stage_shared_lse.data_ptr());
  EXPECT_EQ(cache.shared_o.data_ptr(), rec.two_stage_shared_o.data_ptr());
  EXPECT_EQ(cache.unshared_lse.data_ptr(),
            rec.two_stage_unshared_lse.data_ptr());
  EXPECT_EQ(cache.unshared_o.data_ptr(), rec.two_stage_unshared_o.data_ptr());
  EXPECT_EQ(cache.q_cu_seq_lens_shared.data_ptr(),
            rec.two_stage_q_cu_seq_lens_shared.data_ptr());
  EXPECT_EQ(cache.shared_lse_kernel.data_ptr(),
            rec.two_stage_shared_lse_kernel.data_ptr());
  EXPECT_EQ(cache.decode_slot_mapping.data_ptr(),
            rec.two_stage_decode_slot_mapping.data_ptr());
  EXPECT_EQ(cache.unshared_seq_lens.data_ptr(),
            rec.two_stage_unshared_seq_lens.data_ptr());
  EXPECT_EQ(cache.cached_batch_size, kBatchSize);
  EXPECT_EQ(cache.cached_beam_size, kBeamWidth);
  EXPECT_EQ(cache.cached_max_decode_step, kMaxDecodeSteps);
  EXPECT_EQ(metadata.total_kv_len, 13);
}

TEST(MluAttentionMetadataBuilderDeathTest, RejectsMissingDecodeField) {
  EXPECT_DEATH(
      {
        ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
        WrapperFixture fixture;
        fixture.init({5, 8});
        auto params = make_rec_params(fixture, BatchForwardType::DECODE);
        params.mutable_llmrec_params().two_stage_unshared_seq_lens =
            torch::Tensor();
        AttentionMetadataBuilder::build(params);
      },
      "two_stage_unshared_seq_lens");
}

TEST(MluAttentionMetadataBuilderDeathTest, RejectsMixedRecForward) {
  EXPECT_DEATH(
      {
        ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
        WrapperFixture fixture;
        fixture.init({5, 8});
        auto params = make_rec_params(fixture, BatchForwardType::MIXED);
        AttentionMetadataBuilder::build(params);
      },
      "only supports PREFILL or DECODE");
}

TEST(MluXAttentionKernelTest, PackedFlashReturnsKernelLseLayout) {
  constexpr int64_t total_beam = 4;
  auto query = to_mlu(torch::randn({total_beam, kNumHeads, kHeadDim}));
  auto key = to_mlu(torch::randn({13, kNumKvHeads, kHeadDim}));
  auto value = to_mlu(torch::randn({13, kNumKvHeads, kHeadDim}));
  auto output = torch::empty_like(query);
  auto lse = torch::empty({kNumHeads, total_beam},
                          query.options().dtype(torch::kFloat32));
  std::optional<torch::Tensor> output_lse = lse;
  auto q_cu = to_mlu(torch::tensor({0, 2, 4}, cpu_int()), torch::kInt32);
  auto kv_cu = to_mlu(torch::tensor({0, 5, 13}, cpu_int()), torch::kInt32);

  xllm::kernel::mlu::batch_prefill(
      query,
      key,
      value,
      output,
      output_lse,
      q_cu,
      kv_cu,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      /*max_query_len=*/2,
      /*max_seq_len=*/8,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      /*is_causal=*/false,
      /*window_size_left=*/-1,
      /*window_size_right=*/-1,
      /*compute_dtype=*/"float",
      /*return_lse=*/true);

  EXPECT_EQ(output.sizes(), query.sizes());
  EXPECT_EQ(lse.sizes(), (at::IntArrayRef{kNumHeads, total_beam}));
  EXPECT_TRUE(torch::isfinite(lse).all().item<bool>());
}

TEST(MluXAttentionKernelTest, ReshapePagedWritesOnlyMappedStep) {
  constexpr int64_t total_beam = 4;
  auto key = to_mlu(torch::randn({total_beam, kNumKvHeads, kHeadDim}));
  auto value = to_mlu(torch::randn({total_beam, kNumKvHeads, kHeadDim}));
  auto k_cache =
      torch::full({total_beam, kNumKvHeads, kMaxDecodeSteps, kHeadDim},
                  kPoison,
                  key.options());
  auto v_cache = torch::full_like(k_cache, kPoison);
  auto slots = to_mlu(torch::tensor({1, 3, 5, 7}, cpu_int()), torch::kInt32);
  std::optional<torch::Tensor> value_opt = value;
  std::optional<torch::Tensor> v_cache_opt = v_cache;

  xllm::kernel::mlu::reshape_paged_cache(key,
                                         value_opt,
                                         k_cache,
                                         v_cache_opt,
                                         slots,
                                         /*direction=*/false);

  EXPECT_TRUE(torch::equal(k_cache.select(2, 1).cpu(), key.cpu()));
  EXPECT_TRUE(torch::equal(v_cache.select(2, 1).cpu(), value.cpu()));
  EXPECT_TRUE(torch::all(k_cache.select(2, 0) == kPoison).item<bool>());
  EXPECT_TRUE(torch::all(v_cache.select(2, 0) == kPoison).item<bool>());
}

TEST(MluXAttentionKernelTest, PadDecodeReturnsPadLseLayout) {
  constexpr int64_t total_beam = 4;
  auto query = to_mlu(torch::randn({total_beam, 1, kNumHeads, kHeadDim}));
  auto k_cache = to_mlu(
      torch::randn({total_beam, kNumKvHeads, kMaxDecodeSteps, kHeadDim}));
  auto v_cache = to_mlu(
      torch::randn({total_beam, kNumKvHeads, kMaxDecodeSteps, kHeadDim}));
  auto output = torch::empty_like(query);
  auto lse = torch::empty({total_beam, kNumHeads, 1},
                          query.options().dtype(torch::kFloat32));
  auto int_opts = cpu_int();
  auto block_table = to_mlu(
      torch::arange(total_beam, int_opts).view({total_beam, 1}), torch::kInt32);
  auto seq_lens = to_mlu(torch::full({total_beam}, 2, int_opts), torch::kInt32);
  std::optional<torch::Tensor> v_cache_opt = v_cache;
  std::optional<torch::Tensor> output_lse = lse;

  xllm::kernel::mlu::batch_decode(
      query,
      k_cache,
      output,
      block_table,
      seq_lens,
      v_cache_opt,
      output_lse,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      "float",
      kMaxDecodeSteps,
      -1,
      -1,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      /*return_lse=*/true,
      /*kv_cache_quant_bit_size=*/-1);

  EXPECT_EQ(output.sizes(), query.sizes());
  EXPECT_EQ(lse.sizes(), (at::IntArrayRef{total_beam, kNumHeads, 1}));
  EXPECT_TRUE(torch::isfinite(lse).all().item<bool>());
}

TEST(MluXAttentionKernelTest, PadMergeAcceptsEmptyOffsets) {
  constexpr int64_t total_beam = 4;
  auto out = to_mlu(torch::randn({total_beam, 1, kNumHeads, kHeadDim}));
  auto block_out = to_mlu(torch::randn({total_beam, 1, kNumHeads, kHeadDim}));
  auto lse = to_mlu(torch::randn({total_beam, kNumHeads, 1}), torch::kFloat32);
  auto block_lse =
      to_mlu(torch::randn({total_beam, kNumHeads, 1}), torch::kFloat32);

  xllm::kernel::mlu::update_out_and_lse(
      out, lse, block_out, block_lse, std::nullopt, std::nullopt, std::nullopt);

  EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(lse).all().item<bool>());
}

TEST(MluXAttentionComponentTest, PrefillIsCausalAndWritesPackedSharedCache) {
  torch::manual_seed(20260729);
  ComponentFixture fixture;
  fixture.init({5, 8});
  auto query_cpu = torch::randn({13, kNumHeads, kHeadDim}, cpu_float()) * 0.1;
  auto key_cpu = torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto value_cpu = torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto query = to_mlu(query_cpu);
  auto key = to_mlu(key_cpu);
  auto value = to_mlu(value_cpu);

  auto actual = fixture.run_prefill(query, key, value);
  auto reference =
      packed_prefill_ref(query_cpu, key_cpu, value_cpu, {5, 8}, fixture.scale)
          .reshape({13, -1});

  EXPECT_TRUE(
      torch::allclose(actual.cpu().to(torch::kFloat32), reference, 2e-2, 2e-2));
  EXPECT_TRUE(torch::equal(fixture.full_k.slice(0, 0, 13).cpu(), key.cpu()));
  EXPECT_TRUE(torch::equal(fixture.full_v.slice(0, 0, 13).cpu(), value.cpu()));
  EXPECT_TRUE(torch::all(fixture.full_k.slice(0, 13, kMaxTokens) == kPoison)
                  .item<bool>());
  EXPECT_TRUE(torch::all(fixture.full_v.slice(0, 13, kMaxTokens) == kPoison)
                  .item<bool>());
  EXPECT_TRUE(torch::all(fixture.unshared_k == kPoison).item<bool>());
  EXPECT_TRUE(torch::all(fixture.unshared_v == kPoison).item<bool>());
}

TEST(MluXAttentionComponentTest, PrefillAcceptsStridedQkvViews) {
  torch::manual_seed(20260804);
  ComponentFixture fixture;
  fixture.init({5, 8});
  constexpr int64_t total_heads = kNumHeads + 2 * kNumKvHeads;
  auto qkv_cpu = torch::randn({13, total_heads * kHeadDim}, cpu_float()) * 0.1;
  auto qkv = to_mlu(qkv_cpu);
  auto query = qkv.slice(-1, 0, kNumHeads * kHeadDim);
  auto key =
      qkv.slice(-1, kNumHeads * kHeadDim, (kNumHeads + kNumKvHeads) * kHeadDim);
  auto value = qkv.slice(
      -1, (kNumHeads + kNumKvHeads) * kHeadDim, total_heads * kHeadDim);
  ASSERT_FALSE(query.is_contiguous());
  ASSERT_FALSE(key.is_contiguous());
  ASSERT_FALSE(value.is_contiguous());

  auto actual = fixture.run_prefill(query, key, value);
  auto query_cpu = qkv_cpu.slice(-1, 0, kNumHeads * kHeadDim)
                       .view({13, kNumHeads, kHeadDim});
  auto key_cpu =
      qkv_cpu
          .slice(-1, kNumHeads * kHeadDim, (kNumHeads + kNumKvHeads) * kHeadDim)
          .view({13, kNumKvHeads, kHeadDim});
  auto value_cpu =
      qkv_cpu
          .slice(
              -1, (kNumHeads + kNumKvHeads) * kHeadDim, total_heads * kHeadDim)
          .view({13, kNumKvHeads, kHeadDim});
  auto reference =
      packed_prefill_ref(query_cpu, key_cpu, value_cpu, {5, 8}, fixture.scale)
          .reshape({13, -1});

  EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
  EXPECT_TRUE(
      torch::allclose(actual.cpu().to(torch::kFloat32), reference, 2e-2, 2e-2));
}

TEST(MluXAttentionComponentTest, DecodeTwoStagesMatchFullAttentionAtBothSteps) {
  torch::manual_seed(20260729);
  ComponentFixture fixture;
  fixture.init({5, 8});
  auto prompt_q = to_mlu(torch::randn({13, kNumHeads, kHeadDim}) * 0.1);
  auto prompt_k_cpu =
      torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto prompt_v_cpu =
      torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto prompt_k = to_mlu(prompt_k_cpu);
  auto prompt_v = to_mlu(prompt_v_cpu);
  fixture.run_prefill(prompt_q, prompt_k, prompt_v);

  std::vector<torch::Tensor> suffix_k;
  std::vector<torch::Tensor> suffix_v;
  for (int64_t step = 0; step < kMaxDecodeSteps; ++step) {
    auto query_cpu =
        torch::randn({fixture.total_beam, kNumHeads, kHeadDim}, cpu_float()) *
        0.1;
    auto key_cpu =
        torch::randn({fixture.total_beam, kNumKvHeads, kHeadDim}, cpu_float()) *
        0.1;
    auto value_cpu =
        torch::randn({fixture.total_beam, kNumKvHeads, kHeadDim}, cpu_float()) *
        0.1;
    suffix_k.push_back(key_cpu);
    suffix_v.push_back(value_cpu);
    auto query = to_mlu(query_cpu);
    auto key = to_mlu(key_cpu);
    auto value = to_mlu(value_cpu);

    auto actual = fixture.run_decode(query, key, value, step)
                      .cpu()
                      .to(torch::kFloat32)
                      .view({fixture.total_beam, kNumHeads, kHeadDim});
    std::vector<torch::Tensor> refs;
    int64_t prompt_offset = 0;
    for (int64_t request = 0; request < kBatchSize; ++request) {
      for (int64_t beam = 0; beam < kBeamWidth; ++beam) {
        const int64_t row = request * kBeamWidth + beam;
        std::vector<torch::Tensor> beam_k = {prompt_k_cpu.slice(
            0, prompt_offset, prompt_offset + fixture.prompt_lens[request])};
        std::vector<torch::Tensor> beam_v = {prompt_v_cpu.slice(
            0, prompt_offset, prompt_offset + fixture.prompt_lens[request])};
        for (int64_t suffix_step = 0; suffix_step <= step; ++suffix_step) {
          beam_k.push_back(suffix_k[suffix_step].slice(0, row, row + 1));
          beam_v.push_back(suffix_v[suffix_step].slice(0, row, row + 1));
        }
        refs.push_back(attention_ref(query_cpu.slice(0, row, row + 1),
                                     torch::cat(beam_k, 0),
                                     torch::cat(beam_v, 0),
                                     fixture.scale,
                                     /*causal=*/false));
      }
      prompt_offset += fixture.prompt_lens[request];
    }
    auto reference = torch::cat(refs, 0);
    auto abs_diff = (actual - reference).abs();
    EXPECT_TRUE(torch::allclose(actual, reference, 2e-2, 2e-2))
        << "step=" << step << ", max_abs=" << abs_diff.max().item<float>()
        << ", mean_abs=" << abs_diff.mean().item<float>() << ", fail_ratio="
        << (abs_diff > (2e-2 + 2e-2 * reference.abs()))
               .to(torch::kFloat32)
               .mean()
               .item<float>();

    auto written_k = fixture.unshared_k.select(3, step).reshape(
        {fixture.total_beam, kNumKvHeads, kHeadDim});
    auto written_v = fixture.unshared_v.select(3, step).reshape(
        {fixture.total_beam, kNumKvHeads, kHeadDim});
    EXPECT_TRUE(torch::equal(written_k.cpu(), key.cpu()));
    EXPECT_TRUE(torch::equal(written_v.cpu(), value.cpu()));
    if (step == 0) {
      EXPECT_TRUE(
          torch::all(fixture.unshared_k.select(3, 1) == kPoison).item<bool>());
      EXPECT_TRUE(
          torch::all(fixture.unshared_v.select(3, 1) == kPoison).item<bool>());
    }
    const auto& cache = fixture.meta.xattention_two_stage_decode_cache.value();
    EXPECT_EQ(cache.shared_lse.scalar_type(), torch::kFloat32);
    EXPECT_EQ(cache.unshared_lse.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::isfinite(cache.shared_lse).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(cache.unshared_lse).all().item<bool>());
  }
}

TEST(MluAttentionDispatchTest, PrefillUsesXAttentionWithoutDecodeMetadata) {
  ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
  torch::manual_seed(20260729);
  WrapperFixture fixture;
  fixture.init({5, 8}, /*with_decode_cache=*/false);
  auto query_cpu = torch::randn({13, kNumHeads, kHeadDim}, cpu_float()) * 0.1;
  auto key_cpu = torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto value_cpu = torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto query = to_mlu(query_cpu);
  auto key = to_mlu(key_cpu);
  auto value = to_mlu(value_cpu);

  auto actual = fixture.run_prefill(query, key, value);
  auto reference =
      packed_prefill_ref(query_cpu, key_cpu, value_cpu, {5, 8}, fixture.scale)
          .reshape({13, -1});

  EXPECT_TRUE(
      torch::allclose(actual.cpu().to(torch::kFloat32), reference, 2e-2, 2e-2));
  EXPECT_TRUE(torch::equal(fixture.full_k.slice(0, 0, 13).cpu(), key.cpu()));
  EXPECT_TRUE(torch::equal(fixture.full_v.slice(0, 0, 13).cpu(), value.cpu()));
  EXPECT_TRUE(torch::all(fixture.full_k.slice(0, 13, kMaxTokens) == kPoison)
                  .item<bool>());
  EXPECT_TRUE(torch::all(fixture.full_v.slice(0, 13, kMaxTokens) == kPoison)
                  .item<bool>());
  EXPECT_FALSE(fixture.meta.xattention_two_stage_decode_cache.has_value());
  EXPECT_FALSE(fixture.meta.unshared_k_cache.defined());
  EXPECT_FALSE(fixture.meta.unshared_v_cache.defined());
}

TEST(MluAttentionDispatchTest, DecodeUsesBuilderAndCombinedCache) {
  ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
  torch::manual_seed(20260729);
  WrapperFixture fixture;
  fixture.init({5, 8});
  auto prompt_q = to_mlu(torch::randn({13, kNumHeads, kHeadDim}) * 0.1);
  auto prompt_k_cpu =
      torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto prompt_v_cpu =
      torch::randn({13, kNumKvHeads, kHeadDim}, cpu_float()) * 0.1;
  auto prompt_k = to_mlu(prompt_k_cpu);
  auto prompt_v = to_mlu(prompt_v_cpu);
  fixture.run_prefill(prompt_q, prompt_k, prompt_v);

  std::vector<torch::Tensor> suffix_k;
  std::vector<torch::Tensor> suffix_v;
  for (int64_t step = 0; step < kMaxDecodeSteps; ++step) {
    auto query_cpu =
        torch::randn({fixture.total_beam, kNumHeads, kHeadDim}, cpu_float()) *
        0.1;
    auto key_cpu =
        torch::randn({fixture.total_beam, kNumKvHeads, kHeadDim}, cpu_float()) *
        0.1;
    auto value_cpu =
        torch::randn({fixture.total_beam, kNumKvHeads, kHeadDim}, cpu_float()) *
        0.1;
    suffix_k.push_back(key_cpu);
    suffix_v.push_back(value_cpu);
    auto query = to_mlu(query_cpu);
    auto key = to_mlu(key_cpu);
    auto value = to_mlu(value_cpu);

    auto params = make_rec_params(fixture, BatchForwardType::DECODE);
    auto& rec = params.mutable_llmrec_params();
    runtime::detail::update_mlu_two_stage(params.block_tables,
                                          step,
                                          fixture.max_decode_steps,
                                          rec.two_stage_decode_slot_mapping,
                                          rec.two_stage_unshared_seq_lens);
    auto metadata = AttentionMetadataBuilder::build(params);
    metadata.full_k_cache = fixture.full_k;
    metadata.full_v_cache = fixture.full_v;
    metadata.unshared_k_cache = fixture.unshared_k;
    metadata.unshared_v_cache = fixture.unshared_v;
    fixture.meta = std::move(metadata);

    auto actual = fixture.run_decode(query, key, value, step)
                      .cpu()
                      .to(torch::kFloat32)
                      .view({fixture.total_beam, kNumHeads, kHeadDim});
    std::vector<torch::Tensor> refs;
    int64_t prompt_offset = 0;
    for (int64_t request = 0; request < kBatchSize; ++request) {
      for (int64_t beam = 0; beam < kBeamWidth; ++beam) {
        const int64_t row = request * kBeamWidth + beam;
        std::vector<torch::Tensor> beam_k = {prompt_k_cpu.slice(
            0, prompt_offset, prompt_offset + fixture.prompt_lens[request])};
        std::vector<torch::Tensor> beam_v = {prompt_v_cpu.slice(
            0, prompt_offset, prompt_offset + fixture.prompt_lens[request])};
        for (int64_t suffix_step = 0; suffix_step <= step; ++suffix_step) {
          beam_k.push_back(suffix_k[suffix_step].slice(0, row, row + 1));
          beam_v.push_back(suffix_v[suffix_step].slice(0, row, row + 1));
        }
        refs.push_back(attention_ref(query_cpu.slice(0, row, row + 1),
                                     torch::cat(beam_k, 0),
                                     torch::cat(beam_v, 0),
                                     fixture.scale,
                                     /*causal=*/false));
      }
      prompt_offset += fixture.prompt_lens[request];
    }
    auto reference = torch::cat(refs, 0);
    auto abs_diff = (actual - reference).abs();
    EXPECT_TRUE(torch::allclose(actual, reference, 2e-2, 2e-2))
        << "step=" << step << ", max_abs=" << abs_diff.max().item<float>()
        << ", mean_abs=" << abs_diff.mean().item<float>() << ", fail_ratio="
        << (abs_diff > (2e-2 + 2e-2 * reference.abs()))
               .to(torch::kFloat32)
               .mean()
               .item<float>();

    auto written_k = fixture.unshared_k.select(3, step).reshape(
        {fixture.total_beam, kNumKvHeads, kHeadDim});
    auto written_v = fixture.unshared_v.select(3, step).reshape(
        {fixture.total_beam, kNumKvHeads, kHeadDim});
    EXPECT_TRUE(torch::equal(written_k.cpu(), key.cpu()));
    EXPECT_TRUE(torch::equal(written_v.cpu(), value.cpu()));
    if (step == 0) {
      EXPECT_TRUE(
          torch::all(fixture.unshared_k.select(3, 1) == kPoison).item<bool>());
      EXPECT_TRUE(
          torch::all(fixture.unshared_v.select(3, 1) == kPoison).item<bool>());
    }
    const auto& cache = fixture.meta.xattention_two_stage_decode_cache.value();
    EXPECT_TRUE(torch::isfinite(cache.shared_lse).all().item<bool>());
    EXPECT_TRUE(torch::isfinite(cache.unshared_lse).all().item<bool>());
  }
}

TEST(MluAttentionDispatchDeathTest, RejectsOneStageRecMode) {
  EXPECT_DEATH(
      {
        ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/true);
        AttentionImpl attention(kNumHeads,
                                kHeadDim,
                                1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                                kNumKvHeads,
                                /*sliding_window=*/-1);
      },
      "MLU.*two-stage.*enable_xattention_one_stage");
}

TEST(MluAttentionDispatchDeathTest, RejectsExtendedConstructorInRecMode) {
  EXPECT_DEATH(
      {
        ScopedRecFlags flags(/*max_decode_rounds=*/3, /*one_stage=*/false);
        AttentionImpl attention(kNumHeads,
                                kHeadDim,
                                kNumKvHeads,
                                /*v_head_dim=*/kHeadDim,
                                /*sliding_window=*/-1,
                                1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                                /*use_fused_mla_qkv=*/false,
                                /*enable_lighting_indexer=*/false,
                                /*enable_mla=*/false);
      },
      "MLU REC xAttention.*(MLA|lighting indexer)");
}

TEST(MluAttentionDispatchTest, OneStageFlagDoesNotRejectNonRecMode) {
  ScopedRecFlags flags(/*max_decode_rounds=*/0, /*one_stage=*/true);
  EXPECT_NO_FATAL_FAILURE({
    AttentionImpl attention(kNumHeads,
                            kHeadDim,
                            1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                            kNumKvHeads,
                            /*sliding_window=*/-1);
  });
}

TEST(MluXAttentionComponentTest, TargetShapeReusesCombinedCache32Times) {
  constexpr int64_t input_len = 1024;
  constexpr int64_t beam_width = 256;
  constexpr int64_t num_heads = 16;
  constexpr int64_t num_kv_heads = 8;
  constexpr int64_t head_dim = 128;
  ComponentFixture fixture(beam_width,
                           input_len,
                           num_heads,
                           num_kv_heads,
                           head_dim,
                           kMaxDecodeSteps);
  fixture.init({input_len});
  for (int64_t iteration = 0; iteration < 32; ++iteration) {
    auto prompt_q =
        to_mlu(torch::randn({input_len, num_heads, head_dim}) * 0.01);
    auto prompt_k =
        to_mlu(torch::randn({input_len, num_kv_heads, head_dim}) * 0.01);
    auto prompt_v =
        to_mlu(torch::randn({input_len, num_kv_heads, head_dim}) * 0.01);
    auto prefill = fixture.run_prefill(prompt_q, prompt_k, prompt_v);
    EXPECT_TRUE(torch::isfinite(prefill).all().item<bool>());
    for (int64_t step = 0; step < kMaxDecodeSteps; ++step) {
      auto query =
          to_mlu(torch::randn({beam_width, num_heads, head_dim}) * 0.01);
      auto key =
          to_mlu(torch::randn({beam_width, num_kv_heads, head_dim}) * 0.01);
      auto value =
          to_mlu(torch::randn({beam_width, num_kv_heads, head_dim}) * 0.01);
      auto output = fixture.run_decode(query, key, value, step);
      EXPECT_TRUE(torch::isfinite(output).all().item<bool>())
          << "iteration=" << iteration << ", step=" << step;
    }
  }
}

}  // namespace
}  // namespace xllm::layer::test
