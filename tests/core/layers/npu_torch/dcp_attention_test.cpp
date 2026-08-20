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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/kernels/npu/npu_ops_api.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/parallel_state/process_group.h"
#include "layers/npu_torch/attention.h"

namespace xllm::layer::test {
namespace {

class ScriptedDcpProcessGroup final : public ProcessGroup {
 public:
  ScriptedDcpProcessGroup(const torch::Device& device,
                          torch::Tensor peer_partial_out,
                          torch::Tensor peer_partial_lse)
      : ProcessGroup(1, 2, device),
        peer_partial_out_(std::move(peer_partial_out)),
        peer_partial_lse_(std::move(peer_partial_lse)) {}

  torch::Tensor allgather_base_sync(const torch::Tensor& input) override {
    if (call_count_ == 0) {
      ++call_count_;
      return torch::stack({input, input}, 0);
    }
    if (call_count_ == 1) {
      CHECK_EQ(input.sizes(), peer_partial_out_.sizes());
      normalized_out_before_gather_ =
          torch::equal(input, torch::zeros_like(input));
      ++call_count_;
      return torch::stack({peer_partial_out_, input}, 0);
    }
    if (call_count_ == 2) {
      CHECK_EQ(input.sizes(), peer_partial_lse_.sizes());
      normalized_lse_before_gather_ = torch::equal(
          input,
          torch::full_like(input, -std::numeric_limits<float>::infinity()));
      ++call_count_;
      return torch::stack({peer_partial_lse_, input}, 0);
    }
    LOG(FATAL) << "Unexpected DCP all-gather call " << call_count_;
    return torch::Tensor();
  }

  int32_t call_count() const { return call_count_; }
  bool normalized_out_before_gather() const {
    return normalized_out_before_gather_;
  }
  bool normalized_lse_before_gather() const {
    return normalized_lse_before_gather_;
  }

 private:
  torch::Tensor peer_partial_out_;
  torch::Tensor peer_partial_lse_;
  int32_t call_count_ = 0;
  bool normalized_out_before_gather_ = false;
  bool normalized_lse_before_gather_ = false;
};

class DcpAttentionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }
  static void TearDownTestSuite() { torch_npu::finalize_npu(); }

  torch::Device device_ = torch::Device("npu:0");
};

TEST_F(DcpAttentionTest, ZeroLocalKvNormalizesBeforeMergeAndSlicesLocalHeads) {
  const int64_t block_size = 128;
  const int64_t head_size = 128;
  const int64_t local_num_heads = 4;
  const int64_t num_kv_heads = 1;
  const int64_t group_num_heads = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  torch::Tensor peer_partial_out =
      torch::ones({1, group_num_heads, head_size}, fp32_options);
  peer_partial_out.slice(1, local_num_heads, group_num_heads).fill_(2.0f);
  const torch::Tensor peer_partial_lse =
      torch::zeros({1, group_num_heads, 1}, fp32_options);
  ScriptedDcpProcessGroup dcp_group(
      device_, peer_partial_out, peer_partial_lse);

  torch::Tensor key = torch::zeros({1, num_kv_heads, head_size}, bf16_options);
  torch::Tensor value = torch::zeros_like(key);
  torch::Tensor query =
      torch::randn({1, local_num_heads * head_size}, bf16_options);
  const torch::Tensor k_cache =
      torch::zeros({1, block_size, num_kv_heads, head_size}, bf16_options);
  const torch::Tensor v_cache = torch::zeros_like(k_cache);
  KVCache kv_cache(KVCacheTensors{k_cache, v_cache});

  AttentionMetadata attn_metadata{};
  attn_metadata.slot_mapping =
      torch::tensor(std::vector<int32_t>{-1},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device_);
  attn_metadata.block_table =
      torch::tensor(std::vector<int32_t>{0},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device_)
          .view({1, 1});
  attn_metadata.q_cu_seq_lens_host_vec = {1};
  attn_metadata.kv_seq_lens_host_vec = {1};

  AttentionImpl attention(
      local_num_heads, head_size, scale, num_kv_heads, -1, 2, 1, &dcp_group);
  const auto [output, output_lse] =
      attention.forward(attn_metadata, query, key, value, kv_cache);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  EXPECT_FALSE(output_lse.has_value());
  EXPECT_EQ(dcp_group.call_count(), 3);
  EXPECT_TRUE(dcp_group.normalized_out_before_gather());
  EXPECT_TRUE(dcp_group.normalized_lse_before_gather());
  const torch::Tensor output_cpu =
      output.cpu().to(torch::kFloat32).view({1, local_num_heads, head_size});
  const torch::Tensor expected =
      torch::full({1, local_num_heads, head_size},
                  2.0f,
                  torch::TensorOptions().dtype(torch::kFloat32));
  EXPECT_LT((output_cpu - expected).abs().max().item<float>(), 1e-4f);
}

TEST_F(DcpAttentionTest,
       ChunkedFirstChunkNormalizesLeadingZeroAndSkipsContextCollectives) {
  const int64_t token_count = 5;
  const int64_t block_size = 128;
  const int64_t head_size = 128;
  const int64_t local_num_heads = 4;
  const int64_t num_kv_heads = 1;
  const int64_t group_num_heads = 8;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  const torch::TensorOptions bf16_options =
      torch::TensorOptions().device(device_).dtype(torch::kBFloat16);
  const torch::TensorOptions fp32_options =
      torch::TensorOptions().device(device_).dtype(torch::kFloat32);

  ScriptedDcpProcessGroup dcp_group(device_,
                                    /*peer_partial_out=*/torch::Tensor(),
                                    /*peer_partial_lse=*/torch::Tensor());
  torch::Tensor query =
      torch::randn({token_count, local_num_heads * head_size}, bf16_options) *
      0.1;
  torch::Tensor key =
      torch::randn({token_count, num_kv_heads, head_size}, bf16_options) * 0.1;
  torch::Tensor value =
      torch::randn({token_count, num_kv_heads, head_size}, bf16_options) * 0.1;
  const torch::Tensor k_cache =
      torch::zeros({2, block_size, num_kv_heads, head_size}, bf16_options);
  const torch::Tensor v_cache = torch::zeros_like(k_cache);
  KVCache kv_cache(KVCacheTensors{k_cache, v_cache});

  AttentionMetadata attn_metadata{};
  attn_metadata.is_chunked_prefill = true;
  attn_metadata.slot_mapping =
      torch::full({token_count},
                  -1,
                  torch::TensorOptions().dtype(torch::kInt32).device(device_));
  attn_metadata.block_table =
      torch::tensor(std::vector<int32_t>{0, 1},
                    torch::TensorOptions().dtype(torch::kInt32))
          .to(device_)
          .view({2, 1});
  attn_metadata.fia_attn_mask =
      torch::triu(torch::ones({2048, 2048}, fp32_options), 1)
          .to(torch::kInt8)
          .contiguous();
  attn_metadata.q_cu_seq_lens_host_vec = {0, 2, 5};
  attn_metadata.kv_seq_lens_host_vec = {2, 3};

  const torch::Tensor local_query =
      query.view({token_count, local_num_heads, head_size});
  const torch::Tensor query_group =
      torch::cat({local_query, local_query}, /*dim=*/1).contiguous();
  const std::vector<int64_t> normalized_q_cu_seq_lens = {2, 5};
  const auto [expected_group_out, expected_group_lse] =
      xllm::kernel::npu::npu_fused_infer_attention(
          query_group,
          key,
          value,
          std::make_optional(attn_metadata.fia_attn_mask),
          /*block_table=*/std::nullopt,
          normalized_q_cu_seq_lens,
          normalized_q_cu_seq_lens,
          group_num_heads,
          num_kv_heads,
          scale,
          /*block_size=*/0,
          /*sparse_mode=*/3,
          "TND",
          /*softmax_lse_flag=*/true);
  ASSERT_EQ(expected_group_lse.sizes(),
            torch::IntArrayRef({token_count, group_num_heads, 1}));
  const torch::Tensor expected =
      expected_group_out.slice(1, local_num_heads, group_num_heads)
          .contiguous();

  AttentionImpl attention(
      local_num_heads, head_size, scale, num_kv_heads, -1, 2, 1, &dcp_group);
  const auto [output, output_lse] =
      attention.forward(attn_metadata, query, key, value, kv_cache);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream()),
            ACL_SUCCESS);

  EXPECT_FALSE(output_lse.has_value());
  EXPECT_EQ(dcp_group.call_count(), 1)
      << "all-zero global context must skip context FIA collectives";
  const torch::Tensor output_3d =
      output.view({token_count, local_num_heads, head_size});
  EXPECT_LT(
      (output_3d.cpu().to(torch::kFloat32) - expected.cpu().to(torch::kFloat32))
          .abs()
          .max()
          .item<float>(),
      2e-2f);
}

}  // namespace
}  // namespace xllm::layer::test
