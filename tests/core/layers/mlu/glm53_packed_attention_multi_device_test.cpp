/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/config/parallel_config.h"
#include "core/layers/mlu/multi_device_test_utils.h"
#include "framework/parallel_state/collective_communicator.h"
#include "layers/mlu/attention.h"
#include "layers/mlu/dcp_decode_context.h"
#include "layers/mlu/deepseek_v32_cp_context.h"
#include "platform/device.h"

namespace xllm::layer {
namespace {
void check_packed_values(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         const std::string& label) {
  const torch::Tensor a = actual.cpu().to(torch::kFloat32);
  const torch::Tensor b = expected.cpu().to(torch::kFloat32);
  ASSERT_EQ(a.sizes(), b.sizes()) << label;
  ASSERT_TRUE(torch::equal(torch::isfinite(a), torch::isfinite(b))) << label;
  ASSERT_TRUE(torch::equal(torch::isneginf(a), torch::isneginf(b))) << label;
  ASSERT_TRUE(torch::equal(torch::isposinf(a), torch::isposinf(b))) << label;
  ASSERT_FALSE(torch::isnan(a).any().item<bool>()) << label;
  const torch::Tensor finite = torch::isfinite(b);
  const torch::Tensor av = a.masked_select(finite);
  const torch::Tensor bv = b.masked_select(finite);
  ASSERT_TRUE(torch::allclose(av, bv, /*rtol=*/1e-2, /*atol=*/1e-3)) << label;
  const float error =
      av.numel() > 0 ? (av - bv).abs().max().item<float>() : 0.0f;
  LOG(INFO) << label << " bitwise_equal=" << torch::equal(a, b)
            << " max_abs_error=" << error;
}

std::pair<torch::Tensor, torch::Tensor> packed_local_attention(
    torch::Tensor query,
    const DsaTopkState& topk,
    torch::Tensor key_cache) {
  Attention attention(/*num_heads=*/32,
                      /*head_size=*/576,
                      /*num_kv_heads=*/1,
                      /*v_head_dim=*/512,
                      /*sliding_window=*/-1,
                      /*scale=*/0.0625f,
                      /*use_fused_mla_qkv=*/true,
                      /*enable_lighting_indexer=*/true,
                      /*enable_mla=*/true);
  AttentionMetadata metadata{};
  metadata.is_chunked_prefill = true;
  metadata.compute_dtype = "half";
  metadata.max_seq_len = 2048;
  metadata.block_table = topk.block_tables();
  metadata.kv_seq_lens = topk.context_lens();
  KVCache cache(KVCacheTensors{std::move(key_cache), torch::Tensor()});
  torch::Tensor unused;
  auto [output, lse] = attention(metadata,
                                 query,
                                 unused,
                                 unused,
                                 cache,
                                 /*return_lse=*/true);
  CHECK(lse.has_value());
  return {output.view({query.size(0), 1, 32, 512}), lse.value()};
}

void check_sparse_reference(const torch::Tensor& output,
                            const torch::Tensor& query,
                            const torch::Tensor& full_cache,
                            const DsaTopkState& topk,
                            const torch::Tensor& slots) {
  const torch::Tensor values =
      full_cache.cpu().view({512, 576}).to(torch::kFloat64);
  const torch::Tensor q = query.cpu().to(torch::kFloat64);
  const torch::Tensor table = topk.block_tables().cpu().to(torch::kInt64);
  const torch::Tensor lengths = topk.context_lens().cpu();
  const torch::Tensor mapping = slots.cpu();
  const torch::Tensor result = output.cpu().view({query.size(0), 32, 512});
  for (int64_t row : {int64_t{0},
                      int64_t{1},
                      int64_t{3},
                      int64_t{4},
                      int64_t{15},
                      int64_t{31},
                      query.size(0) - 2,
                      query.size(0) - 1}) {
    const int64_t length = lengths[row].item<int32_t>();
    torch::Tensor expected = torch::zeros({32, 512}, torch::kFloat64);
    if (length > 0 && mapping[row].item<int32_t>() >= 0) {
      const torch::Tensor keys =
          values.index_select(0, table[row].narrow(0, 0, length));
      const torch::Tensor scores =
          torch::matmul(q[row], keys.transpose(0, 1)) * 0.0625;
      expected =
          torch::matmul(torch::softmax(scores, -1), keys.narrow(1, 0, 512));
    }
    check_packed_values(result[row],
                        expected,
                        "FP64 sparse reference row=" + std::to_string(row));
  }
}

void run_packed_chain(int32_t rank, int32_t port) {
  torch::NoGradGuard no_grad;
  torch::set_num_threads(1);
  Device device(rank);
  device.set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device.unwrap());
  ParallelConfig::get_instance().kv_split_size(4);
  CollectiveCommunicator communicator(rank, 8, 1, 1, 4);
  communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                     device.unwrap());
  const ParallelArgs& parallel = *communicator.parallel_args();
  const int32_t shard_rank = parallel.cp_group_->rank();
  DcpDecodeContext dcp(KVShardLayout(/*physical_block_size=*/16,
                                     /*dcp_size=*/4,
                                     shard_rank),
                       parallel.cp_group_);
  for (const std::vector<int32_t>& lengths : {std::vector<int32_t>{64},
                                              std::vector<int32_t>{33, 34},
                                              std::vector<int32_t>{8192}}) {
    const int32_t rows =
        std::accumulate(lengths.begin(), lengths.end(), int32_t{0});
    SCOPED_TRACE("rank=" + std::to_string(rank) +
                 " rows=" + std::to_string(rows));
    const auto segments = v32_cp::build_all_cp_segments(4, lengths, lengths);
    const auto artifacts =
        v32_cp::build_cp_runtime_artifacts(shard_rank, 4, segments, rows);
    v32_cp::DeepseekV32CPContext cp;
    cp.rank = shard_rank;
    cp.total_tokens = rows;
    cp.process_group = parallel.cp_group_;
    cp.comm_plan = artifacts.comm_plan;
    cp.gathered_reorder_index =
        torch::tensor(artifacts.gathered_reorder_index_cpu, torch::kInt64)
            .to(device.unwrap());
    torch::manual_seed(530053);
    const torch::Tensor query =
        (torch::randn({rows, 32, 576}, torch::kFloat32) * 0.1).to(options);
    const torch::Tensor full_cache =
        (torch::randn({32, 1, 16, 576}, torch::kFloat32) * 0.1).to(options);
    const torch::Tensor cache = full_cache.view({8, 4, 16, 576})
                                    .select(1, shard_rank)
                                    .contiguous()
                                    .view({8, 1, 16, 576});
    const auto ints = torch::TensorOptions().dtype(torch::kInt32);
    const torch::Tensor indices = torch::arange(rows, ints);
    torch::Tensor table =
        (indices.unsqueeze(1) * 17 + torch::arange(2048, ints) * 7)
            .remainder(512);
    torch::Tensor lens = indices.remainder(193);
    table[1].copy_(torch::arange(2048, ints).remainder(16));
    lens[1].fill_(8);
    torch::Tensor slots = indices.clone();
    slots[3].fill_(-1);
    slots[rows - 1].fill_(-1);
    table = table.to(device.unwrap());
    lens = lens.to(device.unwrap());
    slots = slots.to(device.unwrap());
    cp.gathered_slot_mapping = slots.index_select(0, cp.gathered_reorder_index);
    const DsaTopkState global_topk(table, lens);
    const DsaTopkState local_topk(v32_cp::reorder_to_local_shard(table, cp),
                                  v32_cp::reorder_to_local_shard(lens, cp));
    const torch::Tensor local_q = v32_cp::reorder_to_local_shard(query, cp);
    const torch::Tensor original_q =
        v32_cp::gather_and_restore_global(local_q, cp);
    const DsaTopkState original_topk(
        v32_cp::gather_and_restore_global(local_topk.block_tables(), cp),
        v32_cp::gather_and_restore_global(local_topk.context_lens(), cp));
    const torch::Tensor packed_q = v32_cp::all_gather_across_ranks(local_q, cp);
    const DsaTopkState packed_topk(
        v32_cp::all_gather_across_ranks(local_topk.block_tables(), cp),
        v32_cp::all_gather_across_ranks(local_topk.context_lens(), cp));
    EXPECT_TRUE(torch::equal(packed_q,
                             query.index_select(0, cp.gathered_reorder_index)));
    EXPECT_TRUE(torch::equal(original_q, query));
    const DsaTopkState original_local = dcp.localize_topk(original_topk);
    const DsaTopkState packed_local = dcp.localize_topk(packed_topk);
    EXPECT_TRUE(torch::equal(packed_local.block_tables(),
                             original_local.block_tables().index_select(
                                 0, cp.gathered_reorder_index)));
    EXPECT_TRUE(torch::equal(packed_local.context_lens(),
                             original_local.context_lens().index_select(
                                 0, cp.gathered_reorder_index)));
    auto [original_output, original_lse] =
        packed_local_attention(original_q, original_local, cache);
    auto [packed_output, packed_lse] =
        packed_local_attention(packed_q, packed_local, cache);
    check_packed_values(
        packed_output,
        original_output.index_select(0, cp.gathered_reorder_index),
        "local attention rows=" + std::to_string(rows));
    check_packed_values(packed_lse,
                        original_lse.index_select(0, cp.gathered_reorder_index),
                        "local LSE rows=" + std::to_string(rows));
    const torch::Tensor original_merged =
        dcp.merge(original_output, original_lse, slots, /*head_sharded=*/false);
    const torch::Tensor packed_merged = dcp.merge(packed_output,
                                                  packed_lse,
                                                  cp.gathered_slot_mapping,
                                                  /*head_sharded=*/false);
    check_packed_values(
        packed_merged,
        original_merged.index_select(0, cp.gathered_reorder_index),
        "merged attention rows=" + std::to_string(rows));
    check_packed_values(v32_cp::slice_local_packed(packed_merged, cp),
                        v32_cp::reorder_to_local_shard(original_merged, cp),
                        "local slice rows=" + std::to_string(rows));
    check_sparse_reference(
        original_merged, query, full_cache, global_topk, slots);
  }
}

TEST(PackedAttentionParallelTest, EightRanksMatchGlobalOrderAndFp64Reference) {
  xllm::test::run_multi_device(8, run_packed_chain);
}

}  // namespace
}  // namespace xllm::layer
