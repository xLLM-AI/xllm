/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/npu/tilelang/h3_rainfusion_runtime.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "acl/acl.h"

namespace xllm::kernel::npu::tilelang {
namespace {

// Reference geometry exercised by both capacity kernels.
H3RainfusionGeometry production_geometry() {
  return H3RainfusionGeometry{
      /*total_rows=*/60160,
      /*valid_rows=*/60132,
      /*video_start=*/22836,
      /*grid_t=*/37,
      /*grid_h=*/24,
      /*grid_w=*/42,
      /*topk_r=*/94,
  };
}

uint64_t fnv1a64(const void* data, size_t bytes) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (size_t index = 0; index < bytes; ++index) {
    hash ^= cursor[index];
    hash *= 0x100000001b3ULL;
  }
  return hash;
}

// Golden values captured from the frozen Python reference
// build_reorder_records(PRODUCTION_GEOMETRY).
constexpr uint64_t kRecordsGoldenChecksum = 0x9eb20996347119a6ULL;
constexpr int64_t kGoldenRecordCount = 5557;
constexpr int64_t kGoldenRecordsPadded = 5568;
constexpr int64_t kGoldenFirstFrameRecords = 16;
constexpr int64_t kGoldenDenseRecords = 357;
constexpr int64_t kGoldenTailRecords = 5186;
constexpr int64_t kGoldenZeroSlots = 11;

TEST(H3RainfusionRecordsTest, DerivedGeometryMatchesFrozenValues) {
  const H3RainfusionDerived derived =
      h3_rainfusion_derive(production_geometry());
  EXPECT_EQ(derived.video_rows, 37296);
  EXPECT_EQ(derived.video_end, 60132);
  EXPECT_EQ(derived.total_blocks, 470);
  EXPECT_EQ(derived.tail_rows, 100);
  EXPECT_EQ(derived.ff_rows, 1008);
  EXPECT_EQ(derived.ff_blocks, 8);
  EXPECT_EQ(derived.video_full_blocks, 291);
  EXPECT_EQ(derived.num_sq, 283);
  EXPECT_EQ(derived.q_pool_pad, 288);
  EXPECT_EQ(derived.k_pool_pad, 480);
  EXPECT_EQ(derived.plan_width, 472);
}

TEST(H3RainfusionRecordsTest, RecordExtentsMatchFrozenValues) {
  const auto geometry = production_geometry();
  EXPECT_EQ(h3_rainfusion_record_count(geometry), kGoldenRecordCount);
  EXPECT_EQ(h3_rainfusion_records_padded(geometry), kGoldenRecordsPadded);
  EXPECT_EQ(kGoldenRecordsPadded % kH3RainfusionWorkers, 0);
}

TEST(H3RainfusionRecordsTest, TableMatchesPythonReferenceBitwise) {
  const auto table = build_h3_rainfusion_reorder_records(
      production_geometry(), torch::Device(torch::kCPU));
  ASSERT_EQ(table.sizes(), c10::IntArrayRef({kGoldenRecordsPadded, 4}));
  ASSERT_EQ(table.scalar_type(), torch::kInt32);
  ASSERT_TRUE(table.is_contiguous());
  EXPECT_EQ(fnv1a64(table.data_ptr<int32_t>(),
                    static_cast<size_t>(table.numel()) * sizeof(int32_t)),
            kRecordsGoldenChecksum);
}

TEST(H3RainfusionRecordsTest, WorkerMajorSlotsMatchFrozenRecords) {
  const torch::Tensor table = build_h3_rainfusion_reorder_records(
      production_geometry(), torch::Device(torch::kCPU));
  const std::array<int64_t, 6> slots = {0, 116, 1740, 1856, 1972, 4291};
  const std::array<std::array<int32_t, 4>, 6> expected = {
      {{22836, 0, 64, 1},
       {22900, 64, 64, 1},
       {23796, 960, 48, 5},
       {23844, 1008, 8, 4},
       {23886, 1016, 8, 4},
       {22784, 60080, 52, 6}}};
  const int32_t* data = table.data_ptr<int32_t>();
  for (size_t index = 0; index < slots.size(); ++index) {
    for (int64_t field = 0; field < 4; ++field) {
      EXPECT_EQ(data[slots[index] * 4 + field], expected[index][field])
          << "slot=" << slots[index] << " field=" << field;
    }
  }
}

TEST(H3RainfusionRecordsTest, ActiveRecordsPermuteEveryValidRow) {
  const auto geometry = production_geometry();
  const auto table =
      build_h3_rainfusion_reorder_records(geometry, torch::Device(torch::kCPU));
  const int32_t* rows = table.data_ptr<int32_t>();

  std::vector<int64_t> permutation(geometry.valid_rows, -1);
  int64_t active = 0;
  int64_t first_frame = 0;
  int64_t dense = 0;
  int64_t tail = 0;
  int64_t zero_slots = 0;
  std::set<int32_t> chunk_sizes;

  for (int64_t slot = 0; slot < kGoldenRecordsPadded; ++slot) {
    const int32_t src = rows[slot * 4 + 0];
    const int32_t dst = rows[slot * 4 + 1];
    const int32_t count = rows[slot * 4 + 2];
    const int32_t flags = rows[slot * 4 + 3];
    if (src == 0 && dst == 0 && count == 0 && flags == 0) {
      ++zero_slots;
      continue;
    }
    ++active;
    ASSERT_GT(count, 0) << "slot " << slot;
    ASSERT_GE(src, 0);
    ASSERT_GE(dst, 0);
    ASSERT_LE(src + count, geometry.valid_rows) << "slot " << slot;
    ASSERT_LE(dst + count, geometry.valid_rows) << "slot " << slot;
    chunk_sizes.insert(count);
    if ((flags & kH3RainfusionFlagFirstFrame) != 0) {
      ++first_frame;
    }
    if ((flags & kH3RainfusionFlagDense) != 0) {
      ++dense;
    }
    if ((flags & kH3RainfusionFlagTail) != 0) {
      ++tail;
    }
    for (int32_t offset = 0; offset < count; ++offset) {
      ASSERT_EQ(permutation[dst + offset], -1)
          << "destination row " << dst + offset << " is written twice";
      permutation[dst + offset] = src + offset;
    }
  }

  EXPECT_EQ(active, kGoldenRecordCount);
  EXPECT_EQ(zero_slots, kGoldenZeroSlots);
  EXPECT_EQ(first_frame, kGoldenFirstFrameRecords);
  EXPECT_EQ(dense, kGoldenDenseRecords);
  EXPECT_EQ(tail, kGoldenTailRecords);
  // The 2-row chunk is the W remainder (42 % 8 == 2) and the 48/52-row chunks
  // are the H remainder tails of the 64-row dense context spans.
  const std::set<int32_t> expected_chunks{2, 8, 48, 52, 64};
  EXPECT_EQ(chunk_sizes, expected_chunks);

  for (int64_t row = 0; row < geometry.valid_rows; ++row) {
    ASSERT_NE(permutation[row], -1) << "row " << row << " is never written";
  }
}

TEST(H3RainfusionRecordsTest, RejectsUnqualifiedGeometry) {
  const auto qualified = production_geometry();

  auto unaligned = qualified;
  unaligned.total_rows = 60161;
  EXPECT_THROW(h3_rainfusion_derive(unaligned), std::invalid_argument);

  auto packed_too_small = qualified;
  packed_too_small.total_rows = 60032;
  EXPECT_THROW(h3_rainfusion_derive(packed_too_small), std::invalid_argument);

  auto video_overruns = qualified;
  video_overruns.video_start = 22837;
  EXPECT_THROW(h3_rainfusion_derive(video_overruns), std::invalid_argument);

  auto degenerate_grid = qualified;
  degenerate_grid.grid_t = 0;
  EXPECT_THROW(h3_rainfusion_derive(degenerate_grid), std::invalid_argument);

  // A grid whose video span leaves no sparse query block is not qualified.
  auto no_sparse_region = qualified;
  no_sparse_region.grid_t = 1;
  EXPECT_THROW(h3_rainfusion_derive(no_sparse_region), std::invalid_argument);
}

// Independent CPU reference, following rf_v2_full_permutation and
// golden_selector_rfa in the frozen Python delivery. It never consumes the
// C++ record builder or device-generated plans when computing expected data.
torch::Tensor reference_permutation() {
  const torch::Tensor video = torch::arange(37 * 24 * 42, torch::kInt64);
  const torch::Tensor first = video.slice(0, 0, 1008);
  const torch::Tensor rest = video.slice(0, 1008).reshape({36, 24, 42});
  const torch::Tensor main = rest.slice(2, 0, 40)
                                 .reshape({36, 3, 8, 5, 8})
                                 .permute({0, 1, 3, 2, 4})
                                 .reshape({36, 960});
  const torch::Tensor remainder = rest.slice(2, 40).reshape({36, 48});
  const torch::Tensor reordered_video =
      torch::cat({first, torch::cat({main, remainder}, 1).flatten()}) + 22836;
  return torch::cat({reordered_video, torch::arange(22836, torch::kInt64)});
}

struct ReferenceSelector {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor plan_ids;
  torch::Tensor plan_count;
};

ReferenceSelector reference_selector(const torch::Tensor& query,
                                     const torch::Tensor& key,
                                     const torch::Tensor& value) {
  const torch::Tensor permutation = reference_permutation();
  ReferenceSelector result;
  result.query = query.index_select(0, permutation);
  result.key = key.index_select(0, permutation);
  result.value = value.index_select(0, permutation);
  std::vector<torch::Tensor> q_pools;
  std::vector<torch::Tensor> k_pools;
  for (int64_t block = 0; block < 470; ++block) {
    const int64_t end = std::min((block + 1) * 128, int64_t{60132});
    k_pools.push_back(
        result.key.slice(0, block * 128, end).to(torch::kFloat32).mean(0));
    if (block >= 8 && block < 291) {
      q_pools.push_back(
          result.query.slice(0, block * 128, end).to(torch::kFloat32).mean(0));
    }
  }
  const torch::Tensor q_pool = torch::stack(q_pools)
                                   .permute({1, 0, 2})
                                   .to(torch::kBFloat16)
                                   .to(torch::kFloat32);
  const torch::Tensor k_pool = torch::stack(k_pools)
                                   .permute({1, 0, 2})
                                   .to(torch::kBFloat16)
                                   .to(torch::kFloat32);
  const torch::Tensor scores =
      torch::bmm(q_pool, k_pool.transpose(1, 2)).permute({1, 0, 2});
  const torch::Tensor top_ids = std::get<1>(scores.topk(94, -1)).contiguous();
  const int64_t* top = top_ids.data_ptr<int64_t>();
  result.plan_ids = torch::full({470, 4, 472}, -1, torch::kInt32);
  result.plan_count = torch::zeros({470, 8}, torch::kInt32);
  int32_t* ids = result.plan_ids.data_ptr<int32_t>();
  int32_t* counts = result.plan_count.data_ptr<int32_t>();
  for (int64_t block = 0; block < 470; ++block) {
    for (int64_t head = 0; head < 4; ++head) {
      std::array<bool, 470> keep = {};
      const bool dense = block < 8 || block >= 291;
      if (!dense) {
        for (int64_t rank = 0; rank < 94; ++rank) {
          keep[top[((block - 8) * 4 + head) * 94 + rank]] = true;
        }
      }
      int32_t count = 0;
      for (int64_t kv_block = 0; kv_block < 470; ++kv_block) {
        if (dense || kv_block < 8 || kv_block >= 291 || keep[kv_block]) {
          ids[(block * 4 + head) * 472 + count++] = kv_block;
        }
      }
      counts[block * 8 + head] = count;
    }
  }
  return result;
}

// Same BF16 acceptance metrics as test_h3_rainfusion_attention.py in the
// frozen delivery. Samples include both dense boundaries, three sparse
// blocks, and the final partial block; values are checked in original order.
void expect_attention_precision(const torch::Tensor& actual,
                                const torch::Tensor& expected) {
  const torch::Tensor actual_f = actual.flatten().to(torch::kFloat64);
  const torch::Tensor expected_f = expected.flatten().to(torch::kFloat64);
  ASSERT_TRUE(torch::isfinite(actual_f).all().item<bool>());
  const torch::Tensor error = (actual_f - expected_f).abs();
  EXPECT_GE((error <= 9.77e-4 + 1.56e-2 * expected_f.abs())
                .to(torch::kFloat64)
                .mean()
                .item<double>(),
            0.99);
  EXPECT_LE(error.max().item<double>(), 1.0);
  EXPECT_LE((error.norm() / expected_f.norm().clamp_min(1e-30)).item<double>(),
            4e-3);
  EXPECT_GE((torch::dot(actual_f, expected_f) /
             (actual_f.norm() * expected_f.norm()).clamp_min(1e-30))
                .item<double>(),
            0.99999);
}

// cc_test's shared NPU environment owns initialization/finalization.
class H3RainfusionRuntimeTest : public ::testing::Test {};

TEST_F(H3RainfusionRuntimeTest, WorkspaceHasQualifiedShapes) {
  const auto geometry = production_geometry();
  ASSERT_TRUE(has_h3_rainfusion_specialization(geometry))
      << "H3 RainFusion AOT specializations are not compiled in";
  const auto device = torch::Device("npu:0");
  const auto& cap = kH3RainfusionQualifiedCapacity;
  const auto workspace = allocate_h3_rainfusion_workspace(geometry, device);

  EXPECT_EQ(workspace.record_count, kGoldenRecordCount);
  EXPECT_EQ(workspace.records_padded, kGoldenRecordsPadded);
  EXPECT_EQ(workspace.reorder_records.sizes(),
            c10::IntArrayRef({kGoldenRecordsPadded, 4}));
  EXPECT_EQ(workspace.plan_ids.sizes(),
            c10::IntArrayRef({cap.num_blocks, 4, cap.plan_width}));
  EXPECT_EQ(workspace.plan_count.sizes(),
            c10::IntArrayRef({cap.num_blocks, 8}));
  EXPECT_EQ(workspace.reordered_q.sizes(),
            c10::IntArrayRef({geometry.total_rows, 512}));
  EXPECT_EQ(workspace.q_pool.sizes(),
            c10::IntArrayRef({4, cap.q_pool_pad, 128}));
  EXPECT_EQ(workspace.k_pool.sizes(),
            c10::IntArrayRef({4, cap.k_pool_pad, 128}));
  EXPECT_EQ(workspace.score_ring.sizes(),
            c10::IntArrayRef({24, 2, 16, cap.k_pool_pad}));
  EXPECT_EQ(workspace.workspace_s.sizes(), c10::IntArrayRef({3, 24, 128, 512}));
  EXPECT_EQ(workspace.workspace_p.sizes(), c10::IntArrayRef({3, 24, 128, 512}));
  EXPECT_EQ(workspace.workspace_o.sizes(), c10::IntArrayRef({3, 24, 128, 128}));
  EXPECT_EQ(workspace.packed_k.sizes(),
            c10::IntArrayRef({4, cap.total_rows, 128}));
  EXPECT_EQ(workspace.reordered_output.sizes(),
            c10::IntArrayRef({geometry.total_rows, 512}));
  EXPECT_EQ(workspace.reorder_records.device(), device);
  const std::array<const torch::Tensor*, 15> tensors = {
      &workspace.reorder_records,
      &workspace.plan_ids,
      &workspace.plan_count,
      &workspace.reordered_q,
      &workspace.reordered_k,
      &workspace.reordered_v,
      &workspace.q_pool,
      &workspace.k_pool,
      &workspace.score_ring,
      &workspace.workspace_s,
      &workspace.workspace_p,
      &workspace.workspace_o,
      &workspace.packed_k,
      &workspace.packed_v,
      &workspace.reordered_output};
  int64_t workspace_bytes = 0;
  for (const torch::Tensor* tensor : tensors) {
    EXPECT_TRUE(tensor->is_contiguous());
    EXPECT_EQ(tensor->device(), device);
    workspace_bytes += tensor->nbytes();
  }
  EXPECT_EQ(workspace_bytes, 412722144);
  EXPECT_EQ(workspace_bytes + geometry.total_rows * 512 * 2, 474325984);
}

TEST_F(H3RainfusionRuntimeTest, UnqualifiedGeometryReportsNoSpecialization) {
  auto unaligned = production_geometry();
  unaligned.total_rows = 60161;
  EXPECT_FALSE(has_h3_rainfusion_specialization(unaligned));

  auto other_topk = production_geometry();
  other_topk.topk_r = 64;
  EXPECT_FALSE(has_h3_rainfusion_specialization(other_topk));
}

// The complete hot path on the current torch NPU stream: one workspace reused
// by two consecutive requests, with the pad rows left caller-owned.
TEST_F(H3RainfusionRuntimeTest, TwoSequentialRequestsReuseOneWorkspace) {
  const auto geometry = production_geometry();
  ASSERT_TRUE(has_h3_rainfusion_specialization(geometry))
      << "H3 RainFusion AOT specializations are not compiled in";
  const auto device = torch::Device("npu:0");
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  auto workspace = allocate_h3_rainfusion_workspace(geometry, device);
  const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);

  const int64_t rows = geometry.total_rows;
  std::vector<torch::Tensor> digests;
  for (int32_t request = 0; request < 2; ++request) {
    torch::manual_seed(1234 + request);
    const auto query = torch::randn({rows, 512}, options);
    const auto key = torch::randn({rows, 512}, options);
    const auto value = torch::randn({rows, 512}, options);
    // 7.0 is the pad canary: the kernels must never write rows
    // [valid_rows, total_rows).
    auto output = torch::full({rows, 512}, 7.0, options);

    h3_rainfusion_attention_out(query, key, value, geometry, workspace, output);
    const aclrtStream stream =
        c10_npu::getCurrentNPUStream(device.index()).stream();
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
        << "request " << request << " stream synchronize failed";

    const auto viewed = output.view({rows, 4, 128});
    const auto valid = viewed.slice(0, 0, geometry.valid_rows);
    EXPECT_TRUE(
        torch::isfinite(valid.toType(torch::kFloat32)).all().item<bool>())
        << "request " << request << " produced a non-finite output";
    EXPECT_GT(valid.count_nonzero().item<int64_t>(), 0)
        << "request " << request << " wrote nothing";
    EXPECT_TRUE(torch::eq(viewed.slice(0, geometry.valid_rows, rows), 7.0)
                    .all()
                    .item<bool>())
        << "request " << request << " clobbered the caller-owned pad rows";

    const auto counts =
        workspace.plan_count.slice(0, 0, derived.total_blocks).slice(1, 0, 4);
    EXPECT_EQ(counts.min().item<int32_t>() > 0, true);
    EXPECT_LE(counts.max().item<int32_t>(), 470);
    // Dense query rows (the video first frame) select every KV block.
    EXPECT_EQ(counts.slice(0, 0, derived.ff_blocks).min().item<int32_t>(), 470);
    digests.push_back(valid.toType(torch::kFloat32).sum());
  }
  // Different inputs must give different outputs, i.e. the second request did
  // not silently reuse a stale workspace result.
  EXPECT_NE(digests[0].item<float>(), digests[1].item<float>());
}

TEST_F(H3RainfusionRuntimeTest, MatchesIndependentCpuPlansAndAttention) {
  const H3RainfusionGeometry geometry = production_geometry();
  ASSERT_TRUE(has_h3_rainfusion_specialization(geometry));
  const torch::Device device("npu:0");
  torch::manual_seed(2024);
  const torch::Tensor query_cpu =
      torch::randn({60160, 4, 128}).to(torch::kBFloat16);
  const torch::Tensor key_cpu =
      torch::randn({60160, 4, 128}).to(torch::kBFloat16);
  const torch::Tensor value_cpu =
      torch::randn({60160, 4, 128}).to(torch::kBFloat16);
  const ReferenceSelector golden =
      reference_selector(query_cpu, key_cpu, value_cpu);
  const torch::Tensor query = query_cpu.view({60160, 512}).to(device);
  const torch::Tensor key = key_cpu.view({60160, 512}).to(device);
  const torch::Tensor value = value_cpu.view({60160, 512}).to(device);
  H3RainfusionWorkspace workspace =
      allocate_h3_rainfusion_workspace(geometry, device);
  torch::Tensor output = torch::full({60160, 512}, 7.0, query.options());
  h3_rainfusion_attention_out(query, key, value, geometry, workspace, output);
  ASSERT_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(0).stream()),
            ACL_SUCCESS);
  const H3RainfusionDerived derived = h3_rainfusion_derive(geometry);
  const auto plan_ids =
      workspace.plan_ids.cpu().slice(0, 0, derived.total_blocks);
  const auto plan_count = workspace.plan_count.cpu()
                              .slice(0, 0, derived.total_blocks)
                              .slice(1, 0, 4);
  ASSERT_TRUE(torch::equal(plan_count, golden.plan_count.slice(1, 0, 4)));
  for (int64_t block = 0; block < derived.total_blocks; ++block) {
    for (int64_t head = 0; head < kH3RainfusionHeads; ++head) {
      const int64_t count = golden.plan_count[block][head].item<int32_t>();
      ASSERT_TRUE(torch::equal(plan_ids[block][head].slice(0, 0, count),
                               golden.plan_ids[block][head].slice(0, 0, count)))
          << "block " << block << ", head " << head;
    }
  }
  EXPECT_TRUE(torch::equal(
      workspace.reordered_q.cpu().slice(0, 0, 60132).view({60132, 4, 128}),
      golden.query));
  EXPECT_TRUE(torch::equal(
      workspace.reordered_k.cpu().slice(0, 0, 60132).view({60132, 4, 128}),
      golden.key));
  EXPECT_TRUE(torch::equal(
      workspace.reordered_v.cpu().slice(0, 0, 60132).view({60132, 4, 128}),
      golden.value));
  const torch::Tensor actual = output.cpu().view({60160, 4, 128});
  EXPECT_TRUE(torch::eq(actual.slice(0, 60132), 7.0).all().item<bool>());
  const torch::Tensor permutation = reference_permutation();
  const int64_t* source_rows = permutation.data_ptr<int64_t>();
  const int32_t* plan = golden.plan_ids.data_ptr<int32_t>();
  const int32_t* counts = golden.plan_count.data_ptr<int32_t>();
  const std::array<int64_t, 7> blocks = {0, 7, 8, 100, 290, 291, 469};
  for (const int64_t block : blocks) {
    for (int64_t head = 0; head < 4; ++head) {
      std::vector<int64_t> selected_rows;
      for (int32_t index = 0; index < counts[block * 8 + head]; ++index) {
        const int64_t kv_block = plan[(block * 4 + head) * 472 + index];
        for (int64_t row = kv_block * 128;
             row < std::min((kv_block + 1) * 128, int64_t{60132});
             ++row) {
          selected_rows.push_back(row);
        }
      }
      const torch::Tensor indices = torch::tensor(selected_rows, torch::kInt64);
      const torch::Tensor selected_key = golden.key.index_select(0, indices)
                                             .select(1, head)
                                             .to(torch::kFloat32);
      const torch::Tensor selected_value = golden.value.index_select(0, indices)
                                               .select(1, head)
                                               .to(torch::kFloat32);
      const int64_t end = std::min((block + 1) * 128, int64_t{60132});
      for (const int64_t row : {block * 128, block * 128 + 63, end - 1}) {
        SCOPED_TRACE("block=" + std::to_string(block) + " head=" +
                     std::to_string(head) + " row=" + std::to_string(row));
        const torch::Tensor q = golden.query.select(0, row)
                                    .select(0, head)
                                    .to(torch::kFloat32)
                                    .unsqueeze(0);
        const torch::Tensor probabilities = torch::softmax(
            torch::matmul(q, selected_key.t()) / std::sqrt(128.0), -1);
        const torch::Tensor expected =
            torch::matmul(probabilities, selected_value)
                .to(torch::kBFloat16)
                .squeeze(0);
        expect_attention_precision(
            actual.select(0, source_rows[row]).select(0, head), expected);
      }
    }
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
