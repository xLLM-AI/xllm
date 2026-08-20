/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

// Fused MoE token index computation — 3 kernels replacing:
//   torch::bincount + 2 × torch::argsort + torch::cumsum + CPU sync
//
// Phase 1  histogram:      atomicAdd per-expert token counts
// Phase 2  prefix_sum:     1 block, exclusive scan → expert_offsets
// Phase 3  place_indices:  atomicAdd on offsets, write dst_src + src_dst
//
// expert_sizes  = per-expert token count [num_experts]  (preserved)
// expert_offsets = exclusive prefix sum of counts       (scratch, reused)

#include <c10/cuda/CUDAGuard.h>

#include <cub/block/block_scan.cuh>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::cuda {

constexpr int32_t kMoeIndexBlock = 256;

// ---- Phase 1: histogram ----
__global__ void
#ifdef USE_DCU
__launch_bounds__(kMoeIndexBlock, 1)
#endif
    moe_histogram_kernel(const int32_t* __restrict__ expert_id,
                         int32_t* __restrict__ expert_sizes,
                         int64_t num_elements,
                         int32_t num_experts) {
  int64_t tid = int64_t(blockIdx.x) * kMoeIndexBlock + threadIdx.x;
  if (tid < num_elements) {
    int32_t eid = expert_id[tid];
    if (eid >= 0 && eid < num_experts) {
      atomicAdd(&expert_sizes[eid], 1);
    }
  }
}

// ---- Phase 2: exclusive prefix sum (1 block) ----
// input:  expert_sizes  (per-expert counts)
// output: expert_offsets (exclusive scan of counts)
//         total_out      (total number of tokens, scalar)
__global__ void
#ifdef USE_DCU
__launch_bounds__(kMoeIndexBlock, 1)
#endif
    moe_prefix_sum_kernel(const int32_t* __restrict__ expert_sizes,
                          int32_t* __restrict__ expert_offsets,
                          int32_t num_experts,
                          int64_t* __restrict__ total_out) {
  using BlockScan = cub::BlockScan<int32_t, kMoeIndexBlock>;
  __shared__ typename BlockScan::TempStorage s_scan;

  int32_t val = (threadIdx.x < num_experts) ? expert_sizes[threadIdx.x] : 0;
  int32_t offset;
  BlockScan(s_scan).ExclusiveSum(val, offset);
  __syncthreads();

  // total = all elements sum = last thread's exclusive output + its input
  int32_t total = offset + val;

  if (threadIdx.x < num_experts) {
    expert_offsets[threadIdx.x] = offset;
  }
  if (threadIdx.x == 0 && total_out != nullptr) {
    *total_out = total;
  }
}

// ---- Phase 3: place indices ----
// atomicAdd on expert_offsets to assign a unique position within
// [start(e), start(e)+count(e)), then write both direction mappings.
__global__ void
#ifdef USE_DCU
__launch_bounds__(kMoeIndexBlock, 1)
#endif
    moe_place_indices_kernel(const int32_t* __restrict__ expert_id,
                             int32_t* __restrict__ expert_offsets,
                             int32_t* __restrict__ dst_src,
                             int32_t* __restrict__ src_dst,
                             int64_t num_elements,
                             int32_t num_experts) {
  int64_t flat_idx = int64_t(blockIdx.x) * kMoeIndexBlock + threadIdx.x;
  if (flat_idx >= num_elements) return;

  int32_t eid = expert_id[flat_idx];
  if (eid < 0 || eid >= num_experts) return;

  int32_t pos = atomicAdd(&expert_offsets[eid], 1);
  dst_src[pos] = static_cast<int32_t>(flat_idx);
  src_dst[flat_idx] = pos;
}

// ---- Host-side orchestrator ----
// Returns {src_dst, dst_src, expert_sizes}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_compute_index(
    const torch::Tensor& expert_id,
    int64_t num_experts) {
  auto device = expert_id.device();
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = expert_id.numel();
  int32_t E = static_cast<int32_t>(num_experts);
  CHECK_LE(E, kMoeIndexBlock) << "num_experts cannot exceed " << kMoeIndexBlock;
  auto expert_id_i32 = expert_id.to(torch::kInt32).contiguous();
  auto opt_i32 = expert_id_i32.options();

  auto expert_sizes = torch::zeros({num_experts}, opt_i32);
  auto expert_offsets = torch::empty({num_experts}, opt_i32);
  auto dst_src = torch::empty({N}, opt_i32);
  auto src_dst = torch::empty({N}, opt_i32);

  int64_t grid = (N + kMoeIndexBlock - 1) / kMoeIndexBlock;

  // Phase 1: histogram
  moe_histogram_kernel<<<grid, kMoeIndexBlock, 0, stream>>>(
      expert_id_i32.data_ptr<int32_t>(),
      expert_sizes.data_ptr<int32_t>(),
      N,
      E);

  // Phase 2: prefix sum (1 block)
  moe_prefix_sum_kernel<<<1, kMoeIndexBlock, 0, stream>>>(
      expert_sizes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      E,
      nullptr);

  // Phase 3: place indices
  moe_place_indices_kernel<<<grid, kMoeIndexBlock, 0, stream>>>(
      expert_id_i32.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      dst_src.data_ptr<int32_t>(),
      src_dst.data_ptr<int32_t>(),
      N,
      E);

  return std::make_tuple(src_dst, dst_src, expert_sizes);
}

}  // namespace xllm::kernel::cuda
