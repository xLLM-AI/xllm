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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "xattention_ops_api.h"

namespace {

// In-place cache selection kernel for Xattention.
// Reorders KV cache entries based on beam search results. After beam search,
// the beam indices may have changed, and this kernel copies KV cache data from
// old beam positions to new beam positions to maintain consistency.
// Inputs:
//   k_ptrs_i64      : [Layer] - pointers to K cache tensors for each layer
//   v_ptrs_i64      : [Layer] - pointers to V cache tensors for each layer
//   beam_index      : [B*Beam] - mapping from new beam index to old beam index
//   block_table     : [B] - request ID per batch item (extracted from [B*Beam,
//   1]) B               : batch size (actual batch size, not batch_size *
//   beam_size) Beam            : beam width Kv              : number of KV
//   heads MaxStep         : maximum decode steps D               : head
//   dimension MaxReq          : maximum number of requests Layer           :
//   number of transformer layers decode_step     : current decode step
//   (0-indexed)
// Cache layout: [MaxReq, Beam, MaxStep, Kv, D]
// The kernel performs two passes to avoid overwriting data:
//   pass-1: copy from old_beam > new_beam (increasing new_beam)
//   pass-2: copy from old_beam < new_beam (decreasing new_beam)
template <typename scalar_t>
__global__ void cache_select_inplace_ptrs_kernel(
    const int64_t* __restrict__ k_ptrs_i64,   // [Layer]
    const int64_t* __restrict__ v_ptrs_i64,   // [Layer]
    const int32_t* __restrict__ beam_index,   // [B*Beam]
    const int32_t* __restrict__ block_table,  // [B]
    int32_t B,
    int32_t Beam,
    int32_t Kv,
    int32_t MaxStep,
    int32_t D,
    int32_t MaxReq,
    int32_t Layer,
    int32_t decode_step) {
  const int32_t b = static_cast<int32_t>(blockIdx.x);
  const int32_t kv = static_cast<int32_t>(blockIdx.y);
  const int32_t layer = static_cast<int32_t>(blockIdx.z);

  if (b >= B || kv >= Kv || layer >= Layer) {
    return;
  }

  const int32_t step_end =
      decode_step < (MaxStep - 1) ? decode_step : (MaxStep - 1);

  const int32_t req = block_table[b];
  if (req < 0 || req >= MaxReq) {
    return;
  }

  scalar_t* __restrict__ k_cache =
      reinterpret_cast<scalar_t*>(static_cast<uintptr_t>(k_ptrs_i64[layer]));
  scalar_t* __restrict__ v_cache =
      reinterpret_cast<scalar_t*>(static_cast<uintptr_t>(v_ptrs_i64[layer]));

  // base(req, beam, s, kv, d) = ((((req*Beam + beam)*MaxStep + s)*Kv + kv) * D
  // + d)
  const int64_t req_base = static_cast<int64_t>(req) * Beam;
  const int64_t step_kv_stride = static_cast<int64_t>(Kv) * D;
  const int64_t kv_d_base = static_cast<int64_t>(kv) * D;

  // grid_step is typically small; loop over s in-kernel to reduce launch
  // blocks.
  for (int32_t s = 0; s <= step_end; ++s) {
    // pass-1: new_beam increasing, copy if old_beam > new_beam
    for (int32_t new_beam = 0; new_beam < Beam; ++new_beam) {
      const int32_t old_beam = beam_index[b * Beam + new_beam] / Beam;
      if (old_beam >= 0 && old_beam < Beam && old_beam > new_beam) {
        const int64_t dst_base =
            ((req_base + new_beam) * MaxStep + s) * step_kv_stride + kv_d_base;
        const int64_t src_base =
            ((req_base + old_beam) * MaxStep + s) * step_kv_stride + kv_d_base;
        for (int32_t d = static_cast<int32_t>(threadIdx.x); d < D;
             d += static_cast<int32_t>(blockDim.x)) {
          k_cache[dst_base + d] = k_cache[src_base + d];
          v_cache[dst_base + d] = v_cache[src_base + d];
        }
      }
    }

    // pass-2: new_beam decreasing, copy if old_beam < new_beam
    for (int32_t new_beam = Beam - 1; new_beam >= 0; --new_beam) {
      const int32_t old_beam = beam_index[b * Beam + new_beam] / Beam;
      if (old_beam >= 0 && old_beam < Beam && old_beam < new_beam) {
        const int64_t dst_base =
            ((req_base + new_beam) * MaxStep + s) * step_kv_stride + kv_d_base;
        const int64_t src_base =
            ((req_base + old_beam) * MaxStep + s) * step_kv_stride + kv_d_base;
        for (int32_t d = static_cast<int32_t>(threadIdx.x); d < D;
             d += static_cast<int32_t>(blockDim.x)) {
          k_cache[dst_base + d] = k_cache[src_base + d];
          v_cache[dst_base + d] = v_cache[src_base + d];
        }
      }
    }
  }
}

void cache_select_cuda_launch_ptrs(
    torch::Tensor k0,
    torch::Tensor v0,
    torch::Tensor k_ptrs_i64,       // [Layer] int64 (CUDA)
    torch::Tensor v_ptrs_i64,       // [Layer] int64 (CUDA)
    torch::Tensor beam_index_i32,   // [B*Beam, 1] int32
    torch::Tensor block_table_i32,  // [B] int32
    int64_t decode_step,
    int64_t layer_num) {
  CHECK(k_ptrs_i64.is_cuda() && v_ptrs_i64.is_cuda())
      << "k_ptrs_i64/v_ptrs_i64 must be CUDA";
  CHECK_EQ(k_ptrs_i64.scalar_type(), torch::kInt64)
      << "k_ptrs_i64/v_ptrs_i64 must be int64";
  CHECK_EQ(v_ptrs_i64.scalar_type(), torch::kInt64)
      << "k_ptrs_i64/v_ptrs_i64 must be int64";
  CHECK(k_ptrs_i64.is_contiguous() && v_ptrs_i64.is_contiguous())
      << "k_ptrs_i64/v_ptrs_i64 must be contiguous";

  const int64_t B64 = block_table_i32.size(0);
  const int64_t Beam64 = k0.size(1);
  const int64_t MaxStep64 = k0.size(2);
  const int64_t Kv64 = k0.size(3);
  const int64_t D64 = k0.size(4);
  const int64_t MaxReq64 = k0.size(0);
  const int64_t Layer64 = layer_num;

  const int32_t B = static_cast<int32_t>(B64);
  const int32_t Beam = static_cast<int32_t>(Beam64);
  const int32_t Kv = static_cast<int32_t>(Kv64);
  const int32_t MaxStep = static_cast<int32_t>(MaxStep64);
  const int32_t D = static_cast<int32_t>(D64);
  const int32_t MaxReq = static_cast<int32_t>(MaxReq64);
  const int32_t Layer = static_cast<int32_t>(Layer64);
  const int32_t decode_step_i32 = static_cast<int32_t>(decode_step);

  // Warp-aligned threads, capped to keep occupancy reasonable.
  int threads_per_block = ((D + 31) / 32) * 32;
  if (threads_per_block < 32) {
    threads_per_block = 32;
  }
  if (threads_per_block > 256) {
    threads_per_block = 256;
  }
  dim3 block_dim(static_cast<unsigned int>(threads_per_block), 1, 1);

  CHECK_LE(Kv64, static_cast<int64_t>(UINT32_MAX)) << "Kv too large for grid.y";
  CHECK_LE(Layer64, 65535) << "layer_num too large for grid.z";
  dim3 grid_dim(static_cast<unsigned int>(B),
                static_cast<unsigned int>(Kv),
                static_cast<unsigned int>(Layer));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(torch::ScalarType::Half,
                                  torch::ScalarType::BFloat16,
                                  k0.scalar_type(),
                                  "cache_select_inplace_ptrs_kernel",
                                  [&] {
                                    cache_select_inplace_ptrs_kernel<scalar_t>
                                        <<<grid_dim, block_dim, 0, stream>>>(
                                            k_ptrs_i64.data_ptr<int64_t>(),
                                            v_ptrs_i64.data_ptr<int64_t>(),
                                            beam_index_i32.data_ptr<int32_t>(),
                                            block_table_i32.data_ptr<int32_t>(),
                                            B,
                                            Beam,
                                            Kv,
                                            MaxStep,
                                            D,
                                            MaxReq,
                                            Layer,
                                            decode_step_i32);
                                  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

namespace xllm::kernel::cuda {
void cache_select(const torch::Tensor& beam_index,  // [B*Beam, 1]
                  std::vector<torch::Tensor>& unshared_k_cache,
                  std::vector<torch::Tensor>& unshared_v_cache,
                  const torch::Tensor& block_table,  // [B*Beam, 1]
                  int64_t decode_step,
                  int64_t beam_size,
                  int64_t layer_num) {
  CHECK_GE(layer_num, 0) << "layer_num must be >= 0";
  if (layer_num == 0) {
    return;
  }
  CHECK_EQ(static_cast<int64_t>(unshared_k_cache.size()), layer_num)
      << "unshared_k_cache length mismatch";
  CHECK_EQ(static_cast<int64_t>(unshared_v_cache.size()), layer_num)
      << "unshared_v_cache length mismatch";

  CHECK(beam_index.is_cuda()) << "beam_index must be CUDA";
  CHECK(block_table.is_cuda()) << "block_table must be CUDA";
  CHECK_EQ(block_table.dim(), 2) << "block_table must be [B*Beam, 1]";
  CHECK_EQ(block_table.size(1), 1) << "block_table must be [B*Beam, 1]";
  CHECK_EQ(beam_index.dim(), 2) << "beam_index must be [B*Beam, 1]";
  CHECK_EQ(beam_index.size(1), 1) << "beam_index must be [B*Beam, 1]";
  CHECK_GE(decode_step, 0) << "decode_step must be >= 0";
  CHECK_GT(beam_size, 0) << "beam_size must be > 0";

  // block_table is [B*Beam, 1] with sequential values [0,1,2,3,...]
  // Infer actual batch_size
  CHECK_EQ(block_table.size(0) % beam_size, 0)
      << "block_table.size(0) must be divisible by beam_size";
  const int64_t B = block_table.size(0) / beam_size;
  CHECK_EQ(beam_index.size(0), B * beam_size)
      << "beam_index size mismatch with B*beam_size";

  // Prepare indices (int32, contiguous).
  auto beam_index_i32 = beam_index.to(torch::kInt32).contiguous();
  auto block_table_i32 = torch::arange(
      0,
      B,
      torch::TensorOptions().dtype(torch::kInt32).device(block_table.device()));
  // Validate shapes/dtypes against layer 0.
  const auto& k0 = unshared_k_cache[0];
  const auto& v0 = unshared_v_cache[0];
  CHECK(k0.is_cuda() && v0.is_cuda()) << "cache must be CUDA";
  CHECK(k0.is_contiguous() && v0.is_contiguous()) << "cache must be contiguous";
  CHECK_EQ(k0.dim(), 5) << "cache must be 5D [MaxReq, Beam, MaxStep, Kv, D]";
  CHECK_EQ(v0.sizes(), k0.sizes()) << "k/v cache shapes must match";
  CHECK_EQ(k0.size(1), beam_size) << "beam_size mismatch with cache";
  CHECK_LT(decode_step, k0.size(2)) << "decode_step must be < max_decode_step";

  // Pack layer pointers into CUDA int64 tensors so we can launch once.
  // Note: pointer values are produced on host (data_ptr()), then copied to GPU.
  c10::cuda::CUDAGuard device_guard(k0.device());
  auto ptr_cuda_opts =
      torch::TensorOptions().dtype(torch::kInt64).device(k0.device());
  auto k_ptrs_i64 = torch::empty({layer_num}, ptr_cuda_opts);
  auto v_ptrs_i64 = torch::empty({layer_num}, ptr_cuda_opts);
  std::vector<int64_t> k_ptrs_host(static_cast<size_t>(layer_num));
  std::vector<int64_t> v_ptrs_host(static_cast<size_t>(layer_num));

  for (int64_t layer = 0; layer < layer_num; ++layer) {
    auto k = unshared_k_cache[static_cast<size_t>(layer)];
    auto v = unshared_v_cache[static_cast<size_t>(layer)];
    CHECK(k.is_cuda() && v.is_cuda()) << "cache must be CUDA";
    CHECK(k.is_contiguous() && v.is_contiguous()) << "cache must be contiguous";
    CHECK_EQ(k.sizes(), k0.sizes()) << "all layers must have same cache shape";
    CHECK_EQ(v.sizes(), k0.sizes()) << "all layers must have same cache shape";
    CHECK_EQ(k.scalar_type(), k0.scalar_type())
        << "all layers must have same dtype";
    CHECK_EQ(v.scalar_type(), k0.scalar_type())
        << "all layers must have same dtype";
    CHECK_EQ(k.get_device(), k0.get_device())
        << "all layers must be on the same CUDA device";
    CHECK_EQ(v.get_device(), k0.get_device())
        << "all layers must be on the same CUDA device";

    k_ptrs_host[static_cast<size_t>(layer)] =
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(k.data_ptr()));
    v_ptrs_host[static_cast<size_t>(layer)] =
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(v.data_ptr()));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(
      cudaMemcpyAsync(k_ptrs_i64.data_ptr<int64_t>(),
                      k_ptrs_host.data(),
                      static_cast<size_t>(layer_num) * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      stream));
  C10_CUDA_CHECK(
      cudaMemcpyAsync(v_ptrs_i64.data_ptr<int64_t>(),
                      v_ptrs_host.data(),
                      static_cast<size_t>(layer_num) * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      stream));

  cache_select_cuda_launch_ptrs(k0,
                                v0,
                                k_ptrs_i64,
                                v_ptrs_i64,
                                beam_index_i32,
                                block_table_i32,
                                decode_step,
                                layer_num);
}

}  // namespace xllm::kernel::cuda
