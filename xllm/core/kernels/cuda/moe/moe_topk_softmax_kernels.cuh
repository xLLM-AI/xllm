// Adapt from
// https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/moe/topk_softmax_kernels.cu
// which is originally adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cub/util_type.cuh>

#if !defined(USE_DCU) && !defined(USE_MACA)
#include <cuda/functional>
#endif

#include "kernels/cuda/device_utils.cuh"

using cub_kvp = cub::KeyValuePair<int, float>;

namespace {

using namespace xllm::kernel::cuda;

#if defined(USE_DCU)
static constexpr unsigned long long kSoftmaxFullMask = 0xffffffffffffffffULL;
#else
static constexpr unsigned int kSoftmaxFullMask = 0xffffffffU;
#endif

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <typename T, int TPB>
__launch_bounds__(TPB) __global__
    void moe_softmax(const T* input,
                     const bool* finished,
                     float* output,
                     const int num_cols,
                     const float moe_softcapping,
                     const float* correction_bias) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  const int thread_row_offset = blockIdx.x * num_cols;

  float threadData(-FLT_MAX);

  // Don't touch finished rows.
  if ((finished != nullptr) && finished[blockIdx.x]) {
    return;
  }

  // First pass: Apply transformation, find max, and write transformed values to
  // output
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    float val = convert_to_float<T>(input[idx]);

    // Apply tanh softcapping if enabled
    if (moe_softcapping != 0.0f) {
      val = tanhf(val / moe_softcapping) * moe_softcapping;
    }

    // Apply correction bias if provided
    if (correction_bias != nullptr) {
      val = val + correction_bias[ii];
    }

    output[idx] = val;  // Store transformed value
    threadData = max(val, threadData);
  }

  const float maxElem =
      BlockReduce(tmpStorage).Reduce(threadData, MaxReduceOp());

  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  // Second pass: Compute sum using transformed values from output
  threadData = 0;
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((output[idx] - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Sum(threadData);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  // Third pass: Compute final softmax using transformed values from output
  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float softmax_val =
        exp((output[idx] - float_max)) * normalizing_factor;
    output[idx] = softmax_val;
  }
}

namespace moe {
class TopKPair {
 public:
  static constexpr int kPair = 2;
  static constexpr int kMaxIndex = 0;
  cub_kvp max;
  cub_kvp secondMax;

  __device__ TopKPair() {}
  __device__ TopKPair(cub_kvp max, cub_kvp secondMax)
      : max(max), secondMax(secondMax) {}
};

class TopKPairArgMax {
 public:
  __device__ TopKPairArgMax() {}
  __device__ __forceinline__ TopKPair
  operator()(const TopKPair& candidate1, const TopKPair& candidate2) const {
    cub_kvp globalMax, globalSecondMax;

    // Determine the global maximum
    if (candidate1.max.value > candidate2.max.value) {
      globalMax = candidate1.max;
    } else {
      globalMax = candidate2.max;
    }

    // Determine the global second maximum
    if (globalMax.key == candidate1.max.key) {
      // If candidate1 contributed the max, compare its secondMax with
      // candidate2's max
      globalSecondMax = (candidate1.secondMax.value > candidate2.max.value)
                            ? candidate1.secondMax
                            : candidate2.max;
    } else {
      // If candidate2 contributed the max, compare its secondMax with
      // candidate1's max
      globalSecondMax = (candidate2.secondMax.value > candidate1.max.value)
                            ? candidate2.secondMax
                            : candidate1.max;
    }
    return TopKPair(globalMax, globalSecondMax);
  }
};
}  // namespace moe

template <int TPB>
__launch_bounds__(TPB) __global__
    void moe_topk_fast(float* inputs_after_softmax,
                       const bool* finished,
                       float* output,
                       int* indices,
                       const int num_experts,
                       const int k,
                       const int start_expert,
                       const int end_expert,
                       const bool renormalize) {
  using namespace moe;
  using BlockReduce = cub::BlockReduce<TopKPair, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;
  TopKPair thread_pair;

  const int block_row = blockIdx.x;

  const bool row_is_active = finished ? !finished[block_row] : true;
  const int thread_read_offset = blockIdx.x * num_experts;
  float row_sum_for_renormalize = 0;
  // Each loop finds the top 2 elements,
  // thus requiring only ceil(k / 2) loops (calculated as (k + 1) / 2).
  for (int k_idx = 0; k_idx < (k + TopKPair::kPair - 1) / TopKPair::kPair;
       ++k_idx) {
    // Initializing the top 2 elements by the minimum value.
    thread_pair.max.key = 0;
    thread_pair.max.value = -1.f;
    thread_pair.secondMax.key = 0;
    thread_pair.secondMax.value = -1.f;

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];
      // updating the thread_pair according to inp_kvp's value
      if (inp_kvp.value > thread_pair.max.value) {
        thread_pair.secondMax = thread_pair.max;
        thread_pair.max = inp_kvp;
      } else if (inp_kvp.value > thread_pair.secondMax.value) {
        thread_pair.secondMax = inp_kvp;
      }
    }

    TopKPairArgMax reducer;
    const TopKPair result_pair =
        BlockReduce(tmpStorage).Reduce(thread_pair, reducer);
    if (threadIdx.x == 0) {
#pragma unroll
      // updating 2 elements to the result.
      for (int i = 0; i < TopKPair::kPair; i++) {
        if (k_idx * 2 + i >= k) {
          break;
        }
        cub_kvp result = (i == TopKPair::kMaxIndex) ? result_pair.max
                                                    : result_pair.secondMax;
        int expert = result.key;
        bool node_uses_expert = expert >= start_expert && expert < end_expert;
        bool should_process_row = row_is_active && node_uses_expert;
        // The inputs_after_softmax is modified in-place to avoid unnecessary
        // loops for finding the top k-1 value. 1.f represents the minimum
        // value.
        inputs_after_softmax[thread_read_offset + expert] = -1.f;
        int idx = k * block_row + k_idx * 2 + i;
        output[idx] = result.value;
        indices[idx] =
            should_process_row ? (expert - start_expert) : num_experts;
        assert(indices[idx] >= 0);
        row_sum_for_renormalize += result.value;
      }
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * block_row + k_idx;
      output[idx] = output[idx] * row_sum_for_renormalize_inv;
    }
  }
}

template <int TPB>
__launch_bounds__(TPB) __global__ void moe_topK(float* inputs_after_softmax,
                                                const bool* finished,
                                                float* output,
                                                int* indices,
                                                const int num_experts,
                                                const int k,
                                                const int start_expert,
                                                const int end_expert,
                                                const bool renormalize) {
  using cub_kvp = cub::KeyValuePair<int, float>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x;

  const bool row_is_active = finished ? !finished[block_row] : true;
  const int thread_read_offset = blockIdx.x * num_experts;
  float row_sum_for_renormalize = 0;
  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = -1.f;  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];
      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      // Ignore experts the node isn't responsible for with expert parallelism
      const int expert = result_kvp.key;
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      const int idx = k * block_row + k_idx;
      output[idx] = result_kvp.value;
      indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
      assert(indices[idx] >= 0);
      row_sum_for_renormalize += result_kvp.value;
      // The inputs_after_softmax is modified in-place to avoid unnecessary
      // loops for finding the top k-1 value. 1.f represents the minimum value.
      inputs_after_softmax[thread_read_offset + expert] = -1.f;
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * block_row + k_idx;
      output[idx] = output[idx] * row_sum_for_renormalize_inv;
    }
  }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k.
*/

template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk_gating_softmax(const T* input,
                             const bool* finished,
                             float* output,
                             const int num_rows,
                             int* indices,
                             const int k,
                             const int start_expert,
                             const int end_expert,
                             const bool renormalize,
                             const float moe_softcapping,
                             const float* correction_bias) {
  // We begin by enforcing compile time assertions and setting up compile time
  // constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int kEltsPerLdg = BYTES_PER_LDG / sizeof(T);
  static constexpr int kEltsPerRow = NUM_EXPERTS;
  static constexpr int kThreadsPerRow = kEltsPerRow / VPT;
  static constexpr int kLdgPerThread = VPT / kEltsPerLdg;

  // Restrictions based on previous section.
  static_assert(
      VPT % kEltsPerLdg == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % kThreadsPerRow == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(kThreadsPerRow == (kThreadsPerRow & -kThreadsPerRow),
                "THREADS_PER_ROW must be power of 2");
  static_assert(kThreadsPerRow <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int kEltsPerWarp = WARP_SIZE * VPT;
  static constexpr int kRowsPerWarp = kEltsPerWarp / kEltsPerRow;
  static constexpr int kRowsPerCta = WARPS_PER_CTA * kRowsPerWarp;

  // Restrictions for previous section.
  static_assert(kEltsPerWarp % kEltsPerRow == 0,
                "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time
  // variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a
  // block contains WARPS_PER_CTA warps. This, each block processes a chunk of
  // rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * kRowsPerCta;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * kRowsPerWarp;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / kThreadsPerRow;
  const int thread_row = warp_base_row + thread_row_in_warp;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows) {
    return;
  }
  const bool row_is_active = finished ? !finished[thread_row] : true;

  // We finally start setting up the read pointers for each thread. First, each
  // thread jumps to the start of the row it will read.
  const T* thread_row_ptr = input + thread_row * kEltsPerRow;

  // Now, we compute the group each thread belong to in order to determine the
  // first column to start loads.
  const int thread_group_idx = threadIdx.x % kThreadsPerRow;
  const int first_elt_read_by_thread = thread_group_idx * kEltsPerLdg;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  // Determine the pointer type to use to read in the data depending on the
  // BYTES_PER_LDG template param. In theory, this can support all powers of 2
  // up to 16. NOTE(woosuk): The original implementation uses CUTLASS aligned
  // array here. We defined our own aligned array and use it here to avoid the
  // dependency on CUTLASS.
  using AccessType = AlignedArray<T, kEltsPerLdg>;

  // Finally, we pull in the data from global mem
  T row_chunk_temp[VPT];
  AccessType* row_chunk_vec_ptr =
      reinterpret_cast<AccessType*>(&row_chunk_temp);
  const AccessType* vec_thread_read_ptr =
      reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  // Note(Byron): interleaved loads to achieve better memory coalescing
  // | thread[0] | thread[1] | thread[2] | thread[3] | thread[0] | thread[1] |
  // thread[2] | thread[3] | ...
  for (int ii = 0; ii < kLdgPerThread; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * kThreadsPerRow];
  }

  float row_chunk[VPT];
#pragma unroll
  // Note(Byron): upcast logits to float32
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = convert_to_float<T>(row_chunk_temp[ii]);
  }

  // Apply tanh softcapping and correction bias
  if (moe_softcapping != 0.0f || correction_bias != nullptr) {
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      float val = row_chunk[ii];

      // Apply tanh softcapping if enabled
      if (moe_softcapping != 0.0f) {
        val = tanhf(val / moe_softcapping) * moe_softcapping;
      }

      // Apply correction bias if provided
      if (correction_bias != nullptr) {
        /*
        LDG is interleaved
        |thread0 LDG| |thread1 LDG| |thread0 LDG| |thread1 LDG|
        |--------- group0 --------| |----------group1 --------|
                                      ^ local2
        */
        const int group_id = ii / kEltsPerLdg;
        const int local_id = ii % kEltsPerLdg;
        const int expert_idx = first_elt_read_by_thread +
                               group_id * kThreadsPerRow * kEltsPerLdg +
                               local_id;
        val = val + correction_bias[expert_idx];
      }

      row_chunk[ii] = val;
    }
  }

  // First, we perform a max reduce within the thread. We can do the max in fp16
  // safely (I think) and just convert to float afterwards for the exp + sum
  // reduction.
  float thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

  /*********************************/
  /********* Softmax Begin *********/
  /*********************************/

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce. lane id: 0-31 within a warp
#pragma unroll
  for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
    // butterfly reduce with (lane id ^ mask)
    thread_max = max(thread_max,
                     XLLM_SHFL_XOR_SYNC_WIDTH(
                         kSoftmaxFullMask, thread_max, mask, kThreadsPerRow));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp.
  // We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
    row_sum += XLLM_SHFL_XOR_SYNC_WIDTH(
        kSoftmaxFullMask, row_sum, mask, kThreadsPerRow);
  }

  // From this point, all threads have the max and the sum for their rows in the
  // thread_max and thread_sum variables respectively. Finally, we can scale the
  // rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only
  // compute for the top-k values in the row. However, this kernel will likely
  // not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
  }
  /*******************************/
  /********* Softmax End *********/
  /*******************************/

  // Now, softmax_res contains the softmax of the row chunk. Now, I want to find
  // the topk elements in each row, along with the max index.
  int start_col = first_elt_read_by_thread;
  static constexpr int kColsPerGroupLdg = kEltsPerLdg * kThreadsPerRow;

  float row_sum_for_renormalize = 0;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < kLdgPerThread;
         ++ldg, col += kColsPerGroupLdg) {
#pragma unroll
      for (int ii = 0; ii < kEltsPerLdg; ++ii) {
        float val = row_chunk[ldg * kEltsPerLdg + ii];

        // No check on the experts here since columns with the smallest index
        // are processed first and only updated if > (not >=)
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
      float other_max = XLLM_SHFL_XOR_SYNC_WIDTH(
          kSoftmaxFullMask, max_val, mask, kThreadsPerRow);
      int other_expert = XLLM_SHFL_XOR_SYNC_WIDTH(
          kSoftmaxFullMask, expert, mask, kThreadsPerRow);

      // We want lower indices to "win" in every thread so we break ties this
      // way
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    if (thread_group_idx == 0) {
      // Add a guard to ignore experts not included by this node
      const bool node_uses_expert =
          expert >= start_expert && expert < end_expert;
      const bool should_process_row = row_is_active && node_uses_expert;

      // The lead thread from each sub-group will write out the final results to
      // global memory. (This will be a single) thread per row of the
      // input/output matrices.
      const int idx = k * thread_row + k_idx;
      output[idx] = max_val;
      indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
      row_sum_for_renormalize += max_val;
    }

    // Finally, we clear the value in the thread with the current max if there
    // is another iteration to run.
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / kColsPerGroupLdg;
      const int thread_to_clear_in_group =
          (expert / kEltsPerLdg) % kThreadsPerRow;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % kEltsPerLdg;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk[ldg_group_for_expert * kEltsPerLdg + offset_for_expert] =
            -10000.f;
      }
    }
  }

  // Fuse renormalization of topk_weights into this kernel
  if (renormalize && thread_group_idx == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int idx = k * thread_row + k_idx;
      output[idx] = output[idx] * row_sum_for_renormalize_inv;
    }
  }
}

template <typename T, int EXPERTS, int WARPS_PER_TB>
void topk_gating_softmax_launcher_helper(const T* input,
                                         const bool* finished,
                                         float* output,
                                         int* indices,
                                         const int num_rows,
                                         const int k,
                                         const int start_expert,
                                         const int end_expert,
                                         const bool renormalize,
                                         const float moe_softcapping,
                                         const float* correction_bias,
                                         cudaStream_t stream) {
  static constexpr std::size_t kMaxBytesPerLdg = 16;

  static constexpr int kBytesPerLdg = MIN(kMaxBytesPerLdg, sizeof(T) * EXPERTS);
  using Constants = TopkConstants<T, EXPERTS, kBytesPerLdg>;
  static constexpr int kVpt = Constants::VPT;
  static constexpr int kRowsPerWarp = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + kRowsPerWarp - 1) / kRowsPerWarp;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  topk_gating_softmax<T, kVpt, EXPERTS, WARPS_PER_TB, kBytesPerLdg>
      <<<num_blocks, block_dim, 0, stream>>>(input,
                                             finished,
                                             output,
                                             num_rows,
                                             indices,
                                             k,
                                             start_expert,
                                             end_expert,
                                             renormalize,
                                             moe_softcapping,
                                             correction_bias);
}

#define LAUNCH_SOFTMAX(TYPE, NUM_EXPERTS, WARPS_PER_TB)                 \
  topk_gating_softmax_launcher_helper<TYPE, NUM_EXPERTS, WARPS_PER_TB>( \
      gating_output,                                                    \
      nullptr,                                                          \
      topk_weights,                                                     \
      topk_indices,                                                     \
      num_tokens,                                                       \
      topk,                                                             \
      0,                                                                \
      num_experts,                                                      \
      renormalize,                                                      \
      moe_softcapping,                                                  \
      correction_bias,                                                  \
      stream);

template <typename T>
void topk_gating_softmax_kernel_launcher(const T* gating_output,
                                         float* topk_weights,
                                         int* topk_indices,
                                         float* softmax_workspace,
                                         const int num_tokens,
                                         const int num_experts,
                                         const int topk,
                                         const bool renormalize,
                                         const float moe_softcapping,
                                         const float* correction_bias,
                                         cudaStream_t stream) {
  static constexpr int kWarpsPerTb = 4;
  switch (num_experts) {
    case 1:
      LAUNCH_SOFTMAX(T, 1, kWarpsPerTb);
      break;
    case 2:
      LAUNCH_SOFTMAX(T, 2, kWarpsPerTb);
      break;
    case 4:
      LAUNCH_SOFTMAX(T, 4, kWarpsPerTb);
      break;
    case 8:
      LAUNCH_SOFTMAX(T, 8, kWarpsPerTb);
      break;
    case 16:
      LAUNCH_SOFTMAX(T, 16, kWarpsPerTb);
      break;
    case 32:
      LAUNCH_SOFTMAX(T, 32, kWarpsPerTb);
      break;
    case 64:
      LAUNCH_SOFTMAX(T, 64, kWarpsPerTb);
      break;
    case 128:
      LAUNCH_SOFTMAX(T, 128, kWarpsPerTb);
      break;
    case 256:
      LAUNCH_SOFTMAX(T, 256, kWarpsPerTb);
      break;
    default: {
      CHECK(softmax_workspace != nullptr)
          << "softmax_workspace must be provided for num_experts that are "
             "not a power of 2.";
      static constexpr int kTpb = 256;
      moe_softmax<T, kTpb><<<num_tokens, kTpb, 0, stream>>>(gating_output,
                                                            nullptr,
                                                            softmax_workspace,
                                                            num_experts,
                                                            moe_softcapping,
                                                            correction_bias);
      if (topk == 1) {
        // Note: As an optimization for better performance,
        // the softmax_workspace is overwritten in-place by both moeTopK and
        // moe_topk_fast.
        moe_topK<kTpb><<<num_tokens, kTpb, 0, stream>>>(softmax_workspace,
                                                        nullptr,
                                                        topk_weights,
                                                        topk_indices,
                                                        num_experts,
                                                        topk,
                                                        0,
                                                        num_experts,
                                                        renormalize);
      } else {
        moe_topk_fast<kTpb><<<num_tokens, kTpb, 0, stream>>>(softmax_workspace,
                                                             nullptr,
                                                             topk_weights,
                                                             topk_indices,
                                                             num_experts,
                                                             topk,
                                                             0,
                                                             num_experts,
                                                             renormalize);
      }
    }
  }
}
}  // namespace

namespace xllm::kernel::cuda {
void topk_softmax(torch::Tensor& topk_weights,   // [num_tokens, topk]
                  torch::Tensor& topk_indices,   // [num_tokens, topk]
                  torch::Tensor& gating_output,  // [num_tokens, num_experts]
                  const bool renormalize,
                  const double moe_softcapping,
                  const std::optional<torch::Tensor>& correction_bias) {
  // Check data type
  CHECK(gating_output.scalar_type() == at::ScalarType::Float ||
        gating_output.scalar_type() == at::ScalarType::Half ||
        gating_output.scalar_type() == at::ScalarType::BFloat16)
      << "gating_output must be float32, float16, or bfloat16";

  // Check dimensions
  CHECK(gating_output.dim() == 2)
      << "gating_output must be 2D tensor [num_tokens, num_experts]";
  CHECK(topk_weights.dim() == 2)
      << "topk_weights must be 2D tensor [num_tokens, topk]";
  CHECK(topk_indices.dim() == 2)
      << "topk_indices must be 2D tensor [num_tokens, topk]";

  // Check shapes
  CHECK(gating_output.size(0) == topk_weights.size(0))
      << "First dimension of topk_weights must match num_tokens in "
         "gating_output"
      << "First dimension of topk_indices must match num_tokens in "
         "gating_output";

  CHECK(topk_weights.size(-1) == topk_indices.size(-1))
      << "Second dimension of topk_indices must match topk in topk_weights"
      << "topk must be less than or equal to num_experts";

  const int num_experts = static_cast<int>(gating_output.size(-1));
  const int num_tokens = static_cast<int>(gating_output.size(0));
  const int topk = static_cast<int>(topk_weights.size(-1));

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size = needs_workspace ? num_tokens * num_experts : 0;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor softmax_workspace = torch::empty(
      {workspace_size}, gating_output.options().dtype(at::ScalarType::Float));

  const at::ScalarType dtype = gating_output.scalar_type();

  // Validate correction_bias if provided - must always be float32
  const float* bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    const torch::Tensor& bias_tensor = correction_bias.value();
    CHECK(bias_tensor.dim() == 1)
        << "correction_bias must be 1D tensor [num_experts]";
    CHECK(bias_tensor.size(0) == num_experts)
        << "correction_bias size must match num_experts";
    CHECK(bias_tensor.scalar_type() == at::ScalarType::Float)
        << "correction_bias must be float32, got " << bias_tensor.scalar_type();
    bias_ptr = bias_tensor.data_ptr<float>();
  }

  // Cast moe_softcapping from double to float for CUDA kernels
  const float moe_softcapping_f = static_cast<float>(moe_softcapping);

  if (dtype == at::ScalarType::Float) {
    topk_gating_softmax_kernel_launcher<float>(
        gating_output.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        softmax_workspace.data_ptr<float>(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        moe_softcapping_f,
        bias_ptr,
        stream);
  } else if (dtype == at::ScalarType::Half) {
    topk_gating_softmax_kernel_launcher<__half>(
        reinterpret_cast<const __half*>(gating_output.data_ptr<at::Half>()),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        softmax_workspace.data_ptr<float>(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        moe_softcapping_f,
        bias_ptr,
        stream);
  } else if (dtype == at::ScalarType::BFloat16) {
    topk_gating_softmax_kernel_launcher<BFloat16Type>(
        reinterpret_cast<const BFloat16Type*>(
            gating_output.data_ptr<at::BFloat16>()),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        softmax_workspace.data_ptr<float>(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        moe_softcapping_f,
        bias_ptr,
        stream);
  } else {
    LOG(FATAL) << "Unsupported gating_output dtype: " << dtype;
  }
}
}  // namespace xllm::kernel::cuda
