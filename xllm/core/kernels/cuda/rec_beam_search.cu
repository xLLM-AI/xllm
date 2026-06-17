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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <utility>

#include "common/global_flags.h"
#include "core/framework/config/beam_search_config.h"
#include "cuda.h"
#include "topk_last_dim.cuh"

// ensure half type is available (consistent with topk_last_dim.cuh)
using half = __half;

namespace xllm::kernel::cuda {

template <typename T>
inline size_t invoke_topk_workspace_size(SizeType32 batch_size,
                                         SizeType32 input_length,
                                         SizeType32 k,
                                         bool is_largest,
                                         bool sorted) {
  return reduce_topk::invokeComputeTopkLastDimWorkspaceSize<T>(
      batch_size, input_length, k, is_largest, sorted);
}

template <typename KernelT>
inline void invoke_topk_last_dim(SizeType32 batch_size,
                                 SizeType32 input_length,
                                 SizeType32 k,
                                 bool is_largest,
                                 const void* input,
                                 void* output_values,
                                 void* output_indices,
                                 void* workspace,
                                 cudaStream_t stream,
                                 bool sorted) {
  reduce_topk::invokeTopkLastDim<KernelT>(batch_size,
                                          input_length,
                                          k,
                                          is_largest,
                                          input,
                                          output_values,
                                          output_indices,
                                          workspace,
                                          stream,
                                          sorted);
}

template <typename T>
std::pair<torch::Tensor, torch::Tensor> compute_topk_general_impl(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device,
    bool sorted) {
  input = input.contiguous();

  auto output_dtype = input.dtype();
  auto top_k_values =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto top_k_indices =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto workspace_size =
      invoke_topk_workspace_size<T>(static_cast<SizeType32>(batch_size),
                                    static_cast<SizeType32>(input_length),
                                    static_cast<SizeType32>(k),
                                    true,
                                    sorted);

  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  invoke_topk_last_dim<T>(static_cast<SizeType32>(batch_size),
                          static_cast<SizeType32>(input_length),
                          static_cast<SizeType32>(k),
                          true,
                          input.data_ptr<T>(),
                          top_k_values.data_ptr<T>(),
                          top_k_indices.data_ptr<int32_t>(),
                          workspace.data_ptr<uint8_t>(),
                          stream,
                          sorted);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(top_k_values, top_k_indices);
}

template <>
std::pair<torch::Tensor, torch::Tensor> compute_topk_general_impl<half>(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device,
    bool sorted) {
  input = input.contiguous();

  auto output_dtype = input.dtype();
  auto top_k_values =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto top_k_indices =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto workspace_size =
      invoke_topk_workspace_size<half>(static_cast<SizeType32>(batch_size),
                                       static_cast<SizeType32>(input_length),
                                       static_cast<SizeType32>(k),
                                       true,
                                       sorted);

  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  invoke_topk_last_dim<half>(static_cast<SizeType32>(batch_size),
                             static_cast<SizeType32>(input_length),
                             static_cast<SizeType32>(k),
                             true,
                             input.data_ptr<at::Half>(),
                             top_k_values.data_ptr<at::Half>(),
                             top_k_indices.data_ptr<int32_t>(),
                             workspace.data_ptr<uint8_t>(),
                             stream,
                             sorted);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(top_k_values, top_k_indices);
}

// specialization for __nv_bfloat16
#ifdef ENABLE_BF16
template <>
std::pair<torch::Tensor, torch::Tensor>
compute_topk_general_impl<__nv_bfloat16>(torch::Tensor input,
                                         uint32_t batch_size,
                                         uint32_t input_length,
                                         uint32_t k,
                                         torch::Device device,
                                         bool sorted) {
  input = input.contiguous();

  auto output_dtype = input.dtype();
  auto top_k_values =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto top_k_indices =
      torch::empty({batch_size, k},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto workspace_size = invoke_topk_workspace_size<__nv_bfloat16>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(input_length),
      static_cast<SizeType32>(k),
      true,
      sorted);

  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  invoke_topk_last_dim<__nv_bfloat16>(static_cast<SizeType32>(batch_size),
                                      static_cast<SizeType32>(input_length),
                                      static_cast<SizeType32>(k),
                                      true,
                                      input.data_ptr<__nv_bfloat16>(),
                                      top_k_values.data_ptr<__nv_bfloat16>(),
                                      top_k_indices.data_ptr<int32_t>(),
                                      workspace.data_ptr<uint8_t>(),
                                      stream,
                                      sorted);

  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(top_k_values, top_k_indices);
}
#endif

std::pair<torch::Tensor, torch::Tensor> compute_topk_general(
    torch::Tensor input,
    uint32_t batch_size,
    uint32_t input_length,
    uint32_t k,
    torch::Device device,
    bool sorted) {
  auto dtype = input.dtype();

  if (dtype == torch::kFloat32) {
    return compute_topk_general_impl<float>(
        input, batch_size, input_length, k, device, sorted);
  } else if (dtype == torch::kFloat16) {
    return compute_topk_general_impl<half>(
        input, batch_size, input_length, k, device, sorted);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    // ensure type match
    input = input.to(torch::kBFloat16).contiguous();
    return compute_topk_general_impl<__nv_bfloat16>(
        input, batch_size, input_length, k, device, sorted);
#else
    // if BF16 is not supported, convert to float32
    input = input.to(torch::kFloat32).contiguous();
    return compute_topk_general_impl<float>(
        input, batch_size, input_length, k, device, sorted);
#endif
  } else {
    // default convert to float32
    input = input.to(torch::kFloat32).contiguous();
    return compute_topk_general_impl<float>(
        input, batch_size, input_length, k, device, sorted);
  }
}

// template function: wrap topK calculation logic, support different precision
// types
template <typename T>
std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search_impl(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t num_return_sequences,
    torch::Device device,
    bool is_largest = true,
    bool sorted = false) {
  combined_probs = combined_probs.contiguous();

  // create output tensor, output type is the same as input type
  auto output_dtype = combined_probs.dtype();
  auto top_k_probs =
      torch::empty({batch_size, num_return_sequences},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto top_k_indices =
      torch::empty({batch_size, num_return_sequences},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // calculate workspace size
  auto workspace_size = reduce_topk::invokeComputeTopkLastDimWorkspaceSize<T>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(num_return_sequences),
      is_largest,
      sorted);  // is_largest = true

  // allocate workspace memory
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  // call TensorRT-LLM's topK function
  // note: the data type of combined_probs must match the template parameter T
  reduce_topk::invokeTopkLastDim<T>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(num_return_sequences),
      is_largest,  // is_largest = true
      combined_probs.data_ptr<T>(),
      top_k_probs.data_ptr<T>(),
      top_k_indices.data_ptr<int32_t>(),
      workspace.data_ptr<uint8_t>(),
      stream,
      sorted);

  // synchronize CUDA stream
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(top_k_probs, top_k_indices);
}

// specialization for half (float16)
template <>
std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search_impl<half>(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t num_return_sequences,
    torch::Device device,
    bool is_largest,
    bool sorted) {
  combined_probs = combined_probs.contiguous();

  // create output tensor, output type is the same as input type
  auto output_dtype = combined_probs.dtype();
  auto top_k_probs =
      torch::empty({batch_size, num_return_sequences},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto top_k_indices =
      torch::empty({batch_size, num_return_sequences},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // calculate workspace size
  auto workspace_size =
      reduce_topk::invokeComputeTopkLastDimWorkspaceSize<half>(
          static_cast<SizeType32>(batch_size),
          static_cast<SizeType32>(beam_size * top_k),
          static_cast<SizeType32>(num_return_sequences),
          is_largest,
          sorted);  // is_largest = true

  // allocate workspace memory
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  // call TensorRT-LLM's topK function
  // use at::Half for data_ptr, then cast to half* for CUDA kernel
  invoke_topk_last_dim<half>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(num_return_sequences),
      is_largest,  // is_largest = true
      reinterpret_cast<half const*>(combined_probs.data_ptr<at::Half>()),
      reinterpret_cast<half*>(top_k_probs.data_ptr<at::Half>()),
      top_k_indices.data_ptr<int32_t>(),
      workspace.data_ptr<uint8_t>(),
      stream,
      sorted);

  // synchronize CUDA stream
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(top_k_probs, top_k_indices);
}

// specialization for __nv_bfloat16
#ifdef ENABLE_BF16
template <>
std::pair<torch::Tensor, torch::Tensor>
compute_topk_for_beam_search_impl<__nv_bfloat16>(torch::Tensor combined_probs,
                                                 uint32_t batch_size,
                                                 uint32_t beam_size,
                                                 uint32_t top_k,
                                                 uint32_t num_return_sequences,
                                                 torch::Device device,
                                                 bool is_largest,
                                                 bool sorted) {
  combined_probs = combined_probs.contiguous();

  // create output tensor, output type is the same as input type
  auto output_dtype = combined_probs.dtype();
  auto top_k_probs =
      torch::empty({batch_size, num_return_sequences},
                   torch::TensorOptions().dtype(output_dtype).device(device));
  auto top_k_indices =
      torch::empty({batch_size, num_return_sequences},
                   torch::TensorOptions().dtype(torch::kInt32).device(device));

  // get CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // calculate workspace size
  auto workspace_size =
      reduce_topk::invokeComputeTopkLastDimWorkspaceSize<__nv_bfloat16>(
          static_cast<SizeType32>(batch_size),
          static_cast<SizeType32>(beam_size * top_k),
          static_cast<SizeType32>(num_return_sequences),
          is_largest,
          sorted);  // is_largest = true

  // allocate workspace memory
  auto workspace =
      torch::empty({static_cast<int64_t>(workspace_size)},
                   torch::TensorOptions().dtype(torch::kUInt8).device(device));

  // call TensorRT-LLM's topK function
  // use at::BFloat16 for data_ptr, then cast to __nv_bfloat16* for CUDA kernel
  invoke_topk_last_dim<__nv_bfloat16>(
      static_cast<SizeType32>(batch_size),
      static_cast<SizeType32>(beam_size * top_k),
      static_cast<SizeType32>(num_return_sequences),
      is_largest,  // is_largest = true
      reinterpret_cast<__nv_bfloat16 const*>(
          combined_probs.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(top_k_probs.data_ptr<at::BFloat16>()),
      top_k_indices.data_ptr<int32_t>(),
      workspace.data_ptr<uint8_t>(),
      stream,
      sorted);

  // synchronize CUDA stream
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_pair(top_k_probs, top_k_indices);
}
#endif

// dispatch to the correct template instantiation based on the dtype of the
// input tensor
std::pair<torch::Tensor, torch::Tensor> compute_topk_for_beam_search(
    torch::Tensor combined_probs,
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t num_return_sequences,
    torch::Device device,
    bool is_largest = true,
    bool sorted = false) {
  auto dtype = combined_probs.dtype();

  if (dtype == torch::kFloat32) {
    // ensure type match
    combined_probs = combined_probs.to(torch::kFloat32).contiguous();
    return compute_topk_for_beam_search_impl<float>(combined_probs,
                                                    batch_size,
                                                    beam_size,
                                                    top_k,
                                                    num_return_sequences,
                                                    device,
                                                    is_largest,
                                                    sorted);
  } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
    // ensure type match, use half (consistent with topk_last_dim.cuh)
    combined_probs = combined_probs.to(torch::kFloat16).contiguous();
    return compute_topk_for_beam_search_impl<half>(combined_probs,
                                                   batch_size,
                                                   beam_size,
                                                   top_k,
                                                   num_return_sequences,
                                                   device,
                                                   is_largest,
                                                   sorted);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    // ensure type match
    combined_probs = combined_probs.to(torch::kBFloat16).contiguous();
    return compute_topk_for_beam_search_impl<__nv_bfloat16>(
        combined_probs,
        batch_size,
        beam_size,
        top_k,
        num_return_sequences,
        device,
        is_largest,
        sorted);
#else
    // if BF16 is not supported, convert to float32
    combined_probs = combined_probs.to(torch::kFloat32).contiguous();
    return compute_topk_for_beam_search_impl<float>(combined_probs,
                                                    batch_size,
                                                    beam_size,
                                                    top_k,
                                                    num_return_sequences,
                                                    device,
                                                    is_largest,
                                                    sorted);
#endif
  } else {
    // default convert to float32
    combined_probs = combined_probs.to(torch::kFloat32).contiguous();
    return compute_topk_for_beam_search_impl<float>(combined_probs,
                                                    batch_size,
                                                    beam_size,
                                                    top_k,
                                                    num_return_sequences,
                                                    device,
                                                    is_largest,
                                                    sorted);
  }
}

template <typename T>
__global__ void beam_search_init_kernel(
    const int32_t* __restrict__ top_tokens,    // [batch_size, top_k]
    const T* __restrict__ top_logprobs,        // [batch_size, top_k]
    int32_t* __restrict__ out_token_ids,       // [batch_size, beam_size]
    T* __restrict__ out_acc_logprob,           // [batch_size, beam_size]
    int32_t* __restrict__ out_token_index,     // [batch_size * beam_size, 1]
    int32_t* __restrict__ out_sequence_group,  // [batch_size, beam_size,
                                               // total_rounds]
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t total_rounds) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_elements = batch_size * beam_size;

  if (tid >= total_elements) return;

  const uint32_t batch_idx = tid / beam_size;
  const uint32_t beam_idx = tid % beam_size;

  // source index: read the first beam_size tokens from [batch_size, top_k]
  const uint32_t src_idx = batch_idx * top_k + beam_idx;

  // destination index
  const uint32_t dst_idx = batch_idx * beam_size + beam_idx;

  // copy all the tokens in one go
  uint32_t token = top_tokens[src_idx];
  out_token_ids[dst_idx] = token;
  out_acc_logprob[dst_idx] = top_logprobs[src_idx];
  out_token_index[tid] = static_cast<int32_t>(beam_idx);

  // out_sequence_group[:, :, 0] = tokens
  const size_t seq_idx =
      batch_idx * beam_size * total_rounds + beam_idx * total_rounds + 0;
  out_sequence_group[seq_idx] = token;
}

// initialize the beam search tensors for first step
void beam_search_init(torch::Tensor top_tokens,
                      torch::Tensor top_logprobs,
                      torch::Tensor out_token_ids,
                      torch::Tensor out_acc_logprob,
                      torch::Tensor out_token_index,
                      torch::Tensor out_sequence_group,
                      uint32_t batch_size,
                      uint32_t beam_size,
                      uint32_t top_k,
                      uint32_t total_rounds) {
  constexpr uint32_t kBlockSize = 256;
  const uint32_t total_elements = batch_size * beam_size;
  const uint32_t num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto dtype = top_logprobs.dtype();

  if (dtype == torch::kFloat32) {
    beam_search_init_kernel<float><<<num_blocks, kBlockSize, 0, stream>>>(
        top_tokens.data_ptr<int32_t>(),
        top_logprobs.data_ptr<float>(),
        out_token_ids.data_ptr<int32_t>(),
        out_acc_logprob.data_ptr<float>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds);
    //   } else if (dtype == torch::kFloat16) {
    //     beam_search_init_kernel<half><<<num_blocks, kBlockSize, 0, stream>>>(
    //         top_tokens.data_ptr<int32_t>(),
    //         top_logprobs.data_ptr<half>(),
    //         out_token_ids.data_ptr<int32_t>(),
    //         out_acc_logprob.data_ptr<half>(),
    //         out_token_index.data_ptr<int32_t>(),
    //         out_sequence_group.data_ptr<int32_t>(),
    //         batch_size,
    //         beam_size,
    //         top_k,
    //         total_rounds);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    beam_search_init_kernel<__nv_bfloat16>
        <<<num_blocks, kBlockSize, 0, stream>>>(
            top_tokens.data_ptr<int32_t>(),
            top_logprobs.data_ptr<__nv_bfloat16>(),
            out_token_ids.data_ptr<int32_t>(),
            out_acc_logprob.data_ptr<__nv_bfloat16>(),
            out_token_index.data_ptr<int32_t>(),
            out_sequence_group.data_ptr<int32_t>(),
            batch_size,
            beam_size,
            top_k,
            total_rounds);
#else
    beam_search_init_kernel<float><<<num_blocks, kBlockSize, 0, stream>>>(
        top_tokens.data_ptr<int32_t>(),
        top_logprobs.to(torch::kFloat32).data_ptr<float>(),
        out_token_ids.data_ptr<int32_t>(),
        out_acc_logprob.data_ptr<float>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds);
#endif
  } else {
    beam_search_init_kernel<float><<<num_blocks, kBlockSize, 0, stream>>>(
        top_tokens.data_ptr<int32_t>(),
        top_logprobs.to(torch::kFloat32).data_ptr<float>(),
        out_token_ids.data_ptr<int32_t>(),
        out_acc_logprob.data_ptr<float>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused kernel for beam search step > 0 post-processing
// Combines: parent_beam/token_in_beam calculation, token lookup,
//           output updates, and sequence group updates
template <typename T>
__global__ void beam_search_step_kernel(
    const int32_t* __restrict__ selected_k_indices,  // [batch_size, beam_size]
    const T* __restrict__ new_probs,                 // [batch_size, beam_size]
    const int32_t* __restrict__ top_tokens,    // [batch_size, beam_size, top_k]
    const int32_t* __restrict__ in_seq_group,  // [batch_size, beam_size,
                                               // total_rounds]
    T* __restrict__ out_acc_logprob,           // [batch_size, beam_size]
    int32_t* __restrict__ out_token_ids,       // [batch_size, beam_size]
    int32_t* __restrict__ out_token_index,     // [batch_size, beam_size]
    int32_t* __restrict__ out_seq_group,       // [batch_size, beam_size,
                                               // total_rounds]
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t total_rounds,
    uint32_t current_step) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_elements = batch_size * beam_size;

  if (tid >= total_elements) return;

  const uint32_t batch_idx = tid / beam_size;
  const uint32_t beam_idx = tid % beam_size;
  const uint32_t flat_idx = batch_idx * beam_size + beam_idx;

  // Read index and compute parent_beam, token_in_beam
  const int32_t idx = selected_k_indices[flat_idx];
  const int32_t parent_beam = idx / static_cast<int32_t>(top_k);
  const int32_t token_in_beam = idx % static_cast<int32_t>(top_k);

  // Lookup new token from top_tokens[batch_idx, parent_beam, token_in_beam]
  const uint32_t token_idx =
      batch_idx * beam_size * top_k + parent_beam * top_k + token_in_beam;
  const int32_t new_token = top_tokens[token_idx];

  // Write outputs
  out_acc_logprob[flat_idx] = new_probs[flat_idx];
  out_token_index[flat_idx] = idx;
  out_token_ids[flat_idx] = new_token;

  // Copy sequence history from parent beam: in_seq_group[batch_idx,
  // parent_beam, 0:current_step]
  const uint32_t src_seq_base =
      batch_idx * beam_size * total_rounds + parent_beam * total_rounds;
  const uint32_t dst_seq_base =
      batch_idx * beam_size * total_rounds + beam_idx * total_rounds;

  for (uint32_t s = 0; s < current_step; ++s) {
    out_seq_group[dst_seq_base + s] = in_seq_group[src_seq_base + s];
  }

  // Write new token at current_step
  out_seq_group[dst_seq_base + current_step] = new_token;
}

// Fused kernel with sorting for non-final steps
// Performs argsort + gather + post-processing in one kernel
template <typename T>
__global__ void beam_search_step_with_sort_kernel(
    const int32_t* __restrict__ selected_k_indices_in,  // [batch_size,
                                                        // beam_size]
    const T* __restrict__ new_probs_in,        // [batch_size, beam_size]
    const int32_t* __restrict__ top_tokens,    // [batch_size, beam_size,
                                               // top_k]
    const int32_t* __restrict__ in_seq_group,  // [batch_size, beam_size,
                                               // total_rounds]
    T* __restrict__ out_acc_logprob,           // [batch_size, beam_size]
    int32_t* __restrict__ out_token_ids,       // [batch_size, beam_size]
    int32_t* __restrict__ out_token_index,     // [batch_size, beam_size]
    int32_t* __restrict__ out_seq_group,       // [batch_size, beam_size,
                                               // total_rounds]
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t total_rounds,
    uint32_t current_step) {
  // Each block handles one batch
  extern __shared__ char shared_mem[];

  const uint32_t batch_idx = blockIdx.x;
  if (batch_idx >= batch_size) return;

  // Shared memory layout: indices_with_order pairs for sorting
  int32_t* shared_indices = reinterpret_cast<int32_t*>(shared_mem);
  int32_t* shared_order = shared_indices + beam_size;
  T* shared_probs = reinterpret_cast<T*>(shared_order + beam_size);

  // Load indices and probs into shared memory
  const uint32_t base_idx = batch_idx * beam_size;
  for (uint32_t i = threadIdx.x; i < beam_size; i += blockDim.x) {
    shared_indices[i] = selected_k_indices_in[base_idx + i];
    shared_probs[i] = new_probs_in[base_idx + i];
    shared_order[i] = i;
  }
  __syncthreads();

  // Simple insertion sort for small beam_size (typically 2-8)
  // Only thread 0 performs the sort
  if (threadIdx.x == 0 && beam_size <= 32) {
    for (uint32_t i = 1; i < beam_size; ++i) {
      int32_t key_idx = shared_indices[i];
      int32_t key_order = shared_order[i];
      T key_prob = shared_probs[i];
      int j = i - 1;
      while (j >= 0 && shared_indices[j] > key_idx) {
        shared_indices[j + 1] = shared_indices[j];
        shared_order[j + 1] = shared_order[j];
        shared_probs[j + 1] = shared_probs[j];
        --j;
      }
      shared_indices[j + 1] = key_idx;
      shared_order[j + 1] = key_order;
      shared_probs[j + 1] = key_prob;
    }
  }
  __syncthreads();

  // Now each thread processes one beam element
  for (uint32_t beam_idx = threadIdx.x; beam_idx < beam_size;
       beam_idx += blockDim.x) {
    const uint32_t flat_idx = base_idx + beam_idx;

    // After sorting, shared_indices[beam_idx] is the sorted index
    const int32_t idx = shared_indices[beam_idx];
    const T prob = shared_probs[beam_idx];
    const int32_t parent_beam = idx / static_cast<int32_t>(top_k);
    const int32_t token_in_beam = idx % static_cast<int32_t>(top_k);

    // Lookup new token
    const uint32_t token_idx =
        batch_idx * beam_size * top_k + parent_beam * top_k + token_in_beam;
    const int32_t new_token = top_tokens[token_idx];

    // Write outputs
    out_acc_logprob[flat_idx] = prob;
    out_token_index[flat_idx] = idx;
    out_token_ids[flat_idx] = new_token;

    // Copy sequence history
    const uint32_t src_seq_base =
        batch_idx * beam_size * total_rounds + parent_beam * total_rounds;
    const uint32_t dst_seq_base =
        batch_idx * beam_size * total_rounds + beam_idx * total_rounds;

    for (uint32_t s = 0; s < current_step; ++s) {
      out_seq_group[dst_seq_base + s] = in_seq_group[src_seq_base + s];
    }
    out_seq_group[dst_seq_base + current_step] = new_token;
  }
}

// Host function to launch beam search step kernel
template <typename T, typename TorchType>
void launch_beam_search_step_kernel(torch::Tensor selected_k_indices,
                                    torch::Tensor new_probs,
                                    torch::Tensor top_tokens,
                                    torch::Tensor in_sequence_group,
                                    torch::Tensor out_acc_logprob,
                                    torch::Tensor out_token_ids,
                                    torch::Tensor out_token_index,
                                    torch::Tensor out_sequence_group,
                                    uint32_t batch_size,
                                    uint32_t beam_size,
                                    uint32_t top_k,
                                    uint32_t total_rounds,
                                    uint32_t current_step,
                                    bool need_sort,
                                    cudaStream_t stream) {
  if (need_sort && beam_size <= 32) {
    // Use fused kernel with sorting for small beam sizes
    const uint32_t threads_per_block = 32;
    const size_t shared_mem_size =
        beam_size * sizeof(int32_t) * 2 + beam_size * sizeof(T);

    beam_search_step_with_sort_kernel<T>
        <<<batch_size, threads_per_block, shared_mem_size, stream>>>(
            selected_k_indices.data_ptr<int32_t>(),
            reinterpret_cast<T*>(new_probs.data_ptr<TorchType>()),
            top_tokens.data_ptr<int32_t>(),
            in_sequence_group.data_ptr<int32_t>(),
            reinterpret_cast<T*>(out_acc_logprob.data_ptr<TorchType>()),
            out_token_ids.data_ptr<int32_t>(),
            out_token_index.data_ptr<int32_t>(),
            out_sequence_group.data_ptr<int32_t>(),
            batch_size,
            beam_size,
            top_k,
            total_rounds,
            current_step);
  } else if (need_sort) {
    // For larger beam sizes, use PyTorch's argsort + gather, then fused kernel
    auto ordered_indices = selected_k_indices.argsort(1, false);
    new_probs = new_probs.gather(1, ordered_indices.to(torch::kInt64));
    selected_k_indices =
        selected_k_indices.gather(1, ordered_indices.to(torch::kInt64));

    constexpr uint32_t kBlockSize = 256;
    const uint32_t total_elements = batch_size * beam_size;
    const uint32_t num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;

    beam_search_step_kernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
        selected_k_indices.data_ptr<int32_t>(),
        reinterpret_cast<T*>(new_probs.data_ptr<TorchType>()),
        top_tokens.data_ptr<int32_t>(),
        in_sequence_group.data_ptr<int32_t>(),
        reinterpret_cast<T*>(out_acc_logprob.data_ptr<TorchType>()),
        out_token_ids.data_ptr<int32_t>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds,
        current_step);
  } else {
    // No sorting needed (final step)
    constexpr uint32_t kBlockSize = 256;
    const uint32_t total_elements = batch_size * beam_size;
    const uint32_t num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;

    beam_search_step_kernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
        selected_k_indices.data_ptr<int32_t>(),
        reinterpret_cast<T*>(new_probs.data_ptr<TorchType>()),
        top_tokens.data_ptr<int32_t>(),
        in_sequence_group.data_ptr<int32_t>(),
        reinterpret_cast<T*>(out_acc_logprob.data_ptr<TorchType>()),
        out_token_ids.data_ptr<int32_t>(),
        out_token_index.data_ptr<int32_t>(),
        out_sequence_group.data_ptr<int32_t>(),
        batch_size,
        beam_size,
        top_k,
        total_rounds,
        current_step);
  }
}

void beam_search_step(torch::Tensor top_tokens,
                      torch::Tensor selected_k_indices,
                      torch::Tensor selected_k_logprobs,
                      torch::Tensor in_sequence_group,
                      torch::Tensor out_acc_logprob,
                      torch::Tensor out_token_ids,
                      torch::Tensor out_token_index,
                      torch::Tensor out_sequence_group,
                      uint32_t batch_size,
                      uint32_t beam_size,
                      uint32_t top_k,
                      uint32_t total_rounds,
                      uint32_t current_step) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  bool need_sort = (current_step < total_rounds - 1);
  auto dtype = selected_k_logprobs.dtype();

  if (dtype == torch::kFloat32) {
    launch_beam_search_step_kernel<float, float>(selected_k_indices,
                                                 selected_k_logprobs,
                                                 top_tokens,
                                                 in_sequence_group,
                                                 out_acc_logprob,
                                                 out_token_ids,
                                                 out_token_index,
                                                 out_sequence_group,
                                                 batch_size,
                                                 beam_size,
                                                 top_k,
                                                 total_rounds,
                                                 current_step,
                                                 need_sort,
                                                 stream);
  } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
    launch_beam_search_step_kernel<half, at::Half>(selected_k_indices,
                                                   selected_k_logprobs,
                                                   top_tokens,
                                                   in_sequence_group,
                                                   out_acc_logprob,
                                                   out_token_ids,
                                                   out_token_index,
                                                   out_sequence_group,
                                                   batch_size,
                                                   beam_size,
                                                   top_k,
                                                   total_rounds,
                                                   current_step,
                                                   need_sort,
                                                   stream);
  } else if (dtype == torch::kBFloat16) {
#ifdef ENABLE_BF16
    launch_beam_search_step_kernel<__nv_bfloat16, at::BFloat16>(
        selected_k_indices,
        selected_k_logprobs,
        top_tokens,
        in_sequence_group,
        out_acc_logprob,
        out_token_ids,
        out_token_index,
        out_sequence_group,
        batch_size,
        beam_size,
        top_k,
        total_rounds,
        current_step,
        need_sort,
        stream);
#else
    launch_beam_search_step_kernel<float, float>(
        selected_k_indices,
        selected_k_logprobs.to(torch::kFloat32),
        top_tokens,
        in_sequence_group,
        out_acc_logprob.to(torch::kFloat32),
        out_token_ids,
        out_token_index,
        out_sequence_group,
        batch_size,
        beam_size,
        top_k,
        total_rounds,
        current_step,
        need_sort,
        stream);
#endif
  } else {
    launch_beam_search_step_kernel<float, float>(
        selected_k_indices,
        selected_k_logprobs.to(torch::kFloat32),
        top_tokens,
        in_sequence_group,
        out_acc_logprob.to(torch::kFloat32),
        out_token_ids,
        out_token_index,
        out_sequence_group,
        batch_size,
        beam_size,
        top_k,
        total_rounds,
        current_step,
        need_sort,
        stream);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused kernel for beam search step > 0 post-processing
// Combines: parent_beam/token_in_beam calculation, token lookup,
//           output updates, and sequence group updates
template <typename T>
__global__ void beam_search_last_step_kernel(
    const int32_t* __restrict__ selected_k_indices,  // [batch_size,
                                                     // num_return_sequences]
    const T* __restrict__ new_probs,                 // [batch_size,
                                                     // num_return_sequences]
    const int32_t* __restrict__ top_tokens,    // [batch_size, beam_size, top_k]
    const int32_t* __restrict__ in_seq_group,  // [batch_size, beam_size,
                                               // total_rounds]
    T* __restrict__ out_acc_logprob,           // [batch_size,
                                               // num_return_sequences]
    int32_t* __restrict__ out_token_ids,       // [batch_size,
                                               // num_return_sequences]
    int32_t* __restrict__ out_token_index,     // [batch_size,
                                               // num_return_sequences]
    int32_t* __restrict__ out_seq_group,       // [batch_size,
                                               // num_return_sequences,
                                               // total_rounds]
    uint32_t batch_size,
    uint32_t beam_size,
    uint32_t top_k,
    uint32_t num_return_sequences,
    uint32_t total_rounds,
    uint32_t current_step) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_elements = batch_size * num_return_sequences;

  if (tid >= total_elements) {
    return;
  }
  const uint32_t batch_idx = tid / num_return_sequences;
  const uint32_t beam_idx = tid % num_return_sequences;
  const uint32_t flat_idx = batch_idx * num_return_sequences + beam_idx;

  // Read index and compute parent_beam, token_in_beam
  const int32_t idx = selected_k_indices[flat_idx];
  const int32_t parent_beam = idx / static_cast<int32_t>(top_k);
  const int32_t token_in_beam = idx % static_cast<int32_t>(top_k);

  // Lookup new token from top_tokens[batch_idx, parent_beam, token_in_beam]
  const uint32_t token_idx =
      batch_idx * beam_size * top_k + parent_beam * top_k + token_in_beam;
  const int32_t new_token = top_tokens[token_idx];

  // Write outputs
  out_acc_logprob[flat_idx] = new_probs[flat_idx];
  out_token_index[flat_idx] = idx;
  out_token_ids[flat_idx] = new_token;

  // Copy sequence history from parent beam: in_seq_group[batch_idx,
  // parent_beam, 0:current_step]
  const uint32_t src_seq_base =
      batch_idx * beam_size * total_rounds + parent_beam * total_rounds;
  const uint32_t dst_seq_base =
      batch_idx * num_return_sequences * total_rounds + beam_idx * total_rounds;

  for (uint32_t s = 0; s < current_step; ++s) {
    out_seq_group[dst_seq_base + s] = in_seq_group[src_seq_base + s];
  }

  // Write new token at current_step
  out_seq_group[dst_seq_base + current_step] = new_token;
}

// Host function to launch beam search step kernel
template <typename T, typename TorchType>
void launch_beam_search_last_step_kernel(torch::Tensor selected_k_indices,
                                         torch::Tensor new_probs,
                                         torch::Tensor top_tokens,
                                         torch::Tensor in_sequence_group,
                                         torch::Tensor out_acc_logprob,
                                         torch::Tensor out_token_ids,
                                         torch::Tensor out_token_index,
                                         torch::Tensor out_sequence_group,
                                         uint32_t batch_size,
                                         uint32_t beam_size,
                                         uint32_t top_k,
                                         uint32_t num_return_sequences,
                                         uint32_t total_rounds,
                                         uint32_t current_step,
                                         cudaStream_t stream) {
  constexpr uint32_t kBlockSize = 256;
  const uint32_t total_elements = batch_size * num_return_sequences;
  const uint32_t num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;
  beam_search_last_step_kernel<T><<<num_blocks, kBlockSize, 0, stream>>>(
      selected_k_indices.data_ptr<int32_t>(),
      reinterpret_cast<T*>(new_probs.data_ptr<TorchType>()),
      top_tokens.data_ptr<int32_t>(),
      in_sequence_group.data_ptr<int32_t>(),
      reinterpret_cast<T*>(out_acc_logprob.data_ptr<TorchType>()),
      out_token_ids.data_ptr<int32_t>(),
      out_token_index.data_ptr<int32_t>(),
      out_sequence_group.data_ptr<int32_t>(),
      batch_size,
      beam_size,
      top_k,
      num_return_sequences,
      total_rounds,
      current_step);
}

void beam_search_last_step(torch::Tensor top_tokens,
                           torch::Tensor selected_k_indices,
                           torch::Tensor selected_k_logprobs,
                           torch::Tensor in_sequence_group,
                           torch::Tensor out_acc_logprob,
                           torch::Tensor out_token_ids,
                           torch::Tensor out_token_index,
                           torch::Tensor out_sequence_group,
                           uint32_t batch_size,
                           uint32_t beam_size,
                           uint32_t top_k,
                           uint32_t num_return_sequences,
                           uint32_t total_rounds,
                           uint32_t current_step) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto dtype = selected_k_logprobs.dtype();
  if (dtype == torch::kFloat32) {
    launch_beam_search_last_step_kernel<float, float>(selected_k_indices,
                                                      selected_k_logprobs,
                                                      top_tokens,
                                                      in_sequence_group,
                                                      out_acc_logprob,
                                                      out_token_ids,
                                                      out_token_index,
                                                      out_sequence_group,
                                                      batch_size,
                                                      beam_size,
                                                      top_k,
                                                      num_return_sequences,
                                                      total_rounds,
                                                      current_step,
                                                      stream);
  } else if (dtype == torch::kFloat16 || dtype == torch::kHalf) {
    launch_beam_search_last_step_kernel<half, at::Half>(selected_k_indices,
                                                        selected_k_logprobs,
                                                        top_tokens,
                                                        in_sequence_group,
                                                        out_acc_logprob,
                                                        out_token_ids,
                                                        out_token_index,
                                                        out_sequence_group,
                                                        batch_size,
                                                        beam_size,
                                                        top_k,
                                                        num_return_sequences,
                                                        total_rounds,
                                                        current_step,
                                                        stream);
#ifdef ENABLE_BF16
  } else if (dtype == torch::kBFloat16) {
    launch_beam_search_last_step_kernel<__nv_bfloat16, at::BFloat16>(
        selected_k_indices,
        selected_k_logprobs,
        top_tokens,
        in_sequence_group,
        out_acc_logprob,
        out_token_ids,
        out_token_index,
        out_sequence_group,
        batch_size,
        beam_size,
        top_k,
        num_return_sequences,
        total_rounds,
        current_step,
        stream);
#endif
  } else {
    launch_beam_search_last_step_kernel<float, float>(
        selected_k_indices,
        selected_k_logprobs.to(torch::kFloat32),
        top_tokens,
        in_sequence_group,
        out_acc_logprob.to(torch::kFloat32),
        out_token_ids,
        out_token_index,
        out_sequence_group,
        batch_size,
        beam_size,
        top_k,
        num_return_sequences,
        total_rounds,
        current_step,
        stream);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void beam_search(torch::Tensor acc_logprob,
                 torch::Tensor in_sequence_group,
                 torch::Tensor top_tokens,
                 torch::Tensor top_logprobs,
                 torch::Tensor out_acc_logprob,
                 torch::Tensor out_token_ids,
                 torch::Tensor out_token_index,
                 torch::Tensor out_beam_count_prefix_sums,
                 torch::Tensor out_sequence_group,
                 uint32_t batch_size,
                 uint32_t num_return_sequences,
                 uint32_t current_step) {
  (void)out_beam_count_prefix_sums;
  torch::Device device = acc_logprob.device();

  uint32_t beam_size = in_sequence_group.size(1);

  uint32_t top_k = top_tokens.size(1);
  uint32_t total_rounds = in_sequence_group.size(2);
  num_return_sequences =
      num_return_sequences > beam_size ? num_return_sequences : beam_size;
  //   CHECK_EQ(beam_size, top_k) << "beam_size must be equal with top_k.";

  if (current_step == 0) {
    // NvtxRange range("==== beam_search step 0 ====");
    beam_search_init(top_tokens,
                     top_logprobs,
                     out_token_ids,
                     out_acc_logprob,
                     out_token_index,
                     out_sequence_group,
                     batch_size,
                     beam_size,
                     top_k,
                     total_rounds);

  } else if (num_return_sequences > beam_size &&
             current_step == (total_rounds - 1)) {
    // Step 1: Compute combined_probs and topk
    auto combined_probs = (acc_logprob + top_logprobs)
                              .view({batch_size, beam_size * top_k})
                              .contiguous();
    auto [selected_logprobs, selected_indices] =
        compute_topk_for_beam_search(combined_probs,
                                     batch_size,
                                     beam_size,
                                     top_k,
                                     num_return_sequences,
                                     device,
                                     true,
                                     true);
    // Step 2: update the widened final result
    beam_search_last_step(top_tokens,
                          selected_indices,
                          selected_logprobs,
                          in_sequence_group,
                          out_acc_logprob,
                          out_token_ids,
                          out_token_index,
                          out_sequence_group,
                          batch_size,
                          beam_size,
                          top_k,
                          num_return_sequences,
                          total_rounds,
                          current_step);
  } else {
    // NvtxRange range("==== beam_search step " + std::to_string(current_step) +
    // " ====");

    // Step 1: Compute combined_probs and topk
    auto combined_probs =
        (acc_logprob + top_logprobs).view({batch_size, beam_size * top_k});

    // Use optimized topk function
    auto [selected_k_logprobs, selected_k_indices] =
        compute_topk_for_beam_search(combined_probs,
                                     batch_size,
                                     beam_size,
                                     top_k,
                                     beam_size,
                                     device,
                                     true,
                                     FLAGS_enable_topk_sorted);

    // Step 2: Launch fused post-processing kernel
    beam_search_step(top_tokens,
                     selected_k_indices,
                     selected_k_logprobs,
                     in_sequence_group,
                     out_acc_logprob,
                     out_token_ids,
                     out_token_index,
                     out_sequence_group,
                     batch_size,
                     beam_size,
                     top_k,
                     total_rounds,
                     current_step);
  }
}

}  // namespace xllm::kernel::cuda
