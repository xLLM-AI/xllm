/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <torch/torch.h>

#include "framework/parallel_state/parallel_args.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace layer {

// Buffer struct used to store communication tensors for DeepEP
struct DeepEPBuffer {
  torch::Tensor dispatch_send_token_tensor;
  torch::Tensor dispatch_recv_token_tensor;
  torch::Tensor combine_send_token_tensor;
  torch::Tensor combine_recv_token_tensor;
};

// parameters that facilitate the usage of deep ep dispatch and combine
// the following variables are only assigned once during initialization
struct DeepEPParams {
  int64_t dispatch_token_size;
  int64_t combine_token_size;
  int64_t max_num_tokens_per_rank;
  int64_t max_num_tokens_recv;
  torch::Tensor dispatch_recv_layout;
  torch::Tensor dispatch_recv_token_num;
};

// the metadata after dispatch processing
struct DeepEPMetaResult {
  torch::Tensor gather_rank_index;  // Used for combine step
  torch::Tensor token_count_slice;  // Used for GEMM
  torch::Tensor token_sum;          // Valid token count
};

// DeepEPImpl manages distributed dispatch and combine operations for tokens
// between expert ranks
class DeepEPImpl : public torch::nn::Module {
 public:
  DeepEPImpl() = default;
  /**
   * @param dispatch_token_size Number of bytes in one token for the dispatch
   * stage
   * @param combine_token_size Number of bytes in one token for the combine
   * stage
   * @param max_num_tokens_per_rank Maximum number of tokens per rank
   * @param num_global_experts Number of global experts
   * @param rank Current process rank
   * @param nranks Total number of processes
   * @param parallel_args Parallel arguments
   * @param options Tensor options
   */
  DeepEPImpl(int64_t dispatch_token_size,
             int64_t combine_token_size,
             int64_t max_num_tokens_per_rank,
             int64_t num_global_experts,
             const ParallelArgs& parallel_args,
             const torch::TensorOptions& options);

  ~DeepEPImpl();

  // The following three steps form the core of expert token movement in MoE
  // All2All communication.
  // 1. Dispatch Step: Automatically generates routing layout and
  // carries out the All2All token dispatch
  void dispatch_step(int64_t num_token_expand,
                     const torch::Tensor& token_count_slice);

  // 2. Process Dispatch Result: Processes the outputs from All2All
  // communication by generating gather indices,splitting the received buffer
  // into expert-local outputs.
  DeepEPMetaResult process_dispatch_result(
      int64_t num_experts_per_rank,
      torch::Tensor& output_head,
      std::optional<torch::Tensor> output_tail = std::nullopt);

  // 3. Combine Step: Performs the All2All combine operation using the
  // generated gather indices and valid token counts.
  // we take two steps to complete so that the combine communication
  //  can be parallelized with the computation of shared experts if needed
  torch::Tensor combine_step_pack(const torch::Tensor& input,
                                  const torch::Tensor& gather_rank_index,
                                  const torch::Tensor& valid_token_num,
                                  int64_t hidden_size,
                                  torch::ScalarType dtype);
  torch::Tensor combine_step_comm(const torch::Tensor& combine_send_layout,
                                  int64_t num_token_expand,
                                  int64_t hidden_size,
                                  torch::ScalarType dtype);

  // Dispatch tokens to other experts using all-to-all communication
  void dispatch(int64_t token_byte,
                int64_t token_num,
                const torch::Tensor& send_layout,
                const torch::Tensor& send_token_num,
                const torch::Tensor& recv_layout,
                const torch::Tensor& recv_token_num,
                const std::optional<torch::Tensor>& send_token = std::nullopt,
                const std::optional<torch::Tensor>& recv_token = std::nullopt);

  // Combine tokens from other experts using all-to-all communication
  void combine(int64_t token_byte,
               int64_t token_num,
               const torch::Tensor& send_src_layout,
               const torch::Tensor& send_dst_layout,
               const std::optional<torch::Tensor>& send_token = std::nullopt,
               const std::optional<torch::Tensor>& recv_token = std::nullopt);

  // Utility function to get the communication buffer
  DeepEPBuffer get_buffer() const;

  DeepEPParams get_params() const;

 private:
  int64_t handle_ = 0;  // Communication handle
  bool is_initialized_ = false;
  const ParallelArgs& parallel_args_;
  const torch::TensorOptions& options_;

  // Buffers created for communication; registered to support module
  // state/save/load and .to(device)
  torch::Tensor dispatch_send_token_tensor_;
  torch::Tensor dispatch_recv_token_tensor_;
  torch::Tensor combine_send_token_tensor_;
  torch::Tensor combine_recv_token_tensor_;

  DeepEPParams deep_ep_params_;
};
TORCH_MODULE(DeepEP);

// DeepEPManager is a singleton class that manages the creation and destruction
// of DeepEP instances
class DeepEPManager {
 public:
  static DeepEP get_instance(int64_t dispatch_token_size,
                             int64_t combine_token_size,
                             int64_t max_num_tokens_per_rank,
                             int64_t num_global_experts,
                             const ParallelArgs& parallel_args,
                             const torch::TensorOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_.is_empty()) {
      instance_ = DeepEP(dispatch_token_size,
                         combine_token_size,
                         max_num_tokens_per_rank,
                         num_global_experts,
                         parallel_args,
                         options);
    }
    return instance_;
  };

  static void destroy() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!instance_.is_empty()) {
      instance_ = DeepEP(nullptr);
    }
  }

 private:
  static std::mutex mutex_;
  static DeepEP instance_;
};

}  // namespace layer
}  // namespace xllm
