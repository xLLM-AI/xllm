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

#include "deep_ep.h"

#include <glog/logging.h>

#include "framework/parallel_state/process_group.h"
#include "kernels/ops_api.h"

namespace {
/**
 * @brief AllGather a metadata tensor on CPU by using Device communication
 *
 * Transfers local tensor from CPU to Device, performs Device AllGather,
 * and copies gathered results back to the allocated CPU output tensor.
 *
 * @param process_group ProcessGroup pointer
 * @param local_exchange_info_cpu Local metadata tensor on CPU [M]
 * @param all_exchange_info_cpu Output metadata tensor on CPU [nranks, M],
 * pre-allocated
 * @param device Device for communication
 */
void offload_exchange_info_allgather(
    xllm::ProcessGroup* process_group,
    const torch::Tensor& local_exchange_info_cpu,
    torch::Tensor& all_exchange_info_cpu,
    const torch::Device& device) {
  // Checks for shape and device
  CHECK(local_exchange_info_cpu.device().is_cpu())
      << "Input tensor must be on CPU";
  CHECK(all_exchange_info_cpu.device().is_cpu())
      << "Output tensor must be on CPU";

  int64_t nranks = all_exchange_info_cpu.size(0);
  int64_t info_size = local_exchange_info_cpu.size(0);

  // Copy input from CPU to device
  torch::Tensor local_tensor_device = local_exchange_info_cpu.to(device);

  // Prepare allgather input/output containers
  std::vector<torch::Tensor> device_outputs;
  device_outputs.reserve(nranks);

  // Allocate device tensors for all_exchange_info_cpu of each rank
  for (int64_t i = 0; i < nranks; ++i) {
    device_outputs.push_back(torch::empty_like(local_tensor_device));
  }

  // Perform device AllGather and wait for completion
  process_group->allgather(local_tensor_device, device_outputs);

  // Copy gathered device tensors back to all_exchange_info_cpu
  for (int64_t i = 0; i < nranks; ++i) {
    torch::Tensor row_cpu = all_exchange_info_cpu.select(0, i);
    row_cpu.copy_(device_outputs[i]);
  }
}
}  // namespace

namespace xllm {
namespace layer {

// initialize the static members of DeepEPManager
std::mutex DeepEPManager::mutex_;
DeepEP DeepEPManager::instance_ = DeepEP(nullptr);

DeepEPImpl::DeepEPImpl(int64_t dispatch_token_size,
                       int64_t combine_token_size,
                       int64_t max_num_tokens_per_rank,
                       int64_t num_global_experts,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options)
    : parallel_args_(parallel_args), options_(options) {
  // make sure the size of moe_ep_group is equal nranks
  CHECK(parallel_args.ep_size() == parallel_args.world_size_)
      << "DeepEp only support the condition of ep_size == world_size";

  ProcessGroup* moe_ep_group = parallel_args.moe_ep_group_;
  int32_t rank = parallel_args.rank_;
  int32_t nranks = parallel_args.world_size_;

  // Step 1: Call moe_all2all_create to obtain the deep ep handle and buffer
  // tensors
  //  for All-to-All communication.
  // Only needs to be done once.
  xllm::kernel::MoeAll2AllCreateParams create_params;
  create_params.dispatch_token_byte = dispatch_token_size;
  create_params.combine_token_byte = combine_token_size;
  create_params.max_expert_num = num_global_experts;
  create_params.max_token_num = max_num_tokens_per_rank;
  create_params.rank = rank;
  create_params.nrank = nranks;
  create_params.device = options_.device();

  // moe_all2all_create returns a vector of tensors:
  // index 0 is a tensor contains [handle, exchange_info_size]
  // Index 1: exchange_info
  // Index 2: dispatch_send
  // Index 3: dispatch_recv
  // Index 4: combine_send
  // Index 5: combine_recv
  std::vector<torch::Tensor> created_tensors =
      xllm::kernel::moe_all2all_create(create_params);
  CHECK(created_tensors.size() == 6)
      << "moe_all2all_create returned incorrect number of tensors";
  // Extract handle and exchange_info_size from created_tensors[0]
  handle_ = created_tensors[0][0].item<int64_t>();
  torch::Tensor exchange_info = created_tensors[1];

  // register buffers to the module
  dispatch_send_token_tensor_ =
      register_buffer("dispatch_send_token", created_tensors[2]);
  dispatch_recv_token_tensor_ =
      register_buffer("dispatch_recv_token", created_tensors[3]);
  combine_send_token_tensor_ =
      register_buffer("combine_send_token", created_tensors[4]);
  combine_recv_token_tensor_ =
      register_buffer("combine_recv_token", created_tensors[5]);

  // Step 2: Gather all_exchange_info by performing an All-Gather operation
  //  on exchange_info across nrank processes.
  // Only needs to be done once.
  auto cpu_options =
      torch::TensorOptions().dtype(exchange_info.dtype()).device(torch::kCPU);
  torch::Tensor all_exchange_info =
      torch::empty({nranks, exchange_info.size(0)}, cpu_options);
  offload_exchange_info_allgather(
      moe_ep_group, exchange_info, all_exchange_info, options_.device());

  // Step 3: moe_all2all_init to configure the all_exchange_info into the
  // handle. Only needs to be done once.
  xllm::kernel::MoeAll2AllInitParams init_params;
  init_params.handle = handle_;
  init_params.all_exchange_info = all_exchange_info;
  init_params.device = options_.device();
  xllm::kernel::moe_all2all_init(init_params);

  // prepare variables that facilitate the usage of deep ep dispatch and combine
  deep_ep_params_.dispatch_token_size = dispatch_token_size;
  deep_ep_params_.combine_token_size = combine_token_size;
  deep_ep_params_.max_num_tokens_per_rank = max_num_tokens_per_rank;
  deep_ep_params_.dispatch_recv_layout =
      torch::empty({nranks, 2}, options_.dtype(torch::kInt32));
  deep_ep_params_.dispatch_recv_token_num =
      torch::empty({num_global_experts}, options_.dtype(torch::kInt32));
  deep_ep_params_.max_num_tokens_recv = max_num_tokens_per_rank * nranks;

  is_initialized_ = true;
}

DeepEPImpl::~DeepEPImpl() {
  if (is_initialized_) {
    xllm::kernel::MoeAll2AllDestroyParams destroy_params;
    destroy_params.handle = handle_;
    destroy_params.device = options_.device();
    xllm::kernel::moe_all2all_destroy(destroy_params);
  }
}

void DeepEPImpl::dispatch(int64_t token_byte,
                          int64_t token_num,
                          const torch::Tensor& send_layout,
                          const torch::Tensor& send_token_num,
                          const torch::Tensor& recv_layout,
                          const torch::Tensor& recv_token_num,
                          const std::optional<torch::Tensor>& send_token,
                          const std::optional<torch::Tensor>& recv_token) {
  xllm::kernel::MoeAll2AllDispatchParams params;
  params.handle = handle_;
  params.token_byte = token_byte;
  params.token_num = token_num;
  params.send_layout = send_layout;
  params.send_token_num = send_token_num;
  params.recv_layout = recv_layout;
  params.recv_token_num = recv_token_num;
  params.send_token = send_token;
  params.recv_token = recv_token;

  xllm::kernel::moe_all2all_dispatch(params);
}

void DeepEPImpl::combine(int64_t token_byte,
                         int64_t token_num,
                         const torch::Tensor& send_src_layout,
                         const torch::Tensor& send_dst_layout,
                         const std::optional<torch::Tensor>& send_token,
                         const std::optional<torch::Tensor>& recv_token) {
  xllm::kernel::MoeAll2AllCombineParams params;
  params.handle = handle_;
  params.token_byte = token_byte;
  params.token_num = token_num;
  params.send_src_layout = send_src_layout;
  params.send_dst_layout = send_dst_layout;
  params.send_token = send_token;
  params.recv_token = recv_token;

  xllm::kernel::moe_all2all_combine(params);
}

void DeepEPImpl::dispatch_step(int64_t num_token_expand,
                               const torch::Tensor& token_count_slice) {
  // 1. Generate send layout
  xllm::kernel::MoeAll2AllGenSendLayoutParams gen_send_layout_params;
  gen_send_layout_params.token_count = token_count_slice;
  gen_send_layout_params.nrank = parallel_args_.ep_size();
  torch::Tensor dispatch_send_layout =
      xllm::kernel::moe_all2all_gen_send_layout(gen_send_layout_params);

  // 2. Perform Dispatch
  // Use member variables for params usually passed from outside
  this->dispatch(deep_ep_params_.dispatch_token_size,
                 num_token_expand,
                 dispatch_send_layout,
                 token_count_slice,
                 deep_ep_params_.dispatch_recv_layout,
                 deep_ep_params_.dispatch_recv_token_num);
}

DeepEPMetaResult DeepEPImpl::process_dispatch_result(
    int64_t num_experts_per_rank,
    torch::Tensor& output_head,
    std::optional<torch::Tensor> output_tail) {
  // 1. Obtain Gather Indices (Meta Calculation)
  xllm::kernel::MoeAll2AllGenGatherIndexParams gen_gather_index_params;
  // Note: we need to reshape token_num view based on current EP config
  gen_gather_index_params.token_num =
      deep_ep_params_.dispatch_recv_token_num.view(
          {parallel_args_.ep_size(), num_experts_per_rank});
  gen_gather_index_params.pad_num = deep_ep_params_.max_num_tokens_per_rank;

  std::vector<torch::Tensor> gen_gather_index_output =
      xllm::kernel::moe_all2all_gen_gather_index(gen_gather_index_params);

  torch::Tensor gather_by_expert_index = gen_gather_index_output[0];
  torch::Tensor gather_by_rank_index = gen_gather_index_output[1];
  torch::Tensor token_count_slice_out = gen_gather_index_output[2];
  torch::Tensor token_sum = gen_gather_index_output[3];

  // 2. Gather Split (Unpack Buffer to User Tensor)
  xllm::kernel::GatherSplitParams gather_split_params;
  int64_t max_tokens_bytes_recv =
      deep_ep_params_.max_num_tokens_recv * deep_ep_params_.dispatch_token_size;

  // Create a view of the internal recv buffer
  torch::Tensor gather_input =
      dispatch_recv_token_tensor_.narrow(0, 0, max_tokens_bytes_recv)
          .view({deep_ep_params_.max_num_tokens_recv,
                 deep_ep_params_.dispatch_token_size});

  gather_split_params.input = gather_input;
  gather_split_params.gather_index = gather_by_expert_index;
  gather_split_params.valid_token_num = token_sum;
  gather_split_params.output_head = output_head;
  gather_split_params.output_tail = output_tail.value_or(torch::Tensor());

  xllm::kernel::gather_split(gather_split_params);

  return DeepEPMetaResult{
      gather_by_rank_index, token_count_slice_out, token_sum};
}

torch::Tensor DeepEPImpl::combine_step_pack(
    const torch::Tensor& input,
    const torch::Tensor& gather_rank_index,
    const torch::Tensor& valid_token_num,
    int64_t hidden_size,
    torch::ScalarType dtype) {
  // 1. Gather Split (Pack User Tensor to Send Buffer)
  xllm::kernel::GatherSplitParams gather_split_params;
  gather_split_params.input = input;
  gather_split_params.gather_index = gather_rank_index;
  gather_split_params.valid_token_num = valid_token_num;

  // View send buffer with correct dtype structure
  gather_split_params.output_head =
      view_as_dtype(combine_send_token_tensor_, dtype)
          .view({deep_ep_params_.max_num_tokens_recv, -1});
  gather_split_params.output_tail = torch::Tensor();

  xllm::kernel::gather_split(gather_split_params);

  // 2. Generate Combine Layout
  xllm::kernel::MoeAll2AllGenSendLayoutParams gen_send_layout_params;
  gen_send_layout_params.token_count = deep_ep_params_.dispatch_recv_token_num;
  gen_send_layout_params.nrank = parallel_args_.ep_size();
  torch::Tensor combine_send_layout =
      xllm::kernel::moe_all2all_gen_send_layout(gen_send_layout_params);

  return combine_send_layout;
}

torch::Tensor DeepEPImpl::combine_step_comm(
    const torch::Tensor& combine_send_layout,
    int64_t num_token_expand,
    int64_t hidden_size,
    torch::ScalarType dtype) {
  // Use the cached dispatch recv layout as combine recv layout
  torch::Tensor combine_recv_layout = deep_ep_params_.dispatch_recv_layout;
  // 3. Perform Combine
  this->combine(deep_ep_params_.combine_token_size,
                num_token_expand,
                combine_send_layout,
                combine_recv_layout);

  // 4. Return formatted output
  int64_t recv_len = num_token_expand * hidden_size;
  return view_as_dtype(combine_recv_token_tensor_, dtype)
      .narrow(0, 0, recv_len)
      .view({num_token_expand, hidden_size});
}

DeepEPBuffer DeepEPImpl::get_buffer() const {
  return DeepEPBuffer{dispatch_send_token_tensor_,
                      dispatch_recv_token_tensor_,
                      combine_send_token_tensor_,
                      combine_recv_token_tensor_};
}

DeepEPParams DeepEPImpl::get_params() const { return deep_ep_params_; }

}  // namespace layer
}  // namespace xllm
