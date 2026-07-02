/* Copyright 2025-2026 The xLLM Authors.

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

#include "process_group.h"

#if defined(USE_NPU)
#include "npu_process_group.h"
#elif defined(USE_MLU)
#include "mlu_process_group.h"
#elif defined(USE_CUDA) || defined(USE_DCU)
#include "cuda_process_group.h"
#elif defined(USE_MUSA)
#include "musa_process_group.h"
#elif defined(USE_ILU)
#include "ilu_process_group.h"
#endif

namespace {
std::pair<int, std::vector<uint64_t>> get_trans_group_rank(int world_size,
                                                           int global_rank,
                                                           int split_size) {
  int trans_group_count = split_size;
  int trans_group_size = world_size / split_size;
  int trans_group_index = global_rank % trans_group_size;
  int trans_index = global_rank / trans_group_size;
  std::vector<uint64_t> trans_group_ranks;
  for (int i = 0; i < trans_group_count; i++) {
    uint64_t rank = i * trans_group_size + trans_group_index;
    trans_group_ranks.push_back(rank);
  }

  return {trans_index, trans_group_ranks};
}

std::vector<int64_t> get_gather_shape(int32_t world_size,
                                      const torch::Tensor& input) {
  std::vector<int64_t> out_shape;
  out_shape.reserve(input.dim() + 1);
  out_shape.push_back(world_size);
  for (int64_t dim_size : input.sizes()) {
    out_shape.push_back(dim_size);
  }
  return out_shape;
}
}  // namespace

namespace xllm {

ProcessGroup::~ProcessGroup() { shutdown_backend(); }

void ProcessGroup::shutdown_backend() {
  if (pg_ == nullptr) {
    return;
  }
  pg_->shutdown();
  pg_.reset();
}

std::pair<int, std::vector<uint64_t>> get_group_rank(int world_size,
                                                     int global_rank,
                                                     int split_size,
                                                     bool trans) {
  if (trans) {
    return get_trans_group_rank(world_size, global_rank, split_size);
  }
  int target_group_index = global_rank / split_size;
  uint64_t start_rank = target_group_index * split_size;
  uint64_t end_rank = start_rank + split_size;
  std::vector<uint64_t> group_rank;
  int index = global_rank - start_rank;
  for (uint64_t rank = start_rank; rank < end_rank; rank++) {
    group_rank.push_back(rank);
  }
  return {index, group_rank};
}

c10::intrusive_ptr<c10d::Store> create_tcp_store(const std::string& host,
                                                 int port,
                                                 int rank) {
  c10d::TCPStoreOptions tcp_options;
  tcp_options.isServer = (rank == 0);
  tcp_options.port = port;
  return c10::make_intrusive<c10d::TCPStore>(host, tcp_options);
}

void ProcessGroup::allreduce(torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  allreduce_async(input)->wait();
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allreduce_async(
    torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  std::vector<torch::Tensor> input_tensors = {input};
  return pg_->allreduce(input_tensors);
}

void ProcessGroup::allgather(const torch::Tensor& input,
                             std::vector<torch::Tensor>& outputs) {
  allgather_async(input, outputs)->wait();
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allgather_async(
    const torch::Tensor& input,
    std::vector<torch::Tensor>& outputs) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<std::vector<torch::Tensor>> output_tensors = {outputs};
  return pg_->allgather(output_tensors, input_tensors);
}

c10::intrusive_ptr<c10d::Work> ProcessGroup::allgather_base_async(
    const torch::Tensor& input,
    torch::Tensor& output) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK_EQ(input.device(), device())
      << "input should be on the same device as the process group";
  CHECK(output.defined()) << "output should be preallocated";
  CHECK_EQ(output.device(), device())
      << "output should be on the same device as the process group";
  CHECK(output.is_contiguous()) << "output should be contiguous";

  torch::Tensor input_buf = input.contiguous();
  const std::vector<int64_t> out_shape =
      get_gather_shape(world_size(), input_buf);
  CHECK_EQ(output.sizes(), torch::IntArrayRef(out_shape))
      << "output shape mismatch for allgather_base_async";
  c10d::AllgatherOptions opts;
  return pg_->_allgather_base(output, input_buf, opts);
}

torch::Tensor ProcessGroup::allgather_base_sync(const torch::Tensor& input) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK_EQ(input.device(), device())
      << "input should be on the same device as the process group";
  torch::Tensor output =
      torch::empty(get_gather_shape(world_size(), input), input.options());
  allgather_base_async(input, output)->wait();
  return output;
}

void ProcessGroup::reduce_scatter(const torch::Tensor& input,
                                  torch::Tensor& output) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  // make sure input is contiguous
  CHECK(input.is_contiguous()) << "input is not contiguous.";
  std::vector<torch::Tensor> input_tensors = {input};
  std::vector<torch::Tensor> output_tensors = {output};

  c10d::ReduceScatterOptions opts;
  // we use reduce operation SUM for reduce_scatter for default.
  opts.reduceOp = c10d::ReduceOp::SUM;
  pg_->reduce_scatter_tensor_coalesced(output_tensors, input_tensors, opts)
      ->wait();
}

void ProcessGroup::broadcast(torch::Tensor& input, int32_t root_rank) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  // single-rank group: nothing to unify, the local tensor is already the
  // source of truth.
  if (world_size() <= 1) {
    return;
  }
  CHECK(input.is_contiguous()) << "input is not contiguous.";
  std::vector<torch::Tensor> tensors = {input};
  c10d::BroadcastOptions opts;
  opts.rootRank = root_rank;
  pg_->broadcast(tensors, opts)->wait();
}

void ProcessGroup::all_to_all_single(
    torch::Tensor output,
    torch::Tensor input,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool async_op,
    c10::intrusive_ptr<c10d::Work>* async_work) {
  CHECK(pg_ != nullptr) << "Process group is not initialized.";
  CHECK(output.defined())
      << "Output of all_to_all_single function is not defined";
  CHECK(input.defined())
      << "Input of all_to_all_single function is not defined";
  if (input.is_complex()) {
    input = torch::view_as_real(input);
  }
  if (output.is_complex()) {
    output = torch::view_as_real(output);
  }

  auto opts = c10d::AllToAllOptions();
  auto work = pg_->alltoall_base(
      output, input, output_split_sizes, input_split_sizes, opts);
  if (async_op) {
    *async_work = work;
  } else {
    work->wait();
  }
}

std::string ProcessGroup::hccl_comm_name(bool init_comm) {
  (void)init_comm;
  CHECK(false) << "hccl_comm_name is only supported on NPU HCCL process group.";
  return "";
}

std::unique_ptr<ProcessGroup> create_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupImpl>(
      rank, world_size, rank_size, port, trans, host, group_name, device);
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
// We currently support explicit DiT communication groups on NPU, MLU, and DCU.
// TODO: This function is used by DiT models, since the DiT communication group
// info have already been calculated by rank_generator, we only need to pass the
// info to create the process groups. For any device that want to reuse the
// function and dit process groups, please implement the corresponding
// ProcessGroupImpl construct function.
std::unique_ptr<ProcessGroup> create_process_group(
    int32_t global_rank,
    int32_t local_rank,
    const std::vector<int32_t>& group_ranks,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device) {
  return std::make_unique<ProcessGroupImpl>(global_rank,
                                            local_rank,
                                            group_ranks,
                                            world_size,
                                            rank_size,
                                            port,
                                            host,
                                            group_name,
                                            device);
}
#endif
}  // namespace xllm
