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

#pragma once

#include <torch/torch.h>

#include <string>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <vector>

#if defined(USE_NPU)
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>
#endif

namespace xllm {

class ProcessGroupImpl;

std::pair<int, std::vector<uint64_t>> get_group_rank(int world_size,
                                                     int global_rank,
                                                     int split_size,
                                                     bool trans);

c10::intrusive_ptr<c10d::Store> create_tcp_store(const std::string& host,
                                                 int port,
                                                 int rank);

class ProcessGroup {
 public:
  ProcessGroup(int32_t rank, int32_t world_size, const torch::Device& device)
      : rank_(rank), world_size_(world_size), device_(device) {}

  virtual ~ProcessGroup();

  int32_t rank() const {
    if (pg_ == nullptr) {
      return rank_;
    }
    return pg_->getRank();
  }

  int32_t world_size() const {
    if (pg_ == nullptr) {
      return world_size_;
    }
    return pg_->getSize();
  }

  const torch::Device& device() const { return device_; }

  // allreduce: reduce the input tensor across all processes, and all processes
  // get the result.
  virtual void allreduce(torch::Tensor& input);

  // allreduce_async: enqueue allreduce and return an async work handle.
  virtual c10::intrusive_ptr<c10d::Work> allreduce_async(torch::Tensor& input);

  // allgather: gather tensors from all processes and concatenate them.
  virtual void allgather(const torch::Tensor& input,
                         std::vector<torch::Tensor>& outputs);

  // allgather_async: enqueue allgather and return an async work handle.
  virtual c10::intrusive_ptr<c10d::Work> allgather_async(
      const torch::Tensor& input,
      std::vector<torch::Tensor>& outputs);

  // allgather_base_async: gather tensors from all processes into one stacked
  // output with shape [world_size, *input.shape].
  virtual c10::intrusive_ptr<c10d::Work> allgather_base_async(
      const torch::Tensor& input,
      torch::Tensor& output);

  // allgather_base_sync: gather tensors from all processes into one stacked
  // output with shape [world_size, *input.shape].
  virtual torch::Tensor allgather_base_sync(const torch::Tensor& input);

  // reduce_scatter: reduce the input tensor across all processes, scatter the
  // reduced chunks to all processes so that each process gets one chunk of the
  // result. we use default dim 0 for reduce_scatter.
  virtual void reduce_scatter(const torch::Tensor& input,
                              torch::Tensor& output);

  // broadcast: in-place overwrite the input tensor on every process with the
  // copy held by root_rank. Used to unify per-rank sampling results to a single
  // source of truth across the consensus group.
  virtual void broadcast(torch::Tensor& input, int32_t root_rank = 0);

  virtual void all_to_all_single(
      torch::Tensor output,
      torch::Tensor input,
      std::vector<int64_t> output_split_sizes = {},
      std::vector<int64_t> input_split_sizes = {},
      bool async_op = false,
      c10::intrusive_ptr<c10d::Work>* async_work = nullptr);

  virtual std::string hccl_comm_name(bool init_comm = true);

 private:
  // rank of current process.
  int32_t rank_ = 0;

  // number of processes.
  int32_t world_size_ = 0;

  // device of current process.
  torch::Device device_;

 protected:
  void shutdown_backend();

#if defined(USE_NPU) &&         \
    (TORCH_VERSION_MAJOR < 2 || \
     (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 7))
  // Using ProcessGroupHCCL for NPU devices
  // Note: torch_npu uses an older torch version where c10d::Backend lacks
  // shutdown() method
  std::unique_ptr<c10d_npu::ProcessGroupHCCL> pg_{nullptr};
#else
  std::unique_ptr<c10d::Backend> pg_{nullptr};
#endif
};

std::unique_ptr<xllm::ProcessGroup> create_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    bool trans,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
// for DiT models
std::unique_ptr<xllm::ProcessGroup> create_process_group(
    int32_t global_rank,
    int32_t local_rank,
    const std::vector<int32_t>& group_ranks,
    int32_t world_size,
    int32_t rank_size,
    int32_t port,
    const std::string& host,
    const std::string& group_name,
    const torch::Device& device);
#endif

}  // namespace xllm
