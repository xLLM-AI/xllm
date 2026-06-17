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

#pragma once

#include <brpc/channel.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common/macros.h"
#include "util/threadpool.h"
#include "xtensor.h"  // For offset_t type definition
#include "xtensor_dist.pb.h"

namespace xllm {

// Memory info returned by GetMemoryInfo RPC
struct MemoryInfo {
  int64_t available_memory;  // Available memory in bytes
  int64_t total_memory;      // Total memory in bytes
};

// Remote client for distributed XTensor operations via brpc
class XTensorDistClient {
 public:
  explicit XTensorDistClient(int32_t global_rank,
                             const std::string& server_address,
                             const torch::Device& device);
  ~XTensorDistClient() = default;

  // Wait for server to be ready
  bool wait_for_server_ready(const std::string& server_address);

  // Get memory info from remote worker (available/total memory)
  folly::SemiFuture<MemoryInfo> get_memory_info_async();

  // Initialize PhyPagePool on remote worker with specified number of pages
  folly::SemiFuture<bool> init_phy_page_pool_async(int64_t num_pages);

  // KV tensor operations (partial mapping by offsets)
  folly::SemiFuture<bool> map_to_kv_tensors_async(
      const std::string& model_id,
      const std::vector<offset_t>& offsets);
  folly::SemiFuture<bool> unmap_from_kv_tensors_async(
      const std::string& model_id,
      const std::vector<offset_t>& offsets);

  // Weight pages allocation from GlobalXTensor
  folly::SemiFuture<bool> alloc_weight_pages_async(const std::string& model_id,
                                                   size_t num_pages);
  folly::SemiFuture<bool> free_weight_pages_async(const std::string& model_id);

  // Get XTensor offsets for KV cache blocks (used in PD disaggregation)
  // Returns per-layer K/V offsets for each block
  // Result: layer_offsets[layer_id] = {k_offsets, v_offsets}
  // Returns empty vector on error
  folly::SemiFuture<
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>>
  get_xtensor_offsets_async(const std::string& model_id,
                            const std::vector<int32_t>& block_ids,
                            uint64_t block_size_bytes);

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorDistClient);

 private:
  int32_t global_rank_;
  torch::Device device_;

  // brpc connection resources
  brpc::Channel channel_;
  brpc::ChannelOptions options_;
  std::unique_ptr<proto::XTensorDist_Stub> stub_;

  // Thread pool for async operations
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"XTensorDistClient.async"};
};

}  // namespace xllm
