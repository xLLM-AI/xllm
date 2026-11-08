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

#include <torch/types.h>

#include <atomic>

#include "core/common/macros.h"
#include "core/util/threadpool.h"
#include "model_memory_dist.pb.h"

namespace xllm {

// Executes memory resource commands on a worker. The protobuf service name is
// retained for compatibility with existing RPC clients.
class WorkerMemoryService final : public proto::ModelMemoryDist {
 public:
  WorkerMemoryService(int32_t global_rank,
                      int32_t world_size,
                      const torch::Device& device);
  ~WorkerMemoryService() override = default;

  // Mark service as initialized
  void set_initialized(bool initialized) { initialized_ = initialized; }

  // Service functions
  void Hello(::google::protobuf::RpcController* controller,
             const proto::Status* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  // Memory info query (get available/total memory before init)
  void GetMemoryInfo(::google::protobuf::RpcController* controller,
                     const proto::Status* request,
                     proto::MemoryInfoResponse* response,
                     ::google::protobuf::Closure* done) override;

  // Initialize PhysicalPagePool with specified number of pages
  void InitPhysicalPagePool(::google::protobuf::RpcController* controller,
                            const proto::InitPhysicalPagePoolRequest* request,
                            proto::Status* response,
                            ::google::protobuf::Closure* done) override;

  // KV tensor operations (partial mapping by offsets)
  void PrepareKvMapping(::google::protobuf::RpcController* controller,
                        const proto::PrepareKvMappingRequest* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;
  void CommitKvMapping(::google::protobuf::RpcController* controller,
                       const proto::KvMappingTransactionRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;
  void AbortKvMapping(::google::protobuf::RpcController* controller,
                      const proto::KvMappingTransactionRequest* request,
                      proto::Status* response,
                      ::google::protobuf::Closure* done) override;

  void PrepareWeightPages(::google::protobuf::RpcController* controller,
                          const proto::PrepareWeightPagesRequest* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) override;
  void CommitWeightPages(::google::protobuf::RpcController* controller,
                         const proto::KvMappingTransactionRequest* request,
                         proto::Status* response,
                         ::google::protobuf::Closure* done) override;
  void AbortWeightPages(::google::protobuf::RpcController* controller,
                        const proto::KvMappingTransactionRequest* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void MapToKvTensors(::google::protobuf::RpcController* controller,
                      const proto::KvTensorRequest* request,
                      proto::Status* response,
                      ::google::protobuf::Closure* done) override;

  void UnmapFromKvTensors(::google::protobuf::RpcController* controller,
                          const proto::KvTensorRequest* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) override;

  // Weight pages allocation from GlobalVirtualMemoryRegion
  void AllocWeightPages(::google::protobuf::RpcController* controller,
                        const proto::AllocWeightPagesRequest* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void FreeWeightPages(::google::protobuf::RpcController* controller,
                       const proto::FreeWeightPagesRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // Get virtual-memory KV block byte offsets in the shared physical-page pool
  // mapping for PD disaggregation.
  void GetVirtualKVCacheOffsets(
      ::google::protobuf::RpcController* controller,
      const proto::GetVirtualKVCacheOffsetsRequest* request,
      proto::GetVirtualKVCacheOffsetsResponse* response,
      ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(WorkerMemoryService);

 private:
  std::atomic<bool> initialized_{false};
  int32_t global_rank_;
  int32_t world_size_;
  torch::Device device_;
  // Budget queries and VMM operations require a device context on every RPC
  // thread, including threads that have not handled a memory snapshot yet.
  ThreadPool threadpool_;
};

}  // namespace xllm
