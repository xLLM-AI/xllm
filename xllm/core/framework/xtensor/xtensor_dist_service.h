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

#include <torch/torch.h>

#include "common/macros.h"
#include "util/threadpool.h"
#include "xtensor_dist.pb.h"

namespace xllm {

class XTensorDistService : public proto::XTensorDist {
 public:
  XTensorDistService(int32_t global_rank,
                     int32_t world_size,
                     const torch::Device& device);
  ~XTensorDistService() = default;

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

  // Initialize PhyPagePool with specified number of pages
  void InitPhyPagePool(::google::protobuf::RpcController* controller,
                       const proto::InitPhyPagePoolRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // KV tensor operations (partial mapping by offsets)
  void MapToKvTensors(::google::protobuf::RpcController* controller,
                      const proto::KvTensorRequest* request,
                      proto::Status* response,
                      ::google::protobuf::Closure* done) override;

  void UnmapFromKvTensors(::google::protobuf::RpcController* controller,
                          const proto::KvTensorRequest* request,
                          proto::Status* response,
                          ::google::protobuf::Closure* done) override;

  // Weight pages allocation from GlobalXTensor
  void AllocWeightPages(::google::protobuf::RpcController* controller,
                        const proto::AllocWeightPagesRequest* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void FreeWeightPages(::google::protobuf::RpcController* controller,
                       const proto::FreeWeightPagesRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  // Get XTensor offsets for KV cache blocks (used in PD disaggregation)
  void GetXTensorOffsets(::google::protobuf::RpcController* controller,
                         const proto::GetXTensorOffsetsRequest* request,
                         proto::GetXTensorOffsetsResponse* response,
                         ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorDistService);

 private:
  bool initialized_;
  int32_t global_rank_;
  int32_t world_size_;
  torch::Device device_;
  ThreadPool threadpool_{/*num_threads=*/4,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"XTensorDistService.rpc"};
};

}  // namespace xllm
