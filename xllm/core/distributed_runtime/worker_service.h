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

#include <string>

#include "runtime/forward_shared_memory_manager.h"
#include "runtime/worker.h"
#include "worker.pb.h"

namespace xllm {

class WorkerService : public proto::DistributeWorker {
 public:
  WorkerService(runtime::Options options, const torch::Device& device);
  WorkerService(runtime::Options options,
                const torch::Device& device,
                std::unique_ptr<Worker> worker);

  virtual ~WorkerService();

  void set_worker(std::unique_ptr<Worker> worker);

  void create_polling_shm_thread(
      std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager,
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager);

  // service functions
  void Hello(::google::protobuf::RpcController* controller,
             const proto::Status* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  void InitModel(::google::protobuf::RpcController* controller,
                 const proto::InitModelRequest* request,
                 proto::Status* response,
                 ::google::protobuf::Closure* done) override;

  void ProcessGroupTest(::google::protobuf::RpcController* controller,
                        const proto::Empty* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void ProfileDeviceMemory(::google::protobuf::RpcController* controller,
                           const proto::Empty* request,
                           proto::DeviceMemory* response,
                           ::google::protobuf::Closure* done) override;

  void AllocateKVCache(::google::protobuf::RpcController* controller,
                       const proto::AllocateKVCacheRequest* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  void AllocateKVCacheWithTransfer(
      ::google::protobuf::RpcController* controller,
      const proto::AllocateKVCacheRequest* req,
      proto::Status* resp,
      ::google::protobuf::Closure* done) override;

  void PullKVCache(::google::protobuf::RpcController* controller,
                   const proto::PullKVCacheRequest* req,
                   proto::Status* resp,
                   ::google::protobuf::Closure* done) override;

  void TransferBlocks(::google::protobuf::RpcController* controller,
                      const proto::BlockTransferInfos* req,
                      proto::TransferStatus* resp,
                      ::google::protobuf::Closure* done) override;

  void PrefetchFromStorage(google::protobuf::RpcController* controller,
                           const proto::BlockTransferInfos* req,
                           proto::Status* resp,
                           google::protobuf::Closure* done) override;

  void GetCacheInfo(::google::protobuf::RpcController* controller,
                    const proto::Empty* req,
                    proto::CacheInfo* resp,
                    ::google::protobuf::Closure* done) override;

  void LinkCluster(::google::protobuf::RpcController* controller,
                   const proto::ClusterInfo* req,
                   proto::Status* resp,
                   ::google::protobuf::Closure* done) override;

  void UnlinkCluster(::google::protobuf::RpcController* controller,
                     const proto::ClusterInfo* req,
                     proto::Status* resp,
                     ::google::protobuf::Closure* done) override;

  void LinkP2P(::google::protobuf::RpcController* controller,
               const proto::P2PLinkWorkerRequest* req,
               proto::Status* resp,
               ::google::protobuf::Closure* done) override;

  void UnlinkP2P(::google::protobuf::RpcController* controller,
                 const proto::P2PLinkWorkerRequest* req,
                 proto::Status* resp,
                 ::google::protobuf::Closure* done) override;

  void ExecuteModel(::google::protobuf::RpcController* controller,
                    const proto::ForwardInput* pb_fwd_input,
                    proto::ForwardOutput* pb_forward_output,
                    ::google::protobuf::Closure* done) override;

  void GetLastStepResult(::google::protobuf::RpcController* controller,
                         const proto::Empty* req,
                         proto::ForwardOutput* pb_forward_output,
                         ::google::protobuf::Closure* done) override;

  void GetActiveActivationMemory(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::ActivationMemory* resp,
                                 ::google::protobuf::Closure* done) override;

  void UpdateWeights(::google::protobuf::RpcController* controller,
                     const proto::UpdateWeightsRequest* req,
                     proto::Status* resp,
                     ::google::protobuf::Closure* done);

  void Sleep(::google::protobuf::RpcController* controller,
             const proto::SleepRequest* req,
             proto::Status* resp,
             ::google::protobuf::Closure* done) override;

  void Wakeup(::google::protobuf::RpcController* controller,
              const proto::WakeupRequest* req,
              proto::Status* resp,
              ::google::protobuf::Closure* done) override;

  void StartProfile(::google::protobuf::RpcController* controller,
                    const proto::Empty* req,
                    proto::Status* resp,
                    ::google::protobuf::Closure* done) override;

  void StopProfile(::google::protobuf::RpcController* controller,
                   const proto::Empty* req,
                   proto::Status* resp,
                   ::google::protobuf::Closure* done) override;

 private:
  void step(ForwardInput& fwd_input,
            torch::Tensor& next_tokens,
            torch::Tensor& logprobs,
            torch::Tensor& top_tokens,
            torch::Tensor& top_logprobs,
            torch::Tensor& embeddings,
            std::vector<std::vector<torch::Tensor>>& mm_embeddings,
            std::vector<torch::Tensor>& dit_images,
            std::vector<std::string>& dit_text_output,
            torch::Tensor& expert_load_data,
            int32_t& prepared_layer_id,
            torch::Tensor& src_seq_idxes,
            torch::Tensor& out_tokens,
            torch::Tensor& out_logprobs);
  DISALLOW_COPY_AND_ASSIGN(WorkerService);

 private:
  // runtime options
  runtime::Options options_;

  bool initialized_;

  Device device_;

  std::unique_ptr<Stream> stream_;

  std::unique_ptr<Worker> worker_;

  std::unique_ptr<std::thread> polling_thread_;

  std::unique_ptr<ThreadPool> threadpool_;

  ThreadPool copy_threadpool_{/*num_threads=*/5,
                              /*cpu_binding=*/false,
                              /*pool_name=*/"WorkerService.copy"};
};

}  // namespace xllm
