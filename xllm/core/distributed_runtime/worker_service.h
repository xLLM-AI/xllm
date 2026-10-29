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

#include <string>
#include <vector>

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

  // Call after RPC ingress stops and before joining outstanding requests.
  void begin_shutdown();

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

  void DSparkCalibration(::google::protobuf::RpcController* controller,
                         const proto::DSparkCalibrationRequest* request,
                         proto::DSparkCalibrationResponse* response,
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
                           const proto::PrefetchRequest* req,
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

  void ExecuteLLMModel(::google::protobuf::RpcController* controller,
                       const proto::LLMForwardRequest* request,
                       proto::LLMForwardResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ExecuteVLMModel(::google::protobuf::RpcController* controller,
                       const proto::VLMForwardRequest* request,
                       proto::VLMForwardResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ExecuteRECModel(::google::protobuf::RpcController* controller,
                       const proto::RECForwardRequest* request,
                       proto::RECForwardResponse* response,
                       ::google::protobuf::Closure* done) override;

  void ExecuteDITModel(::google::protobuf::RpcController* controller,
                       const proto::DITForwardRequest* request,
                       proto::DITForwardResponse* response,
                       ::google::protobuf::Closure* done) override;

  void GetLastLLMStepResult(::google::protobuf::RpcController* controller,
                            const proto::Empty* request,
                            proto::LLMForwardResponse* response,
                            ::google::protobuf::Closure* done) override;

  void GetLastVLMStepResult(::google::protobuf::RpcController* controller,
                            const proto::Empty* request,
                            proto::VLMForwardResponse* response,
                            ::google::protobuf::Closure* done) override;

  void GetActiveActivationMemory(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::ActivationMemory* resp,
                                 ::google::protobuf::Closure* done) override;

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
  friend class WorkerServiceOutputTest;

  void step_llm_model(
      LlmForwardInput& input,
      ForwardOutput& cpu_output,
      std::vector<SpeculativeTokenStats>& speculative_token_stats);
  void step_vlm_model(
      VlmForwardInput& input,
      ForwardOutput& cpu_output,
      std::vector<SpeculativeTokenStats>& speculative_token_stats);
  void step_rec_model(
      RecForwardInput& input,
      ForwardOutput& cpu_output,
      std::vector<SpeculativeTokenStats>& speculative_token_stats);
  void step_dit_model(const DiTForwardInput& input, ForwardOutput& cpu_output);

  void collect_llm_output(const ForwardOutput& output,
                          ForwardOutput& cpu_output,
                          bool cpu_ready_result,
                          std::vector<SpeculativeTokenStats>& stats);
  void collect_vlm_output(const ForwardOutput& output,
                          ForwardOutput& cpu_output,
                          std::vector<SpeculativeTokenStats>& stats);

  void poll_llm_shm(
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager);
  void poll_vlm_shm(
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager);
  void poll_rec_shm(
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager);
  void poll_dit_shm(
      std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager);

  DISALLOW_COPY_AND_ASSIGN(WorkerService);

 private:
  // runtime options
  runtime::Options options_;

  bool initialized_;

  Device device_;

  std::unique_ptr<Stream> stream_;

  std::unique_ptr<Worker> worker_;

  std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager_;
  std::unique_ptr<std::thread> polling_thread_;

  std::unique_ptr<ThreadPool> threadpool_;

  ThreadPool copy_threadpool_{/*num_threads=*/5,
                              /*cpu_binding=*/false,
                              /*pool_name=*/"WorkerService.copy"};
};

}  // namespace xllm
