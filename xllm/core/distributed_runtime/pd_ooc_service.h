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

#include "common/macros.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_service.h"
#include "pd_ooc_service_impl.h"

namespace xllm {

class PDOOCService : public DisaggPDService {
 public:
  explicit PDOOCService(PDOOCScheduler* scheduler, Engine* engine);
  virtual ~PDOOCService() = default;

  // for decode recv multiple tokens from prefill
  void MultiGenerations(::google::protobuf::RpcController* controller,
                        const proto::DisaggGenerationsRequests* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void SendPullSignal(::google::protobuf::RpcController* controller,
                      const proto::PullSignal* request,
                      proto::Status* response,
                      ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(PDOOCService);
  PDOOCServiceImpl* pd_ooc_service_impl_;  // owned by base class
};

}  // namespace xllm
