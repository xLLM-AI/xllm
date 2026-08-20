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

#include "disagg_pd.pb.h"
#include "disagg_pd_service_impl.h"

namespace xllm {

class Engine;
class Request;
class PDOOCScheduler;

// a class to handle disagg_pd OOC requests
class PDOOCServiceImpl final : public DisaggPDServiceImpl {
 public:
  explicit PDOOCServiceImpl(PDOOCScheduler* scheduler, Engine* engine);
  ~PDOOCServiceImpl() = default;

  virtual void decode_recv_multi_generations(
      const proto::DisaggGenerationsRequests* request,
      proto::Status* response);

  virtual void prefill_recv_pull_signal(const proto::PullSignal* request,
                                        proto::Status* response);

 private:
  PDOOCScheduler* pd_ooc_scheduler_;  // not owned
};

}  // namespace xllm
