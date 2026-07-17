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

#include <memory>
#include <vector>

#include "common/macros.h"
#include "mode_switch.pb.h"
#include "runtime/worker_client.h"

namespace xllm {

class LLMEngine;
class ContinuousScheduler;

class ModeSwitchService : public proto::ModeSwitchService {
 public:
  // engine is required. scheduler is optional but required to safely quiesce
  // async surfaces (dispatch_thread, brpc handlers) before rebuilding the
  // KV allocator; passing nullptr falls back to the pre-drain path (only
  // safe for tests where nothing else touches the pool).
  explicit ModeSwitchService(LLMEngine* engine,
                             ContinuousScheduler* scheduler = nullptr);
  ModeSwitchService() : engine_(nullptr), scheduler_(nullptr) {}
  virtual ~ModeSwitchService() = default;

  void SwitchMode(::google::protobuf::RpcController* controller,
                  const proto::InstanceModeSwitchRequest* request,
                  proto::InstanceModeSwitchResponse* response,
                  ::google::protobuf::Closure* done) override;

 private:
  LLMEngine* engine_;
  // Optional; if the deployment is disaggregated PD this points at the
  // DisaggPDScheduler that owns the request-dispatch async surface. The
  // service uses it to pause the loop and take switch_gate_ unique across
  // engine_->switch_mode + rebuild_block_manager_pool.
  ContinuousScheduler* scheduler_;
  std::atomic<int32_t> current_mode_{0};

  DISALLOW_COPY_AND_ASSIGN(ModeSwitchService);
};

}  // namespace xllm
