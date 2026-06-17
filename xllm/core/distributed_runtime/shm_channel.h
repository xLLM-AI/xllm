/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include "comm_channel.h"
#include "runtime/forward_shared_memory_manager.h"
#include "runtime/options.h"

namespace xllm {

class ShmChannel : public CommChannel {
 public:
  explicit ShmChannel(int dp_group,
                      int rank,
                      bool is_driver,
                      const runtime::Options& options);
  ~ShmChannel() = default;

  void execute_model_async(
      const ForwardInput& input,
      folly::Promise<std::optional<RawForwardOutput>>& promise) override;

 private:
  bool execute_model_with_shm(const ForwardInput& input,
                              RawForwardOutput& raw_output);

  bool enable_shm_ = false;
  std::unique_ptr<ForwardSharedMemoryManager> input_shm_manager_ = nullptr;
  std::unique_ptr<ForwardSharedMemoryManager> output_shm_manager_ = nullptr;
};

}  // namespace xllm
