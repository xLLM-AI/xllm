/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>

#include <cstdint>
#include <functional>
#include <memory>

#include "core/util/threadpool.h"

namespace xllm {

// One ordered RPC stream per stage. A stage retires all its TP workers before
// reusing input buffers, while other stages can execute different microbatches.
class PipelineDispatcher final {
 public:
  explicit PipelineDispatcher(int32_t stages) : stages_(stages) {
    CHECK_GT(stages, 0);
    pool_ =
        std::make_unique<ThreadPool>(stages, /*cpu_binding=*/false, "pipeline");
  }

  void run(int32_t microbatches,
           const std::function<void(int32_t, int32_t)>& execute) {
    CHECK_GT(microbatches, 0);
    TaskGroup group(stages_);
    for (int32_t stage = 0; stage < stages_; ++stage) {
      pool_->schedule(group.wrap([&, stage] {
        for (int32_t microbatch = 0; microbatch < microbatches; ++microbatch) {
          execute(stage, microbatch);
        }
      }));
    }
    group.wait();
  }

 private:
  int32_t stages_;
  std::unique_ptr<ThreadPool> pool_;
};

}  // namespace xllm
