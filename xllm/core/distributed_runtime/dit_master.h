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

#include <folly/Function.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "framework/request/dit_request_params.h"
#include "master.h"
#include "scheduler/dit_scheduler.h"

namespace xllm {

class Call;

class DiTMaster : public Master {
 public:
  explicit DiTMaster(const Options& options);
  ~DiTMaster();

  // handle a request, the engine will execute the request asynchronously
  void handle_request(DiTRequestParams params,
                      std::optional<Call*> call,
                      DiTOutputCallback callback);

  // batch generation
  void handle_batch_request(std::vector<DiTRequestParams> params_vec,
                            BatchDiTOutputCallback callback);

  // start running loop
  void run() override;

  // generate will run all request done at once,
  // this is a blocking call
  void generate();

 private:
  std::unique_ptr<DiTScheduler> scheduler_;

  // thread pool for handling requests
  std::unique_ptr<ThreadPool> threadpool_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

class DiTAssistantMaster : public Master {
 public:
  DiTAssistantMaster(const Options& options);
  ~DiTAssistantMaster();
  void run() override;
  void wait() override;

  static void handle_signal(int signum) { running_ = false; }

 private:
  std::thread loop_thread_;
  static volatile bool running_;
};

}  // namespace xllm
