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

#include <folly/Function.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "common/options.h"
#include "common/types.h"
#include "core/framework/multimodal/mm_input.h"
#include "engine.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/tokenizer/tokenizer.h"
#include "master.h"
#include "scheduler/continuous_scheduler.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {

struct MMData;

class VLMMaster : public Master {
 public:
  explicit VLMMaster(const Options& options);
  ~VLMMaster();

  // completion
  void handle_request(std::string prompt,
                      MMData mm_data,
                      RequestParams sp,
                      OutputCallback callback);

  // chat
  void handle_request(std::vector<Message> messages,
                      RequestParams sp,
                      std::string payload,
                      OutputCallback callback);

  // batch completion
  void handle_batch_request(std::vector<std::string> prompts,
                            std::vector<MMData> mm_datas,
                            std::vector<RequestParams> sps,
                            BatchOutputCallback callback);

  // batch completion with image urls/paths (no python image processor)
  void handle_batch_request_with_image_urls(
      std::vector<std::string> prompts,
      std::vector<std::vector<std::string>> image_urls,
      std::vector<RequestParams> sps,
      BatchOutputCallback callback);

  // batch chat
  void handle_batch_request(std::vector<std::vector<Message>> conversations,
                            std::vector<RequestParams> sps,
                            BatchOutputCallback callback);

  // start the handling loop
  void run() override;

  // generate will run all requests, this is an blocking call
  void generate();

  int get_image_limit() { return options_.limit_image_per_prompt(); }

 private:
  using Task = folly::Function<void()>;
  std::shared_ptr<Request> build_request(std::string prompt,
                                         std::vector<int32_t> prompt_tokens,
                                         MMData mm_data,
                                         RequestParams sp,
                                         OutputCallback callback);

  std::shared_ptr<Request> generate_request(std::string prompt,
                                            MMData mm_data,
                                            RequestParams sp,
                                            OutputCallback callback);

  std::shared_ptr<Request> generate_request(std::vector<Message> messages,
                                            RequestParams sp,
                                            std::string payload,
                                            OutputCallback callback);

  std::unique_ptr<Scheduler> scheduler_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  std::unique_ptr<ThreadPool> threadpool_;

  std::unique_ptr<JinjaChatTemplate> chat_template_;
  std::unique_ptr<MultimodalProcessorBase> processor_;
  std::shared_ptr<Tokenizer> tokenizer_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

class VLMAssistantMaster : public Master {
 public:
  explicit VLMAssistantMaster(const Options& options);
  ~VLMAssistantMaster();
  void run() override;

  static void handle_signal(int signum) { running_ = false; }

 private:
  std::thread loop_thread_;
  static volatile bool running_;
};

}  // namespace xllm
