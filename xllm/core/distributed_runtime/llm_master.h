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

#include <glog/logging.h>

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/options.h"
#include "common/rate_limiter.h"
#include "common/types.h"
#include "framework/chat_template/chat_template.h"
#include "framework/request/llm_request_factory.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "llm_engine.h"
#include "master.h"
#include "scheduler/continuous_scheduler.h"
#include "speculative_engine.h"

namespace xllm {

class Call;
class Tokenizer;

class LLMMaster : public Master {
 public:
  explicit LLMMaster(const Options& options);
  ~LLMMaster() override;

  const ModelArgs* model_args() const override { return &model_args_; }

  // handle a request, the engine will execute the request asynchronously
  // completion/encode
  void handle_request(std::string prompt,
                      std::optional<std::vector<int>> prompt_tokens,
                      RequestParams sp,
                      std::optional<Call*> call,
                      OutputCallback callback);

  // chat
  void handle_request(std::vector<Message> messages,
                      std::optional<std::vector<int>> prompt_tokens,
                      RequestParams sp,
                      std::optional<Call*> call,
                      OutputCallback callback);

  // Render and tokenize on the request workers without scheduling generation.
  void count_chat_tokens(std::vector<Message> messages,
                         RequestParams params,
                         std::function<void(Status, int32_t)> callback);

  // batch completion
  void handle_batch_request(std::vector<std::string> prompts,
                            std::vector<RequestParams> sp,
                            BatchOutputCallback callback);

  // batch chat
  void handle_batch_request(std::vector<std::vector<Message>> conversations,
                            std::vector<RequestParams> sp,
                            BatchOutputCallback callback);

  // handle rpc response, send stream generations to xllm service
  bool handle_rpc_response(const RequestOutput& output);

  std::vector<bool> handle_rpc_responses(
      const std::vector<RequestOutput>& outputs);

  const Tokenizer& tokenizer() const {
    CHECK(tokenizer_ != nullptr)
        << "tokenizer() is only available on the leader rank (node_rank == 0).";
    return *tokenizer_;
  }

  const ChatTemplate& chat_template() const {
    CHECK(chat_template_ != nullptr);
    return *chat_template_;
  }

  // start running loop
  void run() override;

  // generate will run all request done at once,
  // this is a blocking call
  void generate();

  MasterStatus get_master_status() const { return master_status_; }

  bool is_sleeping() const { return master_status_ != MasterStatus::WAKEUP; }

  void set_master_status(MasterStatus master_status) {
    master_status_ = master_status;
  }

  bool sleep();

  bool wakeup();

  bool wakeup(const WakeupOptions& options);

  bool link_p2p(const std::vector<std::string>& remote_addrs);

  bool unlink_p2p(const std::vector<std::string>& remote_addrs);

  bool start_profile();
  bool stop_profile();

 private:
  MasterStatus master_status_;
  XServiceClient* xservice_client_ = nullptr;

  // Exactly one of these owners is populated, depending on the configured
  // LLM execution mode.
  std::unique_ptr<LLMEngine> llm_engine_;
  std::unique_ptr<SuffixSpeculativeEngine> suffix_engine_;
  std::unique_ptr<SpeculativeEngineBase<LLMEngine>> speculative_engine_;

  // Scheduler must be destroyed before the engine it references.
  std::unique_ptr<Scheduler> scheduler_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests
  std::unique_ptr<ThreadPool> threadpool_;

  // we don't know if tokenizer is thread safe, so we create one for each thread
  // for now
  std::unique_ptr<Tokenizer> tokenizer_;

  // chat template instance
  std::unique_ptr<ChatTemplate> chat_template_;

  // builds Request aggregates from prompts/messages + RequestParams
  std::unique_ptr<LLMRequestFactory> request_factory_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};

  std::string task_type_;
};

}  // namespace xllm
