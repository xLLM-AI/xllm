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
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/options.h"
#include "common/types.h"
#include "core/framework/multimodal/mm_input.h"
#include "framework/chat_template/jinja_chat_template.h"
#include "framework/request/request_output.h"
#include "framework/request/request_params.h"
#include "framework/request/vlm_request_factory.h"
#include "framework/tokenizer/tokenizer.h"
#include "master.h"
#include "scheduler/continuous_scheduler.h"
#include "speculative_engine.h"
#include "xllm/processors/multimodal_processor.h"

namespace xllm {

class ModelResidencyCoordinator;
class KVTransferCoordinator;
class Call;
class XServiceClient;

class VLMMaster : public Master {
 public:
  explicit VLMMaster(const Options& options);
  ~VLMMaster() override;

  const ModelArgs* model_args() const override { return &model_args_; }

  void handle_request(std::string prompt,
                      std::optional<std::vector<int32_t>> prompt_tokens,
                      RequestParams sp,
                      std::optional<Call*> call,
                      OutputCallback callback);

  void handle_request(std::vector<Message> messages,
                      std::optional<std::vector<int32_t>> prompt_tokens,
                      RequestParams sp,
                      std::optional<Call*> call,
                      OutputCallback callback);

  void handle_request(std::vector<Message> messages,
                      std::optional<std::vector<int32_t>> prompt_tokens,
                      RequestParams sp,
                      MMInput mm_input,
                      std::optional<Call*> call,
                      OutputCallback callback,
                      bool use_prompt_as_is = false);

  // completion
  void handle_request(std::string prompt,
                      MMData mm_data,
                      RequestParams sp,
                      OutputCallback callback);

  // chat
  void handle_request(std::vector<Message> messages,
                      RequestParams sp,
                      MMInput mm_input,
                      OutputCallback callback,
                      bool use_prompt_as_is = false);

  // batch completion
  void handle_batch_request(std::vector<std::string> prompts,
                            std::vector<RequestParams> sps,
                            BatchOutputCallback callback);

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

  void count_chat_tokens(std::vector<Message> messages,
                         RequestParams params,
                         std::function<void(Status, int32_t)> callback);

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

  // start the handling loop
  void run() override;

  // generate will run all requests, this is an blocking call
  void generate();

  bool start_profile();
  bool stop_profile();

  MasterStatus get_master_status() const { return master_status_; }
  bool is_sleeping() const { return master_status_ != MasterStatus::WAKEUP; }
  void set_master_status(MasterStatus status) { master_status_ = status; }
  bool sleep();
  bool requires_wakeup() const;
  bool wakeup(const WakeupOptions& options = {});
  bool link_p2p(const std::vector<std::string>& remote_addrs);
  bool unlink_p2p(const std::vector<std::string>& remote_addrs);

  int get_image_limit() { return options_.limit_image_per_prompt(); }

  const std::string& model_type() const { return model_args_.model_type(); }

 private:
  void schedule_completion(std::string prompt,
                           std::optional<std::vector<int32_t>> prompt_tokens,
                           MMData mm_data,
                           RequestParams sp,
                           std::optional<Call*> call,
                           OutputCallback callback);

  void warmup_multimodal();

  MasterStatus master_status_;
  std::shared_ptr<DistributedWorkerManager> distributed_worker_manager_;
  std::shared_ptr<ModelResidencyCoordinator> model_residency_coordinator_;
  std::shared_ptr<KVTransferCoordinator> kv_transfer_coordinator_;
  XServiceClient* xservice_client_ = nullptr;

  // The LLM draft prepares VirtualMemory pages before loading worker weights.
  std::unique_ptr<ModelResidencyCoordinator> draft_model_residency_coordinator_;

  // Exactly one of these owners is populated for VLM and VLM speculative
  // execution respectively.
  std::unique_ptr<VLMEngine> vlm_engine_;
  std::unique_ptr<SpeculativeEngineBase<VLMEngine>> speculative_engine_;

  // Scheduler must be destroyed before the engine it references.
  std::unique_ptr<Scheduler> scheduler_;

  // model args
  ModelArgs model_args_;

  // thread pool for handling requests. Its worker lambdas dereference
  // request_factory_ and scheduler_, so the pool is explicitly reset
  // (drained/joined) in ~VLMMaster before those members are destroyed.
  std::unique_ptr<ThreadPool> threadpool_;

  std::unique_ptr<JinjaChatTemplate> chat_template_;
  std::unique_ptr<MultimodalProcessorBase> processor_;
  std::shared_ptr<Tokenizer> tokenizer_;

  // builds Request aggregates from prompts/messages + multimodal data
  std::unique_ptr<VLMRequestFactory> request_factory_;

  // thread for moving forward the scheduler
  std::thread loop_thread_;

  // flag to stop the loop
  std::atomic_bool stoped_{false};

  // flag to indicate if the handler is running
  std::atomic_bool running_{false};
};

}  // namespace xllm
