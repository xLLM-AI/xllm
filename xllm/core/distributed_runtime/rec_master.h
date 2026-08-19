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

#include <atomic>
#include <functional>
#include <optional>
#include <thread>

#include "framework/chat_template/jinja_chat_template.h"
#include "framework/model/model_args.h"
#include "framework/request/rec_type.h"
#include "master.h"
#include "rec.pb.h"
#include "rec_engine.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/fixed_steps_scheduler.h"
#include "util/threadpool.h"

namespace xllm {

class RecMaster : public Master {
 public:
  explicit RecMaster(const Options& options);
  ~RecMaster();

  // handle a request, the engine will execute the request asynchronously
  // completion/encode
  void handle_request(
      std::string prompt,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      OutputCallback callback);

  // chat
  // Only supported for LlmRec models.
  void handle_request(
      std::vector<Message> messages,
      std::optional<std::vector<int>> prompt_tokens,
      std::optional<std::vector<proto::InferInputTensor>> input_tensors,
      RequestParams sp,
      OutputCallback callback);

  void handle_request(const std::vector<int>& prompt_tokens,
                      std::optional<MMData> mm_data,
                      RequestParams sp,
                      OutputCallback callback);

  // start the handling loop
  void run() override;

  RecType rec_type() const { return rec_type_; }

 private:
  using RequestBuilder =
      std::function<std::shared_ptr<Request>(const RequestParams&,
                                             OutputCallback)>;

  // ============================================================
  // RecMasterPipeline: Abstract base class for request processing
  // ============================================================
  class RecMasterPipeline {
   public:
    explicit RecMasterPipeline(RecMaster& master) : master_(master) {}
    virtual ~RecMasterPipeline() = default;

    // For prompt-based input (OneRec and LlmRec without mm_data)
    virtual std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback);

    // For raw input (LlmRec with mm_data)
    virtual std::shared_ptr<Request> generate_request(
        const std::vector<int>& prompt_tokens,
        std::optional<MMData> mm_data,
        const RequestParams& sp,
        OutputCallback callback);

   protected:
    std::shared_ptr<Request> generate_onerec_request_common(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback,
        bool build_stop_checker);

    RecMaster& master_;
  };

  // LlmRecMasterPipeline - pure qwen3 (prompt-based, no mm_data)
  class LlmRecMasterPipeline final : public RecMasterPipeline {
   public:
    explicit LlmRecMasterPipeline(RecMaster& master);
    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // LlmRecWithMmDataMasterPipeline - qwen3 with embedding (raw input)
  class LlmRecWithMmDataMasterPipeline final : public RecMasterPipeline {
   public:
    explicit LlmRecWithMmDataMasterPipeline(RecMaster& master);
    std::shared_ptr<Request> generate_request(
        const std::vector<int>& prompt_tokens,
        std::optional<MMData> mm_data,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // OneRecPrefillOnlyMasterPipeline - legacy OneRec without stop checker.
  class OneRecPrefillOnlyMasterPipeline final : public RecMasterPipeline {
   public:
    explicit OneRecPrefillOnlyMasterPipeline(RecMaster& master);
    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // OneRecXAttentionMasterPipeline - OneRec xattention with stop checker for
  // device-side multi-round decode.
  class OneRecXAttentionMasterPipeline final : public RecMasterPipeline {
   public:
    explicit OneRecXAttentionMasterPipeline(RecMaster& master)
        : RecMasterPipeline(master) {}

    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
  };

  // Factory method to create pipeline (can access private classes)
  static std::unique_ptr<RecMasterPipeline> create_pipeline(
      RecPipelineType type,
      RecMaster& master);

  void schedule_request(RequestParams sp,
                        OutputCallback callback,
                        RequestBuilder build_request);

  std::shared_ptr<Request> build_request_common(
      std::string prompt,
      std::vector<int32_t> prompt_tokens,
      MMData mm_data,
      const RequestParams& sp,
      OutputCallback callback,
      bool build_stop_checker);

  // Pipeline instances
  std::unique_ptr<RecMasterPipeline> pipeline_;
  std::unique_ptr<RecMasterPipeline> mm_data_pipeline_;

  std::unique_ptr<FixedStepsScheduler> scheduler_;
  // model args
  ModelArgs model_args_;
  RecType rec_type_ = RecType::kNone;
  std::unique_ptr<ThreadPool> threadpool_;
  std::unique_ptr<Tokenizer> tokenizer_;
  // chat template instance
  std::unique_ptr<JinjaChatTemplate> chat_template_;
  // thread for moving forward the scheduler
  std::thread loop_thread_;
  // flag to stop the loop
  std::atomic<bool> stopped_{false};

  // flag to indicate if the handler is running
  std::atomic<bool> running_{false};
};

}  // namespace xllm
