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
#include <cstdint>
#include <functional>

#include "common/macros.h"
#include "common/types.h"
#include "util/threadpool.h"

namespace xllm {

class BlockManager;
class Request;
class Sequence;
class Tokenizer;
class AsyncResponseProcessor final {
 public:
  AsyncResponseProcessor(
      const Tokenizer* tokenizer,
      const std::optional<InstanceRole>& role,
      bool enable_service_routing,
      bool disable_log_stats,
      std::function<void(std::shared_ptr<Request>)> cancel_request);
  ~AsyncResponseProcessor();

  void process_completed_request(std::shared_ptr<Request> request);

  void process_failed_request(std::shared_ptr<Request> request, Status status);

  // in disagg pd mode, decode send requests' responses to prefill
  void process_completed_requests(
      std::vector<std::shared_ptr<Request>>& requests);

  void process_stream_request(std::shared_ptr<Request> request);

  // in disagg pd mode, decode send requests' responses to prefill
  void process_stream_requests(std::vector<std::shared_ptr<Request>>& requests);

  void batch_process_stream_requests(
      std::vector<std::shared_ptr<Request>>& requests);

  void batch_process_completed_requests(
      std::vector<std::shared_ptr<Request>>& requests);
  // wait for all responses in queue to be handled
  void wait_completion();

 private:
  DISALLOW_COPY_AND_ASSIGN(AsyncResponseProcessor);
  // the threadpool to handle responses
  ThreadPool response_threadpool_;

  // the threadpool to handle rpc
  ThreadPool rpc_threadpool_;

  // the threadpool to generate outputs
  ThreadPool generate_output_threadpool_;

  // tokenizer instance to decode token ids
  std::unique_ptr<Tokenizer> tokenizer_;

  InstanceRole role_ = InstanceRole::DEFAULT;

  // for decode instance in disagg pd mode,
  // `True` means decode instance will response batch request_outputs to
  // prefill or xllm service, this will decrease rpc cost.
  bool enable_batch_response_ = false;

  bool disable_log_stats_ = false;

  std::function<void(std::shared_ptr<Request>)> cancel_request_;
};

}  // namespace xllm
