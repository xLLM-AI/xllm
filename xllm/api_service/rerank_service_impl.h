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
#include <absl/container/flat_hash_set.h>

#include <atomic>
#include <functional>
#include <vector>

#include "api_service/api_service_impl.h"
#include "api_service/call.h"
#include "api_service/non_stream_call.h"
#include "rerank.pb.h"

namespace xllm {

using RerankCall = NonStreamCall<proto::RerankRequest, proto::RerankResponse>;

struct RerankRequestOutput {
  int32_t index = 0;
  std::string document = "";
  float score = 0.0f;

  RerankRequestOutput(int32_t index, std::string document, float score)
      : index(index), document(std::move(document)), score(score) {}
};

// Score computer function type: computes scores from request outputs
// Returns vector of RerankRequestOutput with computed scores
using ScoreComputer = std::function<std::vector<RerankRequestOutput>(
    const std::vector<std::string>& documents,
    const std::vector<RequestOutput>& req_outputs)>;

// Shared context for async aggregation of rerank sub-request results
// Template parameter allows different score computation strategies
struct RerankContext {
  std::shared_ptr<RerankCall> call;
  std::vector<std::string> documents;
  std::string model;
  std::string request_id;
  int32_t top_n;

  std::vector<RequestOutput> req_outputs;
  std::atomic<size_t> pending_count;

  ScoreComputer compute_scores;

  RerankContext(std::shared_ptr<RerankCall> call,
                std::vector<std::string> documents,
                std::string model,
                std::string request_id,
                int32_t top_n,
                size_t num_requests,
                ScoreComputer compute_scores)
      : call(std::move(call)),
        documents(std::move(documents)),
        model(std::move(model)),
        request_id(std::move(request_id)),
        top_n(top_n),
        pending_count(num_requests),
        compute_scores(std::move(compute_scores)) {
    req_outputs.resize(num_requests);
  }

  void on_complete(size_t index, RequestOutput output) {
    req_outputs[index] = std::move(output);

    if (pending_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      finalize();
    }
  }

  void finalize();
};

class RerankServiceImpl : public APIServiceImpl<RerankCall> {
 public:
  RerankServiceImpl(LLMMaster* master, const std::vector<std::string>& models);

  virtual void process_async_impl(std::shared_ptr<RerankCall> call);

 protected:
  DISALLOW_COPY_AND_ASSIGN(RerankServiceImpl);
  LLMMaster* master_ = nullptr;
};

}  // namespace xllm
